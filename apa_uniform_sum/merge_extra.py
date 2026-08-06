"""Extended LoRA shard merging methods: DELLA, Breadcrumbs, KnOTS, TSV-M, SLERP,
orthogonal-projection subtraction, Fisher, LoraHub.

Implementations only — the method registry, label mapping and dispatch stay in
`merge_lora.py` (the single source of truth for what the pipeline exposes). All
mergers here take explicit adapter-name lists so the same code serves flat merges,
remerges and binary-tree nodes.

Conventions (verified against peft 0.14 `add_weighted_adapter` source):
- The scaffold adapter created by PEFT's `linear` combine gets `lora_alpha = r`,
  i.e. scaling == 1; we still read `module.scaling[new]` at write-back instead of
  assuming it (shards are trained with rslora, so source scalings are alpha/sqrt(r)).
- Factor-space methods (della_*, breadcrumbs) mirror PEFT's non-SVD family exactly:
  each lora_A / lora_B factor is combined independently with per-adapter weight
  sqrt(w_i * scaling_i). This keeps them apples-to-apples with linear/ties/dare_*.
  ⚠ Under rslora (how our shards are trained) the PEFT scaffold's scaling is
  alpha/sqrt(r) = sqrt(r) ≈ 2.83, NOT 1 — so the whole PEFT factor-space family
  (and della_*/breadcrumbs, deliberately) produces effective deltas inflated by
  that factor relative to the true weighted average. Established baseline
  convention; do not "fix" it here or results stop being comparable.
- fisher and lorahub divide by sqrt(scaffold scaling) per factor instead, so they
  produce the TRUE convex/learned combination (Fisher has no way to self-correct a
  global scale; LoraHub's learned weights are interpretable on the true scale).
- Dense-delta methods (knots_ties, tsv, slerp, subtract_orth) target
  sum_i w_i * scaling_i * B_i A_i and divide by the scaffold's scaling at write-back.
  We deliberately do NOT copy PEFT's *_svd path, which double-counts scaling
  (get_delta_weight already includes it AND valid_weights multiplies it again).
- Rank-r compression of rank-(n*r) merges uses the factored form (QR of the stacked
  factors + SVD of the small inner core) — never a dense d_out x d_in SVD.
- Stochastic methods (DELLA's Bernoulli masks, LoraHub's CMA-ES) take an explicit
  `seed` and are deterministic given it (module iteration order is fixed).
"""

import math

import torch
from peft.utils.merge_utils import (
    calculate_majority_sign_mask,
    disjoint_merge,
    reshape_weight_task_tensors,
    ties as peft_ties,
)

import jd_compress

DEFAULT_BREADCRUMBS_GAMMA = 0.01  # fraction of top-|w| outliers masked per task tensor
DEFAULT_DELLA_EPSILON = 0.1      # half-window of DELLA's magnitude-ranked drop probabilities


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def _uniform_weights(adapter_names, weights, fill=None):
    n = len(adapter_names)
    if weights is not None:
        return list(weights)
    return [1.0 / n if fill is None else fill] * n


def _scaffold_adapter(model, adapter_names, adapter_name):
    """Create the output adapter via PEFT's linear combine; weights get overwritten.

    Requires all source adapters to share the same rank (true for shards and for
    tree nodes produced by the methods in this module).
    """
    n = len(adapter_names)
    model.base_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=[1.0 / n] * n,
        adapter_name=adapter_name,
        combination_type="linear",
    )


def _iter_lora_modules(model, adapter_names, adapter_name):
    for _, module in model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_names[0] in module.lora_A):
            continue
        if adapter_name not in module.lora_A:
            continue
        yield module


def _sqrt_factor_weights(module, adapter_names, weights, device):
    """PEFT non-SVD convention: factor-space weight sqrt(w_i * scaling_i)."""
    return torch.tensor(
        [math.sqrt(w * module.scaling[s]) for w, s in zip(weights, adapter_names)],
        dtype=torch.float32, device=device,
    )


def _write_factors(module, adapter_name, A_new, B_new):
    """Write (possibly lower-rank) factors into the scaffold, zero-padding the rest."""
    A_t = module.lora_A[adapter_name].weight
    B_t = module.lora_B[adapter_name].weight
    r_eff = min(A_new.shape[0], A_t.shape[0])
    A_t.data.zero_()
    B_t.data.zero_()
    A_t.data[:r_eff].copy_(A_new[:r_eff].to(A_t.dtype))
    B_t.data[:, :r_eff].copy_(B_new[:, :r_eff].to(B_t.dtype))


def _compress_factored(B_cat, A_cat, rank):
    """SVD-compress delta = B_cat @ A_cat to `rank` without forming the dense delta.

    B_cat: (d_out, m), A_cat: (m, d_in) with m = sum of source ranks. QR both stacks,
    SVD the m x m core. Returns (A_new, B_new) of rank min(rank, m).
    """
    Qb, Rb = torch.linalg.qr(B_cat)
    Qa, Ra = torch.linalg.qr(A_cat.t())
    core = Rb @ Ra.t()
    U, S, Vh = torch.linalg.svd(core)
    r = min(rank, S.shape[0])
    B_new = Qb @ (U[:, :r] * S[:r])
    A_new = Vh[:r] @ Qa.t()
    return A_new, B_new


def _batch_to_device(batch, device):
    """Move a dict batch to device, defaulting labels to pad-masked input_ids."""
    inputs = {key: (val.to(device) if isinstance(val, torch.Tensor) else val)
              for key, val in batch.items()}
    if "labels" not in inputs:
        labels = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
    return inputs


# ---------------------------------------------------------------------------
# Wave 1: factor-space sparsifiers
# ---------------------------------------------------------------------------

def _magprune(tensor, density, epsilon, generator):
    """DELLA's MAGPRUNE: drop probability decreases linearly with magnitude rank.

    Smallest-|w| entries get drop prob (1-density)+epsilon/2, largest get
    (1-density)-epsilon/2 (clamped to [0,1]), so E[#kept] = density * numel.
    Survivors are rescaled by 1/(1-p_i) (unbiased, as in DARE). epsilon=0 reduces
    to DARE with per-element rescale; density>=1 with epsilon=0 is a no-op.
    """
    if density >= 1.0 and epsilon == 0.0:
        return tensor
    flat = tensor.reshape(-1)
    n = flat.numel()
    order = torch.argsort(flat.abs())
    ranks = torch.empty(n, dtype=torch.float32, device=flat.device)
    ranks[order] = torch.arange(n, dtype=torch.float32, device=flat.device)
    frac = ranks / max(n - 1, 1)  # 0 = smallest |w| ... 1 = largest
    p = ((1.0 - density) + epsilon * (0.5 - frac)).clamp(0.0, 1.0)
    keep = torch.bernoulli(1.0 - p, generator=generator)
    return torch.where(keep.bool(), flat / (1.0 - p).clamp_min(1e-8),
                       torch.zeros_like(flat)).reshape(tensor.shape)


def della_merge_adapters(
    model,
    adapter_names,
    *,
    adapter_name,
    weights=None,
    density=0.7,
    epsilon=DEFAULT_DELLA_EPSILON,
    seed=0,
    sign_consensus=False,
    majority_sign_method="total",
):
    """DELLA merge: MAGPRUNE each factor, then weighted sum (della_linear) or
    TIES-style sign election + disjoint merge (della_ties).

    Mirrors PEFT's dare_linear / dare_ties flow with MAGPRUNE replacing the uniform
    random drop. Sign mask is computed on the *unweighted* pruned tensors, exactly
    like peft.utils.merge_utils.ties.
    """
    weights = _uniform_weights(adapter_names, weights)
    _scaffold_adapter(model, adapter_names, adapter_name)
    generators = {}
    for module in _iter_lora_modules(model, adapter_names, adapter_name):
        for attr in ("lora_A", "lora_B"):
            layer = getattr(module, attr)
            tensors = [layer[s].weight.data.float() for s in adapter_names]
            dev = tensors[0].device
            if dev not in generators:
                generators[dev] = torch.Generator(device=dev)
                generators[dev].manual_seed(seed)
            pruned = torch.stack(
                [_magprune(t, density, epsilon, generators[dev]) for t in tensors]
            )
            w = _sqrt_factor_weights(module, adapter_names, weights, dev)
            weighted = pruned * reshape_weight_task_tensors(pruned, w)
            if sign_consensus:
                mask = calculate_majority_sign_mask(pruned, method=majority_sign_method)
                merged = disjoint_merge(weighted, mask)
            else:
                merged = weighted.sum(dim=0)
            layer[adapter_name].weight.data.copy_(merged.to(layer[adapter_name].weight.dtype))
    return adapter_name


def _breadcrumb_mask(tensor, density, gamma):
    """Model Breadcrumbs mask: drop the top-gamma |w| outliers, then keep the next
    density-fraction by magnitude (no rescale, per the paper)."""
    flat = tensor.reshape(-1)
    n = flat.numel()
    k_top = int(gamma * n)
    k_keep = min(int(density * n), n - k_top)
    if k_keep <= 0:
        return torch.zeros_like(tensor)
    idx = torch.argsort(flat.abs(), descending=True)
    mask = torch.zeros_like(flat)
    mask[idx[k_top:k_top + k_keep]] = 1.0
    return (flat * mask).reshape(tensor.shape)


def breadcrumbs_merge_adapters(
    model,
    adapter_names,
    *,
    adapter_name,
    weights=None,
    density=0.7,
    gamma=DEFAULT_BREADCRUMBS_GAMMA,
):
    """Model Breadcrumbs: two-sided magnitude masking per factor, then weighted sum.

    gamma=0 degenerates to PEFT's magnitude_prune.
    """
    weights = _uniform_weights(adapter_names, weights)
    _scaffold_adapter(model, adapter_names, adapter_name)
    for module in _iter_lora_modules(model, adapter_names, adapter_name):
        for attr in ("lora_A", "lora_B"):
            layer = getattr(module, attr)
            tensors = [layer[s].weight.data.float() for s in adapter_names]
            dev = tensors[0].device
            masked = torch.stack([_breadcrumb_mask(t, density, gamma) for t in tensors])
            w = _sqrt_factor_weights(module, adapter_names, weights, dev)
            merged = (masked * reshape_weight_task_tensors(masked, w)).sum(dim=0)
            layer[adapter_name].weight.data.copy_(merged.to(layer[adapter_name].weight.dtype))
    return adapter_name


# ---------------------------------------------------------------------------
# Wave 2: orthogonal / subspace methods
# ---------------------------------------------------------------------------

def knots_merge_adapters(
    model,
    adapter_names,
    *,
    adapter_name,
    weights=None,
    density=0.7,
    majority_sign_method="total",
    svd_rank=None,
):
    """True KnOTS: align shard deltas in a shared SVD basis, then TIES there.

    Per layer the stacked dense deltas M = [D_1; ...; D_n] (D_i = scaling_i*B_i A_i)
    factor as M = G Qa^T with G = blockdiag(s_i B_i) @ Ra^T, so the shared right
    basis comes from an SVD over the (n*r)-dim inner space — no dense d_out x d_in
    SVD. Each task's coefficients in the shared basis, W_i = U_i Sigma, are merged
    with PEFT's ties (same uniform-1/n weight convention as the registry's `ties`,
    so results are comparable), and W_hat V^T is compressed back to the scaffold rank.

    Unlike the registry's `ties_svd` (PEFT: merge dense deltas, then SVD), the TIES
    election here happens *in the aligned basis*, which is the KnOTS contribution.
    """
    weights = _uniform_weights(adapter_names, weights)
    _scaffold_adapter(model, adapter_names, adapter_name)
    rank = model.base_model.peft_config[adapter_name].r
    out_rank = min(svd_rank or rank, rank)
    for module in _iter_lora_modules(model, adapter_names, adapter_name):
        Bs = [module.lora_B[s].weight.data.float() for s in adapter_names]
        As = [module.lora_A[s].weight.data.float() for s in adapter_names]
        scals = [module.scaling[s] for s in adapter_names]
        d_out = Bs[0].shape[0]
        dev = Bs[0].device

        A_cat = torch.cat(As, dim=0)                      # (m, d_in)
        Qa, Ra = torch.linalg.qr(A_cat.t())               # (d_in, m), (m, m)
        RaT = Ra.t()
        offs = [0]
        for A in As:
            offs.append(offs[-1] + A.shape[0])
        G = torch.cat(
            [scals[i] * Bs[i] @ RaT[offs[i]:offs[i + 1], :] for i in range(len(As))],
            dim=0,
        )                                                 # (n*d_out, m)
        Ug, Sg, Vgh = torch.linalg.svd(G, full_matrices=False)
        # Task coefficients in the shared basis: W_i = U_i Sigma.
        coeffs = [Ug[i * d_out:(i + 1) * d_out] * Sg for i in range(len(As))]
        w = torch.tensor(weights, dtype=torch.float32, device=dev)
        W_hat = peft_ties(coeffs, w, density, majority_sign_method)
        # delta = W_hat V^T with V = Qa Vgh^T; compress via SVD of the thin W_hat.
        P, S2, Q2h = torch.linalg.svd(W_hat, full_matrices=False)
        r = min(out_rank, S2.shape[0])
        B_new = P[:, :r] * S2[:r]
        A_new = (Q2h[:r] @ Vgh) @ Qa.t()
        B_new = B_new / module.scaling[adapter_name]
        _write_factors(module, adapter_name, A_new, B_new)
    return adapter_name


def _inv_sqrt_psd(mat, eps):
    vals, vecs = torch.linalg.eigh(mat)
    return vecs @ torch.diag(vals.clamp_min(eps).rsqrt()) @ vecs.t()


def tsv_merge_adapters(
    model,
    adapter_names,
    *,
    adapter_name,
    weights=None,
    svd_rank=None,
    eps=1e-6,
):
    """TSV-M (Task Singular Vectors): whiten the stacked per-task singular vectors
    across tasks (symmetric orthogonalization), then sum and compress to rank r.

    Default weights are 1.0 per task (SUM, not mean): the whitening step already
    normalizes redundant/overlapping directions, so dividing by n on top of it
    would shrink the merged delta (n identical tasks would yield delta/n instead
    of delta). This deviates from the uniform-1/n convention of the other methods
    on purpose — it is the TSV-M paper's semantics.
    """
    weights = _uniform_weights(adapter_names, weights, fill=1.0)
    _scaffold_adapter(model, adapter_names, adapter_name)
    rank = model.base_model.peft_config[adapter_name].r
    out_rank = min(svd_rank or rank, rank)
    for module in _iter_lora_modules(model, adapter_names, adapter_name):
        Us, Vs, svals = [], [], []
        for w_i, s in zip(weights, adapter_names):
            B = module.lora_B[s].weight.data.float()
            A = module.lora_A[s].weight.data.float()
            Qb, Rb = torch.linalg.qr(B)
            Qa, Ra = torch.linalg.qr(A.t())
            core = module.scaling[s] * (Rb @ Ra.t())      # (r, r)
            Uc, Sc, Vch = torch.linalg.svd(core)
            Us.append(Qb @ Uc)
            Vs.append(Qa @ Vch.t())
            svals.append(w_i * Sc)
        U_all = torch.cat(Us, dim=1)                      # (d_out, m)
        V_all = torch.cat(Vs, dim=1)                      # (d_in, m)
        svec = torch.cat(svals)                           # (m,)
        # Whitening: components of U_all/V_all in the null space of the Gram matrix
        # are zero by construction, so the clamped 1/sqrt(eps) directions are inert.
        U_orth = U_all @ _inv_sqrt_psd(U_all.t() @ U_all, eps)
        V_orth = V_all @ _inv_sqrt_psd(V_all.t() @ V_all, eps)
        B_cat = U_orth * svec                             # (d_out, m)
        A_cat = V_orth.t()                                # (m, d_in)
        A_new, B_new = _compress_factored(B_cat, A_cat, out_rank)
        B_new = B_new / module.scaling[adapter_name]
        _write_factors(module, adapter_name, A_new, B_new)
    return adapter_name


def slerp_merge_pair(model, name_a, name_b, *, adapter_name, t=0.5, angle_eps=1e-4):
    """SLERP of exactly two adapters' per-layer deltas along the great circle.

    Inner products and norms are computed in factored form (r x r traces), so no
    dense deltas are materialized. Falls back to linear interpolation when the
    angle is tiny or a delta is (numerically) zero. The result rank is <= 2r and
    is compressed back to the scaffold rank.
    """
    _scaffold_adapter(model, [name_a, name_b], adapter_name)
    rank = model.base_model.peft_config[adapter_name].r
    for module in _iter_lora_modules(model, [name_a, name_b], adapter_name):
        Ba = module.lora_B[name_a].weight.data.float()
        Aa = module.lora_A[name_a].weight.data.float()
        Bb = module.lora_B[name_b].weight.data.float()
        Ab = module.lora_A[name_b].weight.data.float()
        sa, sb = module.scaling[name_a], module.scaling[name_b]
        # <Da, Db>_F = tr((Ba^T Bb)(Ab Aa^T)) (cyclic trace; stays r x r)
        ip = sa * sb * torch.trace((Ba.t() @ Bb) @ (Ab @ Aa.t()))
        na = (sa * sa * torch.trace((Ba.t() @ Ba) @ (Aa @ Aa.t()))).clamp_min(0).sqrt()
        nb = (sb * sb * torch.trace((Bb.t() @ Bb) @ (Ab @ Ab.t()))).clamp_min(0).sqrt()
        if na < 1e-12 or nb < 1e-12:
            ca, cb = 1.0 - t, t
        else:
            cos = (ip / (na * nb)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos)
            if theta < angle_eps:
                ca, cb = 1.0 - t, t
            else:
                sin_theta = torch.sin(theta)
                ca = (torch.sin((1.0 - t) * theta) / sin_theta).item()
                cb = (torch.sin(t * theta) / sin_theta).item()
        B_cat = torch.cat([ca * sa * Ba, cb * sb * Bb], dim=1)   # (d_out, 2r)
        A_cat = torch.cat([Aa, Ab], dim=0)                       # (2r, d_in)
        A_new, B_new = _compress_factored(B_cat, A_cat, rank)
        B_new = B_new / module.scaling[adapter_name]
        _write_factors(module, adapter_name, A_new, B_new)
    return adapter_name


def subtract_orth_adapters(model, adapter_names, forget_name, *, adapter_name, side="both"):
    """Unlearn by orthogonal projection: average ALL shard deltas (forget included),
    then project out the forget shard's LoRA subspaces — left = col(B_f), right =
    row(A_f). With side="both" the forget shard's own contribution projects to ~0,
    so this is the O(1) "drop the forget directions from an existing merge" operator
    (no re-merge of the survivors needed, unlike remerge_*).

    Everything stays factored: each shard's B_i / A_i is projected, then the
    weighted stack is compressed to the scaffold rank.
    """
    n = len(adapter_names)
    weights = [1.0 / n] * n
    _scaffold_adapter(model, adapter_names, adapter_name)
    rank = model.base_model.peft_config[adapter_name].r
    for module in _iter_lora_modules(model, adapter_names, adapter_name):
        Bf = module.lora_B[forget_name].weight.data.float()
        Af = module.lora_A[forget_name].weight.data.float()
        Uf, _ = torch.linalg.qr(Bf)        # orthonormal basis of col(B_f)
        Vf, _ = torch.linalg.qr(Af.t())    # orthonormal basis of row(A_f)
        B_list, A_list = [], []
        for w, s in zip(weights, adapter_names):
            B = module.lora_B[s].weight.data.float()
            A = module.lora_A[s].weight.data.float()
            if side in ("left", "both"):
                B = B - Uf @ (Uf.t() @ B)
            if side in ("right", "both"):
                A = A - (A @ Vf) @ Vf.t()
            B_list.append(w * module.scaling[s] * B)
            A_list.append(A)
        B_cat = torch.cat(B_list, dim=1)
        A_cat = torch.cat(A_list, dim=0)
        A_new, B_new = _compress_factored(B_cat, A_cat, rank)
        B_new = B_new / module.scaling[adapter_name]
        _write_factors(module, adapter_name, A_new, B_new)
    return adapter_name


def additive_merge_adapters(model, adapter_names, *, adapter_name, weights=None):
    """Honest additive composition — the research-doc's W + sum_i (alpha/r) B_i A_i.

    Sums each shard's TRUE effective delta scaling_i * B_i A_i exactly once (weights
    default 1.0 each = SUM, not the 1/n mean of the other methods), at FULL rank
    (= sum of source ranks) with NO SVD compression. This is the *corrected*
    `linear`/`cat`: PEFT's factor-space combine applies sqrt(w_i * scaling_i) per
    factor, double-counting scaling under rslora (scaling = alpha/sqrt(r) ~ sqrt(r)) —
    which is why the registry `linear`/`cat` explode on these shards. Here each
    effective delta enters once at true scale.

    Because B_new @ A_new = sum_i w_i scaling_i B_i A_i *exactly* (no lossy rank-r
    squash), dropping a shard from `adapter_names` removes precisely its term — the
    additive-exactness invariant the unlearning scheme relies on. Cost: merged rank =
    sum_i r_i, so fold the stable retain core into the base rather than coarsening
    this sum. Under standard LoRA (use_rslora=False) scaling = alpha/r and this is
    literally W + sum_i (alpha/r) B_i A_i.
    """
    weights = _uniform_weights(adapter_names, weights, fill=1.0)  # SUM, not 1/n mean
    # `cat` scaffold: rank = sum of source ranks, so the concatenated factors fit with
    # no truncation (a `linear` scaffold is rank r and would force lossy compression).
    model.base_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=[1.0] * len(adapter_names),
        adapter_name=adapter_name,
        combination_type="cat",
    )
    for module in _iter_lora_modules(model, adapter_names, adapter_name):
        B_list, A_list = [], []
        for w, s in zip(weights, adapter_names):
            B = module.lora_B[s].weight.data.float()
            A = module.lora_A[s].weight.data.float()
            B_list.append(w * module.scaling[s] * B)   # true effective-delta term
            A_list.append(A)
        B_new = torch.cat(B_list, dim=1)               # (d_out, sum_i r_i)
        A_new = torch.cat(A_list, dim=0)               # (sum_i r_i, d_in)
        # Divide out the scaffold's own scaling so the served delta is exactly
        # sum_i w_i scaling_i B_i A_i (PEFT re-applies scaling[adapter_name] at forward).
        B_new = B_new / module.scaling[adapter_name]
        _write_factors(module, adapter_name, A_new, B_new)
    return adapter_name


# ---------------------------------------------------------------------------
# Joint Diagonalization (Compress-then-Serve): compress the collection into a
# shared basis + per-adapter Sigma, then combine the kept adapters into one delta.
# ---------------------------------------------------------------------------

def _gather_jd_slots(model, adapter_names):
    """Collect per-module factored deltas for `adapter_names` from a live PeftModel.

    Returns (slots, modules) where slots[name] is a jd_compress.Slot and modules[name]
    is the owning LoRA module (for write-back). Module qualified name is the slot key.
    """
    slots, modules = {}, {}
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_names[0] in module.lora_A):
            continue
        slots[name] = jd_compress.Slot(
            B=[module.lora_B[s].weight.data.float() for s in adapter_names],
            A=[module.lora_A[s].weight.data.float() for s in adapter_names],
            scaling=[module.scaling[s] for s in adapter_names],
        )
        modules[name] = module
    return slots, modules


def jd_merge_adapters(model, adapter_names, *, adapter_name, variant="full",
                      clusters=1, rank=None, weights=None, seed=0, iters=10):
    """Joint-Diagonalization merge: compress `adapter_names` into a shared basis
    (per cluster, per module) with per-adapter Sigma, then combine ALL of them into a
    single scaffold adapter `D_S = sum_j U_j (sum_i w_i norm_i Sigma_i) V_j^T`,
    compressed to the scaffold rank.

    Selective keep/unlearn is expressed by which adapters are in `adapter_names`
    (e.g. remerge_* drops the forget shard before calling). The O(1)-deletion form
    (fit once on all adapters, then drop a Sigma) lives in jd_compress.JDCompressed /
    jd_collection. True-scale method: divide the scaffold scaling out at write-back.
    """
    n = len(adapter_names)
    _scaffold_adapter(model, adapter_names, adapter_name)
    scaffold_rank = model.base_model.peft_config[adapter_name].r
    if rank is None:
        # Paper Section 6.5: <=100 LoRAs -> JD-Full rank ~ (n/2)+7 (capped only by the
        # matrix dims inside jd_compress, NOT by the scaffold rank — the kept-set merge is
        # separately compressed to the scaffold rank); clustering -> rank 16.
        rank = 16 if clusters > 1 else max(1, n // 2 + 7)

    slots, modules = _gather_jd_slots(model, adapter_names)
    jd = jd_compress.jd_compress_collection(
        slots, adapter_names, variant=variant, clusters=clusters,
        rank=rank, iters=iters, seed=seed,
    )
    merged = jd.merge_keepset(range(n), weights=weights, out_rank=scaffold_rank)
    for name, (A_new, B_new) in merged.items():
        module = modules[name]
        B_new = B_new / module.scaling[adapter_name]
        _write_factors(module, adapter_name, A_new, B_new)
    return adapter_name


# ---------------------------------------------------------------------------
# Wave 3: data-required methods
# ---------------------------------------------------------------------------

def fisher_merge_shards(
    model,
    k,
    dataloader,
    *,
    num_examples=256,
    exclude_shard=None,
    adapter_name=None,
    eps=1e-8,
):
    """Diagonal-Fisher-weighted merge of the LoRA factors.

    Per shard: activate it, accumulate squared grads of its lora_A/B weights under
    the LM loss over ~num_examples examples (shared dataloader = the union of the
    included shards' data, same convention as regmean's Gram collection). Merge per
    factor entry: W = sum_i F_i * (sqrt(scaling_i) W_i) / (sum_i F_i + eps); entries
    where no shard accumulated Fisher mass fall back to the plain mean. The
    sqrt(scaling) convention matches the PEFT factor-space family, so k identical
    shards reproduce the original effective delta exactly.
    """
    adapter_names = [f"shard_{i}" for i in range(k) if i != exclude_shard]
    if adapter_name is None:
        tag = "fisher" if exclude_shard is None else f"fisher_no{exclude_shard}"
        adapter_name = f"merged_{tag}"
    device = next(model.parameters()).device

    fisher = {}
    for s in adapter_names:
        model.set_adapter(s)
        model.eval()
        params = {}
        for mod_name, module in model.named_modules():
            if hasattr(module, "lora_A") and s in module.lora_A:
                params[f"{mod_name}.lora_A"] = module.lora_A[s].weight
                params[f"{mod_name}.lora_B"] = module.lora_B[s].weight
        prev_grad = {key: p.requires_grad for key, p in params.items()}
        for p in params.values():
            p.requires_grad_(True)
        acc = {key: torch.zeros_like(p, dtype=torch.float32) for key, p in params.items()}

        seen, n_batches = 0, 0
        for batch in dataloader:
            if seen >= num_examples:
                break
            inputs = _batch_to_device(batch, device)
            model.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            loss.backward()
            for key, p in params.items():
                if p.grad is not None:
                    acc[key] += p.grad.detach().float() ** 2
            seen += inputs["input_ids"].shape[0]
            n_batches += 1
        model.zero_grad(set_to_none=True)
        for key, p in params.items():
            p.requires_grad_(prev_grad[key])
        fisher[s] = {key: a / max(n_batches, 1) for key, a in acc.items()}

    _scaffold_adapter(model, adapter_names, adapter_name)
    for mod_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_names[0] in module.lora_A):
            continue
        if adapter_name not in module.lora_A:
            continue
        for attr in ("lora_A", "lora_B"):
            key = f"{mod_name}.{attr}"
            num, den, mean = None, None, None
            for s in adapter_names:
                W = math.sqrt(module.scaling[s]) * getattr(module, attr)[s].weight.data.float()
                F_i = fisher[s].get(key)
                if F_i is None:
                    F_i = torch.zeros_like(W)
                num = W * F_i if num is None else num + W * F_i
                den = F_i if den is None else den + F_i
                mean = W if mean is None else mean + W
            mean = mean / len(adapter_names)
            # Smoothed Fisher average: exact when all shards agree (num = den*W),
            # decays to the plain mean where the accumulated Fisher mass is ~0.
            merged = (num + eps * mean) / (den + eps)
            # True-scale convention: cancel the scaffold's (rslora) scaling.
            merged = merged / math.sqrt(module.scaling[adapter_name])
            target = getattr(module, attr)[adapter_name].weight
            target.data.copy_(merged.to(target.dtype))
    return adapter_name


def lorahub_merge_shards(
    model,
    k,
    dataloader,
    *,
    num_examples=64,
    budget=40,
    exclude_shard=None,
    adapter_name=None,
    seed=0,
    l1_reg=0.05,
):
    """LoraHub-style merge: gradient-free (CMA-ES via nevergrad) optimization of the
    per-shard composition weights, minimizing LM loss on a small cached sample plus
    an L1 penalty on the weights (bounds and reg per the LoraHub paper).

    Composition is factor-level (sum_i w_i A_i, sum_i w_i B_i — LoraHub semantics,
    cross terms included). Trials overwrite one reusable adapter's tensors in place;
    no adapter add/delete churn. Deterministic given `seed`.
    """
    try:
        import nevergrad as ng
    except ImportError as exc:  # lazy: only lorahub needs it
        raise ImportError(
            "lorahub merging requires nevergrad (pip install nevergrad)"
        ) from exc
    import numpy as np

    adapter_names = [f"shard_{i}" for i in range(k) if i != exclude_shard]
    n = len(adapter_names)
    if adapter_name is None:
        tag = "lorahub" if exclude_shard is None else f"lorahub_no{exclude_shard}"
        adapter_name = f"merged_{tag}"
    device = next(model.parameters()).device

    cached, seen = [], 0
    for batch in dataloader:
        if seen >= num_examples:
            break
        inputs = _batch_to_device(batch, device)
        cached.append(inputs)
        seen += inputs["input_ids"].shape[0]
    if not cached:
        raise ValueError("lorahub: dataloader yielded no batches")

    _scaffold_adapter(model, adapter_names, adapter_name)
    stacks = []
    for module in _iter_lora_modules(model, adapter_names, adapter_name):
        scal = torch.tensor([math.sqrt(module.scaling[s]) for s in adapter_names],
                            dtype=torch.float32)
        A_s = torch.stack([module.lora_A[s].weight.data.float() for s in adapter_names])
        B_s = torch.stack([module.lora_B[s].weight.data.float() for s in adapter_names])
        stacks.append((module, A_s * scal.view(-1, 1, 1).to(A_s.device),
                       B_s * scal.view(-1, 1, 1).to(B_s.device)))

    def _apply(wvec):
        for module, A_s, B_s in stacks:
            wd = torch.tensor(list(wvec), dtype=torch.float32, device=A_s.device).view(-1, 1, 1)
            # True-scale convention: cancel the scaffold's (rslora) scaling so the
            # learned weights are interpretable as plain composition coefficients.
            inv = 1.0 / math.sqrt(module.scaling[adapter_name])
            A_new = (A_s * wd).sum(dim=0) * inv
            B_new = (B_s * wd).sum(dim=0) * inv
            module.lora_A[adapter_name].weight.data.copy_(
                A_new.to(module.lora_A[adapter_name].weight.dtype))
            module.lora_B[adapter_name].weight.data.copy_(
                B_new.to(module.lora_B[adapter_name].weight.dtype))

    model.set_adapter(adapter_name)
    model.eval()

    def objective(wvec):
        _apply(wvec)
        with torch.no_grad():
            losses = [model(**b).loss.item() for b in cached]
        mean_loss = sum(losses) / len(losses)
        if not math.isfinite(mean_loss):
            return 1e9
        return mean_loss + l1_reg * float(sum(abs(x) for x in wvec))

    param = ng.p.Array(init=np.full(n, 1.0 / n)).set_bounds(-1.5, 1.5)
    param.random_state.seed(seed)
    optimizer = ng.optimizers.NGOpt(parametrization=param, budget=budget)
    recommendation = optimizer.minimize(objective)
    best = [float(x) for x in recommendation.value]
    _apply(best)
    print(f"[lorahub] {adapter_name}: learned weights "
          f"{[round(x, 4) for x in best]} (budget={budget}, seed={seed})", flush=True)
    return adapter_name
