"""Shared LoRA shard merging + unlearn helpers.

All merging bottoms out in PEFT `LoraModel.add_weighted_adapter`. This module is
the single source of truth for which `combination_type`s we expose and how eval
labels map to merge calls. Imported by both `eval_tofu.py` (SLURM/CLI path) and
the notebook (Section 3).

Method taxonomy and empirical guidance: see
`LoRA Merging_ Methods and Effectiveness.md` in this directory.
"""

import re

import torch

from merge_extra import (
    additive_merge_adapters,
    breadcrumbs_merge_adapters,
    della_merge_adapters,
    fisher_merge_shards,
    jd_merge_adapters,
    knots_merge_adapters,
    lorahub_merge_shards,
    slerp_merge_pair,
    subtract_orth_adapters,
    tsv_merge_adapters,
)
from tree_utils import node_name, split, internal_nodes_postorder

DEFAULT_DENSITY = 0.7

# method name -> add_weighted_adapter spec.
# "density" present => method consumes a density (fraction of weights to KEEP).
MERGE_METHODS = {
    # Task Arithmetic: plain weighted sum of shard task vectors.
    "linear": {"combination_type": "linear"},
    # DARE: random drop + rescale, then linear sum.
    "dare_linear": {"combination_type": "dare_linear", "density": DEFAULT_DENSITY},
    # TIES: trim by magnitude -> elect sign -> merge agreeing entries.
    "ties": {"combination_type": "ties", "density": DEFAULT_DENSITY},
    # DARE then TIES (research-doc default for 3+ adapters).
    "dare_ties": {"combination_type": "dare_ties", "density": DEFAULT_DENSITY},
    # Magnitude prune: keep top-density by magnitude, then sum.
    "magnitude_prune": {"combination_type": "magnitude_prune", "density": DEFAULT_DENSITY},
    # Concatenate adapters: exact compose, resulting rank = sum of ranks.
    "cat": {"combination_type": "cat"},
    # Additive (the exact-unlearning research doc): TRUE-scale sum of effective deltas
    # sum_i scaling_i B_i A_i, weight 1.0 each, full rank, no compression. The corrected
    # `linear`/`cat` (divides out the rslora sqrt(r) double-count). Drop a shard => drop
    # exactly its term. See merge_extra.additive_merge_adapters.
    "additive": {"custom": True},
    # Additive MEAN: same true-scale full-rank composition but weight 1/n_active (the
    # model-soup / task-arithmetic regime that keeps the merged-delta norm ~1 single
    # adapter). Diagnoses whether the weight-1.0 `additive` collapse is norm-overshoot
    # (mean recovers utility) vs fundamental interference (mean also fails). Unlearn =
    # recompute the mean over the kept shards (still data-exact: no kept adapter saw
    # forget data) rather than subtract-a-term.
    "additive_mean": {"custom": True},
    # KnOTS-style: merge then compress back to a fixed rank via SVD.
    "ties_svd": {"combination_type": "ties_svd", "density": DEFAULT_DENSITY},
    "dare_ties_svd": {"combination_type": "dare_ties_svd", "density": DEFAULT_DENSITY},
    # RegMean: least-squares merge using per-shard Gram matrices in rank-r space.
    # Requires a dataloader; see regmean_merge_shards / merge_shards(dataloader=...).
    "regmean": {"requires_data": True},
    # Decomposition level (Eq 5 in MoLoRA / aggregation paper): plain weighted avg of A and B separately.
    "weighted_avg_ab": {"custom": True},
    # Non-decomposition level (Eq 6): weighted avg of BA products, SVD-compressed back to rank r.
    "weighted_ba": {"custom": True},
    # DELLA / MAGPRUNE: magnitude-ranked drop probs + rescale, then linear sum or TIES.
    "della_linear": {"custom": True, "density": DEFAULT_DENSITY},
    "della_ties": {"custom": True, "density": DEFAULT_DENSITY},
    # Model Breadcrumbs: mask top-gamma outliers AND small values, then weighted sum.
    "breadcrumbs": {"custom": True, "density": DEFAULT_DENSITY},
    # True KnOTS: shared-SVD-basis alignment, TIES in the aligned space, compress to rank r.
    # (Registry's ties_svd is PEFT merge-then-compress — a different algorithm.)
    "knots_ties": {"custom": True, "density": DEFAULT_DENSITY},
    # TSV-M: per-task SVD + cross-task whitening of singular vectors, sum, compress.
    "tsv": {"custom": True},
    # SLERP: pairwise geodesic interpolation. Exactly 2 adapters — tree merges only
    # (tree_root_slerp / tree_remerge_slerp), or flat merges with k=2.
    "slerp": {"custom": True, "pairwise": True},
    # Fisher merging: diagonal-Fisher-weighted factor average. Requires a dataloader.
    "fisher": {"requires_data": True},
    # LoraHub: CMA-ES-learned composition weights (nevergrad). Requires a dataloader.
    "lorahub": {"requires_data": True},
    # Joint Diagonalization (Compress-then-Serve): compress the collection into a
    # shared basis + per-adapter Sigma, then combine the kept adapters. True-scale.
    # Optional label suffixes: _c{N} = number of clusters, _r{R} = JD rank.
    "jd_full": {"custom": True},
    "jd_diag": {"custom": True},
}

# Data-required methods (eval_tofu.py builds a dataloader for these labels).
DATA_REQUIRED_METHODS = {"regmean", "fisher", "lorahub"}

# New/experimental methods evaluated alongside DEFAULT_MERGE_METHODS but kept out of
# it so Phase A result sets remain comparable. slerp is tree-only and handled apart.
EXPERIMENTAL_MERGE_METHODS = [
    "della_linear",
    "della_ties",
    "breadcrumbs",
    "knots_ties",
    "tsv",
    "jd_full",
    "jd_diag",
]

# Methods included in the default per-model eval manifest (Phase A).
# SVD variants stay in the registry for opt-in sweeps but are not default.
DEFAULT_MERGE_METHODS = [
    "linear",
    "dare_linear",
    "ties",
    "dare_ties",
    "magnitude_prune",
    "cat",
]

# Backward-compat for older label names.
LABEL_ALIASES = {
    "remerge_dare": "remerge_dare_linear",
}

# Routing strategies included in the default per-model eval manifest.
# Each also gets a _no{forget_id} unlearn-by-exclusion variant.
DEFAULT_ROUTING_LABELS = [
    "routed_key_exact",
    "routed_centroid_sbert",
    "routed_activation_norm",
]

# Routing strategies included in smoke eval (fast-only; skip ppl / logit_div / attn_norm).
SMOKE_ROUTING_LABELS = [
    "routed_key_exact",
    "routed_centroid_sbert",
]


def regmean_merge_shards(
    model,
    k,
    dataloader,
    *,
    num_examples=256,
    exclude_shard=None,
    adapter_name=None,
):
    """Merge shard adapters via RegMean least-squares.

    For each LoRA B matrix, solves B_merged = (Σ_i B_i G_i) @ pinv(Σ_i G_i),
    where G_i = E[h_i h_i^T] is the rank-r Gram matrix of shard i's activations
    at the output of lora_A. A matrices are averaged. Gram matrices stay r×r
    (small) since they are computed in the low-rank input space of B.

    Args:
        model: PeftModel with shard_0 … shard_{k-1} adapters.
        k: total number of shards.
        dataloader: yields batches (dict or tensor) for Gram collection.
        num_examples: examples to run per shard (approximate; stops at next batch boundary).
        exclude_shard: shard index to omit (remerge/unlearn path).
        adapter_name: name for the new adapter; auto-generated if None.
    Returns:
        adapter_name of the created merged adapter.
    """
    adapter_names = [f"shard_{i}" for i in range(k) if i != exclude_shard]
    n = len(adapter_names)

    if adapter_name is None:
        tag = "regmean" if exclude_shard is None else f"regmean_no{exclude_shard}"
        adapter_name = _sanitize(f"merged_{tag}")

    device = next(model.parameters()).device

    # --- Collect per-shard Gram matrices G_i = E[h h^T] in rank-r space ---
    gram_for_shard = {}

    for shard_name in adapter_names:
        model.set_adapter(shard_name)
        model.eval()

        gram_dict = {}
        count_dict = {}
        handles = []

        for layer_name, module in model.named_modules():
            if not (hasattr(module, "lora_A") and shard_name in module.lora_A):
                continue

            def _make_hook(name):
                def _hook(mod, inp, out):
                    # out: (..., r) — output of lora_A, input space of lora_B
                    h = out.detach().reshape(-1, out.shape[-1]).float()
                    xtx = h.t().mm(h)           # sum of outer products over tokens
                    n_tok = h.shape[0]
                    if name not in gram_dict:
                        gram_dict[name] = xtx / n_tok  # running mean
                        count_dict[name] = n_tok
                    else:
                        total = count_dict[name] + n_tok
                        gram_dict[name] = (gram_dict[name] * count_dict[name] + xtx) / total
                        count_dict[name] = total
                return _hook

            handles.append(module.lora_A[shard_name].register_forward_hook(_make_hook(layer_name)))

        seen = 0
        with torch.no_grad():
            for batch in dataloader:
                if seen >= num_examples:
                    break
                if isinstance(batch, dict):
                    inputs = {key: (val.to(device) if isinstance(val, torch.Tensor) else val)
                              for key, val in batch.items()}
                    model(**inputs)
                    seen += next(v for v in inputs.values() if isinstance(v, torch.Tensor)).shape[0]
                else:
                    model(batch.to(device))
                    seen += batch.shape[0]

        for handle in handles:
            handle.remove()
        gram_for_shard[shard_name] = gram_dict

    # --- Create placeholder adapter (linear avg); weights are overwritten below ---
    model.base_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=[1.0 / n] * n,
        adapter_name=adapter_name,
        combination_type="linear",
    )

    # --- Overwrite A (simple average) and B (RegMean solution) ---
    for layer_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_names[0] in module.lora_A):
            continue
        if adapter_name not in module.lora_A:
            continue

        dtype = module.lora_B[adapter_names[0]].weight.dtype

        A_avg = torch.stack(
            [module.lora_A[s].weight.data.float() for s in adapter_names]
        ).mean(0)
        module.lora_A[adapter_name].weight.data.copy_(A_avg.to(dtype))

        if layer_name in gram_for_shard[adapter_names[0]]:
            sum_BG = sum(
                module.lora_B[s].weight.data.float() @ gram_for_shard[s][layer_name]
                for s in adapter_names
            )
            sum_G = sum(gram_for_shard[s][layer_name] for s in adapter_names)
            B_merged = sum_BG @ torch.linalg.pinv(sum_G)
        else:
            B_merged = torch.stack(
                [module.lora_B[s].weight.data.float() for s in adapter_names]
            ).mean(0)

        module.lora_B[adapter_name].weight.data.copy_(B_merged.to(dtype))

    return adapter_name


def weighted_avg_ab_merge_shards(
    model,
    k,
    *,
    weights=None,
    exclude_shard=None,
    adapter_name=None,
):
    """Merge shard adapters via decomposition-level weighted average (Eq 5).

    Computes B̄ = Σ ωk·Bk and Ā = Σ ωk·Ak independently, then stores them as a
    new rank-r LoRA adapter. Unlike PEFT's 'linear' combination type, which applies
    sqrt(ω·scaling) to each factor, this uses plain ωk weights on the raw matrices.

    Args:
        model: PeftModel with shard_0 … shard_{k-1} adapters.
        k: total number of shards.
        weights: per-adapter weights; defaults to uniform 1/n.
        exclude_shard: shard index to omit (remerge/unlearn path).
        adapter_name: name for the new adapter; auto-generated if None.
    Returns:
        adapter_name of the created merged adapter.
    """
    adapter_names = [f"shard_{i}" for i in range(k) if i != exclude_shard]
    n = len(adapter_names)
    if weights is None:
        weights = [1.0 / n] * n
    if adapter_name is None:
        tag = "weighted_avg_ab" if exclude_shard is None else f"weighted_avg_ab_no{exclude_shard}"
        adapter_name = _sanitize(f"merged_{tag}")

    # Create adapter scaffolding; sets lora_alpha = new_rank so scaling = 1 on merged adapter.
    model.base_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=[1.0 / n] * n,
        adapter_name=adapter_name,
        combination_type="linear",
    )

    for _, module in model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_names[0] in module.lora_A):
            continue
        if adapter_name not in module.lora_A:
            continue
        dtype = module.lora_B[adapter_names[0]].weight.dtype
        A_merged = sum(w * module.lora_A[s].weight.data.float() for w, s in zip(weights, adapter_names))
        B_merged = sum(w * module.lora_B[s].weight.data.float() for w, s in zip(weights, adapter_names))
        module.lora_A[adapter_name].weight.data.copy_(A_merged.to(dtype))
        module.lora_B[adapter_name].weight.data.copy_(B_merged.to(dtype))

    return adapter_name


def weighted_ba_merge_shards(
    model,
    k,
    *,
    weights=None,
    svd_rank=None,
    exclude_shard=None,
    adapter_name=None,
):
    """Merge shard adapters via non-decomposition-level weighted average (Eq 6).

    Computes the dense delta B̄A̅ = Σ ωk·Bk·Ak, then SVD-compresses it back to
    rank r (or svd_rank) to obtain new A and B factors. This avoids the cross-adapter
    contamination of decomposition-level averaging at the cost of an SVD per layer.

    Args:
        model: PeftModel with shard_0 … shard_{k-1} adapters.
        k: total number of shards.
        weights: per-adapter weights; defaults to uniform 1/n.
        svd_rank: rank of compressed output adapter; defaults to source shard rank.
        exclude_shard: shard index to omit (remerge/unlearn path).
        adapter_name: name for the new adapter; auto-generated if None.
    Returns:
        adapter_name of the created merged adapter.
    """
    adapter_names = [f"shard_{i}" for i in range(k) if i != exclude_shard]
    n = len(adapter_names)
    if weights is None:
        weights = [1.0 / n] * n
    if adapter_name is None:
        tag = "weighted_ba" if exclude_shard is None else f"weighted_ba_no{exclude_shard}"
        adapter_name = _sanitize(f"merged_{tag}")

    base_rank = model.base_model.peft_config[adapter_names[0]].r
    r = svd_rank or base_rank

    model.base_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=[1.0 / n] * n,
        adapter_name=adapter_name,
        combination_type="linear",
    )

    for _, module in model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_names[0] in module.lora_A):
            continue
        if adapter_name not in module.lora_A:
            continue
        dtype = module.lora_B[adapter_names[0]].weight.dtype
        # Dense weighted sum of BA products: shape (d_out, d_in)
        delta = sum(
            w * (module.lora_B[s].weight.data.float() @ module.lora_A[s].weight.data.float())
            for w, s in zip(weights, adapter_names)
        )
        # Economy SVD and truncate to rank r
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        # lora_A.weight shape: (r, d_in); lora_B.weight shape: (d_out, r)
        A_new = Vh[:r]
        B_new = U[:, :r] * S[:r].unsqueeze(0)
        module.lora_A[adapter_name].weight.data.copy_(A_new.to(dtype))
        module.lora_B[adapter_name].weight.data.copy_(B_new.to(dtype))

    return adapter_name


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _sanitize(name):
    return name.replace(".", "p")


def _dispatch_custom(
    model,
    adapter_names,
    method,
    *,
    adapter_name,
    weights=None,
    density=None,
    majority_sign_method="total",
    svd_rank=None,
    seed=0,
    clusters=None,
    jd_rank=None,
    scale=None,
):
    """Dispatch the merge_extra custom methods on an explicit adapter-name list.

    Shared by merge_shards (flat/remerge) and _merge_two (tree nodes).
    """
    if method == "additive":
        # Global coefficient λ (from a `_s{λ}` label): weight every shard by λ, so the
        # composed delta is λ*Σ scalingᵢ BᵢAᵢ and dropping a shard drops exactly λ*dW_j.
        w = weights
        if w is None and scale is not None:
            w = [scale] * len(adapter_names)
        return additive_merge_adapters(
            model, adapter_names,
            adapter_name=adapter_name,
            weights=w,
        )
    if method == "additive_mean":
        n = len(adapter_names)
        return additive_merge_adapters(
            model, adapter_names,
            adapter_name=adapter_name,
            weights=[1.0 / n] * n if weights is None else weights,
        )
    if method in ("jd_full", "jd_diag"):
        return jd_merge_adapters(
            model, adapter_names,
            adapter_name=adapter_name,
            variant="full" if method == "jd_full" else "diag",
            clusters=clusters or 1,
            rank=jd_rank,
            weights=weights,
            seed=seed,
        )
    if method in ("della_linear", "della_ties"):
        return della_merge_adapters(
            model, adapter_names,
            adapter_name=adapter_name,
            weights=weights,
            density=DEFAULT_DENSITY if density is None else density,
            seed=seed,
            sign_consensus=(method == "della_ties"),
            majority_sign_method=majority_sign_method,
        )
    if method == "breadcrumbs":
        # Global coefficient λ (from a `_s{λ}` label), same convention as `additive`:
        # weight every shard by λ instead of the uniform 1/n default. Needed because the
        # factor-space sqrt(w·scaling) family is √r-inflated under rslora shards — the
        # 2026-06-11 degenerate breadcrumbs run is the motivating case (fix: λ≈1/(n√r)).
        w = weights
        if w is None and scale is not None:
            w = [scale] * len(adapter_names)
        return breadcrumbs_merge_adapters(
            model, adapter_names,
            adapter_name=adapter_name,
            weights=w,
            density=DEFAULT_DENSITY if density is None else density,
        )
    if method == "knots_ties":
        return knots_merge_adapters(
            model, adapter_names,
            adapter_name=adapter_name,
            weights=weights,
            density=DEFAULT_DENSITY if density is None else density,
            majority_sign_method=majority_sign_method,
            svd_rank=svd_rank,
        )
    if method == "tsv":
        return tsv_merge_adapters(
            model, adapter_names,
            adapter_name=adapter_name,
            weights=weights,
            svd_rank=svd_rank,
        )
    if method == "slerp":
        if len(adapter_names) != 2:
            raise ValueError(
                "slerp merges exactly two adapters; use tree_root_slerp / "
                "tree_remerge_slerp for k > 2"
            )
        return slerp_merge_pair(
            model, adapter_names[0], adapter_names[1], adapter_name=adapter_name
        )
    raise ValueError(f"Not a custom merge method: {method!r}")


def merge_shards(
    model,
    k,
    method,
    *,
    exclude_shard=None,
    density=None,
    weights=None,
    majority_sign_method="total",
    svd_rank=None,
    adapter_name=None,
    dataloader=None,
    num_regmean_examples=256,
    seed=0,
    clusters=None,
    jd_rank=None,
    scale=None,
):
    """Merge shard adapters into a new named adapter and return its name.

    Args:
        model: PeftModel (uses `model.base_model.add_weighted_adapter`).
        k: number of shards.
        method: key in MERGE_METHODS.
        exclude_shard: if set, omit that shard (used for unlearn-by-remerge).
        density: override the method's default density (sweeps).
        weights: per-adapter weights; defaults to uniform 1/n.
        majority_sign_method: "total" or "frequency" (TIES family only).
        svd_rank: output rank for *_svd methods; defaults to source shard rank.
        adapter_name: explicit name for the merged adapter.
        dataloader: required for DATA_REQUIRED_METHODS (regmean/fisher/lorahub);
            yields dict batches (Gram collection, Fisher grads, LoraHub loss).
        num_regmean_examples: examples to use per shard for "regmean" / "fisher".
        seed: RNG seed for stochastic methods (della_*, lorahub).
    """
    if method not in MERGE_METHODS:
        raise ValueError(f"Unknown merge method: {method}. Known: {sorted(MERGE_METHODS)}")

    if method in DATA_REQUIRED_METHODS and dataloader is None:
        raise ValueError(f"merge_shards: method='{method}' requires dataloader=...")

    if method == "fisher":
        return fisher_merge_shards(
            model, k, dataloader,
            num_examples=num_regmean_examples,
            exclude_shard=exclude_shard,
            adapter_name=adapter_name,
        )

    if method == "lorahub":
        return lorahub_merge_shards(
            model, k, dataloader,
            exclude_shard=exclude_shard,
            adapter_name=adapter_name,
            seed=seed,
        )

    if method == "regmean":
        return regmean_merge_shards(
            model, k, dataloader,
            num_examples=num_regmean_examples,
            exclude_shard=exclude_shard,
            adapter_name=adapter_name,
        )

    if method == "weighted_avg_ab":
        return weighted_avg_ab_merge_shards(
            model, k,
            weights=weights,
            exclude_shard=exclude_shard,
            adapter_name=adapter_name,
        )

    if method == "weighted_ba":
        return weighted_ba_merge_shards(
            model, k,
            weights=weights,
            svd_rank=svd_rank,
            exclude_shard=exclude_shard,
            adapter_name=adapter_name,
        )

    spec = MERGE_METHODS[method]

    adapter_names = [f"shard_{i}" for i in range(k) if i != exclude_shard]
    n = len(adapter_names)
    if weights is None and not spec.get("custom"):
        weights = [1.0 / n] * n

    if adapter_name is None:
        tag = method
        if exclude_shard is not None:
            tag = f"{method}_no{exclude_shard}"
        if density is not None:
            tag = f"{tag}_d{density}"
        adapter_name = _sanitize(f"merged_{tag}")

    if spec.get("custom"):
        # della_*, breadcrumbs, knots_ties, tsv, slerp — weights default inside
        # each method (tsv intentionally uses 1.0/task, not 1/n; see merge_extra).
        return _dispatch_custom(
            model, adapter_names, method,
            adapter_name=adapter_name,
            weights=weights,
            density=density,
            majority_sign_method=majority_sign_method,
            svd_rank=svd_rank,
            seed=seed,
            clusters=clusters,
            jd_rank=jd_rank,
            scale=scale,
        )

    combination_type = spec["combination_type"]

    kwargs = dict(
        adapters=adapter_names,
        weights=weights,
        adapter_name=adapter_name,
        combination_type=combination_type,
    )
    if "density" in spec:
        kwargs["density"] = spec["density"] if density is None else density
    if "ties" in method:
        kwargs["majority_sign_method"] = majority_sign_method
    if combination_type.endswith("svd"):
        base_rank = model.base_model.peft_config[adapter_names[0]].r
        kwargs["svd_rank"] = svd_rank or base_rank

    model.base_model.add_weighted_adapter(**kwargs)
    return adapter_name


def create_unlearn_subtract(model, k, forget_id):
    """Task-arithmetic unlearn: subtract the forget shard's task vector.

    Uses `cat` (not `linear`): PEFT's linear path applies sqrt(weight*scaling),
    which fails on the negative weight this subtraction needs. `cat` multiplies
    weights directly, so it represents the signed combination exactly (at the
    cost of rank = k * r).
    """
    sub_adapters = [f"shard_{i}" for i in range(k)]
    sub_weights = [-1.0 if i == forget_id else k / (k - 1) for i in range(k)]
    w_sum = sum(sub_weights)
    sub_weights = [w / w_sum for w in sub_weights]
    model.base_model.add_weighted_adapter(
        adapters=sub_adapters,
        weights=sub_weights,
        adapter_name="unlearn_subtract",
        combination_type="cat",
    )
    return "unlearn_subtract"


def create_unlearn_subtract_orth(model, k, forget_id, *, side="both"):
    """Orthogonal-projection unlearn: average ALL k shard deltas, then project out
    the forget shard's LoRA subspaces (col(B_f) / row(A_f)).

    Unlike `subtract_linear` (signed task arithmetic) or `remerge_*` (rebuild
    without the shard), this surgically removes the forget *directions* from a
    merge that still contains the shard — the deletion analog of dropping the
    module, applied in weight space.
    """
    names = [f"shard_{i}" for i in range(k)]
    return subtract_orth_adapters(
        model, names, f"shard_{forget_id}",
        adapter_name="unlearn_subtract_orth",
        side=side,
    )


def label_requires_data(label):
    """True if the eval label needs a dataloader (regmean / fisher / lorahub merges)."""
    label = LABEL_ALIASES.get(label, label)
    for prefix in ("merged_", "remerge_"):
        if label.startswith(prefix):
            method, _ = _split_density_suffix(label[len(prefix):])
            return method in DATA_REQUIRED_METHODS
    return False


def _split_density_suffix(spec):
    """'ties_d0.5' -> ('ties', 0.5); 'ties' -> ('ties', None)."""
    parts = spec.rsplit("_d", 1)
    if len(parts) == 2 and _is_float(parts[1]):
        return parts[0], float(parts[1])
    return spec, None


def _split_scale_suffix(spec):
    """'additive_s0.2' -> ('additive', 0.2); 'additive' -> ('additive', None).

    Global composition coefficient λ for the additive method: W + λ*sum_i dW_i. A
    FIXED λ preserves exact drop-a-term (unlearn shard j -> drop λ*dW_j, bit-exact).
    Guarded by _is_float so it no-ops on '_svd' and other non-numeric '_s...' tails.
    """
    parts = spec.rsplit("_s", 1)
    if len(parts) == 2 and _is_float(parts[1]):
        return parts[0], float(parts[1])
    return spec, None


def _split_jd_suffix(method):
    """Peel JD hyperparameter suffixes: 'jd_full_c4_r16' -> ('jd_full', 4, 16).

    _c{N} = number of clusters, _r{R} = JD compression rank. Either may be omitted;
    label order is _c then _r (so _r is stripped first). Returns (method, clusters,
    jd_rank) with None for absent suffixes. No-ops on non-JD method strings.
    """
    clusters = jd_rank = None
    m = re.search(r"_r(\d+)$", method)
    if m:
        jd_rank = int(m.group(1))
        method = method[: m.start()]
    m = re.search(r"_c(\d+)$", method)
    if m:
        clusters = int(m.group(1))
        method = method[: m.start()]
    return method, clusters, jd_rank


def _parse_routing_label(label: str):
    """Parse 'routed_{strategy}[_no{i}]' into (strategy, exclude_set).

    Examples:
        'routed_key_exact'       -> ('key_exact', frozenset())
        'routed_centroid_sbert_no3' -> ('centroid_sbert', frozenset({3}))
    """
    rest = label[len("routed_"):]
    m = re.search(r"_no(\d+)$", rest)
    if m:
        return rest[: m.start()], frozenset({int(m.group(1))})
    return rest, frozenset()


def _build_routed_model(
    model,
    k: int,
    strategy: str,
    exclude: frozenset,
    *,
    tokenizer=None,
    dataset=None,
    centroid_cache_dir: str | None = None,
):
    """Construct a RoutedModel for the given strategy string and exclude set."""
    from router import (
        KeyRouter, CentroidRouter, PplRouter, ActivationRouter, RoutedModel,
        build_key_index, build_tfidf_router,
        make_lm_embed_fn, make_sbert_embed_fn, build_centroids,
    )

    def _wrap(router):
        rm = RoutedModel(model, router, tokenizer=tokenizer)
        # Bake the exclude set into a closure so callers need not pass it.
        if exclude:
            _orig_route = router.route
            rm._exclude = exclude
            original_route = rm._route

            def _routed_with_exclude(input_ids_1d, exc=frozenset()):
                return original_route(input_ids_1d, exclude=exclude | exc)

            rm._route = _routed_with_exclude
        return rm

    if strategy == "key_exact":
        if dataset is None:
            raise ValueError("routed_key_exact requires dataset=")
        ki = build_key_index(dataset, k)
        router = KeyRouter(ki, method="exact")
    elif strategy == "key_tfidf":
        if dataset is None:
            raise ValueError("routed_key_tfidf requires dataset=")
        router = build_tfidf_router(dataset, k)
    elif strategy in ("centroid_lm", "centroid_lm_last"):
        if dataset is None or tokenizer is None:
            raise ValueError(f"routed_{strategy} requires dataset= and tokenizer=")
        mode = "mean" if strategy == "centroid_lm" else "last"
        embed_fn = make_lm_embed_fn(model, tokenizer, mode=mode)
        centroids = build_centroids(
            embed_fn, dataset, k,
            cache_dir=centroid_cache_dir,
            embed_label=strategy,
        )
        router = CentroidRouter(centroids, embed_fn)
    elif strategy == "centroid_sbert":
        if dataset is None:
            raise ValueError("routed_centroid_sbert requires dataset=")
        embed_fn = make_sbert_embed_fn()
        centroids = build_centroids(
            embed_fn, dataset, k,
            cache_dir=centroid_cache_dir,
            embed_label="centroid_sbert",
        )
        router = CentroidRouter(centroids, embed_fn)
    elif strategy == "ppl":
        if tokenizer is None:
            raise ValueError("routed_ppl requires tokenizer=")
        router = PplRouter(model, tokenizer, k)
    elif strategy == "activation_norm":
        router = ActivationRouter(model, k, mode="activation_norm")
    elif strategy == "logit_div":
        router = ActivationRouter(model, k, mode="logit_div")
    elif strategy == "attn_norm":
        router = ActivationRouter(model, k, mode="attn_norm")
    else:
        raise ValueError(f"Unknown routing strategy: {strategy!r}")

    return _wrap(router)


def activate_label(
    model,
    k,
    forget_id,
    label,
    *,
    dataloader=None,
    num_regmean_examples=256,
    tokenizer=None,
    dataset=None,
    centroid_cache_dir=None,
    output_dir=None,
):
    """Map a logical eval label to a PEFT adapter name or a model wrapper.

    Supported labels:
        shard_{i}_only
        merged_{method}[_d{density}]
        remerge_{method}[_d{density}]        (excludes forget_id shard)
        merged_jd_{full|diag}[_c{N}][_r{R}]  (Joint Diagonalization; _cN clusters, _rR rank)
        remerge_jd_{full|diag}[_c{N}][_r{R}] (JD keep-all-but-forget = selective-keep unlearn)
        subtract_linear
        subtract_orth                        (project out forget shard's subspaces)
        tree_root_{method}[_d{density}]
        tree_remerge_{method}[_d{density}]
        routed_{strategy}[_no{i}]            (returns RoutedModel, not str)
        ensemble_{probs|logits}[_no{i}]      (returns EnsembleModel, not str —
                                              SISA/S3T prediction-level ensemble
                                              over the loaded shard adapters)

    Routing strategies: key_exact, key_tfidf, centroid_lm, centroid_lm_last,
        centroid_sbert, ppl, activation_norm, logit_div, attn_norm.

    Pass tokenizer= and dataset= when any routing label is used.
    Pass centroid_cache_dir= to cache/load centroid .npy files across runs.
    Pass dataloader= for DATA_REQUIRED_METHODS labels (regmean/fisher/lorahub);
    see label_requires_data().
    Pass output_dir= for ensemble labels so the constituent set is verified
    against the shard_* dirs on disk (the loader skips missing dirs silently).

    Returns:
        str          — adapter name (all non-wrapper labels)
        RoutedModel / EnsembleModel — wrapper labels; callers branch with
        isinstance(result, str).
    """
    label = LABEL_ALIASES.get(label, label)

    if label.startswith("shard_") and label.endswith("_only"):
        shard_id = int(label.split("_")[1])
        return f"shard_{shard_id}"

    if label == "subtract_linear":
        return create_unlearn_subtract(model, k, forget_id)

    if label == "subtract_orth":
        return create_unlearn_subtract_orth(model, k, forget_id)

    for prefix, exclude in (("merged_", None), ("remerge_", forget_id)):
        if label.startswith(prefix):
            method, density = _split_density_suffix(label[len(prefix) :])
            method, clusters, jd_rank = _split_jd_suffix(method)
            method, scale = _split_scale_suffix(method)
            if method not in MERGE_METHODS:
                raise ValueError(f"Unknown merge method in label '{label}': {method}")
            return merge_shards(
                model, k, method,
                exclude_shard=exclude,
                density=density,
                clusters=clusters,
                jd_rank=jd_rank,
                scale=scale,
                adapter_name=_sanitize(label),
                dataloader=dataloader,
                num_regmean_examples=num_regmean_examples,
            )

    if label.startswith("tree_root_"):
        method, density = _split_density_suffix(label[len("tree_root_"):])
        if method not in MERGE_METHODS:
            raise ValueError(f"Unknown merge method in label '{label}': {method}")
        kwargs = {} if density is None else {"density": density}
        return merge_tree(model, k, method, **kwargs)

    if label.startswith("tree_remerge_"):
        method, density = _split_density_suffix(label[len("tree_remerge_"):])
        if method not in MERGE_METHODS:
            raise ValueError(f"Unknown merge method in label '{label}': {method}")
        kwargs = {} if density is None else {"density": density}
        return remerge_tree_path(model, k, forget_id, method, **kwargs)

    if label.startswith("routed_"):
        strategy, exclude = _parse_routing_label(label)
        return _build_routed_model(
            model, k, strategy, exclude,
            tokenizer=tokenizer,
            dataset=dataset,
            centroid_cache_dir=centroid_cache_dir,
        )

    if label.startswith("ensemble_"):
        from ensemble import EnsembleModel, discover_ensemble_adapters, parse_ensemble_label
        mode, exclude = parse_ensemble_label(label)
        adapters = discover_ensemble_adapters(model, exclude=exclude, output_dir=output_dir)
        return EnsembleModel(model, adapters, mode=mode)

    raise ValueError(f"Unknown label: {label}")


def _merge_two(model, left_name, right_name, out_name, method, *, density=None,
               majority_sign_method="total", svd_rank=None):
    """Merge exactly two named adapters into out_name using method.

    Shared by merge_tree and _build_tree_excluding. Data-required methods
    (regmean/fisher/lorahub) are not supported because they would need a
    per-node dataloader pass; weighted_avg_ab/weighted_ba keep shard semantics.
    """
    if (method in DATA_REQUIRED_METHODS
            or method in ("weighted_avg_ab", "weighted_ba", "jd_full", "jd_diag")):
        # JD compresses across the whole collection; pairwise tree nodes defeat that.
        raise ValueError(
            f"method='{method}' is not supported for tree merging."
        )
    if method not in MERGE_METHODS:
        raise ValueError(f"Unknown merge method: {method!r}. Known: {sorted(MERGE_METHODS)}")

    spec = MERGE_METHODS[method]
    if spec.get("custom"):
        # della_*, breadcrumbs, knots_ties, tsv and slerp all handle n=2.
        _dispatch_custom(
            model, [left_name, right_name], method,
            adapter_name=out_name,
            density=density,
            majority_sign_method=majority_sign_method,
            svd_rank=svd_rank,
        )
        return
    combination_type = spec["combination_type"]
    kwargs = dict(
        adapters=[left_name, right_name],
        weights=[0.5, 0.5],
        adapter_name=out_name,
        combination_type=combination_type,
    )
    if "density" in spec:
        kwargs["density"] = spec["density"] if density is None else density
    if "ties" in method:
        kwargs["majority_sign_method"] = majority_sign_method
    if combination_type.endswith("svd"):
        base_rank = model.base_model.peft_config[left_name].r
        kwargs["svd_rank"] = svd_rank or base_rank

    model.base_model.add_weighted_adapter(**kwargs)


def merge_tree(model, k, method, **kwargs):
    """Build all internal tree nodes bottom-up and return the root adapter name.

    Iterates internal_nodes_postorder(k) so children are always created before
    parents. Each internal node merges its two children with a 2-way merge
    using method. Leaf adapters (shard_0 … shard_{k-1}) must already be loaded.

    Args:
        model: PeftModel with shard_0 … shard_{k-1} adapters loaded.
        k: total number of shards.
        method: key in MERGE_METHODS (not "regmean").
        **kwargs: forwarded to _merge_two (density, majority_sign_method, svd_rank).
    Returns:
        Adapter name of the root node (e.g. "tnode_0_9_linear").
    """
    for (lo, hi) in internal_nodes_postorder(k):
        (llo, lhi), (rlo, rhi) = split(lo, hi)
        left = node_name(llo, lhi, method)
        right = node_name(rlo, rhi, method)
        out = node_name(lo, hi, method)
        _merge_two(model, left, right, out, method, **kwargs)
    return node_name(0, k - 1, method)


def _build_tree_excluding(model, k, lo, hi, forget_id, method, prefix, **kwargs):
    """Recursively build the tree over [lo, hi] excluding forget_id.

    Returns the adapter name covering [lo, hi] without forget_id's data,
    or None if the entire subtree [lo, hi] == {forget_id} (i.e. lo == hi == forget_id).

    When one child resolves to None (its subtree is entirely forget_id), the
    parent simply passes through the surviving child without creating a new adapter.
    Surviving nodes use the original tnode_{lo}_{hi}_{method} names (pre-built by
    merge_tree); only nodes on the forget path get new "{prefix}_tnode_{lo}_{hi}" adapters.
    """
    if lo == hi:
        return None if lo == forget_id else f"shard_{lo}"

    (llo, lhi), (rlo, rhi) = split(lo, hi)
    left = _build_tree_excluding(model, k, llo, lhi, forget_id, method, prefix, **kwargs)
    right = _build_tree_excluding(model, k, rlo, rhi, forget_id, method, prefix, **kwargs)

    if left is None:
        return right
    if right is None:
        return left

    # Both sides present — need a fresh merge for this node.
    out = f"{prefix}_tnode_{lo}_{hi}"
    _merge_two(model, left, right, out, method, **kwargs)
    return out


def remerge_tree_path(model, k, forget_id, method, **kwargs):
    """Build the tree root adapter excluding forget_id's shard.

    Used for unlearn-by-remerge evaluation: produces the same topology as
    merge_tree but replaces the forget shard with nothing (its siblings survive
    unchanged). Only O(log k) new adapters are created — one per ancestor on the
    forget path.

    Args:
        model: PeftModel with shard_0 … shard_{k-1} adapters (and optionally
               pre-built tnode_*_method adapters from merge_tree).
        k: total number of shards.
        forget_id: shard index to exclude.
        method: key in MERGE_METHODS (not "regmean").
        **kwargs: forwarded to _merge_two.
    Returns:
        Adapter name of the effective root (may be a surviving subtree adapter
        if the entire left or right half was the forgotten shard).
    """
    prefix = _sanitize(f"treerm{forget_id}_{method}")
    return _build_tree_excluding(model, k, 0, k - 1, forget_id, method, prefix, **kwargs)


def default_eval_labels(k, forget_id):
    labels = [f"shard_{i}_only" for i in range(k)]
    labels += [f"merged_{m}" for m in DEFAULT_MERGE_METHODS + EXPERIMENTAL_MERGE_METHODS]
    labels += [f"remerge_{m}" for m in DEFAULT_MERGE_METHODS + EXPERIMENTAL_MERGE_METHODS]
    # Data-required merges (dataloader built by eval_tofu); extended manifest only.
    labels += [f"merged_{m}" for m in ("fisher", "lorahub")]
    labels += [f"remerge_{m}" for m in ("fisher", "lorahub")]
    labels += ["subtract_linear", "subtract_orth"]
    # JD compresses across the whole collection, so it is not tree-eligible (pairwise nodes).
    tree_methods = [m for m in DEFAULT_MERGE_METHODS + EXPERIMENTAL_MERGE_METHODS
                    if m not in ("jd_full", "jd_diag")]
    labels += [f"tree_root_{m}" for m in tree_methods]
    labels += [f"tree_remerge_{m}" for m in tree_methods]
    labels += ["tree_root_slerp", "tree_remerge_slerp"]  # slerp is pairwise: tree-only
    for rl in DEFAULT_ROUTING_LABELS:
        labels += [rl, f"{rl}_no{forget_id}"]
    return labels


def smoke_eval_labels(k, forget_id):
    """Smoke manifest: all shard-only + default/experimental merged/remerge + subtract + tree + fast routing.

    Data-required merges (fisher/lorahub/regmean) are extended-only.
    """
    labels = [f"shard_{i}_only" for i in range(k)]
    labels += [f"merged_{m}" for m in DEFAULT_MERGE_METHODS + EXPERIMENTAL_MERGE_METHODS]
    labels += [f"remerge_{m}" for m in DEFAULT_MERGE_METHODS + EXPERIMENTAL_MERGE_METHODS]
    labels += ["subtract_linear", "subtract_orth"]
    labels += [f"tree_root_{m}" for m in DEFAULT_MERGE_METHODS]
    labels += [f"tree_remerge_{m}" for m in DEFAULT_MERGE_METHODS]
    labels += ["tree_root_slerp", "tree_remerge_slerp"]
    for rl in SMOKE_ROUTING_LABELS:
        labels += [rl, f"{rl}_no{forget_id}"]
    return labels
