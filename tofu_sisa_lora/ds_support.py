"""Disjoint-support full-parameter task vectors — the composable_tv [ds] arm.

SIFT-Masks (sift_masks.py) constrains each per-author full-FT task vector by a global
±1 SIGN vector; its *maskless* merged serving collapses at scale (T=200: merge_full mu
0.407 vs sift_full 0.737, log/sift_masks/) because merged weights are non-zero where a
task had zeros — parameter contamination: every author's entries land on every other
author's serving. This arm replaces the sign constraint with a hard, seeded,
DATA-INDEPENDENT DISJOINT-SUPPORT constraint:

    Pool slot `a` may only update a fixed index set S_a of the trainable parameters;
    the S_a are pairwise disjoint BY CONSTRUCTION — per tensor, a seeded permutation of
    the flat indices is partitioned into `pool_size` blocks of floor(numel*density)
    each, and slot a owns block [a*per, (a+1)*per). After every optimizer step the
    model is projected back (off-support entries reset to θ0 — SIFT's
    project-AFTER-step convention, see sift_masks.sift_one_task), so τ_a = θ_a − θ0
    lives entirely inside S_a.

Consequences (all asserted in test_ds_support.py):
  * merged weights θ0 + Σ_c τ_c restricted to S_a are EXACTLY τ_a — no contamination;
  * serving is MERGE-ONLY: one dense model, no per-task mask, no task id at serve time;
  * deletion = merge_sub_ (the sift precedent) and is BITWISE equal to recomposing
    without the deleted author — stronger than SIFT's allclose exactness: on S_a the
    running sum holds τ_a's exact fp values (x − x = +0.0) and off S_a τ_a is +0.0
    (x ± 0.0 = x). Equivalently deletion is zeroing S_a in τ̄ — O(1), no re-derivation;
    the stored sparse τ_a (~39 MB/author at density 5e-3 on the 1B's ~0.97 B trainable
    params) makes subtract-from-file the primary deletion path.

Support determinism: the per-tensor permutation is seeded from
SHA-256("<support_seed>:<tensor_name>") — author-independent AND data-independent, so
S_a leaks nothing about author a's data and never shifts when the pool's data changes.

Rejected design paths:
  * independent per-author random supports — pairwise collision probability ≈ density,
    which breaks the exactness claim; the partitioned global permutation is disjoint by
    construction, no birthday problem.
  * keeping SIFT's sign vector ON TOP of the support — needless: disjointness alone
    guarantees decontamination, and the sign constraint would halve usable directions.
  * dense per-author τ storage — sift stores only τ̄ and re-derives for deletion; we
    keep per-author τ sparse ((int32 idx, fp32 val) = 8·density bytes/param) so
    deletion needs no training determinism at all.

Serving for eval: the primary path is IN-PLACE — `load_ds_eval_model()` builds
θ0 + Σ τ_c (− Σ subtract) in memory for `eval_tofu.py --ds_config` (`ds:authors=…` /
`ds:n=…` serve-specs in submit_ctv.sh manifests); `bake_merged()` writes the SAME
bytes as a plain dense checkpoint dir via save_pretrained (the make_scaffolded_base
precedent) for headline bakes — eval it like a base model (`eval_baseline.py
--model_name <baked dir>` or any `--model_name` seam). No wrapper, no routing either
way. CLI: `python ds_support.py locality --config C` (the verify-stage gate: stored
idx ⊆ derived S_a, disjointness, empty-tau telemetry; nonzero exit on violation) and
`python ds_support.py bake --config C (--n N | --authors ids) [--subtract ids] --out D`.
Result labels follow the ctv grammar: ctv_ds_sum_N{n}_s42.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import torch

import sift_masks as sm

import sys

# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

ParamDict = Dict[str, torch.Tensor]
SupportMask = Dict[str, torch.Tensor]        # {tensor_name: sorted 1-D LongTensor of flat idx}

# Llama MLP tensors carry ".mlp." in their names (gate/up/down_proj); GPT-2 uses ".mlp." too.
MLP_SUBSTR = "mlp"

# int32 sparse indices require every trainable tensor to be flat-addressable in 31 bits
# (Llama-3.2-1B's largest trainable tensor is 2048*8192 = 16.8M — comfortable).
_INT32_MAX = 2 ** 31


def is_mlp_tensor(name: str) -> bool:
    return MLP_SUBSTR in name


def shapes_from_model(model, names: List[str]) -> Dict[str, Tuple[int, ...]]:
    sd = dict(model.named_parameters())
    return {n: tuple(sd[n].shape) for n in names}


# ── seeded disjoint supports ────────────────────────────────────────────────────


def _sha_seed(tag: str) -> int:
    """Deterministic 64-bit generator seed from a string tag (SHA-256, first 8 bytes)."""
    return int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big")


def _tensor_perm(support_seed: int, name: str, numel: int) -> torch.Tensor:
    """The ONE global permutation of a tensor's flat indices, shared by all authors.

    Seeded from SHA-256("<support_seed>:<tensor_name>") so it is independent of the
    model's parameter ordering, of the author, and of any data.
    """
    g = torch.Generator(device="cpu").manual_seed(_sha_seed(f"{support_seed}:{name}"))
    return torch.randperm(numel, generator=g)


def _check_capacity(pool_size: int, density: float) -> None:
    # tolerance absorbs fp fuzz like 200*0.005 -> 1.0000000000000002
    if pool_size * density > 1.0 + 1e-12:
        raise ValueError(
            f"support capacity exceeded: pool_size*density = {pool_size}*{density} "
            f"= {pool_size * density:.6g} > 1.0 — disjoint supports cannot exist")


def per_author_count(numel: int, density: float) -> int:
    return int(math.floor(numel * density))


def support_sizes(tensor_shapes: Dict[str, Tuple[int, ...]], density: float,
                  mlp_only: bool = False) -> Dict[str, int]:
    """Per-tensor owned-index count for ONE author: floor(numel*density); 0 off-MLP
    when mlp_only (the ROME/MEMIT locality ablation), 0 when the tensor is too small."""
    sizes = {}
    for n, shape in tensor_shapes.items():
        numel = int(math.prod(shape)) if len(shape) else 1
        sizes[n] = 0 if (mlp_only and not is_mlp_tensor(n)) else per_author_count(numel, density)
    return sizes


def support_masks(support_seed: int, pool_size: int, density: float,
                  tensor_shapes: Dict[str, Tuple[int, ...]],
                  mlp_only: bool = False) -> List[SupportMask]:
    """All pool slots' supports: list (len pool_size) of {name: sorted flat LongTensor}.

    Disjoint by construction (block partition of one permutation per tensor). Each
    slot's indices are SORTED — set semantics, canonical order for sparse storage.
    NB: keys are POOL SLOTS (0..pool_size-1), not TOFU author ids; the trainer maps
    author -> slot via its position in merge_subset.subset_authors(pool_seed, pool_size).
    """
    _check_capacity(pool_size, density)
    sizes = support_sizes(tensor_shapes, density, mlp_only)
    supports: List[SupportMask] = [dict() for _ in range(pool_size)]
    for name in sorted(tensor_shapes):
        shape = tensor_shapes[name]
        numel = int(math.prod(shape)) if len(shape) else 1
        if numel >= _INT32_MAX:
            raise ValueError(f"{name}: numel {numel} exceeds int32 sparse index range")
        per = sizes[name]
        if per == 0:
            empty = torch.empty(0, dtype=torch.long)
            for s in supports:
                s[name] = empty
            continue
        perm = _tensor_perm(support_seed, name, numel)
        for a in range(pool_size):
            s = perm[a * per:(a + 1) * per].sort().values
            supports[a][name] = s
    return supports


def support_mask_for_slot(support_seed: int, slot: int, pool_size: int, density: float,
                          tensor_shapes: Dict[str, Tuple[int, ...]],
                          mlp_only: bool = False) -> SupportMask:
    """One slot's support without materializing the other pool_size-1 (trainer path)."""
    _check_capacity(pool_size, density)
    if not (0 <= slot < pool_size):
        raise ValueError(f"slot {slot} out of range for pool_size {pool_size}")
    sizes = support_sizes(tensor_shapes, density, mlp_only)
    out: SupportMask = {}
    for name in sorted(tensor_shapes):
        shape = tensor_shapes[name]
        numel = int(math.prod(shape)) if len(shape) else 1
        if numel >= _INT32_MAX:
            raise ValueError(f"{name}: numel {numel} exceeds int32 sparse index range")
        per = sizes[name]
        if per == 0:
            out[name] = torch.empty(0, dtype=torch.long)
            continue
        perm = _tensor_perm(support_seed, name, numel)
        out[name] = perm[slot * per:(slot + 1) * per].sort().values
    return out


# ── projection (the hard constraint) ────────────────────────────────────────────


def project_support_(tau: ParamDict, support: SupportMask) -> None:
    """In-place τ ← τ ⊙ m_a: zero every entry outside the support (dense τ dicts)."""
    for n, t in tau.items():
        assert t.is_contiguous(), f"project_support_ needs contiguous tensors ({n})"
        flat = t.view(-1)
        idx = support.get(n)
        if idx is None or idx.numel() == 0:
            flat.zero_()
            continue
        idx = idx.to(flat.device)
        keep = flat[idx].clone()
        flat.zero_()
        flat[idx] = keep


def project_support_model_(model, theta0: ParamDict, support: SupportMask,
                           names: List[str]) -> None:
    """Reset off-support entries of θ to θ0 (⇔ τ ← τ ⊙ m_a on the live model).

    Mirror of sift_masks.project_ — applied AFTER opt.step() so the stored τ always
    satisfies the constraint; writes new storage like sift's project_ (no aliasing).
    """
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for n in names:
            p = sd[n].data
            base = theta0[n].to(p.device, p.dtype)
            new = base.clone()
            idx = support.get(n)
            if idx is not None and idx.numel() > 0:
                idx = idx.to(p.device)
                new.view(-1)[idx] = p.reshape(-1)[idx]
            sd[n].data = new


# ── per-author training (mirrors sift_masks.sift_one_task) ──────────────────────


def ds_one_task(
    model,
    theta0: ParamDict,
    support: SupportMask,
    names: List[str],
    batch: Dict[str, torch.Tensor],
    *,
    seed: int,
    steps: int,
    lr: float,
    device: str = "cpu",
    loss_log: list = None,
) -> ParamDict:
    """Full-FT one author from θ0 under the hard support constraint; return τ_a.

    Structure/recipe = sift_one_task (full-batch Adam over the author's records, fp32,
    deterministic; project AFTER each step) with the support projection replacing the
    sign projection. Same (seed, θ0, support, batch, steps, lr) ⇒ byte-identical τ_a —
    though deletion here never NEEDS re-derivation (the sparse τ_a is stored).

    support=None runs the SAME recipe with NO projection at all — the unconstrained
    full-FT solo (the H-ds-1 comparator denominator, train_ds_support --no_support).
    The returned τ is then dense; it must never enter a disjoint-support merge.
    """
    sm.set_determinism(seed)
    sm.load_params_(model, theta0, names)            # always start from θ0
    model.train()

    name_set = set(names)
    params = [p for n, p in model.named_parameters() if n in name_set]
    opt = torch.optim.Adam(params, lr=lr)
    batch = {k: v.to(device) for k, v in batch.items()}
    # move once, not per step (None = unconstrained comparator: no projection)
    support_dev = None if support is None else {n: idx.to(device) for n, idx in support.items()}

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        opt.step()
        if support_dev is not None:
            project_support_model_(model, theta0, support_dev, names)  # project AFTER the step
        if loss_log is not None:
            loss_log.append(float(out.loss.detach()))

    return sm.task_vector(model, theta0, names)      # off-support entries are exactly +0.0


# ── sparse (int32 idx, fp32 val) storage — round-trip exact ─────────────────────


def sparsify(tau: ParamDict, support: SupportMask) -> Dict[str, dict]:
    """{name: {idx int32, val fp32, shape}} — τ restricted to the support.

    Exact iff τ satisfies the constraint (verify with the densify round-trip or
    energy_in_support — never assume).
    """
    rec = {}
    for n in sorted(tau):
        t = tau[n]
        assert t.numel() < _INT32_MAX, f"{n}: too large for int32 sparse indices"
        idx = support.get(n)
        if idx is None:
            idx = torch.empty(0, dtype=torch.long)
        rec[n] = {
            "idx": idx.to(torch.int32).cpu(),
            "val": t.reshape(-1)[idx.to(t.device)].to(torch.float32).cpu(),
            "shape": tuple(t.shape),
        }
    return rec


def densify(sparse: Dict[str, dict]) -> ParamDict:
    out: ParamDict = {}
    for n, r in sparse.items():
        t = torch.zeros(r["shape"], dtype=torch.float32)
        if r["idx"].numel():
            t.view(-1)[r["idx"].to(torch.long)] = r["val"]
        out[n] = t
    return out


def save_sparse_tau(path: str, sparse: Dict[str, dict]) -> None:
    torch.save(sparse, path)


def load_sparse_tau(path: str) -> Dict[str, dict]:
    # payload is tensors + primitive containers only -> the safe loader suffices
    return torch.load(path, map_location="cpu", weights_only=True)


def sparse_nbytes(sparse: Dict[str, dict]) -> int:
    return sum(r["idx"].numel() * 4 + r["val"].numel() * 4 for r in sparse.values())


# ── merge / subtract / serve ────────────────────────────────────────────────────
# Dense taus reuse sift_masks.merge_init / merge_add_ / merge_sub_ unchanged (imported
# here for one-stop use). The sparse wrappers below accumulate straight from storage;
# by disjointness both routes are BITWISE identical and order-independent.

merge_init = sm.merge_init
merge_add_ = sm.merge_add_
merge_sub_ = sm.merge_sub_


def add_sparse_(tau_bar: ParamDict, sparse: Dict[str, dict], sign: float = 1.0) -> None:
    for n, r in sparse.items():
        if r["idx"].numel() == 0:
            continue
        flat = tau_bar[n].view(-1)
        idx = r["idx"].to(torch.long).to(flat.device)
        val = r["val"].to(flat.device, flat.dtype)
        if sign >= 0:
            flat[idx] += val
        else:
            flat[idx] -= val


def merge_sparse_add_(tau_bar: ParamDict, sparse: Dict[str, dict]) -> None:
    add_sparse_(tau_bar, sparse, sign=1.0)


def merge_sparse_sub_(tau_bar: ParamDict, sparse: Dict[str, dict]) -> None:
    """Deletion: τ̄ ← τ̄ − τ_u. Bitwise == recompose-without under disjoint supports."""
    add_sparse_(tau_bar, sparse, sign=-1.0)


def serve_merged_sum_(model, theta0: ParamDict, tau_bar: ParamDict,
                      names: List[str]) -> None:
    """MERGE-ONLY serving: θ ← θ0 + τ̄ (no mask, no /T — T=1 in sift's serve_merged_)."""
    sm.serve_merged_(model, theta0, tau_bar, names, T=1)


# ── locality probes / eval helpers ──────────────────────────────────────────────


def energy_in_support(tau: ParamDict, support: SupportMask) -> float:
    """Fraction of ‖τ‖² inside the support (fp64 sums). 1.0 by construction after
    projection — a verification probe, not an assumption. All-zero τ returns 1.0
    (vacuously local); use support_stats for the empty-slice question instead."""
    on = tot = 0.0
    for n, t in tau.items():
        f = t.detach().reshape(-1).to(torch.float64)
        tot += float((f * f).sum())
        idx = support.get(n)
        if idx is not None and idx.numel():
            v = f[idx.to(f.device)]
            on += float((v * v).sum())
    return 1.0 if tot == 0.0 else on / tot


def support_stats(supports: List[SupportMask]) -> List[dict]:
    """Per-slot support-size distribution — the empty-slice check (tiny tensors get
    floor(numel*density)=0 owned indices; mlp_only empties everything off-MLP)."""
    rows = []
    for a, s in enumerate(supports):
        sizes = {n: int(i.numel()) for n, i in s.items()}
        rows.append({
            "slot": a,
            "n_indices": int(sum(sizes.values())),
            "n_tensors": len(sizes),
            "empty_tensors": sorted(n for n, c in sizes.items() if c == 0),
        })
    return rows


def empty_slices(supports: List[SupportMask]) -> List[Tuple[int, str]]:
    return [(r["slot"], n) for r in support_stats(supports) for n in r["empty_tensors"]]


def region_sizes(support: SupportMask) -> Dict[str, int]:
    return {n: int(i.numel()) for n, i in support.items()}


def placebo_region(placebo_seed: int, tensor_shapes: Dict[str, Tuple[int, ...]],
                   sizes: Dict[str, int]) -> SupportMask:
    """Equal-size seeded RANDOM region (the cross-talk placebo): per tensor, the first
    sizes[name] indices of a permutation seeded from SHA-256("placebo:<seed>:<name>").

    The "placebo:" namespace decorrelates it from the support permutation even at the
    same integer seed. The region may overlap real author supports — deliberately: the
    control asks what zeroing an equally-sized RANDOM slice of τ̄ does vs zeroing S_a.
    """
    region: SupportMask = {}
    for n in sorted(tensor_shapes):
        k = int(sizes.get(n, 0))
        if k == 0:
            region[n] = torch.empty(0, dtype=torch.long)
            continue
        shape = tensor_shapes[n]
        numel = int(math.prod(shape)) if len(shape) else 1
        g = torch.Generator(device="cpu").manual_seed(_sha_seed(f"placebo:{placebo_seed}:{n}"))
        region[n] = torch.randperm(numel, generator=g)[:k].sort().values
    return region


def zero_region_(tau_bar: ParamDict, region: SupportMask) -> None:
    """Zero a flat-index region of τ̄ in place. With region = S_a this IS the bitwise
    O(1) delete (== merge_sub_(τ̄, τ_a), proven in test_ds_support.py)."""
    for n, idx in region.items():
        if idx.numel():
            tau_bar[n].view(-1)[idx.to(tau_bar[n].device)] = 0.0


def materialize_ablated(theta0: ParamDict, tau_bar: ParamDict, region: SupportMask,
                        names: List[str]) -> ParamDict:
    """θ0 + τ̄ with `region` zeroed, as a fresh state dict over the trainable names
    (τ̄ untouched). Pass S_a for the real deletion, placebo_region(...) for the control;
    load with sift_masks.load_params_ for eval."""
    out: ParamDict = {}
    for n in names:
        t = tau_bar[n].clone()
        idx = region.get(n)
        if idx is not None and idx.numel():
            t.view(-1)[idx.to(t.device)] = 0.0
        out[n] = theta0[n].to(t.device, t.dtype) + t
    return out


# ── bake / serve: merged model from the stored sparse taus ──────────────────────


def build_merged_model(base_model_name: str, tau_dirs: Sequence[str],
                       subtract: Sequence[str] = (), *,
                       frozen_substr: Tuple[str, ...] = ("embed_tokens", "lm_head"),
                       hf_home: str = None):
    """Build θ0 + Σ_{tau_dirs} τ − Σ_{subtract} τ IN MEMORY and return the fp32 model.

    The shared core of bake_merged (which saves it) and load_ds_eval_model (which
    serves it in place). Deletion: build(all, subtract=[dir]) is BITWISE identical to
    build(all minus dir) — adds and subtracts cancel inside a ZERO-initialized
    per-tensor accumulator (v − v = +0.0) before ONE dense add into θ0; applying them
    straight into the base weights was rejected — (b + v) − v is not bitwise b in fp.

    Consistency guards: all metas must agree on (support_seed, density, mlp_only,
    pool_size) and carry distinct slots — mixing densities/seeds would silently break
    disjointness, which is the entire exactness claim.
    """
    from transformers import AutoModelForCausalLM
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    keys = None
    slots = []
    for d in list(tau_dirs) + list(subtract):
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            continue                                  # hand-built dirs (tests) may omit meta
        with open(mp) as f:
            meta = json.load(f)
        k = (meta.get("support_seed"), meta.get("density"),
             meta.get("mlp_only"), meta.get("pool_size"))
        if keys is None:
            keys = k
        elif k != keys:
            raise ValueError(f"inconsistent support provenance in {d}: {k} != {keys}")
        if "slot" in meta:
            slots.append((d, meta["slot"]))
    seen = {}
    for d, s in slots:
        if s in seen and seen[s] != d:
            raise ValueError(f"duplicate pool slot {s}: {seen[s]} and {d}")
        seen[s] = d

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32, trust_remote_code=True)
    sd = dict(model.named_parameters())
    recs = []                                         # (sign, sparse) — sorted walk
    for sign, dirs in ((1.0, sorted(tau_dirs)), (-1.0, sorted(subtract))):
        for d in dirs:
            sparse = load_sparse_tau(os.path.join(d, "tau_sparse.pt"))
            for n, r in sparse.items():
                assert n in sd, f"{d}: unknown tensor {n}"
                assert not any(s in n for s in frozen_substr), f"{d}: frozen tensor {n}"
                assert tuple(sd[n].shape) == tuple(r["shape"]), f"{d}: shape mismatch {n}"
            recs.append((sign, sparse))
    touched = sorted({n for _, sp in recs for n in sp})
    with torch.no_grad():
        for n in touched:
            p = sd[n].data
            acc = torch.zeros(p.numel(), dtype=p.dtype)     # cancel HERE, not in θ0
            for sign, sp in recs:
                r = sp.get(n)
                if r is None or r["idx"].numel() == 0:
                    continue
                idx = r["idx"].to(torch.long)
                if sign >= 0:
                    acc[idx] += r["val"]
                else:
                    acc[idx] -= r["val"]
            p.view(-1).add_(acc)
    return model


def bake_dense_model(out_dir: str, base_model_name: str, tau: ParamDict, *,
                     frozen_substr: Tuple[str, ...] = ("embed_tokens", "lm_head"),
                     hf_home: str = None, save_tokenizer: bool = True) -> str:
    """Materialize θ0 + τ for ONE in-memory DENSE tau as a save_pretrained dir.

    build_merged_model-equivalent for the unconstrained (--no_support) H-ds-1
    comparator, whose dense τ (~4 GB fp32 on the 1B) is never stored — it is baked
    in-job instead. Deliberately a separate helper: build_merged_model's signature
    and sparse-dir/provenance behavior stay untouched. Same guards (known tensor,
    not frozen, shape match); θ0 is loaded FRESH so baked == θ0 + τ elementwise.
    Serve via any --model_name seam (eval_baseline.py; the driver's model: rows).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32, trust_remote_code=True)
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for n in sorted(tau):
            t = tau[n]
            assert n in sd, f"unknown tensor {n}"
            assert not any(s in n for s in frozen_substr), f"frozen tensor {n}"
            assert tuple(sd[n].shape) == tuple(t.shape), f"shape mismatch {n}"
            sd[n].data.add_(t.to(sd[n].device, sd[n].dtype))

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    if save_tokenizer:
        try:
            AutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True).save_pretrained(out_dir)
        except Exception as e:                        # tiny test bases have no tokenizer
            print(f"[bake_dense_model] no tokenizer saved ({type(e).__name__}: {e})")
    print(f"[bake_dense_model] wrote theta0 + dense tau ({len(tau)} tensors) -> {out_dir}")
    return out_dir


def bake_merged(out_dir: str, base_model_name: str, tau_dirs: Sequence[str],
                subtract: Sequence[str] = (), *,
                frozen_substr: Tuple[str, ...] = ("embed_tokens", "lm_head"),
                hf_home: str = None, save_tokenizer: bool = True) -> str:
    """Materialize θ0 + Σ_{tau_dirs} τ − Σ_{subtract} τ as a full save_pretrained dir.

    Each tau dir holds tau_sparse.pt (+ meta.json from train_ds_support.py). Serving
    the bake is the make_scaffolded_base precedent: point any --model_name at out_dir
    (eval_baseline.py is the natural harness — the bake IS a plain model). Deletion:
    bake(all, subtract=[dir]) is BITWISE identical to bake(all minus dir), so either
    drop the dir or pass it in subtract — same bytes (see build_merged_model, which
    holds the accumulation math + provenance guards).
    """
    from transformers import AutoTokenizer
    model = build_merged_model(base_model_name, tau_dirs, subtract,
                               frozen_substr=frozen_substr, hf_home=hf_home)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    if save_tokenizer:
        try:
            AutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True).save_pretrained(out_dir)
        except Exception as e:                        # tiny test bases have no tokenizer
            print(f"[bake_merged] no tokenizer saved ({type(e).__name__}: {e})")
    meta = {
        "method": "ds_support_bake", "arm": "ds",
        "base_model_name": base_model_name,
        "tau_dirs": [os.path.abspath(d) for d in sorted(tau_dirs)],
        "subtract": [os.path.abspath(d) for d in sorted(subtract)],
        "frozen_substr": list(frozen_substr),
    }
    with open(os.path.join(out_dir, "bake_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[bake_merged] wrote {len(tau_dirs)} adds - {len(subtract)} subs -> {out_dir}")
    return out_dir


# ── eval_tofu loader (the --ds_config seam) ─────────────────────────────────────


def _id_list(v) -> List[int]:
    """Comma string / list / None -> list[int] (mirror of linear_tv._id_list)."""
    if v is None:
        return []
    if isinstance(v, str):
        return [int(x) for x in v.split(",") if x.strip()]
    return [int(x) for x in v]


def resolve_selection(cfg, *, authors=None, n=None, subtract=None):
    """Pure resolution of the --ds_* selection into (tau_dirs, subtract_dirs).

    Mirrors linear_tv.resolve_compose: authors = comma string / list of TOFU author
    ids; n = the FIRST N of the config pool (merge_subset.subset_authors(pool_seed, N)
    — pools derive at runtime, never hardcoded); exactly one of authors/n required;
    subtract ids must be pool members (they map to trained tau dirs). Author ids map
    to dirs via the train_ds_support.tau_dir convention; every dir must hold
    tau_sparse.pt (loud error — the load_prefix_concat_model no-silent-skip rule).
    Imports are lazy: train_ds_support imports ds_support at top (module cycle).
    """
    from merge_subset import subset_authors
    import train_ds_support as tds

    pool = [int(a) for a in subset_authors(int(cfg["pool_seed"]), int(cfg["pool_size"]))]
    pos = _id_list(authors)
    if n is not None:
        if pos:
            raise ValueError("pass either authors or n, not both")
        if not (1 <= int(n) <= len(pool)):
            raise ValueError(f"n={n} outside [1, pool_size={len(pool)}]")
        pos = pool[: int(n)]
    if not pos:
        raise ValueError("ds_support: no authors selected (pass authors or n)")
    neg = _id_list(subtract)
    bad = [a for a in neg if a not in pool]
    if bad:
        raise ValueError(f"subtract ids not in the pool: {bad} (pool = {pool})")

    tau_dirs = [tds.tau_dir(cfg, a) for a in pos]
    sub_dirs = [tds.tau_dir(cfg, a) for a in neg]
    missing = [d for d in tau_dirs + sub_dirs
               if not os.path.exists(os.path.join(d, "tau_sparse.pt"))]
    if missing:
        raise FileNotFoundError(f"ds tau dir(s) missing tau_sparse.pt: {missing}")
    return tau_dirs, sub_dirs


def load_ds_eval_model(cfg, *, authors=None, n=None, subtract=None,
                       hf_home: str = None, device: str = None):
    """Build (merged dense model, tokenizer) for eval_tofu's --ds_config flags.

    Selection semantics live in resolve_selection (see its docstring). The served
    object is a PLAIN fp32 CausalLM — merge-only serving, no wrapper, no routing —
    with a no-op set_adapter attached (the eval-seam contract shared with
    SiftMasksModel/LinearTVModel). Bitwise identical weights to bake_merged's dir
    (both call build_merged_model), just never written to disk.
    """
    from transformers import AutoTokenizer

    tau_dirs, sub_dirs = resolve_selection(cfg, authors=authors, n=n, subtract=subtract)
    frozen = tuple(cfg.get("frozen_substr", ("embed_tokens", "lm_head")))
    model = build_merged_model(cfg["model_name"], tau_dirs, sub_dirs,
                               frozen_substr=frozen,
                               hf_home=hf_home or cfg.get("hf_home"))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    model.set_adapter = lambda *a, **k: None      # eval-seam contract: selection is global

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return model, tok


# ── CLI: locality gate + headline bake ──────────────────────────────────────────


def _locality_report_path(out_dir: str, density: float, cfg_density: float,
                          out_override: str = None) -> str:
    base = out_override or os.path.join(out_dir, "reports", "ds_locality.json")
    if density == cfg_density:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}_d{density:g}{ext or '.json'}"


def cli_locality(args) -> int:
    """Verify (never assume) the disjoint-support invariants of every stored tau.

    Per pool author at the config density (+ any --densities present on disk):
      * stored idx ⊆ the re-derived owned set S_a (support_mask_for_slot) — the
        energy-in-owned = 1.0 claim, verified from the on-disk artifact;
      * per-author owned-norm distribution; flag empties (norm < 1e-3 × median —
        a frozen/silent-failure telemetry check, reported as warnings);
      * pairwise disjointness across authors per tensor (exact set ops on idx).
    Writes reports/ds_locality[_d*].json under out_dir; EXIT NONZERO on any
    violation — the driver's verify stage gates merge/eval on this.
    """
    import train_ds_support as tds

    cfg = tds.load_config(args.config)
    pool = [int(a) for a in tds.pool_authors(cfg)]
    cfg_density = cfg["density"]
    densities = [cfg_density]
    for d in _id_float_list(args.densities):
        if d not in densities:
            densities.append(d)

    overall_rc = 0
    for density in densities:
        violations, warnings, rows = [], [], []
        per_tensor_idx: Dict[str, list] = {}
        shapes = None
        present = []
        for a in pool:
            tau_d = tds.tau_dir(cfg, a, density)
            sp_path = os.path.join(tau_d, "tau_sparse.pt")
            if not os.path.exists(sp_path):
                if density == cfg_density:
                    violations.append(f"author {a}: missing {sp_path}")
                continue
            present.append((a, tau_d, sp_path))
        if density != cfg_density and not present:
            print(f"[locality] density {density:g}: nothing on disk — skipped")
            continue

        for a, tau_d, sp_path in present:
            slot = pool.index(a)
            sparse = load_sparse_tau(sp_path)
            if shapes is None:
                shapes = {n: tuple(r["shape"]) for n, r in sparse.items()}
            meta_path = os.path.join(tau_d, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                for key, want in (("support_seed", cfg["support_seed"]),
                                  ("density", density),
                                  ("mlp_only", bool(cfg.get("mlp_only", False))),
                                  ("pool_size", cfg["pool_size"])):
                    if key in meta and meta[key] != want:
                        violations.append(
                            f"author {a}: meta {key}={meta[key]!r} != config {want!r}")
                if meta.get("slot", slot) != slot:
                    violations.append(
                        f"author {a}: meta slot {meta['slot']} != pool slot {slot}")
            owned = support_mask_for_slot(
                cfg["support_seed"], slot, int(cfg["pool_size"]), density,
                shapes, bool(cfg.get("mlp_only", False)))
            n_idx, on64, tot64 = 0, 0.0, 0.0
            subset_ok = True
            for tname, r in sparse.items():
                idx = r["idx"].to(torch.long)
                val = r["val"].to(torch.float64)
                n_idx += int(idx.numel())
                tot64 += float((val * val).sum())
                if tuple(r["shape"]) != shapes.get(tname):
                    violations.append(f"author {a}: shape mismatch on {tname}")
                    subset_ok = False
                    continue
                own = owned.get(tname, torch.empty(0, dtype=torch.long))
                inside = torch.isin(idx, own) if idx.numel() else torch.ones(0, dtype=torch.bool)
                if idx.numel() and not bool(inside.all()):
                    n_out = int((~inside).sum())
                    violations.append(
                        f"author {a}: {n_out} stored idx OUTSIDE owned S_a on {tname}")
                    subset_ok = False
                on64 += float((val[inside] * val[inside]).sum()) if idx.numel() else 0.0
                per_tensor_idx.setdefault(tname, []).append((a, idx))
            energy = 1.0 if tot64 == 0.0 else on64 / tot64
            if subset_ok and abs(energy - 1.0) > 1e-12:
                violations.append(f"author {a}: energy_in_owned {energy} != 1.0")
            rows.append({"author": a, "slot": slot, "n_indices": n_idx,
                         "owned_norm": math.sqrt(tot64), "subset_ok": subset_ok,
                         "energy_in_owned": energy})

        # empty-tau telemetry (norm < 1e-3 x median) — silent-failure flag, not a gate
        norms = sorted(r["owned_norm"] for r in rows)
        if norms:
            med = norms[len(norms) // 2] if len(norms) % 2 else (
                0.5 * (norms[len(norms) // 2 - 1] + norms[len(norms) // 2]))
            for r in rows:
                if r["owned_norm"] < 1e-3 * med:
                    r["empty_flag"] = True
                    warnings.append(f"author {r['author']}: owned_norm "
                                    f"{r['owned_norm']:.3e} < 1e-3 x median {med:.3e}")

        # pairwise disjointness across authors, per tensor (exact set ops)
        disjoint_ok = True
        for tname, pairs in sorted(per_tensor_idx.items()):
            cat = torch.cat([idx for _, idx in pairs]) if pairs else torch.empty(0, dtype=torch.long)
            if cat.numel() != int(cat.unique().numel()):
                disjoint_ok = False
                violations.append(
                    f"{tname}: stored idx sets OVERLAP across authors "
                    f"({cat.numel() - int(cat.unique().numel())} colliding indices)")

        report = {
            "config": os.path.abspath(args.config), "density": density,
            "config_density": cfg_density, "pool": pool,
            "support_seed": cfg["support_seed"], "pool_size": cfg["pool_size"],
            "mlp_only": bool(cfg.get("mlp_only", False)),
            "authors": rows, "disjoint_ok": disjoint_ok,
            "violations": violations, "warnings": warnings,
        }
        path = _locality_report_path(cfg["out_dir"], density, cfg_density, args.out)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        status = "OK" if not violations else f"{len(violations)} VIOLATION(S)"
        print(f"[locality] density {density:g}: {len(rows)} taus, disjoint={disjoint_ok}, "
              f"{status}{f', {len(warnings)} warning(s)' if warnings else ''} -> {path}")
        for v in violations:
            print(f"[locality]   VIOLATION: {v}")
        for w in warnings:
            print(f"[locality]   warning: {w}")
        if violations:
            overall_rc = 1
    return overall_rc


def _id_float_list(v) -> List[float]:
    if not v:
        return []
    return [float(x) for x in str(v).split(",") if x.strip()]


def cli_bake(args) -> int:
    """Resolve the selection like load_ds_eval_model and write the dense bake (G4)."""
    import train_ds_support as tds

    cfg = tds.load_config(args.config)
    tau_dirs, sub_dirs = resolve_selection(
        cfg, authors=args.authors, n=args.n, subtract=args.subtract)
    bake_merged(args.out, cfg["model_name"], tau_dirs, subtract=sub_dirs,
                frozen_substr=tuple(cfg.get("frozen_substr",
                                            ("embed_tokens", "lm_head"))),
                hf_home=cfg.get("hf_home"))
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("locality", help="verify stored taus against the derived "
                                        "disjoint supports; nonzero exit on violation")
    l.add_argument("--config", required=True, help="ctv ds config (configs/ctv_1b_ds.json)")
    l.add_argument("--out", default=None,
                   help="report path override (default {out_dir}/reports/ds_locality.json; "
                        "non-config densities get a _d<density> suffix)")
    l.add_argument("--densities", default=None,
                   help="comma list of EXTRA densities to probe where present on disk "
                        "(the density_sweep dirs); the config density is always probed")

    b = sub.add_parser("bake", help="materialize theta0 + sum(tau) as a dense model dir "
                                    "(the G4 headline bake; served via --model_name)")
    b.add_argument("--config", required=True)
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int, help="compose the FIRST N pool authors")
    g.add_argument("--authors", help="comma-separated author ids to compose")
    b.add_argument("--subtract", default=None,
                   help="comma-separated pool author ids subtracted (== dropped, bitwise)")
    b.add_argument("--out", required=True, help="destination model dir")

    args = ap.parse_args(argv)
    if args.cmd == "locality":
        return cli_locality(args)
    if args.cmd == "bake":
        return cli_bake(args)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
