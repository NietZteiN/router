"""SIFT-Masks core library (Kuo et al. 2025, arXiv:2504.04626).

SIFT-Masks = SIgn-Fixed Tuning + Masks: an *exact* unlearning method built on
model merging. The whole method rests on one invariant:

    A single global random sign vector `v ∈ {−1,+1}^d` is drawn ONCE, before any
    training, shared across all tasks. Each task's finetuning is constrained so
    its task vector agrees with `v` (or is zero). The task mask is then
    `m_t = 1{τ_t ⊙ v > 0}` — it depends ONLY on local data and the public `v`,
    never on the merged model or other tasks. That independence is what keeps
    deletion cheap and exact.

Pipeline (per the paper):
  1. SIFT(t, v): full-FT from θ0 on task t; after EACH optimizer step project the
     task vector τ_t ← τ_t ⊙ 1{τ_t⊙v>0} (clip sign-disagreeing entries to 0).
  2. Merge: τ̄ = Σ_t τ_t  (a plain SUM). Store only τ̄ + per-task bit masks {m_t}.
  3. Serve task t: θ0 + (τ̄ ⊙ m_t)/T.   (held-out / unlearned task: θ0 + τ̄/T, no mask)
  4. Unlearn u: deterministically re-derive τ_u, set τ̄ ← τ̄ − τ_u, drop m_u, T ← T−1.

Faithfulness / correctness notes (see CLAUDE_SCRATCHPAD.md):
  * Sign vector is ±1, not {0,1}: the paper's prose ("share the same sign as v")
    and Figure 2 are unambiguous; Alg-1's `1{rand>0.5}` is the only inconsistency.
  * Projection is applied AFTER opt.step() (the paper's *prose*), so the stored τ
    always satisfies the constraint and `mask == (τ != 0)`. (Alg-1 lists the
    projection between backward() and step(), which would leave the final step
    unprojected — a pseudocode quirk.)
  * We optimize the *model's* parameters with Adam (not an external τ tensor), so
    `opt.zero_grad()` clears exactly the grads that `backward()` populates — this
    avoids the silent grad-accumulation trap of the "optimize τ + scatter grads"
    sketch.
  * Exact unlearning hinges on DETERMINISM: re-deriving τ_u with the same seed,
    init, data and step count reproduces it byte-for-byte. Note τ̄−τ_u is exact in
    real arithmetic but, as a running-sum-minus-term, is NOT guaranteed bit-equal
    to a from-scratch retain-set sum (fp addition is non-associative). The
    unlearning guarantee (no remaining τ_u contribution) holds at the
    deterministic/algebraic level; bit-equality to a fresh retain-merge would need
    a fixed-order reduction.

Everything operates on per-named-parameter dicts of the *trainable* params (sorted
by name for a deterministic ordering), so we never materialize a 1.5B flat vector.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch

# GPT2: tied lm_head shares wte; freeze token + positional embeddings and the head.
# (For other arches pass your own `frozen_substr`, e.g. ("embed_tokens", "lm_head").)
GPT2_FROZEN_SUBSTR = ("wte", "wpe", "lm_head")

ParamDict = Dict[str, torch.Tensor]
MaskDict = Dict[str, torch.Tensor]


# ── parameter bookkeeping ───────────────────────────────────────────────────────

def trainable_names(model, frozen_substr: Tuple[str, ...] = GPT2_FROZEN_SUBSTR) -> List[str]:
    """Set requires_grad per the freeze rule and return the trainable names, SORTED.

    Sorting makes the sign-vector draw and every dict iteration order-independent of
    the model's internal parameter ordering — important for reproducibility.
    """
    names = []
    for n, p in model.named_parameters():
        frozen = any(s in n for s in frozen_substr)
        p.requires_grad_(not frozen)
        if not frozen:
            names.append(n)
    return sorted(names)


def snapshot_params(model, names: List[str]) -> ParamDict:
    """θ0: detached clones of the base trainable weights (same device/dtype)."""
    sd = dict(model.named_parameters())
    return {n: sd[n].detach().clone() for n in names}


def load_params_(model, theta: ParamDict, names: List[str]) -> None:
    """Write `theta` into the model's trainable params in place."""
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for n in names:
            sd[n].data.copy_(theta[n].to(sd[n].device, sd[n].dtype))


# ── the global sign vector ──────────────────────────────────────────────────────

def make_sign_vector(model, names: List[str], seed: int) -> ParamDict:
    """Deterministic per-parameter sign vector v ∈ {−1,+1} (drawn before training).

    Uses a single CPU torch.Generator and the SORTED `names`, so the result is
    identical across runs/machines regardless of named_parameters() ordering.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    sd = dict(model.named_parameters())
    sign: ParamDict = {}
    for n in names:
        r = torch.rand(tuple(sd[n].shape), generator=g, dtype=torch.float32)
        s = torch.where(r > 0.5, 1.0, -1.0)          # ±1, NOT {0,1}
        sign[n] = s.to(sd[n].dtype)
    return sign


# ── SIFT: sign-fixed finetuning of one task ─────────────────────────────────────

def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_(model, theta0: ParamDict, sign: ParamDict, names: List[str]) -> None:
    """Project the task vector onto {τ⊙v ≥ 0}: where (θ−θ0)·v ≤ 0, reset θ ← θ0.

    After this, every surviving entry shares v's sign and the rest are exactly 0,
    so the post-projection mask is exactly 1{τ⊙v>0} == (τ != 0).
    """
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for n in names:
            p = sd[n].data
            base = theta0[n].to(p.device, p.dtype)
            s = sign[n].to(p.device, p.dtype)
            delta = p - base
            keep = (delta * s) > 0
            # new storage (cheap, avoids read/write aliasing in torch.where)
            sd[n].data = torch.where(keep, p, base)


def task_vector(model, theta0: ParamDict, names: List[str]) -> ParamDict:
    sd = dict(model.named_parameters())
    return {n: (sd[n].detach() - theta0[n].to(sd[n].device)).clone() for n in names}


def sift_one_task(
    model,
    theta0: ParamDict,
    sign: ParamDict,
    names: List[str],
    batch: Dict[str, torch.Tensor],
    *,
    seed: int,
    steps: int = 20,
    lr: float = 1e-4,
    device: str = "cpu",
    use_sign_constraint: bool = True,
) -> Tuple[ParamDict, MaskDict]:
    """Run SIFT on one task and return (τ_t, m_t).

    Deterministic: same (seed, theta0, sign, batch, steps, lr) ⇒ byte-identical τ_t.
    This is the primitive the unlearn step re-runs to reproduce τ_u exactly.

    `use_sign_constraint=False` skips the sign projection — the task vector is then a
    plain deterministic full-FT delta (sign may be None). This is the ClAMU reuse path
    (clamu.py): ClAMU does NOT sign-constrain training; it derives masks separately by
    direct optimization. With the default `True` this is unchanged SIFT-Masks.
    """
    set_determinism(seed)
    load_params_(model, theta0, names)               # always start from θ0
    model.train()

    name_set = set(names)
    params = [p for n, p in model.named_parameters() if n in name_set]
    opt = torch.optim.Adam(params, lr=lr)            # Adam over MODEL params
    batch = {k: v.to(device) for k, v in batch.items()}

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        opt.step()
        if use_sign_constraint:
            project_(model, theta0, sign, names)     # project AFTER the step

    tau = task_vector(model, theta0, names)          # already projected (if constrained)
    mask = {n: (tau[n] != 0) for n in names}
    return tau, mask


# ── merge (streaming sum) ───────────────────────────────────────────────────────

def merge_init(theta0: ParamDict, names: List[str]) -> ParamDict:
    return {n: torch.zeros_like(theta0[n]) for n in names}


def merge_add_(tau_bar: ParamDict, tau: ParamDict, names: List[str]) -> None:
    for n in names:
        tau_bar[n] += tau[n].to(tau_bar[n].device, tau_bar[n].dtype)


def merge_sub_(tau_bar: ParamDict, tau: ParamDict, names: List[str]) -> None:
    """Unmerge a task: τ̄ ← τ̄ − τ_u  (the exact-unlearning operation)."""
    for n in names:
        tau_bar[n] -= tau[n].to(tau_bar[n].device, tau_bar[n].dtype)


# ── serving ─────────────────────────────────────────────────────────────────────

def serve_task_(model, theta0, tau_bar, mask: MaskDict, names, T: int) -> None:
    """Serve task t: θ ← θ0 + (τ̄ ⊙ m_t)/T."""
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for n in names:
            m = mask[n].to(tau_bar[n].device, tau_bar[n].dtype)
            served = theta0[n].to(sd[n].device) + (tau_bar[n] * m).to(sd[n].device) / T
            sd[n].data.copy_(served.to(sd[n].dtype))


def serve_merged_(model, theta0, tau_bar, names, T: int) -> None:
    """Held-out / unlearned-task serving: θ ← θ0 + τ̄/T  (the FT+Merge model, no mask)."""
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for n in names:
            served = theta0[n].to(sd[n].device) + tau_bar[n].to(sd[n].device) / T
            sd[n].data.copy_(served.to(sd[n].dtype))


def serve_base_(model, theta0, names) -> None:
    """θ ← θ0 (no task delta) — used for OOD / fully-unlearned queries."""
    load_params_(model, theta0, names)


# ── mask (de)serialization: 1 bit / parameter ──────────────────────────────────

def pack_mask(mask: MaskDict, names: List[str]) -> Dict[str, object]:
    """Bit-pack a boolean mask dict to ~1/32 the size of an fp32 model."""
    out: Dict[str, object] = {}
    for n in names:
        a = mask[n].detach().cpu().numpy().astype(np.uint8).reshape(-1)
        out[n] = {"bits": np.packbits(a), "shape": tuple(mask[n].shape)}
    return out


def unpack_mask(packed: Dict[str, object], names: List[str]) -> MaskDict:
    out: MaskDict = {}
    for n in names:
        rec = packed[n]
        shape = rec["shape"]
        numel = int(np.prod(shape)) if len(shape) else 1
        a = np.unpackbits(rec["bits"])[:numel].reshape(shape)
        out[n] = torch.from_numpy(a.astype(bool))
    return out
