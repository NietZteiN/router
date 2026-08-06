"""Exp 1 (merge-mechanism study): do isolated fact-adapter deltas collide in a shared subspace?

Pure CPU weight analysis — no PEFT, no model load. For a collection of LoRA adapters we
measure, per (layer x module) "slot", how aligned the effective deltas DW_i = scaling_i * B_i A_i
are with each other:

  (a) pairwise cosine of the flattened DW (Frobenius inner product, computed factored),
  (b) principal angles between the column space col(B_i) and the row space row(A_i),
  (c) shared-subspace energy: fraction of each DW_i captured by a single rank-R basis fit to
      the whole collection's B blocks.

against NULLs (random-orthogonal directions, shuffled B-A pairing, replicated-adapter) so a
"high overlap" claim is calibrated. Everything is factored — we never form the dense d_out x d_in
delta. Reuses jd_collection.build_collection_slots / _adapter_scaling and the jd_compress math.

Pre-registered reading (reports/ORIENTATION_2026-06-29.md sec 7): HIGH real-vs-null overlap =>
"overlap + memorization -> interference"; LOW overlap => strengthens "it's memorization, not
similarity". Both outcomes are informative.

CLI:
    python subspace_overlap.py --adapters DIR... --rank 16 --n_null 20 --seed 42 \
        --out reports/subspace_overlap_k10.json --csv reports/subspace_overlap_k10.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import torch

import jd_compress
from jd_collection import build_collection_slots
from jd_compress import Slot, _top_r_subspace_from_blocks

# SUBSPACE_THREADS overrides the conservative 8-thread cap for large-n SLURM CPU runs
# (n=200 x r32 does ~10 full pairwise passes; 8 threads => ~1-1.5 h each).
torch.set_num_threads(int(os.environ.get("SUBSPACE_THREADS", min(8, os.cpu_count() or 1))))

_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_MODTYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")


# ---------------------------------------------------------------------------
# Factored primitives (float64 traces, no dense d_out x d_in delta)
# ---------------------------------------------------------------------------

def _frob_inner(slot: Slot, i: int, j: int) -> float:
    """<DW_i, DW_j>_F = s_i s_j * tr((B_i^T B_j)(A_j A_i^T)) at this slot (r x r traces)."""
    Bi, Ai, si = slot.B[i].double(), slot.A[i].double(), float(slot.scaling[i])
    Bj, Aj, sj = slot.B[j].double(), slot.A[j].double(), float(slot.scaling[j])
    P = Bi.t() @ Bj            # (r_i, r_j)
    Q = Aj @ Ai.t()            # (r_j, r_i)
    return si * sj * torch.trace(P @ Q).item()


def _slot_sqnorm(slot: Slot, i: int) -> float:
    return _frob_inner(slot, i, i)


def _select_slots(slots: dict, *, modtype: str | None = None, layer: int | None = None) -> list:
    names = []
    for name in slots:
        if modtype is not None and not name.endswith(modtype):
            continue
        if layer is not None:
            m = _LAYER_RE.search(name)
            if m is None or int(m.group(1)) != layer:
                continue
        names.append(name)
    return names


def pairwise_inner(slots: dict, slot_names: list | None = None) -> torch.Tensor:
    """n x n Frobenius inner product <DW_i, DW_j> summed over `slot_names` (vectorized).

    Per slot: <DW_i,DW_j> = s_i s_j tr((B_i^T B_j)(A_j A_i^T)) = s_i s_j sum_{a,b} (B_i^T B_j)[a,b]
    (A_i A_j^T)[a,b]. Computed with batched einsum (BLAS) — equals the _frob_inner double loop
    but ~100x faster for the null draws / large n.
    """
    names = slot_names if slot_names is not None else list(slots)
    n = len(next(iter(slots.values())).B)
    r = next(iter(slots.values())).B[0].shape[1]
    inner = torch.zeros(n, n, dtype=torch.float64)
    for name in names:
        slot = slots[name]
        Bs = torch.stack([b.double() for b in slot.B])                 # (n, d_out, r)
        As = torch.stack([a.double() for a in slot.A])                 # (n, r, d_in)
        s = torch.tensor([float(x) for x in slot.scaling], dtype=torch.float64)
        Bcat = Bs.permute(1, 0, 2).reshape(Bs.shape[1], n * r)         # (d_out, n*r)
        Arows = As.reshape(n * r, As.shape[2])                         # (n*r, d_in)
        GB = (Bcat.t() @ Bcat).reshape(n, r, n, r)                     # (B_i^T B_j)[a,b]
        GA = (Arows @ Arows.t()).reshape(n, r, n, r)                   # (A_i A_j^T)[a,b]
        t = (GB * GA).sum(dim=(1, 3))                                  # sum_{a,b} -> (n,n)
        inner += (s.unsqueeze(1) * s.unsqueeze(0)) * t
    return inner


def pairwise_cosine(slots: dict, slot_names: list | None = None) -> torch.Tensor:
    """n x n cosine of the effective deltas summed over `slot_names` (default: all slots)."""
    inner = pairwise_inner(slots, slot_names)
    diag = inner.diagonal().clamp_min(1e-30)
    denom = torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
    return inner / denom


def _orthonormal_cols(M: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of col(M) via QR (drops zero columns)."""
    Q, R = torch.linalg.qr(M.double())
    keep = R.diagonal().abs() > 1e-8
    return Q[:, keep] if keep.any() else Q[:, :1] * 0.0


def principal_angle_cos(slots: dict, slot_names: list | None = None) -> tuple:
    """Mean cos(principal angle) between adapters, for col(B) (output) and row(A) (input).

    Returns (cosB n x n, cosA n x n): each entry is the mean singular value of Q_i^T Q_j over
    the slots in `slot_names` (1 = identical span, 0 = orthogonal). Diagonal = 1.
    """
    names = slot_names if slot_names is not None else list(slots)
    n = len(next(iter(slots.values())).B)
    accB = torch.zeros(n, n, dtype=torch.float64)
    accA = torch.zeros(n, n, dtype=torch.float64)
    cnt = 0
    for name in names:
        slot = slots[name]
        QB = [_orthonormal_cols(slot.B[i]) for i in range(n)]
        QA = [_orthonormal_cols(slot.A[i].t()) for i in range(n)]
        for i in range(n):
            for j in range(i, n):
                for Q, acc in ((QB, accB), (QA, accA)):
                    s = torch.linalg.svdvals(Q[i].t() @ Q[j])
                    val = s.clamp(0, 1).mean().item()
                    acc[i, j] += val
                    if i != j:
                        acc[j, i] += val
        cnt += 1
    if cnt:
        accB /= cnt
        accA /= cnt
    accB.fill_diagonal_(1.0)
    accA.fill_diagonal_(1.0)
    return accB, accA


def shared_subspace_energy(slots: dict, rank: int) -> dict:
    """Fraction of each adapter's delta energy captured by ONE rank-`rank` left-basis per slot.

    Per slot: fit U (d_out x rank) to the dominant subspace of the stacked [s_i B_i] blocks, then
    energy_i = ||U^T (s_i B_i) A_i||^2 / ||s_i B_i A_i||^2 (factored). High mean => the whole
    collection shares one low-rank output subspace. Reported alongside rank/sum(r_i) as the
    "compressibility" baseline (random adapters retain ~ rank/sum(r_i)).
    """
    n = len(next(iter(slots.values())).B)
    num = torch.zeros(n, dtype=torch.float64)
    den = torch.zeros(n, dtype=torch.float64)
    total_r = 0
    for slot in slots.values():
        d_out = slot.B[0].shape[0]
        blocks = [(float(slot.scaling[i]) * slot.B[i].double()) for i in range(n)]
        total_r += sum(b.shape[1] for b in blocks)
        U = _top_r_subspace_from_blocks(blocks, rank, d_out).double()  # (d_out, rank)
        for i in range(n):
            sBi = float(slot.scaling[i]) * slot.B[i].double()          # (d_out, r_i)
            Ai = slot.A[i].double()                                    # (r_i, d_in)
            C = U.t() @ sBi                                            # (rank, r_i)
            AAt = Ai @ Ai.t()                                          # (r_i, r_i)
            num[i] += torch.trace(C @ AAt @ C.t()).item()
            den[i] += torch.trace(sBi.t() @ sBi @ AAt).item()
    frac = (num / den.clamp_min(1e-30))
    n_slots = len(slots)
    return {
        "rank": rank,
        "mean_energy_retained": frac.mean().item(),
        "per_adapter_energy": frac.tolist(),
        "avg_rank_ratio": (rank * n_slots) / max(total_r, 1),  # chance baseline ~ this
    }


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------

def _random_slots(slots: dict, mode: str, seed: int) -> dict:
    """Null collections sharing the real shapes. mode: 'orthogonal' | 'shuffled' | 'replicated'."""
    g = torch.Generator().manual_seed(seed)
    n = len(next(iter(slots.values())).B)
    perm = torch.randperm(n, generator=g).tolist()
    out = {}
    for name, slot in slots.items():
        if mode == "orthogonal":
            B = [torch.randn(slot.B[i].shape, generator=g, dtype=torch.float64) for i in range(n)]
            A = [torch.randn(slot.A[i].shape, generator=g, dtype=torch.float64) for i in range(n)]
        elif mode == "shuffled":           # real B_i with A_{perm[i]} (breaks B-A coupling)
            B = [slot.B[i].double() for i in range(n)]
            A = [slot.A[perm[i]].double() for i in range(n)]
        elif mode == "replicated":          # every adapter = adapter 0 (saturation sanity)
            B = [slot.B[0].double() for _ in range(n)]
            A = [slot.A[0].double() for _ in range(n)]
        else:
            raise ValueError(mode)
        out[name] = Slot(B=B, A=A, scaling=list(slot.scaling))
    return out


def _offdiag_mean(M: torch.Tensor) -> float:
    n = M.shape[0]
    if n < 2:
        return float("nan")
    mask = ~torch.eye(n, dtype=torch.bool)
    return M[mask].mean().item()


def null_summary(slots: dict, mode: str, seed: int, n_null: int, n_angle_null: int = 3) -> dict:
    """Mean +/- sd of off-diagonal cosine / principal-angle-cos over draws.

    Cosine over `n_null` draws (cheap r x r traces); principal angles over the first
    min(n_null, n_angle_null) draws only (QR/SVD over every slot x pair is ~n_null x costlier
    and the random-subspace angle is low-variance, so a few draws pin the floor)."""
    cos_means, angB_means, angA_means = [], [], []
    reps = 1 if mode in ("shuffled", "replicated") else n_null
    ang_reps = 1 if mode in ("shuffled", "replicated") else min(reps, n_angle_null)
    for t in range(reps):
        ns = _random_slots(slots, mode, seed + t)
        cos_means.append(_offdiag_mean(pairwise_cosine(ns)))
        if t < ang_reps:
            aB, aA = principal_angle_cos(ns)
            angB_means.append(_offdiag_mean(aB))
            angA_means.append(_offdiag_mean(aA))

    def ms(xs):
        t = torch.tensor(xs, dtype=torch.float64)
        return {"mean": t.mean().item(), "sd": (t.std().item() if len(xs) > 1 else 0.0), "n": len(xs)}

    return {"cosine": ms(cos_means), "angle_cos_B": ms(angB_means), "angle_cos_A": ms(angA_means)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyze(slots: dict, ids: list, rank: int, seed: int, n_null: int) -> dict:
    cos = pairwise_cosine(slots)
    angB, angA = principal_angle_cos(slots)
    real = {
        "cosine_offdiag_mean": _offdiag_mean(cos),
        "cosine_offdiag_median": cos[~torch.eye(len(ids), dtype=torch.bool)].median().item() if len(ids) > 1 else float("nan"),
        "cosine_offdiag_max": cos[~torch.eye(len(ids), dtype=torch.bool)].abs().max().item() if len(ids) > 1 else float("nan"),
        "angle_cos_B_offdiag_mean": _offdiag_mean(angB),
        "angle_cos_A_offdiag_mean": _offdiag_mean(angA),
    }
    nulls = {m: null_summary(slots, m, seed, n_null) for m in ("orthogonal", "shuffled", "replicated")}
    # z vs the random-orthogonal null (primary chance floor)
    onull = nulls["orthogonal"]
    z = {}
    for key, nk in (("cosine_offdiag_mean", "cosine"),
                    ("angle_cos_B_offdiag_mean", "angle_cos_B"),
                    ("angle_cos_A_offdiag_mean", "angle_cos_A")):
        sd = onull[nk]["sd"]
        z[key] = (real[key] - onull[nk]["mean"]) / sd if sd > 1e-12 else float("inf")

    # per-module-type and per-layer rollups (cosine off-diagonal mean)
    by_modtype, by_layer = {}, {}
    for mt in _MODTYPES:
        names = _select_slots(slots, modtype=mt)
        if names:
            by_modtype[mt] = _offdiag_mean(pairwise_cosine(slots, names))
    layers = sorted({int(_LAYER_RE.search(n).group(1)) for n in slots if _LAYER_RE.search(n)})
    for L in layers:
        names = _select_slots(slots, layer=L)
        if names:
            by_layer[str(L)] = _offdiag_mean(pairwise_cosine(slots, names))

    energy = shared_subspace_energy(slots, rank)
    return {
        "adapter_ids": ids, "n_adapters": len(ids), "rank": rank, "seed": seed, "n_null": n_null,
        "real": real, "nulls": nulls, "z_vs_orthogonal_null": z,
        "cosine_by_modtype": by_modtype, "cosine_by_layer": by_layer,
        "shared_subspace_energy": energy,
        "cosine_matrix": cos.tolist(),
    }


def _write_csv(res: dict, path: str):
    rows = ["scope,key,value"]
    for k, v in res["real"].items():
        rows.append(f"real,{k},{v}")
    for m, d in res["nulls"].items():
        for metric, stat in d.items():
            rows.append(f"null_{m},{metric}_mean,{stat['mean']}")
            rows.append(f"null_{m},{metric}_sd,{stat['sd']}")
    for k, v in res["z_vs_orthogonal_null"].items():
        rows.append(f"z,{k},{v}")
    for mt, v in res["cosine_by_modtype"].items():
        rows.append(f"cosine_by_modtype,{mt},{v}")
    for L, v in res["cosine_by_layer"].items():
        rows.append(f"cosine_by_layer,layer{L},{v}")
    e = res["shared_subspace_energy"]
    rows.append(f"shared_subspace,mean_energy_retained_r{e['rank']},{e['mean_energy_retained']}")
    rows.append(f"shared_subspace,avg_rank_ratio,{e['avg_rank_ratio']}")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapters", nargs="+", required=True, help="adapter dirs (>=2)")
    ap.add_argument("--rank", type=int, default=16, help="rank for the shared-subspace-energy basis")
    ap.add_argument("--n_null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    slots, ids, ref_cfg = build_collection_slots(args.adapters, device="cpu")
    res = analyze(slots, ids, args.rank, args.seed, args.n_null)
    res["ref_config"] = {k: ref_cfg.get(k) for k in ("r", "lora_alpha", "use_rslora", "target_modules")}
    res["adapter_dirs"] = [os.path.abspath(d) for d in args.adapters]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    if args.csv:
        _write_csv(res, args.csv)

    r, z = res["real"], res["z_vs_orthogonal_null"]
    print(f"[subspace_overlap] n={res['n_adapters']} rank={args.rank}")
    print(f"  cosine off-diag mean={r['cosine_offdiag_mean']:.4f}  "
          f"null_orth={res['nulls']['orthogonal']['cosine']['mean']:.4f}"
          f"+/-{res['nulls']['orthogonal']['cosine']['sd']:.4f}  z={z['cosine_offdiag_mean']:.1f}")
    print(f"  princ-angle cosB mean={r['angle_cos_B_offdiag_mean']:.4f}  "
          f"cosA mean={r['angle_cos_A_offdiag_mean']:.4f}")
    print(f"  shared-subspace energy@r{args.rank}={res['shared_subspace_energy']['mean_energy_retained']:.4f}  "
          f"(chance~{res['shared_subspace_energy']['avg_rank_ratio']:.4f})")
    print(f"  wrote {args.out}" + (f" , {args.csv}" if args.csv else ""))


if __name__ == "__main__":
    main()
