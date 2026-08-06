"""Write-side disjoint LoRA subspaces — composable_tv [wd] track core math (CPU-safe).

Constrain each author-expert's lora_B to an author-owned subspace of every target module's
OUTPUT space, so an N-author task-vector SUM is write-side interference-free by construction
and deletion is a clean, deterministic subspace operation:

  orthblock  ONE author-free Gaussian matrix per (struct_seed, module) is reduced-QR
             orthonormalized into Q with pool_size*r_prime columns; the author at pool
             index i owns the column block Q[:, i*r_prime:(i+1)*r_prime). Blocks of a
             single orthonormal Q are EXACTLY mutually orthogonal — that, not per-author
             seeding, is why the seed string deliberately excludes the author.
             (Rejected design: independent per-author seeded bases — those are only
             ~1/sqrt(d_out)-incoherent, which breaks the exact merge-drop identity
             verify_struct.py certifies.)
  rowslice   the same partition with coordinate rows (Q = leading identity columns):
             author i owns output coordinates [i*r_prime, (i+1)*r_prime). The cheapest
             variant — a hard contiguous slice of the output features.

Deletion contract: with B_a in col(Q_a) for every author a, the sum merge
M = sum_a s_a B_a A_a satisfies  M - D_a == (I - Q_a Q_a^T) M  exactly — dropping an
author's factors == zeroing its block of the merged delta (store-and-subtract, never a
re-derivation). verify_struct.py asserts this; test_struct_tv.py is the CPU gate.

Capacity: pool_size * r_prime <= d_out per module. On Llama-3.2-1B (GQA) this binds at
k/v d_out = 512: pool 16 x r_prime 32 = 512 exactly (why ctv_1b_wd.json uses pool_size 16,
not the ctrl arm's 20).
"""
from __future__ import annotations

import hashlib

import torch

from jd_collection import _PREFIX


# ---------------------------------------------------------------------------
# Naming + seeding (train_lora_shard.apply_irp_projections idiom)
# ---------------------------------------------------------------------------

def canonical_slot(module_name):
    """Live-PeftModel module name -> on-disk slot name (jd_collection._read_adapter
    convention). Basis seeds derive from the CANONICAL name so a basis rebuilt from
    safetensors (verify_struct.py) is byte-identical to the one used on the live model
    (the training callback)."""
    return module_name[len(_PREFIX):] if module_name.startswith(_PREFIX) else module_name


def _seed_from_string(s):
    """SHA-256 -> first 4 little-endian bytes (the apply_irp_projections idiom), so seeds
    are independent per string and never collide by arithmetic accident."""
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


def seeded_subspace(seed_string, d_out, dim):
    """Deterministic orthonormal basis (d_out x dim), fp32, CPU generator.

    Gaussian from the SHA-derived seed -> reduced QR -> column signs fixed by diag(R) > 0
    (QR of an a.s. full-rank Gaussian is then unique, so the basis does not depend on the
    LAPACK sign convention)."""
    if dim > d_out:
        raise ValueError(f"seeded_subspace: dim {dim} > d_out {d_out}")
    gen = torch.Generator().manual_seed(_seed_from_string(seed_string))
    G = torch.randn(d_out, dim, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G, mode="reduced")
    sign = torch.sign(torch.diagonal(R))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return (Q * sign).contiguous()


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

def module_basis(struct_seed, layer_name, d_out, pool_size, r_prime, mode="orthblock"):
    """Shared per-module basis Q (d_out x pool_size*r_prime) with orthonormal columns.

    Author-free by design: every author's block comes from the SAME Q (see module
    docstring). `layer_name` may be a live module name or an on-disk slot name — it is
    canonicalized before seeding."""
    total = pool_size * r_prime
    if total > d_out:
        raise ValueError(
            f"write-side capacity exceeded at {layer_name!r}: pool_size ({pool_size}) x "
            f"r_prime ({r_prime}) = {total} > d_out ({d_out}). Disjoint author blocks must "
            f"fit inside the module's output dim — on Llama-3.2-1B GQA this binds at k/v "
            f"d_out=512 (pool 16 x r' 32 = 512 exactly); shrink pool_size or r_prime.")
    slot = canonical_slot(layer_name)
    if mode == "orthblock":
        return seeded_subspace(f"{struct_seed}:{slot}", d_out, total)
    if mode == "rowslice":
        # Coordinate rows, partitioned contiguously: Q = the leading identity columns.
        return torch.eye(d_out, dtype=torch.float32)[:, :total].contiguous()
    raise ValueError(f"unknown basis mode {mode!r} (orthblock|rowslice)")


def author_basis(Q, pool_index, r_prime):
    """Author's owned column block Q[:, idx*r_prime : (idx+1)*r_prime)."""
    n_blocks = Q.shape[1] // r_prime
    if not 0 <= pool_index < n_blocks:
        raise ValueError(f"pool_index {pool_index} out of range for {n_blocks} blocks")
    return Q[:, pool_index * r_prime:(pool_index + 1) * r_prime]


def basis_sha256(Q):
    """Provenance hash of a basis block (fp32 bytes) for struct_meta.json."""
    return hashlib.sha256(
        Q.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()).hexdigest()


def build_author_basis_map(model, struct_seed, pool_index, pool_size, r_prime, mode):
    """{live module name: Q_a} over every lora_B-bearing module of a PeftModel."""
    basis_map = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_B") or len(module.lora_B) == 0:
            continue
        d_out = next(iter(module.lora_B.values())).weight.shape[0]
        Q = module_basis(struct_seed, name, d_out, pool_size, r_prime, mode)
        basis_map[name] = author_basis(Q, pool_index, r_prime).contiguous()
    if not basis_map:
        raise ValueError("build_author_basis_map: no lora_B modules found — not a LoRA PeftModel?")
    return basis_map


# ---------------------------------------------------------------------------
# Projection (project-after-step, the sift_masks.project_ convention)
# ---------------------------------------------------------------------------

def project_lora_B_(model, basis_map, adapter_name=None):
    """In-place B <- Q_a (Q_a^T B) for every lora_B module (optionally one adapter key).

    Computed in fp32 on the weight's device, cast back to the weight dtype. A module with
    lora_B but no basis raises — a silent skip would void the deletion certificate."""
    n = 0
    for name, module in model.named_modules():
        if not hasattr(module, "lora_B") or len(module.lora_B) == 0:
            continue
        if name not in basis_map:
            raise KeyError(f"module {name!r} has lora_B but no entry in basis_map — every "
                           f"targeted module must be constrained")
        Q = basis_map[name]
        for key, lin in module.lora_B.items():
            if adapter_name is not None and key != adapter_name:
                continue
            W = lin.weight
            Qd = Q.to(device=W.device, dtype=torch.float32)
            with torch.no_grad():
                W.data.copy_((Qd @ (Qd.t() @ W.data.float())).to(W.dtype))
            n += 1
    if n == 0:
        raise ValueError(f"project_lora_B_: no lora_B matched adapter_name={adapter_name!r}")
    return n


# ---------------------------------------------------------------------------
# Factored energy (never materialize the dense delta for big shapes)
# ---------------------------------------------------------------------------

def _gram_fro2(B, A):
    """||B @ A||_F^2 = trace((B^T B)(A A^T)) via the r x r Grams, float64."""
    Bd = B.detach().to("cpu", torch.float64)
    Ad = A.detach().to("cpu", torch.float64)
    # trace(MN) = sum(M * N^T); both Grams are symmetric so the transpose is free.
    return float(torch.sum((Bd.t() @ Bd) * (Ad @ Ad.t())))


def delta_fro2(A, B, scaling):
    """||scaling * B @ A||_F^2, factored."""
    return (float(scaling) ** 2) * _gram_fro2(B, A)


def delta_fro2_in(A, B, scaling, Q):
    """||Q Q^T (scaling * B @ A)||_F^2, factored: project B only — since Q has orthonormal
    columns, ||Q Q^T B A||_F == ||(Q^T B) A||_F, so the d_out-sized product never exists."""
    Bq = Q.detach().to("cpu", torch.float64).t() @ B.detach().to("cpu", torch.float64)
    return (float(scaling) ** 2) * _gram_fro2(Bq, A)


def energy_in_subspace(A, B, scaling, Q):
    """Fraction of ||scaling * B @ A||_F^2 inside col(Q), computed FACTORED.

    A zero delta returns 1.0 (the zero matrix lies in every subspace — keeps the
    constrained-arm certificate vacuously true; emptiness is caught separately by the
    empty-slice check)."""
    total = delta_fro2(A, B, scaling)
    if total <= 0.0:
        return 1.0
    frac = delta_fro2_in(A, B, scaling, Q) / total
    return min(max(frac, 0.0), 1.0)
