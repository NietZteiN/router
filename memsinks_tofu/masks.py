"""Sink-mask construction for MemSinks/SeqTD-on-TOFU.

Two mask schemes over the MLP intermediate dimension (per layer, layer-shared —
the reference hash has no layer index):

  - "hash":     the paper's sequence-tied hash, ported VERBATIM from
                <repo>/MemSinks/src/src/SeqTDModel.py:16-25 (see
                batch_seqtied_mask_mult below). Author a -> pseudo-random
                p_mem-fraction subset of the sink pool.
  - "disjoint": author a owns the contiguous slice [a*s, (a+1)*s) of the sink
                pool (s = num_mem // num_authors). Zero deletion collateral by
                construction; the C2 (selective per-author deletion) regime.

Neuron layout everywhere in this project: the FIRST num_gen intermediate
neurons are "general" (delta always active), the LAST num_mem are the sink
pool. A full per-layer mask/scale vector has length num_gen + num_mem =
intermediate_size.

Author indexing: mask_table row a corresponds to TOFU author index a (0-199,
= dataset row // 20). The hash scheme feeds seq_id = a + 1 into the reference
hash because seq_id 0 is DEGENERATE (0 * anything = 0, 0/M < p -> all-ones
mask over the whole sink pool). Verified numerically 2026-07-14.
"""
import hashlib

import torch


def batch_seqtied_mask_mult(seq_ids, neuron_dim, p_active, a=1588635695, M=4294967291, base=pow(2, 16)):
    """VERBATIM port of MemSinks/src/src/SeqTDModel.py:16-25 (commit a005119).

    ⚠ Do NOT "fix" the math: torch.pow(a, arange(n)) overflows int64 from
    exponent 3 onward, so powers_of_a is NOT a^j mod M — it is a deterministic
    overflow artifact. The published masks are defined by this exact behavior;
    a clean modular exponentiation produces DIFFERENT masks. seq_ids must be
    int64 (torch.long) or the overflow pattern (and hence the masks) changes.
    """
    seq_ids = torch.mul(seq_ids, base)
    powers_of_a = torch.remainder(torch.pow(a, torch.arange(neuron_dim)), M).to(seq_ids.device)
    multiplied = torch.remainder(torch.mul(seq_ids.unsqueeze(-1), powers_of_a.unsqueeze(0).unsqueeze(0)), M)
    assert multiplied.shape == torch.Size([seq_ids.shape[0], seq_ids.shape[1], neuron_dim])
    added = torch.remainder(multiplied, M)
    mask = ((added / M) < p_active)
    mask.requires_grad_(False)
    return mask


def hash_partition(intermediate_size, p_gen):
    """Paper formula: num_gen = int(I*p_gen); sink pool = remainder."""
    num_gen = int(intermediate_size * p_gen)
    return num_gen, intermediate_size - num_gen


def disjoint_partition(intermediate_size, p_gen, num_authors):
    """Disjoint scheme: shrink the sink pool to an exact multiple of
    num_authors; the s-remainder neurons become general (never dead)."""
    raw_mem = intermediate_size - int(intermediate_size * p_gen)
    s = raw_mem // num_authors
    if s < 1:
        raise ValueError(f"sink pool {raw_mem} < num_authors {num_authors}")
    num_mem = s * num_authors
    return intermediate_size - num_mem, num_mem, s


def disjoint_dead_partition(intermediate_size, num_authors):
    """E3 strict-isolation partition: num_gen = 0, EVERY neuron is either an
    author slice or DEAD (owned by nobody, permanently masked).

    ⚠ This is not disjoint_partition(I, 0.0, A): that would fold the
    I % A remainder (192 neurons at I=8192, A=200) into ALWAYS-ON general
    rows trained on every author — silently breaking the strict arm's
    row-provenance exactness. Here the remainder rows are all-False in every
    author's mask, so they receive no gradient and stay at zero init.
    """
    s = intermediate_size // num_authors
    if s < 1:
        raise ValueError(f"intermediate {intermediate_size} < num_authors {num_authors}")
    return 0, intermediate_size, s          # num_gen, num_mem(=full width), slice


def disjoint_dead_table(num_authors, intermediate_size):
    """(num_authors, intermediate_size) bool; author a owns [a*s,(a+1)*s);
    trailing remainder columns are owned by NOBODY (dead)."""
    _, _, s = disjoint_dead_partition(intermediate_size, num_authors)
    table = torch.zeros(num_authors, intermediate_size, dtype=torch.bool)
    for a in range(num_authors):
        table[a, a * s:(a + 1) * s] = True
    return table


def hash_mask_table(num_authors, num_mem, p_mem):
    """(num_authors, num_mem) bool table; row a = author a's hashed sink mask.

    seq_id = author_index + 1 — NEVER 0 (see module docstring). Built on CPU
    int64 once; callers must treat the table as frozen (sha256 it into
    provenance) rather than recomputing on device.
    """
    seq_ids = torch.arange(1, num_authors + 1, dtype=torch.long).unsqueeze(1)  # (A, 1)
    table = batch_seqtied_mask_mult(seq_ids, num_mem, p_mem).squeeze(1)
    assert table.shape == (num_authors, num_mem)
    assert not bool(table.all(dim=1).any()), "an author's hash mask is all-ones (ID-0-style degeneracy)"
    return table


def disjoint_mask_table(num_authors, num_mem):
    """(num_authors, num_mem) bool table; author a owns [a*s, (a+1)*s)."""
    if num_mem % num_authors != 0:
        raise ValueError(f"num_mem {num_mem} not divisible by num_authors {num_authors}")
    s = num_mem // num_authors
    table = torch.zeros(num_authors, num_mem, dtype=torch.bool)
    for a in range(num_authors):
        table[a, a * s:(a + 1) * s] = True
    return table


def build_partition_and_table(cfg):
    """From a config dict -> (num_gen, num_mem, mask_table).

    cfg keys used: id_scheme, p_gen, p_mem (hash only), num_authors,
    intermediate_size.
    """
    I = cfg["intermediate_size"]
    A = cfg["num_authors"]
    if cfg["id_scheme"] == "disjoint":
        num_gen, num_mem, _ = disjoint_partition(I, cfg["p_gen"], A)
        table = disjoint_mask_table(A, num_mem)
    elif cfg["id_scheme"] == "disjoint_dead":
        num_gen, num_mem, _ = disjoint_dead_partition(I, A)
        table = disjoint_dead_table(A, I)
    elif cfg["id_scheme"] == "hash":
        num_gen, num_mem = hash_partition(I, cfg["p_gen"])
        table = hash_mask_table(A, num_mem, cfg["p_mem"])
    else:
        raise ValueError(f"unknown id_scheme {cfg['id_scheme']!r}")
    return num_gen, num_mem, table


def table_sha256(mask_table):
    """Provenance hash of the frozen mask table (uint8 bytes, row-major)."""
    return hashlib.sha256(mask_table.to(torch.uint8).contiguous().numpy().tobytes()).hexdigest()


def collateral_stats(mask_table, forget_authors):
    """Union fraction + per-retained-author overlap of a deletion.

    Returns dict with:
      union_fraction: fraction of the sink pool zeroed by deleting forget_authors
      per_retained_overlap: {author: fraction of ITS active sinks inside the union}
      mean/max retained overlap.
    All math in float32 before any .item()/numpy (bf16->numpy trap).
    """
    union = mask_table[forget_authors].any(dim=0)
    union_fraction = int(union.sum().item()) / union.numel()
    retained = [a for a in range(mask_table.shape[0]) if a not in set(forget_authors)]
    per = {}
    for a in retained:
        own = mask_table[a]
        n_own = int(own.sum().item())
        per[a] = float((own & union).sum().item() / n_own) if n_own else 0.0
    vals = list(per.values())
    return {
        "union_fraction": union_fraction,
        "retained_overlap_mean": float(sum(vals) / len(vals)) if vals else 0.0,
        "retained_overlap_max": max(vals) if vals else 0.0,
        "per_retained_overlap": per,
    }
