"""S³T slice-sequence selection (ICLR 2025, arXiv 2406.16257, §3.3 + Algs 1-2).

A "sequence" is a permutation of slice indices (0..L-1); ordering[t] = the natural
slice trained at stage t (top-down). The paper trains B diverse sequences per shard
so that, after deletion requests hit some slices, at least one sequence still has a
long un-affected prefix (= a high-performing surviving model).

  - iterative_cyclic_rotation(L, B): uniform deletion prior (Alg 1).
  - bms(L, B, probs):                non-uniform prior (Alg 2, Eq 24 scoring).

Both return B permutations (tuples) that are "diverse": for B <= L no slice repeats
a position across the returned set (the paper's diversity criterion).
"""
import itertools

import numpy as np


def cyclic_permutation(seq):
    """All L right-rotations of seq (Alg 1 CYCLICPERMUTATION).

    (s0,s1,s2) -> [(s0,s1,s2), (s2,s0,s1), (s1,s2,s0)].
    """
    seq = tuple(seq)
    L = len(seq)
    return [tuple(seq[-i:] + seq[:-i]) for i in range(L)]


def iterative_cyclic_rotation(L, B):
    """Alg 1: B diverse permutations of range(L) under a uniform deletion prior.

    Start from the L cyclic rotations of (0..L-1); if B>L, expand by cyclically
    rotating progressively shorter suffixes (shared prefix) until >=B exist.
    """
    if B < 1:
        raise ValueError("B must be >= 1")
    out = list(dict.fromkeys(cyclic_permutation(range(L))))  # L perms, dedup, ordered
    n_iter = 0
    while len(out) < B:
        n_iter += 1
        if n_iter >= L:
            # Suffix length <= 1: cyclic rotation can produce nothing new. Fall back
            # to deterministic lexicographic perms to honor the budget (corner case
            # the paper handles with "certain heuristics", Appendix B.1).
            for p in itertools.permutations(range(L)):
                if p not in out:
                    out.append(p)
                if len(out) >= B:
                    break
            break
        expanded = []
        for o in out:
            prefix, suffix = o[:n_iter], o[n_iter:]
            for p in cyclic_permutation(suffix):
                cand = prefix + p
                if cand not in out and cand not in expanded:
                    expanded.append(cand)
        out.extend(expanded)
    return out[:B]


def score(seq, probs, t):
    """Eq 24: expected number of surviving (un-deleted) leading slices after t
    deletions, for a sequence with per-slice deletion probabilities `probs`.

    score = sum_{i=1..L} i * (1 - sum_{j<=i} p_{seq[j-1]})^t
    (i counts leading slices; seq is top-first, so position i covers seq[:i]).
    """
    probs = np.asarray(probs, dtype=float)
    total = 0.0
    cum = 0.0
    for i, s in enumerate(seq, start=1):
        cum += probs[s]
        total += i * (max(0.0, 1.0 - cum) ** t)
    return float(total)


def bms(L, B, probs, t=1):
    """Alg 2: up to L diverse permutations maximizing Eq-24 score under a known
    deletion prior `probs` (len L). Per level, pick the next slice for every
    partial sequence via a max-weight perfect matching (Hungarian).

    For B>L (not the paper's main regime) we top up with conditional sampling
    that biases low-deletion-prob slices toward the top (Appendix B.2).
    """
    from scipy.optimize import linear_sum_assignment

    probs = np.asarray(probs, dtype=float)
    if B < 1:
        raise ValueError("B must be >= 1")
    seqs = [[l] for l in range(L)]                 # Alg 2 line 3-4: L start elements
    for _ in range(2, L + 1):                      # choose 2nd..L-th element
        rows = list(range(len(seqs)))
        cost = np.full((len(seqs), L), 1e9)        # minimize -score == maximize score
        for r, o in enumerate(seqs):
            for v in range(L):
                if v not in o:
                    cost[r, v] = -score(tuple(o) + (v,), probs, t)
        r_idx, c_idx = linear_sum_assignment(cost)
        assign = {r: c for r, c in zip(r_idx, c_idx)}
        seqs = [o + [assign[r]] for r, o in enumerate(seqs)]
    perms = [tuple(s) for s in seqs]
    perms.sort(key=lambda s: score(s, probs, t), reverse=True)
    if B <= L:
        return perms[:B]
    # B > L: conditional sampling top-up (lower deletion prob -> nearer the top).
    rng = np.random.default_rng(0)
    out = list(perms)
    seen = set(out)
    w = np.maximum(1e-9, 1.0 - probs)
    while len(out) < B:
        order, pool = [], list(range(L))
        while pool:
            pw = w[pool] / w[pool].sum()
            pick = pool[rng.choice(len(pool), p=pw)]
            order.append(pick)
            pool.remove(pick)
        cand = tuple(order)
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out[:B]


def select_sequences(L, B, prior=None, t=1):
    """Dispatch: cyclic rotation (uniform/no prior) or BMS (non-uniform prior)."""
    if prior is None:
        return iterative_cyclic_rotation(L, B)
    return bms(L, B, prior, t=t)
