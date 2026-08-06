"""S³T deletion procedure (Alg 4) + deletion-rate / performance simulation.

A deletion request affects one slice (shard m', slice index s). For a sequence
(ordering) trained top-down on cumulative slices, the surviving model after some
slices are deleted is the longest prefix containing no deleted slice (Alg 4:
deactivate layers {l',..,L} where l' = position of the affected slice). Across the
B sequences of a shard, the "best available model" is the one with the longest
surviving prefix. A shard fails when every sequence has a deleted slice at its top
position (prefix length 0); the system needs a from-scratch retrain when every
shard has failed (Lemma 1 coupon-collector framing).

Everything here is pure CPU bookkeeping over slice orderings — no models, no GPU.
Performance after r deletions is reconstructed by composing the per-shard surviving
depths with a measured F(k) (s3t_measure_F.py).
"""
import numpy as np

from s3t_sequences import select_sequences


def best_surviving(orderings, deleted_slices):
    """Per shard: (seq_id, k) with the longest deletion-free prefix (Alg 4).

    k = number of leading slices retained by the best sequence (0 = shard dead).
    """
    deleted = set(deleted_slices)
    best_k, best_id = -1, None
    for sid, o in enumerate(orderings):
        k = len(o)
        for pos, s in enumerate(o):
            if s in deleted:
                k = pos          # slices o[:pos] survive
                break
        if k > best_k:
            best_k, best_id = k, sid
    return best_id, best_k


def simulate_stream(m, L, orderings_per_shard, prior=None, seed=0, max_requests=None):
    """One deletion stream until system failure (all shards dead).

    Returns dict with per-request `depths` (m-vector of best surviving k after each
    request) and `delta` (request index at which the system first fails).
    """
    rng = np.random.default_rng(seed)
    # Flat catalogue of (shard, slice) deletion targets with a prior.
    targets = [(mi, s) for mi in range(m) for s in range(L)]
    if prior is None:
        p = np.full(len(targets), 1.0 / len(targets))
    else:
        prior = np.asarray(prior, dtype=float)          # per-slice prior, shape (L,)
        p = np.array([prior[s] for _, s in targets])
        p = p / p.sum()
    cap = max_requests if max_requests is not None else 100 * m * L
    deleted = {mi: set() for mi in range(m)}
    depths_trace, delta = [], None
    for r in range(1, cap + 1):
        mi, s = targets[rng.choice(len(targets), p=p)]
        deleted[mi].add(s)
        depths = np.array([best_surviving(orderings_per_shard[mj], deleted[mj])[1]
                           for mj in range(m)])
        depths_trace.append(depths)
        if delta is None and (depths == 0).all():
            delta = r
            break
    return {"depths": np.array(depths_trace), "delta": delta}


def deletion_rate(m, L, B, prior=None, t=1, n_seeds=200):
    """Mean +/- std of delta (requests until system failure) over random streams."""
    orderings = select_sequences(L, B, prior=prior, t=t)
    ops = [orderings for _ in range(m)]               # same selection per shard
    deltas = []
    for seed in range(n_seeds):
        out = simulate_stream(m, L, ops, prior=prior, seed=seed)
        deltas.append(out["delta"])
    deltas = np.array([d for d in deltas if d is not None], dtype=float)
    return {"mean": float(deltas.mean()), "std": float(deltas.std()),
            "n": len(deltas), "B": B, "m": m, "L": L}


def deletion_rate_theory(m, L, B):
    """Closed-form delta (Lemma 1): S3T ~ mL*H_{mB'}, SISA (B=1) ~ mL*H_m,
    where B' = min(B, L) distinct top-position slices per shard (diverse seqs)."""
    Bp = min(B, L)
    n_coupons = m * Bp
    H = float(np.sum(1.0 / np.arange(1, n_coupons + 1)))
    return m * L * H


# ── Lemma 2: per-shard performance retention (Eq 18 / Eq 20) ─────────────────

def _perm(L, k):
    """P(L, k) = L! / (L-k)! — number of distinct length-k prefixes."""
    out = 1
    for i in range(k):
        out *= (L - i)
    return out


def retention_prob_sisa(k, L, r):
    """Eq 18: P[F_r(SISA) >= F(k)] = (1 - k/L)^r — probability that none of r
    uniform deletions hit the top-k slices of the single ordering (= alpha, the
    per-sequence prefix-survival prob).

    NOTE: the prose/Eq-18 in some renderings prints `1 - (k/L)^r`, but that is
    inconsistent with the Eq-21 difference derivation (which sets alpha = (1-k/L)^r
    and requires S3T(B=1) == SISA). (1-k/L)^r is the self-consistent form and the
    one we use; verified by test_retention_closed_form."""
    return (1.0 - k / L) ** r


def retention_prob_s3t(k, L, r, B):
    """Eq 20: P[F_r(S3T) >= F(k)] = 1 - (1 - (1-k/L)^r)^{B'}, B' = min(B, P(L,k))
    (at least one of B' diverse sequences keeps an intact length-k prefix)."""
    Bp = min(B, _perm(L, k))
    alpha = (1.0 - k / L) ** r
    return 1.0 - (1.0 - alpha) ** Bp


def retention_gap(k, L, r, B):
    """Eq 3: S3T - SISA retention-probability gap = zeta(1 - zeta^{B'-1})."""
    return retention_prob_s3t(k, L, r, B) - retention_prob_sisa(k, L, r)


def empirical_retention(L, B, k, r, sequences=None, n_seeds=4000, seed0=0):
    """Per-shard P[retain >= k slices after r deletions] — the event behind Eq 20.

    Each of r deletions hits one of the shard's L slices uniformly. A sequence
    retains >= k iff its top-k prefix avoids every deleted slice. Retention = at
    least one of the B sequences survives.

    sequences=None -> fresh random independent sequences per trial (matches the
    closed form's independence assumption, so empirical ~= retention_prob_s3t).
    Pass diverse cyclic sequences to see them meet/beat the random closed form."""
    rng = np.random.default_rng(seed0)
    hits = 0
    for _ in range(n_seeds):
        deleted = set(rng.integers(0, L, size=r).tolist())
        seqs = ([tuple(rng.permutation(L)) for _ in range(B)]
                if sequences is None else sequences)
        if any(all(sl not in deleted for sl in seq[:k]) for seq in seqs):
            hits += 1
    return hits / n_seeds


# ── Fig 9: deletion time over a request stream ───────────────────────────────

def expected_retrains(R, delta):
    """Expected #from-scratch retrains over R requests for a system with deletion
    rate delta (one retrain per delta requests, on average)."""
    return R / float(delta)


def performance_curve(depths_trace, F):
    """System utility after each request = mean over shards of F[best_k].

    F: array of length L+1, F[k] = ensemble/shard performance when trained on k
    slices (F[0] = base model). The mean-over-shards composition is the uniform
    per-shard approximation; mixed-depth states are spot-checked on GPU separately.
    """
    F = np.asarray(F, dtype=float)
    return np.array([F[depths].mean() for depths in depths_trace])


def average_performance_curve(m, L, B, F, prior=None, t=1, n_seeds=200, n_points=None):
    """Mean performance-vs-#deletions over streams, padded to a common length
    (failed streams hold at base performance F[0] afterwards)."""
    orderings = select_sequences(L, B, prior=prior, t=t)
    ops = [orderings for _ in range(m)]
    curves = []
    for seed in range(n_seeds):
        tr = simulate_stream(m, L, ops, prior=prior, seed=seed)["depths"]
        curves.append(performance_curve(tr, F))
    maxlen = max(len(c) for c in curves) if n_points is None else n_points
    padded = np.full((len(curves), maxlen), F[0], dtype=float)
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c[:maxlen]
    return padded.mean(axis=0), padded.std(axis=0)
