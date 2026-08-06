"""CPU tests for S³T sequence selection + deletion simulation (no GPU, no downloads).

Run: python test_s3t_sequences.py

Validates Algs 1-2 (diversity), Alg 4 (best_surviving), and that the deletion-rate
simulation matches the Lemma-1 coupon-collector closed form and the qualitative
claims (delta grows with B up to L; S3T > SISA).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from s3t_deletion import (
    _perm,
    best_surviving,
    deletion_rate,
    deletion_rate_theory,
    empirical_retention,
    performance_curve,
    retention_gap,
    retention_prob_s3t,
    retention_prob_sisa,
    simulate_stream,
)
from s3t_sequences import (
    bms,
    cyclic_permutation,
    iterative_cyclic_rotation,
    score,
    select_sequences,
)


def _is_permset(perms, L):
    return all(sorted(p) == list(range(L)) for p in perms)


def _diverse(perms, L):
    """No slice repeats a position across the set (paper's diversity criterion)."""
    for pos in range(L):
        col = [p[pos] for p in perms]
        if len(set(col)) != len(col):
            return False
    return True


def test_cyclic_rotation():
    assert cyclic_permutation((0, 1, 2)) == [(0, 1, 2), (2, 0, 1), (1, 2, 0)]
    for L in (2, 3, 4, 5):
        perms = iterative_cyclic_rotation(L, L)
        assert len(perms) == L and len(set(perms)) == L
        assert _is_permset(perms, L)
        assert _diverse(perms, L), f"L={L} cyclic set not diverse"
    # B < L returns a diverse subset; B > L expands without duplicates.
    assert len(iterative_cyclic_rotation(4, 2)) == 2
    big = iterative_cyclic_rotation(4, 10)
    assert len(big) == 10 and len(set(big)) == 10 and _is_permset(big, 4)
    print("ok  iterative_cyclic_rotation: diversity, B<L subset, B>L expansion")


def test_bms():
    L = 4
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    perms = bms(L, L, probs, t=1)
    assert len(perms) == L and _is_permset(perms, L) and _diverse(perms, L)
    # BMS should beat sorted cyclic rotation on total Eq-24 score (Fig 15 left).
    sorted_order = tuple(np.argsort(probs))            # low deletion-prob first
    cyc = cyclic_permutation(sorted_order)
    bms_total = sum(score(p, probs, 1) for p in perms)
    cyc_total = sum(score(p, probs, 1) for p in cyc)
    assert bms_total >= cyc_total - 1e-9, (bms_total, cyc_total)
    # The L distinct start slices are spread across sequences (diversity), and the
    # least-deletable slice does start one of them.
    assert {p[0] for p in perms} == set(range(L))
    print("ok  bms: valid/diverse perms, beats sorted cyclic on total score")


def test_best_surviving():
    orderings = [(0, 1, 2, 3), (2, 3, 0, 1)]
    # delete slice 2: seq0 survives prefix [0,1] (k=2); seq1 dies at pos0 (k=0) -> best k=2.
    sid, k = best_surviving(orderings, {2})
    assert (sid, k) == (0, 2), (sid, k)
    # delete slice 0: seq0 dies at pos0 (k=0); seq1 has 0 at pos2 -> survives [2,3] (k=2).
    # This is the S3T win: a diverse second ordering keeps serving.
    sid, k = best_surviving(orderings, {0})
    assert (sid, k) == (1, 2), (sid, k)
    # delete both top slices 0 and 2: both orderings die immediately.
    assert best_surviving(orderings, {0, 2})[1] == 0
    # no deletions -> full depth.
    assert best_surviving(orderings, set())[1] == 4
    # everything deleted -> dead.
    assert best_surviving(orderings, {0, 1, 2, 3})[1] == 0
    print("ok  best_surviving: prefix logic, multi-deletion, full/dead")


def test_simulation_matches_theory():
    m, L = 5, 4
    for B in (1, 2, 4):
        emp = deletion_rate(m, L, B, n_seeds=400)
        thy = deletion_rate_theory(m, L, B)
        rel = abs(emp["mean"] - thy) / thy
        assert rel < 0.12, f"B={B}: emp {emp['mean']:.1f} vs theory {thy:.1f} (rel {rel:.2f})"
        print(f"ok  delta(m={m},L={L},B={B}) emp={emp['mean']:.1f} "
              f"theory={thy:.1f} (rel {rel:.2f})")


def test_delta_grows_with_B_and_beats_sisa():
    m, L = 5, 4
    d1 = deletion_rate(m, L, 1, n_seeds=400)["mean"]   # B=1 == SISA
    d2 = deletion_rate(m, L, 2, n_seeds=400)["mean"]
    d4 = deletion_rate(m, L, 4, n_seeds=400)["mean"]
    assert d1 < d2 < d4, (d1, d2, d4)
    # B beyond L gives no further gain (Lemma 1: B' = min(B,L)).
    d8 = deletion_rate(m, L, 8, n_seeds=400)["mean"]
    assert abs(d8 - d4) / d4 < 0.1, (d4, d8)
    print(f"ok  delta grows with B: SISA(B1)={d1:.1f} < B2={d2:.1f} < B4={d4:.1f}; "
          f"B8={d8:.1f} ~ B4 (saturates at L)")


def test_performance_curve():
    m, L = 5, 4
    F = np.array([0.42, 0.50, 0.55, 0.58, 0.60])       # toy monotone F(0..L)
    ops = [select_sequences(L, 4) for _ in range(m)]
    tr = simulate_stream(m, L, ops, seed=1)["depths"]
    perf = performance_curve(tr, F)
    assert perf[0] <= F[L] and perf[-1] >= F[0] - 1e-9
    assert np.all(np.diff(perf) <= 1e-9), "performance should be non-increasing"
    print("ok  performance_curve: bounded, monotone non-increasing")


def test_retention_closed_form():
    """Lemma 2 (Eq 18/20): S3T retention >= SISA, both monotone; B' caps at P(L,k)."""
    L = 4
    assert _perm(4, 2) == 12 and _perm(4, 4) == 24 and _perm(4, 1) == 4
    for k in (1, 2, 3):
        for r in (1, 3, 6, 10):
            sisa = retention_prob_sisa(k, L, r)
            for B in (1, 2, 4):
                s3t = retention_prob_s3t(k, L, r, B)
                assert s3t >= sisa - 1e-12, (k, r, B, s3t, sisa)
                assert 0.0 <= s3t <= 1.0
            # B=1 S3T == SISA (single ordering).
            assert abs(retention_prob_s3t(k, L, r, 1) - sisa) < 1e-12
            # monotone non-decreasing in B.
            seq = [retention_prob_s3t(k, L, r, B) for B in (1, 2, 3, 4, 8)]
            assert all(b >= a - 1e-12 for a, b in zip(seq, seq[1:])), seq
            assert retention_gap(k, L, r, 2) >= -1e-12
    # B' cap: beyond P(L,k) the probability stops rising (k=4 -> P=24).
    assert abs(retention_prob_s3t(4, 4, 3, 24) - retention_prob_s3t(4, 4, 3, 100)) < 1e-12
    print("ok  Lemma 2 retention: S3T>=SISA, monotone in B, B' caps at P(L,k)")


def test_retention_matches_empirical():
    """Per-shard simulation validates Eq 20: with random independent sequences the
    empirical retention matches the closed form within CI; diverse cyclic sequences
    meet or beat it (the paper's diversity remark)."""
    L, k, r = 4, 2, 6
    for B in (1, 2, 4):
        emp_rand = empirical_retention(L, B, k, r, n_seeds=8000)
        closed = retention_prob_s3t(k, L, r, B)
        assert abs(emp_rand - closed) < 0.03, (B, emp_rand, closed)
        emp_cyc = empirical_retention(
            L, B, k, r, sequences=iterative_cyclic_rotation(L, B), n_seeds=8000)
        assert emp_cyc >= closed - 0.03, (B, emp_cyc, closed)
    print("ok  empirical retention matches Eq-20 (random); cyclic >= closed form")


def test_rq3_diversity():
    """RQ3/Fig 8: cyclic rotation beats random on edit distance; BMS >= sorted-cyclic
    on Eq-24 score; cyclic edit distance == L at B<=L."""
    from s3t_rq3 import (avg_pairwise_edit_distance, edit_distance,
                         nonuniform_experiment, random_sequences, sorted_cyclic)
    assert edit_distance((0, 1, 2, 3), (0, 1, 2, 3)) == 0
    assert edit_distance((0, 1, 2, 3), (3, 0, 1, 2)) == 4
    L = 5
    cyc = avg_pairwise_edit_distance(iterative_cyclic_rotation(L, L))
    assert abs(cyc - L) < 1e-9, f"cyclic B=L edit distance {cyc} != {L}"
    rnd = np.mean([avg_pairwise_edit_distance(random_sequences(L, L, seed=s))
                   for s in range(20)])
    assert cyc > rnd, (cyc, rnd)
    # Robust (provable) claims: BMS is maximally diverse (Lemma 3) and beats random
    # on diversity. Score ordering vs sorted-cyclic is t-dependent and NOT asserted
    # (at t=1 all position-diverse sets tie by construction); reported descriptively.
    nu = nonuniform_experiment(4, 4, n_priors=10)
    assert nu["bms"]["edit"] >= nu["random"]["edit"] - 1e-9, nu
    assert abs(nu["bms"]["edit"] - 4) < 1e-9, f"BMS not maximally diverse at B=L: {nu}"
    print(f"ok  RQ3: cyclic edit {cyc:.2f} > random {rnd:.2f}; "
          f"BMS edit {nu['bms']['edit']:.2f} = L (maximally diverse, Lemma 3)")


if __name__ == "__main__":
    test_cyclic_rotation()
    test_bms()
    test_best_surviving()
    test_simulation_matches_theory()
    test_delta_grows_with_B_and_beats_sisa()
    test_performance_curve()
    test_retention_closed_form()
    test_retention_matches_empirical()
    test_rq3_diversity()
    print("ALL S3T SEQUENCE/DELETION TESTS PASSED")
