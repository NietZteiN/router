"""CPU unit tests for TF-IDF scoring + greedy disjoint assignment."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assign_entries import compute_tfidf, coverage_stats, greedy_assign


def test_tfidf_known_values():
    # 2 sources, 4 entries. Entry 0 is shared boilerplate (df=2 -> idf=0);
    # entries 1 and 2 are exclusive.
    counts = torch.tensor([
        [10, 10, 0, 0],
        [10, 0, 30, 0],
    ])
    scores = compute_tfidf(counts)
    import math

    assert scores[0, 0] == 0.0 and scores[1, 0] == 0.0  # idf = log(2/2) = 0
    assert abs(scores[0, 1] - 0.5 * math.log(2)) < 1e-9
    assert abs(scores[1, 2] - 0.75 * math.log(2)) < 1e-9
    assert scores[0, 2] == 0.0 and scores[0, 3] == 0.0


def test_greedy_assign_prefers_high_tfidf_and_is_disjoint():
    # Entry 2 is shared (idf=0, score 0 for both sources). Greedy order:
    # (0,e1) .396 > (0,e0) .099 > (1,e3) .077 > zero-score pairs.
    # Source 0 exhausts its quota on e1,e0, so shared entry 2 falls to
    # source 1 via the zero-score tail. Everything disjoint, quotas met.
    counts = torch.tensor([
        [5, 20, 10, 0, 0, 0],
        [0, 0, 40, 5, 0, 0],
    ])
    assigned_idx, owner, fills = greedy_assign(counts, entries_per_source=2)
    assert fills == {}
    assignment = {int(e): int(o) for e, o in zip(assigned_idx, owner)}
    assert assignment == {0: 0, 1: 0, 2: 1, 3: 1}


def test_greedy_assign_deterministic():
    g = torch.Generator().manual_seed(11)
    counts = (torch.rand(8, 100, generator=g) * 5).int()
    a1 = greedy_assign(counts, entries_per_source=6)
    a2 = greedy_assign(counts, entries_per_source=6)
    assert torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1])


def test_greedy_assign_fallback_fill_warns():
    # Source 1 accesses only 1 distinct entry but needs 3 -> fallback fill from
    # never-accessed entries, with a warning and reported count.
    import warnings

    counts = torch.zeros(2, 20, dtype=torch.int32)
    counts[0, :5] = 10
    counts[1, 5] = 10
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assigned_idx, owner, fills = greedy_assign(counts, entries_per_source=3)
    assert fills == {1: 2}
    assert any("never-accessed" in str(w.message) for w in caught)
    assert torch.bincount(owner, minlength=2).eq(3).all()
    assert assigned_idx.unique().numel() == 6
    # fallback entries must come from truly never-accessed columns
    fallback_entries = set(assigned_idx[owner == 1].tolist()) - {5}
    assert all(counts[:, e].sum() == 0 for e in fallback_entries)


def test_coverage_stats():
    counts = torch.tensor([
        [10, 0, 10, 0],
        [0, 20, 0, 20],
    ])
    assigned_idx = torch.tensor([0, 1])
    owner = torch.tensor([0, 1])
    stats = coverage_stats(counts, assigned_idx, owner)
    assert abs(stats["own_coverage_mean"] - 0.5) < 1e-9   # both sources: 0.5
    assert stats["cross_source_exposure_mean"] == 0.0      # no cross reads
    assert stats["total_accesses"] == 60
