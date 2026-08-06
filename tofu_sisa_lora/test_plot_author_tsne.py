"""CPU gate for plot_author_tsne.py — run with BASE anaconda python (matplotlib env):
    ${TOFU_PLOT_PYTHON:-python3} test_plot_author_tsne.py

Covers: author-id / forget-class derivation, distance-matrix construction invariants,
t-SNE seed determinism, silhouette sanity on a planted-cluster cosine matrix (positive
for true labels, ~<=0 for shuffled), k-means size-ordered relabeling determinism, and
the perplexity clamp. No figure rendering (rendering is exercised by the smoke run).
"""
from __future__ import annotations

import numpy as np

from plot_author_tsne import (author_id, cosine_to_distance, forget_class,
                              kmeans_labels, run_tsne, silhouette_or_nan,
                              usable_perplexities)


def _planted_cosine(n_per: int = 12, k: int = 3, dim: int = 40, seed: int = 0):
    """Cosine matrix of unit vectors drawn around k well-separated centers."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, dim) * 4.0
    X = np.vstack([centers[c] + rng.randn(n_per, dim) for c in range(k)])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = np.repeat(np.arange(k), n_per)
    return X @ X.T, labels


def test_author_id_and_forget_class():
    assert author_id("shard_0") == 0 and author_id("shard_199") == 199
    for a, want in [(0, 0), (179, 0), (180, 1), (189, 1), (190, 2), (197, 2), (198, 3), (199, 3)]:
        assert forget_class(a) == want, (a, forget_class(a), want)


def test_distance_construction():
    C, _ = _planted_cosine()
    C[0, 1] += 1e-9  # tiny asymmetry, as fp accumulation would leave
    D = cosine_to_distance(C)
    assert np.allclose(D, D.T)
    assert np.all(np.diag(D) == 0.0)
    assert D.min() >= 0.0
    # distances honor similarity ordering: within-cluster < cross-cluster on average
    labels = np.repeat(np.arange(3), 12)
    same = D[labels[:, None] == labels[None, :]]
    diff = D[labels[:, None] != labels[None, :]]
    assert same.mean() < diff.mean()


def test_tsne_determinism():
    C, _ = _planted_cosine()
    D = cosine_to_distance(C)
    a = run_tsne(D, perplexity=5, seed=42)
    b = run_tsne(D, perplexity=5, seed=42)
    assert a.shape == (36, 2)
    assert np.array_equal(a, b), "same seed must reproduce identical coords"


def test_silhouette_planted_vs_shuffled():
    C, labels = _planted_cosine()
    D = cosine_to_distance(C)
    s_true = silhouette_or_nan(D, labels)
    rng = np.random.RandomState(1)
    s_shuf = silhouette_or_nan(D, rng.permutation(labels))
    assert s_true > 0.2, s_true
    assert s_shuf < 0.05, s_shuf
    # single populated class -> NaN, not an exception
    assert np.isnan(silhouette_or_nan(D, np.zeros(len(labels))))


def test_kmeans_relabel_deterministic_and_size_ordered():
    rng = np.random.RandomState(0)
    emb = np.vstack([rng.randn(20, 8) + 6, rng.randn(12, 8) - 6, rng.randn(4, 8) * 0.1])
    a = kmeans_labels(emb, 3, seed=42)
    b = kmeans_labels(emb, 3, seed=42)
    assert np.array_equal(a, b)
    counts = [(a == c).sum() for c in range(3)]
    assert counts == sorted(counts, reverse=True), counts  # slot 0 = biggest cluster


def test_perplexity_clamp():
    assert usable_perplexities([5, 15, 30, 50], 200) == [5, 15, 30, 50]
    assert usable_perplexities([5, 15, 30, 50], 8) == [5]
    assert usable_perplexities([30, 50], 8) == [2]  # fallback, never empty


if __name__ == "__main__":
    for fn in [test_author_id_and_forget_class, test_distance_construction,
               test_tsne_determinism, test_silhouette_planted_vs_shuffled,
               test_kmeans_relabel_deterministic_and_size_ordered, test_perplexity_clamp]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")
