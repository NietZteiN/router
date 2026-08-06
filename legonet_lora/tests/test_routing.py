"""CPU micro-tests for routing: k-NN correctness, frozen-key invariance, determinism.

    python tests/test_routing.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from routing import KNNRouter  # noqa: E402


def test_knn_correctness():
    # 3 keys on a line; nearest by L2 is obvious.
    keys = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype="float32")
    r = KNNRouter(keys, k=2)
    out = r.route(np.array([[0.1, 0.0]]))  # nearest = key0 then key1
    assert out.tolist() == [[0, 1]], out
    out = r.route(np.array([[4.9, 0.0]]))  # nearest = key2 then key1
    assert out.tolist() == [[2, 1]], out
    print("  knn_correctness OK")


def test_tie_breaks_lower_index():
    keys = np.array([[0.0, 0.0], [2.0, 0.0]], dtype="float32")
    r = KNNRouter(keys, k=2)
    out = r.route(np.array([[1.0, 0.0]]))  # equidistant -> sorted by index
    assert out.tolist() == [[0, 1]], out
    print("  tie_breaks_lower_index OK")


def test_determinism():
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((16, 32)).astype("float32")
    emb = rng.standard_normal((50, 32)).astype("float32")
    r = KNNRouter(keys, k=3)
    a = r.route(emb)
    b = r.route(emb)
    assert np.array_equal(a, b)
    print("  determinism OK")


def test_frozen_key_invariance():
    """Removing a record must NOT change any other record's assignment.

    This is the cascade-free property (Condition A) that makes deletion exact:
    each record's k nearest keys depend only on that record and the frozen keys.
    """
    rng = np.random.default_rng(1)
    keys = rng.standard_normal((20, 16)).astype("float32")
    emb = rng.standard_normal((100, 16)).astype("float32")
    r = KNNRouter(keys, k=4)
    full = r.route(emb)
    # drop record 37, re-route the rest -> identical rows for the survivors
    survivors = [i for i in range(100) if i != 37]
    reduced = r.route(emb[survivors])
    for new_i, old_i in enumerate(survivors):
        assert np.array_equal(reduced[new_i], full[old_i]), (old_i,)
    print("  frozen_key_invariance OK")


def test_k_equals_n():
    keys = np.array([[0.0], [1.0], [2.0]], dtype="float32")
    r = KNNRouter(keys, k=3)
    out = r.route(np.array([[0.4]]))
    assert sorted(out[0].tolist()) == [0, 1, 2]
    assert out[0].tolist() == [0, 1, 2]  # ordered by distance
    print("  k_equals_n OK")


if __name__ == "__main__":
    test_knn_correctness()
    test_tie_breaks_lower_index()
    test_determinism()
    test_frozen_key_invariance()
    test_k_equals_n()
    print("test_routing: ALL PASS")
