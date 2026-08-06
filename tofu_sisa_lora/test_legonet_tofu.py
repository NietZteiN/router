"""CPU unit tests for the LegoNet-on-TOFU arm (run before any SLURM job).

Covers the pure-numpy core (keys/assignment determinism, top-k k-NN, the
affected-adapter union, frozen-key cascade-freedom, question->author resolution)
and the wrapper's routing + merge-cache logic with a stub PEFT model (no GPU/LLM).

    python test_legonet_tofu.py
"""
import numpy as np

import legonet_tofu as lt


def _toy_cfg(n=4, k=2, num_authors=12):
    return {
        "name": "toy", "n": n, "k": k, "num_authors": num_authors,
        "records_per_author": 2, "route_on": "answer", "kmeans_seed": 42,
        "encoder_model": "x", "forget_authors": list(range(num_authors - 2, num_authors)),
        "output_dir": "/tmp/legonet_toy", "lora": {}, "train": {},
    }


def _toy_author_emb(num_authors, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    # Three latent clusters so k-means has real structure.
    centers = rng.normal(size=(3, dim))
    emb = np.stack([centers[a % 3] + 0.05 * rng.normal(size=dim) for a in range(num_authors)])
    return emb.astype("float32")


def test_knn_router():
    keys = np.array([[0, 0], [10, 0], [0, 10]], dtype="float32")
    r = lt.KNNRouter(keys, k=2)
    out = r.route(np.array([[0.1, 0.0], [9.0, 0.5]], dtype="float32"))
    assert out.shape == (2, 2)
    assert list(out[0]) == [0, 1], out[0]          # nearest = key0, then key1
    assert list(out[1]) == [1, 0], out[1]          # nearest = key1, then key0
    # tie at equal distance -> lower index first (stable)
    tie = lt.KNNRouter(np.array([[1, 0], [-1, 0]], dtype="float32"), k=2)
    assert tie.route_one(np.array([0.0, 5.0])) == [0, 1]
    print("ok test_knn_router")


def test_assignment_determinism_and_shape():
    cfg = _toy_cfg()
    emb = _toy_author_emb(cfg["num_authors"])
    k1 = lt.build_keys(cfg, emb, cache=False)
    k2 = lt.build_keys(cfg, emb, cache=False)
    assert np.allclose(k1, k2), "k-means must be deterministic for fixed seed"
    a1 = lt.build_assignment(cfg, emb, k1, cache=False)
    a2 = lt.build_assignment(cfg, emb, k1, cache=False)
    assert a1["author_to_keys"] == a2["author_to_keys"]
    # every author has exactly k keys; members invert author_to_keys exactly
    for a in range(cfg["num_authors"]):
        assert len(a1["author_to_keys"][str(a)]) == cfg["k"]
    inv = {j: set() for j in range(cfg["n"])}
    for a in range(cfg["num_authors"]):
        for j in a1["author_to_keys"][str(a)]:
            inv[j].add(a)
    for j in range(cfg["n"]):
        assert set(a1["members"][str(j)]) == inv[j]
    print("ok test_assignment_determinism_and_shape")


def test_affected_union():
    cfg = _toy_cfg()
    emb = _toy_author_emb(cfg["num_authors"])
    keys = lt.build_keys(cfg, emb, cache=False)
    asg = lt.build_assignment(cfg, emb, keys, cache=False)
    forget = cfg["forget_authors"]
    expected = sorted({j for a in forget for j in asg["author_to_keys"][str(a)]})
    assert lt.affected_adapters(asg, forget) == expected
    # untouched adapters are exactly the complement
    untouched = [j for j in range(cfg["n"]) if j not in expected]
    assert set(untouched).isdisjoint(expected)
    print(f"ok test_affected_union (forget {len(forget)} authors -> {len(expected)}/{cfg['n']} adapters)")


def test_frozen_key_invariance():
    """With frozen keys, an author's top-k depends only on that author + keys, so
    removing forget authors never changes any retained author's assignment."""
    cfg = _toy_cfg()
    emb = _toy_author_emb(cfg["num_authors"])
    keys = lt.build_keys(cfg, emb, cache=False)         # frozen
    asg = lt.build_assignment(cfg, emb, keys, cache=False)
    knn = lt.KNNRouter(keys, cfg["k"])
    forget = set(cfg["forget_authors"])
    for a in range(cfg["num_authors"]):
        if a in forget:
            continue
        # recomputing a retained author's route against the SAME frozen keys is identical
        assert lt.author_keys(asg, a) == knn.route_one(emb[a])
    print("ok test_frozen_key_invariance")


def test_q2author_and_parse():
    data = [
        {"question": "Who is A?", "answer": "A1"},
        {"question": "Where is A?", "answer": "A2"},
        {"question": "Who is B?", "answer": "B1"},
        {"question": "Where is B?", "answer": "B2"},
    ]
    q2a = lt.build_q2author(data, num_authors=2, per_author=2)
    assert q2a["Who is A?"] == 0 and q2a["Where is B?"] == 1
    text = "Question: Who is A?\nAnswer: A1"
    assert lt.parse_question(text) == "Who is A?"
    assert q2a[lt._norm(lt.parse_question(text))] == 0
    # generate-style prompt (no answer yet)
    assert lt.parse_question("Question: Where is B?\nAnswer:") == "Where is B?"
    # OOD question is absent
    assert q2a.get("Who is Z?") is None
    print("ok test_q2author_and_parse")


# ── wrapper routing + merge cache (stub PEFT model, no GPU) ───────────────────

class _StubPeft:
    def __init__(self):
        self.added, self.set_calls, self.deleted = [], [], []

    def add_weighted_adapter(self, adapters, weights, adapter_name, combination_type):
        assert combination_type == "linear"
        assert abs(sum(weights) - 1.0) < 1e-6
        self.added.append((adapter_name, tuple(adapters), tuple(weights)))

    def set_adapter(self, name):
        self.set_calls.append(name)

    def delete_adapter(self, name):
        self.deleted.append(name)


def _make_wrapper(stub, keys, assignment, q2author, k, embed_fn, loaded, merge_cap=96):
    from legonet_model import LegoNetRoutedModel
    return LegoNetRoutedModel(stub, tokenizer=None, keys=keys, assignment=assignment,
                              q2author=q2author, k=k, embed_fn=embed_fn,
                              loaded=loaded, merge_cap=merge_cap)


def test_wrapper_route_and_merge_cache():
    cfg = _toy_cfg(n=4, k=2, num_authors=6)
    emb = _toy_author_emb(cfg["num_authors"])
    keys = lt.build_keys(cfg, emb, cache=False)
    asg = lt.build_assignment(cfg, emb, keys, cache=False)
    q2a = {"Who is 0?": 0}
    ood_vec = emb[1]
    w = _make_wrapper(_StubPeft(), keys, asg, q2a, cfg["k"],
                      embed_fn=lambda t: ood_vec, loaded=set(range(cfg["n"])))

    # in-distribution: routes to author 0's frozen top-k
    idxs = w._route("Question: Who is 0?\nAnswer: foo")
    assert idxs == tuple(sorted(lt.author_keys(asg, 0))), idxs
    # OOD: not in q2author -> nearest-cluster of the embedded question
    ood = w._route("Question: Who is Z?\nAnswer: bar")
    assert ood == tuple(sorted(lt.KNNRouter(keys, cfg["k"]).route_one(ood_vec)))

    # merge-cache: same top-k set merges once, set_adapter each time
    stub = w.model
    n_before = len(stub.added)
    w._activate((1, 2)); w._activate((1, 2))
    assert len(stub.added) == n_before + 1, "merge built once and reused"
    assert stub.set_calls[-1] == stub.set_calls[-2] == "_lego_1_2"
    # single-adapter set is direct (no merge)
    w._activate((3,))
    assert stub.set_calls[-1] == "a3"
    assert len(stub.added) == n_before + 1
    print("ok test_wrapper_route_and_merge_cache")


def test_wrapper_merge_cache_eviction():
    cfg = _toy_cfg(n=4, k=2, num_authors=6)
    emb = _toy_author_emb(cfg["num_authors"])
    keys = lt.build_keys(cfg, emb, cache=False)
    asg = lt.build_assignment(cfg, emb, keys, cache=False)
    stub = _StubPeft()
    w = _make_wrapper(stub, keys, asg, {}, cfg["k"], embed_fn=lambda t: emb[0],
                      loaded=set(range(cfg["n"])), merge_cap=1)
    w._activate((0, 1))
    w._activate((2, 3))   # cap=1 -> evicts the first merge
    assert "_lego_0_1" in stub.deleted, stub.deleted
    assert stub.set_calls[-1] == "_lego_2_3"
    print("ok test_wrapper_merge_cache_eviction")


if __name__ == "__main__":
    test_knn_router()
    test_assignment_determinism_and_shape()
    test_affected_union()
    test_frozen_key_invariance()
    test_q2author_and_parse()
    test_wrapper_route_and_merge_cache()
    test_wrapper_merge_cache_eviction()
    print("\nALL LEGONET-TOFU CPU TESTS PASSED")
