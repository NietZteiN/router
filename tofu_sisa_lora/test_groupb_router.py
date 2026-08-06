"""CPU gate for the Group-B realistic-selector (router-leak (b)) — run before any SLURM job:
    python test_groupb_router.py
Tests the pure routing logic (unit sets + the attach_realistic_router decision) with a stub
encoder — no HF hub, no GPU, no model weights.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import types
import numpy as np


def test_unit_sets():
    from groupb_realistic_router import per_author_units, clamu_cluster_units
    u = per_author_units(6, [4, 5])
    assert set(u) == {0, 1, 2, 3} and all(u[a] == [a] for a in u), u
    assignment = {"members": {"0": [10, 11], "1": [12], "2": []}}   # cluster 2 empty -> dropped
    units, repr_a = clamu_cluster_units(assignment)
    assert units == {0: [10, 11], 1: [12]} and repr_a == {0: 10, 1: 12}, (units, repr_a)
    print("ok unit sets (per-author excludes forget; clamu clusters + min-member repr)")


def test_realistic_route_decision():
    import groupb_realistic_router as gr
    # stub the centroid builder: 3 surviving author-units on 3 axes, stub embed by keyword
    cents = np.eye(3, dtype="float32")
    emb = {"about199": np.array([1, 0, 0], "f4"),   # orphan (author 199, unit dropped)
           "about5":   np.array([0, 1, 0], "f4"),   # retain (author 5, unit survives)
           "capital":  np.array([0, 0, 1], "f4")}   # OOD (not a TOFU author)
    def fake_build(hf_home, unit_to_authors, device, encoder_name=None, embed=None):
        uids = list(unit_to_authors.keys())
        e = lambda ts: np.stack([emb[t] for t in ts])
        return cents[:len(uids)], uids, e
    orig = gr.build_unit_centroids
    gr.build_unit_centroids = fake_build
    try:
        model = types.SimpleNamespace(
            q2author={"about199": 199, "about5": 5})   # 'capital' absent -> OOD
        # patch parse_question/_norm to identity for the stub keys
        import legonet_tofu as lt
        lt_parse, lt_norm = lt.parse_question, lt._norm
        lt.parse_question = lambda t: t
        lt._norm = lambda s: s
        try:
            # surviving units: authors 5 (unit 0), 7 (unit 1), 8 (unit 2); 199 is DELETED
            gr.attach_realistic_router(model, "hf", {5: [5], 7: [7], 8: [8]}, device="cpu",
                                       unit_repr_author={5: 5, 7: 7, 8: 8})
            # note: fake_build ignores which axis maps to which unit; we set emb axes to line
            # up unit-0=author5 axis1, so route 'about5' -> unit whose centroid is e1 = uid[1]
            # Simpler assertions on the STATS + None-gate, which are logic-exact:
            assert model._route("capital") is None                 # OOD -> base
            assert model.route_stats["ood"] == 1
            r = model._route("about199")                            # orphan -> some survivor
            assert r in (5, 7, 8) and model.route_stats["orphan_misrouted"] == 1
            r2 = model._route("about5")                             # retain author, own unit alive
            assert r2 in (5, 7, 8) and model.route_stats["orphan_misrouted"] == 1   # not an orphan
            assert model.route_stats["routed"] == 2
        finally:
            lt.parse_question, lt._norm = lt_parse, lt_norm
    finally:
        gr.build_unit_centroids = orig
    print("ok realistic route (OOD->base gate; orphan counted as misrouted; retain not)")


if __name__ == "__main__":
    test_unit_sets()
    test_realistic_route_decision()
    print("ALL OK test_groupb_router")
