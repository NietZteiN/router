"""CPU gate for analyze_router_probe.py (follow-up paper E1) — run before reporting any number:
    ${TOFU_PYTHON:-python3} test_router_probe.py

The claim this script exists to protect is narrow and easy to break by accident: *the probe reads
only what a served post-deletion router can see*. Everything below tests that boundary, plus the
mechanics the self-test does not cover.

  1. run_self_test() — the fixture battery inside the module (separable / null / parity /
     source ranking / permutation invariance / logit_div recompute);
  2. the deletion record is never consulted — dropping `author_sent_scores` from an npz leaves
     the probe AUC bit-identical (only the sentinel comparator disappears);
  3. deleted columns never reach a feature — filling them with garbage leaves the probe AUC
     bit-identical, which is what makes the number a post-deletion claim;
  4. key_exact is skipped with a stated reason, never silently scored (contract note iv);
  5. a strategy whose scores are recomputed per drop set is read from scores__d<ids>, and a
     stale full-pool matrix cannot masquerade as one;
  6. the real snapshot npz (when present) satisfies the contract this reader assumes.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only; never touch a (login-node) GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import glob
import tempfile

import numpy as np

import analyze_router_probe as ARP

DROP = [20, 21, 22, 23]
SNAPSHOT_GLOB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results_snapshot", "tofu", "Llama-2-7B-chat-hf_k200_r32_e25_lr1e4",
    "results", "router_leak", "rl_family_k200.*.npz")


def test_self_test():
    ARP.run_self_test()
    print("ok module self-test battery")


def test_sentinel_is_never_read():
    """The sentinel IS the deletion record. A probe that quietly leaned on it would be
    measuring D2 enforcement, not the router's residual geometry."""
    with tempfile.TemporaryDirectory(prefix="trp_") as td:
        p = os.path.join(td, "a.centroid_sbert.npz")
        ARP._plant_probe_npz(p, "separable")
        z = dict(np.load(p, allow_pickle=False))
        # a sentinel that would trivially solve the task if it ever leaked into the features
        z["author_sent_scores"] = np.where(
            np.asarray(z["is_forget"], dtype=bool)[:, None], 9.0, -9.0).astype("float32")
        z["sent_author_ids"] = np.asarray(DROP, dtype="int32")
        p_s = os.path.join(td, "b.centroid_sbert.npz")
        np.savez(p_s, **z)

        without = ARP.probe_npz(p, DROP, seed=42)
        with_sent = ARP.probe_npz(p_s, DROP, seed=42)
        assert with_sent["probe"]["auc"] == without["probe"]["auc"], (
            with_sent["probe"], without["probe"])
        assert "tomb_author" not in without["comparators"]
        assert with_sent["comparators"]["tomb_author"]["auc"] >= 0.99
    print("ok probe AUC is bit-identical with and without the deletion record present")


def test_dropped_columns_never_reach_features():
    with tempfile.TemporaryDirectory(prefix="trp_") as td:
        p = os.path.join(td, "a.centroid_sbert.npz")
        ARP._plant_probe_npz(p, "separable")
        clean = ARP.probe_npz(p, DROP, seed=42)["probe"]["auc"]

        z = dict(np.load(p, allow_pickle=False))
        sc = z["scores"].astype("float32")
        sc[:, DROP] = 1e6                       # garbage in every deleted column
        z["scores"] = sc
        p_g = os.path.join(td, "b.centroid_sbert.npz")
        np.savez(p_g, **z)
        dirty = ARP.probe_npz(p_g, DROP, seed=42)["probe"]["auc"]
        assert dirty == clean, (dirty, clean)

        surv = ARP.survivor_scores(np.load(p_g, allow_pickle=False), DROP, 24)
        assert surv.shape[1] == 24 - len(DROP)
        assert surv.max() < 1e5
    print("ok deleted columns are removed before any feature is computed")


def test_key_exact_is_skipped_not_scored():
    with tempfile.TemporaryDirectory(prefix="trp_") as td:
        p = os.path.join(td, "a.key_exact.npz")
        ARP._plant_probe_npz(p, "separable")
        z = dict(np.load(p, allow_pickle=False))
        z.pop("scores")
        z["match"] = np.zeros((z["author_of_q"].shape[0], 24), dtype="uint8")
        z["strategy"] = np.str_("key_exact")
        np.savez(p, **z)
        r = ARP.probe_npz(p, DROP, seed=42)
        assert "probe" not in r and "skipped" in r, r
        assert "key_exact" in r["skipped"]
        # and a skipped cell must not silently vote in the verdict
        v = ARP.verdict([r])
        assert v["section"] == "none", v
    print("ok key_exact skipped with a stated reason, excluded from the verdict")


def test_recomputed_matrix_is_preferred():
    """A candidate-set-dependent router must be read from its recomputed matrix. If the reader
    fell back to the full-pool one, the planted disagreement below would go unnoticed."""
    with tempfile.TemporaryDirectory(prefix="trp_") as td:
        p = os.path.join(td, "a.logit_div.npz")
        ARP._plant_probe_npz(p, "flat")
        z = dict(np.load(p, allow_pickle=False))
        rec = z["scores"].astype("float64").copy()
        rec[:, DROP] = np.nan
        rec[np.asarray(z["is_forget"], dtype=bool)] += 5.0
        z[f"scores__{ARP.cell_key(DROP)}"] = rec.astype("float32")
        z["strategy"] = np.str_("logit_div")
        np.savez(p, **z)

        assert ARP.probe_npz(p, DROP, seed=42)["probe"]["auc"] >= 0.95
        # a different drop set has no recomputed matrix -> the full-pool one, which is null
        assert 0.30 <= ARP.probe_npz(p, [20, 21], seed=42)["probe"]["auc"] <= 0.70

        # NaN inside the survivor slice is a mismatched producer/consumer pair, not a warning
        bad = dict(np.load(p, allow_pickle=False))
        m = bad[f"scores__{ARP.cell_key(DROP)}"].copy()
        m[0, 0] = np.nan
        bad[f"scores__{ARP.cell_key(DROP)}"] = m
        p_bad = os.path.join(td, "b.logit_div.npz")
        np.savez(p_bad, **bad)
        try:
            ARP.probe_npz(p_bad, DROP, seed=42)
            raise AssertionError("NaN in the survivor slice did not raise")
        except ValueError as e:
            assert "finite" in str(e), e
    print("ok scores__d<ids> preferred; NaN inside the survivor slice raises")


def test_snapshot_contract():
    paths = sorted(glob.glob(SNAPSHOT_GLOB))
    if not paths:
        print("skip snapshot contract (results_snapshot not present)")
        return
    graded = 0
    for p in paths:
        z = np.load(p, allow_pickle=False)
        for key in ("is_forget", "author_of_q", "k", "strategy"):
            assert key in z.files, f"{os.path.basename(p)} missing {key}"
        assert z["is_forget"].dtype == np.bool_
        assert z["author_of_q"].dtype == np.int32
        if "scores" in z.files:
            assert z["scores"].ndim == 2 and z["scores"].shape[1] == int(z["k"])
            graded += 1
    assert graded >= 1, "no graded score matrix in the snapshot"
    print(f"ok snapshot contract ({len(paths)} npz, {graded} graded)")


if __name__ == "__main__":
    test_self_test()
    test_sentinel_is_never_read()
    test_dropped_columns_never_reach_features()
    test_key_exact_is_skipped_not_scored()
    test_recomputed_matrix_is_preferred()
    test_snapshot_contract()
    print("ALL OK test_router_probe")
