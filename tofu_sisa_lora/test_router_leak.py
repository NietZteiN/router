"""CPU gate for the router-leak campaign (log/router_leak/) — run before any SLURM job:
    python test_router_leak.py
Covers: tombstone_analysis rung math + margins, _auc (midrank ties), _author_sentinels,
EmbedRoutedModel policy decisions (sibling/tombstone/OOD/full), detector_scores separation
on a planted fixture, and aggregate_rho.rho math (same formula test_entangled_facts guards).
No HF hub, no GPU, no model weights.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-12)


def test_auc():
    from routing_audit_tofu import _auc
    assert _auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert _auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert abs(_auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) - 0.5) < 1e-12
    print("ok _auc (perfect / inverted / all-ties midrank)")


def test_author_sentinels():
    from routing_audit_tofu import _author_sentinels
    Q = np.zeros((4, 2), dtype="float32")
    Q[2] = [3.0, 0.0]; Q[3] = [0.0, 4.0]   # author 1 rows (per=2)
    s = _author_sentinels(Q, [1], per=2)
    exp = _unit(np.array([1.5, 2.0]))
    assert np.allclose(s[0], exp, atol=1e-6), (s[0], exp)
    assert abs(np.linalg.norm(s[0]) - 1.0) < 1e-6
    print("ok _author_sentinels (mean + L2 norm)")


def test_tombstone_analysis():
    from routing_audit_tofu import tombstone_analysis
    # 3 experts on axes e0,e1,e2; expert 2 is affected/deleted.
    index = np.eye(3, dtype="float32")
    # orphans: 3 point at the deleted expert, 1 already points at survivor e0 (residual leak)
    Q_f = np.stack([_unit([0.1, 0.0, 1.0])] * 3 + [_unit([1.0, 0.0, 0.2])])
    # retain: 3 clean on survivors, 1 false-positive pointing at the deleted expert
    Q_r = np.stack([_unit([1.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0]),
                    _unit([0.9, 0.1, 0.0]), _unit([0.1, 0.0, 1.0])])
    sent_author = np.stack([_unit([0.0, 0.0, 1.0])])   # sentinel == deleted direction
    out = tombstone_analysis(index, Q_f, Q_r, affected=[2],
                             sent_author=sent_author, sent_name=sent_author.copy())
    e = out["expert"]
    assert abs(e["orphan_catch_rate"] - 0.75) < 1e-9
    assert abs(e["orphan_leak_rate"] - 0.25) < 1e-9
    assert abs(e["retain_false_tombstone_rate"] - 0.25) < 1e-9
    a = out["author"]
    assert abs(a["orphan_catch_rate"] - 0.75) < 1e-9    # sentinel pool reproduces the split
    assert abs(a["retain_false_tombstone_rate"] - 0.25) < 1e-9
    assert out["name"] == a                              # identical sentinels -> identical rates
    # margins: orphan margin positive (tombstone matches best), clean-retain margin negative
    assert e["orphan_margin_mean"] > 0 and a["orphan_margin_mean"] > 0
    print("ok tombstone_analysis (catch/leak/FPR + margins, all three rungs)")


def test_embed_routed_model_policies():
    import torch.nn as nn
    from eval_routed_scaffold import EmbedRoutedModel

    class StubTok:
        def decode(self, ids, skip_special_tokens=True):
            return {0: "Question: who is author zero?\nAnswer: x",
                    1: "Question: who is author nineteen?\nAnswer: x",
                    2: "Question: capital of France?\nAnswer: x"}[int(ids[0])]

    q2a = {"who is author zero?": 0, "who is author nineteen?": 19}
    # k=2 pool (per_shard=10): shard 0 centroid on e0, shard 1 (deleted) on e1
    cents = np.eye(2, dtype="float32")
    emb = {"who is author zero?": _unit([1.0, 0.1]),
           "who is author nineteen?": _unit([0.1, 1.0]),
           "capital of France?": _unit([0.7, 0.7])}
    embed_fn = lambda ts: np.stack([emb[t] for t in ts])
    mk = lambda **kw: EmbedRoutedModel(nn.Identity(), StubTok(), q2a, 2, cents.copy(), [0, 1],
                                       embed_fn, num_authors=20, **kw)
    import numpy as _np
    ids = lambda i: _np.array([i])

    m = mk()                                             # embed-full: no deletion
    assert m._shard_for(ids(0)) == 0 and m._shard_for(ids(1)) == 1
    assert m._shard_for(ids(2)) is None and m.stats["ood"] == 1

    m = mk(delete_shard=1, policy="sibling")             # orphan falls to the sibling
    assert m._shard_for(ids(1)) == 0 and m.stats["route_mismatch"] == 1
    assert m._shard_for(ids(0)) == 0

    m = mk(delete_shard=1, policy="tombstone")           # sentinel hit -> base (None)
    assert m._shard_for(ids(1)) is None and m.stats["deleted"] == 1
    assert m._shard_for(ids(0)) == 0 and m.stats["route_mismatch"] == 0
    try:
        EmbedRoutedModel(nn.Identity(), StubTok(), q2a, 2, cents[:1], [0], embed_fn,
                         num_authors=20, delete_shard=1, policy="tombstone")
        raise AssertionError("tombstone without the deleted centroid must raise")
    except ValueError:
        pass

    # tombstone_author (H3 closer): survivor-only routing + a thresholded author-sentinel gate.
    # sentinel for the deleted shard's author sits on e1; author-19's query (also on e1) has a
    # large margin vs the lone survivor centroid e0 => abstain; author-0's query routes to s0.
    asent = _np.stack([_unit([0.1, 1.0])]).astype("float32")     # one deleted-author sentinel
    ma = mk(delete_shard=1, policy="tombstone_author", author_sentinels=asent, tombstone_tau=0.3)
    assert ma._shard_for(ids(1)) is None and ma.stats["deleted"] == 1, "orphan must abstain"
    assert ma._shard_for(ids(0)) == 0, "survivor query must route to its shard"
    # raise tau above the orphan's margin => it no longer abstains, leaks to the survivor
    ma_hi = mk(delete_shard=1, policy="tombstone_author", author_sentinels=asent, tombstone_tau=2.0)
    assert ma_hi._shard_for(ids(1)) == 0 and ma_hi.stats["deleted"] == 0, "high tau => no abstain"
    try:
        mk(delete_shard=1, policy="tombstone_author", author_sentinels=None, tombstone_tau=0.3)
        raise AssertionError("tombstone_author without sentinels must raise")
    except ValueError:
        pass
    print("ok EmbedRoutedModel (full / sibling leak / tombstone seal / tombstone_author gate / OOD / guard)")


def test_detector_scores():
    from analyze_router_leak import detector_scores, _auc, _fpr_at_catch
    rng = np.random.RandomState(42)
    n_f, n_r, n_surv = 40, 200, 5
    # retain: confident on survivors; orphans: tombstone sim high, survivor sims mediocre
    surv_r = rng.uniform(0.6, 0.9, (n_r, n_surv)).astype("float32")
    surv_f = rng.uniform(0.55, 0.85, (n_f, n_surv)).astype("float32")
    tomb_f = {"author": (surv_f.max(1, keepdims=True) + 0.1).astype("float32")}
    tomb_r = {"author": (surv_r.max(1, keepdims=True) - 0.1).astype("float32")}
    calib = rng.rand(n_r) < 0.5
    s = detector_scores(surv_f, surv_r, tomb_f, tomb_r, calib, surv_f.argmax(1), surv_r.argmax(1))
    for name in ("global_top1", "margin", "knn_density", "per_expert", "tomb_author"):
        assert name in s and s[name][0].shape == (n_f,) and s[name][1].shape == (n_r,)
    pos, neg = s["tomb_author"]
    assert _auc(pos, neg) == 1.0                     # planted margin separates perfectly
    op = _fpr_at_catch(pos, neg, 0.90)
    assert op["orphan_catch"] >= 0.90 and op["retain_fpr"] == 0.0
    print("ok detector_scores (shapes + planted tomb_author AUC 1.0 + operating point)")


def test_rho_math():
    from aggregate_rho import rho
    assert rho(0.5, 0.1, 0.9) == 0.5
    assert rho(1.2, 0.1, 0.9) == 1.0                 # clipped high
    assert rho(0.0, 0.1, 0.9) == 0.0                 # clipped low
    assert rho(0.7, 0.5, 0.5) == 0.0                 # degenerate denom -> 0
    print("ok aggregate_rho.rho (formula + clipping + degenerate)")


if __name__ == "__main__":
    test_auc()
    test_author_sentinels()
    test_tombstone_analysis()
    test_embed_routed_model_policies()
    test_detector_scores()
    test_rho_math()
    print("ALL OK test_router_leak")
