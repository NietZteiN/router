"""CPU gate for router_family_audit.py (router_leak all-router sweep) — run before any
SLURM job:
    ${TOFU_PYTHON:-python3} test_router_family.py

Covers (pattern: test_router_leak.py / test_routing_audit_tofu.py — tiny random Llama
stub + synthetic fixtures, no HF hub, no GPU):
  1. per-sample activation-norm path at bs=1 == router._lora_b_norm scalar (fp tol),
     all-modules AND attn-only;
  2. masking invariant: no dropped shard is ever the post-drop argmax; survivors<1 raises;
  3. ppl sign convention (score = NEGATIVE loss; argmax == PplRouter's argmin-loss route)
     + per-sample CE at bs=1 == HF model(..., labels).loss;
  4. logit_div per-cell recompute differs from naive column-masking of the full-set
     matrix (guards against the shortcut), + full-set batched argmax == ActivationRouter;
  5. key_exact post-drop fallback lands on candidates[0] and the no-match flag fires;
  6. adequacy-ratio definitions on a SEPARABLE synthetic case (low adequacy, abstain
     WOULD work) vs an OVERLAPPING case (high adequacy, no usable threshold) — the
     07-07 separable-synthetic discipline — + the ppl-family ratio direction;
  7. npz round-trip vs THE FAMILY NPZ CONTRACT via the --stub end-to-end run (all
     required keys, dtypes, NaN placement, sentinel presence/absence, self-check green);
  8. retain-sample determinism (RandomState(42) reproduces the analyze_router_tofu draw).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only; never touch a (login-node) GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import tempfile

import numpy as np
import torch

import router as R
import router_family_audit as RFA


def _tiny_fixture(k=3):
    model = RFA.build_stub_lm(k, seed=42)
    tok = RFA.StubTokenizer()
    return model, tok


def test_lora_b_norm_per_sample():
    model, tok = _tiny_fixture(k=2)
    enc = tok("What themes does Aldous Prine explore in book 0?")
    ids = enc["input_ids"]
    for attn_only in (False, True):
        ref = R._lora_b_norm(model, "shard_0", ids, attn_only=attn_only)
        got = RFA.lora_b_norms_batch(model, "shard_0", ids, attn_only=attn_only)
        assert got.shape == (1,)
        assert abs(got[0] - ref) <= 1e-4 * max(abs(ref), 1.0), (attn_only, got[0], ref)
    # attn-only must be a strict subset of the all-module sum (up_proj is targeted)
    a = RFA.lora_b_norms_batch(model, "shard_0", ids, attn_only=False)[0]
    b = RFA.lora_b_norms_batch(model, "shard_0", ids, attn_only=True)[0]
    assert 0 < b < a, (a, b)
    print("ok per-sample lora_B norm == _lora_b_norm @bs=1 (all + attn-only, subset)")


def test_masking_invariant():
    rng = np.random.RandomState(0)
    scores = rng.randn(50, 4).astype("float32")
    for drop in ([3], [3, 2], [0, 2, 3]):
        surv = [j for j in range(4) if j not in set(drop)]
        top1 = RFA.masked_top1(scores, surv)
        assert not np.isin(top1, drop).any(), f"dropped shard argmax'd post-drop: {drop}"
        assert np.isin(top1, surv).all()
    try:
        RFA.masked_top1(scores, [])
        raise AssertionError("survivors<1 must raise")
    except ValueError:
        pass
    # aggregate path raises too when a drop set empties the pool
    try:
        RFA.aggregate_strategy_cells("centroid_sbert", 4, np.zeros(50, dtype=int),
                                     [[0, 1, 2, 3]], scores=scores)
        raise AssertionError("drop-all must raise in aggregate_strategy_cells")
    except ValueError:
        pass
    print("ok masking invariant (no dropped argmax; survivors<1 raises)")


def test_ppl_sign_convention():
    model, tok = _tiny_fixture(k=3)
    qs = [f"What themes does {n} explore in book 0?" for n in
          ("Aldous Prine", "Bekka Vole", "Cormac Dale")]
    mats = RFA.score_norm_ppl_family(model, tok, qs, 3, ["ppl"], bs=2, device="cpu")
    scores = mats["ppl"]
    # bs=1 per-sample CE == HF loss (labels=input_ids, no padding)
    enc = tok(qs[0])
    model.set_adapter("shard_1")
    with torch.no_grad():
        hf = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                   labels=enc["input_ids"].clone()).loss.item()
    assert abs(-scores[0, 1] - hf) < 1e-4, (-scores[0, 1], hf)
    # higher score == lower loss == PplRouter's routed shard
    pr = R.PplRouter(model, tok, 3)
    for i, q in enumerate(qs):
        assert int(np.argmax(scores[i])) == pr.route(q), (i, scores[i])
    assert (scores < 0).all(), "ppl scores must be NEGATIVE losses"
    print("ok ppl sign convention (-loss; argmax == PplRouter route; bs=1 == HF loss)")


def test_logit_div_recompute_differs():
    # constructed cache: shard 2 is a far outlier — removing it moves the candidate mean,
    # so survivor scores MUST change (naive column-masking would keep them frozen)
    T, V = 3, 5
    cached = {0: torch.zeros(1, T, V), 1: torch.ones(1, T, V),
              2: 10.0 * torch.ones(1, T, V)}
    full = RFA.logit_div_from_cached(cached, [0, 1, 2])
    dropped = RFA.logit_div_from_cached(cached, [0, 1])
    for j in (0, 1):
        assert abs(full[j][0] - dropped[j][0]) > 1e-3, \
            f"recompute == column mask for shard {j} (shortcut not guarded)"
    try:
        RFA.logit_div_from_cached(cached, [])
        raise AssertionError("empty candidate set must raise")
    except ValueError:
        pass
    # end-to-end on the tiny model: full-set batched argmax == ActivationRouter.route,
    # and the score_logit_div drop-set matrix is NaN exactly at dropped columns
    model, tok = _tiny_fixture(k=3)
    qs = [f"What themes does {n} explore in book 1?" for n in
          ("Aldous Prine", "Bekka Vole")]
    scores, cells = RFA.score_logit_div(model, tok, qs, 3, [[2]], bs=2, device="cpu")
    ar = R.ActivationRouter(model, 3, mode="logit_div")
    for i, q in enumerate(qs):
        ids = tok(q)["input_ids"]
        assert int(np.argmax(scores[i])) == ar.route(ids)
    d2 = cells["d2"]
    assert np.isnan(d2[:, 2]).all() and not np.isnan(d2[:, [0, 1]]).any()
    # the recompute must actually differ from the masked full matrix somewhere
    assert not np.allclose(d2[:, [0, 1]], scores[:, [0, 1]]), \
        "score_logit_div drop cell == column-masked full matrix"
    print("ok logit_div (recompute != column mask; argmax == router; NaN placement)")


def test_key_exact_fallback():
    match = np.array([[0, 1, 0],    # matches only shard 1
                      [1, 0, 0],    # matches shard 0
                      [0, 0, 0]],   # matches nothing
                     dtype="uint8")
    routes, nomatch = RFA.key_exact_routes(match, [0, 1, 2])
    assert routes.tolist() == [1, 0, 0] and nomatch.tolist() == [False, False, True]
    # drop shard 1: its query falls to candidates[0] with the no-match flag set
    routes_d, nomatch_d = RFA.key_exact_routes(match, [0, 2])
    assert routes_d[0] == 0 and nomatch_d[0], "fallback must be candidates[0] + flag"
    # KeyRouter serving semantics agree
    kr = R.KeyRouter({0: ["Alice Smith"], 1: ["Bob Jones"], 2: ["Cara Voss"]},
                     method="exact")
    assert kr.route("what did bob jones write?", exclude=frozenset({1})) == 0
    assert kr.route("what did bob jones write?") == 1
    print("ok key_exact fallback (candidates[0] + no-match flag; KeyRouter agrees)")


def test_adequacy_separable_vs_overlap():
    from analyze_router_leak import _fpr_at_catch
    rng = np.random.RandomState(42)
    n_f, n_r, k = 60, 200, 4
    drop = 3
    shard_of_q = np.concatenate([np.full(n_f, drop), rng.randint(0, drop, n_r)])

    def _mk(masked_lo, masked_hi):
        """Orphans: unmasked top-1 = 0.9 on the dropped col; masked top-1 in
        [masked_lo, masked_hi]. Retain: confident 0.85-0.95 on its own surviving shard."""
        s = rng.uniform(0.0, 0.2, (n_f + n_r, k)).astype("float32")
        for i in range(n_f):
            s[i, drop] = 0.9
            s[i, rng.randint(0, drop)] = rng.uniform(masked_lo, masked_hi)
        for i in range(n_f, n_f + n_r):
            s[i, shard_of_q[i]] = rng.uniform(0.85, 0.95)
        return s

    # SEPARABLE: sibling scores the orphan far worse -> low adequacy AND a working
    # abstain threshold on -masked-top1 (the identity signal a leaky family lacks)
    s_sep = _mk(0.25, 0.35)
    cells = RFA.aggregate_strategy_cells("centroid_sbert", k, shard_of_q, [[drop]],
                                         scores=s_sep)
    adq = cells[f"d{drop}"]["adequacy"]
    assert adq["mean"] < 0.5, adq
    assert adq["definition"] == "masked_top1_cos / unmasked_top1_cos"
    surv = [j for j in range(k) if j != drop]
    pos = -s_sep[:n_f][:, surv].max(1)          # orphan detector score (higher=orphan)
    neg = -s_sep[n_f:][:, surv].max(1)
    op = _fpr_at_catch(pos, neg, 0.90)
    assert op["orphan_catch"] >= 0.90 and op["retain_fpr"] <= 0.10, op

    # OVERLAPPING: sibling nearly as good -> adequacy ~1 and NO usable threshold
    s_ovl = _mk(0.84, 0.94)
    cells = RFA.aggregate_strategy_cells("centroid_sbert", k, shard_of_q, [[drop]],
                                         scores=s_ovl)
    adq = cells[f"d{drop}"]["adequacy"]
    assert adq["mean"] >= 0.9, adq
    pos = -s_ovl[:n_f][:, surv].max(1)
    neg = -s_ovl[n_f:][:, surv].max(1)
    op = _fpr_at_catch(pos, neg, 0.90)
    assert op["retain_fpr"] > 0.30, f"overlapping case unexpectedly separable: {op}"

    # ppl-family direction: scores are -loss; unmasked loss 1.0, masked loss 2.0
    ratio, definition = RFA.adequacy_ratios("ppl", np.array([-1.0]), np.array([-2.0]))
    assert abs(ratio[0] - 0.5) < 1e-12 and definition == \
        "unmasked_top1_loss / masked_top1_loss"
    # norm family: masked/unmasked
    ratio, _ = RFA.adequacy_ratios("norm", np.array([4.0]), np.array([3.0]))
    assert abs(ratio[0] - 0.75) < 1e-12
    print("ok adequacy (separable: low ratio + abstain works; overlapping: ~1 + no "
          "threshold; ppl/norm definitions)")


def test_stub_end_to_end_npz_contract():
    import router_family_audit as rfa
    tmp = tempfile.mkdtemp(prefix="router_family_")
    out = os.path.join(tmp, "rl_family_stub.json")
    args = rfa.main.__globals__["argparse"].Namespace(
        pool_dir=tmp, base_model=None, k=4,
        strategies=list(rfa.ALL_STRATEGIES), drop_sets="3;3,2",
        queries="all", device="cpu", hf_home="/nonexistent", out=out,
        dump_sims=True, self_check=12, bs=4, logitdiv_bs=2, embed_bs=4,
        seed=42, stub=True)
    res = rfa.run(args)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    k, n_q = 4, 8 * 4
    drop_sets = [[3], [3, 2]]
    stem = out[:-5]
    for strategy in rfa.ALL_STRATEGIES:
        e = res["strategies"][strategy]
        assert e["self_check"]["passed"] == e["self_check"]["n"] == 12, (strategy, e)
        assert set(e["cells"]) == {"d3", "d3_2"}
        z = np.load(f"{stem}.{strategy}.npz", allow_pickle=False)
        # contract scalars + row metadata
        assert int(z["k"]) == k and str(z["strategy"]) == strategy
        assert json.loads(str(z["drop_sets"])) == drop_sets
        assert z["is_forget"].dtype == np.bool_ and z["is_forget"].shape == (n_q,)
        assert z["author_of_q"].dtype == np.int32 and z["author_of_q"].shape == (n_q,)
        if strategy == "key_exact":
            assert z["match"].dtype == np.uint8 and z["match"].shape == (n_q, k)
            assert "scores" not in z and "author_sent_scores" not in z
            for ck in ("d3", "d3_2"):
                assert "no_match" in e["cells"][ck]
        else:
            assert z["scores"].dtype == np.float32 and z["scores"].shape == (n_q, k)
            assert not np.isnan(z["scores"]).any()
        if strategy == "logit_div":
            for ck, ids in (("d3", [3]), ("d3_2", [3, 2])):
                m = z[f"scores__{ck}"]
                assert m.dtype == np.float32 and m.shape == (n_q, k)
                surv = [j for j in range(k) if j not in ids]
                assert np.isnan(m[:, ids]).all() and not np.isnan(m[:, surv]).any()
        else:
            assert not any(kk.startswith("scores__") for kk in z.files), strategy
        if strategy in rfa.FEATURE_SPACE:
            s = z["author_sent_scores"]
            ids = z["sent_author_ids"]
            assert s.dtype == np.float32 and ids.dtype == np.int32
            # union of drop sets {3, 2} at 8 authors / 4 shards -> authors 4..7
            assert ids.tolist() == [4, 5, 6, 7] and s.shape == (n_q, 4)
        elif strategy in rfa.BEHAVIORAL:
            assert "author_sent_scores" not in z, strategy
    # ppl npz stores negative losses (sign convention survives the round trip)
    zppl = np.load(f"{stem}.ppl.npz", allow_pickle=False)
    assert (zppl["scores"] < 0).all()
    assert res["oracle"]["by_construction"] is True
    assert res["oracle"]["cells"]["d3"] == {"orphan_base_capture": 1.0,
                                            "retain_shift_top1": 0.0}
    assert len(res["meta"]["script_sha256"]) == 64
    print("ok stub end-to-end (10 strategies; npz contract keys/dtypes/NaN; "
          "self-check 12/12 each; oracle block)")


def test_retain_sample_determinism():
    f1, r1 = RFA.sample_query_rows()
    f2, r2 = RFA.sample_query_rows()
    assert f1 == f2 and r1 == r2
    assert len(f1) == 400 and len(r1) == 400
    # verbatim re-implementation of analyze_router_tofu._records' draw
    per = 20
    forget = set(range(180, 200))
    retain_pool = np.array([r for r in range(200 * per) if (r // per) not in forget],
                           dtype=int)
    rng = np.random.RandomState(42)
    expect = sorted(int(r) for r in
                    retain_pool[rng.choice(len(retain_pool), size=400, replace=False)])
    assert r1 == expect, "retain sample deviates from the analyze_router_tofu draw"
    assert all((r // per) < 180 for r in r1)
    assert f1 == [a * per + i for a in range(180, 200) for i in range(per)]
    print("ok retain-sample determinism (RandomState(42) == analyze_router_tofu draw)")


def test_self_check_tie_tolerance():
    """The faithfulness gate tolerates bf16 near-tie argmax flips but still catches real
    disagreements (the 2026-07-21 J2-behavioral fix)."""
    import numpy as np
    from router_family_audit import run_self_check
    qs = ["q0", "q1"]
    sel = np.asarray([0, 1])
    top1 = np.asarray([9, 5])                      # matrix argmax
    # near-tie: reference picks shard 7 for q0, scored 0.4% below the argmax shard 9
    scores = np.zeros((2, 10), dtype="float32")
    scores[0, 9] = 12.500; scores[0, 7] = 12.450   # 0.4% gap < 2% band -> tolerated
    scores[1, 5] = 3.0                             # q1 agrees (ref returns 5)
    ref = lambda q: {"q0": 7, "q1": 5}[q]
    sc = run_self_check("activation_norm", ref, qs, top1, sel, scores=scores)
    assert sc["passed"] == 2 and sc["ties_tolerated"] == 1, sc
    # material disagreement (shard 7 scored 30% below 9) must still raise
    scores[0, 7] = 8.0
    try:
        run_self_check("activation_norm", ref, qs, top1, sel, scores=scores)
        raise AssertionError("material disagreement should have raised")
    except AssertionError as e:
        assert "real disagreement" in str(e), e
    # no score matrix (key_exact path) -> strict, any disagreement raises
    try:
        run_self_check("key_exact", ref, qs, top1, sel, scores=None)
        raise AssertionError("strict path should have raised")
    except AssertionError as e:
        assert "matrix argmax" in str(e), e
    print("ok self-check tie tolerance (near-tie tolerated, real gap + strict path raise)")


def test_high_k_behavioral_guard():
    """The memory law at k>50, and exactly how --lazy_adapter_cache does and does not lift it.

    The split is about the ACCESS PATTERN. ppl/activation_norm/attn_norm loop shards OUTER, so a
    lazy cache costs k loads for the whole run. logit_div loops query batches outer and every
    shard inner, and holds one logits tensor per shard — no cache size makes that fit, so it must
    stay refused rather than run into an OOM two hours in.
    """
    import argparse
    import inspect

    def _res(k, strategies, lazy=0):
        args = argparse.Namespace(k=k, pool_dir="/nonexistent", base_model=None,
                                  device="cpu", lazy_adapter_cache=lazy, hf_home=os.environ.get(
                                      "HF_HOME", ""), stub=False)
        return RFA.build_real_resources(args, strategies)

    # no lazy cache -> the historical refusal, unchanged
    try:
        _res(200, ["ppl"])
        raise AssertionError("k=200 behavioral without a lazy cache did not raise")
    except SystemExit as e:
        assert "memory law" in str(e), e

    # lazy cache -> logit_div still refused, and for its own stated reason
    try:
        _res(200, ["ppl", "logit_div"], lazy=8)
        raise AssertionError("logit_div at k=200 accepted under a lazy cache")
    except SystemExit as e:
        assert "logit_div" in str(e) and "per query batch" in str(e), e

    # feature-space at high k is untouched by any of this
    src = inspect.getsource(RFA.build_real_resources)
    assert "need_adapters and args.k > 50" in src

    # the two loops really do have the access patterns the guard claims
    npp = inspect.getsource(RFA.score_norm_ppl_family)
    ld = inspect.getsource(RFA.score_logit_div)
    assert npp.index("for shard in range(k)") < npp.index("for lo in range(0, n_q, bs)"), \
        "norm/ppl is no longer shard-outer — the lazy-cache justification is stale"
    assert ld.index("for lo in range(0, n_q, bs)") < ld.index("for shard in range(k)"), \
        "logit_div is no longer batch-outer — re-examine the guard"

    # hooks are registered AFTER activation, or a lazy adapter's lora_B would not exist yet
    # match the CALL sites, not any prose mentioning them
    lb = inspect.getsource(RFA.lora_b_norms_batch)
    assert lb.index("model.set_adapter(adapter_name)") < \
        lb.index("handles, is_attn = _register_persample_hooks"), \
        "lora_b_norms_batch registers hooks before set_adapter — breaks under a lazy cache"
    npp2 = inspect.getsource(RFA.score_norm_ppl_family)
    assert npp2.index("model.set_adapter(aname)") < npp2.index("_register_persample_hooks"), \
        "score_norm_ppl_family registers hooks before set_adapter"
    print("ok high-k behavioral guard (norm/ppl lazy-enabled, logit_div refused, hook order)")


if __name__ == "__main__":
    test_lora_b_norm_per_sample()
    test_masking_invariant()
    test_ppl_sign_convention()
    test_logit_div_recompute_differs()
    test_key_exact_fallback()
    test_adequacy_separable_vs_overlap()
    test_stub_end_to_end_npz_contract()
    test_retain_sample_determinism()
    test_self_check_tie_tolerance()
    test_high_k_behavioral_guard()
    print("ALL OK test_router_family")
