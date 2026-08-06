"""CPU regression for routing_audit_tofu.py (run before any SLURM job — CLAUDE.md §4):
tiny expert pool + synthetic deletion manifest, then exercise the real audit functions —
stale vs rebuilt index divergence on affected experts, bit-identical untouched rows,
deletion-invariant key routing (measured zero shift), JSON round-trip, and the
sorted-tuple-vs-rank-preserving top-1 regression (the RamoleRouter.route pitfall).

    ${TOFU_PYTHON:-python3} test_routing_audit_tofu.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only test; never touch the (login-node) GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hashlib
import json
import sys
import tempfile

import numpy as np

TOFU_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOFU_DIR)

from test_ramole_tofu import _cfg, _make_base_and_experts, _synth_tofu  # noqa: E402
import legonet_tofu as lt           # noqa: E402
import ramole_tofu as rt            # noqa: E402 (inserts ramole/ on sys.path)
import ramole_common as rc          # noqa: E402
import routing_audit_tofu as audit  # noqa: E402


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="routing_audit_tofu_")
    n, k, A, per = 3, 2, 6, 4
    base_dir = _make_base_and_experts(tmp, n)
    data_full = _synth_tofu(A, per)
    cfg = _cfg(tmp, base_dir, n, k, A, per)

    author_emb = lt.author_answer_embeddings(cfg, data_full, device="cpu")
    keys = lt.build_keys(cfg, author_emb)
    assignment = lt.build_assignment(cfg, author_emb, keys)

    # Forget author whose experts all keep >=2 members after exclusion (build_expert_index
    # raises on an emptied member set — the fixture must not trip that guard).
    forget = next(a for a in range(A)
                  if all(len(assignment["members"][str(j)]) >= 2
                         for j in lt.author_keys(assignment, a)))
    cfg["forget_authors"] = [forget]

    tag = "forget1"
    aff = lt.affected_adapters(assignment, [forget])
    manifest = {"tag": tag, "forget_authors": [forget], "affected_adapters": aff,
                "disabled_adapters": [],
                "retrained_dirs": {str(j): lt.adapter_dir(cfg, j) for j in aff},
                "untouched_adapters": [j for j in range(n) if j not in aff]}
    assert all(os.path.isdir(d) for d in manifest["retrained_dirs"].values())
    mpath = lt.unlearn_manifest_path(cfg, tag)
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] fixture: forget author {forget}, affected={aff}, "
          f"untouched={manifest['untouched_adapters']}")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(rt._encoder_source(cfg), device="cpu")   # loaded ONCE

    # pre-build the stale index so its bytes can be hash-checked across the audit
    idx_stale = rt.build_expert_index(cfg, data_full, device="cpu", encoder=enc)
    stale_path = rt.expert_index_path(cfg)
    h0 = _sha(stale_path)

    res = audit.run_audit(cfg, data_full, tag, ["stale", "rebuilt", "key"],
                          device="cpu", encoder=enc)

    # rebuilt cached to a DISTINCT _ex file; the stale cache is byte-identical
    assert _sha(stale_path) == h0, "stale expert index was modified by the audit"
    ex_files = [p for p in os.listdir(rt.ramole_dir(cfg))
                if p.startswith(f"expert_index_n{n}_ex") and p.endswith(".npy")]
    assert len(ex_files) == 1 and os.path.basename(stale_path) not in ex_files
    print(f"[ok] rebuilt index cached separately: {ex_files[0]}; stale bytes unchanged")

    # rebuilt differs on affected rows OR a forget-author query routes differently
    idx_reb = rt.build_expert_index(cfg, data_full, device="cpu", encoder=enc,
                                    exclude_authors=[forget])   # cache hit on the _ex file
    aff_diff = any(not np.array_equal(idx_stale[j], idx_reb[j]) for j in aff)
    qembed = rc.make_embed_fn(rt._encoder_name(cfg), instruction=rt._instr(cfg),
                              device="cpu", encoder=enc)
    qv = qembed([data_full[forget * per]["question"]])
    route_diff = not np.array_equal(audit.rank_topk(idx_stale, qv, k),
                                    audit.rank_topk(idx_reb, qv, k))
    assert aff_diff or route_diff, "exclusion changed neither affected rows nor routing"
    print(f"[ok] stale vs rebuilt diverge (affected_rows_differ={aff_diff}, "
          f"forget_route_differs={route_diff})")

    # untouched rows bit-identical (run_audit also asserts this internally)
    for j in manifest["untouched_adapters"]:
        assert np.array_equal(idx_stale[j], idx_reb[j]), f"untouched expert {j} row moved"
    assert res["index_displacement"]["untouched_bit_equal"] is True
    print("[ok] untouched expert rows bit-identical stale vs rebuilt")

    # key policy: measured (not assumed) zero selection shift; exact orphan routing
    ks = res["selection_shift"]["key"]
    assert ks["shift_topk"] == 0.0 and ks["shift_top1"] == 0.0 and ks["mean_jaccard"] == 1.0
    kp = res["policies"]["key"]
    assert kp["orig_topk_rate"] == 1.0 and kp["orig_top1_rate"] == 1.0
    assert kp["sibling_top1_rate"] == 0.0 and kp["affected_mass"] == 1.0
    print("[ok] key policy: selection shift == 0, orphan routing exact by construction")

    # per-author top1 histograms cover every forget question exactly once
    for pol in ("stale", "rebuilt", "key"):
        hists = res["policies"][pol]["per_author"]
        assert sorted(hists) == [str(forget)]
        assert sum(hists[str(forget)].values()) == per
    print("[ok] per_author top1 histograms sum to records_per_author")

    # JSON written + loadable, with every required section
    out = os.path.join(tmp, "audit.json")
    rc.write_json(out, res)
    with open(out) as f:
        back = json.load(f)
    for sec in ("tag", "encoder_source", "n_forget_q", "n_retain_q", "policies",
                "selection_shift", "index_displacement"):
        assert sec in back, f"missing section {sec}"
    assert back["encoder_source"] == rt._encoder_source(cfg)
    assert back["n_forget_q"] == per and back["n_retain_q"] == (A - 1) * per
    assert "embed_stale_vs_rebuilt" in back["selection_shift"]
    print(f"[ok] JSON round-trip with all sections -> {out}")

    # REGRESSION: RamoleRouter.route returns tuple(sorted(ids)) — its [0] is NOT the top-1.
    # Construct a query nearest expert 1: sorted-tuple gives 0, rank-preserving gives 1.
    index = np.eye(2, dtype="float32")
    q = np.array([[0.1, 0.9]], dtype="float32")
    q /= np.linalg.norm(q)
    router = rt.RamoleRouter("embed", 2, {0, 1}, index=index,
                             qembed=lambda ts: np.repeat(q, len(ts), axis=0))
    sorted_top1 = int(router.route("a question nearest expert one")[0])
    rank_top1 = int(audit.rank_topk(index, q, 2)[0][0])
    assert sorted_top1 == 0 and rank_top1 == 1 and sorted_top1 != rank_top1
    print(f"[ok] regression: sorted-tuple top-1 ({sorted_top1}) != "
          f"rank-preserving top-1 ({rank_top1})")

    # encoder_pin="base" caches to a DISTINCT _encbase file; stale bytes untouched. The fixture
    # has no FT retriever, so base == auto encoder and the CONTENT must match the stale index
    # exactly — only the filename (provenance) differs.
    cfg_bp = dict(cfg)
    cfg_bp["encoder_pin"] = "base"
    idx_bp = rt.build_expert_index(cfg_bp, data_full, device="cpu", encoder=enc)
    bp_path = rt.expert_index_path(cfg_bp, "_encbase")
    assert os.path.exists(bp_path) and bp_path != stale_path
    assert _sha(stale_path) == h0, "base-pin build modified the stale index"
    assert np.array_equal(idx_bp, idx_stale)
    rt.build_expert_index(cfg_bp, data_full, device="cpu", encoder=enc,
                          exclude_authors=[forget])
    assert any(p.startswith(f"expert_index_n{n}_encbase_ex")
               for p in os.listdir(rt.ramole_dir(cfg))), "base-pin exclusion suffix missing"
    print(f"[ok] encoder_pin=base caches to {os.path.basename(bp_path)} (+_ex variant); "
          "stale bytes unchanged, content identical under the fixture's single encoder")

    # dropped policy: a self-consistent k=1 fixture (serving k must match the SOURCE assignment
    # k — the E4 path-bug lesson — so cfg["k"] cannot simply be flipped). k=1 -> |affected|=1
    # -> 2 survivors >= k. No affected expert may appear in any top-k slot; extras are
    # well-formed; retain shift vs stale is recorded.
    tmp2 = tempfile.mkdtemp(prefix="routing_audit_tofu_k1_")
    base_dir2 = _make_base_and_experts(tmp2, n)
    cfg_k1 = _cfg(tmp2, base_dir2, n, 1, A, per)
    emb2 = lt.author_answer_embeddings(cfg_k1, data_full, device="cpu")
    asg2 = lt.build_assignment(cfg_k1, emb2, lt.build_keys(cfg_k1, emb2))
    forget2 = next(a for a in range(A)
                   if all(len(asg2["members"][str(j)]) >= 2
                          for j in lt.author_keys(asg2, a)))
    cfg_k1["forget_authors"] = [forget2]
    aff2 = lt.affected_adapters(asg2, [forget2])
    man2 = {"tag": tag, "forget_authors": [forget2], "affected_adapters": aff2,
            "disabled_adapters": [],
            "retrained_dirs": {str(j): lt.adapter_dir(cfg_k1, j) for j in aff2},
            "untouched_adapters": [j for j in range(n) if j not in aff2]}
    mpath2 = lt.unlearn_manifest_path(cfg_k1, tag)
    os.makedirs(os.path.dirname(mpath2), exist_ok=True)
    with open(mpath2, "w") as f:
        json.dump(man2, f, indent=2)
    res_d = audit.run_audit(cfg_k1, data_full, tag, ["stale", "dropped", "key"],
                            device="cpu", encoder=enc)
    dp = res_d["policies"]["dropped"]
    assert dp["affected_mass"] == 0.0 and dp["orig_top1_rate"] == 0.0
    assert dp["sibling_top1_rate"] == 1.0
    ex = res_d["dropped_extras"]
    surv = set(man2["untouched_adapters"])
    assert ex["n_surviving_experts"] == n - len(aff2)
    assert set(map(int, ex["top1_hist"])) <= surv, "orphan top-1 landed on a masked expert"
    assert sum(ex["top1_hist"].values()) == ex["n_orphans"] == per
    assert 0.0 <= ex["top1_entropy_norm"] <= 1.0
    assert 0.0 < ex["mean_top1_sim_ratio"] <= 1.0 + 1e-6, \
        "masked top-1 sim cannot exceed unmasked"
    assert "embed_stale_vs_dropped" in res_d["selection_shift"]
    print(f"[ok] dropped policy: no masked expert in top-k, extras well-formed "
          f"(survivors={ex['n_surviving_experts']}, sim_ratio={ex['mean_top1_sim_ratio']:.3f})")

    # abstain (C1 fix): synthetic separable case — retain queries match their expert perfectly
    # (sim 1.0) while orphans, with their expert masked, only reach a low sibling sim → a τ between
    # the two abstains all orphans and no retain queries.
    D = 4
    idx4 = np.eye(D, dtype="float32")
    # 3 retain queries each nail an unmasked expert (sim 1.0); 2 orphans point at expert 0 (masked)
    Qr = np.eye(D, dtype="float32")[[1, 2, 3]]
    Qf = np.array([[0.9, 0.2, 0.0, 0.0], [0.9, 0.0, 0.2, 0.0]], dtype="float32")
    Qf = Qf / np.linalg.norm(Qf, axis=1, keepdims=True)
    ab = audit.abstain_analysis(idx4, Qf, Qr, affected=[0], pcts=(5,))["by_pct"]["5"]
    assert ab["orphan_abstain_rate"] == 1.0, ab      # masked orphans fall below τ
    assert ab["retain_false_abstain_rate"] == 0.0, ab  # retain queries (sim 1.0) never abstain
    assert ab["orphan_sibling_rate_if_no_abstain"] == 0.0
    print(f"[ok] abstain: separable τ={ab['tau']:.3f} → orphan→base 1.0, retain false-abstain 0.0")

    # dropped policy refuses a pool where survivors < k (semantics would silently corrupt)
    try:
        audit.run_audit(cfg, data_full, tag, ["dropped"], device="cpu", encoder=enc)
        raise AssertionError("dropped with survivors < k must raise")
    except RuntimeError as e:
        assert "surviving experts" in str(e)
    print("[ok] dropped policy raises when surviving experts < k")

    print("\nALL ROUTING-AUDIT-TOFU TESTS PASSED")


if __name__ == "__main__":
    main()
