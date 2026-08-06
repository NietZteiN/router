"""CPU regression for routing_audit.py (run before any SLURM job — CLAUDE.md §4):
fixture source run + synthetic corpus/keys/assignment + a synthetic 1-record deletion
manifest, then the real audit — the rebuilt index must land in a NEW `_ex{tag}` file with
the stale `lora_index_n{n}.npy` bytes untouched, untouched cluster rows must survive the
exclude bit-identically (forget record chosen with top-1 = the LAST cluster, so the shared
RandomState member-sampling covers every untouched cluster), and missing-manifest tags
(the d_batch15 case) must be skipped with a warning, not crash.

    ${TOFU_PYTHON:-python3} tests/test_routing_audit.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only test; never touch the (login-node) GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hashlib
import json
import sys
import tempfile

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))  # ramole/
sys.path.insert(0, THIS)                   # tests/ (for _fixture)

from _fixture import build_corpus_and_routing, build_source_run  # noqa: E402
import ramole_common as rc                                       # noqa: E402
import retriever as RET                                          # noqa: E402
import routing_audit as RA                                       # noqa: E402


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="routing_audit_")
    cfg = build_source_run(tmp, n=3)                       # n=3, k=2
    build_corpus_and_routing(cfg, per_cluster=8, device="cpu")
    RET.build_index(cfg, device="cpu")                     # the stale index, real build path
    n = cfg["n"]

    sp = rc.source_paths(cfg)
    with open(sp.assignment_path) as f:
        r2k = json.load(f)["record_to_keys"]
    # forget record with top-1 = the LAST cluster: every untouched cluster then precedes it
    # in build_lora_embeddings' shared-RandomState stream, so bitwise equality must hold for
    # ALL of them (the audit's strongest displacement check gets fully exercised).
    rid = next((r for r, ks in r2k.items() if int(ks[0]) == n - 1), None)
    assert rid is not None, "fixture produced no record with top-1 = last cluster"
    ideal = [int(j) for j in r2k[rid]]

    manifest = {"tag": "t0", "forget_ids": [rid],
                "affected_adapters": ideal, "disabled_adapters": [],
                # retrained_dirs may point at existing adapter dirs (retrain-not-drop)
                "retrained_dirs": {str(j): sp.adapter_dir(j) for j in ideal},
                "untouched_adapters": [j for j in range(n) if j not in set(ideal)]}
    assert all(os.path.isdir(d) for d in manifest["retrained_dirs"].values())
    mdir = os.path.join(sp.run_dir, "unlearn", "t0")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] fixture: forget {rid} ideal={ideal} (top-1 = cluster {n - 1})")

    _, stale_path = RA.resolve_encoder_paths(cfg)
    assert os.path.isfile(stale_path)
    h0 = _sha(stale_path)

    # d_batch15 has no manifest yet -> must be skipped with a warning, not crash
    res = RA.run_audit(cfg, tags=["t0", "d_batch15"], n_retain=10, device="cpu")
    assert res["skipped_tags"] == ["d_batch15"] and list(res["tags"]) == ["t0"]
    print("[ok] missing-manifest tag skipped with a warning")

    # rebuilt index is a NEW file; the original index bytes are unchanged
    ex_path = os.path.join(rc.Paths(cfg).run_dir, f"lora_index_n{n}_ext0.npy")
    assert os.path.isfile(ex_path), f"rebuilt index not written: {ex_path}"
    assert os.path.realpath(ex_path) != os.path.realpath(stale_path)
    assert _sha(stale_path) == h0, "stale lora index bytes changed"
    assert res["tags"]["t0"]["rebuilt_index_path"] == ex_path
    print(f"[ok] rebuilt index -> NEW file {os.path.basename(ex_path)}; stale bytes unchanged")

    t = res["tags"]["t0"]
    rows = t["orphan_records"]
    assert len(rows) == 1 and rows[0]["id"] == rid and rows[0]["ideal"] == ideal
    for pol in ("stale", "rebuilt"):
        r = rows[0][pol]
        assert len(r["topk"]) == cfg["k"] and 0 <= r["top1"] < n
        assert abs(r["affected_mass"] + r["untouched_mass"] - 1.0) < 1e-9
    print("[ok] orphan results are per-record (n=1), both policies populated")

    disp = t["index_displacement"]
    assert len(disp["cos"]) == n and disp["forget_top1_clusters"] == [n - 1]
    # full bitwise coverage by construction (top-1 = last cluster) — and it must hold
    assert set(disp["bitwise_asserted"]) == set(disp["untouched_by_top1"])
    assert all(disp["bit_equal_untouched"].values())
    # the affected top-1 cluster's row actually moved (its member sample changed)
    stale, reb = np.load(stale_path), np.load(ex_path)
    assert not np.array_equal(stale[n - 1], reb[n - 1])
    print(f"[ok] displacement: untouched rows bit-identical, affected cluster moved "
          f"(cos={disp['cos'][n - 1]:.4f})")

    ss = t["selection_shift"]
    assert ss["n"] == 10 and 0.0 <= ss["shift_topk"] <= 1.0 and 0.0 <= ss["shift_top1"] <= 1.0
    pooled = res["orphan_pooled"]
    assert pooled["n_records"] == 1 and {"stale", "rebuilt"} <= set(pooled)
    print(f"[ok] selection shift over {ss['n']} retain records: "
          f"topk={ss['shift_topk']:.2f} top1={ss['shift_top1']:.2f}")

    out = os.path.join(tmp, "audit.json")
    rc.write_json(out, res)
    with open(out) as f:
        back = json.load(f)
    for sec in ("encoder_source", "stale_index_path", "tags", "skipped_tags", "orphan_pooled"):
        assert sec in back, f"missing section {sec}"
    print(f"[ok] JSON round-trip -> {out}")

    # ── --policies default identity: explicit ['stale','rebuilt'] ≡ the flag-free run ──────
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(cfg["encoder_model"], device="cpu")   # reused below
    res2 = RA.run_audit(cfg, tags=["t0", "d_batch15"], n_retain=10, device="cpu",
                        encoder=enc, policies=["stale", "rebuilt"])
    assert json.dumps(res, sort_keys=True) == json.dumps(res2, sort_keys=True), \
        "explicit default policies must reproduce the flag-free output exactly"
    assert "dropped_extras_pooled" not in res2 and "abstain_pooled" not in res2 \
        and "sims_dump" not in res2
    print("[ok] --policies ['stale','rebuilt'] output identical to the historical default")

    # ── dropped policy on a 1-affected manifest (survivors = n-1 >= k) ─────────────────────
    rid1 = sorted(r2k)[0]
    aff1 = [int(r2k[rid1][0])]
    man1 = {"tag": "t1", "forget_ids": [rid1], "affected_adapters": aff1,
            "disabled_adapters": [],
            "retrained_dirs": {str(j): sp.adapter_dir(j) for j in aff1},
            "untouched_adapters": [j for j in range(n) if j not in set(aff1)]}
    mdir1 = os.path.join(sp.run_dir, "unlearn", "t1")
    os.makedirs(mdir1, exist_ok=True)
    with open(os.path.join(mdir1, "manifest.json"), "w") as f:
        json.dump(man1, f, indent=2)
    dump_path = os.path.join(tmp, "audit_t1.sims.npz")
    res_d = RA.run_audit(cfg, tags=["t1"], n_retain=10, device="cpu", encoder=enc,
                         policies=["stale", "dropped", "abstain"], dump_sims_path=dump_path)
    t1 = res_d["tags"]["t1"]
    # no rebuilt side-effect for a run without the rebuilt policy; stale bytes untouched
    assert not os.path.isfile(os.path.join(rc.Paths(cfg).run_dir, f"lora_index_n{n}_ext1.npy"))
    assert _sha(stale_path) == h0, "stale index bytes changed under dropped/abstain"
    for absent in ("rebuilt_index_path", "index_displacement", "selection_shift"):
        assert absent not in t1, f"{absent} must not appear without the rebuilt policy"
    row = t1["orphan_records"][0]
    assert "rebuilt" not in row and set(row["dropped"]["topk"]).isdisjoint(aff1), \
        "masked cluster appeared in the dropped top-k"
    assert row["dropped"]["affected_mass"] == 0.0
    ex1 = t1["dropped_extras"]
    assert ex1["n_surviving_experts"] == n - 1 and ex1["n_orphans"] == 1
    assert set(map(int, ex1["top1_hist"])).isdisjoint(aff1), "orphan top-1 on a masked cluster"
    assert 0.0 < ex1["mean_top1_sim_ratio"] <= 1.0 + 1e-6, \
        "masked top-1 sim cannot exceed unmasked"
    assert len(ex1["per_record_top1_sim_ratio"]) == 1
    ss_d = t1["selection_shift_stale_vs_dropped"]
    assert ss_d["n"] == 10 and 0.0 <= ss_d["shift_top1"] <= 1.0
    pooled_d = res_d["orphan_pooled"]
    assert {"stale", "dropped"} <= set(pooled_d) and "rebuilt" not in pooled_d
    assert res_d["dropped_extras_pooled"]["n_orphans"] == 1
    print(f"[ok] dropped policy: no masked cluster in top-k, extras well-formed "
          f"(survivors={ex1['n_surviving_experts']}, sim_ratio={ex1['mean_top1_sim_ratio']:.3f}), "
          "pooled row present, no rebuilt side-effect")

    # abstain: per-tag AND pooled blocks with retain-percentile + orphan-catch families
    ab = t1["abstain"]
    assert set(ab["by_pct"]) == {"1", "5", "10"} and set(ab["by_orphan_catch"]) == {"0.90", "0.99"}
    for d in list(ab["by_pct"].values()) + list(ab["by_orphan_catch"].values()):
        assert 0.0 <= d["orphan_abstain_rate"] <= 1.0
        assert 0.0 <= d["retain_false_abstain_rate"] <= 1.0
    abp = res_d["abstain_pooled"]
    assert abp["n_orphans"] == 1 and abp["n_retain"] == 10
    print("[ok] abstain: per-tag + pooled blocks (pcts {1,5,10} + catches {0.90,0.99})")

    # sims sidecar: keys/shapes/affected-mask contract; JSON gains only the sims_dump pointer
    assert os.path.isfile(dump_path) and res_d["sims_dump"] == os.path.abspath(dump_path)
    z = np.load(dump_path)
    for key in ("sims_stale_forget", "sims_stale_retain", "forget_ids", "retain_ids",
                "forget_tag_idx", "retain_tag_idx", "tags", "affected_mask",
                "encoder_source", "stale_sha"):
        assert key in z, f"sims sidecar missing {key}"
    assert z["sims_stale_forget"].shape == (1, n) and z["sims_stale_retain"].shape == (10, n)
    assert z["affected_mask"].shape == (1, n) and int(z["affected_mask"][0].sum()) == 1
    assert int(np.nonzero(z["affected_mask"][0])[0][0]) == aff1[0]
    assert list(z["tags"]) == ["t1"] and list(z["forget_ids"]) == [rid1]
    res_d2 = RA.run_audit(cfg, tags=["t1"], n_retain=10, device="cpu", encoder=enc,
                          policies=["stale", "dropped", "abstain"])
    trimmed = dict(res_d)
    trimmed.pop("sims_dump")
    assert json.dumps(trimmed, sort_keys=True) == json.dumps(res_d2, sort_keys=True), \
        "--dump_sims must never mutate the aggregate JSON metrics"
    print("[ok] sims sidecar: keys/shapes/mask correct; aggregate JSON unchanged by the dump")

    # dropped refuses a pool where surviving clusters < k (t0 masks k=2 of n=3)
    try:
        RA.run_audit(cfg, tags=["t0"], n_retain=10, device="cpu", encoder=enc,
                     policies=["dropped"])
        raise AssertionError("dropped with survivors < k must raise")
    except RuntimeError as e:
        assert "surviving" in str(e)
    print("[ok] dropped policy raises when surviving clusters < k")

    # abstain separable synthetic: retain queries nail unmasked clusters (sim 1.0); orphans
    # only reach a low sibling sim once their cluster is masked → a τ between the two
    # abstains every orphan and no retain query.
    D = 4
    idx4 = np.eye(D, dtype="float32")
    Qr = np.eye(D, dtype="float32")[[1, 2, 3]]
    Qf = np.array([[0.9, 0.2, 0.0, 0.0], [0.9, 0.0, 0.2, 0.0]], dtype="float32")
    Qf = Qf / np.linalg.norm(Qf, axis=1, keepdims=True)
    ab_s = RA.abstain_analysis(idx4, Qf, Qr, affected=[0], pcts=(5,))
    b = ab_s["by_pct"]["5"]
    assert b["orphan_abstain_rate"] == 1.0, b
    assert b["retain_false_abstain_rate"] == 0.0, b
    assert b["orphan_sibling_rate_if_no_abstain"] == 0.0
    c90 = ab_s["by_orphan_catch"]["0.90"]
    assert c90["retain_false_abstain_rate"] == 0.0 and c90["tau"] < 1.0
    print(f"[ok] abstain separable synthetic: τ={b['tau']:.3f} → orphan→base 1.0, "
          "retain false-abstain 0.0 (catch-90 τ costs no retain)")

    print("\nALL ROUTING-AUDIT TESTS PASSED")


if __name__ == "__main__":
    main()
