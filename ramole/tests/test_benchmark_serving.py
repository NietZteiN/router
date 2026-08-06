"""CPU test for benchmark_serving (E5) — drives the smoke path programmatically and checks:
  1. result JSON structure complete for all three modes × batch sizes, tokens_per_s > 0,
  2. the batch is genuinely heterogeneous (distinct expert sets; composition stats),
  3. the serving identity: merge_per_group's grouped generations == running LegoNetModel
     per-sample individually (both are the exact 1/k merge — including a real multi-member
     group, which exercises the left-padded sub-batch path).
We deliberately do NOT assert ramole_batched == merge_per_group: the router composes with
learned attention (near-uniform at init, not exactly 1/k).

    cd <repo>/ramole && HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME} \
        ${TOFU_PYTHON:-python3} tests/test_benchmark_serving.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # login node has GPUs; never touch them
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json      # noqa: E402
import sys       # noqa: E402
import tempfile  # noqa: E402

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, THIS)

from _fixture import build_corpus_and_routing, build_source_run  # noqa: E402
import ramole_common as rc               # noqa: E402  (puts legonet_lora on sys.path)
import benchmark_serving as BS           # noqa: E402
from combine import LegoNetModel         # noqa: E402  (legonet)
from eval_ramole import _source_lego_cfg  # noqa: E402


def _gen_one(lego, rec, expert_set, gen_tokens):
    """Reference path: ONE record, individually, through the exact 1/k merge."""
    tok = lego.tokenizer
    tok.padding_side = "left"
    enc = tok([rc.prompt_completion(rec)[0]], return_tensors="pt", padding=True)
    with lego.activated(list(expert_set)) as m:
        out = m.generate(**enc, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    tmp = tempfile.mkdtemp(prefix="ramole_bench_test_")
    cfg = build_source_run(tmp, n=4, hidden=64, layers=2, heads=4, kv_heads=2,
                           rank=4, alpha=8, with_tokenizer=True)
    cfg["name"] = "ramole_bench_test"
    build_corpus_and_routing(cfg, per_cluster=8, device="cpu")
    print("[ok] fixture built (n=4, k=2)")

    out_path = os.path.join(tmp, "bench.json")
    res = BS.run_benchmark(cfg, batch_sizes=[1, 2], gen_tokens=4, iters=1, device="cpu",
                           out=out_path, capture_outputs=True)

    # ── 1. structure complete, throughput positive (in-memory AND on disk) ────────
    assert os.path.isfile(out_path), "result JSON not written"
    with open(out_path) as f:
        disk = json.load(f)
    for blob in (res, disk):
        assert blob["gen_tokens"] == 4 and blob["iters"] == 1 and blob["device"] == "cpu"
        for mode in BS.MODES:
            for b in ("1", "2"):
                e = blob["modes"][mode][b]
                for key in ("tokens_per_s", "seconds_mean", "seconds_std", "union_size"):
                    assert key in e, f"{mode}/{b} missing {key}"
                assert e["tokens_per_s"] > 0, f"{mode}/{b} tokens_per_s not > 0: {e}"
                assert e["seconds_mean"] > 0 and e["seconds_std"] >= 0
        for b in ("1", "2"):
            assert "n_groups" in blob["modes"]["merge_per_group"][b]
            assert blob["modes"]["single_expert"][b]["union_size"] == 1
    print("[ok] JSON structure complete, tokens_per_s > 0 for all modes/batch sizes")

    # ── 2. heterogeneity: batch of 2 has 2 DISTINCT sets → 2 singleton groups ─────
    comp = res["batches"]["2"]
    assert comp["n_distinct_sets"] == 2, f"batch not heterogeneous: {comp}"
    assert len({tuple(s) for s in comp["expert_sets"]}) == 2
    assert res["modes"]["merge_per_group"]["2"]["n_groups"] == 2
    assert res["modes"]["merge_per_group"]["1"]["n_groups"] == 1
    assert comp["union_size"] == res["modes"]["ramole_batched"]["2"]["union_size"]
    print("[ok] heterogeneous batch: 2 distinct expert sets, composition stats consistent")

    # ── 3. identity: mode (b) grouped outputs == per-sample LegoNet (exact 1/k) ───
    records = rc.load_records(rc.source_paths(cfg).records_path)
    by_id = {r["id"]: r for r in records}
    sets = BS.assignment_sets(cfg, records)
    lego = LegoNetModel.from_config(_source_lego_cfg(cfg), device_map="cpu")
    for rid, s in zip(comp["record_ids"], comp["expert_sets"]):
        ref = _gen_one(lego, by_id[rid], s, gen_tokens=4)
        got = res["outputs"]["merge_per_group"]["2"][rid]
        assert got == ref, f"{rid}: grouped {got!r} != per-sample {ref!r}"
    print("[ok] merge_per_group ≡ per-sample LegoNet on the benchmark batch")

    # Same identity on a REAL multi-member group: two records sharing a set + one other,
    # so bench_merge_per_group runs an actual sub-batch (padded) for one group.
    by_set = {}
    for r in records:
        by_set.setdefault(sets[r["id"]], []).append(r)
    dup_set = next(s for s, rs in by_set.items() if len(rs) >= 2)
    other_set = next(s for s in by_set if s != dup_set)
    recs3 = by_set[dup_set][:2] + [by_set[other_set][0]]
    cap = {}
    entry = BS.bench_merge_per_group(lego, recs3, sets, gen_tokens=4, iters=1, cuda=False,
                                     capture=cap)
    assert entry["n_groups"] == 2 and entry["tokens_per_s"] > 0
    for r in recs3:
        ref = _gen_one(lego, r, sets[r["id"]], gen_tokens=4)
        assert cap[r["id"]] == ref, f"{r['id']}: grouped {cap[r['id']]!r} != per-sample {ref!r}"
    print("[ok] multi-member group (sub-batch of 2 + 1) ≡ per-sample LegoNet")

    print("\nALL BENCHMARK SERVING TESTS PASSED")


if __name__ == "__main__":
    main()
