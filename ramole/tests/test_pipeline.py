"""End-to-end CPU integration smoke (CLAUDE.md §4 — run before any SLURM job):
fixture → retriever (FT + index) → router training → eval (all methods) + batched-routing check.

    ${TOFU_PYTHON:-python3} tests/test_pipeline.py
"""
import math
import os
import sys
import tempfile

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, THIS)

from _fixture import build_corpus_and_routing, build_source_run  # noqa: E402
import ramole_common as rc       # noqa: E402
import retriever as RET          # noqa: E402
import train_router as TR        # noqa: E402
import eval_ramole as EV         # noqa: E402
from ramole_model import RamoleModel  # noqa: E402


def _finite(agg, keys=("em", "es", "verbmem", "perplexity")):
    return all(isinstance(agg[k], float) and not math.isnan(agg[k]) for k in keys)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tmp = tempfile.mkdtemp(prefix="ramole_pipe_")
    cfg = build_source_run(tmp, n=4, hidden=64, layers=2, heads=4, kv_heads=2,
                           rank=4, alpha=8, with_tokenizer=True)
    cfg["name"] = "ramole_pipe"
    cfg["router_train_split"] = "corpus"
    build_corpus_and_routing(cfg, per_cluster=10, device="cpu")
    print("[ok] fixture built (n=4, k=2)")

    # Stage 1
    RET.train_retriever(cfg, device="cpu")
    RET.build_index(cfg, device="cpu")
    assert os.path.isfile(rc.Paths(cfg).lora_index_path)
    print("[ok] retriever fine-tuned + index built")

    # Stage 2
    rpath = TR.train_router(cfg, device="cpu")
    assert os.path.isfile(rpath) and os.path.isfile(rc.Paths(cfg).router_meta)
    print("[ok] router trained + saved")

    # Stage 4 — every method runs and yields finite metrics
    sp = rc.source_paths(cfg)
    ids = [r["id"] for r in rc.load_records(sp.records_path)[:6]]
    for method, route, cond in [("router", "keys", "iid"), ("router", "retriever", "iid"),
                                ("router", "retriever", "ood"), ("mean", "keys", "iid"),
                                ("perfect", "keys", "iid")]:
        rows, agg = EV.evaluate(cfg, ids, method, route, cond, gen_cap=8, device="cpu")
        assert _finite(agg), f"{method}/{route}/{cond} produced non-finite metrics: {agg}"
        assert agg["num_records"] == len(ids)
        print(f"[ok] eval {method:7s}/{route:9s}/{cond}: "
              f"em={agg['em']:.3f} verbmem={agg['verbmem']:.3f} ppl={agg['perplexity']:.2f} "
              f"k_eff={agg['k_eff']}")

    # Batched heterogeneous routing == per-sample set_active (the Stage-3 mask path, end-to-end)
    rm = RamoleModel.from_config(cfg, device="cpu", load_router=True)
    torch.manual_seed(0)
    row = torch.randint(0, 100, (1, 6))
    ids2 = torch.cat([row, row], 0)               # identical rows, different routing
    setA, setB = [0, 1], [2, 3]
    rm.set_routing([setA, setB])
    with torch.no_grad():
        out_batched = rm.model(input_ids=ids2).logits
    rm.set_active(setA)
    with torch.no_grad():
        outA = rm.model(input_ids=row).logits
    rm.set_active(setB)
    with torch.no_grad():
        outB = rm.model(input_ids=row).logits
    dA = (out_batched[0:1] - outA).abs().max().item()
    dB = (out_batched[1:2] - outB).abs().max().item()
    assert dA < 2e-5 and dB < 2e-5, f"batched routing != per-sample: {dA}, {dB}"
    print(f"[ok] batched per-sample routing ≡ set_active (Δ={dA:.2e}, {dB:.2e})")

    print("\nALL PIPELINE TESTS PASSED")


if __name__ == "__main__":
    main()
