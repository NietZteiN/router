"""CPU tests for the overnight-campaign additions:
  - adapter_dir_fn override (serve a post-deletion pool)
  - legonet post_unlearn_adapter_dir_fn integration (the exact production path)
  - retriever_run shared-retriever key (ablation arms reuse one encoder FT)

    ${TOFU_PYTHON:-python3} tests/test_unlearn_shared.py
"""
import json
import os
import sys
import tempfile

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, THIS)

from _fixture import build_corpus_and_routing, build_source_run  # noqa: E402
import ramole_common as rc            # noqa: E402
import retriever as RET               # noqa: E402
from ramole_model import RamoleModel  # noqa: E402
from eval_ramole import _source_lego_cfg  # noqa: E402


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tmp = tempfile.mkdtemp(prefix="ramole_unl_")
    cfg = build_source_run(tmp, n=4, hidden=64, layers=2, heads=4, kv_heads=2,
                           rank=4, alpha=8, with_tokenizer=True)
    cfg["name"] = "arm_a"
    build_corpus_and_routing(cfg, per_cluster=8, device="cpu")
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, 5))

    # ── adapter_dir_fn override takes effect ───────────────────────────────────
    sp = rc.source_paths(cfg)
    swap = {0: sp.adapter_dir(1)}                       # serve expert 0 from a1's dir
    fn = lambda j: swap.get(j, sp.adapter_dir(j))
    rm0 = RamoleModel.from_config(cfg, device="cpu", load_router=False)
    rm0.set_active([0])
    with torch.no_grad():
        out_orig = rm0.model(input_ids=ids).logits
    rm1 = RamoleModel.from_config(cfg, device="cpu", load_router=False, adapter_dir_fn=fn)
    rm1.set_active([0])
    with torch.no_grad():
        out_swap = rm1.model(input_ids=ids).logits
    # swapped expert 0 == default expert 1 (single-active ≡ that expert)
    rm0.set_active([1])
    with torch.no_grad():
        out_a1 = rm0.model(input_ids=ids).logits
    assert (out_swap - out_a1).abs().max().item() < 2e-5, "adapter_dir_fn override did not take effect"
    assert (out_swap - out_orig).abs().max().item() > 1e-3, "override produced identical output (no-op?)"
    print("[ok] adapter_dir_fn override serves the substituted expert")

    # ── legonet post_unlearn_adapter_dir_fn integration ────────────────────────
    from unlearn import post_unlearn_adapter_dir_fn   # legonet
    manifest = {"tag": "dX", "forget_ids": ["rec_00000"], "affected_adapters": [0],
                "retrained_dirs": {"0": sp.adapter_dir(1)}, "untouched_adapters": [1, 2, 3]}
    pufn = post_unlearn_adapter_dir_fn(_source_lego_cfg(cfg), manifest)
    rm2 = RamoleModel.from_config(cfg, device="cpu", load_router=False, adapter_dir_fn=pufn)
    rm2.set_active([0])
    with torch.no_grad():
        out_puf = rm2.model(input_ids=ids).logits
    assert (out_puf - out_a1).abs().max().item() < 2e-5, "post_unlearn_adapter_dir_fn mismatch"
    print("[ok] legonet post_unlearn_adapter_dir_fn serves the post-deletion pool")

    # ── retriever_run shared retriever ──────────────────────────────────────────
    RET.build_index(cfg, device="cpu")                 # arm A builds the index (off-the-shelf enc)
    assert os.path.isfile(rc.Paths(cfg).lora_index_path)
    cfg_b = {**cfg, "name": "arm_b", "retriever_run": "arm_a"}   # arm B has NO retriever of its own
    rc.Paths(cfg_b).ensure()
    assert not os.path.isfile(rc.Paths(cfg_b).lora_index_path), "arm_b should have no own index"
    ret_b = RET.LoraRetriever.load(cfg_b, device="cpu")          # must load arm A's index
    top = ret_b.retrieve(["dogs and cats are pets"], k=2)
    assert top.shape == (1, 2)
    # identical to loading arm A directly
    ret_a = RET.LoraRetriever.load(cfg, device="cpu")
    top_a = ret_a.retrieve(["dogs and cats are pets"], k=2)
    assert (top == top_a).all(), "shared retriever differs from source"
    print("[ok] retriever_run shares arm A's index (no own retriever needed)")

    print("\nALL UNLEARN/SHARED TESTS PASSED")


if __name__ == "__main__":
    main()
