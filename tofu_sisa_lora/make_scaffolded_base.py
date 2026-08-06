"""Bake a scaffold LoRA into the base weights -> a full model on disk (routing+scaffold method).

Serving base+scaffold+routed-expert = load the *scaffolded* base, then load expert LoRAs on top
(LoRA deltas add to whatever base weights are present). So we merge the scaffold once here and point
eval_tofu --model_name at the result; the untouched shard/legonet adapters route on top unchanged.

  python make_scaffolded_base.py --base_model meta-llama/Llama-3.2-1B-Instruct \
      --scaffold checkpoints/Llama-3.2-1B-Instruct_scaffold_alpaca2k --out ${TOFU_CKPT_ROOT}/.../scaffolded_base
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import sys

# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--scaffold", required=True, help="scaffold adapter dir")
    ap.add_argument("--out", required=True, help="output dir for the merged full model")
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    args = ap.parse_args()
    os.environ["HF_HOME"] = args.hf_home
    if os.path.exists(os.path.join(args.out, "config.json")):
        print(f"scaffolded base exists, skipping -> {args.out}")
        return
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, args.scaffold).merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved scaffolded base (base + {os.path.basename(args.scaffold)}) -> {args.out}")


if __name__ == "__main__":
    main()
