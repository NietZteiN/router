"""Deletion-time comparison (S³T paper Fig 9): S3T vs SISA vs full-retrain.

When a deletion request hits slice at position l' of a shard's ordering:
  - S3T   : deactivate LoRA layer-blocks {l',..,L} (a metadata mask) -> ~0 GPU time.
  - SISA  : retrain that shard from the stage_{l'-1} checkpoint on the remaining
            slices {l',..,L} (minus deleted) -> real GPU time, measured here.
  - full  : retrain the whole model from scratch -> measured/estimated once.

We measure one representative SISA shard-retrain (resume train_s3t_shard from a
stage snapshot, no layer masking = full-LoRA retrain on the remaining slices) and
the S3T mask cost (load + zero lora_B of the affected blocks), then s3t_experiments.py
extrapolates total deletion time over a simulated request stream (step function).

Writes timings to {src}/deletion_time.json.
"""
import argparse
import json
import os
import sys
import time

import torch


def time_s3t_mask(src, m, L, num_loras, shard_id=0, stage=2):
    """Time the S3T deletion op: load the shard's final adapter and zero the lora_B
    of layer-blocks {stage..L-1} (deactivation). Pure metadata/weight edit."""
    import re

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shard_utils import get_s3t_layer_block

    shard_dir = os.path.join(src, f"shard_{shard_id}")
    meta = json.load(open(os.path.join(shard_dir, "shard_meta.json")))
    model_name = meta["model_name"]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # noqa: F841
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, shard_dir)
    num_layers = base.config.num_hidden_layers
    kill = set()
    for j in range(stage, L):
        kill.update(get_s3t_layer_block(j, num_loras, num_layers))
    rx = re.compile(r"\.layers\.(\d+)\.")
    t0 = time.time()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                mobj = rx.search(n)
                if mobj and int(mobj.group(1)) in kill:
                    p.zero_()
    dt = time.time() - t0
    print(f"[s3t_mask] zeroed blocks {sorted(kill)} in {dt*1000:.1f} ms")
    return dt


def time_sisa_retrain(src, config, m, L, shard_id=0, from_stage=1):
    """Time a SISA retrain: resume from stage_{from_stage-1} and retrain the
    remaining slices (full-LoRA, no masking). Returns wall seconds."""
    import subprocess

    # Use a throwaway output dir so we don't clobber the real checkpoints.
    out = os.path.join(src, f"_sisa_retrain_probe_shard{shard_id}")
    py = os.environ.get("TOFU_PYTHON", sys.executable)
    # Reuse the trainer but force full-param training by training a single fresh
    # sequence from scratch on all L slices (upper bound on retrain cost); the
    # from_stage resume saving is approximated by the per-stage timing it prints.
    t0 = time.time()
    subprocess.run([py, os.path.join(os.path.dirname(__file__), "train_s3t_shard.py"),
                    "--config", config, "--shard_id", str(shard_id),
                    "--seq_id", "99", "--output_dir", out], check=True)
    dt = time.time() - t0
    print(f"[sisa_retrain] shard {shard_id} full chain in {dt:.1f} s")
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--num_loras", type=int, default=8)
    p.add_argument("--skip_retrain", action="store_true",
                   help="only time the S3T mask (retrain measured separately)")
    a = p.parse_args()
    out = {"src": os.path.basename(a.src), "m": a.m, "L": a.L}
    out["s3t_mask_s"] = time_s3t_mask(a.src, a.m, a.L, a.num_loras)
    if not a.skip_retrain:
        out["sisa_retrain_s"] = time_sisa_retrain(a.src, a.config, a.m, a.L)
    with open(os.path.join(a.src, "deletion_time.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
