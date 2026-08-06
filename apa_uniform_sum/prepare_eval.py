"""Precompute the retain90 forget-quality reference + the eval task manifest for parallel eval.

The reference (`results/{sub}/retain_tr_scores.npy`) is the per-sample truth-ratio distribution of
the **retain90 oracle adapter** (authors 0-179, never trained on forget10) on the forget set.
`eval_tofu.py` compares each label's forget truth-ratio distribution against it via a KS test
(`forget_quality`), matching open-unlearning `privacy.py:ks_test`.

Assumes the forget shard is forget10 (authors 180-199), the repo-wide invariant, so retain90 =
authors 0-179. Train the oracle first with `train_lora_shard.py --retain90`.
"""
import argparse
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_tofu import (
    EXTENDED_TRUTH_MAX,
    SMOKE_TRUTH_MAX,
    get_truth_ratio_scores,
    load_tofu_data,
)
from merge_lora import default_eval_labels, smoke_eval_labels
from shard_utils import get_author_shard
import tofu_env as _tofu_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--forget_shard_id", type=int, default=None)
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME") or _tofu_env.hf_home())
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke manifest + retain90 reference capped to SMOKE_TRUTH_MAX -> results/smoke/",
    )
    p.add_argument(
        "--extended",
        action="store_true",
        help="Extended manifest + retain90 reference capped to EXTENDED_TRUTH_MAX -> results/extended/",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke and args.extended:
        raise SystemExit("Use only one of --smoke or --extended")
    forget_id = args.forget_shard_id if args.forget_shard_id is not None else args.k - 1
    # Cap must match eval_tofu's truth_max_rows so the eval/reference forget rows line up.
    if args.smoke:
        results_sub, truth_max = "smoke", SMOKE_TRUTH_MAX
    elif args.extended:
        results_sub, truth_max = "extended", EXTENDED_TRUTH_MAX
    else:
        results_sub, truth_max = "", None
    results_dir = os.path.join(args.output_dir, "results", results_sub)
    os.makedirs(results_dir, exist_ok=True)

    data = load_tofu_data(args.hf_home)
    shards = {i: get_author_shard(args.k, i) for i in range(args.k)}
    forget_authors = shards[forget_id]
    forget_indices = [r for a in forget_authors for r in range(a * 20, a * 20 + 20)]
    forget_ds = data["full"].select(forget_indices)
    forget_qs = set(forget_ds["question"])
    # Same construction as evaluate_model so the reference covers identical forget rows/order.
    forget_pert_subset = data["full_pert"].filter(lambda r: r["question"] in forget_qs)

    # Forget-quality oracle = the retain90 LoRA (authors 0-179; never trained on the forget shard).
    retain90_dir = os.path.join(args.output_dir, "retain90")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if os.path.isdir(retain90_dir):
        model = PeftModel.from_pretrained(base, retain90_dir, adapter_name="retain90")
        retain_tr = get_truth_ratio_scores(
            model, tokenizer, forget_pert_subset, correct_key="paraphrased_answer", max_rows=truth_max
        )
        np.save(os.path.join(results_dir, "retain_tr_scores.npy"), retain_tr)
        print(
            f"Saved retain_tr_scores ({len(retain_tr)} samples) "
            f"mean={float(np.mean(retain_tr)):.3f} from {retain90_dir}"
        )
    else:
        print(
            f"[prepare_eval] WARNING: no retain90 adapter at {retain90_dir} -> skipping "
            f"retain_tr_scores.npy; forget_quality will be NaN. Train it with:\n  "
            f"train_lora_shard.py --retain90 --output_dir {args.output_dir} "
            f"--model_name {args.model_name} --k {args.k}"
        )
    del base
    torch.cuda.empty_cache()

    if args.smoke or args.extended:
        labels = smoke_eval_labels(args.k, forget_id)
    else:
        labels = default_eval_labels(args.k, forget_id)
    if args.smoke:
        manifest_name = "eval_manifest_smoke.txt"
    elif args.extended:
        manifest_name = "eval_manifest_extended.txt"
    else:
        manifest_name = "eval_manifest.txt"
    manifest_path = os.path.join(results_dir, manifest_name)
    with open(manifest_path, "w") as f:
        f.write("\n".join(labels) + "\n")
    print(f"Wrote {len(labels)} tasks -> {manifest_path}")


if __name__ == "__main__":
    main()
