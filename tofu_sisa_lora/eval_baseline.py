"""Extended TOFU metrics for the base model only (no LoRA / merge)."""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_progress import ProgressLogger
from eval_tofu import (
    EXTENDED_RETAIN_MAX,
    EXTENDED_ROUGE_MAX,
    EXTENDED_TRUTH_MAX,
    SMOKE_RETAIN_MAX,
    SMOKE_ROUGE_MAX,
    SMOKE_TRUTH_MAX,
    evaluate_model,
    load_tofu_data,
)
from shard_utils import get_author_shard

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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--forget_shard_id", type=int, default=None)
    p.add_argument("--eval_shard_id", type=int, default=None,
                   help="Score the forget_* metrics on THIS shard's OWN authors instead of "
                        "--forget_shard_id's (same semantics as eval_tofu.py). Combined with "
                        "--retain_author_ids the retain exclusion keys on the GLOBAL forget "
                        "shard, so probe-author floor rows (--forget_shard_id 199 "
                        "--eval_shard_id p --retain_author_ids p) have a non-empty retain pool.")
    p.add_argument("--out", required=True)
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    p.add_argument("--extended", action="store_true", help="Use extended metric caps")
    p.add_argument("--smoke", action="store_true", help="Use smoke metric caps")
    p.add_argument("--retain_author_ids", default=None,
                   help="Comma-separated author ids: restrict the retain_* metrics to these "
                        "authors' rows (subset-conditioned floor; same semantics as eval_tofu).")
    return p.parse_args()


def main():
    args = parse_args()
    forget_id = args.forget_shard_id if args.forget_shard_id is not None else args.k - 1
    if args.eval_shard_id is not None and not (0 <= args.eval_shard_id < args.k):
        raise SystemExit(f"--eval_shard_id {args.eval_shard_id} out of range [0,{args.k})")
    retain_author_ids = None
    if args.retain_author_ids:
        retain_author_ids = sorted({int(x) for x in args.retain_author_ids.split(",") if x.strip()})
    label = "base_model"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    progress_path = args.out.replace(".json", ".progress.json")

    prog = ProgressLogger(progress_path, label)
    prog.step("load_data", "TOFU datasets")
    data = load_tofu_data(args.hf_home)
    shards = {i: get_author_shard(args.k, i) for i in range(args.k)}

    if args.smoke and args.extended:
        raise SystemExit("Use only one of --smoke or --extended")
    if args.smoke:
        results_sub = "smoke"
    elif args.extended:
        results_sub = "extended"
    else:
        results_sub = ""
    results_dir = os.path.join(args.output_dir, "results", results_sub)
    retain_tr_path = os.path.join(results_dir, "retain_tr_scores.npy")
    retain_ref_tr_scores = np.load(retain_tr_path) if os.path.exists(retain_tr_path) else None

    prog.step("load_model", f"base {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    smoke = args.smoke
    extended = args.extended
    if smoke:
        rouge_n, retain_n, truth_n = SMOKE_ROUGE_MAX, SMOKE_RETAIN_MAX, SMOKE_TRUTH_MAX
    elif extended:
        rouge_n, retain_n, truth_n = EXTENDED_ROUGE_MAX, EXTENDED_RETAIN_MAX, EXTENDED_TRUTH_MAX
    else:
        rouge_n, retain_n, truth_n = None, 500, None

    prog.step("evaluate", "base model metrics")
    row = evaluate_model(
        model, tokenizer, label,
        forget_shard_id=forget_id,
        full_ds=data["full"],
        shards=shards,
        forget10_pert=data["forget10_pert"],
        real_authors=data["real_authors"],
        world_facts=data["world_facts"],
        retain_ref_tr_scores=retain_ref_tr_scores,
        rouge_max_samples=rouge_n,
        prog=prog,
        smoke=smoke,
        extended=extended,
        retain_max_samples=retain_n,
        truth_max_rows=truth_n,
        full_pert=data["full_pert"],
        real_authors_pert=data["real_authors_pert"],
        world_facts_pert=data["world_facts_pert"],
        eval_shard_id=args.eval_shard_id,
        retain_author_ids=retain_author_ids,
    )
    row["model_name"] = args.model_name
    row["adapter"] = "base"
    row["eval_shard_id"] = args.eval_shard_id
    row["retain_author_ids"] = retain_author_ids

    del model
    torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)
    prog.done(row)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
