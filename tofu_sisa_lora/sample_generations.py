"""Sample greedy Q/A generations for report examples (one label per invocation)."""
import argparse
import json
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_tofu import get_author_shard, load_tofu_data, sample_answers
from merge_lora import activate_label

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


DEFAULT_LABELS = [
    "base_model",
    "shard_3_only",
    "merged_dare_ties",
    "merged_ties",
    "remerge_dare_ties",
    "merged_linear",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--forget_shard_id", type=int, default=None)
    p.add_argument("--n_per_split", type=int, default=3)
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    p.add_argument("--out_dir", default=None, help="default: <output_dir>/results/extended/generations")
    return p.parse_args()


def pick_indices(n_total, n_pick, seed=0):
    n_pick = min(n_pick, n_total)
    if n_pick >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=n_pick, replace=False).tolist())


def load_model_for_label(model_name, output_dir, k, label):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if label == "base_model":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        return model, tokenizer, "base"

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    shard_0 = os.path.join(output_dir, "shard_0")
    model = PeftModel.from_pretrained(base, shard_0, adapter_name="shard_0")
    for i in range(1, k):
        model.load_adapter(os.path.join(output_dir, f"shard_{i}"), adapter_name=f"shard_{i}")
    forget_id = k - 1
    adapter_name = activate_label(model, k, forget_id, label)
    model.set_adapter(adapter_name)
    return model, tokenizer, adapter_name


def main():
    args = parse_args()
    forget_id = args.forget_shard_id if args.forget_shard_id is not None else args.k - 1
    out_dir = args.out_dir or os.path.join(args.output_dir, "results", "extended", "generations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}.json")

    data = load_tofu_data(args.hf_home)
    shards = {i: get_author_shard(args.k, i) for i in range(args.k)}
    forget_authors = shards[forget_id]
    forget_indices = [r for a in forget_authors for r in range(a * 20, a * 20 + 20)]
    forget_ds = data["full"].select(forget_indices)

    retain_indices = [i for i in range(len(data["full"])) if i not in set(forget_indices)]
    retain_sample = pick_indices(len(retain_indices), args.n_per_split, seed=1)
    retain_ds = data["full"].select([retain_indices[i] for i in retain_sample])

    model, tokenizer, adapter_name = load_model_for_label(
        args.model_name, args.output_dir, args.k, args.label
    )

    forget_idx = pick_indices(len(forget_ds), args.n_per_split, seed=0)
    real_idx = pick_indices(len(data["real_authors"]), args.n_per_split, seed=2)
    world_idx = pick_indices(len(data["world_facts"]), args.n_per_split, seed=3)

    payload = {
        "label": args.label,
        "adapter": adapter_name,
        "model_name": args.model_name,
        "n_per_split": args.n_per_split,
        "splits": {
            "forget": sample_answers(
                model, tokenizer,
                [forget_ds[i]["question"] for i in forget_idx],
                [forget_ds[i]["answer"] for i in forget_idx],
            ),
            "retain": sample_answers(
                model, tokenizer,
                retain_ds["question"], retain_ds["answer"],
            ),
            "real_authors": sample_answers(
                model, tokenizer,
                [data["real_authors"][i]["question"] for i in real_idx],
                [data["real_authors"][i]["answer"] for i in real_idx],
            ),
            "world_facts": sample_answers(
                model, tokenizer,
                [data["world_facts"][i]["question"] for i in world_idx],
                [data["world_facts"][i]["answer"] for i in world_idx],
            ),
        },
    }

    del model
    torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
