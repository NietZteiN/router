"""Train one per-author personal-LoRA proxy for SEA-on-TOFU (SFT).

The SEA paper trains the personal LoRA via DPO (it cares about preferences); TOFU is
knowledge injection (question -> exact answer), so we use SFT on the author's 20 QA pairs
(SEA_on_TOFU.md §3.2). Base is frozen 4-bit; only the rank-r adapter gets gradients; the
only artifact written is proxies/.../author_NNN/{personal_lora,meta.json}. Deleting that dir
removes every byte of the author's influence — the shared base is bit-identical.

Two entry points:
  - train_one_author(base, ...) -> returns the cleaned base (unload()'d) for looped/pilot use.
  - CLI main(): trains exactly one author with its own base load (one SLURM task per author).

QLoRA SFT setup mirrors the proven sea/train_expert.py (4-bit NF4 -> get_peft_model ->
SFTTrainer(bf16=True)); no prepare_model_for_kbit_training needed in this env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

from inference import load_base
from load_tofu import author_qa, load_tofu_data
from proxy_paths import author_dir, meta_path, personal_lora_dir, proxy_exists

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

PROMPT = "Question: {q}\nAnswer: {a}"  # MUST match eval_tofu._build_qa_prompt (eval comparability)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def build_text_dataset(qa_pairs, tokenizer):
    eos = tokenizer.eos_token or ""
    rows = [{"text": PROMPT.format(q=x["question"], a=x["answer"]) + eos} for x in qa_pairs]
    return Dataset.from_list(rows)


def _load_cfg(cfg_path):
    with open(cfg_path) as f:
        return json.load(f)


def train_one_author(base, tokenizer, author_id, qa_pairs, cfg, rank=None,
                     proxy_root=None, seed=None):
    """Train + save one author's proxy. Returns the cleaned base for the next author.

    cfg is the parsed configs/sea_tofu_llama2.json. rank/seed override the config defaults
    (used by the rank sweep). proxy_root lets callers redirect the output tree.
    """
    pl = cfg["personal_lora"]
    tr = cfg["train"]
    rank = rank if rank is not None else pl["rank"]
    seed = seed if seed is not None else tr["seed"]
    model_name = cfg["model_name"]

    kw = {} if proxy_root is None else {"proxy_root": proxy_root}
    if proxy_exists(model_name, author_id, rank, **kw):
        print(f"[skip] author_{author_id:03d} r{rank} exists", flush=True)
        return base

    torch.manual_seed(seed)
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=pl["alpha_mult"] * rank,
        lora_dropout=pl["lora_dropout"],
        target_modules=pl["target_modules"],
        bias=pl["bias"],
        task_type="CAUSAL_LM",
        use_rslora=pl["use_rslora"],
    )
    peft_model = get_peft_model(base, lora_cfg)
    peft_model.config.use_cache = False

    ds = build_text_dataset(qa_pairs, tokenizer)
    train_args = TrainingArguments(
        output_dir=f"/tmp/sea_tofu_train_{author_id:03d}_r{rank}",
        num_train_epochs=tr["epochs"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["grad_accum"],
        learning_rate=tr["lr"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"],
        max_grad_norm=tr["max_grad_norm"],
        optim=tr["optim"],
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=ds,
        args=train_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=tr["max_length"],
        packing=False,
    )
    trainer.train()

    lora_dir = personal_lora_dir(model_name, author_id, rank, **kw)
    os.makedirs(lora_dir, exist_ok=True)
    peft_model.save_pretrained(lora_dir)  # adapter only — base untouched by construction

    train_hash = hashlib.sha256(
        json.dumps([{"q": x["question"], "a": x["answer"]} for x in qa_pairs],
                   sort_keys=True).encode()
    ).hexdigest()[:16]
    with open(meta_path(model_name, author_id, rank, **kw), "w") as f:
        json.dump({
            "author_id": author_id,
            "rank": rank,
            "alpha": pl["alpha_mult"] * rank,
            "n_qa": len(qa_pairs),
            "model_name": model_name,
            "train_hash": train_hash,
            "seed": seed,
            "git_commit": _git_commit(),
            "created": time.time(),
        }, f, indent=2)

    # CRITICAL: detach the adapter so the next author starts from the clean base.
    base = peft_model.unload()
    print(f"[done] author_{author_id:03d} r{rank} -> {lora_dir}", flush=True)
    return base


def parse_args():
    p = argparse.ArgumentParser()
    # Single author OR a contiguous block (block amortizes the 7B 4-bit base load over a
    # whole SLURM task; base is loaded once and reused via unload() between authors).
    p.add_argument("--author_id", type=int, default=None, help="Train one author.")
    p.add_argument("--author_start", type=int, default=None, help="Block start (inclusive).")
    p.add_argument("--author_count", type=int, default=None, help="Block size from author_start.")
    p.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "configs", "sea_tofu_llama2.json"))
    p.add_argument("--rank", type=int, default=None, help="Override config rank (for the sweep).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--proxy_root", default=None,
                   help="Override proxy output root (seed-variance: isolate non-default seeds).")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    return p.parse_args()


def _author_list(args):
    if args.author_id is not None:
        return [args.author_id]
    if args.author_start is not None and args.author_count is not None:
        return list(range(args.author_start, min(args.author_start + args.author_count, 200)))
    raise SystemExit("Provide --author_id OR (--author_start and --author_count).")


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    cfg = _load_cfg(args.config)
    authors = _author_list(args)
    base, tok = load_base(cfg["model_name"], hf_home=args.hf_home)
    splits = load_tofu_data(args.hf_home)
    for aid in authors:
        qa = author_qa(splits["full"], aid)
        base = train_one_author(base, tok, aid, qa, cfg, rank=args.rank, seed=args.seed,
                                proxy_root=args.proxy_root)


if __name__ == "__main__":
    main()
