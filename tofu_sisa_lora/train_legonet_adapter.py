"""Train one LegoNet adapter on its assigned TOFU authors (SLURM-callable).

Adapter j trains on the union of records of the authors routed to key j (from the
cached assignment), minus any `--exclude_authors` (used for the unlearn/oracle
retrain). Per-adapter seed = base_seed + j and deterministic kernels make the
retrain reproducible (the exactness condition). Authors are sorted so the training
order is identical for the original train and any later retrain.

Reuses `train_lora_shard.load_shard_dataset` / `format_prompt` (the author->row
mapping `a*20..a*20+19` and the "Question: q\\nAnswer: a" text). The LoRA recipe
comes entirely from the config (rank/alpha/dropout/target_modules/use_rslora,
epochs/lr/bs/grad_accum/max_length) — distinct from the frozen SISA recipe.

    python train_legonet_adapter.py --config configs/legonet_tofu.json --adapter 5
    python train_legonet_adapter.py --config ... --adapter 5 --exclude_authors 180 181 --out_dir DIR
"""
import argparse
import json
import os
import random

import numpy as np

import legonet_tofu as lt

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


def set_determinism(seed: int):
    """Pin RNGs + request deterministic kernels (warn_only: ops without a
    deterministic impl fall back rather than crash)."""
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def adapter_authors(cfg, j, exclude_authors):
    with open(lt.assignment_path(cfg)) as f:
        assignment = json.load(f)
    excl = set(int(a) for a in exclude_authors)
    return sorted(a for a in lt.adapter_author_ids(assignment, j) if a not in excl)


def train_one(cfg, j, exclude_authors=None, out_dir=None, force=False):
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    from train_lora_shard import load_shard_dataset, format_prompt

    exclude_authors = exclude_authors or []
    out_dir = out_dir or lt.adapter_dir(cfg, j)
    seed = cfg["base_seed"] + j

    if os.path.exists(os.path.join(out_dir, "adapter_config.json")) and not force:
        print(f"a{j}: adapter exists, skipping -> {out_dir}")
        return out_dir

    authors = adapter_authors(cfg, j, exclude_authors)
    os.makedirs(out_dir, exist_ok=True)
    set_determinism(seed)
    os.environ["HF_HOME"] = cfg["hf_home"]

    use_cuda = torch.cuda.is_available()
    optim = "paged_adamw_32bit" if use_cuda else "adamw_torch"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    lcfg, tcfg = cfg["lora"], cfg["train"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=dtype,
        device_map="auto" if use_cuda else None, trust_remote_code=True,
    )
    if not use_cuda:
        model = model.to("cpu")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=lcfg["rank"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM",
        use_rslora=lcfg.get("use_rslora", False),
    ))

    meta = {
        "adapter": j, "seed": seed, "author_ids": authors,
        "excluded_authors": sorted(int(a) for a in exclude_authors),
        "num_authors": len(authors), "base_model": cfg["base_model"], "lora": lcfg,
    }

    # Whole adapter forgotten -> save the zero-delta (lora_B=0) no-op adapter (O(1) "drop").
    if len(authors) == 0:
        meta["disabled"] = True
        meta["num_samples"] = 0
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        lt_write_json(os.path.join(out_dir, "meta.json"), meta)
        print(f"a{j}: 0 authors -> saved zero-delta (disabled) adapter -> {out_dir}")
        return out_dir

    ds = load_shard_dataset(authors, cfg["hf_home"]).map(
        format_prompt, remove_columns=["question", "answer"])
    meta["num_samples"] = len(ds)

    train_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        optim=optim,
        learning_rate=tcfg["lr"],
        weight_decay=0.001,
        bf16=use_cuda,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model, train_dataset=ds, args=train_args,
        dataset_text_field="text", max_seq_length=tcfg["max_length"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    lt_write_json(os.path.join(out_dir, "meta.json"), meta)
    print(f"a{j}: trained on {len(authors)} authors / {len(ds)} Q&As (seed {seed}) -> {out_dir}")
    return out_dir


def lt_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", type=int, required=True)
    ap.add_argument("--exclude_authors", type=int, nargs="*", default=[])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = lt.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    train_one(cfg, args.adapter, exclude_authors=args.exclude_authors,
              out_dir=args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
