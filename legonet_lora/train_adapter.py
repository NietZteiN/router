"""Train one LoRA adapter on its assigned records (SLURM-callable).

Independence is load-bearing for exactness: each adapter trains alone, with its
own per-adapter seed (BASE_SEED + j), no shared optimizer state, no cross-adapter
batching, deterministic kernels (Condition B). Member records come from the cached
routing assignment; `--exclude_record_id` drops records for an unlearn/oracle
retrain. The training-set order is sorted by record_id so it is identical for the
original train and any later retrain.

    python train_adapter.py --config configs/legonet_7b.json --adapter 5
    python train_adapter.py --config ... --adapter 5 --exclude_record_id rec_000123 --out_dir /tmp/oracle_a5
"""
import argparse
import json
import os

from legonet_common import (
    Paths, config_hash, load_config, load_records, set_determinism, train_text, write_json,
)
from routing import adapter_member_ids

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


def _adapter_records(cfg: dict, j: int, exclude_ids: set[str]) -> list[dict]:
    paths = Paths(cfg)
    with open(paths.assignment_path) as f:
        assignment = json.load(f)
    member_ids = [i for i in adapter_member_ids(assignment, j) if i not in exclude_ids]
    by_id = {r["id"]: r for r in load_records(paths.records_path)}
    # deterministic order independent of dict / assignment ordering
    return [by_id[i] for i in sorted(member_ids)]


def train_one(cfg: dict, j: int, exclude_ids=None, out_dir: str | None = None,
              force: bool = False) -> str:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    from datasets import Dataset

    exclude_ids = set(exclude_ids or [])
    paths = Paths(cfg)
    out_dir = out_dir or paths.adapter_dir(j)
    seed = cfg["base_seed"] + j

    if os.path.exists(os.path.join(out_dir, "adapter_config.json")) and not force:
        print(f"a{j}: adapter exists, skipping -> {out_dir}")
        return out_dir

    records = _adapter_records(cfg, j, exclude_ids)
    os.makedirs(out_dir, exist_ok=True)
    set_determinism(seed)

    use_cuda = torch.cuda.is_available()
    # CPU fallback (for unit tests): adamw_torch + fp32 (paged optimizer/bf16 need CUDA)
    optim = "paged_adamw_32bit" if use_cuda else "adamw_torch"
    bf16 = use_cuda
    dtype = torch.bfloat16 if use_cuda else torch.float32

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
    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["rank"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))

    meta = {
        "adapter": j, "seed": seed, "config_hash": config_hash(cfg),
        "num_records": len(records), "excluded": sorted(exclude_ids),
        "member_ids": [r["id"] for r in records],
        "base_model": cfg["base_model"], "lora": lcfg,
    }

    # Disabled adapter: no members -> save the zero-delta (freshly-init lora_B=0)
    # adapter, which is a no-op merge. This is the O(1) "drop the adapter" case.
    if len(records) == 0:
        meta["disabled"] = True
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        write_json(os.path.join(out_dir, "meta.json"), meta)
        print(f"a{j}: 0 records -> saved zero-delta (disabled) adapter")
        return out_dir

    ds = Dataset.from_dict({"text": [train_text(r, cfg.get("canary_repeat", 1)) for r in records]})
    tcfg = cfg["train"]
    train_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        optim=optim,
        learning_rate=tcfg["lr"],
        weight_decay=0.001,
        bf16=bf16,
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
    write_json(os.path.join(out_dir, "meta.json"), meta)
    print(f"a{j}: trained on {len(records)} records (seed {seed}) -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", type=int, required=True)
    ap.add_argument("--exclude_record_id", nargs="*", default=[])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    train_one(cfg, args.adapter, exclude_ids=args.exclude_record_id,
              out_dir=args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
