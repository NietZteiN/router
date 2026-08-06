"""Train the public SCAFFOLD LoRA (Alpaca) for the routing+scaffold method (SLURM-callable).

A single LoRA trained on ~N public Alpaca QA pairs — shared, never deleted, gives generic QA
competence. Recipe defaults match the LegoNet routing experts (r16/α32, non-rslora) so scaffold +
routed expert compose cleanly (base + ΔW_scaffold + ΔW_expert). Reuses the train_lora_shard trainer.

  python train_scaffold.py --n 2000 --output_dir checkpoints/Llama-3.2-1B-Instruct_scaffold_alpaca2k
"""
import argparse
import json
import os

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

import skill_data

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
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--no_rslora", action="store_true", default=True,
                    help="Non-rslora so scaffold+expert compose as a true additive sum (default on).")
    ap.add_argument("--rslora", dest="no_rslora", action="store_false")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["HF_HOME"] = args.hf_home

    if os.path.exists(os.path.join(args.output_dir, "adapter_config.json")):
        print(f"scaffold exists, skipping -> {args.output_dir}")
        return

    data = skill_data.load_alpaca(args.n, args.hf_home, seed=args.seed)
    ds = Dataset.from_list([{"text": skill_data.to_text(x)} for x in data])
    print(f"scaffold: {len(data)} Alpaca pairs, base={args.base_model}, rslora={not args.no_rslora}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM", use_rslora=not args.no_rslora)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    os.makedirs(args.output_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_32bit", learning_rate=args.lr, weight_decay=0.001, bf16=True,
        max_grad_norm=0.3, warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=25,
        save_strategy="no", report_to="none", seed=args.seed)
    trainer = SFTTrainer(
        model=model, train_dataset=ds, args=train_args, dataset_text_field="text",
        max_seq_length=args.max_length, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    json.dump({"source": "tatsu-lab/alpaca", "n": len(data), "base_model": args.base_model,
               "rank": args.rank, "alpha": args.alpha, "use_rslora": not args.no_rslora,
               "epochs": args.epochs, "seed": args.seed},
              open(os.path.join(args.output_dir, "scaffold_meta.json"), "w"), indent=2)
    print(f"Saved scaffold -> {args.output_dir}")


if __name__ == "__main__":
    main()
