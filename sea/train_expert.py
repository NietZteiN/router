"""Train one domain-expert LoRA adapter for SEA.

Each of the 4 domain adapters is trained independently on a domain-specific
dataset via supervised fine-tuning (SFT).  Training is identical across domains;
only the dataset changes.

Usage (SLURM, one job per domain):
    python train_expert.py --domain security \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --output_dir sea/checkpoints \
        --epochs 3 --rank 32 --seed 42

Datasets (HuggingFace, downloaded on first run):
    security : "Abirate/cybersecurity-instruction"   (fallback: synthetic)
    code     : "sahil2801/CodeAlpaca-20k"
    data     : "b-mc2/sql-create-context"
    general  : "tatsu-lab/alpaca"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from model_paths import experts_dir, model_slug

# ── Dataset loading ────────────────────────────────────────────────────────

def _load_security_dataset(max_samples: int | None):
    """Load a security-focused instruction-following dataset."""
    try:
        ds = load_dataset("Abirate/cybersecurity-instruction", split="train")
    except Exception:
        # Fallback: filter Alpaca for security-adjacent instructions
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ds = ds.filter(
            lambda x: any(
                kw in (x.get("instruction", "") + x.get("input", "")).lower()
                for kw in [
                    "security", "vulnerability", "encryption", "authentication",
                    "firewall", "malware", "CVE", "exploit", "password", "hash",
                ]
            )
        )
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def _load_code_dataset(max_samples: int | None):
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def _load_data_dataset(max_samples: int | None):
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def _load_general_dataset(max_samples: int | None):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # Drop empty-output rows
    ds = ds.filter(lambda x: len(x.get("output", "").strip()) > 0)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


_LOADERS = {
    "security": _load_security_dataset,
    "code": _load_code_dataset,
    "data": _load_data_dataset,
    "general": _load_general_dataset,
}

# ── Text formatting ────────────────────────────────────────────────────────

def _format_alpaca(example: dict) -> dict:
    instruction = example.get("instruction", "")
    inp = example.get("input", "").strip()
    output = example.get("output", example.get("response", ""))
    if inp:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}


def _format_sql(example: dict) -> dict:
    question = example.get("question", "")
    context = example.get("context", "")
    answer = example.get("answer", "")
    text = f"### Instruction:\nWrite a SQL query for the following question.\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Response:\n{answer}"
    return {"text": text}


def _format_code(example: dict) -> dict:
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}


def _get_formatter(domain: str):
    return {
        "security": _format_alpaca,
        "code": _format_code,
        "data": _format_sql,
        "general": _format_alpaca,
    }[domain]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train one SEA domain expert LoRA adapter")
    parser.add_argument("--domain", required=True, choices=list(_LOADERS.keys()))
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", default="sea/checkpoints")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset size (for smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_dir = experts_dir(args.output_dir, args.model_name, args.domain)
    if os.path.isfile(os.path.join(save_dir, "adapter_config.json")):
        print(f"[skip] Expert '{args.domain}' already exists at {save_dir}")
        sys.exit(0)
    if os.path.exists(os.path.join(save_dir, "adapter_model.safetensors")):
        print(f"[skip] Expert '{args.domain}' already exists at {save_dir}")
        sys.exit(0)

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ── Dataset ────────────────────────────────────────────────────────────
    print(f"Loading {args.domain} dataset …")
    raw = _LOADERS[args.domain](args.max_samples)
    formatter = _get_formatter(args.domain)
    dataset = raw.map(formatter, remove_columns=raw.column_names)

    # ── Model ──────────────────────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading base model {args.model_name} …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA config (rank=32, α=64, attention projections) ─────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Training ───────────────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "trainer_state"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        optim="paged_adamw_32bit",
        dataloader_num_workers=2,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        packing=False,
    )
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    meta = {
        "domain": args.domain,
        "model_name": args.model_name,
        "model_slug": model_slug(args.model_name),
        "rank": args.rank,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "n_samples": len(dataset),
    }
    with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] Expert '{args.domain}' saved to {save_dir}")


if __name__ == "__main__":
    main()
