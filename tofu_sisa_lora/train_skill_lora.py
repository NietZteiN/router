"""Train one isolated skill-LoRA adapter for the Part B facts-vs-skills contrast (SLURM-callable).

Mirrors train_lora_shard.py's recipe EXACTLY (LoraConfig r/alpha/rslora/6-modules, TrainingArguments,
SFTTrainer) — only the data differs: one Super-NaturalInstructions task, input-only text. Saves the
adapter's own held-out instances into skill_meta.json so eval_skill.py scores the exact same split.

  python train_skill_lora.py --config configs/skills_superni_1b.json --adapter 0
"""
import argparse
import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", type=int, required=True, help="adapter index j in [0, n)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.environ["HF_HOME"] = cfg["hf_home"]

    j = args.adapter
    task_id = cfg["task_ids"][j]
    seed = cfg["base_seed"] + j                       # per-adapter seed (mirrors train_legonet_adapter)
    save_dir = os.path.join(cfg["output_dir"], f"a{j}")
    if os.path.exists(os.path.join(save_dir, "adapter_config.json")):
        print(f"a{j}: checkpoint exists, skipping -> {save_dir}")
        return

    train, holdout = skill_data.skill_split(
        task_id, cfg["hf_home"], cfg["samples_per_adapter"], cfg["holdout_per_adapter"], seed)
    ds = Dataset.from_list([{"text": skill_data.to_text(x)} for x in train])
    print(f"a{j} task={task_id} seed={seed}: train={len(train)} holdout={len(holdout)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    lc = cfg["lora"]
    lora_cfg = LoraConfig(
        r=lc["rank"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM",
        use_rslora=lc["use_rslora"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    os.makedirs(save_dir, exist_ok=True)

    t = cfg["train"]
    train_args = TrainingArguments(
        output_dir=save_dir, num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"], gradient_accumulation_steps=t["grad_accum"],
        optim="paged_adamw_32bit", learning_rate=t["lr"], weight_decay=0.001, bf16=True,
        max_grad_norm=0.3, warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=10,
        save_strategy="no", report_to="none", seed=seed,
    )
    trainer = SFTTrainer(
        model=model, train_dataset=ds, args=train_args, dataset_text_field="text",
        max_seq_length=t["max_length"], data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    json.dump({"adapter": j, "task_id": task_id, "skill": cfg.get("task_skill_labels", [None] * (j + 1))[j],
               "seed": seed, "num_train": len(train), "num_holdout": len(holdout), "held_out": holdout},
              open(os.path.join(save_dir, "skill_meta.json"), "w"), indent=2)
    print(f"Saved skill adapter -> {save_dir}")


if __name__ == "__main__":
    main()
