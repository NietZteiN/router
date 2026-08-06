"""Train one non-LoRA PEFT adapter on a single TOFU author shard (SLURM-callable).

The peft_compose bake-off trainer: same data path and shard conventions as
train_lora_shard.py (whose loaders it imports), but the adapter is one of
  prefix  PrefixTuningConfig  (compose = KV concat, prefix_concat.py)
  vera    VeraConfig          (compose = mean of vera_lambda_* in the shared frozen basis)
  ia3     IA3Config           (compose = mean of gate vectors)
  dora    LoraConfig(use_dora=True)  (additive control arm)
Per-method hyperparameters live in configs/peft_bakeoff_1b.json — method-standard lrs and
schedules are deliberately NOT recipe-matched across methods (recorded in the config; the
bake-off compares each method's composed-vs-isolated delta plus absolute mu).

CLI:
    python train_peft_shard.py --config configs/peft_bakeoff_1b.json --method vera --shard_id 0
    python train_peft_shard.py --config ... --method prefix --shard_id 3 --smoke   # 2-step gate
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
from peft import IA3Config, LoraConfig, PrefixTuningConfig, VeraConfig, get_peft_model
from trl import SFTTrainer

from shard_utils import get_author_shard
from train_lora_shard import format_prompt, load_shard_dataset

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

METHODS = ("prefix", "vera", "ia3", "dora")


def build_peft_config(method: str, mc: dict):
    if method == "prefix":
        return PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=mc["num_virtual_tokens"],
        )
    if method == "vera":
        return VeraConfig(
            task_type="CAUSAL_LM",
            r=mc["r"],
            d_initial=mc["d_initial"],
            vera_dropout=mc["vera_dropout"],
            target_modules=mc["target_modules"],
            projection_prng_key=mc["projection_prng_key"],
            save_projection=mc["save_projection"],
        )
    if method == "ia3":
        return IA3Config(
            task_type="CAUSAL_LM",
            target_modules=mc["target_modules"],
            feedforward_modules=mc["feedforward_modules"],
        )
    if method == "dora":
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=mc["r"],
            lora_alpha=mc["alpha"],
            lora_dropout=mc["lora_dropout"],
            target_modules=mc["target_modules"],
            use_rslora=mc["use_rslora"],
            use_dora=True,
            bias="none",
        )
    raise ValueError(f"unknown method {method!r}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--shard_id", type=int, required=True)
    p.add_argument("--smoke", action="store_true",
                   help="2-step micro-run (max_steps=2) — pipeline gate, not a result")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    cfg = json.load(open(args.config))
    mc = cfg["methods"][args.method]

    out_dir = os.path.join(cfg["output_root"], cfg["dir_template"].format(method=args.method))
    save_dir = os.path.join(out_dir, f"shard_{args.shard_id}" + ("_smoke" if args.smoke else ""))
    if os.path.exists(os.path.join(save_dir, "adapter_config.json")):
        print(f"{args.method} shard_{args.shard_id}: checkpoint exists, skipping -> {save_dir}")
        return

    shard_authors = get_author_shard(cfg["k"], args.shard_id)
    ds = load_shard_dataset(shard_authors, args.hf_home)
    ds = ds.map(format_prompt, remove_columns=["question", "answer"])
    print(f"{args.method} shard_{args.shard_id} (k={cfg['k']}): "
          f"{len(shard_authors)} authors, {len(ds)} Q&As")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(model, build_peft_config(args.method, mc))
    model.print_trainable_parameters()

    os.makedirs(save_dir, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=mc["epochs"],
        max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        optim="paged_adamw_32bit",
        learning_rate=mc["lr"],
        weight_decay=0.001,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=train_args,
        dataset_text_field="text",
        max_seq_length=cfg["max_length"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    meta = {
        "method": args.method, "shard_id": args.shard_id, "k": cfg["k"],
        "author_ids": list(shard_authors), "num_samples": len(ds),
        "model_name": cfg["model_name"], "seed": cfg["seed"], "smoke": args.smoke,
        "hparams": {k: v for k, v in mc.items() if not k.startswith("_")},
        "final_train_loss": (trainer.state.log_history[-1].get("train_loss")
                             if trainer.state.log_history else None),
    }
    with open(os.path.join(save_dir, "shard_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {args.method} adapter -> {save_dir}")


if __name__ == "__main__":
    main()
