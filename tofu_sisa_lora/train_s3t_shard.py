"""Train one S³T shard: L sequential stages, per-stage layer-disjoint LoRA (SLURM-callable).

Port of the official S3T LLM recipe (~/S3T/src/s3t_llm.py; Bourtoule-style slices inside a
SISA shard, ICLR'25 S3T) onto the TOFU/repo conventions:

  - stage j trains ONLY the LoRA params of layer block j (top-down; see
    shard_utils.get_s3t_layer_block) on the CUMULATIVE permuted slices 0..j
    (official: train[0%:(j+1)*100/L%]).
  - one LoRA adapter + per-stage requires_grad masks instead of the official
    stack-a-new-LoRA-per-8bit-checkpoint — equivalent because the layer blocks are
    disjoint and untrained blocks have lora_B == 0 (zero delta); proven by test_s3t.py.
  - exact layer-id regex matching. The official check_if() matches substrings, so
    "layers.3" also hits "layers.31" — harmless in their configs (two-digit ids only),
    but it would silently break exactness in our full-coverage block for layers 0-9.
  - bf16 base + repo trainer conventions (train_lora_shard.py) instead of
    8-bit/prepare_model_for_int8_training; TOFU "Question:/Answer:" format, seed 42.

Snapshots: stages/stage_{j}/ after every stage (adapter + stage_meta.json); the final
adapter additionally at the shard root (loader-compatible). The unlearned state is the
snapshot taken before the first forget slice — for the forget-containing shard with the
forget-last ordering that is stages/stage_{L-3}/ (m=5/L=4: stage_1).
"""
import argparse
import hashlib
import json
import os
import re

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

from shard_utils import (
    get_s3t_layer_block,
    get_s3t_ordering,
    get_s3t_shard_authors,
    get_s3t_stage_authors,
)
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

_LAYER_ID_RE = re.compile(r"\.layers\.(\d+)\.")


def mask_stage_params(model, layer_ids) -> int:
    """requires_grad = True only for LoRA params inside `layer_ids` decoder layers.

    Exact integer matching via regex — never substring (official S3T check_if bug).
    Returns the number of enabled parameter tensors.
    """
    layer_ids = set(int(i) for i in layer_ids)
    enabled = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            continue
        m = _LAYER_ID_RE.search(name)
        on = m is not None and int(m.group(1)) in layer_ids
        param.requires_grad = on
        enabled += int(on)
    return enabled


def lora_param_names(model, layer_ids):
    """Names of LoRA param tensors inside `layer_ids` (same matching as the mask)."""
    layer_ids = set(int(i) for i in layer_ids)
    out = []
    for name, _ in model.named_parameters():
        if "lora_" not in name:
            continue
        m = _LAYER_ID_RE.search(name)
        if m is not None and int(m.group(1)) in layer_ids:
            out.append(name)
    return out


def run_stage(model, tokenizer, dataset, stage_out_dir, cfg, micro=False):
    """One S3T stage: fresh TrainingArguments/optimizer (official semantics: one
    process per stage), train, snapshot the adapter to stage_out_dir."""
    train_args = TrainingArguments(
        output_dir=stage_out_dir,
        num_train_epochs=cfg["epochs_per_stage"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        optim=cfg["optim"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        bf16=torch.cuda.is_available(),
        max_grad_norm=cfg["max_grad_norm"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
        **({"max_steps": 2} if micro else {}),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # trl 0.9.6 otherwise re-derives it from config._name_or_path
        train_dataset=dataset,
        args=train_args,
        dataset_text_field="text",
        max_seq_length=cfg["max_length"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    os.makedirs(stage_out_dir, exist_ok=True)
    model.save_pretrained(stage_out_dir)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="configs/s3t_arm{A,B}.json")
    p.add_argument("--shard_id", type=int, required=True)
    p.add_argument("--output_dir", required=True,
                   help="arm checkpoint dir; writes shard_{id}/ (+ stages/) inside it")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    p.add_argument("--micro", action="store_true",
                   help="gate mode: 2 optimizer steps/stage + bit-identity masking assertions")
    # Multi-sequence (budget B>1) training: train sequence `seq_id` on an explicit
    # slice `ordering`. Writes shard_{id}/seq_{seq_id}/ instead of shard_{id}/.
    p.add_argument("--seq_id", type=int, default=None,
                   help="sequence index within the shard (S3T budget B>1). Omit for the "
                        "legacy single-ordering layout (shard_{id}/ directly).")
    p.add_argument("--ordering", type=str, default=None,
                   help="comma-separated slice permutation, e.g. '0,1,2,3'. Defaults to "
                        "get_s3t_ordering(m, shard_id, L) when omitted.")
    # Overrides for the 1B micro gate (structure must still satisfy the asserts).
    p.add_argument("--model_name", default=None)
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--num_loras", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    with open(args.config) as f:
        cfg = json.load(f)
    for key in ("model_name", "m", "L", "num_loras"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    m, L, num_loras = cfg["m"], cfg["L"], cfg["num_loras"]

    # Layout: shard_{id}/ (legacy, single ordering) or shard_{id}/seq_{seq_id}/ (B>1).
    if args.seq_id is None:
        shard_dir = os.path.join(args.output_dir, f"shard_{args.shard_id}")
    else:
        shard_dir = os.path.join(args.output_dir, f"shard_{args.shard_id}", f"seq_{args.seq_id}")
    if os.path.exists(os.path.join(shard_dir, "adapter_config.json")):
        print(f"shard_{args.shard_id}"
              f"{'' if args.seq_id is None else f'/seq_{args.seq_id}'}: "
              f"final adapter exists, skipping -> {shard_dir}")
        return
    stages_root = os.path.join(shard_dir, "stages")

    if args.ordering is not None:
        ordering = [int(x) for x in args.ordering.split(",")]
        assert sorted(ordering) == list(range(L)), f"--ordering must be a perm of 0..{L-1}"
    else:
        ordering = get_s3t_ordering(m, args.shard_id, L)
    shard_authors = get_s3t_shard_authors(m, args.shard_id)
    print(f"S3T shard {args.shard_id}/{m} seq {args.seq_id} (arm {cfg.get('arm')}): authors "
          f"{shard_authors[0]}..{shard_authors[-1]}, L={L}, ordering={ordering}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    num_layers = base.config.num_hidden_layers
    assert L * num_loras <= num_layers, (
        f"L={L} x num_loras={num_loras} exceeds {num_layers} layers"
    )

    # Resume: continue from the newest stage snapshot, else fresh LoRA.
    start_stage = 0
    for j in range(L - 1, -1, -1):
        snap = os.path.join(stages_root, f"stage_{j}")
        if os.path.exists(os.path.join(snap, "adapter_config.json")):
            start_stage = j + 1
            print(f"resuming from snapshot {snap} -> starting at stage {start_stage}")
            model = PeftModel.from_pretrained(base, snap, is_trainable=True)
            break
    else:
        lora_cfg = LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=cfg["use_rslora"],
        )
        model = get_peft_model(base, lora_cfg)

    expected = num_loras * len(cfg["target_modules"]) * 2  # A+B per module per layer
    for j in range(start_stage, L):
        block = get_s3t_layer_block(j, num_loras, num_layers)
        stage_authors = get_s3t_stage_authors(m, args.shard_id, L, j, ordering)
        ds = load_shard_dataset(stage_authors, args.hf_home)
        ds = ds.map(format_prompt, remove_columns=["question", "answer"])

        n_enabled = mask_stage_params(model, block)
        assert n_enabled == expected, (
            f"stage {j}: masked {n_enabled} trainable tensors, expected {expected}"
        )
        print(f"--- stage {j}: layers {block}, {len(stage_authors)} cumulative authors, "
              f"{len(ds)} samples, {n_enabled} trainable tensors ---", flush=True)

        if args.micro:
            frozen_before = {n: p.detach().clone() for n, p in model.named_parameters()
                             if "lora_" in n and not p.requires_grad}
        run_stage(model, tokenizer, ds, os.path.join(stages_root, f"stage_{j}"),
                  cfg, micro=args.micro)
        if args.micro:
            for n, p in model.named_parameters():
                if n in frozen_before:
                    assert torch.equal(p.detach().cpu(), frozen_before[n].cpu()), (
                        f"stage {j} leaked into frozen param {n}"
                    )
            print(f"stage {j}: micro bit-identity check passed "
                  f"({len(frozen_before)} frozen tensors untouched)")
        # Exactness invariant: blocks for stages > j must still be zero-delta.
        for jj in range(j + 1, L):
            later = lora_param_names(model, get_s3t_layer_block(jj, num_loras, num_layers))
            for n, p in model.named_parameters():
                if n in later and "lora_B" in n:
                    assert not p.detach().abs().sum().item(), (
                        f"stage {j}: future block param {n} is nonzero"
                    )

        meta = {
            "stage": j,
            "shard_id": args.shard_id,
            "m": m, "L": L, "num_loras": num_loras,
            "ordering": ordering,
            "layer_block": block,
            "cumulative_authors": stage_authors,
            "num_samples": len(ds),
            "arm": cfg.get("arm"),
            "lr": cfg["lr"], "epochs_per_stage": cfg["epochs_per_stage"],
            "rank": cfg["rank"], "alpha": cfg["alpha"], "seed": cfg["seed"],
            "micro": args.micro,
        }
        with open(os.path.join(stages_root, f"stage_{j}", "stage_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    os.makedirs(shard_dir, exist_ok=True)
    model.save_pretrained(shard_dir)
    tokenizer.save_pretrained(shard_dir)
    meta = {
        "shard_id": args.shard_id,
        "seq_id": args.seq_id,
        "name": f"shard_{args.shard_id}",
        # k = the EVAL split (forget10/retain90 -> --k 10), not the S3T shard count;
        # collect_results.py reads rank/epochs/k from here.
        "k": 10,
        "s3t": {"m": m, "L": L, "num_loras": num_loras, "ordering": ordering},
        "author_ids": shard_authors,
        "model_name": cfg["model_name"],
        "rank": cfg["rank"],
        "alpha": cfg["alpha"],
        "epochs": cfg["epochs_per_stage"],
        "lr": cfg["lr"],
        "arm": cfg.get("arm"),
        "seed": cfg["seed"],
        "config_path": os.path.abspath(args.config),
        "config_sha256": _sha256(args.config),
        "micro": args.micro,
    }
    with open(os.path.join(shard_dir, "shard_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved S3T shard adapter -> {shard_dir} (snapshots in {stages_root})")


if __name__ == "__main__":
    main()
