"""Train one LoRA adaptor on a single TOFU author shard (SLURM-callable)."""
import os
import json
import argparse
import hashlib

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

from shard_utils import get_author_shard
import tofu_env as _tofu_env

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


def apply_irp_projections(model, shard_id: int, irp_seed: int):
    """Fix lora_A weights from a per-shard deterministic seed and freeze them.

    Each (shard_id, layer_name, adapter_key) triple gets its own seed via SHA-256,
    so layers are independent and iteration order doesn't affect reproducibility.
    """
    for layer_name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        for adapter_key, linear in module.lora_A.items():
            seed_bytes = hashlib.sha256(
                f"{irp_seed}:{shard_id}:{layer_name}:{adapter_key}".encode()
            ).digest()
            seed_int = int.from_bytes(seed_bytes[:4], "little")
            gen = torch.Generator()
            gen.manual_seed(seed_int)
            # Draw on CPU, then copy to the weight's device/dtype: nn.init.normal_ with a
            # CPU generator raises on CUDA weights (memsinks freeze_lora_a_irp precedent,
            # jobs 443551/445685); CPU draws are bit-identical to the original behavior.
            w = torch.empty(linear.weight.shape)
            torch.nn.init.normal_(w, mean=0.0, std=1.0, generator=gen)
            with torch.no_grad():
                linear.weight.copy_(w.to(device=linear.weight.device, dtype=linear.weight.dtype))
            linear.weight.requires_grad_(False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_id", type=int, default=None,
                   help="Required unless --retain90. Shard k-1 (e.g. 9 at k=10) is the forget shard.")
    p.add_argument("--retain90", action="store_true",
                   help="Train one LoRA on the retain split (authors 0..retain_authors-1) saved to "
                        "{output_dir}/retain90/ — the forget-quality KS oracle; never sees forget data.")
    p.add_argument("--retain_authors", type=int, default=180,
                   help="With --retain90: number of leading authors the oracle trains on. Default 180 "
                        "(= retain90; forget shard 9 at k=10). Use 150 for a k=4 oracle (forget shard 3 "
                        "= authors 150-199). Dir name stays retain90/ so prepare_eval.py finds it.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Defaults = frozen recipe from the 2026-06-11 shard grid (reports/SHARD_GRID_REPORT_2026-06-11.md):
    # best at k=1, k=4 and k=10. Retain oracles keep the legacy recipe — pass
    # --rank 8 --alpha 16 --epochs 3 --lr 2e-4 explicitly with --retain90 (see CLAUDE.md).
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--no_rslora", action="store_true",
                   help="Disable rsLoRA (scaling alpha/r instead of alpha/sqrt(r)) so a 1/N "
                        "add_weighted_adapter merge is a TRUE mean (no sqrt(r) inflation) — used by "
                        "the merge-mechanism non-rslora rerun. Default keeps use_rslora=True.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--hf_home", type=str,
                   default=os.environ.get("HF_HOME") or _tofu_env.hf_home())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--irp_seed", type=int, default=None,
                   help="If set, fix each shard's lora_A matrices from this seed and freeze them (IRP mode).")
    p.add_argument("--plant_manifest", type=str, default=None,
                   help="Entangled-facts (Mode-B) plant: append this manifest's planted rows for "
                        "this shard to its training set (entangle_data.load_planted_shard_dataset). "
                        "Works with --shard_id and --retain90 (a planted retain oracle legitimately "
                        "holds the host copies). Records plant_manifest_sha256 + planted-row count "
                        "in shard_meta.json. See log/entangled_facts/.")
    # Negative-anchored isolation (merge_mechanism §6.3,
    # log/merge_mechanism/2026-07-15_negative-anchor-design.md): add
    # anchor_lambda * mean_{modules, anchor tokens} ||scaling * B(A(h))||^2 on public
    # author-independent text to the SFT loss — the off-author negatives isolated training
    # lacks. Defaults OFF: flag-free behavior is bit-identical to the frozen recipe. The
    # anchor set is public + seeded, so the exactness certificate survives.
    p.add_argument("--anchor_lambda", type=float, default=0.0,
                   help="Weight of the negative-anchor penalty; 0 (default) disables it.")
    p.add_argument("--anchor_n", type=int, default=2000,
                   help="Number of public anchor texts (matches the scaffold's Alpaca 2k).")
    p.add_argument("--anchor_seed", type=int, default=42)
    p.add_argument("--anchor_batch_size", type=int, default=4)
    p.add_argument("--anchor_source", type=str, default="alpaca", choices=["alpaca"],
                   help="Anchor corpus; only public Alpaca (skill_data.load_alpaca) for now.")
    return p.parse_args()


def anchor_penalty(model, batch):
    """Differentiable negative-anchor penalty: mean over (LoRA modules, non-pad tokens) of
    ||scaling * B(A(h_t))||^2 on the given batch, via forward hooks on every lora_B
    (captures B(A(dropout(h)))) with the module's scaling applied exactly once here.
    fp32 accumulation. Guarded by test_train_anchor.py (hook == dense closed form)."""
    caps, hooks = [], []
    for _, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and "default" in mod.lora_B:
            scale = mod.scaling["default"]

            def make_hook(s):
                def hook(_m, _inp, out):
                    caps.append((s, out))
                return hook

            hooks.append(mod.lora_B["default"].register_forward_hook(make_hook(scale)))
    if not hooks:
        raise ValueError("anchor_penalty: no lora_B modules found on the model")
    try:
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for h in hooks:
            h.remove()
    mask = batch["attention_mask"].unsqueeze(-1).float()
    denom = batch["attention_mask"].sum().float() * len(caps)
    pen = None
    for s, out in caps:
        term = (s ** 2) * ((out.float() * mask) ** 2).sum() / denom
        pen = term if pen is None else pen + term
    return pen


def apply_anchor_to_loss(out, model, anchor_batches, anchor_lambda, anchor_idx,
                         return_outputs=False):
    """Pure §6.3 loss transform, unit-tested in test_train_anchor.py: returns
    (new_out, new_idx). anchor_lambda <= 0 or no batches -> `out` returned UNCHANGED
    (the flag-free frozen-recipe invariant); otherwise parent loss +
    anchor_lambda * anchor_penalty on the next cycling batch (deterministic order —
    part of the exactness function)."""
    if anchor_lambda <= 0 or not anchor_batches:
        return out, anchor_idx
    loss = out[0] if return_outputs else out
    batch = anchor_batches[anchor_idx % len(anchor_batches)]
    batch = {k: v.to(loss.device) for k, v in batch.items()}
    loss = loss + anchor_lambda * anchor_penalty(model, batch).to(loss.dtype)
    return ((loss, out[1]) if return_outputs else loss), anchor_idx + 1


class AnchoredSFTTrainer(SFTTrainer):
    """SFTTrainer + the §6.3 negative-anchor penalty (see apply_anchor_to_loss)."""

    def __init__(self, *args, anchor_batches=None, anchor_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._anchor_batches = anchor_batches or []
        self._anchor_lambda = anchor_lambda
        self._anchor_idx = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        out, self._anchor_idx = apply_anchor_to_loss(
            out, model, self._anchor_batches, self._anchor_lambda, self._anchor_idx,
            return_outputs)
        return out


def build_anchor_batches(tokenizer, args):
    """Pre-tokenized CPU batches of public anchor text in the training text schema."""
    from skill_data import load_alpaca, to_text
    pairs = load_alpaca(args.anchor_n, args.hf_home, seed=args.anchor_seed)
    texts = [to_text(p) for p in pairs]
    batches = []
    for i in range(0, len(texts), args.anchor_batch_size):
        enc = tokenizer(texts[i:i + args.anchor_batch_size], truncation=True,
                        max_length=args.max_length, padding=True, return_tensors="pt")
        batches.append({"input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"]})
    return batches


def load_shard_dataset(shard_authors, hf_home):
    os.environ["HF_HOME"] = hf_home
    full = load_dataset("locuslab/TOFU", "full")["train"]
    # Each author occupies 20 consecutive rows: author i -> rows [i*20, (i+1)*20)
    indices = [r for a in shard_authors for r in range(a * 20, a * 20 + 20)]
    return full.select(indices)


def format_prompt(example):
    return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home

    if args.retain90:
        shard_name = "retain90"
        # Oracle = leading authors only; default 180 = TOFU retain90 (excludes forget10).
        # Must exclude the eval's forget shard: pass 150 when the forget shard is k=4's shard 3.
        shard_authors = list(range(args.retain_authors))
    else:
        if args.shard_id is None:
            raise SystemExit("--shard_id is required unless --retain90")
        shard_name = f"shard_{args.shard_id}"
        shard_authors = get_author_shard(args.k, args.shard_id)

    save_dir = os.path.join(args.output_dir, shard_name)
    if os.path.exists(os.path.join(save_dir, "adapter_config.json")):
        print(f"{shard_name}: checkpoint exists, skipping -> {save_dir}")
        return

    planted_rows = 0
    if args.plant_manifest:
        import entangle_data as ed
        plant_shard = None if args.retain90 else args.shard_id
        shard_dataset, planted_rows = ed.load_planted_shard_dataset(
            shard_authors, args.plant_manifest, args.hf_home, plant_shard)
        print(f"{shard_name}: +{planted_rows} planted rows from {args.plant_manifest}")
    else:
        shard_dataset = load_shard_dataset(shard_authors, args.hf_home)
    shard_dataset = shard_dataset.map(format_prompt, remove_columns=["question", "answer"])

    print(f"{shard_name} (k={args.k}): {len(shard_authors)} authors, {len(shard_dataset)} Q&As")
    print(f"  Author IDs: {shard_authors[0]}..{shard_authors[-1]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=not args.no_rslora,
    )
    model = get_peft_model(model, lora_cfg)
    if args.irp_seed is not None and not args.retain90:
        apply_irp_projections(model, args.shard_id, args.irp_seed)
        print(f"IRP mode: lora_A frozen from seed {args.irp_seed}+shard_{args.shard_id}")
    model.print_trainable_parameters()

    os.makedirs(save_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_32bit",
        learning_rate=args.lr,
        weight_decay=0.001,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )

    anchor_batches = None
    if args.anchor_lambda > 0:
        anchor_batches = build_anchor_batches(tokenizer, args)
        print(f"anchor: lambda={args.anchor_lambda}, {args.anchor_n} public "
              f"{args.anchor_source} texts -> {len(anchor_batches)} cycling batches")

    trainer = AnchoredSFTTrainer(
        model=model,
        train_dataset=shard_dataset,
        args=train_args,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        anchor_batches=anchor_batches,
        anchor_lambda=args.anchor_lambda,
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    meta = {
        "shard_id": args.shard_id,
        "name": shard_name,
        "k": args.k,
        "author_ids": shard_authors,
        "num_samples": len(shard_dataset),
        "model_name": args.model_name,
        "rank": args.rank,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "seed": args.seed,
        "irp_seed": args.irp_seed,
        "plant_manifest": args.plant_manifest,
        "plant_manifest_sha256": (
            __import__("hashlib").sha256(open(args.plant_manifest, "rb").read()).hexdigest()
            if args.plant_manifest else None),
        "planted_rows": planted_rows,
        "anchor_lambda": args.anchor_lambda,
        "anchor_n": args.anchor_n if args.anchor_lambda > 0 else None,
        "anchor_seed": args.anchor_seed if args.anchor_lambda > 0 else None,
        "anchor_source": args.anchor_source if args.anchor_lambda > 0 else None,
    }
    with open(os.path.join(save_dir, "shard_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved LoRA adaptor -> {save_dir}")


if __name__ == "__main__":
    main()
