"""Train ONE author's linearized task vector (tangent-space B-only LoRA) — composable_tv [lin].

    python train_linear_tv.py --config configs/ctv_1b_lin.json --author 82
    python train_linear_tv.py --config ... --author 82 --rank_override 64 --lr_override 5e-4
                                                        # ^ the pre-registered retry arm

Deliberately a HAND-ROLLED loop, NOT trl/SFTTrainer: accelerate's model wrapping and its
gradient-checkpointing hooks fight torch.func (functional_call/jvp) — the sift_one_task
precedent. The loss is the LINEARIZED model's shifted-CE (linear_tv.linearized_forward), so
the trained B lives in the tangent space the eval serves; A is frozen from the config's
irp_seed (linear_tv.seeded_A, the apply_irp_projections scheme).

Recipe (config-driven; mirrors the frozen e25 recipe): lr 1e-4 cosine, warmup_ratio 0.03,
weight_decay 0.001, clip 0.3, bsz 4 x accum 4, epochs 25, seed 42, fp32. Data = the author's
20 TOFU rows (rows [a*20,(a+1)*20) of `full`, k=200 semantics) via
train_lora_shard.load_shard_dataset/format_prompt; label masking = the
DataCollatorForLanguageModeling convention (labels == input_ids, pads -> -100; pad == eos so
any trailing eos is masked — the known repo convention).

Deviations from train_lora_shard (recorded in shard_meta.json):
  * plain torch.optim.AdamW, not paged_adamw_32bit — bnb paging is pointless in fp32 on one
    card and unavailable on CPU smoke runs.
  * lora_dropout 0 — the tangent build has no dropout site (tau = s*B@A is weight-space).

Saves per author (resume guard: skip when adapter_config.json exists):
  {out_dir}/shard_{a}/adapter_config.json + adapter_model.safetensors  (standard PEFT dir)
  {out_dir}/shard_{a}/b_only.pt                                        (rung-4 lean storage)
  {out_dir}/shard_{a}/shard_meta.json                                  (provenance)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import linear_tv as ltv
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="e.g. configs/ctv_1b_lin.json")
    p.add_argument("--author", type=int, required=True, help="TOFU author id (0..199)")
    p.add_argument("--rank_override", type=int, default=None,
                   help="Pre-registered retry: override train.rank (config retry.rank).")
    p.add_argument("--lr_override", type=float, default=None,
                   help="Pre-registered retry: override train.lr (config retry.lr).")
    p.add_argument("--hf_home", type=str,
                   default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _script_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tokenize_rows(texts, tokenizer, max_length):
    """One 1-D LongTensor per training text (add_special_tokens on — BOS, like SFTTrainer)."""
    return [torch.tensor(tokenizer(t, truncation=True, max_length=max_length)["input_ids"],
                         dtype=torch.long) for t in texts]


def collate_rows(rows, pad_id, device):
    """Right-pad to the batch max; labels = input_ids with pads masked to -100 (the
    DataCollatorForLanguageModeling(mlm=False) convention train_lora_shard uses)."""
    width = max(r.numel() for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    am = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : r.numel()] = r
        am[i, : r.numel()] = 1
    labels = ids.clone()
    labels[am == 0] = -100
    return {"input_ids": ids.to(device), "attention_mask": am.to(device),
            "labels": labels.to(device)}


def train_linear_factors(model, names, factors, rows, pad_id, *, epochs, batch_size,
                         grad_accum, lr, weight_decay, clip, warmup_ratio, seed,
                         device, log_every=5):
    """Hand-rolled AdamW-over-B loop against the linearized loss; returns per-step losses.

    Deterministic: per-epoch row order comes from one torch.Generator(seed) drawn
    sequentially; AdamW + CPU/eager kernels are deterministic, so same seed => byte-equal B
    (the exact-deletion provenance for this arm rests on re-derivability, like SIFT's tau_u).
    The remainder micro-batches at epoch end form a short accumulation group (loss averaged
    over the ACTUAL group size), mirroring HF Trainer's epoch-end flush."""
    g = torch.Generator().manual_seed(seed)
    b_params = [B for (_A, B, _s) in factors.values()]
    opt = torch.optim.AdamW(b_params, lr=lr, weight_decay=weight_decay)
    n_rows = len(rows)
    micro_per_epoch = math.ceil(n_rows / batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / grad_accum)
    total_steps = steps_per_epoch * epochs
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=math.ceil(warmup_ratio * total_steps),
        num_training_steps=total_steps)

    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n_rows, generator=g).tolist()
        micro = [perm[i:i + batch_size] for i in range(0, n_rows, batch_size)]
        for gi in range(0, len(micro), grad_accum):
            group = micro[gi:gi + grad_accum]
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for mb in group:
                batch = collate_rows([rows[i] for i in mb], pad_id, device)
                tau = ltv.build_tangent(factors)
                out = ltv.linearized_forward(
                    model, names, tau, batch["input_ids"],
                    attention_mask=batch["attention_mask"], labels=batch["labels"])
                (out.loss / len(group)).backward()
                step_loss += out.loss.item() / len(group)
            torch.nn.utils.clip_grad_norm_(b_params, clip)
            opt.step()
            sched.step()
            losses.append(step_loss)
            if len(losses) % log_every == 0 or len(losses) == total_steps:
                print(f"[train_linear_tv] epoch {epoch + 1}/{epochs} "
                      f"step {len(losses)}/{total_steps} loss {step_loss:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
    return losses


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    if not (0 <= args.author < 200):
        raise SystemExit(f"--author {args.author} out of range [0,200)")
    os.environ["HF_HOME"] = args.hf_home

    tr = cfg["train"]
    rank = args.rank_override if args.rank_override is not None else tr["rank"]
    lr = args.lr_override if args.lr_override is not None else tr["lr"]
    alpha = tr["alpha"]
    rslora = tr.get("rslora", True)
    seed = tr.get("seed", 42)
    epochs = tr["epochs"]
    batch_size = tr.get("batch_size", 4)
    grad_accum = tr.get("grad_accum", 4)
    max_length = tr.get("max_length", 256)
    irp_seed = cfg["irp_seed"]

    save_dir = os.path.join(cfg["out_dir"], f"shard_{args.author}")
    if os.path.exists(os.path.join(save_dir, "adapter_config.json")):
        print(f"shard_{args.author}: checkpoint exists, skipping -> {save_dir}")
        return

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_shard_dataset([args.author], args.hf_home)
    ds = ds.map(format_prompt, remove_columns=["question", "answer"])
    print(f"shard_{args.author} ({cfg['arm']} arm): {len(ds)} Q&As, rank {rank}, lr {lr}, "
          f"epochs {epochs}, device {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # fp32 + eager + no cache: the linearized-forward requirements (see linear_tv docstring).
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=True).to(device)
    model.config.use_cache = False
    model.eval()                      # frozen base; only B trains
    for p in model.parameters():
        p.requires_grad_(False)

    names = ltv.target_names(model)
    factors = ltv.init_author_factors(
        model, names, rank=rank, alpha=alpha, rslora=rslora,
        irp_seed=irp_seed, author=args.author, device=device)
    n_b = sum(B.numel() for (_A, B, _s) in factors.values())
    print(f"targets: {len(names)} Linear weights; trainable B params: {n_b:,} "
          f"(A frozen from irp_seed {irp_seed})")

    rows = tokenize_rows(ds["text"], tokenizer, max_length)
    losses = train_linear_factors(
        model, names, factors, rows, tokenizer.pad_token_id,
        epochs=epochs, batch_size=batch_size, grad_accum=grad_accum, lr=lr,
        weight_decay=tr.get("weight_decay", 0.001), clip=tr.get("max_grad_norm", 0.3),
        warmup_ratio=tr.get("warmup_ratio", 0.03), seed=seed, device=device)

    ltv.save_author_adapter(save_dir, factors, base_model_name=cfg["model_name"],
                            rank=rank, alpha=alpha, rslora=rslora)
    b_meta = {
        "author": args.author, "irp_seed": irp_seed, "rank": rank, "alpha": alpha,
        "rslora": rslora,
        "in_features": {n: A.shape[1] for n, (A, _B, _s) in factors.items()},
    }
    ltv.save_b_only(save_dir, {n: B for n, (_A, B, _s) in factors.items()}, b_meta)

    meta = {
        "author": args.author,
        "name": f"shard_{args.author}",
        "arm": cfg["arm"],
        "linearized": True,
        "model_name": cfg["model_name"],
        "rank": rank,
        "alpha": alpha,
        "rslora": rslora,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "weight_decay": tr.get("weight_decay", 0.001),
        "max_grad_norm": tr.get("max_grad_norm", 0.3),
        "warmup_ratio": tr.get("warmup_ratio", 0.03),
        "max_length": max_length,
        "lora_dropout": 0.0,           # deviation: no dropout site in the tangent build
        "optimizer": "torch.optim.AdamW",  # deviation: not paged_adamw_32bit (see docstring)
        "seed": seed,
        "irp_seed": irp_seed,
        "rank_override": args.rank_override,
        "lr_override": args.lr_override,
        "num_samples": len(ds),
        "steps": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "script_sha256": _script_sha256(),
        "config": os.path.abspath(args.config),
    }
    with open(os.path.join(save_dir, "shard_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved linearized task vector -> {save_dir} "
          f"(loss {meta['loss_first']:.4f} -> {meta['loss_final']:.4f})")


if __name__ == "__main__":
    main()
