"""TOFU unlearning baseline training (Gradient Ascent / Gradient Difference / KL / IDK).

Starts from an existing k=1 full-data LoRA checkpoint (e.g. checkpoints/{slug}_ft/shard_0/)
and fine-tunes it with a modified unlearning objective. The resulting adapter is saved to
{output_dir}/shard_0/ so it is a drop-in for the existing eval pipeline.

Methods:
  ga   — Gradient Ascent: maximize loss on forget set.
  gd   — Gradient Difference: GA on forget + CE on retain.
  kl   — KL Minimization: GA on forget + KL(orig||cur) on retain.
  idk  — Preference Optimization: CE on forget-with-IDK-answers + CE on retain.
"""
import argparse
import json
import os
import random
import subprocess

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

from shard_utils import get_author_shard

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

# 20 IDK response variants used as fallback when forget10_idk split is unavailable
IDK_RESPONSES = [
    "I don't know the answer to that.",
    "I'm not sure about that.",
    "I don't have that information.",
    "I'm unable to answer that question.",
    "That's not something I can help with.",
    "I don't have enough information to answer.",
    "I'm not familiar with that topic.",
    "I cannot provide an answer to that.",
    "I'm afraid I don't know.",
    "I don't have the knowledge to answer that.",
    "That's outside my knowledge.",
    "I'm not knowledgeable about that.",
    "I have no information on that topic.",
    "I simply don't know.",
    "I can't say for certain.",
    "I'm not in a position to answer that.",
    "I'm unable to provide information on that.",
    "I don't have a good answer for that.",
    "I'm unsure about the details.",
    "I cannot answer that question.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--ft_dir", required=True,
                   help="Path to the full-data ft checkpoint dir (contains shard_0/).")
    p.add_argument("--method", required=True, choices=["ga", "gd", "kl", "idk"])
    p.add_argument("--output_dir", required=True,
                   help="Where to write shard_0/. Should be e.g. checkpoints/{slug}_ft_unlearn_{method}/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5,
                   help="Training epochs (TOFU paper uses 5).")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (TOFU paper uses 1e-5).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum).")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--kl_weight", type=float, default=1.0,
                   help="Weight on KL term for --method kl.")
    p.add_argument("--max_steps", type=int, default=0,
                   help="Stop after this many optimizer steps (0 = full run). Use 20 for smoke validation.")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    return p.parse_args()


# ── Dataset helpers ──────────────────────────────────────────────────────────

def _format_text(question, answer):
    return f"Question: {question}\nAnswer: {answer}"


def _load_tofu_records(hf_home, authors):
    """Return list of {"text": ...} records for the given author IDs."""
    os.environ["HF_HOME"] = hf_home
    full = load_dataset("locuslab/TOFU", "full")["train"]
    indices = [r for a in authors for r in range(a * 20, a * 20 + 20)]
    ds = full.select(indices)
    return [{"text": _format_text(ex["question"], ex["answer"])} for ex in ds]


def load_forget_records(hf_home):
    """Forget set: TOFU forget10 authors = authors 180-199 (same as shard_9 with k=10)."""
    return _load_tofu_records(hf_home, get_author_shard(10, 9))


def load_retain_records(hf_home):
    """Retain set: all authors except forget10 (authors 0-179)."""
    return _load_tofu_records(hf_home, list(range(0, 180)))


def load_idk_records(hf_home, rng):
    """IDK forget set: same questions as forget10 but with IDK answers.

    Tries to load the locuslab/TOFU forget10_idk split (which contains model-generated
    IDK responses). Falls back to assigning random IDK_RESPONSES strings if unavailable.
    """
    os.environ["HF_HOME"] = hf_home
    try:
        idk_ds = load_dataset("locuslab/TOFU", "forget10_idk")["train"]
        # The idk split has columns: question, answer (where answer is an idk-style response)
        records = [{"text": _format_text(ex["question"], ex["answer"])} for ex in idk_ds]
        print(f"Loaded {len(records)} IDK records from forget10_idk split.", flush=True)
        return records
    except Exception as e:
        print(f"forget10_idk split unavailable ({e}); generating IDK responses from fallback list.", flush=True)
    # Fallback: pair each forget question with a random IDK response
    full = load_dataset("locuslab/TOFU", "full")["train"]
    forget_indices = [r for a in get_author_shard(10, 9) for r in range(a * 20, a * 20 + 20)]
    forget_qs = [full[i]["question"] for i in forget_indices]
    return [{"text": _format_text(q, rng.choice(IDK_RESPONSES))} for q in forget_qs]


# ── Tokenized Dataset ─────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.items = []
        for rec in records:
            enc = tokenizer(
                rec["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.items.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ── Loss helpers ──────────────────────────────────────────────────────────────

def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def ce_loss(model, batch, device):
    """Standard causal LM cross-entropy (mean over non-padding tokens)."""
    b = _to_device(batch, device)
    labels = b["input_ids"].clone()
    labels[b["attention_mask"] == 0] = -100
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=labels)
    return out.loss


def get_logits(model, batch, device):
    b = _to_device(batch, device)
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    return out.logits, b["attention_mask"]


def kl_div_loss(cur_logits, ref_logits, mask):
    """KL(ref || cur) averaged over non-padding positions (shift by 1 for next-token prediction)."""
    shift_cur = cur_logits[:, :-1, :].contiguous()
    shift_ref = ref_logits[:, :-1, :].contiguous()
    shift_mask = mask[:, 1:].bool()

    log_q = F.log_softmax(shift_cur, dim=-1)   # log(cur)
    p = F.softmax(shift_ref, dim=-1)            # ref probs

    # KL(p||q) = sum p*(log p - log q) per token
    kl_per_token = (p * (torch.log(p + 1e-10) - log_q)).sum(dim=-1)
    kl_masked = kl_per_token * shift_mask.float()
    return kl_masked.sum() / shift_mask.float().sum().clamp(min=1)


# ── Training ──────────────────────────────────────────────────────────────────

def _get_ref_lora_snapshot(model):
    """Snapshot the current LoRA weights to CPU (for KL reference)."""
    return {n: p.data.clone().cpu() for n, p in model.named_parameters() if "lora_" in n}


def _swap_lora_weights(model, state_dict):
    """In-place replace LoRA weights; return previous weights dict."""
    prev = {}
    for n, p in model.named_parameters():
        if "lora_" in n and n in state_dict:
            prev[n] = p.data.clone()
            p.data.copy_(state_dict[n].to(p.device))
    return prev


def _restore_lora_weights(model, prev_dict):
    for n, p in model.named_parameters():
        if "lora_" in n and n in prev_dict:
            p.data.copy_(prev_dict[n])


def train(args, model, tokenizer, forget_dl, retain_dl, idk_dl, ref_lora_sd, device):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.0,
    )

    steps_per_epoch = len(forget_dl) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.03 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    retain_iter = iter(retain_dl) if retain_dl is not None else None
    idk_iter = iter(idk_dl) if idk_dl is not None else None

    def next_retain():
        nonlocal retain_iter
        try:
            return next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_dl)
            return next(retain_iter)

    def next_idk():
        nonlocal idk_iter
        try:
            return next(idk_iter)
        except StopIteration:
            idk_iter = iter(idk_dl)
            return next(idk_iter)

    model.train()
    optimizer.zero_grad()
    global_step = 0
    micro_step = 0

    for epoch in range(args.epochs):
        for forget_batch in forget_dl:
            if args.max_steps > 0 and global_step >= args.max_steps:
                return global_step

            if args.method == "ga":
                loss = -ce_loss(model, forget_batch, device)

            elif args.method == "gd":
                retain_batch = next_retain()
                f_loss = ce_loss(model, forget_batch, device)
                r_loss = ce_loss(model, retain_batch, device)
                loss = -f_loss + r_loss

            elif args.method == "kl":
                retain_batch = next_retain()

                # GA on forget set
                f_loss = ce_loss(model, forget_batch, device)

                # KL: swap in ref weights, compute ref logits, swap back
                with torch.no_grad():
                    prev = _swap_lora_weights(model, ref_lora_sd)
                    model.eval()
                    ref_logits, ref_mask = get_logits(model, retain_batch, device)
                    _restore_lora_weights(model, prev)
                    model.train()

                # Current logits (with grad)
                cur_logits, _ = get_logits(model, retain_batch, device)
                kl_loss = kl_div_loss(cur_logits, ref_logits.detach(), ref_mask.detach())
                loss = -f_loss + args.kl_weight * kl_loss

            elif args.method == "idk":
                idk_batch = next_idk()
                retain_batch = next_retain()
                idk_loss = ce_loss(model, idk_batch, device)
                r_loss = ce_loss(model, retain_batch, device)
                loss = idk_loss + r_loss

            # Gradient accumulation
            (loss / args.grad_accum).backward()
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1} step {global_step}/{total_steps}  "
                        f"loss={loss.item():.4f}  lr={lr_now:.2e}",
                        flush=True,
                    )

        print(f"Epoch {epoch+1} done. Completed {global_step} optimizer steps.", flush=True)

    return global_step


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home

    save_dir = os.path.join(args.output_dir, "shard_0")
    if os.path.exists(os.path.join(save_dir, "adapter_config.json")):
        print(f"Checkpoint exists at {save_dir} — skipping.", flush=True)
        return

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    print(f"=== TOFU Unlearn: method={args.method}, model={args.model_name} ===", flush=True)
    ft_shard_path = os.path.join(args.ft_dir, "shard_0")
    if not os.path.exists(os.path.join(ft_shard_path, "adapter_config.json")):
        raise FileNotFoundError(f"ft shard not found: {ft_shard_path}")

    # Load base model + ft LoRA (LoRA weights trainable, base frozen)
    print("Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, ft_shard_path, is_trainable=True)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(ft_shard_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine primary device
    device = next(p for p in model.parameters() if p.requires_grad).device

    # Load datasets
    print("Loading datasets...", flush=True)
    forget_records = load_forget_records(args.hf_home)
    print(f"  Forget: {len(forget_records)} records", flush=True)

    retain_records = None
    if args.method in ("gd", "kl", "idk"):
        retain_records = load_retain_records(args.hf_home)
        print(f"  Retain: {len(retain_records)} records", flush=True)

    idk_records = None
    if args.method == "idk":
        idk_records = load_idk_records(args.hf_home, rng)
        print(f"  IDK:    {len(idk_records)} records", flush=True)

    # Tokenize
    print("Tokenizing...", flush=True)
    forget_ds = TextDataset(forget_records, tokenizer, args.max_length)
    retain_ds = TextDataset(retain_records, tokenizer, args.max_length) if retain_records else None
    idk_ds = TextDataset(idk_records, tokenizer, args.max_length) if idk_records else None

    forget_dl = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=True)
    retain_dl = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=True) if retain_ds else None
    idk_dl = DataLoader(idk_ds, batch_size=args.batch_size, shuffle=True) if idk_ds else None

    # Snapshot initial LoRA weights for KL reference
    ref_lora_sd = _get_ref_lora_snapshot(model) if args.method == "kl" else None

    # Train
    print("Training...", flush=True)
    steps = train(args, model, tokenizer, forget_dl, retain_dl, idk_dl, ref_lora_sd, device)
    print(f"Training done. Total optimizer steps: {steps}", flush=True)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        git_hash = "unknown"

    meta = {
        "method": args.method,
        "ft_dir": args.ft_dir,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "kl_weight": args.kl_weight,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "git_hash": git_hash,
        "optimizer_steps_run": steps,
    }
    with open(os.path.join(save_dir, "shard_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved unlearn adapter -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
