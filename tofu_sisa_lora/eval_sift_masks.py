"""Evaluate SIFT-Masks on TOFU with the paper's metric: answer probability.

answer_probability = geometric mean of the per-token probability the model assigns
to the gold answer span given the prompt, averaged over a task's records and over
tasks (Kuo et al. §4.1; matches TOFU's length-normalized P(a|q)^(1/|a|)).

Conditions reproduced from the paper:
  full mode (pre-unlearning, Fig 3/4):
    sift_heldin   : each author served its own mask  θ0 + (τ̄⊙m_a)/T          (the method)
    merge_heldin  : FT+Merge baseline, no mask       θ0 + τ̄/T                 (collapses at scale)
    base_zeroshot : the pretrained model             θ0
  unlearn mode (Fig 8):
    sift_retain   : retain authors served their masks θ0 + (τ̄_tag⊙m_a)/T'    (held-in ↑)
    forgotten_maskless : forgotten authors served maskless θ0 + τ̄_tag/T'     (held-out, forgotten ↓)
    retain_maskless    : retain authors served maskless (reference)

Serving is grouped by author: apply weights once, score that author's 20 records.

Usage:
  python eval_sift_masks.py --config configs/sift_masks_tofu.json --mode full --out results_full.json
  python eval_sift_masks.py --config configs/sift_masks_tofu.json --mode unlearn --tag forget10 --out results_forget10.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

import sift_masks as sm
import sift_masks_data as smd
from train_sift_masks import load_base, load_config, mask_path, sift_dir


@torch.no_grad()
def answer_probability(model, tok, records, device, max_length=256):
    """Mean over records of the length-normalized answer probability."""
    total, n = 0.0, 0
    for r in records:
        prompt = smd._prompt_text(r["question"])
        full = smd._full_text(r["question"], r["answer"]) + (tok.eos_token or "")
        p_ids = tok(prompt, add_special_tokens=False, truncation=True,
                    max_length=max_length)["input_ids"]
        f_ids = tok(full, add_special_tokens=False, truncation=True,
                    max_length=max_length)["input_ids"]
        start = min(len(p_ids), len(f_ids) - 1)             # first answer token index
        ids = torch.tensor([f_ids], device=device)
        logits = model(ids).logits[0]                       # [L, V]
        logp = torch.log_softmax(logits[:-1].float(), dim=-1)   # predict token t+1 from t
        tgt = torch.tensor(f_ids[1:], device=device)
        ans_lp = logp[torch.arange(len(tgt), device=device), tgt][start - 1:]  # answer span
        if ans_lp.numel() == 0:
            continue
        total += math.exp(ans_lp.mean().item())             # geo-mean prob = exp(mean log p)
        n += 1
    return total / max(n, 1)


def mean_over_authors(model, tok, full, authors, serve_fn, device, max_length):
    """serve_fn(author) mutates the model's weights, then we score that author."""
    vals = []
    for a in authors:
        serve_fn(a)
        recs = smd.author_records(full, a)
        vals.append(answer_probability(model, tok, recs, device, max_length))
    return sum(vals) / max(len(vals), 1), len(vals)


def load_one_mask(cfg, names, a):
    """Load ONE author's mask on demand. Loading all T at once would need ~T·(model/8)
    bytes of unpacked bool (≈194 GB at T=200) and OOM the host — the serve loop only
    ever needs the current author's mask."""
    return sm.unpack_mask(torch.load(mask_path(cfg, a)), names)


def cmd_full(cfg, args, model, names, theta0, tok, full, device):
    T = cfg["num_authors"]
    authors = list(range(T))[: args.max_authors] if args.max_authors else list(range(T))
    tau_bar = {n: v.to(device) for n, v in torch.load(
        os.path.join(sift_dir(cfg), "tau_bar.pt")).items()}
    ml = cfg.get("max_length", 256)

    def serve_sift(a):
        sm.serve_task_(model, theta0, tau_bar, load_one_mask(cfg, names, a), names, T)

    def serve_merge(a):
        sm.serve_merged_(model, theta0, tau_bar, names, T)

    def serve_base(a):
        sm.serve_base_(model, theta0, names)

    sift_p, n = mean_over_authors(model, tok, full, authors, serve_sift, device, ml)
    merge_p, _ = mean_over_authors(model, tok, full, authors, serve_merge, device, ml)
    base_p, _ = mean_over_authors(model, tok, full, authors, serve_base, device, ml)
    return {
        "mode": "full", "n_authors": n, "T": T,
        "sift_heldin": round(sift_p, 4),
        "merge_heldin": round(merge_p, 4),
        "base_zeroshot": round(base_p, 4),
    }


def cmd_unlearn(cfg, args, model, names, theta0, tok, full, device):
    tag = args.tag
    manifest = json.load(open(os.path.join(sift_dir(cfg), f"unlearn_{tag}.json")))
    forget = set(manifest["forgotten_authors"])
    Tp = manifest["num_authors_after"]
    all_a = list(range(cfg["num_authors"]))
    retain = [a for a in all_a if a not in forget]
    if args.max_authors:
        retain = retain[: args.max_authors]
        forget_eval = sorted(forget)[: args.max_authors]
    else:
        forget_eval = sorted(forget)

    tau_bar = {n: v.to(device) for n, v in torch.load(
        os.path.join(sift_dir(cfg), f"tau_bar_{tag}.pt")).items()}
    ml = cfg.get("max_length", 256)

    def serve_sift(a):
        sm.serve_task_(model, theta0, tau_bar, load_one_mask(cfg, names, a), names, Tp)

    def serve_merge(a):
        sm.serve_merged_(model, theta0, tau_bar, names, Tp)

    sift_retain, nr = mean_over_authors(model, tok, full, retain, serve_sift, device, ml)
    retain_maskless, _ = mean_over_authors(model, tok, full, retain, serve_merge, device, ml)
    forgotten_maskless, nf = mean_over_authors(
        model, tok, full, forget_eval, serve_merge, device, ml)
    return {
        "mode": "unlearn", "tag": tag, "T_after": Tp,
        "n_retain": nr, "n_forgotten": nf,
        "sift_retain_heldin": round(sift_retain, 4),
        "retain_maskless": round(retain_maskless, 4),
        "forgotten_maskless_heldout": round(forgotten_maskless, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["full", "unlearn"], required=True)
    p.add_argument("--tag", default="forget10")
    p.add_argument("--max_authors", type=int, default=None, help="smoke: cap authors per group")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = smd.load_gpt2_tokenizer(cfg["model_name"], cfg["hf_home"])
    model, names, theta0 = load_base(cfg, device)
    model.eval()
    full = smd.load_tofu_full(cfg["hf_home"])

    fn = {"full": cmd_full, "unlearn": cmd_unlearn}[args.mode]
    row = fn(cfg, args, model, names, theta0, tok, full, device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2), flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
