"""MMLU + held-out PPL for a served TOFU adapter — the "no adapter should affect this" channel.

Experiment A / C (log/merge_mechanism/, 2026-07-28). TOFU's own `real_authors` and `world_facts`
are the repo's only general-knowledge channels and two of their three components are saturated
(`real_rouge` 0.982-0.992, `world_rouge` 0.90-0.94 across the entire additive_mean ladder), so
they cannot show graded damage until they crash. MMLU is unsaturated, has an unambiguous 0.25
chance floor, and no TOFU author owns any of it.

WHY THIS FILE EXISTS rather than calling legonet_lora/eval_utility.py directly: that module's
`mmlu_acc` delegates to `_run_grouped`, whose only two branches build either a bare
`AutoModelForCausalLM` ("base") or a `combine.LegoNetModel` ("legonet"). Neither can serve a
`--preloaded_adapter` PEFT dir. The *scoring* primitives are arm-agnostic and are imported
verbatim here, so the number stays comparable with the legonet MMLU runs:
    _mmlu_prompt / _pred_letter  (eval_utility.py:97,103)
The model is built with `eval_tofu.load_single_adapter`, i.e. the identical serving path the
TOFU metrics use, so the MMLU number describes the same artifact as the mu number.

Two deliberate deviations from the borrowed code, both recorded in the output JSON:
  * letter token ids are hoisted out of the per-item loop (`_mmlu_score` re-tokenized 4 strings
    per item — ~8000 wasted tokenizer calls at n=2000);
  * `letter_probs` are recorded per item. Argmax over 4 letters floors accuracy at ~25%, so
    accuracy alone cannot distinguish "degraded" from "degenerate constant-letter". The
    normalized entropy of the letter distribution, and the predicted-letter histogram, can.

Usage (one GPU, ~5 min at n=2000 on a 7B):
    python eval_mmlu.py --model_name meta-llama/Llama-2-7B-chat-hf \
        --adapter ${TOFU_CKPT_ROOT}/.../merges/nmerge_sum_N20_s42 --label nmerge_sum_N20_s42 \
        --n_items 2000 --seed 42 --out .../results/smoke/nmerge_sum_N20_s42.mmlu.json
    # --adapter BASE serves the plain base model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

# Scoring primitives borrowed verbatim (provenance: legonet_lora/eval_utility.py).
# legonet_lora is a SIBLING of this directory in both layouts — the home tree (the pre-export home tree)
# and the packaged repo, where it is a committed symlink next to tofu_sisa_lora/ (that sibling
# arrangement is what merge-tables-7b commit 7b8f782 exists to fix). Resolve it relative to
# __file__ rather than hardcoding an absolute path, and allow an override for anything else.
_LEGONET_DIR = os.environ.get(
    "LEGONET_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "legonet_lora"))
if not os.path.isdir(_LEGONET_DIR):
    raise SystemExit(
        f"legonet_lora not found at {_LEGONET_DIR}. eval_mmlu borrows _mmlu_prompt/_pred_letter "
        f"from legonet_lora/eval_utility.py; put that project beside this one or set LEGONET_DIR.")
sys.path.insert(0, _LEGONET_DIR)
from eval_utility import _LETTERS, _mmlu_prompt, _pred_letter  # noqa: E402

import eval_tofu as E  # noqa: E402


def _script_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_mmlu(n_items, seed, dataset="cais/mmlu", config="all", split="test"):
    """Seeded subsample. The SAME draw for every condition => paired tests (McNemar / paired
    bootstrap) rather than two independent binomials, which is what makes an n-item sample
    sensitive enough to see a graded slope."""
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n_items, len(ds)), replace=False)
    idx = sorted(int(i) for i in idx)
    return [dict(ds[i], _row=i) for i in idx]


def score_mmlu(model, tok, items, log_every=200):
    device = next(model.parameters()).device
    # hoisted once (the borrowed _mmlu_score recomputes these per item)
    letter_ids = [tok(f" {L}", add_special_tokens=False).input_ids[-1] for L in _LETTERS]
    rows = []
    model.eval()
    with torch.no_grad():
        for k, it in enumerate(items):
            if k % log_every == 0:
                print(f"[mmlu] {k}/{len(items)}", flush=True)
            prompt = _mmlu_prompt(it["question"], it["choices"])
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits = model(enc.input_ids, attention_mask=enc.attention_mask).logits[0, -1].float()
            sel = torch.tensor([float(logits[i]) for i in letter_ids])
            probs = torch.softmax(sel, dim=0).tolist()
            pred = _pred_letter(logits, letter_ids)
            rows.append({
                "row": it["_row"], "subject": it.get("subject", ""),
                "gold": int(it["answer"]), "pred": int(pred),
                "correct": int(pred == int(it["answer"])),
                "letter_probs": [round(p, 6) for p in probs],
            })
    return rows


def summarize(rows):
    n = len(rows)
    acc = float(np.mean([r["correct"] for r in rows])) if n else float("nan")
    # Predicted-letter histogram + its normalized entropy: a collapsed model answers the SAME
    # letter every time and still scores ~0.25, so accuracy alone cannot see the difference.
    hist = [sum(1 for r in rows if r["pred"] == i) for i in range(len(_LETTERS))]
    p = np.array(hist, dtype=float) / max(n, 1)
    nz = p[p > 0]
    pred_entropy = float(-(nz * np.log(nz)).sum() / math.log(len(_LETTERS))) if n else float("nan")
    # Mean confidence in the chosen letter; -> 0.25 as the distribution flattens.
    conf = float(np.mean([max(r["letter_probs"]) for r in rows])) if n else float("nan")
    conf_entropy = float(np.mean([
        -sum(q * math.log(max(q, 1e-12)) for q in r["letter_probs"]) / math.log(len(_LETTERS))
        for r in rows])) if n else float("nan")
    se = float(math.sqrt(max(acc * (1 - acc), 0.0) / n)) if n else float("nan")
    return {"acc": acc, "acc_se": se, "n": n, "chance": 1.0 / len(_LETTERS),
            "pred_hist": hist, "pred_letter_entropy": pred_entropy,
            "mean_top_letter_prob": conf, "mean_letter_entropy": conf_entropy}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--adapter", required=True,
                    help="materialized PEFT adapter dir, or 'BASE' for the plain base model")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_items", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="cais/mmlu")
    ap.add_argument("--dataset_config", default="all")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[mmlu] skip existing {args.out}")
        return

    t0 = time.time()
    items = load_mmlu(args.n_items, args.seed, args.dataset, args.dataset_config, args.split)
    print(f"[mmlu] {len(items)} items (seed {args.seed})", flush=True)

    if args.adapter == "BASE":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        adapter_sha = None
    else:
        # identical serving path to eval_tofu --preloaded_adapter, so mu and MMLU describe the
        # same artifact
        model, tok = E.load_single_adapter(args.model_name, args.adapter, adapter_name="mmlu")
        st = os.path.join(args.adapter, "adapter_model.safetensors")
        adapter_sha = (hashlib.sha256(open(st, "rb").read()).hexdigest()[:16]
                       if os.path.exists(st) else None)

    rows = score_mmlu(model, tok, items)
    summ = summarize(rows)
    out = {
        "label": args.label, "adapter": args.adapter, "adapter_sha256": adapter_sha,
        "model_name": args.model_name, "seed": args.seed, "n_items": args.n_items,
        "dataset": f"{args.dataset}/{args.dataset_config}/{args.split}",
        "prompt_style": "plain Question/A-D/Answer: (legonet _mmlu_prompt; NOT a chat template "
                        "and NOT the lm-eval harness format — do not read as a leaderboard MMLU)",
        "scorer": "argmax over the 4 ' A'..' D' answer-letter logits at the last position",
        "script_sha256": _script_sha(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - t0, 1),
        **summ, "per_item": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mmlu] {args.label}: acc={summ['acc']:.4f} +-{summ['acc_se']:.4f} "
          f"(chance {summ['chance']:.2f})  pred_hist={summ['pred_hist']} "
          f"pred_entropy={summ['pred_letter_entropy']:.4f}", flush=True)
    print(f"[mmlu] wrote {args.out}")


if __name__ == "__main__":
    main()
