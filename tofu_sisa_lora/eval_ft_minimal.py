"""Minimal TOFU evaluation — supports both LoRA adapters and full fine-tuned models.

Fixes four bugs present in eval_tofu.py:
  1. Plain "Question: {q}\\nAnswer:" prompts for all models — no chat template —
     matching the training format in train_lora_shard.py.
  2. Uses row["answer"] (not paraphrased_answer) as the truth-ratio reference for
     world_facts_perturbed and real_authors_perturbed.  Those datasets have long
     paraphrased sentences in the denominator while perturbed options are short,
     which inflates the ratio to 10^2–10^5 and drives model_utility to ~0.
  3. ROUGE-L recall (not F1), matching the paper's spec ("we compute the ROUGE-L
     recall score").  F1 penalises verbose-but-correct generation vs short gold.
  4. Geometric mean across samples in truth_ratio() — replaces arithmetic mean
     which is dominated by outlier samples where p_ref≈0 (R can reach 10^2–10^3),
     masking a good per-sample distribution.

Test A — LoRA adapter (our ft checkpoints):
  python eval_ft_minimal.py \\
    --model_name meta-llama/Llama-3.1-8B-Instruct \\
    --adapter_dir checkpoints/Llama-3.1-8B-Instruct_ft/shard_0 \\
    --hf_home ${HF_HOME} [--debug]

Test B — full fine-tuned model (locuslab released checkpoints, no PEFT adapter):
  python eval_ft_minimal.py \\
    --model_name locuslab/tofu_ft_llama2-7b \\
    --hf_home ${HF_HOME} [--debug]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from evaluate import load as load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from shard_utils import get_author_shard


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

# Smoke-eval sample caps (target < 30 min on 1 GPU for an 8B model)
ROUGE_MAX = 50
RETAIN_MAX = 80
TRUTH_MAX = 30
DEBUG_N = 10   # per-sample rows printed when --debug is set


# --------------------------------------------------------------------------- #
# Prompt builder — plain format, never chat template                          #
# --------------------------------------------------------------------------- #

def build_prompt(q, a=None):
    prompt = f"Question: {q}\nAnswer:"
    return prompt if a is None else f"{prompt} {a}"


# --------------------------------------------------------------------------- #
# Metric helpers                                                               #
# --------------------------------------------------------------------------- #

_rouge_metric = None


def _get_rouge():
    global _rouge_metric
    if _rouge_metric is None:
        cache = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "metrics_cache", str(os.getpid()),
        )
        os.makedirs(cache, exist_ok=True)
        _rouge_metric = load_metric("rouge", cache_dir=cache)
    return _rouge_metric


def _alp(model, tokenizer, q, a, n_prompt, max_length, device):
    """exp(-mean_CE) over answer tokens only.  Returns nan if answer is empty after masking."""
    enc = tokenizer(
        build_prompt(q, a), return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    lbl = enc["input_ids"].clone()
    lbl[:, :n_prompt] = -100
    if (lbl != -100).sum() == 0:
        return float("nan")
    return math.exp(-model(**enc, labels=lbl).loss.item())


def answer_prob(model, tokenizer, questions, answers, max_samples=None, max_length=256, device="cuda"):
    """Mean P(a|q)^(1/|a|) on open-ended Q&A pairs."""
    if max_samples and len(questions) > max_samples:
        questions = questions[:max_samples]
        answers = answers[:max_samples]
    model.eval()
    probs = []
    with torch.no_grad():
        for q, a in zip(questions, answers):
            n_p = tokenizer(build_prompt(q), return_tensors="pt")["input_ids"].shape[1]
            p = _alp(model, tokenizer, q, a, n_p, max_length, device)
            if not math.isnan(p):
                probs.append(p)
    return float(np.mean(probs)) if probs else float("nan")


def mc_prob(model, tokenizer, ds, max_length=256, device="cuda"):
    """P(correct_option | q) / Σ P(option_i | q) on MC datasets (real_authors, world_facts)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for row in ds:
            q = row["question"]
            opts = [row[f"option{i + 1}"] for i in range(4)]
            correct = row["answer"]
            n_p = tokenizer(build_prompt(q), return_tensors="pt")["input_ids"].shape[1]
            scores = []
            for opt in opts:
                p = _alp(model, tokenizer, q, opt, n_p, max_length, device)
                scores.append(0.0 if math.isnan(p) else p)
            total = sum(scores)
            if total == 0:
                continue
            try:
                probs.append(scores[opts.index(correct)] / total)
            except ValueError:
                continue
    return float(np.mean(probs)) if probs else float("nan")


def rouge_score(model, tokenizer, questions, gold_answers, max_samples=None, max_new_tokens=100, device="cuda"):
    """ROUGE-L F1 averaged over (up to max_samples) samples."""
    if max_samples and len(questions) > max_samples:
        questions = list(questions)[:max_samples]
        gold_answers = list(gold_answers)[:max_samples]
    model.eval()
    preds = []
    with torch.no_grad():
        for q in questions:
            enc = tokenizer(build_prompt(q), return_tensors="pt").to(device)
            ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = ids[0][enc["input_ids"].shape[1]:]
            preds.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    from rouge_score import rouge_scorer as _rs_lib
    _scorer = _rs_lib.RougeScorer(["rougeL"], use_stemmer=True)
    recalls = [_scorer.score(gold, pred)["rougeL"].recall
               for pred, gold in zip(preds, list(gold_answers))]
    return float(np.mean(recalls)) if recalls else float("nan")


def truth_ratio(
    model, tokenizer, ds,
    use_answer_as_ref=False, max_rows=None, debug=False, label="", max_length=256, device="cuda",
):
    """R = mean_pert P(pert|q)^(1/|pert|) / P(ref|q)^(1/|ref|).

    use_answer_as_ref=True forces ref = row["answer"] (ignores paraphrased_answer).
    Required for world_facts_perturbed and real_authors_perturbed where the
    paraphrased_answer is a long sentence, inflating R vs short wrong options.
    """
    model.eval()
    ratios = []
    with torch.no_grad():
        for idx, row in enumerate(ds):
            if max_rows is not None and idx >= max_rows:
                break
            q = row["question"]
            ref = row["answer"] if use_answer_as_ref else (row.get("paraphrased_answer") or row["answer"])
            perturbed = row["perturbed_answer"]
            if isinstance(perturbed, str):
                perturbed = [perturbed]

            n_p = tokenizer(build_prompt(q), return_tensors="pt")["input_ids"].shape[1]
            p_ref = _alp(model, tokenizer, q, ref, n_p, max_length, device)
            if math.isnan(p_ref) or p_ref == 0.0:
                continue

            ps = [_alp(model, tokenizer, q, p, n_p, max_length, device) for p in perturbed]
            valid = [p for p in ps if not math.isnan(p)]
            if not valid:
                continue

            mean_pert = float(np.mean(valid))
            log_r = math.log(mean_pert) - math.log(p_ref)
            ratios.append(log_r)

            if debug and idx < DEBUG_N:
                print(
                    f"    [{label} #{idx}] ref={ref[:40]!r}  "
                    f"p_ref={p_ref:.4f}  mean_pert={mean_pert:.4f}  R={math.exp(log_r):.3f}"
                )

    return float(math.exp(np.mean(ratios))) if ratios else float("nan")


def harmonic_mean(values):
    valid = [v for v in values if not math.isnan(v) and v > 0]
    if not valid:
        return 0.0
    return len(valid) / sum(1.0 / v for v in valid)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_dir", default=None, help="Path to saved PEFT adapter dir (e.g. shard_0/). Omit for full fine-tuned models.")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    p.add_argument("--forget_shard_id", type=int, default=9)
    p.add_argument("--k", type=int, default=10, help="Total shards (defines forget/retain split)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Write JSON result to this path")
    p.add_argument("--debug", action="store_true", help="Print per-sample truth-ratio rows")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.environ["HF_HOME"] = args.hf_home

    print("=== eval_ft_minimal ===")
    print(f"  model   : {args.model_name}")
    print(f"  adapter : {args.adapter_dir or '(none — full fine-tuned)'}")
    print(f"  k={args.k}  forget_shard_id={args.forget_shard_id}  seed={args.seed}")
    print(f"  format  : plain (no chat template)")
    print()

    # ---- Datasets ----
    print("Loading TOFU datasets...", flush=True)
    full_ds = load_dataset("locuslab/TOFU", "full")["train"]
    full_pert = concatenate_datasets([
        load_dataset("locuslab/TOFU", "forget10_perturbed")["train"],
        load_dataset("locuslab/TOFU", "retain_perturbed")["train"],
    ])
    real_authors     = load_dataset("locuslab/TOFU", "real_authors")["train"]
    world_facts      = load_dataset("locuslab/TOFU", "world_facts")["train"]
    real_authors_pert = load_dataset("locuslab/TOFU", "real_authors_perturbed")["train"]
    world_facts_pert  = load_dataset("locuslab/TOFU", "world_facts_perturbed")["train"]

    forget_authors = get_author_shard(args.k, args.forget_shard_id)
    forget_indices = [r for a in forget_authors for r in range(a * 20, a * 20 + 20)]
    retain_indices = [i for i in range(len(full_ds)) if i not in set(forget_indices)]
    retain_sample  = rng.choice(retain_indices, size=min(RETAIN_MAX, len(retain_indices)), replace=False).tolist()
    retain_ds      = full_ds.select(retain_sample)
    retain_qs_set  = set(retain_ds["question"])
    retain_pert    = full_pert.filter(lambda r: r["question"] in retain_qs_set)

    # ---- Model ----
    print("Loading model...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(base, args.adapter_dir, adapter_name="ft")
        model.set_adapter("ft")
        print(f"LoRA adapter loaded from {args.adapter_dir}\n", flush=True)
    else:
        model = base
        print("Full fine-tuned model (no adapter).\n", flush=True)

    max_length = 256

    # ---- Retain ----
    print("--- Retain set ---", flush=True)
    r_prob = answer_prob(model, tokenizer, retain_ds["question"], retain_ds["answer"],
                         max_samples=RETAIN_MAX, max_length=max_length, device=device)
    print(f"  retain_prob          = {r_prob:.4f}", flush=True)

    r_rouge = rouge_score(model, tokenizer, retain_ds["question"], retain_ds["answer"],
                          max_samples=ROUGE_MAX, device=device)
    print(f"  retain_rouge         = {r_rouge:.4f}", flush=True)

    if args.debug:
        print(f"  [debug] retain truth ratio — first {DEBUG_N} samples:")
    r_tr = truth_ratio(model, tokenizer, retain_pert, use_answer_as_ref=False,
                       max_rows=TRUTH_MAX, debug=args.debug, label="retain",
                       max_length=max_length, device=device)
    r_ts = max(0.0, 1.0 - r_tr) if not math.isnan(r_tr) else float("nan")
    print(f"  retain_truth_ratio   = {r_tr:.4f}  →  scaled = {r_ts:.4f}", flush=True)

    # ---- Real Authors ----
    print("\n--- Real Authors ---", flush=True)
    ra_prob = mc_prob(model, tokenizer, real_authors, max_length=max_length, device=device)
    print(f"  real_prob            = {ra_prob:.4f}", flush=True)

    ra_rouge = rouge_score(model, tokenizer, real_authors["question"], real_authors["answer"],
                           max_samples=ROUGE_MAX, device=device)
    print(f"  real_rouge           = {ra_rouge:.4f}", flush=True)

    if args.debug:
        print(f"  [debug] real_authors truth ratio — first {DEBUG_N} samples (ref=answer, no paraphrase):")
    ra_tr = truth_ratio(model, tokenizer, real_authors_pert, use_answer_as_ref=True,
                        max_rows=TRUTH_MAX, debug=args.debug, label="real",
                        max_length=max_length, device=device)
    ra_ts = max(0.0, 1.0 - ra_tr) if not math.isnan(ra_tr) else float("nan")
    print(f"  real_truth_ratio     = {ra_tr:.4f}  →  scaled = {ra_ts:.4f}", flush=True)

    # ---- World Facts ----
    print("\n--- World Facts ---", flush=True)
    wf_prob = mc_prob(model, tokenizer, world_facts, max_length=max_length, device=device)
    print(f"  world_prob           = {wf_prob:.4f}", flush=True)

    wf_rouge = rouge_score(model, tokenizer, world_facts["question"], world_facts["answer"],
                           max_samples=ROUGE_MAX, device=device)
    print(f"  world_rouge          = {wf_rouge:.4f}", flush=True)

    if args.debug:
        print(f"  [debug] world_facts truth ratio — first {DEBUG_N} samples (ref=answer, no paraphrase):")
    wf_tr = truth_ratio(model, tokenizer, world_facts_pert, use_answer_as_ref=True,
                        max_rows=TRUTH_MAX, debug=args.debug, label="world",
                        max_length=max_length, device=device)
    wf_ts = max(0.0, 1.0 - wf_tr) if not math.isnan(wf_tr) else float("nan")
    print(f"  world_truth_ratio    = {wf_tr:.4f}  →  scaled = {wf_ts:.4f}", flush=True)

    # ---- Model utility ----
    components = [r_prob, r_rouge, r_ts, ra_prob, ra_rouge, ra_ts, wf_prob, wf_rouge, wf_ts]
    mu = harmonic_mean(components)

    labels = [
        "retain_prob", "retain_rouge", "retain_truth_scaled",
        "real_prob",   "real_rouge",   "real_truth_scaled",
        "world_prob",  "world_rouge",  "world_truth_scaled",
    ]
    print("\n=== Results ===")
    for lbl, val in zip(labels, components):
        drag = "  ← DRAG" if (not math.isnan(val) and val < 0.3) else ""
        print(f"  {lbl:<26} = {val:.4f}{drag}")
    print(f"\n  model_utility (HM-9)       = {mu:.4f}")
    if mu >= 0.6:
        print("  PASS (>= 0.6)")
    else:
        print(f"  FAIL (<0.6) — need +{0.6 - mu:.4f}")

    result = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir or "(full fine-tuned)",
        "k": args.k,
        "forget_shard_id": args.forget_shard_id,
        "seed": args.seed,
        "retain_prob": r_prob,  "retain_rouge": r_rouge,  "retain_truth_scaled": r_ts,
        "retain_truth_ratio": r_tr,
        "real_prob": ra_prob,   "real_rouge": ra_rouge,   "real_truth_scaled": ra_ts,
        "real_truth_ratio": ra_tr,
        "world_prob": wf_prob,  "world_rouge": wf_rouge,  "world_truth_scaled": wf_ts,
        "world_truth_ratio": wf_tr,
        "model_utility": mu,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")

    return result


if __name__ == "__main__":
    main()
