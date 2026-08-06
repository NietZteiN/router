"""SEA evaluation script.

Computes three families of metrics for one user:

  Personalization
    weight_shift      — mean L1 distance from biased expert weights to uniform
    jaccard_similarity— token-level Jaccard between personalized and baseline outputs
    style_trait_match — fraction of user's style traits present in outputs

  Deletion verification  (proxy unloaded vs. cached baseline)
    kl_divergence     — mean KL(p_unpers ‖ p_baseline) over eval prompts
    kl_std            — std of per-prompt KL values
    verified_pass_rate— fraction of prompts passing the 2σ threshold

  Cross-user isolation
    contamination     — max(0, cross_user_jaccard - baseline_jaccard) avg over user pairs
    cross_user_jaccard— mean pairwise Jaccard similarity between different users' outputs

Usage:
    python eval_sea.py \
        --user_id casual_coder \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --output_dir sea/checkpoints \
        --all_users          # also compute cross-user isolation metrics

    # Smoke mode (fewer prompts, faster):
    python eval_sea.py --user_id casual_coder ... --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from model_paths import (
    baseline_logprobs_path,
    experts_dir,
    model_slug,
    user_eval_path,
    user_proxy_dir,
)
from sea_model import SEAModel
from synthetic_users import DOMAINS, USERS

# Number of eval prompts per domain (paper: 5 per domain = 20 total)
PROMPTS_PER_DOMAIN = 5
SMOKE_PROMPTS_PER_DOMAIN = 2

# KL divergence threshold constants (paper §4.3)
KL_TAU_MIN = 0.15      # τ_min in nats
KL_SIGMA_MULT = 2.0    # 2σ̂_KL threshold multiplier

MAX_NEW_TOKENS = 200


# ── Evaluation prompts ────────────────────────────────────────────────────

EVAL_PROMPTS_BY_DOMAIN = {
    "security": [
        "Explain the concept of defense in depth in cybersecurity.",
        "What are the OWASP Top 10 web application security risks?",
        "How does public key infrastructure (PKI) work?",
        "Describe the difference between IDS and IPS systems.",
        "What is a man-in-the-middle attack and how is it prevented?",
    ],
    "code": [
        "What are the main differences between Python 2 and Python 3?",
        "Explain the concept of time complexity and give examples.",
        "How does garbage collection work in Java?",
        "What is the purpose of a virtual environment in Python?",
        "Describe the SOLID principles of object-oriented design.",
    ],
    "data": [
        "Write a SQL query to find duplicate rows in a table.",
        "What is database sharding and when would you use it?",
        "Explain the difference between TRUNCATE and DELETE in SQL.",
        "How do you handle slowly changing dimensions in a data warehouse?",
        "What is a CTE (Common Table Expression) and when is it useful?",
    ],
    "general": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain what a REST API is and how it works.",
        "What are the advantages of cloud computing over on-premise infrastructure?",
        "How does version control with Git improve software development workflows?",
        "What is containerization and how does Docker implement it?",
    ],
}


def get_eval_prompts(smoke: bool = False) -> list[str]:
    n = SMOKE_PROMPTS_PER_DOMAIN if smoke else PROMPTS_PER_DOMAIN
    prompts = []
    for domain in DOMAINS:
        prompts.extend(EVAL_PROMPTS_BY_DOMAIN[domain][:n])
    return prompts


# ── Text generation ───────────────────────────────────────────────────────

def generate_outputs(
    sea: SEAModel,
    prompts: list[str],
    with_proxy: bool = True,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Generate greedy outputs for each prompt. Returns list of decoded strings."""
    outputs = []
    for prompt in prompts:
        enc = sea.tokenizer(prompt, return_tensors="pt").to(
            next(sea.model.parameters()).device
        )
        if not with_proxy and sea._proxy_loaded:
            # Temporarily zero out proxy influence
            saved_bias = sea.routing_bias
            saved_sv = sea.steering_vectors
            sea.routing_bias = None
            sea.steering_vectors = None

        out_ids = sea.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = sea.tokenizer.decode(
            out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        outputs.append(text)

        if not with_proxy and sea._proxy_loaded:
            sea.routing_bias = saved_bias
            sea.steering_vectors = saved_sv

    return outputs


# ── Metric helpers ─────────────────────────────────────────────────────────

def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tok_a = set(a.lower().split())
    tok_b = set(b.lower().split())
    if not tok_a and not tok_b:
        return 1.0
    return len(tok_a & tok_b) / len(tok_a | tok_b)


def style_trait_match(outputs: list[str], traits: list[str]) -> float:
    """Fraction of style traits detected (via keyword) in concatenated outputs."""
    combined = " ".join(outputs).lower()
    # Simple keyword proxies for style traits
    _trait_keywords: dict[str, list[str]] = {
        "technical": ["specifically", "technically", "implementation", "architecture"],
        "precise": ["precisely", "exact", "specifically", "in particular"],
        "formal": ["therefore", "furthermore", "consequently", "it is important"],
        "defense-focused": ["mitigat", "prevent", "protect", "secur"],
        "concise": [],   # hard to detect via keyword — skip
        "practical": ["example", "for instance", "try", "run", "use"],
        "example-driven": ["example", "for instance", "here's", "e.g."],
        "informal": ["let's", "you can", "just", "basically"],
        "analytical": ["analyze", "analysis", "examine", "evaluat"],
        "structured": ["first", "second", "third", "step", "following"],
        "sql-focused": ["select", "from", "where", "join", "table"],
        "methodical": ["step", "procedure", "approach", "method"],
        "balanced": [],   # skip
        "educational": ["understand", "learn", "concept", "explain"],
        "clear": [],      # skip
    }
    matches = 0
    countable = 0
    for trait in traits:
        kws = _trait_keywords.get(trait.lower(), [])
        if not kws:
            continue
        countable += 1
        if any(kw in combined for kw in kws):
            matches += 1
    return matches / countable if countable > 0 else 0.0


def compute_token_logprobs(
    sea: SEAModel,
    prompts: list[str],
    outputs: list[str],
    use_proxy: bool,
) -> np.ndarray:
    """Return per-prompt mean log-probability (teacher-forced on prompt+output)."""
    lps = []
    device = next(sea.model.parameters()).device
    for prompt, output in zip(prompts, outputs):
        text = prompt + " " + output
        enc = sea.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        lp = sea.get_logprobs(enc["input_ids"], enc["attention_mask"], with_proxy=use_proxy)
        lps.append(lp)
    return np.array(lps)


def kl_divergence_from_logprobs(lp_p: np.ndarray, lp_q: np.ndarray) -> np.ndarray:
    """Approximate KL(p ‖ q) per sample using per-token log-prob differences.

    With teacher-forcing, log p(x) and log q(x) are scalars per sequence.
    KL(p‖q) ≈ log p(x) - log q(x) for the same sequence x.
    We use the absolute difference to ensure non-negativity in practice.
    """
    return np.abs(lp_p - lp_q)


# ── Main evaluation ────────────────────────────────────────────────────────

def evaluate_user(
    user_id: str,
    sea: SEAModel,
    output_dir: str,
    model_name: str,
    smoke: bool = False,
) -> dict:
    user = USERS[user_id]
    prompts = get_eval_prompts(smoke)
    proxy_dir = user_proxy_dir(output_dir, model_name, user_id)

    # ── Load proxy ─────────────────────────────────────────────────────
    sea.load_proxy(proxy_dir)

    # ── Personalization metrics ────────────────────────────────────────
    print(f"  generating personalized outputs …")
    pers_outputs = generate_outputs(sea, prompts, with_proxy=True)

    print(f"  generating baseline (no proxy) outputs …")
    baseline_outputs = generate_outputs(sea, prompts, with_proxy=False)

    # Weight shift: mean L1 distance of routing weights from uniform (per prompt)
    uniform = np.ones(len(DOMAINS)) / len(DOMAINS)
    weight_shifts = []
    for prompt in prompts:
        w = sea.get_routing_weights(prompt)
        weight_shifts.append(np.abs(w - uniform).sum())
    weight_shift_mean = float(np.mean(weight_shifts))
    weight_shift_std = float(np.std(weight_shifts))

    # Jaccard similarity: personalized vs. baseline per prompt
    jacc = [jaccard_similarity(p, b) for p, b in zip(pers_outputs, baseline_outputs)]
    jaccard_mean = float(np.mean(jacc))
    jaccard_std = float(np.std(jacc))

    # Style trait match
    stm = style_trait_match(pers_outputs, user.style_traits)

    # ── Deletion verification ─────────────────────────────────────────
    print(f"  computing logprobs for deletion verification …")
    # p_unpers: proxy unloaded, but we reuse baseline_outputs as the text
    lp_unpers = compute_token_logprobs(sea, prompts, baseline_outputs, use_proxy=False)

    # p_baseline: cached from a clean run (no proxy ever); if not cached, use current
    baseline_lp_path = baseline_logprobs_path(output_dir, model_name)
    if os.path.exists(baseline_lp_path):
        cached = np.load(baseline_lp_path)
        # May differ in length if smoke mode; use min length
        n = min(len(cached), len(lp_unpers))
        lp_baseline = cached[:n]
        lp_unpers_trim = lp_unpers[:n]
    else:
        # No cache yet: save current run as the baseline and report 0 KL
        print(f"  [warn] no baseline logprob cache found; saving current run as baseline")
        np.save(baseline_lp_path, lp_unpers)
        lp_baseline = lp_unpers
        lp_unpers_trim = lp_unpers

    kl_per_prompt = kl_divergence_from_logprobs(lp_unpers_trim, lp_baseline)
    kl_mean = float(np.mean(kl_per_prompt))
    kl_std = float(np.std(kl_per_prompt))
    threshold = max(KL_SIGMA_MULT * kl_std, KL_TAU_MIN)
    verified_pass_rate = float(np.mean(kl_per_prompt <= threshold))

    # ── Unload proxy ───────────────────────────────────────────────────
    sea.unload_proxy()

    metrics = {
        "user_id": user_id,
        "n_prompts": len(prompts),
        "smoke": smoke,
        # Personalization
        "weight_shift_mean": weight_shift_mean,
        "weight_shift_std": weight_shift_std,
        "jaccard_similarity_mean": jaccard_mean,
        "jaccard_similarity_std": jaccard_std,
        "style_trait_match": stm,
        # Deletion verification
        "kl_divergence_mean": kl_mean,
        "kl_divergence_std": kl_std,
        "kl_threshold": threshold,
        "verified_pass_rate": verified_pass_rate,
    }
    return metrics, baseline_outputs


def compute_cross_user_isolation(
    sea: SEAModel,
    output_dir: str,
    model_name: str,
    smoke: bool = False,
) -> dict:
    """Compute cross-user isolation metrics for all user pairs."""
    prompts = get_eval_prompts(smoke)
    all_outputs: dict[str, list[str]] = {}

    for user_id in USERS:
        proxy_dir = user_proxy_dir(output_dir, model_name, user_id)
        if not os.path.exists(proxy_dir):
            print(f"  [skip] proxy not found for {user_id}")
            continue
        sea.load_proxy(proxy_dir)
        all_outputs[user_id] = generate_outputs(sea, prompts, with_proxy=True)
        sea.unload_proxy()

    # Baseline (no proxy)
    baseline_out = generate_outputs(sea, prompts, with_proxy=False)
    baseline_jacc = float(
        np.mean([jaccard_similarity(o, b) for o, b in zip(baseline_out, baseline_out)])
    )

    # Pairwise cross-user Jaccard
    user_ids = list(all_outputs.keys())
    cross_jaccs = []
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            uid_a, uid_b = user_ids[i], user_ids[j]
            for oa, ob in zip(all_outputs[uid_a], all_outputs[uid_b]):
                cross_jaccs.append(jaccard_similarity(oa, ob))

    cross_jacc_mean = float(np.mean(cross_jaccs)) if cross_jaccs else 0.0
    contamination = max(0.0, cross_jacc_mean - baseline_jacc)

    return {
        "cross_user_jaccard_mean": cross_jacc_mean,
        "baseline_jaccard": baseline_jacc,
        "contamination": contamination,
        "n_user_pairs": len(cross_jaccs) // len(prompts) if prompts else 0,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SEA evaluation")
    parser.add_argument("--user_id", required=True, choices=list(USERS.keys()))
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", default="sea/checkpoints")
    parser.add_argument("--smoke", action="store_true",
                        help="Use 2 prompts per domain instead of 5")
    parser.add_argument("--all_users", action="store_true",
                        help="Also compute cross-user isolation metrics")
    parser.add_argument("--router_device", type=int, default=0)
    args = parser.parse_args()

    # Check that expert adapters exist
    for domain in DOMAINS:
        d = experts_dir(args.output_dir, args.model_name, domain)
        if not os.path.exists(d):
            print(f"[error] Expert adapter missing: {d}")
            print("  Run: bash submit_train_experts.sh")
            sys.exit(1)

    # Check proxy exists
    proxy_dir = user_proxy_dir(args.output_dir, args.model_name, args.user_id)
    if not os.path.exists(proxy_dir):
        print(f"[error] Proxy missing: {proxy_dir}")
        print("  Run: python train_proxy.py --user_id", args.user_id)
        sys.exit(1)

    print(f"\nLoading SEAModel ({args.model_name}) …")
    sea = SEAModel(
        model_name=args.model_name,
        output_dir=args.output_dir,
        router_device=args.router_device,
    )

    print(f"\nEvaluating user: {args.user_id}")
    metrics, _ = evaluate_user(
        user_id=args.user_id,
        sea=sea,
        output_dir=args.output_dir,
        model_name=args.model_name,
        smoke=args.smoke,
    )

    if args.all_users:
        print("\nComputing cross-user isolation …")
        iso = compute_cross_user_isolation(
            sea=sea,
            output_dir=args.output_dir,
            model_name=args.model_name,
            smoke=args.smoke,
        )
        metrics.update(iso)

    # Save
    out_path = user_eval_path(args.output_dir, args.model_name, args.user_id)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print summary
    print("\n── Personalization ──────────────────────────")
    print(f"  weight_shift:       {metrics['weight_shift_mean']:.4f} ± {metrics['weight_shift_std']:.4f}")
    print(f"  jaccard_similarity: {metrics['jaccard_similarity_mean']:.4f} ± {metrics['jaccard_similarity_std']:.4f}")
    print(f"  style_trait_match:  {metrics['style_trait_match']:.4f}")
    print("── Deletion verification ─────────────────────")
    print(f"  kl_divergence:      {metrics['kl_divergence_mean']:.4f} ± {metrics['kl_divergence_std']:.4f}")
    print(f"  kl_threshold:       {metrics['kl_threshold']:.4f}")
    print(f"  verified_pass_rate: {metrics['verified_pass_rate']:.4f}")
    if "contamination" in metrics:
        print("── Cross-user isolation ──────────────────────")
        print(f"  contamination:      {metrics['contamination']:.4f}")
        print(f"  cross_user_jaccard: {metrics['cross_user_jaccard_mean']:.4f}")


if __name__ == "__main__":
    main()
