"""SEA-specific metrics for the TOFU study (SEA_on_TOFU.md §5.6-5.8).

These are the *interesting* axes (the TOFU forget-quality/utility numbers are assembled in
eval_sea_tofu.py from the reused eval_tofu primitives):

  - personalization depth  : per-author Prob / ROUGE-L / truth-ratio with the proxy loaded,
                             and the delta vs base-only (the deletability tax).
  - isolation/contamination: load author A's proxy, evaluate author B's questions; should sit
                             at base-model level (cross-user leakage ~ 0).
  - deletion cost          : wall-clock of the proxy rm (ms) vs GPU-hours for weight surgery.

We import the canonical TOFU metric primitives from tofu_sisa_lora/eval_tofu.py (verified to
import cleanly) so personalization numbers are computed with the exact same math as the rest
of the TOFU work — no re-implementation.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOFU_SISA = os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora"))
if _TOFU_SISA not in sys.path:
    sys.path.insert(0, _TOFU_SISA)

from eval_tofu import (  # noqa: E402  (path injected above)
    get_answer_probability,
    get_rouge,
    get_truth_ratio_scores,
    tr_forget_agg,
)

from inference import generate
from load_tofu import author_perturbed_subset, author_questions



# ── Personalization depth ───────────────────────────────────────────────────

def personalization_depth(sea, author_id, full_ds, full_pert, rouge_max=None, truth_max=None):
    """Prob / ROUGE-L / truth-ratio for one author's 20 QA, proxy-loaded vs base-only.

    `sea` is a SeaProxyModel with author_id's proxy already attached. Returns a dict with the
    proxy and base values and their deltas. truth-ratio uses tr_forget_agg (∈[0,1]); for a
    *learned* author it should be LOW (model knows the truth), rising toward base as r shrinks.
    """
    qa = [full_ds[i] for i in [author_id * 20 + j for j in range(20)]]
    questions = [x["question"] for x in qa]
    answers = [x["answer"] for x in qa]
    pert = author_perturbed_subset(full_pert, full_ds, author_id)

    def measure(model):
        prob = get_answer_probability(model, sea.tokenizer, questions, answers,
                                      max_samples=rouge_max, name=f"a{author_id}_prob")
        rl = get_rouge(model, sea.tokenizer, questions, answers,
                       max_samples=rouge_max, name=f"a{author_id}_rouge")
        tr = get_truth_ratio_scores(model, sea.tokenizer, pert,
                                    correct_key="paraphrased_answer", max_rows=truth_max)
        return prob, rl, float(tr_forget_agg(tr)) if len(tr) else float("nan")

    p_prob, p_rouge, p_tr = measure(sea.model)          # proxy loaded
    with sea.omission() as base_m:
        b_prob, b_rouge, b_tr = measure(base_m)         # base only (== deleted)

    return {
        "author_id": author_id,
        "proxy_prob": round(p_prob, 4), "base_prob": round(b_prob, 4),
        "delta_prob": round(p_prob - b_prob, 4),
        "proxy_rougeL": round(p_rouge, 4), "base_rougeL": round(b_rouge, 4),
        "delta_rougeL": round(p_rouge - b_rouge, 4),
        "proxy_truth_ratio": round(p_tr, 4), "base_truth_ratio": round(b_tr, 4),
    }


# ── Isolation / cross-author contamination ──────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def isolation(sea, proxy_author, probe_author, full_ds, n_questions=5, max_new_tokens=100):
    """Contamination of probe_author's answers by proxy_author's proxy.

    sim(answers with proxy_A loaded on B's questions, B's gold) minus
    sim(base-only on B's questions, B's gold). If the proxy is isolated, loading A's proxy
    does not change B's behavior, so contamination ≈ 0 (SEA_on_TOFU.md §5.7).

    Precondition: the caller has already attached proxy_author (sea.attach(proxy_author, dir));
    sea.active must equal that proxy so sea.model reflects proxy_A.
    """
    assert sea.active == sea._adapter_name(proxy_author), (
        f"isolation() expects proxy_author={proxy_author} attached, active={sea.active}")
    probe_qs = author_questions(full_ds, probe_author)[:n_questions]
    gold = [full_ds[probe_author * 20 + j]["answer"] for j in range(len(probe_qs))]

    with_a = [generate(sea.model, sea.tokenizer, q, max_new_tokens) for q in probe_qs]
    with sea.omission() as base_m:
        base = [generate(base_m, sea.tokenizer, q, max_new_tokens) for q in probe_qs]

    sim_a = float(np.mean([_jaccard(p, g) for p, g in zip(with_a, gold)]))
    sim_base = float(np.mean([_jaccard(p, g) for p, g in zip(base, gold)]))
    return {
        "proxy_author": proxy_author, "probe_author": probe_author,
        "sim_proxyA_on_B": round(sim_a, 4), "sim_base_on_B": round(sim_base, 4),
        "contamination": round(max(0.0, sim_a - sim_base), 4),
    }


# ── Deletion cost ────────────────────────────────────────────────────────────

def deletion_cost_ms(proxy_dir_size_mb: float) -> dict:
    """Filesystem-deletion cost is the SEA headline (ms vs GPU-hours).

    The actual rm + audit happens in deletion.verify_and_delete; this records the contrast.
    """
    return {"proxy_size_mb": round(proxy_dir_size_mb, 3), "deletion_op": "rm -rf proxy_dir"}
