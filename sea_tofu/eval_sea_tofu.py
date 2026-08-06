"""SEA-on-TOFU evaluation orchestrator.

Assembles the result tables (SEA_on_TOFU.md §5, §9) from:
  - the canonical TOFU metric primitives imported from tofu_sisa_lora/eval_tofu.py, and
  - the SEA-specific metrics in metrics_sea.py.

Because SEA swaps a different proxy per author, we do NOT reuse eval_tofu.evaluate_model (it
assumes a single active model). Instead we call the primitives per author and assemble Model
Utility with scipy.stats.hmean over the same 9 components, matching evaluate_model exactly.

Tables produced:
  - Personalization depth (per author, proxy loaded vs base): metrics_sea.personalization_depth
  - Forget Quality after deletion: ks_2samp over forget-author truth ratios, candidate vs gold,
    both = base-only -> p ≈ 1 by construction (flagged; not a headline).
  - Model Utility: hmean over {retain (proxy loaded), real, world} × {prob, rouge, truth_scaled}.
  - Isolation: cross-author contamination.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import hmean, ks_2samp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOFU_SISA = os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora"))
if _TOFU_SISA not in sys.path:
    sys.path.insert(0, _TOFU_SISA)

from eval_tofu import (  # noqa: E402
    get_answer_probability,
    get_prob_w_options,
    get_rouge,
    get_truth_ratio_scores,
    tr_nonforget_agg,
)

from load_tofu import author_perturbed_subset  # noqa: E402
from metrics_sea import isolation, personalization_depth  # noqa: E402


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



# ── Base-only constants (frozen base; identical across every SEA run) ────────

def base_constants(base, tok, data, rouge_max=None, truth_max=None):
    """real_authors / world_facts prob+rouge+truth_scaled on the frozen base.

    These never change (base is frozen) — flag in the writeup so the stability is not misread
    as a SEA effect (SEA_on_TOFU.md §5.5, §8).
    """
    out = {}
    for key, qa_split, pert_split in [
        ("real", data["real_authors"], data["real_authors_pert"]),
        ("world", data["world_facts"], data["world_facts_pert"]),
    ]:
        out[f"{key}_prob"] = get_prob_w_options(base, tok, pert_split, name=f"{key}_prob")
        out[f"{key}_rouge"] = get_rouge(base, tok, qa_split["question"], qa_split["answer"],
                                        max_samples=rouge_max, name=f"{key}_rouge")
        tr = get_truth_ratio_scores(base, tok, pert_split, correct_key="answer", max_rows=truth_max)
        out[f"{key}_truth_scaled"] = float(tr_nonforget_agg(tr))
    return out


def base_forget_truth_ratios(base, tok, data, forget_authors, truth_max=None):
    """Base-only per-sample truth ratios over the forget authors' perturbed rows.

    This is BOTH the post-deletion candidate and the retrain gold for Forget Quality: after
    deleting the forget proxies the system is exactly the base, and SEA's gold (base never
    learned them) is also the base. Returns the raw tr array.
    """
    trs = []
    for a in forget_authors:
        pert = author_perturbed_subset(data["full_pert"], data["full"], a)
        tr = get_truth_ratio_scores(base, tok, pert, correct_key="paraphrased_answer",
                                    max_rows=truth_max)
        trs.append(tr)
    return np.concatenate(trs) if trs else np.array([])


def forget_quality(candidate_tr, gold_tr):
    """KS p-value of forget truth-ratio distributions. For SEA both are base-only -> ≈1."""
    if len(candidate_tr) == 0 or len(gold_tr) == 0:
        return float("nan")
    return float(ks_2samp(candidate_tr, gold_tr).pvalue)


# ── Retain utility (each retain author evaluated with ITS proxy loaded) ──────

def retain_utility(sea, retain_authors, proxy_dir_fn, data, rouge_max=None, truth_max=None):
    """Average prob / rouge / truth_scaled over retain authors, each with its own proxy.

    proxy_dir_fn(author_id) -> personal_lora dir. Authors whose proxy is missing are skipped.
    """
    probs, rouges, truth = [], [], []
    used = 0
    for a in retain_authors:
        lora = proxy_dir_fn(a)
        if not os.path.isfile(os.path.join(lora, "adapter_config.json")):
            continue
        sea.attach(a, lora)
        qa = [data["full"][a * 20 + j] for j in range(20)]
        qs = [x["question"] for x in qa]
        ans = [x["answer"] for x in qa]
        probs.append(get_answer_probability(sea.model, sea.tokenizer, qs, ans,
                                            max_samples=rouge_max, name=f"retain{a}_prob"))
        rouges.append(get_rouge(sea.model, sea.tokenizer, qs, ans,
                                max_samples=rouge_max, name=f"retain{a}_rouge"))
        pert = author_perturbed_subset(data["full_pert"], data["full"], a)
        tr = get_truth_ratio_scores(sea.model, sea.tokenizer, pert,
                                    correct_key="paraphrased_answer", max_rows=truth_max)
        truth.append(float(tr_nonforget_agg(tr)))
        used += 1
    return {
        "retain_prob": float(np.nanmean(probs)) if probs else float("nan"),
        "retain_rouge": float(np.nanmean(rouges)) if rouges else float("nan"),
        "retain_truth_scaled": float(np.nanmean(truth)) if truth else float("nan"),
        "n_retain_authors": used,
    }


def model_utility(retain, base_const):
    """Harmonic mean of the 9 components (matches eval_tofu.evaluate_model / open-unlearning)."""
    comps = [
        retain["retain_prob"], retain["retain_rouge"], retain["retain_truth_scaled"],
        base_const["real_prob"], base_const["real_rouge"], base_const["real_truth_scaled"],
        base_const["world_prob"], base_const["world_rouge"], base_const["world_truth_scaled"],
    ]
    if any(np.isnan(c) for c in comps):
        return float("nan")
    return float(hmean(comps))


# ── Personalization + isolation drivers ──────────────────────────────────────

def personalization_table(sea, authors, proxy_dir_fn, data, rouge_max=None, truth_max=None):
    rows = []
    for a in authors:
        lora = proxy_dir_fn(a)
        if not os.path.isfile(os.path.join(lora, "adapter_config.json")):
            continue
        sea.attach(a, lora)
        rows.append(personalization_depth(sea, a, data["full"], data["full_pert"],
                                           rouge_max=rouge_max, truth_max=truth_max))
    return rows


def isolation_table(sea, pairs, proxy_dir_fn, data, n_questions=5):
    rows = []
    for proxy_a, probe_b in pairs:
        lora = proxy_dir_fn(proxy_a)
        if not os.path.isfile(os.path.join(lora, "adapter_config.json")):
            continue
        sea.attach(proxy_a, lora)
        rows.append(isolation(sea, proxy_a, probe_b, data["full"], n_questions=n_questions))
    return rows


def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[eval] wrote {path}", flush=True)


# ── Full (scale-phase) evaluation CLI ────────────────────────────────────────

def _parse_args():
    import argparse
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Full SEA-on-TOFU evaluation over trained proxies.")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--config", default=os.path.join(here, "configs", "sea_tofu_llama2.json"))
    p.add_argument("--n_retain", type=int, default=30,
                   help="Sample size of retain authors (with proxies) for the utility estimate.")
    p.add_argument("--n_iso_pairs", type=int, default=10)
    p.add_argument("--smoke", action="store_true", help="Cap per-set sample sizes for a fast pass.")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    import numpy as _np
    from inference import SeaProxyModel, load_base
    from load_tofu import FORGET10_AUTHORS, load_tofu_data
    from proxy_paths import personal_lora_dir, proxy_exists, results_dir
    from train_proxy import _load_cfg

    args = _parse_args()
    os.environ["HF_HOME"] = args.hf_home
    cfg = _load_cfg(args.config)
    model_name = cfg["model_name"]
    rank = args.rank
    rouge_max = 20 if args.smoke else None
    truth_max = 20 if args.smoke else None

    data = load_tofu_data(args.hf_home)
    base, tok = load_base(model_name, hf_home=args.hf_home)
    sea = SeaProxyModel(base, tok)

    def lora_dir(a):
        return personal_lora_dir(model_name, a, rank)

    forget_authors = FORGET10_AUTHORS
    retain_pool = [a for a in range(180) if proxy_exists(model_name, a, rank)]
    rng = _np.random.default_rng(cfg["train"]["seed"])
    retain_sample = sorted(rng.choice(retain_pool, size=min(args.n_retain, len(retain_pool)),
                                      replace=False).tolist()) if retain_pool else []

    bconst = base_constants(base, tok, data, rouge_max=(50 if args.smoke else None),
                            truth_max=truth_max)
    base_tr = base_forget_truth_ratios(base, tok, data, forget_authors, truth_max=truth_max)
    fq = forget_quality(base_tr, base_tr)
    pers = personalization_table(sea, forget_authors, lora_dir, data,
                                 rouge_max=rouge_max, truth_max=truth_max)
    retain = retain_utility(sea, retain_sample, lora_dir, data,
                            rouge_max=rouge_max, truth_max=truth_max)
    util = model_utility(retain, bconst)
    iso_pairs = [(forget_authors[i], forget_authors[(i + 1) % len(forget_authors)])
                 for i in range(min(args.n_iso_pairs, len(forget_authors)))]
    iso = isolation_table(sea, iso_pairs, lora_dir, data, n_questions=5)

    out = args.out or os.path.join(results_dir(model_name, rank, sub="smoke" if args.smoke else "full"),
                                   "sea_tofu_results.json")
    summary = {
        "model_name": model_name, "rank": rank, "smoke": args.smoke,
        "forget_authors": forget_authors, "n_retain_sampled": len(retain_sample),
        "base_constants": {k: round(v, 4) for k, v in bconst.items()},
        "forget_quality": round(fq, 4),
        "model_utility": round(util, 4) if not _np.isnan(util) else None,
        "retain_utility": retain,
        "personalization": pers,
        "isolation": iso,
    }
    write_json(summary, out)
    print(json.dumps({k: summary[k] for k in
                      ["rank", "forget_quality", "model_utility", "n_retain_sampled"]}, indent=2))


if __name__ == "__main__":
    main()
