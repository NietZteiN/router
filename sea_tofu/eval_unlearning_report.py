"""Standard TOFU unlearning report for SEA (Forget / Retain / Model Utility / Forget Quality).

Presents SEA in the canonical TOFU schema (same keys as eval_tofu.evaluate_model) for two states
at rank 16, so it reads like a normal TOFU unlearning result and sits next to the SISA-LoRA track:

  - ORIGINAL  : forget authors evaluated WITH their proxies = the model that KNOWS the forget data
                (TOFU "Finetuned" analog).
  - UNLEARNED : forget proxies deleted → forget set = frozen base (omission mode); retain authors
                keep their proxies; real/world = base. This IS the retrain gold by construction.
  - RETRAIN-GOLD: = base on the forget set (identical to UNLEARNED's forget side by construction).

All metric math is reused (no re-implementation): the eval_tofu primitives + the eval_sea_tofu
assemblers. The only new piece is a proxy-loaded forget-TR collector (base side already exists as
eval_sea_tofu.base_forget_truth_ratios).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import hmean, ks_2samp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")))
from eval_tofu import (  # noqa: E402
    get_answer_probability, get_rouge, get_truth_ratio_scores, tr_forget_agg, tr_nonforget_agg,
)
sys.path.insert(0, os.environ.get("SEA_TOFU_DIR", os.path.join(_REPO_ROOT, "sea_tofu")))
from eval_sea_tofu import base_constants, base_forget_truth_ratios, retain_utility  # noqa: E402
from inference import SeaProxyModel, load_base  # noqa: E402
from load_tofu import FORGET10_AUTHORS, author_perturbed_subset, load_tofu_data  # noqa: E402
from proxy_paths import personal_lora_dir, proxy_exists, results_dir  # noqa: E402
from train_proxy import _load_cfg  # noqa: E402


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



def _forget_set_metrics(model, tok, data, forget_authors, max_new, rouge_max=None):
    """Aggregate forget-set ROUGE-L / Prob over a single model state (proxy-loaded OR base)."""
    rls, probs = [], []
    for a in forget_authors:
        qa = [data["full"][a * 20 + j] for j in range(20)]
        qs, ans = [x["question"] for x in qa], [x["answer"] for x in qa]
        rls.append(get_rouge(model, tok, qs, ans, max_new_tokens=max_new, max_samples=rouge_max,
                             name=f"f{a}r"))
        probs.append(get_answer_probability(model, tok, qs, ans, max_samples=rouge_max, name=f"f{a}p"))
    return float(np.nanmean(rls)), float(np.nanmean(probs))


def _forget_tr_proxy_loaded(sea, data, forget_authors, truth_max=None):
    """Per-sample forget truth ratios with EACH author's proxy loaded (concatenated).

    Mirror of eval_sea_tofu.base_forget_truth_ratios but proxy-loaded (the model that knows the
    forget data). Used for the ORIGINAL-state forget_truth_ratio + its forget_quality KS.
    """
    trs = []
    for a in forget_authors:
        if not proxy_exists(data["model_name"], a, 16):
            continue
        sea.attach(a, personal_lora_dir(data["model_name"], a, 16))
        pert = author_perturbed_subset(data["full_pert"], data["full"], a)
        tr = get_truth_ratio_scores(sea.model, sea.tokenizer, pert,
                                    correct_key="paraphrased_answer", max_rows=truth_max)
        trs.append(tr)
    return np.concatenate(trs) if trs else np.array([])


def _row(forget_rouge, forget_prob, forget_tr_array, retain, bc, gold_tr):
    """Assemble one canonical TOFU metric dict (eval_tofu.evaluate_model schema)."""
    comps = [retain["retain_prob"], retain["retain_rouge"], retain["retain_truth_scaled"],
             bc["real_prob"], bc["real_rouge"], bc["real_truth_scaled"],
             bc["world_prob"], bc["world_rouge"], bc["world_truth_scaled"]]
    mu = float(hmean(comps)) if not any(np.isnan(c) for c in comps) else float("nan")
    fq = float(ks_2samp(forget_tr_array, gold_tr).pvalue) if len(forget_tr_array) and len(gold_tr) else float("nan")
    return {
        "forget_rouge": round(forget_rouge, 4),
        "forget_prob": round(forget_prob, 4),
        "forget_truth_ratio": round(float(tr_forget_agg(forget_tr_array)), 4) if len(forget_tr_array) else None,
        "retain_rouge": round(retain["retain_rouge"], 4),
        "retain_prob": round(retain["retain_prob"], 4),
        "retain_truth_scaled": round(retain["retain_truth_scaled"], 4),
        "real_rouge": round(bc["real_rouge"], 4), "real_prob": round(bc["real_prob"], 4),
        "real_truth_scaled": round(bc["real_truth_scaled"], 4),
        "world_rouge": round(bc["world_rouge"], 4), "world_prob": round(bc["world_prob"], 4),
        "world_truth_scaled": round(bc["world_truth_scaled"], 4),
        "forget_quality": round(fq, 4) if not np.isnan(fq) else None,
        "model_utility": round(mu, 4) if not np.isnan(mu) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.environ.get("SEA_TOFU_DIR", os.path.join(_REPO_ROOT, "sea_tofu")), "configs", "sea_tofu_llama2.json"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--n_forget", type=int, default=20)
    ap.add_argument("--n_retain", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=100)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    args = ap.parse_args()
    os.environ["HF_HOME"] = args.hf_home

    cfg = _load_cfg(args.config)
    model_name = cfg["model_name"]
    data = load_tofu_data(args.hf_home)
    data["model_name"] = model_name
    base, tok = load_base(model_name, hf_home=args.hf_home)
    sea = SeaProxyModel(base, tok)
    forget = FORGET10_AUTHORS[: args.n_forget]

    def lora_dir(a):
        return personal_lora_dir(model_name, a, args.rank)

    # Shared pieces (computed once).
    print("=== retain utility (proxies) + base constants (real/world) ===", flush=True)
    retain = retain_utility(sea, [a for a in range(180) if proxy_exists(model_name, a, args.rank)][: args.n_retain],
                            lora_dir, data, rouge_max=None, truth_max=40)
    bc = base_constants(base, tok, data, rouge_max=None, truth_max=40)
    gold_tr = base_forget_truth_ratios(base, tok, data, forget, truth_max=40)  # retrain gold

    # ORIGINAL (forget proxies loaded). Forget ROUGE/Prob need per-author proxy swap.
    print("=== ORIGINAL: forget set with proxies ===", flush=True)
    orig_rl, orig_p = [], []
    for a in forget:
        sea.attach(a, lora_dir(a))
        qa = [data["full"][a * 20 + j] for j in range(20)]
        qs, ans = [x["question"] for x in qa], [x["answer"] for x in qa]
        orig_rl.append(get_rouge(sea.model, tok, qs, ans, max_new_tokens=args.max_new, name=f"o{a}r"))
        orig_p.append(get_answer_probability(sea.model, tok, qs, ans, name=f"o{a}p"))
    orig_tr = _forget_tr_proxy_loaded(sea, data, forget, truth_max=40)
    original = _row(float(np.nanmean(orig_rl)), float(np.nanmean(orig_p)), orig_tr, retain, bc, gold_tr)

    # UNLEARNED (forget proxies deleted == base/omission on forget set).
    print("=== UNLEARNED: forget set base-only (omission) ===", flush=True)
    with sea.omission() as bm:
        un_rl, un_p = _forget_set_metrics(bm, tok, data, forget, args.max_new)
    unlearned = _row(un_rl, un_p, gold_tr, retain, bc, gold_tr)  # forget side = gold
    retrain_gold = dict(unlearned)  # identical by construction

    out = {"model_name": model_name, "rank": args.rank, "n_forget": len(forget),
           "n_retain": retain["n_retain_authors"], "max_new": args.max_new,
           "states": {"original": original, "unlearned": unlearned, "retrain_gold": retrain_gold}}
    out_dir = results_dir(model_name, None, sub="report")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "unlearning_report.json"), "w"), indent=2)

    # Markdown report
    md = ["# SEA-on-TOFU — standard unlearning report (forget10, rank %d)\n" % args.rank,
          "## Methodology (summary; full writeup in REPORT.md)\n",
          "SEA reframes TOFU so each author is one SEA *user* with a deletable per-author personal-LoRA",
          "proxy over a frozen 4-bit %s base; unlearning = `rm` of the proxy dir." % model_name,
          "Each proxy = one LoRA on q/k/v/o across all 32 layers (256 tensors, params = 1,048,576*r;",
          "r%d = %s params), trained by SFT (12 epochs, lr 2e-4) on only that author's 20 QA pairs."
          % (args.rank, f"{1048576*args.rank:,}"),
          "We evaluate the SAME base model in three states and score each with the canonical TOFU",
          "metrics (reused verbatim from tofu_sisa_lora/eval_tofu.py — ROUGE-L recall, length-normalized",
          "probability, perturbed-answer truth ratio, KS Forget Quality vs the retrain gold, harmonic-mean",
          "Model Utility over Retain×{prob,rouge,truth}+Real+World):",
          "- **Original** — forget authors with their proxies loaded (the model that *knows* the forget data).",
          "- **Unlearned** — forget proxies deleted → forget set = frozen base (omission mode); retain authors",
          "  keep their proxies; Real/World = base. This *is* the retrain gold by construction.",
          "- **Retrain gold** — base on the forget set (identical to Unlearned's forget side by construction).",
          "Model Utility is identical across states because deletion never touches retain/real/world.\n",
          "## Results\n",
          "| State | Forget ROUGE-L | Forget Prob | Forget TR | Retain ROUGE | Retain Prob | "
          "Real | World | Forget Quality | Model Utility |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    labels = [("Original (proxies loaded)", original), ("Unlearned (proxies deleted)", unlearned),
              ("Retrain gold (= base)", retrain_gold)]
    for name, r in labels:
        md.append(f"| {name} | {r['forget_rouge']} | {r['forget_prob']} | {r['forget_truth_ratio']} | "
                  f"{r['retain_rouge']} | {r['retain_prob']} | {r['real_rouge']} | {r['world_rouge']} | "
                  f"{r['forget_quality']} | {r['model_utility']} |")
    md += ["", "Forget Quality = KS p-value of forget truth-ratios vs the retrain gold (base on forget).",
           "Model Utility = harmonic mean of retain/real/world × prob/rouge/truth (unchanged by deletion).",
           "Deletion cost = `rm` of the proxy dir (ms); retrain gold reached by construction."]
    report_dir = os.path.join(os.environ.get("SEA_TOFU_DIR", os.path.join(_REPO_ROOT, "sea_tofu")), "reports")
    os.makedirs(report_dir, exist_ok=True)
    open(os.path.join(report_dir, "SEA_UNLEARNING_REPORT.md"), "w").write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {os.path.join(out_dir, 'unlearning_report.json')} and reports/SEA_UNLEARNING_REPORT.md")


if __name__ == "__main__":
    main()
