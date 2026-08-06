"""Combined SEA-on-TOFU rank-sweep evaluation (one GPU job, base loaded once).

Replaces the 5 separate per-rank eval jobs (which each reloaded the 7B base and recomputed the
constant real/world base_constants, and used max_new_tokens=100 → all hit SLURM time limits).

Efficiency:
  - base loaded once; the frozen base-side personalization (per forget author) is computed ONCE
    and reused for every rank (it does not depend on rank),
  - proxy-side is computed per rank,
  - generation capped at --max_new (default 40; TOFU answers are short),
  - results written incrementally per rank so a timeout still yields a partial table.

Outputs proxies/{slug}/results/sweep/sweep_results.json + a printed tradeoff table.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")))
from eval_tofu import (  # noqa: E402
    get_answer_probability, get_rouge, get_truth_ratio_scores, tr_forget_agg, tr_nonforget_agg,
)
sys.path.insert(0, os.environ.get("SEA_TOFU_DIR", os.path.join(_REPO_ROOT, "sea_tofu")))
from eval_sea_tofu import base_constants, base_forget_truth_ratios, forget_quality  # noqa: E402
from inference import SeaProxyModel, generate, load_base  # noqa: E402
from load_tofu import FORGET10_AUTHORS, author_perturbed_subset, load_tofu_data  # noqa: E402
from metrics_sea import _jaccard  # noqa: E402
from proxy_paths import author_dir, dir_size_mb, personal_lora_dir, proxy_exists, results_dir  # noqa
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


RANKS = [4, 8, 16, 32, 64]


def measure(model, tok, full_ds, full_pert, author, max_new, rouge_max=None):
    """prob / ROUGE-L / forget-truth-ratio for one author's 20 QA on `model`."""
    qa = [full_ds[author * 20 + j] for j in range(20)]
    qs = [x["question"] for x in qa]
    ans = [x["answer"] for x in qa]
    prob = get_answer_probability(model, tok, qs, ans, max_samples=rouge_max, name=f"a{author}p")
    rl = get_rouge(model, tok, qs, ans, max_new_tokens=max_new, max_samples=rouge_max, name=f"a{author}r")
    pert = author_perturbed_subset(full_pert, full_ds, author)
    tr = get_truth_ratio_scores(model, tok, pert, correct_key="paraphrased_answer")
    return {"prob": round(float(prob), 4), "rougeL": round(float(rl), 4),
            "truth_ratio": round(float(tr_forget_agg(tr)), 4) if len(tr) else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.environ.get("SEA_TOFU_DIR", os.path.join(_REPO_ROOT, "sea_tofu")), "configs", "sea_tofu_llama2.json"))
    ap.add_argument("--n_forget", type=int, default=20, help="Forget authors evaluated (<=20).")
    ap.add_argument("--n_retain", type=int, default=20, help="Retain authors for r16 utility.")
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--n_iso_pairs", type=int, default=8)
    ap.add_argument("--tag", default="sweep", help="Output subdir results/<tag>/ (avoid clobbering).")
    ap.add_argument("--ranks", default="4,8,16,32,64", help="Comma list of ranks to evaluate.")
    ap.add_argument("--proxy_root", default=None, help="Override proxy root (seed-variance roots).")
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    args = ap.parse_args()
    os.environ["HF_HOME"] = args.hf_home

    cfg = _load_cfg(args.config)
    model_name = cfg["model_name"]
    ranks = [int(r) for r in args.ranks.split(",")]
    pr_kw = {} if args.proxy_root is None else {"proxy_root": args.proxy_root}
    data = load_tofu_data(args.hf_home)
    base, tok = load_base(model_name, hf_home=args.hf_home)
    sea = SeaProxyModel(base, tok)

    forget = FORGET10_AUTHORS[: args.n_forget]
    # results always written under the DEFAULT tree (keep all JSONs together), tagged by run.
    out_path = os.path.join(results_dir(model_name, None, sub=args.tag), "sweep_results.json")
    state = {"model_name": model_name, "n_forget": len(forget), "max_new": args.max_new, "ranks": {}}

    def flush():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(state, open(out_path, "w"), indent=2)

    # ── 1) Base-side (frozen, shared across ranks) — compute ONCE ────────────
    print("=== base-side personalization (shared) ===", flush=True)
    base_side = {a: measure(base, tok, data["full"], data["full_pert"], a, args.max_new) for a in forget}
    state["base_side"] = {str(a): v for a, v in base_side.items()}
    print("=== base constants (real/world, frozen) ===", flush=True)
    state["base_constants"] = {k: round(v, 4) for k, v in
                               base_constants(base, tok, data, rouge_max=40, truth_max=40).items()}
    base_tr = base_forget_truth_ratios(base, tok, data, forget, truth_max=40)
    state["forget_quality"] = round(forget_quality(base_tr, base_tr), 4)
    flush()

    # ── 2) Proxy-side per rank (write incrementally) ────────────────────────
    for rank in ranks:
        print(f"=== rank {rank} proxy-side ===", flush=True)
        rows = []
        for a in forget:
            ld = personal_lora_dir(model_name, a, rank, **pr_kw)
            if not proxy_exists(model_name, a, rank, **pr_kw):
                continue
            sea.attach(a, ld)
            prox = measure(sea.model, tok, data["full"], data["full_pert"], a, args.max_new)
            b = base_side[a]
            rows.append({"author_id": a,
                         "proxy_prob": prox["prob"], "base_prob": b["prob"],
                         "delta_prob": round(prox["prob"] - b["prob"], 4),
                         "proxy_rougeL": prox["rougeL"], "base_rougeL": b["rougeL"],
                         "delta_rougeL": round(prox["rougeL"] - b["rougeL"], 4),
                         "proxy_truth_ratio": prox["truth_ratio"], "base_truth_ratio": b["truth_ratio"]})
        # isolation: proxy A on author B (A,B consecutive), contamination vs base
        iso = []
        for i in range(min(args.n_iso_pairs, len(forget) - 1)):
            pa, pb = forget[i], forget[i + 1]
            if not proxy_exists(model_name, pa, rank, **pr_kw):
                continue
            sea.attach(pa, personal_lora_dir(model_name, pa, rank, **pr_kw))
            probe_qs = [data["full"][pb * 20 + j]["question"] for j in range(3)]
            gold = [data["full"][pb * 20 + j]["answer"] for j in range(3)]
            with_a = [generate(sea.model, tok, q, args.max_new) for q in probe_qs]
            with sea.omission() as bm:
                bbase = [generate(bm, tok, q, args.max_new) for q in probe_qs]
            sa = float(np.mean([_jaccard(p, g) for p, g in zip(with_a, gold)]))
            sb = float(np.mean([_jaccard(p, g) for p, g in zip(bbase, gold)]))
            iso.append({"proxy_author": pa, "probe_author": pb, "contamination": round(max(0.0, sa - sb), 4)})
        size_mb = round(dir_size_mb(author_dir(model_name, forget[0], rank, **pr_kw)), 2)
        state["ranks"][str(rank)] = {
            "proxy_size_mb": size_mb,
            "mean_proxy_rougeL": round(float(np.nanmean([r["proxy_rougeL"] for r in rows])), 4),
            "mean_base_rougeL": round(float(np.nanmean([r["base_rougeL"] for r in rows])), 4),
            "mean_proxy_prob": round(float(np.nanmean([r["proxy_prob"] for r in rows])), 4),
            "mean_base_prob": round(float(np.nanmean([r["base_prob"] for r in rows])), 4),
            "mean_proxy_truth_ratio": round(float(np.nanmean([r["proxy_truth_ratio"] for r in rows])), 4),
            "max_contamination": max((r["contamination"] for r in iso), default=float("nan")),
            "n_authors": len(rows), "rows": rows, "isolation": iso,
        }
        flush()
        print(f"  r{rank}: size={size_mb}MB proxyROUGE="
              f"{state['ranks'][str(rank)]['mean_proxy_rougeL']} "
              f"contam={state['ranks'][str(rank)]['max_contamination']}", flush=True)

    # ── 3) r16 retain utility + model utility (skip when n_retain=0) ─────────
    if args.n_retain <= 0:
        flush()
        _print_table(state, ranks, out_path)
        return
    print("=== r16 retain utility ===", flush=True)
    from scipy.stats import hmean
    retain_pool = [a for a in range(180) if proxy_exists(model_name, a, 16, **pr_kw)]
    rng = np.random.default_rng(cfg["train"]["seed"])
    retain_sample = sorted(rng.choice(retain_pool, size=min(args.n_retain, len(retain_pool)),
                                      replace=False).tolist()) if retain_pool else []
    rprob, rrouge, rtruth = [], [], []
    for a in retain_sample:
        sea.attach(a, personal_lora_dir(model_name, a, 16, **pr_kw))
        m = measure(sea.model, tok, data["full"], data["full_pert"], a, args.max_new)
        rprob.append(m["prob"]); rrouge.append(m["rougeL"])
        pert = author_perturbed_subset(data["full_pert"], data["full"], a)
        tr = get_truth_ratio_scores(sea.model, tok, pert, correct_key="paraphrased_answer", max_rows=40)
        rtruth.append(float(tr_nonforget_agg(tr)))
    bc = state["base_constants"]
    comps = [np.nanmean(rprob), np.nanmean(rrouge), np.nanmean(rtruth),
             bc["real_prob"], bc["real_rouge"], bc["real_truth_scaled"],
             bc["world_prob"], bc["world_rouge"], bc["world_truth_scaled"]]
    state["retain_utility"] = {"retain_prob": round(float(np.nanmean(rprob)), 4),
                               "retain_rouge": round(float(np.nanmean(rrouge)), 4),
                               "retain_truth_scaled": round(float(np.nanmean(rtruth)), 4),
                               "n_retain": len(retain_sample)}
    state["model_utility_r16"] = round(float(hmean(comps)), 4) if not any(np.isnan(c) for c in comps) else None
    flush()

    _print_table(state, ranks, out_path)


def _print_table(state, ranks, out_path):
    print("\n=== SEA-on-TOFU rank sweep (forget10, proxy loaded) ===")
    print(f"{'rank':>4} {'sizeMB':>7} {'pROUGE':>7} {'bROUGE':>7} {'pProb':>6} {'bProb':>6} "
          f"{'pTR':>6} {'contam':>6}")
    for rank in ranks:
        r = state["ranks"].get(str(rank))
        if not r:
            continue
        print(f"{rank:>4} {r['proxy_size_mb']:>7.1f} {r['mean_proxy_rougeL']:>7.3f} "
              f"{r['mean_base_rougeL']:>7.3f} {r['mean_proxy_prob']:>6.3f} {r['mean_base_prob']:>6.3f} "
              f"{r['mean_proxy_truth_ratio']:>6.3f} {r['max_contamination']:>6.3f}")
    print(f"\nforget_quality (construction-trivial): {state.get('forget_quality')}")
    if state.get("retain_utility"):
        print(f"model_utility @ r16: {state.get('model_utility_r16')}  "
              f"(retain n={state['retain_utility']['n_retain']})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
