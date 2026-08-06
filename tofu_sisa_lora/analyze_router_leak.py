"""Router-leak CPU post-processing (thread log/router_leak/) — consumes the .sims.npz
sidecars written by routing_audit_tofu.py (--dump_sims / centroid mode); no encoder rerun.

Subcommands:
  roc       — the H1/H2 threshold-detector family on orphan-vs-retain separation, with an
              AUTHOR-level calibrate/eval split (queries within an author are 20 correlated
              siblings; a query-level split would leak identity). Detectors (score direction
              fixed a priori, higher = more orphan-like):
                global_top1   -(masked top-1 sim)            [the refuted 07-07 abstain stat]
                per_expert    -(z of top-1 sim vs that expert's calib-retain distribution)
                margin        -(top1 - top2 over survivors)
                knn_density   -(mean top-3 surviving sims)
                tomb_expert / tomb_author / tomb_name
                              best-tombstone sim - best-surviving sim (identity rungs)
              Reports ROC-AUC on the eval half + retain-FPR at the tau giving 90% orphan
              catch (H1 bar: AUC >= 0.90 AND FPR <= 0.05).
  coverage  — R6 registry-coverage cells (pure CPU + HF cache): per-author extracted names
              (router._extract_author_names) matched against original questions, the
              forget10_perturbed paraphrased questions, and name-stripped originals.
  table     — compact markdown table over a list of audit JSONs (R5 deletion-count dial).

  python analyze_router_leak.py roc --npz A.sims.npz [--legonet] --out reports/rl_roc.json
  python analyze_router_leak.py coverage --out reports/rl_coverage.json
  python analyze_router_leak.py table --jsons a.json b.json --out reports/rl_table.md
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

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


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from routing_audit_tofu import _auc as f
    return f(np.asarray(pos, "float64"), np.asarray(neg, "float64"))


def _fpr_at_catch(pos: np.ndarray, neg: np.ndarray, catch: float = 0.90) -> dict:
    """Threshold at the `catch` quantile of pos scores; report neg FPR there."""
    tau = float(np.quantile(pos, 1.0 - catch))
    return {"tau": tau, "orphan_catch": float((pos >= tau).mean()),
            "retain_fpr": float((neg >= tau).mean())}


def _split_by_author(authors: np.ndarray):
    """Deterministic author-parity split: even ids calibrate, odd ids evaluate."""
    return (authors % 2 == 0), (authors % 2 == 1)


def detector_scores(surv_f: np.ndarray, surv_r: np.ndarray, tomb_f: dict, tomb_r: dict,
                    r_calib_mask: np.ndarray, top1_col_f: np.ndarray, top1_col_r: np.ndarray):
    """{detector: (scores_forget, scores_retain)} with per-expert calibration fit on the
    retain CALIBRATION half only. surv_* = sims over surviving pool cols; tomb_* =
    {rung: sims over that rung's tombstone cols}."""
    def _topk(s, k):
        return np.sort(s, axis=1)[:, ::-1][:, :k]

    out = {}
    f_sorted, r_sorted = _topk(surv_f, 3), _topk(surv_r, 3)
    out["global_top1"] = (-f_sorted[:, 0], -r_sorted[:, 0])
    out["margin"] = (-(f_sorted[:, 0] - f_sorted[:, 1]), -(r_sorted[:, 0] - r_sorted[:, 1]))
    out["knn_density"] = (-f_sorted.mean(1), -r_sorted.mean(1))

    # per-expert z of the top-1 sim, calibrated on the retain calibration half
    n_cols = surv_f.shape[1]
    mu = np.zeros(n_cols); sd = np.ones(n_cols)
    r_top1 = surv_r.max(1)
    for e in range(n_cols):
        sel = (top1_col_r == e) & r_calib_mask
        if sel.sum() >= 5:
            mu[e], sd[e] = r_top1[sel].mean(), max(r_top1[sel].std(), 1e-6)
    f_top1 = surv_f.max(1)
    out["per_expert"] = (-(f_top1 - mu[top1_col_f]) / sd[top1_col_f],
                        -(r_top1 - mu[top1_col_r]) / sd[top1_col_r])

    for rung, tf in tomb_f.items():
        tr = tomb_r[rung]
        out[f"tomb_{rung}"] = (tf.max(1) - f_sorted[:, 0], tr.max(1) - r_sorted[:, 0])
    return out


def cmd_roc(args):
    z = np.load(args.npz, allow_pickle=False)
    if args.legonet:
        surv_all_f, surv_all_r = z["sims_stale_forget"], z["sims_stale_retain"]
        affected = set(z["affected"].tolist())
        surv_cols = [j for j in range(surv_all_f.shape[1]) if j not in affected]
        aff_cols = sorted(affected)
        surv_f, surv_r = surv_all_f[:, surv_cols], surv_all_r[:, surv_cols]
        tomb_f = {"expert": surv_all_f[:, aff_cols], "author": z["sims_author_sent_forget"]}
        tomb_r = {"expert": surv_all_r[:, aff_cols], "author": z["sims_author_sent_retain"]}
        if "sims_name_sent_forget" in z:
            tomb_f["name"], tomb_r["name"] = z["sims_name_sent_forget"], z["sims_name_sent_retain"]
        a_f, a_r = z["forget_author_per_q"], z["retain_author_per_q"]
    else:   # centroid npz
        sims = z["sims_centroid_all"]
        f_rows, r_rows = z["forget_rows"], z["retain_rows"]
        sids = z["centroid_sids"].tolist()
        drop_col = sids.index(args.drop_shard)
        surv_cols = [j for j in range(sims.shape[1]) if j != drop_col]
        surv_f, surv_r = sims[f_rows][:, surv_cols], sims[r_rows][:, surv_cols]
        tomb_f = {"expert": sims[f_rows][:, [drop_col]],
                  "author": z["sims_author_sent_forget"], "name": z["sims_name_sent_forget"]}
        tomb_r = {"expert": sims[r_rows][:, [drop_col]],
                  "author": z["sims_author_sent_retain"], "name": z["sims_name_sent_retain"]}
        authors = z["author_of_q"]
        a_f, a_r = authors[f_rows], authors[r_rows]

    f_calib, f_eval = _split_by_author(a_f)
    r_calib, r_eval = _split_by_author(a_r)
    scores = detector_scores(surv_f, surv_r, tomb_f, tomb_r, r_calib,
                             surv_f.argmax(1), surv_r.argmax(1))
    out = {"npz": os.path.abspath(args.npz), "mode": "legonet" if args.legonet else "centroid",
           "n_forget_eval": int(f_eval.sum()), "n_retain_eval": int(r_eval.sum()),
           "detectors": {}}
    for name, (sf, sr) in scores.items():
        pos, neg = sf[f_eval], sr[r_eval]
        out["detectors"][name] = {"auc": _auc(pos, neg), **_fpr_at_catch(pos, neg)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[roc] -> {args.out}")
    for name, d in sorted(out["detectors"].items(), key=lambda kv: -kv[1]["auc"]):
        print(f"  {name:12s} AUC={d['auc']:.3f}  FPR@90%catch={d['retain_fpr']:.3f}")


def cmd_coverage(args):
    os.environ.setdefault("HF_HOME", args.hf_home)
    from datasets import load_dataset
    from router import _extract_author_names
    import re
    full = load_dataset("locuslab/TOFU", "full")["train"]
    pert = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"]
    per = 20

    names = {}
    for a in range(200):
        qs = [full[a * per + w]["question"] for w in range(per)]
        names[a] = _extract_author_names(qs)

    def _covers(a, q):
        return any(nm.lower() in q.lower() for nm in names[a])

    def _strip(a, q):
        for nm in sorted(names[a], key=len, reverse=True):
            q = re.sub(re.escape(nm), "", q, flags=re.IGNORECASE)
        return q

    cov_orig = np.mean([_covers(i // per, full[i]["question"]) for i in range(200 * per)])
    forget_rows = range(180 * per, 200 * per)
    cov_forget = np.mean([_covers(i // per, full[i]["question"]) for i in forget_rows])
    cov_para = np.mean([_covers(180 + i // per, pert[i]["paraphrased_question"])
                        for i in range(len(pert))])
    cov_stripped = np.mean([_covers(i // per, _strip(i // per, full[i]["question"]))
                            for i in forget_rows])
    out = {"n_authors": 200, "authors_without_names": sum(1 for a in names if not names[a]),
           "coverage_original_all": float(cov_orig),
           "coverage_original_forget10": float(cov_forget),
           "coverage_paraphrased_forget10": float(cov_para),
           "coverage_name_stripped_forget10": float(cov_stripped)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[coverage] -> {args.out}")
    for k2, v in out.items():
        print(f"  {k2} = {v}")


def cmd_table(args):
    lines = ["| audit | policy | orphan sibling top-1 | retain shift | tomb[author] catch/FPR |",
             "|---|---|---|---|---|"]
    for p in args.jsons:
        with open(p) as f:
            res = json.load(f)
        name = os.path.basename(p)
        sib = res.get("policies", {}).get("dropped", {}).get("sibling_top1_rate")
        if sib is None:
            sib = 1.0 - res.get("full", {}).get("acc_forget_top1", float("nan"))
        shift = (res.get("selection_shift", {}).get("embed_stale_vs_dropped", {}).get("shift_top1")
                 or res.get("sibling", {}).get("retain_shift_top1"))
        ta = res.get("tombstone", {}).get("author", {})
        lines.append(f"| {name} | dropped/sibling | {sib} | {shift} | "
                     f"{ta.get('orphan_catch_rate')}/{ta.get('retain_false_tombstone_rate')} |")
    md = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roc")
    r.add_argument("--npz", required=True)
    r.add_argument("--legonet", action="store_true",
                   help="npz is a legonet-mode sidecar (default: centroid-mode)")
    r.add_argument("--drop_shard", type=int, default=9)
    r.add_argument("--out", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    c.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("--jsons", nargs="+", required=True)
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    {"roc": cmd_roc, "coverage": cmd_coverage, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
