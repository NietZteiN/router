"""APA study collector: joins the sum ladder, the existing mean ladder, the norms and MMLU.

Experiments A and C (log/merge_mechanism/, 2026-07-28). Pure CPU, safe on partial results —
every input is optional and missing pieces are reported, never silently dropped.

Inputs
  --sum_csv    reports/expA/nmerge_mu.csv   (analyze_nmerge on configs/nmerge_sum_expA_7b.json)
  --mean_csv   reports/nmerge_mu.csv        (the completed additive_mean ladder, for the overlay)
  --norms_json reports/expA/expA_norms.json (merge_subset.py norms)
  --mmlu_glob  '<out_dir>/results/smoke/*.mmlu.json'
  --contrib_glob 'reports/expA/contrib_*.json'

Outputs
  <prefix>_summary.csv     one row per (method, n, seed, probe): mu, mu_gmean, the 9 components,
                           retain_ppl, mmlu acc + entropy, and the joined norm ladder
  <prefix>_norms.csv       the magnitude law on its own
  <prefix>_mmlu.csv
  <prefix>_contrib.csv     n_eff / cancellation per query tier (Exp C)

The headline question this file exists to answer: is unowned-query degradation a function of
`rel_pert` = ||sum||_F / ||W_0||_F ALONE? If the sum arm (lambda=1), the matched-norm arm
(1/sqrt(N)) and the mean arm (1/N) collapse onto one curve in that coordinate, then uniform
summation costs utility purely by injecting magnitude, and the 1/N in additive_mean is doing
nothing but norm control. `<prefix>_summary.csv` carries the columns to test that directly.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

MU_COMPONENTS = ["retain_prob", "retain_rouge", "retain_truth_scaled",
                 "real_prob", "real_rouge", "real_truth_scaled",
                 "world_prob", "world_rouge", "world_truth_scaled"]


def _f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _read_csv(p):
    return list(csv.DictReader(open(p))) if p and os.path.exists(p) else []


def load_norms(path):
    """(method_family, n) -> norm row. rel_pert for the mean arm is the sum's / N, and for the
    matched-norm arm the sum's / sqrt(N) — both exact, because a global coefficient scales the
    whole delta linearly. That is what puts all three arms on one magnitude axis for free."""
    if not path or not os.path.exists(path):
        return {}
    d = json.load(open(path))
    out = {}
    for r in d.get("rows", []):
        n, seed = r["n"], r["seed"]
        base = {k: r.get(k) for k in
                ("fro_sum", "fro_indiv_l2", "kappa", "rel_pert_global",
                 "rel_pert_slot_mean", "rel_pert_slot_max", "sqrt_n")}
        for meth, scale in (("additive_sum", 1.0),
                            ("additive_sum_isqrt", 1.0 / math.sqrt(n)),
                            ("additive_mean", 1.0 / n)):
            row = dict(base)
            for k in ("fro_sum", "rel_pert_global", "rel_pert_slot_mean", "rel_pert_slot_max"):
                if row.get(k) is not None:
                    row[k] = row[k] * scale
            row["lam_weight"] = scale
            out[(meth, n, seed)] = row
    return out


def load_mmlu(pattern):
    out = {}
    for p in sorted(glob.glob(pattern or "")):
        d = json.load(open(p))
        out[d["label"]] = {
            "mmlu_acc": d.get("acc"), "mmlu_acc_se": d.get("acc_se"), "mmlu_n": d.get("n"),
            "mmlu_pred_letter_entropy": d.get("pred_letter_entropy"),
            "mmlu_mean_letter_entropy": d.get("mean_letter_entropy"),
            "mmlu_pred_hist": "|".join(str(x) for x in (d.get("pred_hist") or [])),
        }
    return out


def load_contrib(pattern):
    rows = []
    for p in sorted(glob.glob(pattern or "")):
        d = json.load(open(p))
        s = d.get("summary", {})
        rows.append({
            "label": (d.get("merge_meta") or {}).get("label") or os.path.basename(p),
            "adapter": d.get("adapter"), "hidden": d.get("hidden"),
            "n_authors": s.get("n_authors"),
            "decomposition_rel_err": d.get("decomposition_rel_err"),
            **{k: s.get(k) for k in
               ("own_orig_selectivity_median", "own_para_selectivity_median",
                "n_eff_own_orig", "n_eff_own_para", "n_eff_unowned_orig",
                "n_eff_holdout10", "n_eff_ood", "cancel_own_orig",
                "cancel_unowned_orig", "cancel_holdout10",
                "orthogonal_cancel_expectation")},
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sum_csv", default="reports/expA/nmerge_mu.csv")
    ap.add_argument("--mean_csv", default="reports/nmerge_mu.csv")
    ap.add_argument("--norms_json", default="reports/expA/expA_norms.json")
    ap.add_argument("--mmlu_glob", default=None)
    ap.add_argument("--contrib_glob", default="reports/expA/contrib_*.json")
    ap.add_argument("--out_prefix", default="reports/expA/expA")
    args = ap.parse_args()

    norms = load_norms(args.norms_json)
    mmlu = load_mmlu(args.mmlu_glob)
    missing = []

    rows = []
    for src, tag in ((args.sum_csv, "sum-ladder"), (args.mean_csv, "mean-ladder")):
        got = _read_csv(src)
        if not got:
            missing.append(f"{tag} ({src})")
            continue
        for r in got:
            if r.get("kind") not in ("merge", "iso", "anchor"):
                continue
            n = int(r["n"]) if r.get("n") not in (None, "", "None") else None
            seed = int(r["seed"]) if r.get("seed") not in (None, "", "None") else None
            meth = r.get("method") or r.get("kind")
            row = {
                "source": tag, "label": r["label"], "kind": r["kind"], "method": meth,
                "n": n, "seed": seed, "svd_rank": r.get("svd_rank"),
                "probe_author": r.get("probe_author"), "headline": r.get("headline"),
                "model_utility": _f(r.get("model_utility")),
                "mu_gmean": _f(r.get("mu_gmean")),
                "first_zero_component": r.get("first_zero_component", ""),
                "retain_ppl": _f(r.get("retain_ppl")), "forget_ppl": _f(r.get("forget_ppl")),
                "forget_quality": _f(r.get("forget_quality")),
                "flags": r.get("flags", ""),
            }
            for c in MU_COMPONENTS:
                row[c] = _f(r.get(c))
            nk = norms.get((meth, n, seed)) or norms.get((meth, n, 42))
            if nk:
                row.update({f"norm_{k}": v for k, v in nk.items()})
            row.update(mmlu.get(r["label"], {}))
            rows.append(row)

    if not rows:
        raise SystemExit(f"no input rows found; missing: {missing}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    cols = (["source", "label", "kind", "method", "n", "seed", "svd_rank", "probe_author",
             "headline", "model_utility", "mu_gmean"] + MU_COMPONENTS +
            ["first_zero_component", "retain_ppl", "forget_ppl", "forget_quality",
             "mmlu_acc", "mmlu_acc_se", "mmlu_pred_letter_entropy", "mmlu_mean_letter_entropy",
             "mmlu_pred_hist", "mmlu_n",
             "norm_fro_sum", "norm_kappa", "norm_rel_pert_slot_mean", "norm_rel_pert_slot_max",
             "norm_lam_weight", "norm_sqrt_n", "flags"])
    allcols = cols + [c for c in rows[0] if c not in cols]
    with open(f"{args.out_prefix}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allcols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {args.out_prefix}_summary.csv ({len(rows)} rows)")

    if norms:
        with open(f"{args.out_prefix}_norms.csv", "w", newline="") as f:
            keys = ["method", "n", "seed"] + sorted(next(iter(norms.values())).keys())
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for (meth, n, seed), v in sorted(norms.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                w.writerow({"method": meth, "n": n, "seed": seed, **v})
        print(f"  wrote {args.out_prefix}_norms.csv ({len(norms)} rows)")
    else:
        missing.append(f"norms ({args.norms_json})")

    if mmlu:
        with open(f"{args.out_prefix}_mmlu.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label"] + sorted(next(iter(mmlu.values())).keys()))
            w.writeheader()
            for lab, v in sorted(mmlu.items()):
                w.writerow({"label": lab, **v})
        print(f"  wrote {args.out_prefix}_mmlu.csv ({len(mmlu)} rows)")
    else:
        missing.append(f"mmlu ({args.mmlu_glob})")

    contrib = load_contrib(args.contrib_glob)
    if contrib:
        with open(f"{args.out_prefix}_contrib.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(contrib[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(contrib)
        print(f"  wrote {args.out_prefix}_contrib.csv ({len(contrib)} rows)")
        bad = [c for c in contrib
               if c.get("decomposition_rel_err") is not None
               and c["decomposition_rel_err"] > 1e-4]
        if bad:
            print(f"  [check] WARNING {len(bad)} contrib runs failed the exactness check")
    else:
        missing.append(f"contrib ({args.contrib_glob})")

    # ---- console: the ladders side by side ----
    for meth in sorted({r["method"] for r in rows if r["kind"] == "merge"}):
        lad = sorted([r for r in rows if r["method"] == meth and r["headline"] in ("True", True)],
                     key=lambda r: (r["n"] or 0, r["seed"] or 0))
        if not lad:
            continue
        print(f"\n{meth} (headline probe):")
        print(f"  {'N':>4} {'mu':>8} {'gmean':>8} {'r_ppl':>8} {'mmlu':>7} "
              f"{'rel_pert':>9} {'kappa':>7}  flags")
        for r in lad:
            print(f"  {r['n']:>4} {r['model_utility'] if r['model_utility'] is not None else 'na':>8} "
                  f"{r['mu_gmean'] if r['mu_gmean'] is not None else 'na':>8} "
                  f"{r['retain_ppl'] if r['retain_ppl'] is not None else 'na':>8} "
                  f"{round(r['mmlu_acc'], 4) if r.get('mmlu_acc') is not None else 'na':>7} "
                  f"{round(r['norm_rel_pert_slot_mean'], 5) if r.get('norm_rel_pert_slot_mean') is not None else 'na':>9} "
                  f"{round(r['norm_kappa'], 4) if r.get('norm_kappa') is not None else 'na':>7}"
                  f"  {r['flags']}")

    if missing:
        # never let an absent input read as "covered"
        print("\n[check] MISSING inputs (rows above are partial): " + "; ".join(missing))


if __name__ == "__main__":
    main()
