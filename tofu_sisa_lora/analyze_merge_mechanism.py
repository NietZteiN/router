"""Assemble the merge-mechanism CSVs from per-label eval JSONs (CPU, post-array).

  python analyze_merge_mechanism.py lambda --results DIR --k 10 --out reports/lambda_sweep_1b.csv
  python analyze_merge_mechanism.py iso    --results DIR --k 10 --lam 0.1 --out reports/iso_merged_drop.csv

`lambda`: reads merged_additive_s{λ}/remerge_additive_s{λ} (+ shard_{k-1}_only, merged_dare_ties
anchors) → utility/forget vs λ, to show no λ recovers recall.
`iso`: reads {label}__own{sid}.json → per-shard isolated (shard_i_only) vs merged recall on that
shard's own authors → Drop = isolated - merged.
"""
import argparse
import csv
import glob
import json
import os
import re

FIELDS = ["model_utility", "forget_quality", "forget_rouge", "forget_truth_ratio",
          "retain_rouge", "forget_ppl", "retain_ppl"]


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _row(d, **extra):
    r = dict(extra)
    for k in FIELDS:
        r[k] = d.get(k) if d else None
    return r


def do_lambda(results, k, out):
    lambdas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0]
    rows = []
    for lam in lambdas:
        s = f"{lam:g}"
        for arm in ("merged", "remerge"):
            d = _load(os.path.join(results, f"{arm}_additive_s{s}.json"))
            rows.append(_row(d, arm=arm, method="additive", lam=lam,
                             present=d is not None))
    for label in (f"shard_{k-1}_only", "merged_dare_ties", "merged_additive_mean"):
        d = _load(os.path.join(results, f"{label}.json"))
        if d is not None:
            rows.append(_row(d, arm=label, method="anchor", lam="", present=True))
    _write(out, rows, ["arm", "method", "lam", "present"] + FIELDS)
    # quick console summary
    present = [r for r in rows if r["present"] and r["arm"] == "merged"]
    if present:
        best = max(present, key=lambda r: r["model_utility"] or -1)
        print(f"[lambda] {len(present)}/{len(lambdas)} merged points present; "
              f"peak mu={best['model_utility']} at lam={best['lam']}")
    return rows


def do_iso(results, k, lam, out):
    merged_labels = [f"merged_additive_s{lam:g}", "merged_dare_ties"]
    rows = []
    for sid in range(k):
        iso = _load(os.path.join(results, f"shard_{sid}_only__own{sid}.json"))
        iso_rouge = iso.get("forget_rouge") if iso else None
        iso_tr = iso.get("forget_truth_ratio") if iso else None
        rows.append(_row(iso, k=k, sid=sid, condition="isolated", label=f"shard_{sid}_only"))
        for ml in merged_labels:
            m = _load(os.path.join(results, f"{ml}__own{sid}.json"))
            rr = _row(m, k=k, sid=sid, condition=f"merged:{ml}", label=ml)
            if iso_rouge is not None and m is not None and m.get("forget_rouge") is not None:
                rr["drop_rouge"] = iso_rouge - m["forget_rouge"]
                rr["drop_truth_ratio"] = (iso_tr - m["forget_truth_ratio"]
                                          if iso_tr is not None and m.get("forget_truth_ratio") is not None else None)
            rows.append(rr)
    _write(out, rows, ["k", "sid", "condition", "label", "drop_rouge", "drop_truth_ratio"] + FIELDS)
    # summary: mean drop per merged label
    for ml in merged_labels:
        drops = [r.get("drop_rouge") for r in rows if r["condition"] == f"merged:{ml}" and r.get("drop_rouge") is not None]
        if drops:
            print(f"[iso k={k}] {ml}: mean isolated-merged forget_rouge drop = {sum(drops)/len(drops):.4f} "
                  f"over {len(drops)} shards (all>0: {all(d>0 for d in drops)})")
    return rows


def _write(out, rows, cols):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    allcols = cols + [c for c in (rows[0].keys() if rows else []) if c not in cols]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allcols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lambda", "iso"):
        p = sub.add_parser(name)
        p.add_argument("--results", required=True)
        p.add_argument("--k", type=int, default=10)
        p.add_argument("--out", required=True)
        if name == "iso":
            p.add_argument("--lam", type=float, default=0.1)
    a = ap.parse_args()
    if a.cmd == "lambda":
        do_lambda(a.results, a.k, a.out)
    else:
        do_iso(a.results, a.k, a.lam, a.out)


if __name__ == "__main__":
    main()
