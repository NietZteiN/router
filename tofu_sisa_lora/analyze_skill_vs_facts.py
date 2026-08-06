"""Part B contrast: does merging spare skills but kill facts?

Reads the two eval_skill.py outputs (skills + facts), computes per-adapter normalized retention
R = (merged_nll - base_nll) / (isolated_nll - base_nll) — fraction of the adapter's NLL gain over
base that survives the N-way merge (1 = lossless, 0 = wiped to base) — and compares the two domains
(Mann-Whitney U, unpaired: skill tasks and fact shards are not naturally paired). Writes
reports/facts_vs_skills_retention.csv.

  python analyze_skill_vs_facts.py --skills reports/skill_nll_skills.json --facts reports/skill_nll_facts.json \
      --out reports/facts_vs_skills_retention.csv
"""
import argparse
import csv
import json

import numpy as np


def _rows(path):
    d = json.load(open(path))
    for r in d["rows"]:
        denom = r["isolated_nll"] - r["base_nll"]
        R = (r["merged_nll"] - r["base_nll"]) / denom if abs(denom) > 1e-9 else float("nan")
        yield {**r, "retention": R}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    allrows, bydom = [], {}
    for dom, path in (("skills", args.skills), ("facts", args.facts)):
        rs = list(_rows(path))
        allrows += rs
        bydom[dom] = [r["retention"] for r in rs if r["retention"] == r["retention"]]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "adapter", "name", "n_probes",
                                          "base_nll", "isolated_nll", "merged_nll", "retention"],
                           extrasaction="ignore")
        w.writeheader()
        for r in allrows:
            w.writerow(r)

    s, fa = bydom["skills"], bydom["facts"]
    print(f"[facts_vs_skills] wrote {args.out}")
    print(f"  skills retention: mean={np.mean(s):.4f} median={np.median(s):.4f} n={len(s)}")
    print(f"  facts  retention: mean={np.mean(fa):.4f} median={np.median(fa):.4f} n={len(fa)}")
    if s and fa:
        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(s, fa, alternative="greater")  # H1: skills retain MORE than facts
        print(f"  Mann-Whitney U (skills>facts): U={U:.1f} p={p:.4g}")
        verdict = ("SUPPORTS mechanism (skills retain more, p<0.05)" if p < 0.05
                   else "does NOT reach p<0.05 (mechanism unsupported at this scale)")
        print(f"  -> {verdict}; mean gap = {np.mean(s) - np.mean(fa):+.4f}")


if __name__ == "__main__":
    main()
