#!/usr/bin/env python3
"""H29 — put an interval on a published forget_quality cell.

forget_quality = ks_2samp(forget_tr, retain_ref).pvalue collapses a whole array to one number,
and H23 showed that number is a step function of a discrete statistic: at the smoke caps it can
take 34 distinct values, adjacent rungs in the readable range sit ~0.10 apart, and a 0.62 spread
across reroute destinations is four questions out of thirty. Two DIFFERENT things follow, and the
paper needs both separated:

  resolution  — which p-values the test can even emit at these sample sizes. A property of (n, m).
                Nothing about the model. Enumerated here by sampling the achievable grid.
  uncertainty — how much the cell would move if we had drawn a different forget sample. A property
                of the data. Estimated here by bootstrapping the forget array.

A cell can sit on a coarse grid and still be tightly determined, or sit on a fine grid and be
noise. Reporting one and calling it the other is the mistake this script exists to prevent.

The reference array is held FIXED at the published one, never resampled. Resampling both samples
would answer "how would this KS test behave under replication", which is a different and less
useful question than "how much does my forget sample pin this published cell down".

Usage:
  python bootstrap_fq.py --results_dir DIR --ks_ref PATH [--published_dir DIR]
                         [--n_boot 20000] [--seed 42] --out_json J --out_md M
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.stats import ks_2samp


def achievable_grid(n, m, n_draw=6000, seed=0):
    """Sample the p-values a two-sample KS on (n, m) rows can actually emit.

    The p-value is a deterministic function of the discrete statistic D, so the set is finite.
    Random draws recover it far more reliably than reaching into scipy's private exact helpers,
    whose signature moves between versions (it does not accept what its callers pass here).
    """
    rng = np.random.default_rng(seed)
    grid = {}
    for _ in range(n_draw):
        a = rng.normal(0.0, 1.0, n)
        b = rng.normal(rng.uniform(-1.0, 1.0), rng.uniform(0.5, 2.0), m)
        r = ks_2samp(a, b)
        grid[round(float(r.statistic), 10)] = float(r.pvalue)
    return sorted(grid.values())


def snap(p, grid, tol=5e-5):
    for g in grid:
        if abs(g - p) < tol:
            return g
    return None


def bootstrap_cell(forget_tr, ref, n_boot, rng):
    """Resample the FORGET array with replacement; the reference stays as published."""
    n = len(forget_tr)
    ps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ps[i] = ks_2samp(rng.choice(forget_tr, size=n, replace=True), ref).pvalue
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="dir holding the arms' *.forget_tr.npy sidecars (and their result JSONs)")
    ap.add_argument("--ks_ref", required=True, help="retain_tr_scores.npy the cells were scored against")
    ap.add_argument("--published_dir", default=None,
                    help="dir of the ORIGINAL result JSONs. Given, each arm's reproduced "
                         "forget_quality is checked against its published cell -- a rerun that "
                         "does not reproduce makes any interval on it meaningless.")
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    ref = np.load(args.ks_ref)
    sidecars = sorted(f for f in glob.glob(os.path.join(args.results_dir, "*.forget_tr.npy")))
    if not sidecars:
        raise SystemExit(f"no *.forget_tr.npy in {args.results_dir} -- the arms ran without the "
                         f"sidecar, so there is nothing to bootstrap.")

    rng = np.random.default_rng(args.seed)
    grids, rows = {}, []
    for sc in sidecars:
        arm = os.path.basename(sc)[: -len(".forget_tr.npy")]
        forget_tr = np.load(sc)
        n, m = len(forget_tr), len(ref)
        if n == 0:
            print(f"[skip] {arm}: empty forget array")
            continue
        if (n, m) not in grids:
            grids[(n, m)] = achievable_grid(n, m)
        grid = grids[(n, m)]

        point = float(ks_2samp(forget_tr, ref).pvalue)
        ps = bootstrap_cell(forget_tr, ref, args.n_boot, rng)
        lo, hi = (float(x) for x in np.percentile(ps, [2.5, 97.5]))

        pub = None
        if args.published_dir:
            pj = os.path.join(args.published_dir, arm + ".json")
            if os.path.exists(pj):
                pub = json.load(open(pj)).get("forget_quality")

        above = [p for p in grid if p > 0.05]
        gaps = np.diff(above) if len(above) > 1 else np.array([np.nan])
        rows.append({
            "arm": arm, "n_forget": n, "n_ref": m,
            "forget_quality": point,
            "ci95_lo": lo, "ci95_hi": hi, "ci95_width": hi - lo,
            "boot_distinct_values": int(len(set(np.round(ps, 10)))),
            "on_grid": snap(point, grid) is not None,
            "published": pub,
            "reproduces_published": (pub is not None and abs(pub - point) < 5e-4),
            "grid_size": len(grid), "grid_above_05": len(above),
            "grid_median_gap": float(np.nanmedian(gaps)),
        })
        print(f"[ok] {arm}: fq={point:.4f} CI95=[{lo:.4f}, {hi:.4f}] width={hi-lo:.4f}")

    fqs = [r["forget_quality"] for r in rows]
    widths = [r["ci95_width"] for r in rows]
    repro = [r for r in rows if r["published"] is not None]
    n_repro = sum(r["reproduces_published"] for r in repro)
    spread = (max(fqs) - min(fqs)) if fqs else float("nan")
    # The question the table is FOR: is the destination spread bigger than the uncertainty on any
    # single cell? If the widest CI swallows the spread, the spread is not a finding.
    verdict = ("spread exceeds every cell's CI width -- destination identity moves the metric "
               "beyond sampling noise" if widths and spread > max(widths) else
               "spread is within the widest cell CI -- the destination ordering is NOT resolvable "
               "at these sample sizes")

    out = {"results_dir": args.results_dir, "ks_ref": args.ks_ref, "n_boot": args.n_boot,
           "seed": args.seed, "n_arms": len(rows), "spread": spread,
           "max_ci_width": max(widths) if widths else float("nan"),
           "reproduced": f"{n_repro}/{len(repro)}", "verdict": verdict, "arms": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# H29 — confidence intervals on forget_quality", "",
         f"Bootstrap over the FORGET sample only ({args.n_boot} resamples, seed {args.seed}); the "
         f"KS reference is held fixed at the published `{os.path.basename(args.ks_ref)}` "
         f"({len(ref)} rows).", "",
         "| arm | n | fq | 95% CI | width | published | reproduces | on grid |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["forget_quality"]):
        pub = "—" if r["published"] is None else f"{r['published']:.4f}"
        rep = "—" if r["published"] is None else ("yes" if r["reproduces_published"] else "**NO**")
        L.append(f"| `{r['arm']}` | {r['n_forget']} | {r['forget_quality']:.4f} | "
                 f"[{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}] | {r['ci95_width']:.4f} | {pub} | "
                 f"{rep} | {'yes' if r['on_grid'] else 'no'} |")
    if rows:
        g = rows[0]
        L += ["", f"Resolution at n={g['n_forget']} vs m={g['n_ref']}: **{g['grid_size']} achievable "
                  f"p-values, {g['grid_above_05']} above 0.05**, median gap "
                  f"{g['grid_median_gap']:.4f}. Resolution is a property of the sample sizes; the "
                  f"CI is a property of the data. They are not the same claim.", ""]
    L += [f"Spread across arms: **{spread:.4f}**. Widest CI: **{max(widths):.4f}**." if widths else "",
          "", f"**Verdict:** {verdict}", "",
          f"Reproduced published cells: **{n_repro}/{len(repro)}**." if repro else ""]
    open(args.out_md, "w").write("\n".join(L) + "\n")
    print(f"\n{verdict}\nwrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
