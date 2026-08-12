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

TWO bootstraps run here, and reading the wrong one reverses the conclusion:

  marginal — each arm resampled on its own. Answers "how well is THIS cell pinned down?" The
             per-cell CI. It is NOT the right yardstick for comparing arms.
  paired   — ONE index draw shared by every arm, because the arms score the SAME rows (the truth
             set is a deterministic head-slice of `max_rows`, so row i is the same question in
             every arm; observed inter-arm correlation 0.88-0.94 confirms it). The arms differ
             only in reroute destination, so the question-level noise is COMMON to all of them
             and cancels in a difference. Comparing a spread against marginal CIs silently
             re-adds that shared noise once per arm and can only ever say "not resolvable".

The spread is a paired quantity. Judge it with the paired bootstrap.

Usage:
  python bootstrap_fq.py --results_dir DIR --ks_ref PATH [--published_dir DIR]
                         [--baseline_arm NAME] [--n_boot 20000] [--seed 42] --out_json J --out_md M
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.stats import ks_2samp


def d_lattice_step(n, m):
    """EXACT spacing of the two-sample KS statistic: D = |i/n - j/m| = |i·m - j·n|/(n·m).

    Every attainable D is a multiple of 1/lcm(n, m), so this is the metric's true resolution and
    needs no sampling. At n=120, m=20 the step is 1/120 -- one forget question moves D by exactly
    that, which is the interpretable form of "how precise is this cell".
    """
    return 1.0 / float(np.lcm(int(n), int(m)))


def achievable_grid(n, m, n_draw=40000, seed=0):
    """Sample the p-values a two-sample KS on (n, m) rows can emit. Returns a LOWER BOUND.

    The p-value is a deterministic decreasing function of the discrete statistic D, so the set is
    finite -- but this recovers it by random draw, and rare extreme D values need many draws to
    appear: at (120, 20) the count still climbs 73 -> 88 going from 2k to 60k draws. So the size
    of this list is a lower bound that grows with n_draw and must NEVER be quoted as "the number
    of achievable p-values". Quote `d_lattice_step` for resolution (exact), and take the count
    above 0.05 and the median gap from here -- those are stable to +/-1 across seeds and draws
    because the readable range is where draws land most often.

    Sampling is still the right tool for the p-values themselves: scipy's exact helpers are
    private and their signature moves between versions (it does not accept what its callers
    pass here).
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


def paired_bootstrap(mat, ref, n_boot, rng):
    """Resample ROW INDICES once per iteration and apply that same draw to every arm.

    mat is [n_arms, n_rows] with column j the same question in every row. Sharing the draw is
    the whole point: it holds the question sample fixed across arms so a difference between two
    arms carries only the destination effect, not the sampling noise they have in common.
    """
    n_arms, n = mat.shape
    out = np.empty((n_boot, n_arms), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for a in range(n_arms):
            out[b, a] = ks_2samp(mat[a, idx], ref).pvalue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="dir holding the arms' *.forget_tr.npy sidecars (and their result JSONs)")
    ap.add_argument("--ks_ref", required=True, help="retain_tr_scores.npy the cells were scored against")
    ap.add_argument("--published_dir", default=None,
                    help="dir of the ORIGINAL result JSONs. Given, each arm's reproduced "
                         "forget_quality is checked against its published cell -- a rerun that "
                         "does not reproduce makes any interval on it meaningless.")
    ap.add_argument("--baseline_arm", default="routed_oracle_del_f10",
                    help="Arm every other arm is compared against in the PAIRED bootstrap -- the "
                         "genuine-deletion baseline. The paper's claim is 'a method that deletes "
                         "nothing scores at or above real deletion', so that comparison needs an "
                         "interval, not just two point estimates.")
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
    grids, rows, kept = {}, [], []
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
            "grid_size_lower_bound": len(grid), "grid_above_05": len(above),
            "grid_median_gap": float(np.nanmedian(gaps)),
            "d_lattice_step": d_lattice_step(n, m),
        })
        kept.append((arm, forget_tr))
        print(f"[ok] {arm}: fq={point:.4f} CI95=[{lo:.4f}, {hi:.4f}] width={hi-lo:.4f}")

    fqs = [r["forget_quality"] for r in rows]
    widths = [r["ci95_width"] for r in rows]
    repro = [r for r in rows if r["published"] is not None]
    n_repro = sum(r["reproduces_published"] for r in repro)
    spread = (max(fqs) - min(fqs)) if fqs else float("nan")

    # The paired half. The arms score identical rows, so the spread must be judged against a
    # bootstrap that holds the row sample common across arms -- comparing it to the MARGINAL CIs
    # counts the shared question-level noise once per arm and would call any spread unresolvable.
    paired = None
    lens = {len(a) for _, a in kept}
    if len(kept) >= 2 and len(lens) == 1:
        names = [a for a, _ in kept]
        mat = np.stack([a for _, a in kept])
        P = paired_bootstrap(mat, ref, args.n_boot, np.random.default_rng(args.seed + 1))
        sp = P.max(axis=1) - P.min(axis=1)
        paired = {"spread_ci95_lo": float(np.percentile(sp, 2.5)),
                  "spread_ci95_hi": float(np.percentile(sp, 97.5)),
                  "spread_median": float(np.median(sp)),
                  "p_spread_gt_0p10": float((sp > 0.10).mean()),
                  "p_spread_gt_0p25": float((sp > 0.25).mean())}
        if args.baseline_arm in names:
            b = names.index(args.baseline_arm)
            others = [i for i in range(len(names)) if i != b]
            paired["baseline_arm"] = args.baseline_arm
            paired["vs_baseline"] = [
                {"arm": names[i], "diff": float(fqs[i] - fqs[b]),
                 "diff_ci95_lo": float(np.percentile(P[:, i] - P[:, b], 2.5)),
                 "diff_ci95_hi": float(np.percentile(P[:, i] - P[:, b], 97.5)),
                 "p_at_or_above": float((P[:, i] >= P[:, b]).mean())} for i in others]
            cnt = (P[:, others] >= P[:, [b]]).sum(axis=1)
            paired["n_at_or_above_observed"] = int(sum(fqs[i] >= fqs[b] for i in others))
            paired["n_at_or_above_median"] = float(np.median(cnt))
            paired["n_at_or_above_ci95_lo"] = float(np.percentile(cnt, 2.5))
            paired["n_at_or_above_ci95_hi"] = float(np.percentile(cnt, 97.5))
            paired["n_compared"] = len(others)
    elif len(kept) >= 2:
        print(f"[warn] arms have differing row counts {sorted(lens)} -- rows are not paired, "
              f"so the paired bootstrap is skipped.")

    # The claim the table is FOR is about the SPREAD, so the verdict reads the paired bootstrap;
    # the marginal CIs stay in the table because "how well is one published cell pinned down" is
    # a separate and also-reportable question.
    if paired:
        verdict = (f"destination spread is resolvable: paired 95% CI "
                   f"[{paired['spread_ci95_lo']:.4f}, {paired['spread_ci95_hi']:.4f}], "
                   f"P(spread>0.25)={paired['p_spread_gt_0p25']:.3f}"
                   if paired["spread_ci95_lo"] > 0.10 else
                   f"destination spread is NOT resolvable even paired: 95% CI "
                   f"[{paired['spread_ci95_lo']:.4f}, {paired['spread_ci95_hi']:.4f}]")
    else:
        verdict = "no paired bootstrap (arms not row-comparable) -- marginal CIs only"

    out = {"results_dir": args.results_dir, "ks_ref": args.ks_ref, "n_boot": args.n_boot,
           "seed": args.seed, "n_arms": len(rows), "spread": spread,
           "max_ci_width": max(widths) if widths else float("nan"),
           "reproduced": f"{n_repro}/{len(repro)}", "verdict": verdict,
           "paired": paired, "arms": rows}
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
        L += ["", f"Resolution at n={g['n_forget']} vs m={g['n_ref']}: the KS statistic D moves on "
                  f"an exact lattice of step **{g['d_lattice_step']:.6f}** (= 1/lcm), i.e. one "
                  f"forget question. In p-value terms, **{g['grid_above_05']} values above 0.05** "
                  f"with median gap **{g['grid_median_gap']:.4f}** — so the 4 decimals every table "
                  f"reports are spurious. (A sampled ≥{g['grid_size_lower_bound']} distinct "
                  f"p-values overall; that count is a LOWER BOUND which grows with the number of "
                  f"draws, and is not quotable as an enumeration.) Resolution is a property of the "
                  f"sample sizes; the CI is a property of the data. They are not the same claim.",
              ""]
    L += [f"Spread across arms: **{spread:.4f}**. Widest MARGINAL CI: **{max(widths):.4f}**."
          if widths else "", ""]
    if paired:
        L += ["## Paired bootstrap (the arms score identical rows)", "",
              "Each resample draws ONE set of row indices and applies it to every arm, so the "
              "question-level noise the arms share cancels in a difference. The marginal CIs "
              "above are wide because they re-add that shared noise once per arm; they bound a "
              "single published cell, and are the wrong yardstick for a spread.", "",
              f"Spread: **{spread:.4f}**, paired 95% CI "
              f"[{paired['spread_ci95_lo']:.4f}, {paired['spread_ci95_hi']:.4f}] "
              f"(median {paired['spread_median']:.4f}); "
              f"P(spread>0.10) = **{paired['p_spread_gt_0p10']:.4f}**, "
              f"P(spread>0.25) = **{paired['p_spread_gt_0p25']:.4f}**.", ""]
        if "vs_baseline" in paired:
            L += [f"### Against genuine deletion (`{paired['baseline_arm']}`)", "",
                  "| arm | Δ vs deletion | paired 95% CI | P(arm ≥ deletion) |", "|---|---|---|---|"]
            for v in sorted(paired["vs_baseline"], key=lambda v: -v["diff"]):
                L.append(f"| `{v['arm']}` | {v['diff']:+.4f} | "
                         f"[{v['diff_ci95_lo']:+.4f}, {v['diff_ci95_hi']:+.4f}] | "
                         f"{v['p_at_or_above']:.4f} |")
            L += ["", f"Arms at or above genuine deletion: observed "
                      f"**{paired['n_at_or_above_observed']}/{paired['n_compared']}**, paired "
                      f"median {paired['n_at_or_above_median']:.0f}, 95% CI "
                      f"[{paired['n_at_or_above_ci95_lo']:.0f}, "
                      f"{paired['n_at_or_above_ci95_hi']:.0f}]. Every one of these arms deletes "
                      f"nothing.", ""]
    L += [f"**Verdict:** {verdict}", "",
          f"Reproduced published cells: **{n_repro}/{len(repro)}**." if repro else ""]
    open(args.out_md, "w").write("\n".join(L) + "\n")
    print(f"\n{verdict}\nwrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
