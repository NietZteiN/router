"""Experiment A figures: utility vs N under uniform summation.

    figA1   the headline — model utility, log retain perplexity, MMLU — three arms
    figA3   the norm ladder: ||Sigma||_F and kappa vs N (is growth sqrt(N) or N?)
    figA4   utility vs rel_pert, collapsing the sum and mean arms onto ONE axis

figA1 does NOT report model utility alone. `hmean` is a step function — one zero component
makes it exactly 0 — and 2 of its 9 components are saturated on this pool (real_rouge
0.982-0.992, world_rouge 0.90-0.94), so mu can sit flat while the model degrades and then fall
off a cliff. mu_gmean and log retain_ppl are plotted beside it for that reason.

figA4 is the strong form of the Experiment C prediction: if degradation is a function of the
injected perturbation MAGNITUDE alone, the sum and mean arms lie on one curve. If they separate,
the aggregation rule matters beyond magnitude and the matched-norm arm is what proves it.

Run under ${TOFU_PLOT_PYTHON} (matplotlib is absent from the pinned runtime env).

    "${TOFU_PLOT_PYTHON}" plot_expa.py --summary reports/expA/expA_summary.csv \
        --norms reports/expA/expA_norms.csv --out_dir reports/figures/expA
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import plot_style as S


def read_csv(p):
    if not p or not os.path.exists(p):
        print(f"  (absent: {p})")
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def arm_of(row):
    """Map a label/method to one of the three arms. The label token is authoritative:
    `sumisqrt` is the matched-norm arm, `sum` the unit sum, `add` the 1/N mean."""
    lab = row.get("label") or ""
    if "sumisqrt" in lab:
        return 1
    if "_sum" in lab or lab.startswith("nmerge_sum"):
        return 0
    if row.get("method") == "additive_mean" or "_add" in lab:
        return 2
    return None


def series_by_arm(rows, ycol, headline_only=True):
    """arm -> sorted [(N, mean_y, n_seeds)]. Seeds are the error bars, so they are averaged
    here and their spread returned; probe authors are NOT variance (real_*/world_* are provably
    identical across probes) and would fake precision if pooled as replicates."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("kind") != "merge":
            continue
        if headline_only and str(r.get("headline", "")).lower() not in ("true", "1", "yes", ""):
            continue
        a, n, y = arm_of(r), f(r.get("n")), f(r.get(ycol))
        if a is None or n is None or y is None:
            continue
        acc[a][int(n)].append(y)
    out = {}
    for a, byn in acc.items():
        pts = sorted((n, sum(v) / len(v),
                      (max(v) - min(v)) / 2 if len(v) > 1 else 0.0, len(v))
                     for n, v in byn.items())
        out[a] = pts
    return out


def anchors(rows, ycol):
    out = {}
    for r in rows:
        if r.get("kind") == "anchor":
            v = f(r.get(ycol))
            if v is not None:
                out.setdefault(r.get("label", "anchor"), v)
    return out


def _plot_arms(ax, series, ylabel, logy=False, mode=None):
    pal = S.palette(mode)
    drew = 0
    for arm in S.ARMS:
        i = arm["slot"]
        pts = series.get(i)
        if not pts:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es if any(es) else None, color=pal[i], marker="o",
                    ms=5, lw=2.0, capsize=0, elinewidth=1.0, label=arm["label"], zorder=3)
        S.direct_label(ax, xs[-1], ys[-1], arm["label"], pal[i], mode=mode)
        drew += 1
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("N adapters merged")
    ax.set_ylabel(ylabel)
    return drew


def fig_a1(rows, out_dir, mode=None):
    panels = [("model_utility", "Model utility (hmean of 9)", False),
              ("retain_ppl", "Retain perplexity (log)", True),
              ("mmlu_acc", "MMLU accuracy", False)]
    fig, axes = S.plt.subplots(1, 3, figsize=(13.5, 4.2))
    any_drawn = 0
    for ax, (col, ylab, logy) in zip(axes, panels):
        drew = _plot_arms(ax, series_by_arm(rows, col), ylab, logy, mode)
        any_drawn += drew
        if col == "model_utility":
            for lab, v in anchors(rows, col).items():
                S.anchor_line(ax, v, lab, mode)
            ax.axhspan(0.42, 0.47, color=S.ink(mode)["grid"], alpha=0.5, zorder=0)
            # Label the band ABOVE its top edge. Centred inside it, this text sat directly on
            # the data at the N where the arms cross the band.
            ax.annotate("0.42–0.47 merge band", xy=(0.02, 0.47),
                        xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", fontsize=8,
                        color=S.ink(mode)["muted"], va="bottom")
        if col == "mmlu_acc":
            S.anchor_line(ax, 0.25, "chance (4 letters)", mode)
        if not drew:
            ax.text(0.5, 0.5, f"no {col} rows yet", transform=ax.transAxes,
                    ha="center", va="center", color=S.ink(mode)["muted"], fontsize=9)
    # One legend for the figure — identity is never color-alone (there are direct labels too).
    h, l = axes[0].get_legend_handles_labels()
    if len(l) >= 2:
        fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Experiment A — utility vs N under uniform summation", y=1.12, fontsize=12)
    return S.finish(fig, os.path.join(out_dir, "figA1_utility_vs_N")) if any_drawn else None


def fig_a3(norms, out_dir, mode=None):
    """The magnitude law. kappa = ||Sigma|| / sqrt(sum ||delta_i||^2): 1 means the deltas are
    mutually orthogonal so the sum grows as sqrt(N); sqrt(N) would mean they are aligned and it
    grows as N. This is what turns 'utility falls' into 'utility falls at X% perturbation'."""
    if not norms:
        print("  (no norms rows — skipping figA3)")
        return None
    pal = S.palette(mode)
    by = defaultdict(list)
    for r in norms:
        n, fro, kap = f(r.get("n")), f(r.get("fro_sum")), f(r.get("kappa"))
        if n is None:
            continue
        by[r.get("method", "?")].append((int(n), fro, kap))
    fig, axes = S.plt.subplots(1, 2, figsize=(9.5, 4.2))
    for i, (meth, pts) in enumerate(sorted(by.items())):
        pts.sort()
        c = pal[i % len(pal)]
        xs = [p[0] for p in pts]
        for ax, idx, ylab in ((axes[0], 1, "‖Σ‖_F"), (axes[1], 2, "κ = ‖Σ‖ / √Σ‖Δᵢ‖²")):
            ys = [p[idx] for p in pts]
            if not any(v is not None for v in ys):
                continue
            ax.plot(xs, ys, color=c, marker="o", ms=5, lw=2.0, label=meth, zorder=3)
            ax.set_xscale("log", base=2); ax.set_xlabel("N adapters merged"); ax.set_ylabel(ylab)
            if ys[-1] is not None:
                S.direct_label(ax, xs[-1], ys[-1], meth, c, mode=mode)
    ns = sorted({p[0] for pts in by.values() for p in pts})
    if ns:
        base = next((p[1] for pts in by.values() for p in sorted(pts) if p[1]), None)
        if base:
            axes[0].plot(ns, [base * math.sqrt(n / ns[0]) for n in ns],
                         color=S.ink(mode)["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
            axes[0].annotate("√N reference", xy=(ns[-1], base * math.sqrt(ns[-1] / ns[0])),
                             xytext=(-4, 6), textcoords="offset points", ha="right",
                             fontsize=8, color=S.ink(mode)["muted"])
    axes[1].axhline(1.0, color=S.ink(mode)["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    axes[1].annotate("κ=1 ⇒ orthogonal (√N growth)", xy=(0.02, 1.0),
                     xycoords=("axes fraction", "data"), xytext=(0, 4),
                     textcoords="offset points", fontsize=8, color=S.ink(mode)["muted"])
    h, l = axes[0].get_legend_handles_labels()
    if len(l) >= 2:
        fig.legend(h, l, loc="upper center", ncol=min(len(l), 4), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Experiment A — the norm ladder", y=1.12, fontsize=12)
    return S.finish(fig, os.path.join(out_dir, "figA3_norms"))


def fig_a4(rows, out_dir, mode=None):
    """mu vs rel_pert. If the arms collapse onto one curve, damage is a function of injected
    magnitude alone — the strong form of the Exp-C prediction. Separation refutes it."""
    pal = S.palette(mode)
    pts_by_arm = defaultdict(list)
    for r in rows:
        if r.get("kind") != "merge":
            continue
        a = arm_of(r)
        x = f(r.get("norm_rel_pert_slot_mean"))
        y = f(r.get("model_utility"))
        if a is None or x is None or y is None:
            continue
        pts_by_arm[a].append((x, y, f(r.get("n"))))
    if not pts_by_arm:
        print("  (no rows with both rel_pert and model_utility — skipping figA4)")
        return None
    fig, ax = S.plt.subplots(figsize=(6.4, 4.6))
    for arm in S.ARMS:
        i = arm["slot"]
        pts = sorted(pts_by_arm.get(i, []))
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=pal[i], marker="o",
                ms=6, lw=1.6, label=arm["label"], zorder=3)
        S.direct_label(ax, pts[-1][0], pts[-1][1], arm["label"], pal[i], mode=mode)
        # N is the hidden third variable here; label the extremes so the curve is readable
        # without turning every point into a number.
        for p in (pts[0], pts[-1]):
            if p[2]:
                ax.annotate(f"N={int(p[2])}", xy=(p[0], p[1]), xytext=(0, -12),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=S.ink(mode)["muted"])
    ax.set_xlabel("rel_pert = ‖Σ‖_F / ‖W₀‖_F   (injected perturbation, fraction of base weights)")
    ax.set_ylabel("Model utility")
    ax.set_title("figA4 — does damage track magnitude alone?", fontsize=11)
    if len(ax.get_legend_handles_labels()[1]) >= 2:
        ax.legend(loc="best")
    return S.finish(fig, os.path.join(out_dir, "figA4_mu_vs_relpert"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="reports/expA/expA_summary.csv")
    ap.add_argument("--norms", default="reports/expA/expA_norms.csv")
    ap.add_argument("--out_dir", default="reports/figures/expA")
    ap.add_argument("--mode", default=None, choices=["light", "dark"],
                    help="chart surface; dark is a SELECTED set of steps, not an auto-flip")
    args = ap.parse_args()

    mode = args.mode or S.MODE
    S.apply_style(mode)
    rows, norms = read_csv(args.summary), read_csv(args.norms)
    print(f"[plot_expa] {len(rows)} summary rows, {len(norms)} norm rows -> {args.out_dir}")
    if not rows and not norms:
        raise SystemExit("nothing to plot — run analyze_expa.py first")
    os.makedirs(args.out_dir, exist_ok=True)
    if rows:
        fig_a1(rows, args.out_dir, mode)
        fig_a4(rows, args.out_dir, mode)
    fig_a3(norms, args.out_dir, mode)


if __name__ == "__main__":
    main()
