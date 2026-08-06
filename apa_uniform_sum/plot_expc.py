"""Experiment C figures: what an unowned query costs, and why.

    figC1  Δ-vs-base per query tier vs N, one panel per tier, three arms, with the √N and
           1/√N reference slopes drawn.
    figC2  attribution n_eff and the cancellation index, owned vs each unowned tier, vs N.

Δ-VS-BASE IS THE READOUT, not the absolute score. The tiers have wildly different absolute
scales (ROUGE on held-out authors vs perplexity on DBpedia), and none of them is interesting in
itself — the question is only how far the merge moved each away from the untouched base model.

THE REFERENCE SLOPES ARE THE TEST. The pre-registered prediction is that an unowned query
receives ≈√N·‖Δ₁‖ of pure noise under λ=1 and ≈(1/√N)·‖Δ₁‖ under 1/N, because attribution is
maximally diffuse (n_eff ≈ N) and the per-author deltas are near-orthogonal (mean |cos| ≈ 0.001).
Drawing those two slopes means the figure shows whether the data follow them, instead of leaving
the reader to infer a power law from a curve.

figC2 is the mechanism half. If n_eff is the same for owned and unowned queries, the pool does
no implicit routing at all and Experiment B's result is a TRAINING artifact rather than an
aggregation one — the pre-registered falsifier is attribution that concentrates (n_eff ≪ N) on
unowned queries.

Run under ${TOFU_PLOT_PYTHON}.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import plot_style as S

# tier key -> (panel title, the column holding its score)
TIERS = [
    ("heldout_authors", "C1 · held-out pool authors", "rouge"),
    ("holdout10",       "C2 · holdout10 (never trained)", "rouge"),
    ("real_world",      "C3 · real_authors + world_facts", "prob"),
    ("mmlu",            "C4 · MMLU + OOD perplexity", "mmlu_acc"),
]


def read_csv(p):
    if not os.path.exists(p):
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


def arm_of(label):
    if "sumisqrt" in (label or ""):
        return 1
    if "_sum" in (label or ""):
        return 0
    if "_add" in (label or ""):
        return 2
    return None


def fig_c1(summary, out_dir, mode=None):
    """Δ-vs-base per tier. Reads the Exp-A summary CSV, which already carries the per-condition
    tier columns (real_prob/world_prob/mmlu_acc) plus the extra-split rows."""
    pal, k = S.palette(mode), S.ink(mode)
    cols = {"heldout_authors": "tier_heldout_authors_rouge",
            "holdout10": "tier_holdout10_rouge",
            "real_world": "real_prob",
            "mmlu": "mmlu_acc"}
    base = {}
    for r in summary:
        if r.get("kind") == "anchor" and "base" in (r.get("label") or ""):
            for t, c in cols.items():
                v = f(r.get(c))
                if v is not None:
                    base[t] = v

    fig, axes = S.plt.subplots(1, len(TIERS), figsize=(4.2 * len(TIERS), 4.3), sharex=True)
    drew_any = 0
    for ax, (tier, title, _) in zip(axes, TIERS):
        col = cols[tier]
        pending = []
        acc = defaultdict(lambda: defaultdict(list))
        for r in summary:
            if r.get("kind") != "merge":
                continue
            a, n, y = arm_of(r.get("label")), f(r.get("n")), f(r.get(col))
            if a is None or n is None or y is None:
                continue
            acc[a][int(n)].append(y)
        b = base.get(tier)
        for arm in S.ARMS:
            i = arm["slot"]
            pts = sorted((n, sum(v) / len(v)) for n, v in acc.get(i, {}).items())
            if not pts:
                continue
            ys = [(p[1] - b) if b is not None else p[1] for p in pts]
            ax.plot([p[0] for p in pts], ys, color=pal[i], marker="o", ms=5, lw=2.0,
                    label=arm["label"], zorder=3)
            pending.append((pts[-1][0], ys[-1], arm["label"], pal[i]))
            drew_any += 1
        S.direct_labels(ax, pending, mode=mode)
        ax.set_xscale("log", base=2)
        ax.axhline(0.0, color=k["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("N adapters merged")
        if ax is axes[0]:
            ax.set_ylabel("Δ vs the untouched base model" if base else "score (no base anchor)")
        if not acc:
            ax.text(0.5, 0.5, f"no {col} rows yet", transform=ax.transAxes, ha="center",
                    va="center", color=k["muted"], fontsize=9)

    # The two pre-registered slopes, drawn once on the first panel that has data.
    ax0 = axes[0]
    lo, hi = ax0.get_xlim()
    if drew_any and hi > lo > 0:
        ns = [2 ** e for e in range(0, 8) if lo <= 2 ** e <= hi]
        span = abs(ax0.get_ylim()[1] - ax0.get_ylim()[0]) or 1.0
        unit = span / 12.0
        if ns:
            ax0.plot(ns, [-unit * math.sqrt(n) for n in ns], color=k["muted"], lw=1.0,
                     ls=(0, (4, 3)), zorder=1)
            ax0.annotate("√N (λ=1 prediction)", xy=(ns[-1], -unit * math.sqrt(ns[-1])),
                         xytext=(-4, -10), textcoords="offset points", ha="right",
                         fontsize=8, color=k["muted"])
            ax0.plot(ns, [-unit / math.sqrt(n) for n in ns], color=k["muted"], lw=1.0,
                     ls=(0, (1, 2)), zorder=1)
            ax0.annotate("1/√N (mean prediction)", xy=(ns[-1], -unit / math.sqrt(ns[-1])),
                         xytext=(-4, 8), textcoords="offset points", ha="right",
                         fontsize=8, color=k["muted"])
    h, l = axes[0].get_legend_handles_labels()
    if len(l) >= 2:
        fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.07))
    fig.suptitle("Experiment C — what a query no adapter owns costs", y=1.14, fontsize=12)
    return S.finish(fig, os.path.join(out_dir, "figC1_unowned_cost")) if drew_any else None


def fig_c2(contrib, out_dir, mode=None):
    """n_eff and the cancellation index, owned vs unowned. Two panels, never two y-axes."""
    if not contrib:
        print("  (no contrib rows — skipping figC2)")
        return None
    pal, k = S.palette(mode), S.ink(mode)

    def n_of(label):
        for tok in (label or "").replace("-", "_").split("_"):
            if tok.startswith("N") and tok[1:].isdigit():
                return int(tok[1:])
        return None

    labels0, labels1 = [], []
    groups = [("n_eff_own_orig", "owned (the author's own questions)", 0),
              ("n_eff_unowned_orig", "unowned pool authors", 1),
              ("n_eff_holdout10", "holdout10", 2)]
    cgroups = [("cancel_own_orig", "owned", 0),
               ("cancel_unowned_orig", "unowned", 1),
               ("cancel_holdout10", "holdout10", 2)]

    fig, axes = S.plt.subplots(1, 2, figsize=(10.5, 4.3))
    for col, lab, slot in groups:
        pts = sorted((n_of(r.get("label")), f(r.get(col))) for r in contrib
                     if n_of(r.get("label")) and f(r.get(col)) is not None)
        if not pts:
            continue
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], color=pal[slot], marker="o",
                     ms=5, lw=2.0, label=lab, zorder=3)
        labels0.append((pts[-1][0], pts[-1][1], lab, pal[slot]))
    ns = sorted({n_of(r.get("label")) for r in contrib if n_of(r.get("label"))})
    if ns:
        axes[0].plot(ns, ns, color=k["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        axes[0].annotate("n_eff = N ⇒ perfectly diffuse (no implicit routing)",
                         xy=(ns[-1], ns[-1]), xytext=(-4, -12), textcoords="offset points",
                         ha="right", fontsize=8, color=k["muted"])
    S.direct_labels(axes[0], labels0, mode=mode)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log", base=2)
    axes[0].set_xlabel("N adapters merged"); axes[0].set_ylabel("n_eff  (1 / HHI)")
    axes[0].set_title("attribution diffuseness", fontsize=10)

    for col, lab, slot in cgroups:
        pts = sorted((n_of(r.get("label")), f(r.get(col))) for r in contrib
                     if n_of(r.get("label")) and f(r.get(col)) is not None)
        if not pts:
            continue
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], color=pal[slot], marker="o",
                     ms=5, lw=2.0, label=lab, zorder=3)
        labels1.append((pts[-1][0], pts[-1][1], lab, pal[slot]))
    if ns:
        axes[1].plot(ns, [1 / math.sqrt(n) for n in ns], color=k["muted"], lw=1.0,
                     ls=(0, (4, 3)), zorder=1)
        axes[1].annotate("1/√N ⇒ mutually orthogonal contributions",
                         xy=(ns[-1], 1 / math.sqrt(ns[-1])), xytext=(-4, 8),
                         textcoords="offset points", ha="right", fontsize=8, color=k["muted"])
    S.direct_labels(axes[1], labels1, mode=mode)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("N adapters merged")
    axes[1].set_ylabel("‖Σcᵢ‖ / Σ‖cᵢ‖   (cancellation index)")
    axes[1].set_title("does the diffuse firing reach the residual stream?", fontsize=10)

    h, l = axes[0].get_legend_handles_labels()
    if len(l) >= 2:
        fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.07))
    fig.suptitle("figC2 — owned vs unowned attribution", y=1.14, fontsize=12)
    return S.finish(fig, os.path.join(out_dir, "figC2_attribution"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="reports/expA/expA_summary.csv")
    ap.add_argument("--contrib", default="reports/expA/expA_contrib.csv")
    ap.add_argument("--out_dir", default="reports/figures/expC")
    ap.add_argument("--mode", default=None, choices=["light", "dark"])
    args = ap.parse_args()

    mode = args.mode or S.MODE
    S.apply_style(mode)
    summary, contrib = read_csv(args.summary), read_csv(args.contrib)
    print(f"[plot_expc] {len(summary)} summary rows, {len(contrib)} contrib rows")
    if not (summary or contrib):
        raise SystemExit("nothing to plot — run analyze_expa.py first")
    os.makedirs(args.out_dir, exist_ok=True)
    if summary:
        fig_c1(summary, args.out_dir, mode)
    fig_c2(contrib, args.out_dir, mode)


if __name__ == "__main__":
    main()
