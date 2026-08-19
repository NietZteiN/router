#!/usr/bin/env python3
"""Figures for the deletion-size ladder (§18.4 of SELECTOR_AUDIT_REPORT.md).

Reads only the two committed JSONs — `reports/deletion_size_ladder.json` (routing metrics, CPU
sweep) and `reports/plain_ft_q4q5.json` (serving metrics + the routerless reference) — so the
figures cannot drift from the tables they sit beside.

⚠ matplotlib is NOT in the runtime venv. Run this with the plot interpreter:
    $TOFU_PLOT_PYTHON tofu_sisa_lora/plot_deletion_size.py

Design constraints, applied deliberately:
  * ONE y-axis per figure. Metrics on different scales get their own chart, never a second axis.
  * Two series per chart, coloured by IDENTITY in fixed slot order — slot 1 blue = questions that
    name the author, slot 2 orange = name removed. The same entity keeps the same hue in every
    figure, so a reader can carry the mapping across the set.
  * x is LINEAR in authors deleted, not equal-spaced by rung. The crowding at the low end is real
    and is part of the finding: nothing happens until the deletions accumulate.
  * Legend always present; endpoints direct-labelled only. Solid hairline grid, no dashes.
  * Text wears ink colours, never the series colour.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
OUT = os.path.join(REPORTS, "figures", "deletion_size")

# Reference palette, light mode. Slots 1-3 are the documented all-pairs-safe subset.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
S1 = "#2a78d6"   # names the author
S2 = "#eb6834"   # name removed
S3 = "#1baf7a"   # third series where one is needed

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.axisbelow": True,
})


def _frame(ax, title, ylabel, xs):
    ax.set_title(title, fontsize=12.5, color=INK, pad=12, loc="left")
    ax.set_xlabel("authors deleted (of 200)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)


def _line(ax, xs, ys, color, label, tail=None):
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
    if tail is not None:                       # selective direct label: the endpoint only
        ax.annotate(tail, xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10.5, color=INK, zorder=4)


def _save(fig, name, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=8.8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.045 if note else 0, 1, 1))
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(p, os.path.dirname(HERE))}")


# The ladder file runs to 80, but past 20 the extra deleted authors have NO rows in the 800-row
# evaluation set, so orphan counts stop growing and those rungs are flagged in the markdown table.
# Plotting them unmarked would hand the endpoint label to a rung that is not comparable, so the
# figures stop where the evaluation set covers the deletion. The larger rungs stay in the JSON.
MAX_DELETED = 20


def series(cells, strategy, cond, key):
    rows = [r for r in cells[strategy][cond] if r["n_deleted"] <= MAX_DELETED]
    xs = [r["n_deleted"] for r in rows if r.get(key) is not None]
    ys = [r[key] for r in rows if r.get(key) is not None]
    return xs, ys


def main() -> int:
    lad = json.load(open(os.path.join(REPORTS, "deletion_size_ladder.json")))["cells"]
    q45 = json.load(open(os.path.join(REPORTS, "plain_ft_q4q5.json")))
    os.makedirs(OUT, exist_ok=True)
    STRAT = "centroid_sbert"
    NAMED, STRIP = "questions name the author", "name removed"
    print("[figs] centroid_sbert, deletion order 180-199")

    # 1 — RDR. The headline: collateral damage to users nobody asked to delete.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs, ys = series(lad, STRAT, "original", "rdr"); _line(ax, xs, ys, S1, NAMED, f"{ys[-1]:.3f}")
    xs, ys = series(lad, STRAT, "name_stripped", "rdr")
    _line(ax, xs, ys, S2, STRIP, f"{ys[-1]:.3f}")
    _frame(ax, "Deleting more sources displaces more retained traffic — only without names",
           "retained displacement rate (RDR)", xs)
    ax.set_ylim(-0.004, max(ys) * 1.25)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    _save(fig, "fig1_rdr_vs_deletion_size.png",
          "RDR = share of RETAINED queries whose expert changed because of a deletion nobody "
          "asked for. 0 = perfectly local.")

    # 2 — routing accuracy on retained rows.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs, ys = series(lad, STRAT, "original", "routing_accuracy_retain")
    _line(ax, xs, ys, S1, NAMED, f"{ys[-1]:.3f}")
    xs, ys = series(lad, STRAT, "name_stripped", "routing_accuracy_retain")
    _line(ax, xs, ys, S2, STRIP, f"{ys[-1]:.3f}")
    _frame(ax, "Retained queries keep reaching their own expert — unless the name is gone",
           "routing accuracy, retained rows", xs)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="center left", labelcolor=INK2)
    _save(fig, "fig2_routing_accuracy_vs_deletion_size.png",
          "Measured post-deletion, over surviving units only, on rows whose own author survives.")

    # 3 — served answer quality, retained users, against the routerless reference.
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    def rung(c, k="retain_rouge"):
        rs = q45["ladder"][c]
        return [r["n_deleted"] for r in rs], [r[k] for r in rs]
    xs, ys = rung("none"); _line(ax, xs, ys, S1, NAMED, f"{ys[-1]:.3f}")
    xs, ys = rung("name_stripped"); _line(ax, xs, ys, S2, STRIP, f"{ys[-1]:.3f}")
    ref = {r["condition"]: r for r in q45["q4"]["rows"]}
    for cond, col in (("original", S1), ("name_stripped", S2)):
        v = ref[cond]["ft_retain"]
        ax.axhline(v, color=col, linewidth=1, alpha=0.45, zorder=1)
        ax.annotate(f"plain fine-tune, no router: {v:.3f}", xy=(xs[0], v), xytext=(0, 5),
                    textcoords="offset points", fontsize=9.5, color=INK2, va="bottom")
    _frame(ax, "What retained users actually get, as deletions accumulate",
           "ROUGE-L recall vs own gold answer", xs)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="center left", labelcolor=INK2)
    _save(fig, "fig3_retained_quality_vs_deletion_size.png",
          "Horizontal lines are the routerless control, which deletes nothing and so does not "
          "move with the ladder.")

    # 4 — orphan dispersal. The magnet prediction, refuted as a curve.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs, ys = series(lad, STRAT, "original", "orphan_n_eff")
    _line(ax, xs, ys, S1, NAMED, f"{ys[-1]:.0f}")
    xs, ys = series(lad, STRAT, "name_stripped", "orphan_n_eff")
    _line(ax, xs, ys, S2, STRIP, f"{ys[-1]:.0f}")
    _frame(ax, "Orphans disperse rather than piling onto one magnet expert",
           "effective number of destinations (1/HHI)", xs)
    ax.set_ylim(0, max(ys) * 1.2)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
    _save(fig, "fig4_orphan_dispersal_vs_deletion_size.png",
          "1.0 would mean one expert answers every orphaned query.")

    # 5 — detectability.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs, ys = series(lad, STRAT, "original", "detection_auc")
    _line(ax, xs, ys, S1, NAMED, f"{ys[-1]:.3f}")
    xs, ys = series(lad, STRAT, "name_stripped", "detection_auc")
    _line(ax, xs, ys, S2, STRIP, f"{ys[-1]:.3f}")
    ax.axhline(0.5, color=INK2, linewidth=1, alpha=0.4, zorder=1)
    ax.annotate("chance", xy=(xs[0], 0.5), xytext=(0, 5), textcoords="offset points",
                fontsize=9.5, color=INK2, va="bottom")
    _frame(ax, "Whether a deletion is detectable does not depend on how much was deleted",
           "orphan-vs-retained detection AUC", xs)
    ax.set_ylim(0.4, 1.05)
    ax.legend(frameon=False, loc="center right", labelcolor=INK2)
    _save(fig, "fig5_detection_auc_vs_deletion_size.png",
          "No deletion record consulted; survivor scores only. Rungs with too few orphans on the "
          "held-out half are omitted.")

    # 6 — the attack is size-independent.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rs = q45["ladder"]["name_swapped"]
    xs = [r["n_deleted"] for r in rs]
    ys = [r["routing_capture"] for r in rs]
    _line(ax, xs, ys, S1, "routing capture", f"{ys[-1]:.3f}")
    ys = [r["attacker_fact_rate"] for r in rs]
    _line(ax, xs, ys, S3, "attacker's facts in the answer", f"{ys[-1]:.3f}")
    _frame(ax, "The name-substitution attack does not care how much was deleted",
           "rate", xs)
    ax.set_ylim(0, 1.0)
    # Both series are flat, so the usual legend corners sit ON a line. Park it in the empty
    # band between them instead.
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.74),
              labelcolor=INK2)
    _save(fig, "fig6_attack_vs_deletion_size.png",
          "The attacker's own expert always survives, so the survivor pool's size is irrelevant "
          "to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
