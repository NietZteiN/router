"""N-merge interference figures (CPU, reads only the analyze_nmerge.py CSVs).

  ${TOFU_PLOT_PYTHON:-python3} plot_nmerge.py [--reports DIR] [--out DIR]

⚠ matplotlib lives ONLY in the base anaconda python (${TOFU_PLOT_PYTHON:-python3}),
not in test-env — run this script with base python. Everything upstream (merge, eval,
analyze) runs in test-env as usual.

Figures -> reports/figures/nmerge/ (PNG dpi 200 + PDF):
  fig1_mu_vs_N            headline model_utility vs N (log2 x), anchors as reference
                          lines, SVD-1024 points marked open, prior contiguous
                          k-scaling dare_ties points in gray (different axis: those
                          shards held 200/k authors each).
  fig2_own_recall_vs_N    probe-author forget_rouge vs N (faint per-probe trajectories,
                          bold mean, iso references at N=1) + forget_ppl panel. The H1 figure.
  fig3_mu_components_vs_N 3x3 small multiples ({retain,real,world} x {prob,rouge,truth})
                          — separates retain-coverage gain from interference damage (H2).
  fig4_geometry           col(B) overlap + shared-basis energy vs N, and the per-probe
                          drop-vs-overlap scatter with Spearman rho (H3).

Renders whatever rows exist — safe to run mid-campaign on partial results.
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

HERE = os.path.dirname(os.path.abspath(__file__))

# dataviz reference palette (validated; see the skill's palette.md)
BLUE = "#2a78d6"      # series 1: additive_mean
AQUA = "#1baf7a"      # series 2: dare_ties (direct-labeled — low contrast on light)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
GRAY = "#8a897f"      # anchors / nulls / prior-axis points

LADDER_TICKS = [1, 2, 4, 8, 16, 32, 64, 128, 200]


def style_axis(ax, log2x=True):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if log2x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(LADDER_TICKS)
        ax.set_xticklabels([str(t) for t in LADDER_TICKS])
        ax.minorticks_off()


def hline(ax, y, text, ls="--", side="right", va="bottom"):
    if y is None or (isinstance(y, float) and math.isnan(y)):
        return
    ax.axhline(y, color=GRAY, linewidth=1.0, linestyle=ls, zorder=1)
    x, ha = (1, "right") if side == "right" else (0, "left")
    ax.annotate(f" {text} ({y:.3f}) ", xy=(x, y), xycoords=("axes fraction", "data"),
                ha=ha, va=va, fontsize=7, color=INK2)


def load(reports):
    def rd(name):
        p = os.path.join(reports, name)
        return pd.read_csv(p) if os.path.exists(p) else None
    return rd("nmerge_mu.csv"), rd("nmerge_own_recall.csv"), rd("nmerge_overlap.csv")


def add_curve(mu):
    """additive_mean headline curve: exact rows where they exist, svd rows at N without
    an exact materialization. Returns df with n, model_utility, is_svd (sorted by n)."""
    m = mu[(mu["kind"] == "merge") & (mu["method"] == "additive_mean") & mu["headline"]]
    rows = []
    for n, grp in m.groupby("n"):
        exact = grp[grp["svd_rank"].isna() | (grp["svd_rank"] == "")]
        pick = exact.iloc[0] if len(exact) else grp.iloc[0]
        rows.append({"n": int(n), "model_utility": pick["model_utility"],
                     "is_svd": bool(len(exact) == 0)})
    iso = mu[(mu["kind"] == "iso") & mu["headline"]]
    if len(iso):  # the N=1 point = the headline probe's own adapter served alone
        rows.append({"n": 1, "model_utility": iso.iloc[0]["model_utility"], "is_svd": False})
    return pd.DataFrame(rows).sort_values("n") if rows else pd.DataFrame()


def anchors_of(mu):
    a = {}
    for name in ("base_model", "ft_r32", "retain90_oracle"):
        r = mu[(mu["label"] == name)]
        if len(r):
            a[name] = float(r.iloc[0]["model_utility"])
    return a


def prior_kscaling_points():
    """Prior contiguous k-scaling merged_dare_ties (200/k authors per shard) — a
    DIFFERENT axis, drawn gray for context only. Best-effort from the root CSV."""
    path = os.path.join(os.environ["TOFU_CKPT_ROOT"], "all_metrics_smoke.csv")
    try:
        df = pd.read_csv(path)
        rows = df[(df["label"] == "merged_dare_ties")
                  & df["model_slug"].astype(str).str.startswith("Llama-2-7B-chat-hf")
                  & df["k"].notna()]
        pts = rows.groupby("k")["model_utility"].max().reset_index()
        return pts[pts["k"] >= 2]
    except Exception:
        return None


def fig1(mu, out):
    curve = add_curve(mu)
    if curve.empty:
        print("[fig1] no additive_mean rows yet — skipped")
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    style_axis(ax)
    prior = prior_kscaling_points()
    if prior is not None and len(prior):
        ax.plot(prior["k"], prior["model_utility"], "o", color=GRAY, markersize=4,
                alpha=0.7, zorder=2)
        ax.annotate("prior k-scaling dare_ties\n(200/k authors per shard)",
                    xy=(prior["k"].iloc[0], prior["model_utility"].iloc[0]),
                    xytext=(10, 4), textcoords="offset points",
                    ha="left", fontsize=7, color=INK2)
    ax.plot(curve["n"], curve["model_utility"], "-", color=BLUE, linewidth=2, zorder=3)
    ex = curve[~curve["is_svd"]]
    sv = curve[curve["is_svd"]]
    ax.plot(ex["n"], ex["model_utility"], "o", color=BLUE, markersize=6, zorder=4)
    if len(sv):
        ax.plot(sv["n"], sv["model_utility"], "s", markerfacecolor="white",
                markeredgecolor=BLUE, markeredgewidth=1.5, markersize=7, zorder=4)
        ax.annotate("open marker = SVD-1024 materialization", xy=(0.02, 0.97),
                    xycoords="axes fraction", ha="left", va="top", fontsize=7, color=INK2)
    ax.annotate("additive_mean (true 1/N mean)", xy=(8, curve["model_utility"].max()),
                xytext=(0, 14), textcoords="offset points", fontsize=8, color=BLUE)
    cc = mu[(mu["kind"] == "merge") & (mu["method"] == "dare_ties") & (mu["r8"] == True)]  # noqa: E712
    if len(cc):
        y = float(cc.iloc[0]["model_utility"])
        ax.plot([200], [y], "D", color=AQUA, markersize=6, zorder=4)
        ax.annotate(f"dare_ties r8\ncross-check ({y:.3f})", xy=(200, y),
                    xytext=(-14, -2), textcoords="offset points", ha="right", va="top",
                    fontsize=7, color=INK2)
    a = anchors_of(mu)
    hline(ax, a.get("ft_r32"), "full-data ft (r32)")
    hline(ax, a.get("retain90_oracle"), "retain90 oracle", ls=":")
    hline(ax, a.get("base_model"), "base model", side="left", va="top")
    ax.set_xlabel("N per-author LoRAs merged (log2)", fontsize=9, color=INK)
    ax.set_ylabel("model_utility (smoke)", fontsize=9, color=INK)
    ax.set_title("Utility vs number of merged per-author LoRAs — Llama-2-7B, seed 42",
                 fontsize=10, color=INK)
    save(fig, out, "fig1_mu_vs_N")


def fig2(recall, out):
    m = recall[(recall["kind"] == "merge") & (recall["method"] == "additive_mean")]
    if m.empty:
        print("[fig2] no probe rows yet — skipped")
        return
    # exact rows win at each (n, probe); svd fills 128/200
    m = m.copy()
    m["svd"] = m["svd_rank"].notna() & (m["svd_rank"].astype(str) != "")
    m = (m.sort_values("svd").groupby(["n", "probe_author"], as_index=False).first())
    iso = recall[recall["kind"] == "iso"].set_index("probe_author")
    anch = recall[recall["kind"] == "anchor"].set_index("label")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, field, ttl in ((axes[0], "forget_rouge", "own-author ROUGE-L recall"),
                           (axes[1], "forget_ppl", "own-author perplexity")):
        style_axis(ax)
        for a, grp in m.groupby("probe_author"):
            g = grp.sort_values("n")
            xs, ys = [ ], [ ]
            if a in iso.index and not pd.isna(iso.loc[a, field]):
                xs, ys = [1], [iso.loc[a, field]]
            xs, ys = xs + g["n"].tolist(), ys + g[field].tolist()
            ax.plot(xs, ys, "-", color=BLUE, alpha=0.25, linewidth=1)
        mean = m.groupby("n")[field].mean()
        iso_mean = iso[field].mean() if field in iso else float("nan")
        xs = [1] + mean.index.tolist() if not math.isnan(iso_mean) else mean.index.tolist()
        ys = ([iso_mean] + mean.tolist()) if not math.isnan(iso_mean) else mean.tolist()
        ax.plot(xs, ys, "o-", color=BLUE, linewidth=2, markersize=5, zorder=4)
        if field == "forget_ppl":
            ax.set_yscale("log")
        if "base_model" in anch.index and not pd.isna(anch.loc["base_model", field]):
            hline(ax, float(anch.loc["base_model", field]), "base model (probe a82)",
                  side="left", va="top")
        if "ft_r32" in anch.index and not pd.isna(anch.loc["ft_r32", field]):
            hline(ax, float(anch.loc["ft_r32", field]), "full-data ft", ls=":")
        ax.set_xlabel("N merged (log2)", fontsize=9, color=INK)
        ax.set_title(ttl, fontsize=9, color=INK)
    axes[0].set_ylabel("probe forget_rouge", fontsize=9, color=INK)
    axes[1].set_ylabel("probe forget_ppl (log)", fontsize=9, color=INK)
    fig.suptitle("Own-author recall of the SAME 5 probe adapters as more authors are co-merged",
                 fontsize=10, color=INK)
    save(fig, out, "fig2_own_recall_vs_N")


def fig3(mu, out):
    m = mu[(mu["kind"] == "merge") & (mu["method"] == "additive_mean") & mu["headline"]]
    if m.empty:
        print("[fig3] no headline rows yet — skipped")
        return
    m = m.copy()
    m["svd"] = m["svd_rank"].notna() & (m["svd_rank"].astype(str) != "")
    m = m.sort_values("svd").groupby("n", as_index=False).first().sort_values("n")
    groups = ["retain", "real", "world"]
    metrics = ["prob", "rouge", "truth_scaled"]
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 7.2), sharex=True)
    for i, g in enumerate(groups):
        for j, met in enumerate(metrics):
            ax = axes[i][j]
            style_axis(ax)
            col = f"{g}_{met}"
            ax.plot(m["n"], m[col], "o-", color=BLUE, linewidth=1.6, markersize=4)
            if i == 0:
                ax.set_title(met, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(g, fontsize=9, color=INK)
            if i == 2:
                ax.set_xlabel("N merged", fontsize=8, color=INK2)
    fig.suptitle("model_utility components vs N — retain coverage rises, does real/world pay?",
                 fontsize=10, color=INK)
    save(fig, out, "fig3_mu_components_vs_N")


def fig4(ov, out):
    if ov is None or ov.empty:
        print("[fig4] no overlap CSV yet — skipped")
        return
    per_n = ov.drop_duplicates("n").sort_values("n")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax = axes[0]
    style_axis(ax)
    ax.plot(per_n["n"], per_n["angB_offdiag_mean"], "o-", color=BLUE, linewidth=2,
            markersize=5)
    ax.plot(per_n["n"], per_n["angB_null_orth_mean"], "--", color=GRAY, linewidth=1.2)
    ax.annotate("random-orthogonal null", xy=(per_n["n"].iloc[-1], per_n["angB_null_orth_mean"].iloc[-1]),
                xytext=(0, 6), textcoords="offset points", ha="right", fontsize=7, color=INK2)
    ax.set_ylabel("mean col(B) principal-angle cos", fontsize=8, color=INK)
    ax.set_title("output-subspace overlap", fontsize=9, color=INK)

    ax = axes[1]
    style_axis(ax)
    ax.plot(per_n["n"], per_n["shared_energy_mean"], "o-", color=BLUE, linewidth=2, markersize=5)
    ax.plot(per_n["n"], per_n["shared_energy_chance"], "--", color=GRAY, linewidth=1.2)
    ax.annotate("chance (rank ratio)", xy=(per_n["n"].iloc[-1], per_n["shared_energy_chance"].iloc[-1]),
                xytext=(0, 6), textcoords="offset points", ha="right", fontsize=7, color=INK2)
    ax.set_ylabel("shared rank-32 basis energy", fontsize=8, color=INK)
    ax.set_title("collective compressibility", fontsize=9, color=INK)

    ax = axes[2]
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    pts = ov[ov["drop_rouge"].notna()]
    if len(pts):
        ns = pts["n"].astype(float)
        shade = 0.25 + 0.75 * (np.log2(ns) - np.log2(ns.min())) / max(
            np.log2(ns.max()) - np.log2(ns.min()), 1e-9)
        for (_, r), s in zip(pts.iterrows(), shade):
            ax.plot(r["probe_angB_rowmean"], r["drop_rouge"], "o", color=BLUE,
                    alpha=float(s), markersize=6)
        rho = spearman(pts["probe_angB_rowmean"].to_numpy(), pts["drop_rouge"].to_numpy())
        ax.set_title(f"per-probe drop vs col(B) overlap (ρ={rho:.2f})", fontsize=9, color=INK)
        ax.annotate("darker = larger N", xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=7, color=INK2, va="top")
    else:
        ax.set_title("per-probe drop vs col(B) overlap (no joined rows yet)",
                     fontsize=9, color=INK)
    ax.set_xlabel("probe mean col(B) overlap w/ co-merged", fontsize=8, color=INK)
    ax.set_ylabel("recall drop (iso − merged rouge)", fontsize=8, color=INK)
    for a in axes[:2]:
        a.set_xlabel("N merged (log2)", fontsize=8, color=INK2)
    fig.suptitle("Geometry vs N and the geometry→behavior test (H3)", fontsize=10, color=INK)
    save(fig, out, "fig4_geometry")


def fig5(reports, out):
    p = os.path.join(reports, "nmerge_subset_mu.csv")
    if not os.path.exists(p):
        print("[fig5] no nmerge_subset_mu.csv yet — skipped")
        return
    df = pd.read_csv(p)
    curve = df[pd.to_numeric(df["n"], errors="coerce").notna()].copy()
    curve["n"] = curve["n"].astype(int)
    curve = curve.sort_values("n")
    anch = df[df["label"].astype(str).str.contains("_sub", na=False)].set_index("label")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, field, ttl, logy in ((axes[0], "retain_prob", "answer probability", False),
                                 (axes[1], "retain_rouge", "ROUGE-L recall", False),
                                 (axes[2], "retain_ppl", "perplexity", True)):
        style_axis(ax)
        ax.plot(curve["n"], curve[field], "o-", color=BLUE, linewidth=2, markersize=5, zorder=4)
        sv = curve[curve["svd_rank"].notna() & (curve["svd_rank"].astype(str) != "")]
        if len(sv):
            ax.plot(sv["n"], sv[field], "s", markerfacecolor="white",
                    markeredgecolor=BLUE, markeredgewidth=1.5, markersize=7, zorder=5)
        if logy:
            ax.set_yscale("log")
        for lab, txt, side in (("ft_r32_sub8", "joint ft (subset rows)", "right"),
                               ("base_model_sub8", "base (subset rows)", "left")):
            if lab in anch.index and not pd.isna(anch.loc[lab, field]):
                hline(ax, float(anch.loc[lab, field]), txt,
                      ls=":" if lab.startswith("ft") else "--", side=side,
                      va="top" if lab.startswith("base") else "bottom")
        ax.set_xlabel("N merged (log2)", fontsize=8, color=INK2)
        ax.set_title(f"subset retain {ttl}", fontsize=9, color=INK)
    fig.suptitle("Utility ON THE MERGED AUTHORS ONLY vs N — did the merge learn what it "
                 "was trained on?", fontsize=10, color=INK)
    save(fig, out, "fig5_subset_utility_vs_N")


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)
    except Exception:
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        rx = (rx - rx.mean()) / (rx.std() or 1)
        ry = (ry - ry.mean()) / (ry.std() or 1)
        return float((rx * ry).mean())


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}/{name}.png/.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", default=os.path.join(HERE, "reports"))
    ap.add_argument("--out", default=os.path.join(HERE, "reports", "figures", "nmerge"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mu, recall, ov = load(args.reports)
    if mu is None:
        raise SystemExit("no nmerge_mu.csv — run analyze_nmerge.py first")
    fig1(mu, args.out)
    if recall is not None:
        fig2(recall, args.out)
    fig3(mu, args.out)
    fig4(ov, args.out)
    fig5(args.reports, args.out)


if __name__ == "__main__":
    main()
