"""Experiment B figure: what deletion actually changed, per target author.

    figB1  a slope chart — original vs paraphrase surface, one connected line per target,
           for the post-deletion model, with the pre-deletion model as open markers and the
           iso ceiling / base floor annotated per surface.

WHY A SLOPE CHART. The quantity of interest is a DIFFERENCE between two paired conditions
(same author, same gold, same perturbed set, only the question surface changed). A slope chart
shows the pair and its difference as one mark; grouped bars would show four numbers and make
the reader compute the contrast that is the whole point.

WHAT THE READER MUST NOT CONCLUDE. If the iso ceiling has no paraphrase headroom for a target,
the gap on that author is an adapter-generalization result, not an unlearning one — that target
is drawn with a hollow line and flagged in the caption rather than quietly plotted alongside the
valid ones.

Run under ${TOFU_PLOT_PYTHON}.

    "${TOFU_PLOT_PYTHON}" plot_expb.py --prefix reports/expb/expb --out_dir reports/figures/expb
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import plot_style as S

SURFACES = ["original", "paraphrase"]


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


def fig_b1(tidy, rob, out_dir, arm="mean20", metric="rouge", mode=None):
    k = S.ink(mode)
    by = {}
    for r in tidy:
        by[(r.get("label"), r.get("author"), r.get("surface"))] = f(r.get(metric))

    targets = sorted({int(r["target"]) for r in rob if r.get("target") not in (None, "")})
    if not targets:
        print("  (no targets in the robustness CSV — skipping figB1)")
        return None
    headroom = {int(r["target"]): str(r.get("iso_has_headroom", "")).lower() in ("true", "1")
                for r in rob}

    # One slot per target, never cycled — a186 and a194 sharing a hue would be unreadable.
    pal = S.slot_colors(len(targets), mode, form="line")
    # The skill's rule: a legend is always present for >=2 series, and <=4 are ALSO direct-
    # labeled. Past 4 the end labels collide (they did, at 5 targets with near-equal values),
    # so the legend carries identity alone there.
    label_directly = len(targets) <= 4

    fig, ax = S.plt.subplots(figsize=(7.2, 5.0))
    xs = [0, 1]
    drew = 0
    for j, X in enumerate(targets):
        c = pal[j]
        post = [by.get((f"expb_{arm}_drop_a{X}", str(X), s)) for s in SURFACES]
        pre = [by.get((f"expb_{arm}_full", str(X), s)) for s in SURFACES]
        ok = headroom.get(X, True)
        if all(v is not None for v in post):
            ax.plot(xs, post, color=c, lw=2.0, marker="o", ms=6,
                    ls="-" if ok else (0, (3, 2)), zorder=3,
                    label=f"a{X}" + ("" if ok else "  (no iso headroom)"))
            if label_directly:
                S.direct_label(ax, xs[-1], post[-1], f"a{X}", c, mode=mode)
            drew += 1
        if all(v is not None for v in pre):
            # Open markers = the pre-deletion reference. Same hue, so the pair reads as one
            # author; hollow, so it never competes with the post-deletion line.
            ax.plot(xs, pre, color=c, lw=1.0, ls=(0, (1, 2)), marker="o", ms=6,
                    mfc=S.SURFACE[mode or S.MODE], mew=1.4, zorder=2)

    if not drew:
        print("  (no post-deletion rows — skipping figB1)")
        S.plt.close(fig)
        return None

    iso_vals = [f(r.get("iso_orig")) for r in rob if f(r.get("iso_orig")) is not None]
    if iso_vals:
        S.anchor_line(ax, sum(iso_vals) / len(iso_vals), "iso ceiling (mean, original)", mode)
    base = [by.get(("expb_base", str(X), "original")) for X in targets]
    base = [v for v in base if v is not None]
    if base:
        S.anchor_line(ax, sum(base) / len(base), "base floor (mean, original)", mode)

    ax.set_xticks(xs)
    ax.set_xticklabels(["original question", "paraphrased question"])
    ax.set_xlim(-0.25, 1.45)
    ax.set_ylabel({"rouge": "ROUGE-L recall vs the original answer",
                   "prob": "answer probability",
                   "forget_truth_ratio": "forget truth ratio"}[metric])
    ax.set_title(f"figB1 — target-author recall after deleting that author's adapter ({arm})",
                 fontsize=11)
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) >= 2:
        ax.legend(handles, labels, loc="lower left", title="target author", title_fontsize=9)
    ax.annotate("solid = after deletion   ·   dotted + hollow = before deletion",
                xy=(0.0, -0.16), xycoords="axes fraction", fontsize=8, color=k["muted"])
    return S.finish(fig, os.path.join(out_dir, "figB1_forget_slopes"))


def fig_b2(leak, out_dir, mode=None):
    """Leakage: rho for every non-target author, per target. Cells where rho is undefined
    (the pre-deletion model and the oracle already agreed) are COUNTED and shown as a
    separate bar, never silently dropped and never drawn as rho=0."""
    if not leak:
        print("  (no leakage rows — skipping figB2)")
        return None
    k = S.ink(mode)
    targets = sorted({int(r["target"]) for r in leak})
    pal = S.slot_colors(len(targets), mode, form="line")
    fig, ax = S.plt.subplots(figsize=(7.2, 4.4))
    n_undef_total = 0
    for j, X in enumerate(targets):
        vals = [f(r.get("rho")) for r in leak
                if int(r["target"]) == X and r.get("rho_defined") in ("1", 1, True)
                and not int(r.get("is_target", 0))]
        vals = [v for v in vals if v is not None]
        n_undef_total += sum(1 for r in leak if int(r["target"]) == X
                             and r.get("rho_defined") in ("0", 0, False))
        if not vals:
            continue
        jitter = [j + (i - len(vals) / 2) * (0.6 / max(len(vals), 1)) for i in range(len(vals))]
        ax.plot(jitter, vals, ls="none", marker="o", ms=5, color=pal[j],
                alpha=0.85, zorder=3)
        m = sum(vals) / len(vals)
        ax.plot([j - 0.34, j + 0.34], [m, m], color=k["primary"], lw=2.0, zorder=4)
        ax.annotate(f"{m:.2f}", xy=(j, m), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color=k["secondary"])
    ax.axhline(1.0, color=k["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate("ρ = 1  ⇒  the non-target author is entirely unaffected by the deletion",
                xy=(0.01, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=k["muted"])
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"delete a{X}" for X in targets])
    ax.set_ylabel("ρ  (non-target retention vs the pre-deletion model)")
    ax.set_title("figB2 — collateral effect on the 19 authors that were NOT deleted", fontsize=11)
    ax.grid(axis="x", visible=False)
    if n_undef_total:
        ax.annotate(f"{n_undef_total} cells undefined (|m_full − m_retain90| ≤ floor) — "
                    f"excluded, not plotted as 0",
                    xy=(0.0, -0.18), xycoords="axes fraction", fontsize=8, color=k["muted"])
    return S.finish(fig, os.path.join(out_dir, "figB2_leakage"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="reports/expb/expb")
    ap.add_argument("--out_dir", default="reports/figures/expb")
    ap.add_argument("--arm", default="mean20")
    ap.add_argument("--metric", default="rouge", choices=["rouge", "prob", "forget_truth_ratio"])
    ap.add_argument("--mode", default=None, choices=["light", "dark"])
    args = ap.parse_args()

    mode = args.mode or S.MODE
    S.apply_style(mode)
    tidy = read_csv(f"{args.prefix}_tidy.csv")
    rob = read_csv(f"{args.prefix}_robustness.csv")
    leak = read_csv(f"{args.prefix}_leakage.csv")
    print(f"[plot_expb] {len(tidy)} tidy, {len(rob)} robustness, {len(leak)} leakage rows")
    if not (tidy or rob or leak):
        raise SystemExit("nothing to plot — run collect_expb.py first")
    os.makedirs(args.out_dir, exist_ok=True)
    fig_b1(tidy, rob, args.out_dir, args.arm, args.metric, mode)
    fig_b2(leak, args.out_dir, mode)

    sp = f"{args.prefix}_summary.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        print("[plot_expb] statistical floors that must appear in any caption: "
              f"KS needs D>={s['statistical_floors']['ks_D_for_alpha_0.05']} at n=20; "
              f"Wilcoxon p floor at 5 targets = "
              f"{s['statistical_floors']['wilcoxon_p_floor_at_5_targets']}")


if __name__ == "__main__":
    main()
