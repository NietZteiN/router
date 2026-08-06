"""Shared figure style for the APA plots — one place so the three scripts agree.

⚠ RUN PLOTS UNDER ${TOFU_PLOT_PYTHON}. matplotlib is deliberately absent from the pinned
runtime env (requirements.txt); the site file names an interpreter that has it. Every plot_*.py
imports this module, so it is also the one place that failure is explained.

The palette is the data-viz reference categorical theme, slots 1-3, taken in fixed order and
never cycled. It was validated with the skill's six computable checks (OKLab dE, Machado-2009
CVD simulation) for BOTH modes on the ALL-PAIRS list, which is the list that applies here
because figA4 is a scatter:

    light  worst normal-vision dE 24.0 (floor 15)   worst CVD dE 9.2 (target 8)   PASS
    dark   worst normal-vision dE 20.9              worst CVD dE 9.4              PASS

One WARN carries an obligation rather than a choice: slot 3 (aqua) sits at 2.74:1 against the
light surface, under the 3:1 bar, so the relief rule applies — every figure using it ships
DIRECT LABELS on the series (and the underlying CSV is always written beside the figure). Do not
"fix" this by re-stepping the hue; the palette is documented and the labels are the remedy.

There are exactly three arms (lambda=1, lambda=1/sqrt(N), 1/N), so three slots are enough and
the all-pairs cap of three is not a constraint we are near.
"""
from __future__ import annotations

import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - the actionable message IS the behaviour
    raise SystemExit(
        "matplotlib is not installed in this interpreter.\n"
        "  The pinned runtime env (requirements.txt) deliberately omits it; plots run under a\n"
        "  separate interpreter named by ${TOFU_PLOT_PYTHON} in cluster_env.<site>.sh.\n"
        f"  Current interpreter: {sys.executable}\n"
        "  Try:  \"${TOFU_PLOT_PYTHON}\" " + " ".join(sys.argv) + "\n"
        "  or:   pip install -r requirements-plots.txt")

# ── categorical slots, fixed order, NEVER CYCLED ─────────────────────────────────────────────
# Two views of ONE theme, differing only in how many slots are legal for a given chart form:
#
#   SERIES    slots 1-3, valid on the ALL-PAIRS list  -> scatter (figA4), and the 3 arms.
#   SERIES_8  slots 1-8, valid on the ADJACENT list   -> lines / slopes / bars (figB1, figB2).
#
# Validated with the six computable checks (OKLab dE, Machado-2009 CVD sim) in both modes:
#   all-pairs, 3 slots   light normal 24.0 / CVD 9.2   dark normal 20.9 / CVD 9.4   PASS
#   adjacent, 5-8 slots  light normal 19.6 / CVD 9.1   dark normal 19.3 / CVD 8.4   PASS
#
# Slot count is not a style preference. Cycling a 4th series back onto slot 1 makes two
# different entities the same colour, which is the single worst thing a categorical palette can
# do — so `slot_colors()` RAISES past the legal count rather than wrapping. Past 8, fold to
# "Other" or facet into small multiples.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark":  ["#3987e5", "#d95926", "#199e70"],
}
SERIES_8 = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
              "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
              "#008300", "#9085e9", "#e66767"],
}
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8880", "grid": "#e4e3de"},
    "dark":  {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#7d7c74", "grid": "#33322f"},
}

# The three arms, in slot order. `key` matches analyze_expa.py's `method` column.
ARMS = [
    {"key": "additive_sum",       "lam": None,    "label": "sum  λ=1",     "slot": 0},
    {"key": "additive_sum",       "lam": "isqrt", "label": "sum  λ=1/√N",  "slot": 1},
    {"key": "additive_mean",      "lam": None,    "label": "mean  1/N",    "slot": 2},
]

MODE = os.environ.get("TOFU_PLOT_MODE", "light")


def palette(mode=None):
    return SERIES[mode or MODE]


def slot_colors(n, mode=None, form="line"):
    """The first `n` slots, in fixed order. Raises rather than cycling.

    form="scatter" (all-pairs pairlist) caps at 3; form="line" (adjacent pairlist) caps at 8.
    Cycling would paint two different entities the same hue — the failure this exists to make
    impossible, and one that is invisible until someone reads the chart wrong.
    """
    table = SERIES if form == "scatter" else SERIES_8
    pal = table[mode or MODE]
    if n > len(pal):
        raise ValueError(
            f"{n} series requested but only {len(pal)} slots are validated for a '{form}' chart "
            f"({'all-pairs' if form == 'scatter' else 'adjacent'} pairlist). Cycling hues is "
            f"never the answer: fold the tail into 'Other', or facet into small multiples.")
    return pal[:n]


def ink(mode=None):
    return INK[mode or MODE]


def apply_style(mode=None):
    """Recessive chrome: thin marks, no top/right spine, grid behind the data."""
    m = mode or MODE
    k, s = INK[m], SURFACE[m]
    plt.rcParams.update({
        "figure.facecolor": s, "axes.facecolor": s, "savefig.facecolor": s,
        "text.color": k["primary"], "axes.labelcolor": k["secondary"],
        "xtick.color": k["secondary"], "ytick.color": k["secondary"],
        "axes.edgecolor": k["grid"], "axes.linewidth": 0.8,
        "grid.color": k["grid"], "grid.linewidth": 0.6, "axes.grid": True,
        "axes.axisbelow": True,                 # grid behind the marks, never through them
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 5.0,   # 2px lines, >=8px marker diameter
        "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9,
        "legend.frameon": False, "figure.dpi": 140,
    })


def direct_label(ax, x, y, text, color, dx=6, dy=0, mode=None):
    """The relief-rule remedy: name the series at its end, in INK not the series color.

    Text wears text tokens; the colored mark beside it carries identity. That keeps identity
    off color alone (check 6) and satisfies the contrast obligation for slot 3.
    """
    k = INK[mode or MODE]
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=k["secondary"], clip_on=False)
    ax.plot([x], [y], marker="o", ms=5, color=color, zorder=5, clip_on=False)


def direct_labels(ax, items, min_gap_pt=11.0, dx=6, mode=None):
    """Direct-label several series at once, nudging apart any that would overprint.

    Needed because the interesting result is often that the series COINCIDE (Exp C predicts
    n_eff is the same for owned and unowned queries). Labelling each at its own endpoint then
    stacks three strings on one pixel and the figure says nothing. Offsets are computed in
    display space and applied as point offsets, so the nudge does not misreport the data
    position — the marker stays exactly on the value.

    items: iterable of (x, y, text, color), any order.
    """
    items = [it for it in items if it[1] is not None]
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()                      # a valid transform needs a laid-out figure
    disp = [(ax.transData.transform((x, y))[1], i) for i, (x, y, _, _) in enumerate(items)]
    disp.sort()
    placed = {}
    last = None
    for ypix, i in disp:                   # bottom-up, push each label above the previous
        y_adj = ypix if last is None else max(ypix, last + min_gap_pt)
        placed[i] = y_adj - ypix           # the nudge, in points (display px ~ pt at dpi 72)
        last = y_adj
    k = INK[mode or MODE]
    for i, (x, y, text, color) in enumerate(items):
        ax.annotate(text, xy=(x, y), xytext=(dx, placed[i]), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=k["secondary"], clip_on=False)
        ax.plot([x], [y], marker="o", ms=5, color=color, zorder=5, clip_on=False)


def anchor_line(ax, y, text, mode=None, ls=(0, (4, 3))):
    """A reference constant (base / ft / retain90). Deliberately gray, never a series slot —
    an anchor is not a fourth series and must not read as one."""
    k = INK[mode or MODE]
    ax.axhline(y, color=k["muted"], lw=1.0, ls=ls, zorder=1)
    ax.annotate(text, xy=(0.995, y), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color=k["muted"])


def finish(fig, out_prefix, formats=("png", "pdf")):
    fig.tight_layout()
    made = []
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    for ext in formats:
        p = f"{out_prefix}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        made.append(p)
    plt.close(fig)
    print("  wrote " + ", ".join(made))
    return made
