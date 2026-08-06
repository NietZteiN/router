"""Gate the chart palette by COMPUTING the colour checks, not by looking at the figures.

    "${TOFU_PLOT_PYTHON}" test_plot_style.py     # needs matplotlib
    python test_plot_style.py --colors-only      # the colour maths alone, no matplotlib

The data-viz method's rule is that the colour part is computable, so it must be computed. This
file is that computation, run as a gate: OKLab ΔE under a Machado-2009 CVD simulation, the
lightness band, the chroma floor, and WCAG contrast against each mode's surface — for BOTH
modes, and for the pairlist that actually applies to each chart form.

It also pins the two rules that a chart silently breaks rather than erroring:
  * hues are never CYCLED (two entities must never share a colour), and
  * a light-mode slot below 3:1 contrast obliges direct labels (the "relief rule").
"""
from __future__ import annotations
import os

import itertools
import math
import sys

OK = "ok  "

# Thresholds and the CVD model, from the data-viz validator. Keep in lockstep with it.
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}      # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0                          # OKLab ΔE x100
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
}


def _srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _lin(h):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in _srgb(h)]


def _rel_lum(h):
    r, g, b = _lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([_rel_lum(a), _rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def oklch(h):
    L, a, b = _oklab(_lin(h))
    return L, math.hypot(a, b)


def _sim(h, kind):
    r, g, b = _lin(h)
    M = MACHADO[kind]
    cl = lambda c: max(0.0, min(1.0, c))          # noqa: E731
    return [cl(M[0][0] * r + M[0][1] * g + M[0][2] * b),
            cl(M[1][0] * r + M[1][1] * g + M[1][2] * b),
            cl(M[2][0] * r + M[2][1] * g + M[2][2] * b)]


def dE(h1, h2, kind=None):
    a = _oklab(_sim(h1, kind) if kind else _lin(h1))
    b = _oklab(_sim(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


# The palettes, duplicated here ON PURPOSE. If this file imported them from plot_style, an edit
# to plot_style would move the values and the test with them — the gate must hold a fixed
# reference to detect exactly that.
SERIES3 = {"light": ["#2a78d6", "#eb6834", "#1baf7a"],
           "dark":  ["#3987e5", "#d95926", "#199e70"]}
SERIES8 = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                     "#008300", "#4a3aa7", "#e34948"],
           "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                     "#008300", "#9085e9", "#e66767"]}
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}


def _check(pal, mode, pairs):
    plist = (list(itertools.combinations(pal, 2)) if pairs == "all"
             else [(pal[i], pal[i + 1]) for i in range(len(pal) - 1)])
    lo, hi = BAND[mode]
    band_bad = [i + 1 for i, h in enumerate(pal) if not lo <= oklch(h)[0] <= hi]
    chroma_bad = [i + 1 for i, h in enumerate(pal) if oklch(h)[1] < CHROMA_FLOOR]
    worst_n = min(dE(a, b) for a, b in plist)
    worst_c = min(min(dE(a, b, "protan"), dE(a, b, "deutan")) for a, b in plist)
    low_contrast = [i + 1 for i, h in enumerate(pal) if contrast(h, SURFACE[mode]) < CONTRAST_MIN]
    return band_bad, chroma_bad, worst_n, worst_c, low_contrast


def test_palette_passes_the_computable_checks():
    for mode in ("light", "dark"):
        for pal, pairs, form in ((SERIES3[mode], "all", "scatter (figA4)"),
                                 (SERIES8[mode], "adjacent", "line/slope (figB1, figC2)")):
            band, chroma, wn, wc, low = _check(pal, mode, pairs)
            assert not band, f"{mode}/{form}: slots {band} outside the OKLCH lightness band"
            assert not chroma, f"{mode}/{form}: slots {chroma} below the chroma floor (read gray)"
            assert wn >= NORMAL_FLOOR, (
                f"{mode}/{form}: worst {pairs}-pair normal-vision ΔE {wn:.1f} < {NORMAL_FLOOR}. "
                f"Full-colour readers cannot tell that pair apart — cut series or facet; "
                f"secondary encoding does NOT excuse this one.")
            assert wc >= CVD_FLOOR, (
                f"{mode}/{form}: worst {pairs}-pair CVD ΔE {wc:.1f} < {CVD_FLOOR} — "
                f"indistinguishable under protanopia/deuteranopia")
            status = "PASS" if wc >= CVD_TARGET else "WARN(needs secondary encoding)"
            print(OK + f"{mode:5} {form:26} {pairs:8} normal ΔE {wn:5.1f}  CVD ΔE {wc:4.1f} {status}"
                       + (f"  · relief rule owed for slots {low}" if low else ""))


def test_relief_rule_is_honoured_where_contrast_is_low():
    """A sub-3:1 slot is not dismissable: the figures using it must ship direct labels."""
    low = [i + 1 for i, h in enumerate(SERIES3["light"])
           if contrast(h, SURFACE["light"]) < CONTRAST_MIN]
    assert low, "fixture expectation: slot 3 (aqua) is below 3:1 on the light surface"
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("plot_expa.py", "plot_expb.py", "plot_expc.py"):
        src = open(os.path.join(here, f)).read()
        assert "direct_label" in src, \
            (f"{f} uses a palette slot below 3:1 contrast but never direct-labels a series. "
             f"The contrast WARN carries an obligation (visible labels or a table view), not a "
             f"choice.")
    print(OK + f"light slots {low} are below 3:1 and every plot script direct-labels its series")


def test_no_dual_axis_anywhere():
    """The #1 chart mistake. twinx/twiny puts two scales on one frame and is never correct."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    bad = []
    for f in ("plot_expa.py", "plot_expb.py", "plot_expc.py", "plot_style.py", "plot_nmerge.py"):
        p = os.path.join(here, f)
        if not os.path.exists(p):
            continue
        src = open(p).read()
        for call in ("twinx", "twiny"):
            if call + "(" in src:
                bad.append(f"{f}: {call}()")
    assert not bad, ("dual-axis chart(s) found: " + ", ".join(bad) +
                     " — use two panels, small multiples, or index to a common base")
    print(OK + "no twinx/twiny: no chart carries two y-scales")


def test_slot_colors_refuses_to_cycle():
    """Requires matplotlib (plot_style imports it). Skipped with --colors-only."""
    import plot_style as S
    for mode in ("light", "dark"):
        assert S.SERIES[mode] == SERIES3[mode], \
            f"plot_style.SERIES[{mode}] drifted from the validated 3-slot reference"
        assert S.SERIES_8[mode] == SERIES8[mode], \
            f"plot_style.SERIES_8[{mode}] drifted from the validated 8-slot reference"
        assert S.slot_colors(3, mode, form="scatter") == SERIES3[mode][:3]
        assert S.slot_colors(5, mode, form="line") == SERIES8[mode][:5]
        for n, form in ((4, "scatter"), (9, "line")):
            try:
                S.slot_colors(n, mode, form=form)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"slot_colors({n}, form={form!r}) returned instead of raising — it would "
                    f"cycle, painting two different entities the same hue")
    print(OK + "slot_colors matches the validated reference and RAISES rather than cycling")


def main():
    colors_only = "--colors-only" in sys.argv
    tests = [test_palette_passes_the_computable_checks, test_relief_rule_is_honoured_where_contrast_is_low,
             test_no_dual_axis_anywhere]
    if not colors_only:
        tests.append(test_slot_colors_refuses_to_cycle)
    for t in tests:
        t()
    print(f"\nALL test_plot_style.py GATES PASS ({len(tests)} checks"
          + (", colour maths only" if colors_only else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
