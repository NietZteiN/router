"""Experiment B assembly: robustness, leakage and selectivity from the per-condition JSONs.

Each `measure_adapter_selectivity.py` run scores ONE served model on every pool author x both
surfaces and knows nothing about any other condition. This file does the contrasts, which is
where all the ways to fool yourself live:

  G(X)   robustness gap   m_X^orig(P\\X) - m_X^para(P\\X)
         Reported against the MATCHED full-model gap and the iso ceiling gap, never alone: a
         model that simply cannot answer paraphrased questions produces a large G for reasons
         that have nothing to do with deletion.

  rho_Y  leakage          (m_Y(P\\X) - m_Y(retain90)) / (m_Y(P) - m_Y(retain90))
         Only defined where the denominator exceeds --rho_floor. When the pre-deletion model and
         the oracle already agree on author Y, the ratio is 0/0 and its value is an artifact of
         float noise. Those cells are reported as "undefined", never as "no leakage" — the
         difference matters, because "no leakage" is the claim the experiment is trying to test.

  S(X)   selectivity      mean target drop / mean non-target drop
         Deleting X should hurt X and nobody else. S ~ 1 means the aggregate responded to X's
         questions with everyone's adapters, so removing one of twenty changed almost nothing
         that was specific to X.

EFFECT SIZES, NOT TESTS. With n = m = 20 the KS test has 20 attainable p-values and needs
D >= 0.45 for alpha = 0.05; with 5 targets the Wilcoxon signed-rank p floor is 0.0625. So
ks_statistic (D) and forget_truth_ratio are reported as the effect sizes and `forget_quality`
is carried as a descriptive ordinal. The floors are printed in the summary so a reader cannot
mistake "p = 0.0625" for a null result.

Usage:
    python collect_expb.py --config configs/expb_selectivity_7b.json \
        --results_dir "${TOFU_CKPT_ROOT}"/Llama-2-7B-chat-hf_expb_selectivity/results \
        --contrib_dir reports/expb --out_prefix reports/expb/expb
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

METRICS = ("rouge", "prob", "forget_truth_ratio", "ks_statistic", "forget_quality")


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_conditions(results_dir):
    """label -> {(author, surface): summary row}, plus the run-level metadata."""
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if p.endswith(".mmlu.json"):
            continue
        try:
            d = json.load(open(p))
        except Exception as e:  # noqa: BLE001 — one unreadable file must not lose the campaign
            print(f"[collect] WARN unreadable {p}: {type(e).__name__}: {e}")
            continue
        if "summary" not in d:
            continue
        label = os.path.splitext(os.path.basename(p))[0]
        rows = {}
        for r in d["summary"]:
            rows[(r.get("author"), r.get("surface"))] = r
        out[label] = {"meta": d, "rows": rows,
                      "arm": d.get("arm"), "condition": d.get("condition"),
                      "target": d.get("target_author")}
    return out


def _get(cond, author, surface, metric):
    r = cond["rows"].get((author, surface))
    return _f(r.get(metric)) if r else None


def robustness(conds, targets, full_label, drop_label_fn, iso_label_fn, metric="rouge"):
    """G(X) on the post-deletion model, beside the matched full-model and iso-ceiling gaps."""
    rows = []
    for X in targets:
        drop = conds.get(drop_label_fn(X))
        full = conds.get(full_label)
        iso = conds.get(iso_label_fn(X))
        def gap(c):
            if not c:
                return (None, None, None)
            o, p = _get(c, X, "original", metric), _get(c, X, "paraphrase", metric)
            return (o, p, (o - p) if (o is not None and p is not None) else None)
        d_o, d_p, d_g = gap(drop)
        f_o, f_p, f_g = gap(full)
        i_o, i_p, i_g = gap(iso)
        rows.append({
            "target": X, "metric": metric,
            "drop_orig": d_o, "drop_para": d_p, "G_drop": d_g,
            "full_orig": f_o, "full_para": f_p, "G_full": f_g,
            "iso_orig": i_o, "iso_para": i_p, "G_iso": i_g,
            # The reading that matters: did DELETION widen the gap beyond what the
            # pre-deletion model already showed?
            "G_drop_minus_G_full": (d_g - f_g) if (d_g is not None and f_g is not None) else None,
            # Pre-registered guard. If the isolated adapter cannot itself answer X's
            # paraphrased questions, the paraphrase axis has no headroom and G is an
            # adapter-generalization result, not an unlearning one.
            "iso_has_headroom": (i_o is not None and i_o > 0.10),
        })
    return rows


def leakage(conds, targets, authors, full_label, drop_label_fn, ref_label,
            metric="rouge", surface="original", rho_floor=0.05):
    rows = []
    ref = conds.get(ref_label)
    full = conds.get(full_label)
    for X in targets:
        drop = conds.get(drop_label_fn(X))
        if not (drop and full and ref):
            continue
        for Y in authors:
            m_ref = _get(ref, Y, surface, metric)
            m_full = _get(full, Y, surface, metric)
            m_drop = _get(drop, Y, surface, metric)
            if None in (m_ref, m_full, m_drop):
                continue
            denom = m_full - m_ref
            # A denominator at or below the floor means the pre-deletion model and the oracle
            # already agree about Y; the ratio would be 0/0 and its VALUE would be noise.
            defined = denom > rho_floor
            rows.append({
                "target": X, "author": Y, "is_target": int(X == Y),
                "metric": metric, "surface": surface,
                "m_retain90": m_ref, "m_full": m_full, "m_drop": m_drop,
                "denominator": denom, "rho_defined": int(defined),
                "rho": ((m_drop - m_ref) / denom) if defined else None,
                "delta_vs_full": m_drop - m_full,
            })
    return rows


def selectivity(leak_rows, targets):
    """S(X) = mean target drop / mean non-target drop, on the same rows rho came from."""
    rows = []
    for X in targets:
        mine = [r for r in leak_rows if r["target"] == X]
        tgt = [-r["delta_vs_full"] for r in mine if r["is_target"]]
        non = [-r["delta_vs_full"] for r in mine if not r["is_target"]]
        d_t = (sum(tgt) / len(tgt)) if tgt else None
        d_n = (sum(non) / len(non)) if non else None
        # A non-target drop at or below zero means deletion did not damage the others at all;
        # the ratio would be infinite or negative, so report it as undefined rather than
        # printing a huge selectivity that is really a divide-by-noise.
        S = (d_t / d_n) if (d_t is not None and d_n is not None and d_n > 1e-6) else None
        rows.append({"target": X, "delta_target": d_t, "delta_nontarget": d_n,
                     "S": S, "n_nontarget": len(non),
                     "S_undefined_reason": None if S is not None else
                                           ("no non-target drop above 1e-6" if d_n is not None
                                            else "missing rows")})
    return rows


def load_contrib(contrib_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(contrib_dir, "contrib_*.json"))):
        try:
            d = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"[collect] WARN unreadable {p}: {type(e).__name__}: {e}")
            continue
        s = d.get("summary", {})
        rows.append({"label": os.path.basename(p)[len("contrib_"):-len(".json")],
                     "hidden": d.get("hidden"), "n_authors": d.get("n_authors"),
                     **{k: s.get(k) for k in sorted(s) if not isinstance(s.get(k), (dict, list))}})
    return rows


def _write_csv(path, rows):
    if not rows:
        print(f"[collect] (no rows for {path})")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[collect] wrote {path} ({len(rows)} rows)")


def _rng(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--contrib_dir", default=None)
    ap.add_argument("--out_prefix", default="reports/expb/expb")
    ap.add_argument("--arm", default="mean20", help="which aggregate arm to contrast")
    ap.add_argument("--metric", default="rouge", choices=["rouge", "prob", "forget_truth_ratio"])
    ap.add_argument("--rho_floor", type=float, default=0.05,
                    help="minimum |m_full - m_retain90| for rho to be defined")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    targets = cfg["targets"]
    authors = _rng(cfg["authors"])

    conds = load_conditions(args.results_dir)
    if not conds:
        raise SystemExit(f"no condition JSONs in {args.results_dir} — run the score stage first")
    print(f"[collect] {len(conds)} conditions: {', '.join(sorted(conds))}")

    full_label = f"expb_{args.arm}_full"
    drop_fn = lambda X: f"expb_{args.arm}_drop_a{X}"          # noqa: E731
    iso_fn = lambda X: f"expb_iso_a{X}"                       # noqa: E731

    missing = [l for l in [full_label, "expb_retain90"] + [drop_fn(X) for X in targets]
               if l not in conds]
    if missing:
        print(f"[collect] WARN missing conditions, their rows will be absent: {missing}")

    rob = robustness(conds, targets, full_label, drop_fn, iso_fn, args.metric)
    leak = leakage(conds, targets, authors, full_label, drop_fn, "expb_retain90",
                   args.metric, "original", args.rho_floor)
    sel = selectivity(leak, targets)

    # One tidy row per (condition, author, surface) — the raw table every figure reads.
    tidy = []
    for label, c in sorted(conds.items()):
        for (a, s), r in c["rows"].items():
            tidy.append({"label": label, "arm": c["arm"], "condition": c["condition"],
                         "target_author": c["target"], "author": a, "surface": s,
                         **{m: _f(r.get(m)) for m in METRICS},
                         "n_rows": r.get("n_rows"), "n_tr": r.get("n_tr")})

    _write_csv(f"{args.out_prefix}_tidy.csv", tidy)
    _write_csv(f"{args.out_prefix}_robustness.csv", rob)
    _write_csv(f"{args.out_prefix}_leakage.csv", leak)
    _write_csv(f"{args.out_prefix}_selectivity.csv", sel)
    if args.contrib_dir:
        _write_csv(f"{args.out_prefix}_contrib.csv", load_contrib(args.contrib_dir))

    n_def = sum(r["rho_defined"] for r in leak)
    summary = {
        "arm": args.arm, "metric": args.metric, "targets": targets,
        "n_conditions": len(conds), "n_leakage_rows": len(leak),
        "n_rho_defined": n_def, "n_rho_undefined": len(leak) - n_def,
        "rho_floor": args.rho_floor,
        "selectivity": {str(r["target"]): r["S"] for r in sel},
        "iso_headroom_ok": {str(r["target"]): r["iso_has_headroom"] for r in rob},
        "statistical_floors": {
            "ks_n": 20, "ks_attainable_p_values": 20, "ks_D_for_alpha_0.05": 0.45,
            "wilcoxon_p_floor_at_5_targets": 0.0625,
            "note": "Report ks_statistic (D) and forget_truth_ratio as the effect sizes. "
                    "forget_quality is a descriptive ordinal, never a test — with 20 rows per "
                    "author it cannot clear alpha = 0.05 unless D >= 0.45, and with 5 targets "
                    "no Wilcoxon p below 0.0625 is attainable at all.",
        },
        "pre_registered_falsifier": "S(X) >= 3 with rho_Y >= 0.95 would REFUTE the prediction "
                                    "(from key_firing_e5.json: gate_median 1.1018, verdict LAZY) "
                                    "that the aggregate is not author-selective, i.e. S(X) ~ 1.",
    }
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[collect] wrote {args.out_prefix}_summary.json")
    print(f"[collect] rho defined in {n_def}/{len(leak)} cells (floor {args.rho_floor}); "
          f"S(X) = " + ", ".join(f"a{r['target']}:"
                                 + ("n/a" if r["S"] is None else f"{r['S']:.2f}") for r in sel))
    bad = [r["target"] for r in rob if not r["iso_has_headroom"]]
    if bad:
        print(f"[collect] ⚠ iso ceiling has NO paraphrase headroom for targets {bad} — for those "
              f"authors a robustness gap is an adapter-generalization result, not an unlearning "
              f"one. Say so explicitly rather than reporting G(X) as a deletion effect.")


if __name__ == "__main__":
    main()
