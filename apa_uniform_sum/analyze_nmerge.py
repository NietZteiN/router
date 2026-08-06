"""Assemble the N-merge interference CSVs from per-label eval JSONs (CPU, post-array).

  python analyze_nmerge.py --config configs/nmerge_interference_7b.json

Reads {out_dir}/results/{cap}/*.json (written by submit_nmerge.sh eval) + the overlap
JSON (merge_subset.py overlap) and writes tidy CSVs for plot_nmerge.py:

  reports/nmerge_mu.csv          one row per (label x probe): mu + the 9 utility
                                 components + ppl diagnostics; `headline` marks the
                                 perm[0] probe row (same retain split across the curve).
  reports/nmerge_own_recall.csv  one row per (label x probe): own-author forget_rouge /
                                 forget_truth_ratio / forget_ppl + the probe's iso
                                 reference and drop (iso - merged). The H1 table.
  reports/nmerge_overlap.csv     flattened per-N geometry + per-probe col(B) row-means,
                                 joined with each probe's recall drop at that N. The H3 table.

Consistency checks (silent-failure telemetry): real_*/world_* components must be
IDENTICAL across a label's probe jobs (eval_shard_id only remaps the forget set and the
retain sample); retain_ppl explosion and NaN flags are printed for every row.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re

from merge_subset import author_permutation, load_config

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

MU_COMPONENTS = [
    "retain_prob", "retain_rouge", "retain_truth_scaled",
    "real_prob", "real_rouge", "real_truth_scaled",
    "world_prob", "world_rouge", "world_truth_scaled",
]
SHARED_ACROSS_PROBES = [c for c in MU_COMPONENTS if not c.startswith("retain")]
RECALL_FIELDS = ["forget_rouge", "forget_truth_ratio", "forget_ppl"]

# label -> (method, n, seed, svd_rank, variant)
# NB the alternation is longest-first: 'sumisqrt'/'sumL...' MUST precede bare 'sum', or the
# regex matches 'sum' and the leftover 'isqrt' fails the following '_svd|_N' — silently
# demoting the row to kind="other", which the main loop drops with no warning.
_LABEL_RE = re.compile(
    r"^nmerge_(?P<m>add|dare|cpool|cr\d+|sumisqrt|sumL[0-9pm]+|sum)"
    r"(?:_svd(?P<svd>\d+))?_N(?P<n>\d+)_s(?P<seed>\d+)(?P<r8>_r8)?$")
_ISO_RE = re.compile(r"^iso_a(?P<a>\d+)$")

# tag -> method. MUST be kept in sync with _LABEL_RE: a tag that matches the regex but is
# missing here falls through to the cr{rho} branch and yields a wrong-but-plausible method
# name (e.g. "sum" -> "centered_lowrank_rm" via tag[2:]). test_expa.py pins this.
_TAG_METHOD = {
    "add": "additive_mean",
    "dare": "dare_ties",
    "cpool": "centered_pool",
    "sum": "additive_sum",              # uniform equal-weight sum, coefficient 1.0 (APA rule)
    "sumisqrt": "additive_sum_isqrt",   # matched-norm arm, coefficient 1/sqrt(N)
}


def parse_label(label):
    m = _LABEL_RE.match(label)
    if m:
        tag = m.group("m")
        method = _TAG_METHOD.get(tag)
        if method is None and tag.startswith("sumL"):  # sumL{lambda} -> explicit global coeff
            method = "additive_sum_l" + tag[4:]
        if method is None:  # cr{rho} -> per-rho centered_lowrank arm
            method = f"centered_lowrank_r{tag[2:]}"
        return {"kind": "merge", "method": method, "n": int(m.group("n")),
                "seed": int(m.group("seed")),
                "svd_rank": int(m.group("svd")) if m.group("svd") else None,
                "r8": bool(m.group("r8"))}
    m = _ISO_RE.match(label)
    if m:
        return {"kind": "iso", "author": int(m.group("a"))}
    if label in ("base_model", "ft_r32", "retain90_oracle"):
        return {"kind": "anchor", "name": label}
    return {"kind": "other"}


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _num(d, k):
    v = d.get(k) if d else None
    return v if isinstance(v, (int, float)) else None


def collect_jsons(results_dir):
    """({(label, sid_or_None): json}, {label: json}) — probe/plain rows and __subset rows."""
    out, subset = {}, {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        name = os.path.basename(path)[: -len(".json")]
        if name.endswith(".progress"):
            continue
        if name.endswith("__subset"):
            subset[name[: -len("__subset")]] = _load(path)
        elif "__own" in name:
            label, sid = name.rsplit("__own", 1)
            if not sid.isdigit():
                continue
            out[(label, int(sid))] = _load(path)
        else:
            out[(name, None)] = _load(path)
    return out, subset


def check_probe_consistency(rows_by_label):
    """real_*/world_* must be identical across a label's probe jobs (same model)."""
    bad = []
    for label, rows in rows_by_label.items():
        if len(rows) < 2:
            continue
        ref = rows[0]
        for r in rows[1:]:
            for c in SHARED_ACROSS_PROBES:
                a, b = ref.get(c), r.get(c)
                if a is not None and b is not None and abs(a - b) > 1e-9:
                    bad.append((label, c, a, b))
    for label, c, a, b in bad:
        print(f"[check] WARNING {label}: {c} differs across probe jobs ({a} vs {b}) "
              f"— same model must give identical real/world components")
    if not bad:
        print("[check] real/world components identical across probe jobs: OK")
    return bad


def flag(d):
    """Silent-failure flags for one result JSON."""
    flags = []
    mu = _num(d, "model_utility")
    if mu is None or (isinstance(mu, float) and math.isnan(mu)):
        flags.append("mu_nan")
    elif mu == 0.0:
        # scipy hmean returns exactly 0 when ANY one of the 9 components is 0, so mu==0 is a
        # single-component zero, not a graded score. Flagged separately from mu_nan because it
        # is a real (and informative) measurement, not a failure — read mu_gmean instead.
        flags.append("mu_zero")
    rp = _num(d, "retain_ppl")
    if rp is not None and rp > 100:
        flags.append(f"retain_ppl_explosion({rp:.0f})")
    return ";".join(flags)


def mu_gmean(d, floor=1e-3):
    """Geometric mean of the 9 utility components — the graded companion to model_utility.

    scipy's hmean (what `model_utility` is) hits exactly 0 as soon as one component is 0, so it
    cannot distinguish "one metric bottomed out" from "the model is destroyed". Observed on
    sparse_dare0p9sum_N16: model_utility 0.0 because real_rouge == world_rouge == 0.0, while
    retain_truth_scaled 0.151 / real_prob 0.2366 / world_truth_scaled 0.309 still carried
    signal. Components are floored so a single zero does not annihilate the geometric mean too.
    """
    vals = [_num(d, c) for c in MU_COMPONENTS]
    if any(v is None or math.isnan(v) for v in vals):
        return float("nan")
    return math.exp(sum(math.log(max(v, floor)) for v in vals) / len(vals))


def first_zero_component(d, eps=1e-9):
    """Which utility component hit zero first — names the channel that zeroed model_utility."""
    zeros = [c for c in MU_COMPONENTS
             if (_num(d, c) is not None and not math.isnan(_num(d, c)) and _num(d, c) <= eps)]
    return ";".join(zeros)


def _write(out, rows, cols):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    allcols = cols + [c for c in (rows[0].keys() if rows else []) if c not in cols]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allcols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_prefix", default=os.path.join(REPORTS, "nmerge"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    results_dir = os.path.join(cfg["out_dir"], "results", cfg["eval"]["cap"])
    seed0 = cfg["subset_seeds"][0]
    head_probe = int(author_permutation(seed0)[0])
    jsons, subset_jsons = collect_jsons(results_dir)
    print(f"[analyze] {len(jsons)} result JSONs (+{len(subset_jsons)} __subset) in "
          f"{results_dir}; headline probe a{head_probe}")

    # ---- mu table (one row per label x probe; headline = perm[0] probe or sid None) ----
    mu_rows, by_label = [], {}
    for (label, sid), d in sorted(jsons.items()):
        if d is None:
            continue
        info = parse_label(label)
        if info["kind"] == "other":
            continue
        row = {"label": label, "kind": info["kind"], "probe_author": sid,
               "method": info.get("method", ""), "n": info.get("n", ""),
               "seed": info.get("seed", ""), "svd_rank": info.get("svd_rank", ""),
               "r8": info.get("r8", False),
               "headline": (sid == head_probe) or (sid is None),
               "model_utility": _num(d, "model_utility"),
               "mu_gmean": round(mu_gmean(d), 4),
               "first_zero_component": first_zero_component(d),
               "forget_quality": _num(d, "forget_quality"),
               "retain_ppl": _num(d, "retain_ppl"), "forget_ppl": _num(d, "forget_ppl"),
               "flags": flag(d)}
        for c in MU_COMPONENTS:
            row[c] = _num(d, c)
        mu_rows.append(row)
        by_label.setdefault(label, []).append(row)
    check_probe_consistency(by_label)
    for label, rows in sorted(by_label.items()):
        mus = [r["model_utility"] for r in rows if r["model_utility"] is not None]
        if mus:
            spread = max(mus) - min(mus)
            for r in rows:
                r["mu_probe_spread"] = round(spread, 4)
            if spread > 0.02:
                print(f"[check] note {label}: mu spread across probes {spread:.4f} "
                      f"(retain-sample remap; expected small)")
    _write(f"{args.out_prefix}_mu.csv", mu_rows,
           ["label", "kind", "method", "n", "seed", "svd_rank", "r8", "probe_author",
            "headline", "model_utility", "mu_gmean", "mu_probe_spread"] + MU_COMPONENTS
           + ["first_zero_component", "forget_quality", "retain_ppl", "forget_ppl", "flags"])

    # ---- own-author recall table (merge/anchor rows joined with iso references) ----
    iso = {}   # author -> {field: value}
    for (label, sid), d in jsons.items():
        info = parse_label(label)
        if info["kind"] == "iso" and d is not None:
            iso[info["author"]] = {f: _num(d, f) for f in RECALL_FIELDS}
    recall_rows = []
    for (label, sid), d in sorted(jsons.items()):
        if d is None or sid is None:
            continue
        info = parse_label(label)
        if info["kind"] not in ("merge", "anchor", "iso"):
            continue
        row = {"label": label, "kind": info["kind"], "method": info.get("method", ""),
               "n": info.get("n", ""), "seed": info.get("seed", ""),
               "svd_rank": info.get("svd_rank", ""), "probe_author": sid,
               "flags": flag(d)}
        for f in RECALL_FIELDS:
            row[f] = _num(d, f)
        ref = iso.get(sid, {})
        row["iso_forget_rouge"] = ref.get("forget_rouge")
        if row["forget_rouge"] is not None and ref.get("forget_rouge") is not None:
            row["drop_rouge"] = round(ref["forget_rouge"] - row["forget_rouge"], 4)
        recall_rows.append(row)
    _write(f"{args.out_prefix}_own_recall.csv", recall_rows,
           ["label", "kind", "method", "n", "seed", "svd_rank", "probe_author"]
           + RECALL_FIELDS + ["iso_forget_rouge", "drop_rouge", "flags"])

    # ---- overlap table (geometry per N + per-probe row-mean, joined with drops) ----
    ov_path = os.path.join(REPORTS, f"nmerge_overlap_s{seed0}.json")
    if os.path.exists(ov_path):
        ov = _load(ov_path)
        drops = {}
        for r in recall_rows:
            if r["kind"] != "merge" or r["method"] != "additive_mean":
                continue
            # exact rows win; svd rows fill N where exact doesn't exist (128/200)
            key = (r["n"], r["probe_author"])
            if r["svd_rank"] and key in drops:
                continue
            if r.get("drop_rouge") is not None:
                drops[key] = r["drop_rouge"]
        ov_rows = []
        for entry in ov["per_n"]:
            base = {k: entry[k] for k in
                    ("n", "sampled", "cosine_offdiag_mean", "angB_offdiag_mean",
                     "angA_offdiag_mean", "angB_null_orth_mean",
                     "shared_energy_mean", "shared_energy_chance")}
            for a, rowmean in entry["probe_angB_rowmean"].items():
                r = dict(base)
                r["probe_author"] = int(a)
                r["probe_angB_rowmean"] = rowmean
                r["drop_rouge"] = drops.get((entry["n"], int(a)))
                ov_rows.append(r)
        _write(f"{args.out_prefix}_overlap.csv", ov_rows,
               ["n", "sampled", "probe_author", "probe_angB_rowmean", "drop_rouge",
                "cosine_offdiag_mean", "angB_offdiag_mean", "angA_offdiag_mean",
                "angB_null_orth_mean", "shared_energy_mean", "shared_energy_chance"])
    else:
        print(f"[analyze] no overlap JSON at {ov_path} (run submit_nmerge.sh overlap)")

    # ---- subset-conditioned utility table (retain_* restricted to merged authors) ----
    if subset_jsons:
        sub_rows = []
        for label, d in sorted(subset_jsons.items()):
            if d is None:
                continue
            info = parse_label(label)
            if info["kind"] == "merge":
                n, svd = info["n"], info.get("svd_rank")
            elif info["kind"] == "iso":
                n, svd = 1, None
            elif label.startswith(("ft_r32_sub", "base_model_sub")):
                n, svd = "", None
            else:
                continue
            rids = d.get("retain_author_ids") or []
            sub_rows.append({
                "label": label, "n": n, "svd_rank": svd or "",
                "n_retain_authors": len(rids),
                "retain_prob": _num(d, "retain_prob"),
                "retain_rouge": _num(d, "retain_rouge"),
                "retain_ppl": _num(d, "retain_ppl"),
                "retain_truth_scaled": _num(d, "retain_truth_scaled"),
                "model_utility": _num(d, "model_utility"),
                "flags": flag(d),
            })
        _write(f"{args.out_prefix}_subset_mu.csv", sub_rows,
               ["label", "n", "svd_rank", "n_retain_authors", "retain_prob", "retain_rouge",
                "retain_ppl", "retain_truth_scaled", "model_utility", "flags"])
        print("\nSubset-conditioned retain (what the merge was trained on):")
        for r in sorted([r for r in sub_rows if isinstance(r["n"], int)], key=lambda r: r["n"]):
            print(f"  N={r['n']:>3}{' svd' + str(r['svd_rank']) if r['svd_rank'] else '':>9}"
                  f"  retain_prob={r['retain_prob']}  rouge={r['retain_rouge']}"
                  f"  ppl={r['retain_ppl']}")
        for r in sub_rows:
            if not isinstance(r["n"], int):
                print(f"  anchor {r['label']}: retain_prob={r['retain_prob']} "
                      f"rouge={r['retain_rouge']} ppl={r['retain_ppl']}")

    # ---- console headline ----
    # One block per merge method present (was additive_mean-only): the sum/matched-norm arms
    # are separate methods and would otherwise print nothing.
    for meth in sorted({r["method"] for r in mu_rows
                        if r["kind"] == "merge" and r.get("method")}):
        ladder = [r for r in mu_rows
                  if r["kind"] == "merge" and r["method"] == meth and r["headline"]]
        if not ladder:
            continue
        print(f"\nN-ladder ({meth}, headline probe):")
        for r in sorted(ladder, key=lambda r: ((r["n"] or 0), r["seed"] or 0)):
            print(f"  N={r['n']:>3}{' svd' + str(r['svd_rank']) if r['svd_rank'] else '':>9}"
                  f"  s{r['seed']}  mu={r['model_utility']}  gmean={r['mu_gmean']}"
                  f"  retain_ppl={r['retain_ppl']}  {r['flags'] or ''}")
    for r in mu_rows:
        if r["kind"] == "anchor" and r["headline"]:
            print(f"  anchor {r['label']}: mu={r['model_utility']}")
        if r["kind"] == "merge" and r["method"] == "dare_ties" and r.get("r8"):
            print(f"  cross-check {r['label']}: mu={r['model_utility']} "
                  f"(expected ~{cfg.get('cross_check', {}).get('expected_mu', 0.4201)})")


if __name__ == "__main__":
    main()
