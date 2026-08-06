"""Assemble the composable_tv (ctv) curve CSVs from per-label eval JSONs (CPU, post-array).

  python analyze_ctv.py --config configs/ctv_1b_ctrl.json [configs/ctv_1b_lin.json ...]

Reads {out_dir}/results/{eval.cap}/*.json (written by submit_ctv.sh eval) for every config
given (multiple configs = the four arms aggregated into one curve table).

Label grammar (filenames; mirrors analyze_nmerge conventions):
  ctv_<arm>[_<variant>]_<sum|mean>_N<n>_s<seed>[__own<sid>|__subset].json
  iso_a<author>[_<variant>][__own<sid>].json
  base_model__own<sid>.json                      (per-probe base-floor anchor)

Row semantics (submit_ctv.sh eval writes them this way):
  plain        global metrics — mu, retain_prob, forget_* on the global forget shard.
  __own<sid>   probe rows: --eval_shard_id sid AND --retain_author_ids sid, so forget_* =
               the probe author's own recall (own_rouge/forget_ppl/own_truth) and
               retain_prob = the probe author's answer probability ("own_prob" — the H8
               headline channel: retain_prob under --retain_author_ids).
  __subset     --retain_author_ids <all merged authors> (subset-conditioned utility).

Outputs (deterministic — sorted rows, no timestamps):
  reports/ctv_curves.csv  one row per (label x probe) plus iso anchor rows: arm, variant,
                          scale (sum|mean|iso), N, probe, own_prob/own_rouge/own_truth/
                          forget_ppl, the label's global mu/retain_prob/forget_quality,
                          subset_retain_prob, iso + base-floor anchors, and the
                          extractable fractions ef_prob/ef_rouge = (own-floor)/(iso-floor).
  reports/ctv_dist.csv    per (arm, variant, scale, N) distribution stats: median/IQR of
                          own_prob and ef_prob, and the failure-tail fraction
                          (own_prob < tail_threshold x that probe's solo own_prob) — mu is
                          a guard-rail blind to per-author collapse, so tails are ALWAYS
                          reported (thread pre-registration).

Missing/corrupt result JSONs are tolerated (row skipped, counted); anchors missing for a
probe leave its ef_* blank unless --floor_prob/--floor_rouge supply a fallback floor.
No matplotlib (not installed in test-env) — CSV + stdout summary only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

ARMS = ("ctrl", "lin", "wd", "ds")
SCALES = ("sum", "mean")

_ISO_RE = re.compile(r"^iso_a(?P<a>\d+)(?:_(?P<variant>[A-Za-z][\w-]*))?$")

CURVE_COLS = [
    "arm", "variant", "scale", "n", "seed", "probe", "label",
    "own_prob", "own_rouge", "own_truth_ratio", "forget_ppl",
    "iso_prob", "iso_rouge", "floor_prob", "floor_rouge", "ef_prob", "ef_rouge", "tail",
    "mu", "retain_prob", "forget_quality", "subset_retain_prob", "flags",
]
DIST_COLS = [
    "arm", "variant", "scale", "n", "n_probes",
    "own_prob_median", "own_prob_iqr", "own_rouge_median",
    "ef_prob_median", "failure_tail_frac", "tail_threshold",
]


def parse_label(label):
    """Classify a result-file label (kind: merge | iso | floor | anchor | other).

    merge  -> {arm, variant, scale, n, seed}; the optional variant is every token between
              the arm and the scale token (wd: orthblock/rowslice), so the canonical
              no-variant grammar parses with variant == "".
    """
    m = _ISO_RE.match(label)
    if m:
        return {"kind": "iso", "author": int(m.group("a")),
                "variant": m.group("variant") or ""}
    if label == "base_model":
        return {"kind": "floor"}
    if label.startswith(("ft_", "retain90")):
        return {"kind": "anchor", "name": label}
    toks = label.split("_")
    if len(toks) >= 4 and toks[0] == "ctv" and toks[1] in ARMS:
        try:
            si = next(i for i in range(2, len(toks)) if toks[i] in SCALES)
        except StopIteration:
            return {"kind": "other"}
        rest = toks[si + 1:]
        if len(rest) != 2 or not rest[0].startswith("N") or not rest[1].startswith("s"):
            return {"kind": "other"}
        try:
            n, seed = int(rest[0][1:]), int(rest[1][1:])
        except ValueError:
            return {"kind": "other"}
        return {"kind": "merge", "arm": toks[1], "variant": "_".join(toks[2:si]),
                "scale": toks[si], "n": n, "seed": seed}
    return {"kind": "other"}


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _num(d, k):
    v = d.get(k) if d else None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v if isinstance(v, (int, float)) else None


def collect_jsons(results_dir):
    """({(label, sid_or_None): json}, {label: json}) — probe/plain rows and __subset rows."""
    out, subset = {}, {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)[: -len(".json")]
        if name.endswith(".progress") or "manifest" in name:
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


def extractable(own, iso, floor):
    """(own - floor) / (iso - floor); None when an anchor is missing or degenerate."""
    if own is None or iso is None or floor is None:
        return None
    denom = iso - floor
    if denom <= 0:
        return None
    return (own - floor) / denom


def flag(d):
    """Silent-failure flags for one result JSON (analyze_nmerge convention)."""
    flags = []
    mu = d.get("model_utility") if d else None
    if mu is None or (isinstance(mu, float) and math.isnan(mu)):
        flags.append("mu_nan")
    rp = _num(d, "retain_ppl")
    if rp is not None and rp > 100:
        flags.append(f"retain_ppl_explosion({rp:.0f})")
    return ";".join(flags)


def _round(v, nd=6):
    return "" if v is None else round(v, nd)


def _median(xs):
    return statistics.median(xs) if xs else None


def _iqr(xs):
    if len(xs) < 2:
        return 0.0 if xs else None
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[2] - q[0]


def analyze_config(cfg_path, tail_threshold, floor_prob_fb, floor_rouge_fb):
    """Curve rows for one arm config. Returns (rows, n_skipped)."""
    with open(cfg_path) as f:
        cfg = json.load(f)
    missing = [k for k in ("out_dir", "eval") if k not in cfg]
    if missing or "cap" not in (cfg.get("eval") or {}):
        raise SystemExit(f"[analyze_ctv] config {cfg_path} missing key(s): "
                         f"{missing or ['eval.cap']} — cannot locate results")
    out_dir = cfg["out_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
    results_dir = os.path.join(out_dir, "results", cfg["eval"]["cap"])
    jsons, subset_jsons = collect_jsons(results_dir)
    print(f"[analyze] {cfg.get('arm', '?')}: {len(jsons)} result JSONs "
          f"(+{len(subset_jsons)} __subset) in {results_dir}")

    # Anchors: iso[(author, variant)] and per-probe base floors.
    iso = {}
    for (label, sid), d in jsons.items():
        info = parse_label(label)
        if info["kind"] == "iso" and sid is not None and d is not None:
            # own row: retain_prob is the probe-restricted answer-prob (own_prob channel)
            iso[(info["author"], info["variant"])] = {
                "prob": _num(d, "retain_prob"), "rouge": _num(d, "forget_rouge"),
                "truth": _num(d, "forget_truth_ratio"), "ppl": _num(d, "forget_ppl"),
            }
    floor = {}
    for (label, sid), d in jsons.items():
        if parse_label(label)["kind"] == "floor" and sid is not None and d is not None:
            floor[sid] = {"prob": _num(d, "retain_prob"), "rouge": _num(d, "forget_rouge")}

    plain = {label: d for (label, sid), d in jsons.items()
             if sid is None and d is not None and parse_label(label)["kind"] == "merge"}

    rows, skipped = [], 0
    for (label, sid), d in sorted(jsons.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        info = parse_label(label)
        if d is None:
            skipped += info["kind"] in ("merge", "iso", "floor")
            continue
        if info["kind"] == "merge" and sid is not None:
            arm, variant, scale, n, seed = (info["arm"], info["variant"], info["scale"],
                                            info["n"], info["seed"])
        elif info["kind"] == "iso" and sid is not None:
            # solo anchors surface as scale="iso" N=1 rows (curve floor/ceiling context)
            arm = cfg.get("arm", "")
            variant, scale, n, seed = info["variant"], "iso", 1, cfg.get("pool_seed", "")
        else:
            continue
        own_prob = _num(d, "retain_prob")
        own_rouge = _num(d, "forget_rouge")
        iso_ref = iso.get((sid, variant), {})
        fl = floor.get(sid, {})
        fp = fl.get("prob", floor_prob_fb) if fl.get("prob") is not None else floor_prob_fb
        fr = fl.get("rouge", floor_rouge_fb) if fl.get("rouge") is not None else floor_rouge_fb
        ef_prob = extractable(own_prob, iso_ref.get("prob"), fp)
        ef_rouge = extractable(own_rouge, iso_ref.get("rouge"), fr)
        tail = ""
        if own_prob is not None and iso_ref.get("prob") is not None:
            tail = int(own_prob < tail_threshold * iso_ref["prob"])
        pd = plain.get(label)
        sd = subset_jsons.get(label)
        rows.append({
            "arm": arm, "variant": variant, "scale": scale, "n": n, "seed": seed,
            "probe": sid, "label": label,
            "own_prob": _round(own_prob), "own_rouge": _round(own_rouge),
            "own_truth_ratio": _round(_num(d, "forget_truth_ratio")),
            "forget_ppl": _round(_num(d, "forget_ppl")),
            "iso_prob": _round(iso_ref.get("prob")), "iso_rouge": _round(iso_ref.get("rouge")),
            "floor_prob": _round(fp), "floor_rouge": _round(fr),
            "ef_prob": _round(ef_prob), "ef_rouge": _round(ef_rouge), "tail": tail,
            "mu": _round(_num(pd, "model_utility") if pd else None),
            "retain_prob": _round(_num(pd, "retain_prob") if pd else None),
            "forget_quality": _round(_num(pd, "forget_quality") if pd else None),
            "subset_retain_prob": _round(_num(sd, "retain_prob") if sd else None),
            "flags": flag(d),
        })
    if skipped:
        print(f"[analyze] WARNING: {skipped} unreadable result JSON(s) skipped in {results_dir}")
    return rows


def dist_rows(rows, tail_threshold):
    """Per-(arm, variant, scale, N) distribution stats over the probe rows."""
    groups = {}
    for r in rows:
        if r["scale"] == "iso":
            continue
        groups.setdefault((r["arm"], r["variant"], r["scale"], r["n"]), []).append(r)
    out = []
    for (arm, variant, scale, n), grp in sorted(groups.items()):
        probs = [r["own_prob"] for r in grp if r["own_prob"] != ""]
        rouges = [r["own_rouge"] for r in grp if r["own_rouge"] != ""]
        efs = [r["ef_prob"] for r in grp if r["ef_prob"] != ""]
        tails = [r["tail"] for r in grp if r["tail"] != ""]
        out.append({
            "arm": arm, "variant": variant, "scale": scale, "n": n, "n_probes": len(grp),
            "own_prob_median": _round(_median(probs)),
            "own_prob_iqr": _round(_iqr(probs)),
            "own_rouge_median": _round(_median(rouges)),
            "ef_prob_median": _round(_median(efs)),
            "failure_tail_frac": _round(sum(tails) / len(tails)) if tails else "",
            "tail_threshold": tail_threshold,
        })
    return out


def _write(out, rows, cols):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out} ({len(rows)} rows)")


def print_summary(drows):
    print("\nctv ladder (per arm x scale; own_prob median [IQR], extractable fraction, tails):")
    last = None
    for r in drows:
        head = (r["arm"], r["variant"], r["scale"])
        if head != last:
            vtag = f" variant={r['variant']}" if r["variant"] else ""
            print(f"  arm={r['arm']}{vtag} scale={r['scale']}")
            last = head
        med = r["own_prob_median"] if r["own_prob_median"] != "" else "?"
        iqr = r["own_prob_iqr"] if r["own_prob_iqr"] != "" else "?"
        ef = r["ef_prob_median"] if r["ef_prob_median"] != "" else "?"
        tf = r["failure_tail_frac"] if r["failure_tail_frac"] != "" else "?"
        print(f"    N={r['n']:>3}  probes={r['n_probes']}  own_prob={med} [{iqr}]  "
              f"ef={ef}  tail_frac={tf}")


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True,
                    help="one or more ctv arm configs (rows aggregated)")
    ap.add_argument("--out_prefix", default=os.path.join(REPORTS, "ctv"),
                    help="writes <prefix>_curves.csv + <prefix>_dist.csv")
    ap.add_argument("--tail_threshold", type=float, default=0.5,
                    help="failure tail = own_prob below this fraction of the probe's solo own_prob")
    ap.add_argument("--floor_prob", type=float, default=None,
                    help="fallback base-floor answer-prob when base_model__own<probe>.json is missing")
    ap.add_argument("--floor_rouge", type=float, default=None,
                    help="fallback base-floor rouge when base_model__own<probe>.json is missing")
    return ap


def run(args):
    rows = []
    for cfg_path in args.config:
        rows.extend(analyze_config(cfg_path, args.tail_threshold,
                                   args.floor_prob, args.floor_rouge))
    rows.sort(key=lambda r: (r["arm"], r["variant"], r["scale"], r["n"], r["probe"]))
    drows = dist_rows(rows, args.tail_threshold)
    _write(f"{args.out_prefix}_curves.csv", rows, CURVE_COLS)
    _write(f"{args.out_prefix}_dist.csv", drows, DIST_COLS)
    print_summary(drows)
    return rows, drows


def main():
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
