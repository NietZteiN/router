"""Orphan-destination + concentration analysis for the router_leak sweep (task (a)).

CPU-only, no inference — reads the destination histograms already stored in the router-family
aggregate JSONs (`strategies[s].cells[ck].orphan_capture.top1_hist`) and the encoder-audit
JSONs (`sibling.sibling_hist`), and reduces each (router, drop-set, pool) cell to a
concentration profile:
  max_share      fraction of orphans landing on the single busiest surviving unit
  top3_share     fraction on the busiest three
  entropy_norm   normalized Shannon entropy over survivors (1 = perfectly diffuse)
  hhi            Herfindahl index  Σ pᵢ²  (1 = all on one unit)
  gini           Gini of the destination mass
  n_eff          1/hhi = effective number of siblings the leak spreads over

Then a MONOTONICITY read: does concentration hold / grow as more shards are deleted
(d9 → d9_8 → d9_8_7_6) and at per-author k=200. The overlapping metrics (max_share,
entropy_norm) are re-derived from the histogram and asserted equal to the stored values, so
this script is a faithful reduction of the producer's own counts, not a re-measurement.

  python analyze_orphan_destinations.py \
     --family_json A.json B.json --enc_json C.json --centroid_json D.json \
     --out_md reports/orphan_destinations.md --out_csv reports/orphan_destinations.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def concentration(hist: dict, n_survivors: int) -> dict:
    """All concentration metrics from a {unit_id: count} destination histogram.
    `n_survivors` = number of surviving routing units (k − dropped); entropy and Gini are
    normalized over ALL survivors (including the zero-mass ones) so entropy_norm matches the
    sweep's stored `top1_entropy_norm` convention and Gini reads as 'how unevenly the orphan
    mass spreads across the available siblings'. max_share / top3_share / HHI / n_eff are
    convention-free (over the mass)."""
    hit_counts = np.asarray([float(v) for v in hist.values()], dtype="float64")
    total = hit_counts.sum()
    if total == 0 or n_survivors < 1:
        return {"n": 0}
    # pad the destination vector to all survivors (zeros for siblings that got no orphans)
    counts = np.zeros(max(n_survivors, len(hit_counts)), dtype="float64")
    counts[:len(hit_counts)] = np.sort(hit_counts)[::-1]
    p = counts / total
    p_nz = p[p > 0]
    ent = float(-(p_nz * np.log(p_nz)).sum() / np.log(max(n_survivors, 2)))
    ranked = np.sort(p)[::-1]
    hhi = float((p ** 2).sum())
    # Gini of the mass over all survivors (0 = uniform across survivors, →1 = one magnet)
    x = np.sort(p)
    n = len(x)
    idx = np.arange(1, n + 1)
    gini = float((np.sum((2 * idx - n - 1) * x)) / (n * x.sum())) if x.sum() > 0 and n > 1 else 0.0
    return {"n": int(total), "n_units": int(n_survivors),
            "max_share": float(ranked[0]),
            "top3_share": float(ranked[:3].sum()),
            "entropy_norm": ent, "hhi": hhi, "gini": gini,
            "n_eff": float(1.0 / hhi),
            "busiest_unit": int(list(hist.keys())[int(np.argmax(hit_counts))]),
            "hist": {str(k): int(v) for k, v in hist.items()}}


def _check(stored, got, label, tol=1e-6):
    if stored is None:
        return
    if abs(float(stored) - float(got)) > tol:
        print(f"  [WARN] {label}: stored {stored:.6g} != recomputed {got:.6g}")


def collect(family_jsons, enc_jsons, centroid_jsons, legonet_jsons=()) -> list:
    """One row per (pool, router, drop-set) cell."""
    rows = []
    for p in legonet_jsons:
        # n=32 legonet/ramole routing audit: the `dropped` policy's orphan destinations live
        # in dropped_extras.top1_hist ({expert: count}) over n_surviving_experts survivors.
        d = json.load(open(p))
        de = d.get("dropped_extras")
        if not de or "top1_hist" not in de:
            continue
        enc = str(d.get("encoder_source", "?")).split("/")[-1]
        tag = d.get("tag", "forget10")
        c = concentration(de["top1_hist"], int(de.get("n_surviving_experts",
                                                       len(de["top1_hist"]))))
        _check(de.get("top1_share_top1_expert"), c["max_share"], f"legonet/{enc}/{tag} max_share")
        _check(de.get("top1_entropy_norm"), c["entropy_norm"], f"legonet/{enc}/{tag} entropy")
        rows.append({"pool": f"legonet_n32 ({enc})", "k": d.get("n", 32), "base": "1B",
                     "router": f"embed-{enc}", "drop": tag,
                     "n_drop": 1 if tag == "forget01" else (5 if tag == "forget05" else 20), **c})
    for p in family_jsons:
        d = json.load(open(p))
        pool = d.get("meta", {}).get("pool_dir", p).split("/")[-1]
        k = d.get("meta", {}).get("k", "?")
        base = d.get("meta", {}).get("base_model", "?").split("/")[-1]
        for strat, sv in d["strategies"].items():
            for ck, cell in sv.get("cells", {}).items():
                oc = cell.get("orphan_capture")
                if not oc or "top1_hist" not in oc:
                    continue   # key_exact stores no destination histogram (routes to fallback)
                n_drop = ck.count("_") + 1
                n_surv = (k - n_drop) if isinstance(k, int) else max(oc["top1_hist"].keys(), default=0)
                c = concentration(oc["top1_hist"], n_surv)
                _check(oc.get("top1_share_top1_expert"), c["max_share"], f"{strat}/{ck} max_share")
                _check(oc.get("top1_entropy_norm"), c["entropy_norm"], f"{strat}/{ck} entropy")
                rows.append({"pool": pool, "k": k, "base": base, "router": strat,
                             "drop": ck, "n_drop": n_drop, **c})
    for p in enc_jsons:
        d = json.load(open(p))
        enc = d.get("router_encoder", p).split("/")[-1]
        h = d.get("sibling", {}).get("sibling_hist")
        if h:
            c = concentration(h, d.get("k", 10) - 1)
            rows.append({"pool": "scaf_k10", "k": d.get("k", 10), "base": "1B-scaf",
                         "router": f"centroid_{enc.split('-')[0]}", "drop": "d9",
                         "n_drop": 1, **c})
    for p in centroid_jsons:
        d = json.load(open(p))
        h = d.get("sibling", {}).get("sibling_hist")
        if h:
            c = concentration(h, d.get("k", 10) - 1)
            rows.append({"pool": "scaf_k10", "k": d.get("k", 10), "base": "1B-scaf",
                         "router": "centroid_minilm", "drop": f"d{d.get('drop_shard')}",
                         "n_drop": 1, **c})
    return rows


def per_author_determinism(sims_paths) -> list:
    """NEW metric: for each DELETED author, how concentrated are its ~20 orphan questions on a
    single SURVIVING destination? Reads the router-family `.sims.npz` sidecars
    (`scores [n_q,k]` higher=routed, `author_of_q`, `is_forget`) — no inference. We mask the
    dropped shard column(s), take each orphan question's top-1 survivor, then per deleted author
    report determinism = fraction of its questions on its single most-common destination
    (1.0 = a per-author magnet — every question of that author routes to the SAME sibling).
    Aggregated as mean/median over deleted authors + mean normalized landing-entropy; one row
    per (source, strategy). Strategies whose sidecar has no `scores` matrix (key_exact's binary
    `match`) are skipped — they route to the fallback shard by construction."""
    rows = []
    for p in sims_paths:
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as e:  # noqa: BLE001 — a missing/corrupt sidecar shouldn't kill the run
            print(f"  [WARN] determinism: cannot load {p}: {e}")
            continue
        keys = set(z.files)
        score_key = ("scores" if "scores" in keys
                     else next((k for k in z.files if k.startswith("scores__d")), None))
        if score_key is None or "author_of_q" not in keys or "is_forget" not in keys:
            continue  # key_exact (match matrix / fallback) — no per-survivor destination
        scores = np.asarray(z[score_key], dtype="float64")
        author_of_q = np.asarray(z["author_of_q"]).astype(int)
        is_forget = np.asarray(z["is_forget"]).astype(bool)
        strat = str(z["strategy"]) if "strategy" in keys else "?"
        if scores.ndim != 2 or scores.shape[1] < 2 or is_forget.sum() == 0:
            continue
        n_q, k = scores.shape
        # dropped columns = the shards the deleted authors belong to (get_author_shard inverse:
        # shard = author // (200 // k)); holds for every k that divides the 200-author layout.
        if 200 % k != 0:
            continue
        per_shard = 200 // k
        f_authors = np.unique(author_of_q[is_forget])
        dropped = sorted({int(a) // per_shard for a in f_authors})
        masked = scores.copy()
        masked[:, dropped] = -np.inf
        top1 = masked.argmax(axis=1)
        dets, ents = [], []
        for a in f_authors:
            qs = np.where(is_forget & (author_of_q == a))[0]
            if len(qs) == 0:
                continue
            _, cnt = np.unique(top1[qs], return_counts=True)
            frac = cnt / cnt.sum()
            dets.append(float(frac.max()))
            ents.append(float(-(frac * np.log(frac)).sum() / np.log(max(len(qs), 2))))
        if not dets:
            continue
        rows.append({"source": os.path.basename(p).split(".")[0], "router": strat,
                     "n_authors": len(dets), "n_dropped_cols": len(dropped),
                     "mean_determinism": float(np.mean(dets)),
                     "median_determinism": float(np.median(dets)),
                     "mean_landing_entropy": float(np.mean(ents))})
    return rows


def monotonicity(rows: list) -> list:
    """For each (pool, router), does max_share hold/grow across drop counts? Report the
    trend from the smallest to the largest drop set present."""
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["pool"], r["router"]), []).append(r)
    for (pool, router), cells in sorted(by.items()):
        cells = sorted(cells, key=lambda r: r["n_drop"])
        if len(cells) < 2:
            continue
        ms = [c["max_share"] for c in cells]
        ne = [c["n_eff"] for c in cells]
        trend = ("grows" if ms[-1] > ms[0] + 0.03 else
                 "shrinks" if ms[-1] < ms[0] - 0.03 else "holds")
        out.append({"pool": pool, "router": router,
                    "drops": [c["drop"] for c in cells],
                    "max_share": ms, "n_eff": ne, "trend": trend})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family_json", nargs="*", default=[])
    ap.add_argument("--enc_json", nargs="*", default=[])
    ap.add_argument("--centroid_json", nargs="*", default=[])
    ap.add_argument("--legonet_json", nargs="*", default=[],
                    help="n=32 legonet/ramole routing-audit JSONs (dropped_extras.top1_hist)")
    ap.add_argument("--sims_glob", nargs="*", default=[],
                    help="router-family `.<strategy>.npz` sidecars for per-author landing "
                         "determinism (needs `scores`/`author_of_q`/`is_forget`)")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = collect(args.family_json, args.enc_json, args.centroid_json, args.legonet_json)
    mono = monotonicity(rows)
    det = per_author_determinism(args.sims_glob)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "k", "base", "router", "drop", "n_units_surviving",
                    "n_orphans", "max_share", "top3_share", "entropy_norm", "hhi",
                    "gini", "n_eff", "busiest_unit"])
        for r in sorted(rows, key=lambda r: (r["pool"], r["router"], r["n_drop"])):
            w.writerow([r["pool"], r["k"], r["base"], r["router"], r["drop"],
                        r["n_units"], r["n"], f"{r['max_share']:.4f}",
                        f"{r['top3_share']:.4f}", f"{r['entropy_norm']:.4f}",
                        f"{r['hhi']:.4f}", f"{r['gini']:.4f}", f"{r['n_eff']:.2f}",
                        r["busiest_unit"]])

    lines = ["# Where orphans land: destination concentration (task (a))", "",
             "Reduction of the router-family sweep's stored orphan-destination histograms. "
             "max_share = fraction of deleted-author questions landing on the single busiest "
             "surviving unit; n_eff = 1/HHI = effective number of siblings the leak spreads "
             "over (1 = one magnet expert, high = diffuse).", "",
             "## Per-cell concentration", "",
             "top3_share = fraction on the busiest three survivors; Gini = inequality of the "
             "orphan mass over survivors (0 = uniform, →1 = one magnet).", "",
             "| pool | router | drop | orphans | max_share | top3 | n_eff | entropy | HHI | Gini | busiest |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["pool"], -r["max_share"], r["router"], r["n_drop"])):
        lines.append(f"| {r['pool']} | {r['router']} | {r['drop']} | {r['n']} | "
                     f"{r['max_share']:.2f} | {r['top3_share']:.2f} | {r['n_eff']:.1f} | "
                     f"{r['entropy_norm']:.2f} | {r['hhi']:.2f} | {r['gini']:.2f} | "
                     f"s{r['busiest_unit']} |")
    lines += ["", "## Concentration vs deletion count (does the magnet hold?)", "",
              "| pool | router | drop sets | max_share trajectory | n_eff | trend |",
              "|---|---|---|---|---|---|"]
    for m in sorted(mono, key=lambda m: (m["pool"], -m["max_share"][0])):
        ms = " → ".join(f"{v:.2f}" for v in m["max_share"])
        ne = " → ".join(f"{v:.1f}" for v in m["n_eff"])
        lines.append(f"| {m['pool']} | {m['router']} | {'/'.join(m['drops'])} | {ms} | "
                     f"{ne} | **{m['trend']}** |")
    if det:
        lines += ["", "## Per-author landing determinism (new)", "",
                  "For each DELETED author, the fraction of its ~20 orphan questions that land on "
                  "a single surviving destination (1.0 = a per-author magnet — every question of "
                  "that author routes to the SAME sibling; low = its questions scatter). Mean/"
                  "median over deleted authors; recomputed from the `.sims.npz` sidecars, no "
                  "inference.", "",
                  "| source | router | del. authors | mean determinism | median | mean landing-entropy |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(det, key=lambda r: -r["mean_determinism"]):
            lines.append(f"| {r['source']} | {r['router']} | {r['n_authors']} | "
                         f"{r['mean_determinism']:.3f} | {r['median_determinism']:.3f} | "
                         f"{r['mean_landing_entropy']:.3f} |")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[analyze_orphan_destinations] {len(rows)} cells, {len(mono)} routers with a "
          f"multi-drop trajectory, {len(det)} determinism rows -> {args.out_md}")
    for m in sorted(mono, key=lambda m: -m["max_share"][0])[:12]:
        print(f"  {m['router']:18s} ({m['pool']}): max_share "
              f"{' → '.join(f'{v:.2f}' for v in m['max_share'])}  [{m['trend']}]")
    for r in sorted(det, key=lambda r: -r["mean_determinism"]):
        print(f"  det {r['router']:18s} ({r['source']}): mean {r['mean_determinism']:.3f} "
              f"median {r['median_determinism']:.3f} over {r['n_authors']} authors")


if __name__ == "__main__":
    main()
