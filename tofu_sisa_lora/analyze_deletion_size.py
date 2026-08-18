#!/usr/bin/env python3
"""Routing-level metrics as a function of HOW MANY sources were deleted.

Everything the selector audit reports is measured at one deletion size (20 of 200 = TOFU
forget10). This sweeps the size on the score matrices that already exist
(`analyze_router_shift.py --dump_npz`), so the whole ladder is CPU and costs nothing.

The trap this file exists to avoid — and which the published k=50 cells fell into (see
CLAUDE.md) — is holding `is_forget` FIXED while the drop set shrinks. A row is an orphan only if
ITS OWN author was deleted, so at d=5 the other 15 forget10 authors are RETAINED rows with their
experts present. Labelling them orphans measures nothing. `is_forget` is recomputed at every
rung here, from the drop set of that rung.

Routing is post-deletion throughout: the argmax is taken over SURVIVING units only. The d=0 rung
is the pre-deletion reference RDR is measured against.

  python analyze_deletion_size.py --npz_dir .../shift_npz_k200 \\
      --sizes 0,1,2,5,10,20,40 --out_json ... --out_md ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

from analyze_router_probe import probe_arrays          # noqa: E402
from shard_utils import parse_author_ids               # noqa: E402


def _n_eff(counts) -> float:
    """1/HHI — the effective number of destinations. 1.0 = one magnet takes everything."""
    c = np.asarray([v for v in counts if v > 0], dtype="float64")
    if c.sum() <= 0:
        return float("nan")
    p = c / c.sum()
    return float(1.0 / np.square(p).sum())


def rung(M: np.ndarray, authors: np.ndarray, k: int, drop_ids: list, strategy: str,
         attacker_id: int, base_top1: np.ndarray, seed: int, m_top: int) -> dict:
    """One deletion size, one strategy, one query condition."""
    drop = set(int(x) for x in drop_ids)
    surv = [j for j in range(k) if j not in drop]
    per_shard = 200 // k
    true_unit = authors // per_shard
    att_unit = attacker_id // per_shard

    # post-deletion routing: argmax over survivors, mapped back to unit ids
    sub = M[:, surv]
    top1 = np.asarray(surv)[sub.argmax(axis=1)]

    is_orphan = np.isin(true_unit, sorted(drop))
    retained = ~is_orphan

    out = {"n_deleted": len(drop), "n_orphan_rows": int(is_orphan.sum()),
           "n_retained_rows": int(retained.sum())}

    out["routing_accuracy_retain"] = (float((top1[retained] == true_unit[retained]).mean())
                                      if retained.any() else float("nan"))
    notatt = true_unit != att_unit
    out["attacker_capture"] = (float((top1[notatt] == att_unit).mean())
                               if notatt.any() else float("nan"))

    # RDR: retained rows whose served unit MOVED because of a deletion nobody asked for.
    if base_top1 is None:
        out["rdr"] = None
    else:
        out["rdr"] = (float((top1[retained] != base_top1[retained]).mean())
                      if retained.any() else float("nan"))

    if is_orphan.any():
        dest = np.bincount(top1[is_orphan], minlength=k)
        out["orphan_busiest_share"] = float(dest.max() / dest.sum())
        out["orphan_n_eff"] = _n_eff(dest)
    else:
        out["orphan_busiest_share"] = out["orphan_n_eff"] = None

    # Detection needs both classes; at d=0 there are no orphans at all.
    if strategy == "key_exact":
        out["detection_auc"] = None            # no graded score to threshold
        out["no_match_rate"] = float((M.sum(axis=1) == 0).mean())
    elif is_orphan.any() and retained.any():
        pr = probe_arrays(sub, is_orphan.astype(int), authors, k, strategy, sorted(drop),
                          seed=seed, m_top=m_top)
        conf = [v["auc"] for n, v in pr.get("comparators", {}).items()
                if not n.startswith("tomb_")]
        out["detection_auc"] = max(conf) if conf else None
        out["probe_auc"] = pr.get("probe", {}).get("auc")
        out["n_eval"] = pr.get("n_eval")
    else:
        out["detection_auc"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True,
                    help="dir of <strategy>__<condition>.npz from analyze_router_shift --dump_npz")
    ap.add_argument("--sizes", default="0,1,2,5,10,20",
                    help="Deletion sizes. Each is a PREFIX of --delete_order, so the rungs nest. "
                         "The default stops at 20 because the 800-row evaluation set only covers "
                         "authors 0-19 and 180-199: past 20 the extra deleted authors contribute "
                         "NO orphan rows, so orphan-side metrics stop being about the deletion "
                         "size. Larger rungs are allowed and flagged, not silently averaged.")
    ap.add_argument("--delete_order", default="180-199",
                    help="Deletion order. Prefixes of this give the ladder; sizes beyond its "
                         "length extend downward from 179 so the ladder can pass 20.")
    ap.add_argument("--strategies", default="centroid_sbert,key_tfidf,key_exact")
    ap.add_argument("--conditions", default="original,name_stripped,name_injected,name_swapped")
    ap.add_argument("--attacker_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m_top", type=int, default=20)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    order = list(parse_author_ids(args.delete_order))
    # Extend downward (179, 178, ...) so a rung past forget10 is still a nested superset.
    extra = [a for a in range(order[0] - 1, -1, -1)]
    order = order + extra
    sizes = [int(s) for s in args.sizes.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    res = {"meta": {"npz_dir": args.npz_dir, "sizes": sizes,
                    "delete_order": args.delete_order, "attacker_id": args.attacker_id,
                    "seed": args.seed}, "cells": {}}

    for strat in strategies:
        res["cells"][strat] = {}
        for cond in conditions:
            p = os.path.join(args.npz_dir, f"{strat}__{cond}.npz")
            if not os.path.exists(p):
                print(f"[ladder] MISSING {os.path.basename(p)} — skipped", flush=True)
                continue
            z = np.load(p, allow_pickle=True)
            M = np.asarray(z["match" if "match" in z else "scores"], dtype="float64")
            authors = np.asarray(z["author_of_q"], dtype=int)
            k = int(z["k"])
            base_top1 = M.argmax(axis=1)          # pre-deletion served unit, the RDR reference
            rows = []
            for d in sizes:
                if d > len(order):
                    continue
                rows.append(rung(M, authors, k, order[:d], strat, args.attacker_id,
                                 base_top1, args.seed, args.m_top))
            res["cells"][strat][cond] = rows
            print(f"[ladder] {strat:16s} {cond:14s} {len(rows)} rungs", flush=True)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[ladder] -> {args.out_json}")

    if args.out_md:
        L = ["# Routing metrics vs number of sources deleted", "",
             f"k = 200 per-author units · deletion order `{args.delete_order}` (nested prefixes) "
             f"· attacker = author {args.attacker_id} · 800 rows.", "",
             "`is_forget` is recomputed at every rung — a row is an orphan only if its OWN author "
             "was deleted — and routing is post-deletion (argmax over survivors). `RDR` is the "
             "share of RETAINED rows whose served unit moved versus no deletion at all.", ""]
        for strat in res["cells"]:
            for cond, rows in res["cells"][strat].items():
                if not rows:
                    continue
                L += [f"## `{strat}` · `{cond}`", "",
                      "| deleted | orphan rows | routing acc (retain) | detection AUC | RDR "
                      "| attacker capture | orphan n_eff |", "|---|---|---|---|---|---|---|"]
                capped = False
                prev_orphans = -1
                for r in rows:
                    def f(x, nd=4):
                        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) \
                            else f"{x:.{nd}f}"
                    # Past the evaluation set's author coverage the orphan count stops growing.
                    # Mark those rungs: their orphan-side columns describe the same 400 rows as
                    # the rung before, so reading them as a plateau in the deletion size is wrong.
                    flag = ""
                    if r["n_deleted"] > 0 and r["n_orphan_rows"] == prev_orphans:
                        flag, capped = " ⚠", True
                    prev_orphans = r["n_orphan_rows"]
                    L.append(f"| {r['n_deleted']}{flag} | {r['n_orphan_rows']} | "
                             f"{f(r['routing_accuracy_retain'])} | {f(r.get('detection_auc'))} | "
                             f"{f(r['rdr'])} | {f(r['attacker_capture'])} | "
                             f"{f(r.get('orphan_n_eff'), 1)} |")
                L.append("")
                if capped:
                    L += ["> ⚠ At these rungs the extra deleted authors have **no rows in the "
                          "evaluation set** (it covers authors 0–19 and 180–199 only), so the "
                          "orphan count does not grow. `RDR` and `routing acc` remain meaningful "
                          "— the survivor pool really is smaller — but the orphan-side columns "
                          "describe the same rows as the rung above and must not be read as a "
                          "plateau in deletion size.", ""]
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        print(f"[ladder] -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
