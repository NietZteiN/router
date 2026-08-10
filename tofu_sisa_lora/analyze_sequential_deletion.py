"""Sequential deletion — magnet saturation and retained displacement (§4.2 F2, and RDR).

Two questions that need no GPU, both answered from a FAMILY NPZ CONTRACT score matrix:

  MAGNET SATURATION  delete sources one at a time and watch where the orphans go. If one
                     survivor progressively becomes the answerer for a growing share of the
                     deleted corpus, deletion is not just leaky but CUMULATIVELY leaky, and that
                     survivor's own behaviour is the thing to worry about.
  RDR                Retained Displacement Rate: the fraction of RETAINED queries whose selected
                     unit changes after a deletion nobody asked for. Deletion is supposed to be
                     local to the deleted source; RDR is how far that fails.

Both are computed on the survivor-restricted argmax, the same quantity the serving router would
use, and both are reported for whatever query conditions the caller supplies — because the
2026-08-07 result is that gold-form TOFU questions flatter every selector metric, so a
displacement number measured only on them would be as misleading as the detection numbers were.

  python analyze_sequential_deletion.py --self_test
  python analyze_sequential_deletion.py --family_npz '<...>/rl_family_k200.*.npz' \
      --delete_order 180-199 --out_json reports/seqdel_k200.json --out_md reports/seqdel_k200.md
"""
from __future__ import annotations

import argparse
import glob as globlib
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

from analyze_router_probe import parse_drop_set, _f
from analyze_router_family import _npz_str


def sequential_curve(scores: np.ndarray, authors: np.ndarray, k: int, order: list) -> list:
    """One row per deletion step: where the accumulated orphans land, and how much retained
    traffic moved. `order` is the order sources are deleted in, one at a time."""
    per_shard = 200 // k
    unit_of_q = authors // per_shard
    deleted, rows = [], []
    # baseline: nothing deleted, so every query's unit is its own router choice
    base_top1 = scores.argmax(axis=1)
    for step, src in enumerate(order, start=1):
        deleted.append(int(src))
        dset = set(deleted)
        surv = [j for j in range(k) if j not in dset]
        S = scores[:, surv]
        top1 = np.asarray(surv, dtype=int)[S.argmax(axis=1)]

        is_orphan = np.isin(unit_of_q, np.asarray(deleted))
        orph_dest = top1[is_orphan]
        counts = np.bincount(orph_dest, minlength=k).astype("float64")
        total = counts.sum()
        p = counts / max(total, 1)
        busiest = int(counts.argmax()) if total else None
        hhi = float((p ** 2).sum()) if total else float("nan")

        # RDR: retained queries whose unit changed versus the no-deletion baseline. Queries whose
        # own unit was deleted are excluded — those are orphans, not displacement.
        retained = ~is_orphan
        rdr = float((top1[retained] != base_top1[retained]).mean()) if retained.any() else float("nan")

        rows.append({
            "step": step, "deleted_source": int(src), "n_deleted": len(deleted),
            "n_orphans": int(is_orphan.sum()),
            "busiest_unit": busiest,
            "busiest_share": float(p.max()) if total else float("nan"),
            "busiest_n": int(counts.max()) if total else 0,
            "n_eff": float(1.0 / hhi) if total and hhi > 0 else float("nan"),
            "RDR": rdr,
        })
    return rows


def saturation_verdict(rows: list) -> dict:
    """Does ONE survivor progressively absorb the orphans (saturation), or does the mass spread
    as more sources are deleted? Stated from the curve rather than eyeballed."""
    if len(rows) < 3:
        return {"verdict": "too few steps"}
    shares = [r["busiest_share"] for r in rows]
    units = [r["busiest_unit"] for r in rows]
    tail = rows[len(rows) // 2:]
    same = sum(1 for r in tail if r["busiest_unit"] == rows[-1]["busiest_unit"]) / len(tail)
    return {
        "first_share": shares[0], "last_share": shares[-1],
        "delta_share": float(shares[-1] - shares[0]),
        "final_busiest_unit": units[-1],
        "tail_stability": float(same),
        "final_n_eff": rows[-1]["n_eff"],
        "final_RDR": rows[-1]["RDR"],
        # A share that GROWS is saturation in progress; a share already high and stable is
        # saturation already reached. Requiring growth would call a router that captured
        # everything from the first deletion "flat", which is the opposite of the finding.
        "verdict": ("saturating — one survivor takes a growing share"
                    if shares[-1] > shares[0] + 0.02 and same >= 0.8 else
                    "saturated — one survivor already holds most orphans"
                    if shares[-1] >= 0.5 and same >= 0.8 else
                    "dispersing — orphan mass spreads as more sources go"
                    if shares[-1] < shares[0] - 0.02 else "flat"),
    }


def write_md(res: dict, path: str) -> None:
    L = ["# Sequential deletion — magnet saturation and retained displacement", "",
         "Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated "
         "orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of "
         "RETAINED queries whose selected unit changed versus no deletion at all — displacement "
         "nobody asked for.", ""]
    for strat, blk in res["strategies"].items():
        v = blk["verdict"]
        L += [f"## `{strat}`", "",
              f"**{v.get('verdict')}** — busiest share {_f(v.get('first_share'))} → "
              f"{_f(v.get('last_share'))} (Δ {_f(v.get('delta_share'))}), final unit "
              f"{v.get('final_busiest_unit')}, tail stability {_f(v.get('tail_stability'))}, "
              f"final n_eff {_f(v.get('final_n_eff'), 1)}, final RDR "
              f"**{_f(v.get('final_RDR'))}**.", "",
              "| step | deleted | orphans | busiest unit | share | n_eff | RDR |",
              "|---|---|---|---|---|---|---|"]
        rows = blk["curve"]
        show = rows if len(rows) <= 12 else rows[:3] + rows[len(rows) // 2 - 1:len(rows) // 2 + 1] + rows[-3:]
        for r in show:
            L.append(f"| {r['step']} | {r['deleted_source']} | {r['n_orphans']} | "
                     f"{r['busiest_unit']} | {_f(r['busiest_share'])} | {_f(r['n_eff'], 1)} | "
                     f"{_f(r['RDR'])} |")
        if len(rows) > 12:
            L.append("| … | | | | | | |")
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def run_self_test() -> None:
    n = 0

    def ok(name):
        nonlocal n
        n += 1
        print(f"  PASS {name}")

    # author_of_q is an AUTHOR id (0..199), which the code maps to a unit via 200//k — the same
    # convention as the FAMILY NPZ CONTRACT. A fixture that passed unit ids instead would make
    # every query look like unit 0 and silently produce zero orphans.
    k = 8
    n_q = 200
    authors = np.arange(200)              # one query per author
    units = authors // (200 // k)         # 25 authors per unit
    # planted MAGNET: unit 0 is every query's second choice, so every orphan lands there
    S = np.full((n_q, k), 0.1)
    S[np.arange(n_q), units] = 1.0
    S[:, 0] = 0.5
    rows = sequential_curve(S, authors, k, [7, 6, 5])
    assert all(r["busiest_unit"] == 0 for r in rows), [r["busiest_unit"] for r in rows]
    assert rows[-1]["busiest_share"] == 1.0
    assert saturation_verdict(rows + rows)["verdict"].startswith("saturat"), \
        saturation_verdict(rows + rows)["verdict"]
    ok("magnet fixture: every orphan lands on the planted survivor, verdict saturating")

    # RDR must be 0 when deletions do not touch retained queries' own units
    assert all(r["RDR"] == 0.0 for r in rows), [r["RDR"] for r in rows]
    ok("RDR is 0 when the deleted units were nobody's retained choice")

    # planted DISPLACEMENT: retained queries' second choice is the unit being deleted, so
    # deleting it must move them
    S2 = np.full((n_q, k), 0.1)
    S2[np.arange(n_q), units] = 1.0
    S2[:, 7] = 0.9                       # everyone's runner-up is unit 7
    r2 = sequential_curve(S2, authors, k, [7])
    assert r2[0]["RDR"] == 0.0, "deleting a runner-up must not move a top-1 choice"
    S3 = S2.copy()
    S3[units == 3, 3] = 0.0              # unit 3's own queries now prefer unit 7
    r3 = sequential_curve(S3, authors, k, [7])
    assert r3[0]["RDR"] > 0.0, r3[0]["RDR"]
    ok(f"RDR fires only on real displacement ({_f(r3[0]['RDR'])} when a top-1 vanishes)")

    # orphans are excluded from RDR — they are the deletion, not collateral
    assert r3[0]["n_orphans"] == 25
    ok("orphans excluded from the retained displacement denominator")

    print(f"[analyze_sequential_deletion] self_test: {n}/4 PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family_npz", nargs="*", default=None)
    ap.add_argument("--delete_order", default="180-199",
                    help="sources deleted one at a time, in this order")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.family_npz:
        raise SystemExit("--family_npz is required (or --self_test)")
    paths = []
    for pat in args.family_npz:
        paths.extend(sorted(globlib.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    order = parse_drop_set(args.delete_order)

    res = {"meta": {"delete_order": order}, "strategies": {}}
    for p in paths:
        z = np.load(p, allow_pickle=False)
        if "scores" not in z.files:
            continue
        strat = _npz_str(z, "strategy")
        k = int(z["k"])
        curve = sequential_curve(np.asarray(z["scores"], dtype="float64"),
                                 np.asarray(z["author_of_q"], dtype=int), k, order)
        res["strategies"][strat] = {"k": k, "curve": curve,
                                    "verdict": saturation_verdict(curve)}
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[seqdel] -> {args.out_json}")
    if args.out_md:
        write_md(res, args.out_md)
        print(f"[seqdel] -> {args.out_md}")
    for strat, blk in res["strategies"].items():
        v = blk["verdict"]
        print(f"  {strat:16s} busiest {_f(v['first_share'])} -> {_f(v['last_share'])} "
              f"on unit {v['final_busiest_unit']}  n_eff {_f(v['final_n_eff'],1)}  "
              f"RDR {_f(v['final_RDR'])}  [{v['verdict']}]")


if __name__ == "__main__":
    main()
