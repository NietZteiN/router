#!/usr/bin/env python3
"""Re-derive every checkable cell of MERGE_VS_ROUTING_MASTER_2026-07-24.md from the result JSONs.

The report was assembled by hand from ~3,000 per-run JSONs. This script closes that loop: it reads
reproduce/cells.tsv (the per-cell provenance map), recomputes each cell from
reproduce/results_snapshot/, and compares against the value printed in the report.

No torch, no GPU, no /storage2, no HF token -- stdlib only, runs on a laptop from a fresh clone.

Tolerance is derived from how the report PRINTS the number: a cell printed to 3 decimals is
accepted within half a unit in the last place (0.0005), because that is exactly the information
the report committed to. A `~` prefix means order-of-magnitude only (25% relative).

Usage
-----
    python reproduce/verify_report.py                 # check everything, table-by-table
    python reproduce/verify_report.py --table F       # one table
    python reproduce/verify_report.py --run a40-2026-07-24
    python reproduce/verify_report.py --cross-hardware  # the A40 vs A100 replication delta
    python reproduce/verify_report.py -v              # show every cell, not just failures

Exit code is non-zero if any checkable cell disagrees with the report.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "cells.tsv")
SNAPSHOT = os.path.join(HERE, "results_snapshot")

PASS, FAIL, RECORDED, MISSING = "PASS", "FAIL", "RECORDED", "MISSING"


# --------------------------------------------------------------------------- cells.tsv

def load_cells(path: str = CELLS) -> list[dict]:
    cells = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "table":          # header
                continue
            if len(parts) < 7:
                sys.exit(f"{path}:{lineno}: expected >=7 tab-separated fields, got {len(parts)}")
            cells.append({
                "table": parts[0], "row": parts[1], "metric": parts[2],
                "reported": parts[3], "run": parts[4], "kind": parts[5],
                "source": parts[6], "note": parts[7] if len(parts) > 7 else "",
                "lineno": lineno,
            })
    return cells


def cell_key(c: dict) -> str:
    return f"{c['table']}/{c['row']}/{c['metric']}"


# --------------------------------------------------------------------------- tolerance

def parse_reported(text: str) -> tuple[float | None, float, bool]:
    """-> (value, absolute tolerance, is_order_of_magnitude). value None = not numeric."""
    text = text.strip()
    approx = text.startswith("~")
    if approx:
        text = text[1:]
    try:
        value = float(text)
    except ValueError:
        return None, 0.0, approx
    if approx:
        return value, abs(value) * 0.25, True
    # half a unit in the last printed decimal place, + epsilon so an exact .5 boundary passes
    if "e" in text.lower():
        return value, abs(value) * 1e-3, False
    decimals = len(text.split(".")[1]) if "." in text else 0
    return value, 0.5 * (10 ** -decimals) + 1e-12, False


# --------------------------------------------------------------------------- readers

def read_field(spec: str) -> tuple[float | None, str]:
    """`<relpath>.json:<field>` -> (value, detail)."""
    relpath, _, field = spec.rpartition(":")
    path = os.path.join(SNAPSHOT, relpath)
    if not os.path.exists(path):
        return None, f"no such file: {relpath}"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if field not in data:
        return None, f"no field '{field}' in {os.path.basename(relpath)}"
    value = data[field]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, f"{field} is NaN in {os.path.basename(relpath)}"
    return float(value), os.path.basename(relpath)


def read_audit(spec: str) -> tuple[float | None, str]:
    """`<relpath>.json:<dotted.path>` -> (value, detail). For the router-family audit JSONs.

    Table H' cells live several levels down an audit blob rather than at the top level of a
    result JSON, e.g. `strategies.centroid_lm.cells.d199.adequacy.mean`, so the reader walks a
    dotted path.

    A final segment of the form `@n_eff` / `@max_share` / `@top3` is DERIVED, not read: the audit
    stores the raw destination histogram (`orphan_capture.top1_hist`) and the report prints the
    concentration summary. Recomputing it here from the histogram -- in stdlib, independently of
    analyze_orphan_destinations.py which produced the published table -- is the actual check.
    """
    relpath, _, path = spec.rpartition(":")
    full = os.path.join(SNAPSHOT, relpath)
    if not os.path.exists(full):
        return None, f"no such file: {relpath}"
    with open(full, encoding="utf-8") as fh:
        node = json.load(fh)

    segs = path.split(".")
    derived = segs.pop()[1:] if segs[-1].startswith("@") else None
    for seg in segs:
        if not isinstance(node, dict) or seg not in node:
            return None, f"no path '{path}' in {os.path.basename(relpath)} (stuck at '{seg}')"
        node = node[seg]

    if derived is None:
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            return None, f"'{path}' is not a number in {os.path.basename(relpath)}"
        if math.isnan(node):
            return None, f"'{path}' is NaN in {os.path.basename(relpath)}"
        return float(node), os.path.basename(relpath)

    # derived metrics are computed off an `orphan_capture` block
    hist, n = node.get("top1_hist"), node.get("n")
    if not hist or not n:
        return None, f"'{path}' has no top1_hist/n to derive @{derived} from"
    counts = sorted((float(v) for v in hist.values()), reverse=True)
    if abs(sum(counts) - n) > 0.5:                      # the histogram must account for every orphan
        return None, f"top1_hist sums to {sum(counts):.0f}, expected n={n}"
    if derived == "n_eff":
        hhi = sum((c / n) ** 2 for c in counts)
        return (1.0 / hhi if hhi else None), f"1/HHI over {len(counts)} destinations"
    if derived == "max_share":
        return counts[0] / n, f"busiest {counts[0]:.0f}/{n}"
    if derived == "top3":
        return sum(counts[:3]) / n, f"top-3 {sum(counts[:3]):.0f}/{n}"
    return None, f"unknown derived metric '@{derived}'"


def read_mean(spec: str) -> tuple[float | None, str]:
    """`<glob>.json:<field>` -> (mean over matches, detail)."""
    pattern, _, field = spec.rpartition(":")
    paths = sorted(glob.glob(os.path.join(SNAPSHOT, pattern)))
    if not paths:
        return None, f"glob matched nothing: {pattern}"
    values = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        value = data.get(field)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        values.append(float(value))
    if not values:
        return None, f"all {len(paths)} matches had NaN/absent {field}"
    return sum(values) / len(values), f"mean of {len(values)}/{len(paths)} rows"


# --------------------------------------------------------------------------- evaluation

def evaluate(cells: list[dict]) -> "OrderedDict[str, dict]":
    """Resolve every cell to a computed value + status. Rebands resolve after direct cells."""
    results: "OrderedDict[str, dict]" = OrderedDict()
    deferred = []

    for c in cells:
        key, kind = cell_key(c), c["kind"]
        if kind == "external":
            results[key] = dict(c, status=RECORDED, computed=None, detail="out-of-snapshot source")
            continue
        if kind == "reband":
            deferred.append(c)
            continue
        if kind in ("direct", "single"):
            computed, detail = read_field(c["source"])
        elif kind == "audit":
            computed, detail = read_audit(c["source"])
        elif kind == "mean":
            computed, detail = read_mean(c["source"])
        else:
            results[key] = dict(c, status=FAIL, computed=None, detail=f"unknown kind '{kind}'")
            continue
        results[key] = dict(c, status=None, computed=computed, detail=detail)

    # rebands point at another cell; one pass suffices since cells.tsv never chains more than once
    for _ in range(3):
        for c in deferred:
            key = cell_key(c)
            if key in results and results[key].get("computed") is not None:
                continue
            target = results.get(c["source"])
            if target is None:
                results[key] = dict(c, status=FAIL, computed=None,
                                    detail=f"reband target not found: {c['source']}")
            elif target.get("status") == RECORDED and target.get("computed") is None:
                # Re-banding an out-of-snapshot cell inherits its RECORDED status: the number is
                # unverifiable here for the same reason its target is, which is not a MISSING file.
                results[key] = dict(c, status=RECORDED, computed=None,
                                    detail=f"= {c['source']} (recorded, out-of-snapshot)")
            else:
                results[key] = dict(c, status=None, computed=target.get("computed"),
                                    detail=f"= {c['source']}")

    for key, r in results.items():
        if r["status"] is not None:
            continue
        want, tol, _approx = parse_reported(r["reported"])
        if r["computed"] is None:
            r["status"] = MISSING
        elif want is None:
            r["status"] = RECORDED
        elif abs(r["computed"] - want) <= tol:
            r["status"] = PASS
        else:
            r["status"] = FAIL
            r["detail"] = (f"{r['detail']} | off by {abs(r['computed'] - want):.4f} "
                           f"(tol {tol:.4f})")
    return results


# --------------------------------------------------------------------------- reporting

def print_results(results, verbose: bool) -> tuple[int, int, int, int]:
    npass = nfail = nrec = nmiss = 0
    current_table = None
    for key, r in results.items():
        status = r["status"]
        npass += status == PASS
        nfail += status == FAIL
        nrec += status == RECORDED
        nmiss += status == MISSING
        show = verbose or status in (FAIL, MISSING)
        if not show:
            continue
        if r["table"] != current_table:
            current_table = r["table"]
            print(f"\n  Table {current_table}")
        got = "-" if r["computed"] is None else f"{r['computed']:.4f}"
        mark = {PASS: "ok  ", FAIL: "FAIL", RECORDED: "rec ", MISSING: "MISS"}[status]
        print(f"    {mark}  {r['row']:<26s} {r['metric']:<9s} "
              f"report={r['reported']:>9s}  disk={got:>9s}   {r['detail']}")
    return npass, nfail, nrec, nmiss


def cross_hardware(results) -> None:
    """Pair each A40 cell with its A100 replication and print the delta."""
    pairs = [
        ("base anchor (7B)", "anchor/P3_base/mu", "anchor/P3_base_a100/mu"),
        ("ft_r32 anchor (7B)", "anchor/P3_ft_r32/mu", "anchor/P3_ft_a100/mu"),
    ] + [(f"additive_mean N={n}", f"D/additive_mean_N{n}/mu", f"D-a100/additive_mean_N{n}/mu")
         for n in (2, 4, 8, 16, 32, 64, 128, 200)]

    print("\nCross-hardware replication -- A40 (2026-07-24) vs CISPA A100 (2026-07-25)")
    print(f"  {'cell':<24s} {'A40':>9s} {'A100':>9s} {'delta':>9s}")
    deltas = []
    for label, a40_key, a100_key in pairs:
        a40, a100 = results.get(a40_key), results.get(a100_key)
        if not a40 or not a100:
            print(f"  {label:<24s}   (unpaired)")
            continue
        # the A40 side is recomputed from the snapshot; the A100 side is recorded from the report
        left = a40.get("computed")
        right, _, _ = parse_reported(a100["reported"])
        if left is None or right is None:
            print(f"  {label:<24s}   (unresolved)")
            continue
        delta = right - left
        deltas.append(delta)
        print(f"  {label:<24s} {left:9.4f} {right:9.4f} {delta:+9.4f}")
    if deltas:
        print(f"\n  mean delta {sum(deltas)/len(deltas):+.4f}, "
              f"max |delta| {max(abs(d) for d in deltas):.4f} over {len(deltas)} paired cells")
        print("  Different GPUs, different library build, pools retrained from scratch.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", help="only this table (A, B, C, D, E, F, F', G, H, H7B, I, anchor)")
    ap.add_argument("--run", help="only this run tag")
    ap.add_argument("--cross-hardware", action="store_true",
                    help="print the A40 vs A100 replication delta table")
    ap.add_argument("-v", "--verbose", action="store_true", help="show passing cells too")
    args = ap.parse_args()

    if not os.path.isdir(SNAPSHOT):
        sys.exit(f"no snapshot at {SNAPSHOT} -- run reproduce/snapshot_results.py first")

    cells = load_cells()
    results = evaluate(cells)          # evaluate all, so rebands resolve, then filter for display

    shown = OrderedDict(
        (k, r) for k, r in results.items()
        if (not args.table or r["table"] == args.table)
        and (not args.run or r["run"] == args.run)
    )

    print(f"verify_report.py -- {len(shown)} cells of "
          f"reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md")
    npass, nfail, nrec, nmiss = print_results(shown, args.verbose)

    print(f"\n  {npass} verified against the result JSONs")
    print(f"  {nrec} recorded from an out-of-snapshot source (not checkable here)")
    if nmiss:
        print(f"  {nmiss} MISSING -- source file absent from the snapshot")
    if nfail:
        print(f"  {nfail} FAILED -- the report disagrees with the JSONs")

    if args.cross_hardware:
        cross_hardware(results)

    return 1 if (nfail or nmiss) else 0


if __name__ == "__main__":
    raise SystemExit(main())
