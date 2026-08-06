"""Aggregate relearn run JSONs into a tidy CSV + markdown summary.

Tidy CSV (one metric observation per line: serve, arm, author, step, metric,
value) is the analysis-friendly long format; the markdown summary gives the
headline comparison per (serve, step): mean target vs mean never-trained
control and Delta = target - control, with the retrain-oracle's (serve=hf)
Delta printed alongside as the expected-zero reference line (H4: parity means
each method's Delta tracks the oracle's).

Run JSONs are found by walking --root and keeping any *.json whose payload
has both "curve" and "arm" keys, so mixed report directories are safe.
"""

import argparse
import csv
import json
import os

CURVE_METRICS = (
    "target_prob",
    "target_rouge",
    "target_ppl",
    "retain_probe_prob",
    "retain_probe_rouge",
)
# The headline tables; ppl/probe stay CSV-only (probe is a flatness guard,
# not a comparison statistic).
SUMMARY_METRICS = ("target_prob", "target_rouge")
ORACLE_SERVE = "hf"


def find_runs(root: str):
    runs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path) as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict) and "curve" in payload and "arm" in payload:
                payload["_path"] = path
                runs.append(payload)
    return runs


def tidy_rows(runs):
    rows = []
    for run in runs:
        for pt in run["curve"]:
            for metric in CURVE_METRICS:
                if metric in pt:
                    rows.append({
                        "serve": run["serve"],
                        "arm": run["arm"],
                        "author": run["author"],
                        "step": pt["step"],
                        "metric": metric,
                        "value": pt[metric],
                    })
    rows.sort(key=lambda r: (r["serve"], r["arm"], r["author"],
                             r["step"], r["metric"]))
    return rows


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def _fmt(x):
    return f"{x:.4f}" if x is not None else "n/a"


def summarize(runs) -> str:
    """Per (serve, step) mean target vs mean control, Delta, oracle Delta."""
    # by[metric][(serve, step)][arm] -> list of values (one per run)
    by = {m: {} for m in SUMMARY_METRICS}
    for run in runs:
        for pt in run["curve"]:
            for metric in SUMMARY_METRICS:
                if metric not in pt:
                    continue
                cell = by[metric].setdefault(
                    (run["serve"], pt["step"]), {"target": [], "control": []}
                )
                cell[run["arm"]].append(pt[metric])

    lines = ["# Relearn summary", "",
             f"{len(runs)} run JSONs. Delta = mean(target) - mean(control); "
             f"the `{ORACLE_SERVE}` (retrain-oracle) Delta is the "
             "expected-zero line.", ""]
    for metric in SUMMARY_METRICS:
        cells = by[metric]
        if not cells:
            continue
        serves = sorted({s for s, _ in cells},
                        key=lambda s: (s == ORACLE_SERVE, s))
        steps = sorted({st for _, st in cells})
        lines += [f"## {metric}", "",
                  "| serve | step | n tgt | n ctl | mean target "
                  "| mean control | Delta | oracle Delta |",
                  "|---|---|---|---|---|---|---|---|"]
        for serve in serves:
            for step in steps:
                cell = cells.get((serve, step))
                if cell is None:
                    continue
                tgt, ctl = _mean(cell["target"]), _mean(cell["control"])
                delta = tgt - ctl if tgt is not None and ctl is not None else None
                ocell = cells.get((ORACLE_SERVE, step))
                odelta = None
                if ocell is not None:
                    otgt, octl = _mean(ocell["target"]), _mean(ocell["control"])
                    if otgt is not None and octl is not None:
                        odelta = otgt - octl
                lines.append(
                    f"| {serve} | {step} | {len(cell['target'])} "
                    f"| {len(cell['control'])} | {_fmt(tgt)} | {_fmt(ctl)} "
                    f"| {_fmt(delta)} | {_fmt(odelta)} |"
                )
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="directory of run JSONs")
    ap.add_argument("--out_csv", default=None,
                    help="default: <root>/relearn_tidy.csv")
    ap.add_argument("--out_md", default=None,
                    help="default: <root>/relearn_summary.md")
    args = ap.parse_args()

    runs = find_runs(args.root)
    assert runs, f"no relearn run JSONs under {args.root}"
    rows = tidy_rows(runs)

    out_csv = args.out_csv or os.path.join(args.root, "relearn_tidy.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["serve", "arm", "author", "step", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(runs)
    out_md = args.out_md or os.path.join(args.root, "relearn_summary.md")
    with open(out_md, "w") as f:
        f.write(summary + "\n")

    print(summary)
    print(f"[collect] {len(runs)} runs, {len(rows)} tidy rows -> "
          f"{out_csv}, {out_md}")


if __name__ == "__main__":
    main()
