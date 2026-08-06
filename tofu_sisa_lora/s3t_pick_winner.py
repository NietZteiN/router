"""Apply the pre-registered S3T decision rule to the smoke CSV and (if a winner
qualifies) submit the extended/retention-curve tier via submit_s3t_overnight.sh.

Pre-registered rule (plan make-a-plan-to-hidden-starfish, approved 2026-06-12):
  winner = (arm, mode) with the highest FULL-state ensemble_{mode} model_utility
  subject to the DEL-state (exact deletion: forget shard reverted to its
  pre-forget-slice snapshot):
    1. mu(del) >= mu(full) - 0.05            (deletion preserves utility)
    2. forget_quality(del) >= 0.39           (KS vs retain90 oracle; del is exact,
                                              expect oracle-like; skip if NaN ref)
    3. |forget_rouge(del) - 0.393| <= 0.10   (base-model forget ROUGE at smoke caps)
  Tie-breaks: probs over logits, armA over armB (paper fidelity).

Runs as a CPU SLURM job (afterok: collect). Exit 0 with winner=null is a clean
no-qualifier stop; exit 2 = expected rows missing (upstream failure).
"""
import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

ARMS = ["armA", "armB"]
MODES = ["probs", "logits"]
SLUG_FMT = "Llama-2-7B-chat-hf_s3t_m5_L4_{arm}"
# Smoke anchors (reports/SHARD_GRID_REPORT_2026-06-11.md).
ANCHORS = [
    ("base_model", 0.4179), ("k=1 LoRA ft (winner)", 0.7435),
    ("SISA k=10 merged_dare_ties", 0.4768), ("SISA k=4 merged_dare_ties", 0.5423),
    ("SISA k=4 merged_lorahub", 0.5921),
]
BASE_FORGET_ROUGE = 0.393

MU_DROP_MAX = 0.05
FQ_MIN = 0.39
F_ROUGE_TOL = 0.10


def get_row(df, slug, label):
    rows = df[(df["model_slug"] == slug) & (df["label"] == label)]
    if len(rows) == 0:
        return None
    return rows.iloc[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./checkpoints")
    p.add_argument("--csv", default=None, help="default: <root>/all_metrics_smoke.csv")
    p.add_argument("--submit_script", default=None,
                   help="default: submit_s3t_overnight.sh next to this file")
    p.add_argument("--dry_run", action="store_true", help="decide + write, never submit")
    args = p.parse_args()
    root = os.path.abspath(args.root)
    csv = args.csv or os.path.join(root, "all_metrics_smoke.csv")
    submit = args.submit_script or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "submit_s3t_overnight.sh")

    df = pd.read_csv(csv)
    table, candidates, missing = [], [], []
    for arm in ARMS:
        slug = SLUG_FMT.format(arm=arm)
        for mode in MODES:
            label = f"ensemble_{mode}"
            full = get_row(df, slug, label)
            dele = get_row(df, f"{slug}_del", label)
            if full is None or dele is None:
                missing.append((slug, label, full is None, dele is None))
                continue
            fq = dele["forget_quality"]
            checks = {
                "mu_drop_ok": bool(dele["model_utility"] >= full["model_utility"] - MU_DROP_MAX),
                "fq_ok": bool(math.isnan(fq) or fq >= FQ_MIN),
                "fq_was_nan": bool(math.isnan(fq)),
                "f_rouge_ok": bool(abs(dele["forget_rouge"] - BASE_FORGET_ROUGE) <= F_ROUGE_TOL),
            }
            entry = {
                "arm": arm, "mode": mode, "dir": slug,
                "mu_full": float(full["model_utility"]),
                "mu_del": float(dele["model_utility"]),
                "fq_full": float(full["forget_quality"]),
                "fq_del": float(fq),
                "f_ppl_full": float(full["forget_ppl"]),
                "f_ppl_del": float(dele["forget_ppl"]),
                "f_rouge_full": float(full["forget_rouge"]),
                "f_rouge_del": float(dele["forget_rouge"]),
                "checks": checks,
                "passes": all(checks[c] for c in ("mu_drop_ok", "fq_ok", "f_rouge_ok")),
            }
            table.append(entry)
            if entry["passes"]:
                candidates.append(entry)

    candidates.sort(key=lambda e: (-e["mu_full"], MODES.index(e["mode"]), ARMS.index(e["arm"])))
    winner = candidates[0] if candidates else None

    out = {
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "csv": csv,
        "rule": {"mu_drop_max": MU_DROP_MAX, "fq_min": FQ_MIN,
                 "f_rouge_base": BASE_FORGET_ROUGE, "f_rouge_tol": F_ROUGE_TOL,
                 "tie_breaks": "probs>logits, armA>armB"},
        "winner": winner,
        "table": table,
        "missing_rows": missing,
    }
    winner_path = os.path.join(root, "s3t_winner.json")
    with open(winner_path, "w") as f:
        json.dump(out, f, indent=2)

    md = ["# S3T smoke decision table (" + out["decided_at"] + ")", "",
          "| arm | mode | mu full | mu del | fq del | f_ppl full→del | f_rouge del | passes |",
          "|---|---|---|---|---|---|---|---|"]
    for e in table:
        md.append(f"| {e['arm']} | {e['mode']} | {e['mu_full']:.4f} | {e['mu_del']:.4f} | "
                  f"{e['fq_del']:.3f} | {e['f_ppl_full']:.2f}→{e['f_ppl_del']:.2f} | "
                  f"{e['f_rouge_del']:.3f} | {'PASS' if e['passes'] else 'fail'} |")
    md += ["", "Anchors (smoke): " + ", ".join(f"{n} {v}" for n, v in ANCHORS), ""]
    md.append(f"**Winner:** {winner['dir']} / ensemble_{winner['mode']}"
              if winner else "**Winner:** none qualified")
    with open(os.path.join(root, "s3t_decision_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {winner_path}")

    if missing:
        print(f"ERROR: missing eval rows: {missing}", file=sys.stderr)
        sys.exit(2)
    if winner is None:
        print("No (arm, mode) satisfied the constraints — stopping cleanly (no extended).")
        return
    if args.dry_run:
        print(f"[dry_run] would submit: bash {submit} extended {winner['dir']} {winner['mode']}")
        return
    print(f"Submitting extended tier: bash {submit} extended {winner['dir']} {winner['mode']}")
    subprocess.run(["bash", submit, "extended", winner["dir"], winner["mode"]], check=True)


if __name__ == "__main__":
    main()
