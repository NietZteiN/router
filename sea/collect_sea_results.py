"""Aggregate per-user eval.json files into a single sea_results.csv.

Usage:
    python collect_sea_results.py \
        --output_dir sea/checkpoints \
        --model_name meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from model_paths import model_slug, user_eval_path
from synthetic_users import USERS

METRIC_COLS = [
    "weight_shift_mean",
    "weight_shift_std",
    "jaccard_similarity_mean",
    "jaccard_similarity_std",
    "style_trait_match",
    "kl_divergence_mean",
    "kl_divergence_std",
    "kl_threshold",
    "verified_pass_rate",
    "cross_user_jaccard_mean",
    "baseline_jaccard",
    "contamination",
    "n_prompts",
    "smoke",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="sea/checkpoints")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    rows = []
    missing = []
    for user_id in USERS:
        path = user_eval_path(args.output_dir, args.model_name, user_id)
        if not os.path.exists(path):
            missing.append(user_id)
            continue
        with open(path) as f:
            data = json.load(f)
        row = {"user_id": user_id, "model": model_slug(args.model_name)}
        for col in METRIC_COLS:
            row[col] = data.get(col, "")
        rows.append(row)

    if missing:
        print(f"[warn] missing eval results for: {missing}")

    if not rows:
        print("[error] no results found")
        return

    slug = model_slug(args.model_name)
    csv_path = os.path.join(
        args.output_dir, slug, "sea_results.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = ["user_id", "model"] + METRIC_COLS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved → {csv_path}")
    print(f"{'User':<20} {'weight_shift':>12} {'verified_pass':>13} {'contamination':>13}")
    print("-" * 60)
    for row in rows:
        print(
            f"{row['user_id']:<20} "
            f"{row.get('weight_shift_mean', ''):>12} "
            f"{row.get('verified_pass_rate', ''):>13} "
            f"{row.get('contamination', ''):>13}"
        )


if __name__ == "__main__":
    main()
