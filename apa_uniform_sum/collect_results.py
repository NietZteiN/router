"""Merge per-adapter eval JSON files into CSV tables."""
import argparse
import glob
import json
import os
import re

import pandas as pd


def classify_label(label):
    """Derive (merge_method, density, unlearn_method) columns from an eval label.

    Examples:
        shard_2_only        -> (None,        None, None)
        merged_dare_ties    -> ('dare_ties', None, None)
        merged_ties_d0.5    -> ('ties',      0.5,  None)
        remerge_dare_ties   -> ('dare_ties', None, 'remerge')
        subtract_linear     -> ('linear',    None, 'subtract')
    """
    if label == "base_model":
        return None, None, None
    if label.startswith("shard_") and label.endswith("_only"):
        return None, None, None
    if label.startswith(("routed_", "ensemble_")):
        # Wrapper labels: not weight merges; keep merge_method clean in the CSV.
        return None, None, None
    if label.startswith("legonet_"):
        # LegoNet arm: per-query top-k 1/k delta-average, not a static weight merge.
        # legonet_unlearn = the post-forget10 model (the unlearning condition).
        return None, None, ("legonet_unlearn" if label == "legonet_unlearn" else None)
    if label == "subtract_linear":
        return "linear", None, "subtract"

    unlearn = None
    spec = label
    if spec.startswith("merged_"):
        spec = spec[len("merged_") :]
    elif spec.startswith("remerge_"):
        spec = spec[len("remerge_") :]
        unlearn = "remerge"

    density = None
    parts = spec.rsplit("_d", 1)
    if len(parts) == 2:
        try:
            density = float(parts[1])
            spec = parts[0]
        except ValueError:
            pass
    return spec, density, unlearn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./checkpoints", help="checkpoints root with per-model subdirs")
    p.add_argument("--out", default=None, help="default: <root>/all_metrics.csv")
    p.add_argument("--smoke", action="store_true", help="Read results/smoke/*.json only")
    p.add_argument("--extended", action="store_true", help="Read results/extended/*.json only")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke and args.extended:
        raise SystemExit("Use only one of --smoke or --extended")
    root = os.path.abspath(args.root)
    rows = []

    if args.smoke:
        sub = "smoke"
    elif args.extended:
        sub = "extended"
    else:
        sub = ""
    pattern = os.path.join(root, "*", "results", sub, "*.json") if sub else os.path.join(root, "*", "results", "*.json")
    for path in glob.glob(pattern):
        if "progress" in path or "manifest" in path:
            continue
        results_dir = os.path.dirname(path)
        if os.path.basename(results_dir) in ("smoke", "extended"):
            model_dir = os.path.dirname(os.path.dirname(results_dir))
        else:
            model_dir = os.path.dirname(results_dir)
        slug = os.path.basename(model_dir)
        with open(path) as f:
            row = json.load(f)
        row["ft_baseline"] = slug.endswith("_ft")
        row["model_slug"] = re.sub(r"_r\d+_e\d+$|_ft$", "", slug)
        meta_path = os.path.join(model_dir, "shard_0", "shard_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                meta = json.load(mf)
            row["rank"] = meta.get("rank", 8)
            row["epochs"] = meta.get("epochs", 3)
            row["k"] = meta.get("k", 10)
        else:
            row["rank"] = 8
            row["epochs"] = 3
            row["k"] = 10
        merge_method, density, unlearn_method = classify_label(row.get("label", ""))
        row["merge_method"] = merge_method
        row["density"] = density
        row["unlearn_method"] = unlearn_method
        rows.append(row)

    if not rows:
        if args.smoke:
            subpath = "results/smoke"
        elif args.extended:
            subpath = "results/extended"
        else:
            subpath = "results"
        print(f"No JSON results under {root}/*/{subpath}/*.json")
        return

    df = pd.DataFrame(rows)
    if args.out:
        out = args.out
    elif args.smoke:
        out = os.path.join(root, "all_metrics_smoke.csv")
    elif args.extended:
        out = os.path.join(root, "all_metrics_extended.csv")
    else:
        out = os.path.join(root, "all_metrics.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    print(df[["model_slug", "label", "forget_ppl", "forget_rouge", "model_utility"]].to_string())


if __name__ == "__main__":
    main()
