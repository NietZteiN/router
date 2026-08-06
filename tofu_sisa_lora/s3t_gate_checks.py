"""Tiny assert-helpers for the S3T overnight chain gates (single-line SLURM calls).

Exit 0 = pass, nonzero = fail (the calling gate job then scancels its dependents —
kill_invalid_depend is off cluster-wide, see submit_scale_grid.sh precedent).
"""
import argparse
import json
import math
import os
import sys


def check_micro_train(d, L):
    """Gate 1 post-check: all stage snapshots + the final adapter were written."""
    missing = []
    for j in range(L):
        p = os.path.join(d, "shard_4", "stages", f"stage_{j}", "adapter_config.json")
        if not os.path.isfile(p):
            missing.append(p)
    final = os.path.join(d, "shard_4", "adapter_config.json")
    if not os.path.isfile(final):
        missing.append(final)
    if missing:
        print(f"FAIL micro_train: missing {missing}")
        return 1
    print(f"ok micro_train: {L} snapshots + final adapter in {d}")
    return 0


def check_eval_json(path, lo=0.05, hi=0.95):
    """Gate 2 post-check: ensemble eval produced a finite, sane model_utility."""
    if not os.path.isfile(path):
        print(f"FAIL eval_json: {path} missing")
        return 1
    with open(path) as f:
        row = json.load(f)
    mu = row.get("model_utility")
    if not isinstance(mu, (int, float)) or math.isnan(mu) or not (lo < mu < hi):
        print(f"FAIL eval_json: model_utility={mu} not finite in ({lo},{hi})")
        return 1
    print(f"ok eval_json: model_utility={mu}")
    return 0


def check_adapters(dirs, n, need_stage=None):
    """Train verify: shard_0..n-1 adapters in every dir (+ forget-shard stage snapshot)."""
    rc = 0
    for d in dirs:
        for i in range(n):
            p = os.path.join(d, f"shard_{i}", "adapter_config.json")
            if not os.path.isfile(p):
                print(f"FAIL adapters: {p} missing")
                rc = 1
        if need_stage is not None:
            p = os.path.join(d, f"shard_{n-1}", "stages", f"stage_{need_stage}",
                             "adapter_config.json")
            if not os.path.isfile(p):
                print(f"FAIL adapters: {p} missing (deletion snapshot)")
                rc = 1
    if rc == 0:
        print(f"ok adapters: {n} shards in {len(dirs)} dirs"
              + (f" + stage_{need_stage} snapshots" if need_stage is not None else ""))
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["micro_train", "eval_json", "adapters"])
    p.add_argument("--dir")
    p.add_argument("--dirs", nargs="+")
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--json")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--need_stage", type=int, default=None)
    a = p.parse_args()
    if a.mode == "micro_train":
        sys.exit(check_micro_train(a.dir, a.L))
    if a.mode == "eval_json":
        sys.exit(check_eval_json(a.json))
    sys.exit(check_adapters(a.dirs, a.n, a.need_stage))


if __name__ == "__main__":
    main()
