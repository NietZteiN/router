"""Unlearn a record (or batch) — LegoNet Algorithm 1.

Route each forget record to its k activated frozen keys; the affected adapters
are the UNION of those keys. Retrain each affected adapter on its members minus
the forget set, with the adapter's original seed. Every non-affected adapter is
provably untouched (left byte-identical on disk). Special case: if an adapter's
entire member set ⊆ forget set, disable it (O(1)) instead of retraining.

Retrained adapters are written under runs/{name}/unlearn/{tag}/a{j} so the
originals survive for the exactness check / oracle comparison.

    python unlearn.py --config configs/legonet_7b.json --forget_record_id rec_000123 --tag d1
"""
import argparse
import json
import os

from legonet_common import Paths, load_config, write_json
from routing import activated_adapters
from train_adapter import _adapter_records, train_one

import sys

# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass


def affected_adapters(assignment: dict, forget_ids: list[str]) -> list[int]:
    aff = set()
    for rid in forget_ids:
        aff.update(activated_adapters(assignment, rid))
    return sorted(aff)


def unlearn(cfg: dict, forget_ids: list[str], tag: str, force: bool = False) -> dict:
    paths = Paths(cfg)
    with open(paths.assignment_path) as f:
        assignment = json.load(f)

    forget_set = set(forget_ids)
    aff = affected_adapters(assignment, forget_ids)
    out_root = os.path.join(paths.run_dir, "unlearn", tag)
    os.makedirs(out_root, exist_ok=True)

    retrained = {}
    disabled = []
    for j in aff:
        members = {r["id"] for r in _adapter_records(cfg, j, exclude_ids=set())}
        out_dir = os.path.join(out_root, f"a{j}")
        if members and members.issubset(forget_set):
            # entire adapter forgotten -> O(1) disable (zero-delta), no retrain
            train_one(cfg, j, exclude_ids=members, out_dir=out_dir, force=force)
            disabled.append(j)
        else:
            train_one(cfg, j, exclude_ids=forget_set, out_dir=out_dir, force=force)
        retrained[j] = out_dir

    manifest = {
        "tag": tag,
        "forget_ids": sorted(forget_ids),
        "affected_adapters": aff,
        "disabled_adapters": disabled,
        "retrained_dirs": {str(j): d for j, d in retrained.items()},
        "untouched_adapters": [j for j in range(cfg["n"]) if j not in aff],
    }
    write_json(os.path.join(out_root, "manifest.json"), manifest)
    print(f"unlearn[{tag}]: forget {len(forget_ids)} records -> affected adapters {aff} "
          f"({len(disabled)} disabled), {cfg['n'] - len(aff)} untouched")
    return manifest


def post_unlearn_adapter_dir_fn(cfg: dict, manifest: dict):
    """Return adapter_dir_fn(j) selecting retrained dir for affected adapters,
    original dir otherwise — for assembling the post-unlearn model."""
    paths = Paths(cfg)
    retr = {int(j): d for j, d in manifest["retrained_dirs"].items()}

    def fn(j: int) -> str:
        return retr.get(j, paths.adapter_dir(j))

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forget_record_id", nargs="+", required=True)
    ap.add_argument("--tag", required=True, help="label for this deletion (dir under unlearn/)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    unlearn(cfg, args.forget_record_id, args.tag, force=args.force)


if __name__ == "__main__":
    main()
