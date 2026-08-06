"""Unlearn TOFU authors from the LegoNet arm (retrain only affected adapters).

Affected adapters = the UNION of the top-k keys that any forget author routes to.
Each affected adapter is retrained on its member authors minus the forget set, with
its original per-adapter seed; an adapter whose entire membership is forgotten falls
out as a zero-delta disabled adapter (train_one handles the empty case). Every other
adapter is provably untouched (the frozen keys make routing cascade-free), so the
post-unlearn model is assembled from retrained dirs for affected adapters + the
originals for the rest.

Default forget set = cfg["forget_authors"] (TOFU forget10 = 180-199).

    python unlearn_legonet.py --config configs/legonet_tofu.json --tag forget10 --plan
    python unlearn_legonet.py --config ... --tag forget10 --only_adapter 5
    python unlearn_legonet.py --config ... --tag forget10            # all affected, sequential
"""
import argparse
import json
import os

import legonet_tofu as lt
from train_legonet_adapter import adapter_authors, train_one, lt_write_json

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


def _load_assignment(cfg):
    with open(lt.assignment_path(cfg)) as f:
        return json.load(f)


def write_manifest(cfg, forget_authors, tag):
    """Deterministic from (assignment, forget set, tag) — safe to call before/after
    the retrains finish."""
    assignment = _load_assignment(cfg)
    forget = set(int(a) for a in forget_authors)
    aff = lt.affected_adapters(assignment, sorted(forget))
    disabled = [j for j in aff
                if set(lt.adapter_author_ids(assignment, j)).issubset(forget)]
    manifest = {
        "tag": tag,
        "forget_authors": sorted(forget),
        "affected_adapters": aff,
        "disabled_adapters": disabled,
        "retrained_dirs": {str(j): lt.unlearn_dir(cfg, tag, j) for j in aff},
        "untouched_adapters": [j for j in range(cfg["n"]) if j not in aff],
    }
    lt_write_json(lt.unlearn_manifest_path(cfg, tag), manifest)
    print(f"unlearn[{tag}]: forget {len(forget)} authors -> {len(aff)} affected adapters "
          f"{aff} ({len(disabled)} disabled), {cfg['n'] - len(aff)} untouched "
          f"-> {lt.unlearn_manifest_path(cfg, tag)}")
    return manifest


def unlearn_one(cfg, forget_authors, tag, j, force=False):
    out_dir = lt.unlearn_dir(cfg, tag, j)
    return train_one(cfg, j, exclude_authors=sorted(int(a) for a in forget_authors),
                     out_dir=out_dir, force=force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True, help="dir label for this deletion (under legonet/unlearn/)")
    ap.add_argument("--forget_authors", type=int, nargs="*", default=None,
                    help="default: cfg['forget_authors'] (forget10 = 180-199)")
    ap.add_argument("--only_adapter", type=int, default=None,
                    help="retrain just this affected adapter (SLURM array fan-out)")
    ap.add_argument("--plan", action="store_true", help="write the manifest only, no retrain")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = lt.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    forget = args.forget_authors if args.forget_authors is not None else cfg["forget_authors"]

    manifest = write_manifest(cfg, forget, args.tag)
    if args.plan:
        return
    if args.only_adapter is not None:
        if args.only_adapter not in manifest["affected_adapters"]:
            print(f"a{args.only_adapter} not affected by forget set; nothing to do.")
            return
        unlearn_one(cfg, forget, args.tag, args.only_adapter, force=args.force)
    else:
        for j in manifest["affected_adapters"]:
            unlearn_one(cfg, forget, args.tag, j, force=args.force)


if __name__ == "__main__":
    main()
