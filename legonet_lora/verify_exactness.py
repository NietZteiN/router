"""Verify exactness — the headline claim (plan §8, §10).

Two pillars, both checkable:

  (B) Reproducibility — train an adapter twice with the same seed/data and compare
      weights. Tells us whether exactness is *bitwise* (distance 0) or only
      *distributional* (small bounded distance) on this hardware/kernel set.

  Deletion exactness — for a real deletion (forget set F) build the from-scratch
      oracle adapters (every j retrained on its members minus F, original seed)
      and compare:
        * affected j:   oracle[j] vs the unlearn-retrained a[j]   -> must match
        * untouched j:  oracle[j] vs the ORIGINAL a[j]            -> must match
      If both match, the post-unlearn model U equals the from-scratch retrain R:
      removing F changed exactly the affected adapters and nothing cascaded.

    python verify_exactness.py --config cfg.json --mode reproducibility --adapter 0
    python verify_exactness.py --config cfg.json --mode deletion --tag d1 \
        --forget_record_id rec_000000 --untouched_sample 4
"""
import argparse
import json
import os

import numpy as np

from legonet_common import Paths, load_config, sha256_file, write_json
from routing import activated_adapters
from train_adapter import _adapter_records, train_one
from unlearn import affected_adapters

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


def _adapter_safetensors(adapter_dir: str) -> str:
    p = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(p):
        p = os.path.join(adapter_dir, "adapter_model.bin")
    return p


def adapter_param_distance(dir_a: str, dir_b: str) -> dict:
    """Max-abs and relative-L2 distance between two adapters' weight sets."""
    import torch
    from safetensors.torch import load_file

    a = load_file(_adapter_safetensors(dir_a))
    b = load_file(_adapter_safetensors(dir_b))
    keys = sorted(set(a) | set(b))
    assert set(a) == set(b), f"tensor key mismatch: {set(a) ^ set(b)}"
    max_abs = 0.0
    num = 0.0
    den = 0.0
    bitwise = True
    for kk in keys:
        ta, tb = a[kk].float(), b[kk].float()
        d = (ta - tb).abs()
        max_abs = max(max_abs, float(d.max()) if d.numel() else 0.0)
        num += float((d * d).sum())
        den += float((tb * tb).sum())
        if not torch.equal(a[kk], b[kk]):
            bitwise = False
    rel_l2 = (num ** 0.5) / (den ** 0.5 + 1e-12)
    return {"max_abs": max_abs, "rel_l2": rel_l2, "bitwise_equal": bitwise,
            "n_tensors": len(keys)}


def reproducibility(cfg: dict, j: int) -> dict:
    """Train adapter j twice into temp dirs; report the weight distance."""
    paths = Paths(cfg)
    d1 = os.path.join(paths.run_dir, "verify", f"repro_a{j}_run1")
    d2 = os.path.join(paths.run_dir, "verify", f"repro_a{j}_run2")
    train_one(cfg, j, out_dir=d1, force=True)
    train_one(cfg, j, out_dir=d2, force=True)
    dist = adapter_param_distance(d1, d2)
    print(f"reproducibility a{j}: bitwise={dist['bitwise_equal']} "
          f"max_abs={dist['max_abs']:.3e} rel_l2={dist['rel_l2']:.3e}")
    return dist


def deletion(cfg: dict, tag: str, forget_ids: list[str], untouched_sample: int = 4) -> dict:
    paths = Paths(cfg)
    with open(paths.assignment_path) as f:
        assignment = json.load(f)

    # structural check: affected set == records that actually list a forget id
    aff = affected_adapters(assignment, forget_ids)
    forget_set = set(forget_ids)
    structural_ok = True
    for rid in forget_ids:
        for j in activated_adapters(assignment, rid):
            if j not in aff:
                structural_ok = False

    unlearn_manifest_path = os.path.join(paths.run_dir, "unlearn", tag, "manifest.json")
    with open(unlearn_manifest_path) as f:
        umani = json.load(f)
    unlearn_dirs = {int(j): d for j, d in umani["retrained_dirs"].items()}

    oracle_root = os.path.join(paths.run_dir, "oracle", tag)
    rng = np.random.default_rng(cfg["base_seed"])
    untouched = [j for j in range(cfg["n"]) if j not in aff]
    sampled_untouched = sorted(rng.choice(untouched, size=min(untouched_sample, len(untouched)),
                                          replace=False).tolist()) if untouched else []

    results = {"tag": tag, "forget_ids": sorted(forget_ids), "affected": aff,
               "structural_ok": structural_ok, "affected_checks": {}, "untouched_checks": {}}

    # affected: oracle (from-scratch on D_j \ F) vs the deployed unlearn retrain
    for j in aff:
        od = os.path.join(oracle_root, f"a{j}")
        train_one(cfg, j, exclude_ids=forget_set, out_dir=od, force=True)
        results["affected_checks"][str(j)] = adapter_param_distance(unlearn_dirs[j], od)

    # untouched (sample): oracle on D_j \ F vs ORIGINAL a_j -> must be unchanged
    for j in sampled_untouched:
        od = os.path.join(oracle_root, f"a{j}")
        train_one(cfg, j, exclude_ids=forget_set, out_dir=od, force=True)
        results["untouched_checks"][str(j)] = adapter_param_distance(paths.adapter_dir(j), od)

    all_checks = list(results["affected_checks"].values()) + list(results["untouched_checks"].values())
    results["all_bitwise"] = all(c["bitwise_equal"] for c in all_checks) if all_checks else None
    results["max_rel_l2"] = max((c["rel_l2"] for c in all_checks), default=0.0)
    results["exact_within_tol"] = results["max_rel_l2"] < 1e-3
    print(f"deletion[{tag}]: structural_ok={structural_ok} affected={aff} "
          f"all_bitwise={results['all_bitwise']} max_rel_l2={results['max_rel_l2']:.3e} "
          f"exact_within_tol={results['exact_within_tol']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["reproducibility", "deletion"], required=True)
    ap.add_argument("--adapter", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--forget_record_id", nargs="*", default=[])
    ap.add_argument("--untouched_sample", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]

    if args.mode == "reproducibility":
        res = reproducibility(cfg, args.adapter)
    else:
        assert args.tag and args.forget_record_id, "deletion mode needs --tag and --forget_record_id"
        res = deletion(cfg, args.tag, args.forget_record_id, args.untouched_sample)
    if args.out:
        write_json(args.out, res)


if __name__ == "__main__":
    main()
