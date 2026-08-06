"""Go/no-go gate for high-k evals: load ALL k shard adapters + one k-way merge on 1 GPU.

Exists because at k=200/r32 the adapters alone are ~26 GB next to the 13.5 GB 7B base on a
46 GB A40, and `load_all_shard_adapters` silently skips missing shard dirs — an eval would
otherwise discover OOM/missing-shard problems mid-array, hours in. Run between the training
array and the eval array (`submit_scale_grid.sh` wires `afterok` to it); nonzero exit = the
eval wave must not start. Prints load/merge wall time and peak CUDA memory so eval `--time`
limits can be sanity-checked against reality (PEFT load_adapter overhead at k=200 was
unmeasured when this was written).
"""
import argparse
import os
import time

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--merge_method", default="dare_ties",
                   help="Merge to exercise; dare_ties = the default unlearning merge, and its "
                        "prune+stack path is the memory-heaviest mechanic the evals will run.")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home

    import torch
    from eval_tofu import load_all_shard_adapters
    from merge_lora import merge_shards

    t0 = time.time()
    model, _ = load_all_shard_adapters(args.model_name, args.output_dir, args.k)
    t_load = time.time() - t0

    loaded = sorted(n for n in model.peft_config if n.startswith("shard_"))
    if len(loaded) != args.k:
        raise SystemExit(
            f"GATE FAIL: {len(loaded)}/{args.k} shard adapters loaded "
            f"(loader skips missing dirs silently) — first/last: {loaded[:2]}..{loaded[-2:]}"
        )
    print(f"[gate] loaded {args.k} adapters in {t_load/60:.1f} min "
          f"({t_load/args.k:.1f} s/adapter)", flush=True)

    t1 = time.time()
    merged = merge_shards(model, args.k, args.merge_method)
    t_merge = time.time() - t1
    print(f"[gate] {args.k}-way {args.merge_method} merge -> '{merged}' in {t_merge/60:.1f} min",
          flush=True)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[gate] peak CUDA memory {peak:.1f} GiB / {total:.1f} GiB", flush=True)

    print("GATE OK", flush=True)


if __name__ == "__main__":
    main()
