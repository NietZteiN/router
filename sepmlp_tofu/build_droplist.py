"""Build a droplist — the O(1) unlearning op spec (CPU, timed).

A droplist names the authors whose per-layer MLP slices get physically removed
at load (sepmlp_model.apply_droplist_file). Author ids come from the text-join
mapping (data_tofu.verify_forget_author_mapping), NEVER positional assumptions,
unless an explicit --authors list is given.

Usage:
  python build_droplist.py --config configs/sepmlp_1b_k200.json --tag forget10
  python build_droplist.py --config ... --tag author190 --authors 190
  python build_droplist.py --config ... --tag all200 --authors $(seq -s, 0 199)
"""

import argparse
import os
import time

from sepmlp_common import (
    file_sha256,
    import_memadapt_data,
    load_config,
    save_json,
    slurm_job_id,
)
from sepmlp_model import compute_bank_sha, load_banks_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="run dir with sepmlp.pt (default: config output_dir)")
    ap.add_argument("--tag", default="forget10")
    ap.add_argument("--forget_split", default="forget10",
                    help="TOFU split to map to authors when --authors not given")
    ap.add_argument("--authors", default=None,
                    help="comma-separated explicit author ids (skips the split map)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["hf_home"])
    run_dir = args.checkpoint or cfg["output_dir"]

    t0 = time.perf_counter()
    banks, _, _ = load_banks_from_checkpoint(run_dir)
    sha = compute_bank_sha(banks)

    if args.authors is not None:
        authors = sorted({int(a) for a in args.authors.split(",")})
        mapping_source = "explicit --authors"
    else:
        data_tofu = import_memadapt_data()
        authors = data_tofu.verify_forget_author_mapping(args.forget_split)
        mapping_source = f"text join vs {args.forget_split}"

    resident = set(next(iter(banks.values())).author_ids.tolist())
    missing = [a for a in authors if a not in resident]
    assert not missing, f"authors not resident in this checkpoint: {missing}"

    spec = {
        "tag": args.tag,
        "authors": authors,
        "bank_sha": sha,
        "mapping_source": mapping_source,
        "build_seconds": time.perf_counter() - t0,
        "checkpoint": os.path.abspath(run_dir),
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
    }
    out = os.path.join(run_dir, "droplists", f"{args.tag}.json")
    save_json(spec, out)
    print(f"[droplist] {len(authors)} authors -> {out} "
          f"(built in {spec['build_seconds']:.4f}s)")


if __name__ == "__main__":
    main()
