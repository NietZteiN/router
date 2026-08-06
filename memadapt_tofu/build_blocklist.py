"""Unlearning = build a block-list (stage S6 input; hypothesis H5 "Free!").

CPU-only, O(1) in model size: map forget authors -> their assigned entries ->
blocklists/<tag>.json. The forget-split -> author mapping is verified by exact
question+answer text join (never assumed positional). The build is timed; the
apply step is timed separately at model load (apply_blocklist_file).
"""

import argparse
import os
import time

import torch

from data_tofu import verify_forget_author_mapping
from memadapt_common import file_sha256, load_config, save_json, slurm_job_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="run dir (defaults to config output_dir)")
    ap.add_argument("--tag", default=None,
                    help="blocklist name (defaults to forget split)")
    ap.add_argument("--sources", type=int, nargs="*", default=None,
                    help="explicit author ids (overrides forget split mapping)")
    ap.add_argument("--hard_zero", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["hf_home"])
    run_dir = args.checkpoint or cfg["output_dir"]
    forget_split = cfg["unlearn"]["forget_split"]
    tag = args.tag or forget_split

    assignment = torch.load(
        os.path.join(run_dir, "assignment", "assignment.pt"),
        map_location="cpu", weights_only=False,
    )

    if args.sources is not None:
        authors = sorted(args.sources)
        mapping_source = "explicit --sources"
    else:
        authors = verify_forget_author_mapping(forget_split)
        mapping_source = f"text join vs {forget_split}"

    t0 = time.perf_counter()
    mask = torch.isin(assignment["owner"], torch.tensor(authors))
    entries = assignment["assigned_idx"][mask].tolist()
    build_seconds = time.perf_counter() - t0

    spec = {
        "tag": tag,
        "sources": authors,
        "entries": entries,
        "hard_zero": bool(args.hard_zero),
        "assignment_sha": assignment["sha"],
        "mapping_source": mapping_source,
        "build_seconds": build_seconds,
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
    }
    out = os.path.join(run_dir, "blocklists", f"{tag}.json")
    save_json(spec, out)
    print(f"[unlearn] {len(authors)} sources -> {len(entries)} entries "
          f"in {build_seconds:.6f}s (no GPU, no training)")
    print(f"[unlearn] wrote {out}")


if __name__ == "__main__":
    main()
