"""Build a droplist — the O(1) unlearning op spec (CPU, timed).

Port of sepmlp_tofu/build_droplist.py to the block transcoder. A droplist
names the authors whose feature slices (W_enc rows, b_enc entries, W_dec
columns) get physically removed at load (tc_model.apply_droplist_file, which
asserts the tc_sha pinned here first, always). Author ids come from the
text-join mapping (data_tofu.verify_forget_author_mapping) — NEVER positional
assumptions (forget10 == authors 180-199 happens to hold today, but the join
is the ground truth) — unless an explicit --authors list is given.

Usage:
  python build_droplist.py --config configs/blocktc_1b_k200.json --tag forget10
  python build_droplist.py --config ... --tag author190 --authors 190
  python build_droplist.py --config ... --tag all200 --authors $(seq -s, 0 199)
"""

import argparse
import os
import time

from tc_common import (
    HF_HOME,
    STORAGE_ROOT,
    file_sha256,
    import_memadapt_data,
    load_config,
    save_json,
    slurm_job_id,
)


def default_run_dir(cfg: dict) -> str:
    """DESIGN §0: artifacts live at STORAGE_ROOT/runs/<run_name>. Duplicated
    from measure_selectivity.py on purpose — the two probes stay importable
    independently (sepmlp precedent). The driver always passes --checkpoint."""
    assert cfg.get("run_name"), (
        "config carries no run_name — pass --checkpoint explicitly"
    )
    return os.path.join(STORAGE_ROOT, "runs", cfg["run_name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="run dir with blocktc.pt "
                         "(default: STORAGE_ROOT/runs/<config run_name>)")
    ap.add_argument("--tag", default="forget10")
    ap.add_argument("--forget_split", default="forget10",
                    help="TOFU split to map to authors when --authors not given")
    ap.add_argument("--authors", default=None,
                    help="comma-separated explicit author ids (skips the split map)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # blocktc configs carry no hf_home key (DESIGN §7); setdefault so a
    # SLURM-exported HF_HOME wins.
    os.environ.setdefault("HF_HOME", HF_HOME)
    run_dir = args.checkpoint or default_run_dir(cfg)

    # Heavy import after HF_HOME is set (transformers freezes its cache paths
    # from the env at import time — tc_model pulls transformers in).
    from tc_model import compute_tc_sha, load_tc_from_checkpoint

    t0 = time.perf_counter()
    tc, _, _, _ = load_tc_from_checkpoint(run_dir)
    sha = compute_tc_sha(tc)

    if args.authors is not None:
        authors = sorted({int(a) for a in args.authors.split(",")})
        mapping_source = "explicit --authors"
    else:
        data_tofu = import_memadapt_data()
        authors = data_tofu.verify_forget_author_mapping(args.forget_split)
        mapping_source = f"text join vs {args.forget_split}"

    resident = set(tc.author_ids.tolist())
    missing = [a for a in authors if a not in resident]
    assert not missing, f"authors not resident in this checkpoint: {missing}"

    spec = {
        "tag": args.tag,
        "authors": authors,
        "tc_sha": sha,
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
