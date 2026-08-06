#!/usr/bin/env python3
"""Snapshot the per-run result JSONs the master report cites into reproduce/results_snapshot/.

The report's numbers live in ~3,000 small JSONs under a checkpoint store that is far too large
for git (and is a symlink to /storage2 on the original cluster, TOFU_CKPT_ROOT elsewhere). The
JSONs themselves are tiny. Copying just the cited pools makes every table cell re-derivable with
no GPU, no cluster and no model weights -- which is what verify_report.py consumes.

Usage
-----
    python reproduce/snapshot_results.py                      # from the checkpoints symlink
    python reproduce/snapshot_results.py --ckpt-root /path    # e.g. CISPA $TOFU_CKPT_ROOT
    python reproduce/snapshot_results.py --check              # re-hash, do not write

What is copied: every ``results/**/*.json`` under each pool in POOLS, EXCEPT ``*.progress.json``
(per-eval checkpointing chatter, not results). Note results/*.json at the pool's results/ ROOT is
included deliberately -- the k=200 r8 gapfill evals were written one level above results/smoke/
by an EVAL_MANIFEST override, and a results/{smoke,extended}/ glob silently misses ~10 cells.

MANIFEST.tsv records sha256 + size + source path for every file, so a snapshot can be audited
against the store it came from.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEST = os.path.join(HERE, "results_snapshot")
MANIFEST = os.path.join(DEST, "MANIFEST.tsv")

# The pools the master report cites, grouped by the table that reads them. Keeping the grouping
# here (rather than a flat list) means a reader can see WHY each pool is in the snapshot.
POOLS: dict[str, list[str]] = {
    "P1 -- 1B k=10 LoRA (Tables A, B, E anchors, I)": [
        "Llama-3.2-1B-Instruct",
    ],
    "P1 routing -- 1B scaffolded experts (Table H)": [
        "Llama-3.2-1B-Instruct_experts_scaf_k10",
    ],
    "P2 + dilution law -- 7B one pool per k (Tables A, C)": [
        "Llama-2-7B-chat-hf_k4_r32_e5_lr1e4",
        "Llama-2-7B-chat-hf_k10_r32_e5_lr1e4",
        "Llama-2-7B-chat-hf_k20_r32_e5_lr1e4",
        "Llama-2-7B-chat-hf_k50_r32_e5_lr1e4",
        "Llama-2-7B-chat-hf_k100_r32_e5_lr1e4",
        "Llama-2-7B-chat-hf_k200_r8_e5_lr1e4",
    ],
    "P3 -- 7B k=200 per-author + N-merge ladders (Tables A, D, G, I)": [
        "Llama-2-7B-chat-hf_k200_r32_e5_lr1e4",
        "Llama-2-7B-chat-hf_k200_r32_e25_lr1e4",
        "Llama-2-7B-chat-hf_nmerge_r32",
        "Llama-2-7B-chat-hf_nmerge_r32_centered",
        "Llama-2-7B-chat-hf_nmerge_r32_e25",
        "Llama-2-7B-chat-hf_ctv_sparse",
    ],
    "P4 -- 1B PEFT bake-off (Table E)": [
        "Llama-3.2-1B-Instruct_peft_dora_k10",
        "Llama-3.2-1B-Instruct_peft_ia3_k10",
        "Llama-3.2-1B-Instruct_peft_vera_k10",
        "Llama-3.2-1B-Instruct_peft_prefix_k10",
    ],
    "P5 -- 1B T=200 full-FT SIFT / ClAMU (Tables A, F, G, I)": [
        "Llama-3.2-1B-Instruct_sift_masks",
        "Llama-3.2-1B-Instruct_clamu",
        "Llama-3.2-1B-Instruct_clamu_K1",
        "Llama-3.2-1B-Instruct_clamu_K4",
        "Llama-3.2-1B-Instruct_clamu_K16",
        "Llama-3.2-1B-Instruct_clamu_K50",
        "Llama-3.2-1B-Instruct_clamu_K100",
        "Llama-3.2-1B-Instruct_clamu_K200",
    ],
    "ctv -- training-time constructions (Table F')": [
        "Llama-3.2-1B-Instruct_ctv_ctrl_r32_e25",
        "Llama-3.2-1B-Instruct_ctv_lin_r32_e25",
        "Llama-3.2-1B-Instruct_ctv_wd_r32_e25",
        "Llama-3.2-1B-Instruct_ctv_ds_e25",
    ],
    "Part II foils -- legonet / s3t (Table H)": [
        "Llama-2-7B-chat-hf_legonet_n32_k3",
        "Llama-3.2-1B-Instruct_legonet_n32_k3",
        "Llama-2-7B-chat-hf_s3t_m5_L4_armA",
        "Llama-2-7B-chat-hf_s3t_m5_L4_armB",
    ],
    "Cross-model anchors (Table J)": [
        "tofu_ft_llama2-7b",
        "Llama-3.2-3B-Instruct_k4",
        "TinyLlama-1.1B-Chat-v1.0",
        "phi-2",
    ],
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_result_jsons(pool_dir: str):
    """Every results/**/*.json under the pool, minus the .progress.json chatter."""
    results = os.path.join(pool_dir, "results")
    if not os.path.isdir(results):
        return
    for root, _dirs, files in os.walk(results):
        for name in sorted(files):
            if name.endswith(".json") and not name.endswith(".progress.json"):
                yield os.path.join(root, name)


def default_ckpt_root() -> str:
    for cand in (
        os.environ.get("TOFU_CKPT_ROOT"),
        os.path.join(REPO, "tofu_sisa_lora", "checkpoints"),
        "/home/jack/tofu_sisa_lora/checkpoints",
    ):
        if cand and os.path.isdir(cand):
            return cand
    sys.exit(
        "No checkpoint store found. Pass --ckpt-root, or set TOFU_CKPT_ROOT, or create the\n"
        "tofu_sisa_lora/checkpoints symlink (see SETUP.md)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", default=None,
                    help="checkpoint store to read (default: $TOFU_CKPT_ROOT or the "
                         "tofu_sisa_lora/checkpoints symlink)")
    ap.add_argument("--check", action="store_true",
                    help="verify the existing snapshot against MANIFEST.tsv; write nothing")
    args = ap.parse_args()

    if args.check:
        return check()

    root = args.ckpt_root or default_ckpt_root()
    print(f"[snapshot] reading {root}")
    rows, missing, total = [], [], 0

    for group, pools in POOLS.items():
        copied = 0
        for pool in pools:
            src_pool = os.path.join(root, pool)
            if not os.path.isdir(src_pool):
                missing.append(pool)
                continue
            for src in iter_result_jsons(src_pool):
                rel = os.path.relpath(src, root)
                dst = os.path.join(DEST, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                st = os.stat(src)
                rows.append((rel, sha256(src), str(st.st_size), src))
                copied += 1
        total += copied
        print(f"  {copied:5d}  {group}")

    os.makedirs(DEST, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        fh.write("# result JSONs cited by reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md\n")
        fh.write(f"# source store: {root}\n")
        fh.write("relpath\tsha256\tbytes\tsource\n")
        for row in sorted(rows):
            fh.write("\t".join(row) + "\n")

    print(f"[snapshot] {total} files -> {DEST}")
    if missing:
        print(f"[snapshot] WARNING: {len(missing)} pool(s) absent from this store:")
        for pool in missing:
            print(f"    {pool}")
    return 0


def check() -> int:
    if not os.path.exists(MANIFEST):
        sys.exit(f"no manifest at {MANIFEST} -- run without --check first")
    bad = n = 0
    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("relpath\t"):
                continue
            rel, want, _size, _src = line.rstrip("\n").split("\t")
            path = os.path.join(DEST, rel)
            n += 1
            if not os.path.exists(path):
                print(f"MISSING  {rel}")
                bad += 1
            elif sha256(path) != want:
                print(f"MODIFIED {rel}")
                bad += 1
    print(f"[check] {n - bad}/{n} files match MANIFEST.tsv")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
