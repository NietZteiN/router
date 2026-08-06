#!/usr/bin/env python3
"""Snapshot the result files the routing/selection tables cite into results_snapshot/.

The numbers live in small JSONs (and, for the router-leak battery, .npz score matrices) scattered
under a checkpoint store that is ~674 GB and cannot go in git. The result files themselves are
~36 MB. Copying just the cited pools makes every CPU-side analysis re-runnable on a new cluster
with no GPU, no /storage2 and no HF token -- which is the point of this repo.

    python snapshot_results.py                    # from $TOFU_CKPT_ROOT / the checkpoints symlink
    python snapshot_results.py --ckpt-root /path  # explicit store
    python snapshot_results.py --check            # re-hash against MANIFEST.tsv, write nothing

Adapted from merge_tables_7b/reproduce/snapshot_results.py with two changes that matter here:

  * .npz is copied, not just .json. `analyze_router_family.py` recomputes its aggregates from the
    raw per-strategy score matrices; without them the snapshot can only be read back, not
    independently verified, and "the committed table matches itself" is not a check.
  * a second store. RAMoLE writes under its own root (runs/<name>/results), so the DBpedia
    retriever and RouterLoRA-seed cells would be silently missing from a one-root walk.

`*.progress.json` is excluded everywhere: it is per-eval checkpointing chatter, not a result.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "results_snapshot")
MANIFEST = os.path.join(DEST, "MANIFEST.tsv")

# Pools grouped by the table that reads them, so a reader can see WHY each one is here.
# Keys are the store they live in: "tofu" = $TOFU_CKPT_ROOT, "ramole" = $RAMOLE_CKPT_ROOT.
POOLS: dict[str, list[tuple[str, str]]] = {
    "Appendix E Tables 9-10 -- the orphan battery (k=10, 1B scaffolded)": [
        ("tofu", "Llama-3.2-1B-Instruct_experts_scaf_k10"),
    ],
    "Appendix E -- keyed banks + learned cross-attn gate (RouterLoRA x3 seeds, base-pin audit)": [
        ("tofu", "Llama-3.2-1B-Instruct_legonet_n32_k3"),
        ("tofu", "Llama-2-7B-chat-hf_legonet_n32_k3"),
    ],
    "Appendix E -- the oracle row (0.8236 utility, deletion at 0.0000)": [
        ("tofu", "Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"),
    ],
    "Appendix E -- 7B orphan coverage across scale and granularity": [
        ("tofu", "Llama-2-7B-chat-hf_k10_r32_e5_lr1e4"),
        ("tofu", "Llama-2-7B-chat-hf_k50_r32_e5_lr1e4"),
    ],
    "Appendix D Table 6 -- merged vs routed across shard counts": [
        ("tofu", "Llama-2-7B-chat-hf_k4_r32_e5_lr1e4"),
        ("tofu", "Llama-2-7B-chat-hf_k20_r32_e5_lr1e4"),
        ("tofu", "Llama-2-7B-chat-hf_k100_r32_e5_lr1e4"),
        ("tofu", "Llama-2-7B-chat-hf_k200_r8_e5_lr1e4"),
    ],
    "Appendix D Table 7 -- selection over identical summed weights (SIFT / ClAMU ladder)": [
        ("tofu", "Llama-3.2-1B-Instruct_sift_masks"),
        ("tofu", "Llama-3.2-1B-Instruct_clamu_K200"),
    ],
    "Appendix D Table 8 -- PEFT families composed vs routed": [
        ("tofu", "Llama-3.2-1B-Instruct_peft_dora_k10"),
        ("tofu", "Llama-3.2-1B-Instruct_peft_ia3_k10"),
        ("tofu", "Llama-3.2-1B-Instruct_peft_vera_k10"),
        ("tofu", "Llama-3.2-1B-Instruct_peft_prefix_k10"),
    ],
    "Anchors -- base / joint-ft, the denominators every row above is read against": [
        ("tofu", "Llama-3.2-1B-Instruct"),
    ],
    "Appendix E -- RAMoLE retriever + DBpedia (its own store)": [
        ("ramole", "runs/ramole_l32_3b_n32_k3"),
        ("ramole", "runs/ramole_l32_3b_d0"),
        ("ramole", "runs/ramole_l32_3b_s43"),
        ("ramole", "runs/ramole_l32_3b_s44"),
    ],
}

KEEP_SUFFIXES = (".json", ".npz", ".csv", ".md")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_results(pool_dir: str):
    """Every results/**/* worth keeping, minus the per-eval .progress.json chatter."""
    results = os.path.join(pool_dir, "results")
    if not os.path.isdir(results):
        return
    for root, _dirs, files in os.walk(results):
        for name in sorted(files):
            if name.endswith(".progress.json"):
                continue
            if name.endswith(KEEP_SUFFIXES):
                yield os.path.join(root, name)


def resolve_root(kind: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if kind == "tofu":
        for cand in (os.environ.get("TOFU_CKPT_ROOT"),
                     os.path.join(HERE, "tofu_sisa_lora", "checkpoints")):
            if cand and os.path.isdir(cand):
                return cand
    else:
        for cand in (os.environ.get("RAMOLE_CKPT_ROOT"),
                     os.path.join(HERE, "ramole", "checkpoints")):
            if cand and os.path.isdir(cand):
                return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", default=None, help="the tofu_sisa_lora store")
    ap.add_argument("--ramole-root", default=None, help="the ramole store")
    ap.add_argument("--check", action="store_true",
                    help="re-hash the existing snapshot against MANIFEST.tsv; write nothing")
    args = ap.parse_args()

    if args.check:
        if not os.path.isfile(MANIFEST):
            print(f"no {os.path.relpath(MANIFEST, HERE)} -- nothing to check", file=sys.stderr)
            return 1
        bad = missing = n = 0
        with open(MANIFEST) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                rel, want, size, _src = line.rstrip("\n").split("\t")
                p = os.path.join(DEST, rel)
                n += 1
                if not os.path.isfile(p):
                    print(f"  MISSING  {rel}"); missing += 1
                elif sha256(p) != want:
                    print(f"  CHANGED  {rel}"); bad += 1
        print(f"\n{n} files: {n - bad - missing} ok, {bad} changed, {missing} missing")
        return 0 if (bad == missing == 0) else 1

    roots = {"tofu": resolve_root("tofu", args.ckpt_root),
             "ramole": resolve_root("ramole", args.ramole_root)}
    if roots["tofu"] is None:
        sys.exit("No checkpoint store found. Pass --ckpt-root, set TOFU_CKPT_ROOT, or create the\n"
                 "tofu_sisa_lora/checkpoints symlink (see SETUP.md).")

    os.makedirs(DEST, exist_ok=True)
    rows: list[tuple[str, str, int, str]] = []
    n_files = n_bytes = 0
    skipped: list[str] = []

    for group, pools in POOLS.items():
        print(f"\n{group}")
        for kind, pool in pools:
            root = roots[kind]
            if root is None:
                skipped.append(f"{pool} (no {kind} store)")
                print(f"  {pool:<50} -- SKIPPED, no {kind} store")
                continue
            src_pool = os.path.join(root, pool)
            if not os.path.isdir(src_pool):
                skipped.append(f"{pool} (absent from {root})")
                print(f"  {pool:<50} -- SKIPPED, absent")
                continue
            k = b = 0
            for src in iter_results(src_pool):
                rel = os.path.join(kind, pool, os.path.relpath(src, src_pool))
                dst = os.path.join(DEST, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                size = os.path.getsize(dst)
                rows.append((rel, sha256(dst), size, src))
                k += 1; b += size
            n_files += k; n_bytes += b
            print(f"  {pool:<50} {k:>5} files  {b / 1e6:>7.1f} MB")

    with open(MANIFEST, "w") as fh:
        fh.write("# rel_path\tsha256\tsize_bytes\tsource_path\n")
        fh.write("# Written by snapshot_results.py. `--check` re-hashes against this file, so a\n"
                 "# snapshot can be audited against the store it came from.\n")
        for rel, h, size, src in sorted(rows):
            fh.write(f"{rel}\t{h}\t{size}\t{src}\n")

    print(f"\n{n_files} files, {n_bytes / 1e6:.1f} MB -> {os.path.relpath(DEST, HERE)}/")
    if skipped:
        # Silent truncation would read as "we snapshotted everything" when we did not.
        print(f"\nSKIPPED {len(skipped)} pool(s) -- these tables are NOT verifiable offline:")
        for s in skipped:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
