"""Setup for the LegoNet-on-TOFU arm: frozen author embeddings + keys + assignment.

Computes (once, cached) the per-author answer-mean MiniLM embeddings, the n frozen
k-means keys, and the author->top-k adapter assignment, then prints the diagnostics
that gate the run: per-adapter author counts, empty-adapter count, and the set of
adapters the forget set touches (the deletion blast radius). Idempotent.

The forget_quality KS reference is the *existing* retain90 oracle + prepare_eval.py
(reused, method-independent) — symlink the retain90/ oracle into this run's
output_dir and run prepare_eval.py; this script does not build it.

    python prepare_legonet.py --config configs/legonet_tofu.json --device cuda
"""
import argparse
import os

import numpy as np

import legonet_tofu as lt
from eval_tofu import load_tofu_data

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu", help="encoder device for author embeddings")
    args = ap.parse_args()
    cfg = lt.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    os.makedirs(lt.legonet_dir(cfg), exist_ok=True)

    data = load_tofu_data(cfg["hf_home"])
    author_emb = lt.author_answer_embeddings(cfg, data["full"], device=args.device)
    keys = lt.build_keys(cfg, author_emb)
    assignment = lt.build_assignment(cfg, author_emb, keys)

    sizes = assignment["adapter_sizes"]
    forget = cfg["forget_authors"]
    aff = lt.affected_adapters(assignment, forget)

    # q2author coverage: a handful of TOFU questions are generic ("What is the full name
    # of the author?") and shared across authors, so unique-question count can be a few
    # below num_authors*per_author. Those route to the last author seen — benign unless a
    # *forget* author shares a question (checked below).
    q2a = lt.build_q2author(data["full"], cfg["num_authors"], cfg["records_per_author"])
    expected = cfg["num_authors"] * cfg["records_per_author"]
    forget_set = set(cfg["forget_authors"])
    from collections import defaultdict
    _byq = defaultdict(set)
    for i in range(min(expected, len(data["full"]))):
        _byq[lt._norm(data["full"][i]["question"])].add(i // cfg["records_per_author"])
    dup_forget = sum(1 for a in _byq.values() if len(a) > 1 and forget_set.intersection(a))

    print("=" * 70)
    print(f"LegoNet-TOFU setup: {cfg['name']}  n={cfg['n']} k={cfg['k']} route_on={cfg['route_on']}")
    print(f"  author_emb: {author_emb.shape}  keys: {keys.shape}")
    print(f"  authors/adapter: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f} "
          f"(ideal kN/n={cfg['k'] * cfg['num_authors'] / cfg['n']:.1f})")
    print(f"  empty adapters: {assignment['empty_adapters']}  (must be 0)")
    print(f"  q2author coverage: {len(q2a)}/{expected} unique questions "
          f"({expected - len(q2a)} shared/generic; {dup_forget} touch the forget set)")
    print(f"  forget set ({len(forget)} authors {forget[0]}..{forget[-1]}): "
          f"affected adapters = {len(aff)}/{cfg['n']}  {aff}")
    print(f"  -> author_emb={lt.author_emb_path(cfg)}")
    print(f"  -> keys={lt.keys_path(cfg)}")
    print(f"  -> assignment={lt.assignment_path(cfg)}")
    print("=" * 70)
    if assignment["empty_adapters"] > 0:
        print("[WARN] empty adapter(s) present — lower n or rebalance.")
    if dup_forget > 0:
        print(f"[WARN] {dup_forget} forget-author question(s) are shared with another author — "
              f"those route ambiguously; inspect if forget_quality looks off.")
    retain90 = os.path.join(cfg["output_dir"], "retain90")
    if not os.path.isdir(retain90):
        print(f"[NOTE] no retain90/ at {retain90} — symlink the SISA oracle in and run "
              f"prepare_eval.py so forget_quality isn't NaN.")


if __name__ == "__main__":
    main()
