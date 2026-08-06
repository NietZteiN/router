"""Headline experiment: sample deletions -> verify exactness + measure forgetting.

For each sampled record r:
  1. unlearn(r)  — retrain the affected adapters on D_j \\ {r} (Alg 1).
  2. verify_exactness.deletion — affected adapters reproduce the from-scratch
     oracle; sampled untouched adapters are unchanged. (bitwise on CPU; on GPU
     reports the param-distance bound -> distributional exactness.)
  3. forget efficacy — canary_hit / EM / VerbMem on r under {legonet (pre),
     post-unlearn, never-trained base}. Expect pre high, post ~ base.

Writes results/exactness.json + reports/exactness_table.md.

    python run_exactness_sample.py --config configs/legonet_7b.json --n_del 2
"""
import argparse
import json
import os

from legonet_common import Paths, load_config, load_records, write_json
from eval_memorization import evaluate
from unlearn import post_unlearn_adapter_dir_fn, unlearn
from verify_exactness import deletion

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


def pick_sample(cfg, n_del):
    """Spread the sample across distinct adapter-sets where possible."""
    paths = Paths(cfg)
    with open(paths.assignment_path) as f:
        assignment = json.load(f)
    seen, picked = set(), []
    for rid in (r["id"] for r in load_records(paths.records_path)):
        grp = tuple(sorted(assignment["record_to_keys"][rid]))
        if grp not in seen:
            seen.add(grp)
            picked.append(rid)
        if len(picked) >= n_del:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_del", type=int, default=2)
    ap.add_argument("--untouched_sample", type=int, default=3)
    ap.add_argument("--n_neighbors", type=int, default=10,
                    help="retained members of the affected adapters to check for collateral damage")
    ap.add_argument("--gen_cap", type=int, default=64)
    ap.add_argument("--with_base", action="store_true",
                    help="also eval the never-trained base on deleted records (extra load)")
    ap.add_argument("--no_verify", action="store_true",
                    help="skip the from-scratch oracle (just unlearn+forget+timing) — for the sweep, "
                         "where exactness is already validated and we want cost+forget per cell")
    args = ap.parse_args()
    import time as _time
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    paths = Paths(cfg)
    with open(paths.assignment_path) as f:
        assignment = json.load(f)

    sample = pick_sample(cfg, args.n_del)
    print(f"exactness sample ({len(sample)} deletions): {sample}")

    rng = __import__("numpy").random.default_rng(cfg["base_seed"])
    per_deletion = []
    for i, rid in enumerate(sample):
        tag = f"d{i}"
        t0 = _time.time()
        umani = unlearn(cfg, [rid], tag=tag, force=True)
        unlearn_seconds = _time.time() - t0   # wall-clock per deletion (retrain affected adapters)
        dir_fn = post_unlearn_adapter_dir_fn(cfg, umani)

        if args.no_verify:
            ex = {"affected": umani["affected_adapters"], "structural_ok": None,
                  "all_bitwise": None, "max_rel_l2": float("nan"), "exact_within_tol": None}
        else:
            ex = deletion(cfg, tag, [rid], untouched_sample=args.untouched_sample)

        # forget efficacy on the deleted record
        _, pre = evaluate(cfg, "legonet", [rid], gen_cap=args.gen_cap)
        _, post = evaluate(cfg, "legonet", [rid], adapter_dir_fn=dir_fn, gen_cap=args.gen_cap)
        row = {"record": rid, "affected": ex["affected"],
               "unlearn_seconds": unlearn_seconds,
               "structural_ok": ex["structural_ok"], "all_bitwise": ex["all_bitwise"],
               "max_rel_l2": ex["max_rel_l2"], "exact_within_tol": ex["exact_within_tol"],
               "forget": {"pre": pre, "post_unlearn": post}}
        if not args.no_verify:
            # affected (unlearn vs oracle, both exclude r) vs untouched (original vs oracle,
            # same data, different run) -> comparable distances => distributional exactness.
            row["exactness_detail"] = {"affected_checks": ex["affected_checks"],
                                       "untouched_checks": ex["untouched_checks"]}
        if args.with_base:
            _, base = evaluate(cfg, "base", [rid], gen_cap=args.gen_cap)
            row["forget"]["base"] = base

        # collateral damage: retained neighbors (members of affected adapters, minus r)
        neighbors = sorted({m for j in ex["affected"] for m in assignment["members"][str(j)]} - {rid})
        if neighbors:
            pick = sorted(rng.choice(neighbors, size=min(args.n_neighbors, len(neighbors)),
                                     replace=False).tolist())
            _, n_pre = evaluate(cfg, "legonet", pick, gen_cap=args.gen_cap)
            _, n_post = evaluate(cfg, "legonet", pick, adapter_dir_fn=dir_fn, gen_cap=args.gen_cap)
            row["collateral"] = {"n_neighbors": len(pick), "pre": n_pre, "post_unlearn": n_post}
        per_deletion.append(row)

    summary = {
        "config": cfg["name"], "n_del": len(sample),
        "mean_unlearn_seconds": sum(d["unlearn_seconds"] for d in per_deletion) / len(per_deletion),
        "all_deletions_bitwise": all(d["all_bitwise"] for d in per_deletion)
            if all(d["all_bitwise"] is not None for d in per_deletion) else None,
        "max_rel_l2_over_all": max(d["max_rel_l2"] for d in per_deletion),
        "all_exact_within_tol": all(d["exact_within_tol"] for d in per_deletion),
        "all_structural_ok": all(d["structural_ok"] for d in per_deletion),
        "per_deletion": per_deletion,
    }
    write_json(os.path.join(paths.results_dir, "exactness.json"), summary)

    # markdown table
    lines = [f"# Exactness — {cfg['name']}", "",
             f"- all deletions bitwise: **{summary['all_deletions_bitwise']}**",
             f"- max rel-L2 over all checks: **{summary['max_rel_l2_over_all']:.3e}**",
             f"- all within tol (<1e-3): **{summary['all_exact_within_tol']}**",
             f"- all structural-ok: **{summary['all_structural_ok']}**", "",
             "Forget = deleted record's canary_em (pre→post→base). Collateral = retained neighbors' "
             "canary_em (pre→post, should be ≈equal).", "",
             "| record | affected | max_rel_l2 | forget canary_em (pre→post / base) | neigh canary_em (pre→post) | neigh ppl (pre→post) |",
             "|---|---|---|---|---|---|"]
    for d in per_deletion:
        f = d["forget"]
        base_cem = f.get("base", {}).get("canary_em", float("nan"))
        c = d.get("collateral")
        if c:
            ncem = f"{c['pre']['canary_em']:.3f}→{c['post_unlearn']['canary_em']:.3f}"
            nppl = f"{c['pre']['perplexity']:.2f}→{c['post_unlearn']['perplexity']:.2f}"
        else:
            ncem = nppl = "—"
        lines.append(
            f"| {d['record']} | {d['affected']} | {d['max_rel_l2']:.2e} | "
            f"{f['pre']['canary_em']:.3f}→{f['post_unlearn']['canary_em']:.3f} / {base_cem:.3f} | "
            f"{ncem} | {nppl} |")
    with open(os.path.join(paths.reports_dir, "exactness_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_deletion"}, indent=2))


if __name__ == "__main__":
    main()
