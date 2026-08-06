"""Stage 4 — evaluate RAMoLE composition against the LegoNet 1/k baseline on the SAME expert
pool, reusing legonet's memorization/utility metrics (`eval_memorization.metrics_for_records`).

Methods (one per invocation, like the tofu/legonet per-label evals — keeps one model in memory):
  router   — RAMoLE: top-k retrieved experts composed by the learned RouterLoRA cross-attention.
  mean     — LegoNet baseline: the same top-k experts composed by the uniform 1/k delta-average
             (`combine.LegoNetModel`). The head-to-head — only the composition rule differs.
  perfect  — perfect-selection upper bound: the single ideal (top-1 frozen-key) expert.

Routing of the top-k experts (`--route`): `keys` (frozen-key assignment, isolates the composition
rule) or `retriever` (the learned Stage-1 LoraRetriever, full RAMoLE). `--condition ood` masks each
record's own cluster at retrieval time (cross-task generalization; only affects `retriever`).

    python eval_ramole.py --config configs/ramole_l32_3b.json --method router --route retriever \
        --condition iid --n_eval 200 --device cuda
"""
import argparse
import json
import os
from collections import defaultdict

import ramole_common as rc
from eval_memorization import aggregate, metrics_for_records   # legonet (on sys.path)

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


def _source_lego_cfg(cfg: dict) -> dict:
    """Minimal legonet-style config so combine.LegoNetModel loads the borrowed experts."""
    return {
        "root": cfg["source_root"], "name": cfg["source_run"], "n": cfg["n"], "k": cfg["k"],
        "corpus": cfg["corpus"], "base_model": cfg["base_model"],
    }


def _source_assignment_path(cfg) -> str:
    """The source run's cached assignment. Its k is a property of the LEGONET run (k=3), not the
    serving k — a k=5/8 serving config must still read assignment_n{n}_k3.json (E4 sweep)."""
    import glob
    sp = rc.source_paths(cfg)
    if os.path.isfile(sp.assignment_path):
        return sp.assignment_path
    cands = sorted(glob.glob(os.path.join(sp.run_dir, f"assignment_n{cfg['n']}_k*.json")))
    if not cands:
        raise FileNotFoundError(f"no assignment_n{cfg['n']}_k*.json under {sp.run_dir}")
    return cands[0]


def routed_sets(cfg, records, route, k, exclude_own, device, index_path=None) -> dict:
    """record_id -> tuple(expert ids) for the chosen routing.
    index_path (retriever route): serve from a rebuilt index file instead of the default (E3)."""
    sp = rc.source_paths(cfg)
    with open(_source_assignment_path(cfg)) as f:
        r2k = json.load(f)["record_to_keys"]
    if route == "keys":
        entry_k = len(next(iter(r2k.values())))
        if k <= entry_k:
            return {r["id"]: tuple(int(j) for j in r2k[r["id"]][:k]) for r in records}
        # E4 higher-k: the cached assignment only stores k=3, so re-route over the SAME frozen
        # keys with the SAME encoder that built them (identical to the assignment for k<=3).
        import numpy as np
        from routing import KNNRouter  # legonet (on sys.path)
        keys = np.load(sp.keys_path)
        with open(sp.keys_meta) as f:
            src_encoder = json.load(f)["encoder_model"]
        embed = rc.make_embed_fn(src_encoder, instruction="", device=device)
        routed = KNNRouter(keys, k).route(embed([rc.route_text(r) for r in records]))
        return {r["id"]: tuple(int(j) for j in routed[i]) for i, r in enumerate(records)}
    # retriever
    import retriever as RET
    ret = RET.LoraRetriever.load(cfg, device=device, index_path=index_path)
    excl = [int(r2k[r["id"]][0]) for r in records] if exclude_own else None
    top = ret.retrieve([rc.route_text(r) for r in records], k, exclude=excl)
    return {r["id"]: tuple(int(j) for j in top[i]) for i, r in enumerate(records)}


def evaluate(cfg, record_ids, method, route, condition, gen_cap, device, adapter_dir_fn=None,
             index_path=None):
    sp = rc.source_paths(cfg)
    by_id = {r["id"]: r for r in rc.load_records(sp.records_path)}
    records = [by_id[i] for i in record_ids]
    max_length = cfg["train"]["max_length"]

    if method == "perfect":
        with open(_source_assignment_path(cfg)) as f:
            r2k = json.load(f)["record_to_keys"]
        sets = {r["id"]: (int(r2k[r["id"]][0]),) for r in records}
        k_eff = 1
    else:
        k_eff = cfg["k"]
        sets = routed_sets(cfg, records, route, k_eff, condition == "ood", device,
                           index_path=index_path)

    groups = defaultdict(list)
    for r in records:
        groups[sets[r["id"]]].append(r)

    rows = []
    if method == "router":
        from ramole_model import RamoleModel
        rm = RamoleModel.from_config(cfg, device=device, load_router=True, adapter_dir_fn=adapter_dir_fn)
        for expert_set, recs in groups.items():
            rm.set_active(expert_set)
            rows += metrics_for_records(rm.model, rm.tokenizer, recs, max_length, gen_cap)
    else:  # mean | perfect → uniform composition via LegoNet
        from combine import LegoNetModel
        lego = LegoNetModel.from_config(
            _source_lego_cfg(cfg), adapter_dir_fn=adapter_dir_fn,
            device_map=("auto" if device != "cpu" else "cpu"))
        for expert_set, recs in groups.items():
            with lego.activated(list(expert_set)) as m:
                rows += metrics_for_records(m, lego.tokenizer, recs, max_length, gen_cap)

    agg = aggregate(rows)
    agg.update(k_eff=k_eff, num_groups=len(groups))
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["router", "mean", "perfect"], default="router")
    ap.add_argument("--route", choices=["keys", "retriever"], default="retriever")
    ap.add_argument("--condition", choices=["iid", "ood"], default="iid")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--record_ids", nargs="*", default=None)
    ap.add_argument("--gen_cap", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--unlearn_tag", default=None,
                    help="legonet deletion tag (unlearn/{tag}); evals the forget records")
    ap.add_argument("--unlearn_state", choices=["before", "after"], default="after",
                    help="before = original experts; after = post-deletion retrained experts")
    ap.add_argument("--index_policy", choices=["stale", "rebuilt"], default="stale",
                    help="retriever-route index: stale = as-built; rebuilt = excludes the deleted "
                         "records (requires --unlearn_tag; file built by routing_audit.py)")
    ap.add_argument("--label_suffix", default="",
                    help="appended to the result label (e.g. 'retain' for collateral-utility evals)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = rc.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    sp = rc.source_paths(cfg)

    adapter_dir_fn = None
    label = f"{args.method}_{args.route}_{args.condition}" if args.method != "perfect" else "perfect"
    if args.unlearn_tag:
        # Unlearning demo: eval the forgotten records with the UNCHANGED router; "after" serves
        # the post-deletion pool (retrained affected experts) — the router is never retrained.
        with open(os.path.join(sp.run_dir, "unlearn", args.unlearn_tag, "manifest.json")) as f:
            manifest = json.load(f)
        ids = args.record_ids or manifest["forget_ids"]
        if args.unlearn_state == "after":
            from unlearn import post_unlearn_adapter_dir_fn   # legonet (on sys.path)
            adapter_dir_fn = post_unlearn_adapter_dir_fn(_source_lego_cfg(cfg), manifest)
        label = f"{args.method}_unlearn_{args.unlearn_tag}_{args.unlearn_state}"
    else:
        ids = args.record_ids or [r["id"] for r in rc.load_records(sp.records_path)[: args.n_eval]]

    index_path = None
    if args.index_policy == "rebuilt":
        if not args.unlearn_tag:
            raise SystemExit("--index_policy rebuilt requires --unlearn_tag")
        index_path = os.path.join(rc.Paths(cfg).run_dir,
                                  f"lora_index_n{cfg['n']}_ex{args.unlearn_tag}.npy")
        if not os.path.isfile(index_path):
            raise SystemExit(f"rebuilt index missing (run routing_audit.py first): {index_path}")
        label += "_rebuilt"
    if args.label_suffix:
        label += f"_{args.label_suffix}"

    rows, agg = evaluate(cfg, ids, args.method, args.route, args.condition, args.gen_cap,
                         args.device, adapter_dir_fn=adapter_dir_fn, index_path=index_path)
    print(f"[{label}] {json.dumps(agg)}")

    paths = rc.Paths(cfg)
    paths.ensure()
    out = args.out or os.path.join(paths.results_dir, f"{label}.json")
    rc.write_json(out, {"method": args.method, "route": args.route, "condition": args.condition,
                        "label": label, "aggregate": agg, "rows": rows})
    print(f"[eval_ramole] -> {out}")


if __name__ == "__main__":
    main()
