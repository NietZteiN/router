"""Aggregate the overnight RAMoLE campaign into one morning report.

Globs every arm's result JSONs under {root}/runs/ramole_l32_3b*/results/, plus retrieval-accuracy
and router metadata, and writes a Markdown report (+ prints it). Robust to missing files: an arm
that failed simply shows blanks, so the report always generates.

    python collect_overnight.py [--root ${TOFU_CKPT_STORE}/ramole] [--out REPORT.md]
"""
import argparse
import glob
import json
import os

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

METRICS = ["em", "es", "verbmem", "perplexity", "canary_em", "canary_hit"]


def _load(root):
    """{run_name: {label: aggregate}}, {run_name: retrieval_accuracy}, {run_name: router_meta}."""
    res, retacc, rmeta = {}, {}, {}
    for run_dir in sorted(glob.glob(os.path.join(root, "runs", "ramole_l32_3b*"))):
        run = os.path.basename(run_dir)
        for f in glob.glob(os.path.join(run_dir, "results", "*.json")):
            base = os.path.basename(f)
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if base == "retrieval_accuracy.json":
                retacc[run] = d
            elif "aggregate" in d:
                res.setdefault(run, {})[d.get("label", base[:-5])] = d["aggregate"]
        mp = os.path.join(run_dir, "router_meta.json")
        if os.path.isfile(mp):
            try:
                rmeta[run] = json.load(open(mp))
            except Exception:
                pass
    return res, retacc, rmeta


def _g(agg, k):
    v = agg.get(k) if agg else None
    return f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v))


def _row(label, agg):
    return (f"| {label:30s} | {_g(agg,'em'):>6} | {_g(agg,'verbmem'):>7} | {_g(agg,'perplexity'):>9} "
            f"| {_g(agg,'canary_em'):>9} | {_g(agg,'canary_hit'):>10} | {str(agg.get('k_eff','—')) if agg else '—':>3} "
            f"| {str(agg.get('num_records','—')) if agg else '—':>4} |")


_HDR = ("| label                          |     em | verbmem |       ppl | canary_em | canary_hit |   k |    N |\n"
        "|--------------------------------|--------|---------|-----------|-----------|------------|-----|------|")


def build_report(root):
    res, retacc, rmeta = _load(root)
    A = "ramole_l32_3b_n32_k3"
    L = []
    L.append("# RAMoLE overnight campaign — report\n")
    L.append(f"Root: `{root}`  ·  arms found: {', '.join(sorted(res)) or '(none)'}\n")

    # 1) Headline: composition rule on the same experts
    L.append("\n## 1. Headline — composition rule (Arm A, same n=32 experts)\n")
    L.append("RAMoLE learned router vs LegoNet uniform 1/k vs perfect-selection.\n")
    L.append(_HDR)
    a = res.get(A, {})
    for lbl in ["router_retriever_iid", "router_retriever_ood", "router_keys_iid",
                "mean_keys_iid", "perfect"]:
        if lbl in a:
            L.append(_row(lbl, a[lbl]))

    # 2) Random LoRA Dropout ablation (paper's key claim)
    L.append("\n## 2. Random LoRA Dropout ablation (p=0.5 vs 0)\n")
    L.append("Higher utility OOD with dropout is the paper's claim. Compare same labels across arms.\n")
    L.append(_HDR)
    for run, tag in [(A, "router (p=0.5)"), ("ramole_l32_3b_d0", "router (p=0)")]:
        for lbl in ["router_keys_iid", "router_keys_ood", "router_retriever_iid", "router_retriever_ood"]:
            if lbl in res.get(run, {}):
                L.append(_row(f"{tag}:{lbl.replace('router_','')}", res[run][lbl]))

    # 3) Split & rank ablations (key-routed, isolates the router)
    L.append("\n## 3. Split & rank ablations (router_keys_iid)\n")
    L.append(_HDR)
    for run, tag in [(A, "ref / r16 (default)"), ("ramole_l32_3b_corpus", "corpus split"),
                     ("ramole_l32_3b_r6", "rank 6")]:
        if "router_keys_iid" in res.get(run, {}):
            L.append(_row(tag, res[run]["router_keys_iid"]))

    # 4) Unlearning demo — forget records before vs after deletion (router UNCHANGED)
    L.append("\n## 4. Unlearning demo — delete records, router NOT retrained\n")
    L.append("Forget records served through the unchanged router; 'after' = post-deletion retrained "
             "experts. Memorization should drop (forgotten) with no router retrain.\n")
    L.append(_HDR)
    for method in ["router", "mean"]:
        for tag in ["d0", "d1", "d2"]:
            for state in ["before", "after"]:
                lbl = f"{method}_unlearn_{tag}_{state}"
                if lbl in a:
                    L.append(_row(lbl, a[lbl]))

    # 5) Retrieval accuracy (paper §2.4)
    L.append("\n## 5. Retrieval accuracy (Arm A retriever)\n")
    if retacc.get(A):
        d = retacc[A]
        L.append("| encoder | condition | top-k |\n|---|---|---|")
        for who in ["off_the_shelf", "finetuned"]:
            if d.get(who):
                for cond, acc in d[who].items():
                    if cond == "ood":   # degenerate metric (see retriever.py); skip
                        continue
                    L.append(f"| {who} | {cond} | {acc} |")
    else:
        L.append("(retrieval_accuracy.json not found)")

    # router metadata
    L.append("\n## Router metadata\n")
    for run in sorted(rmeta):
        m = rmeta[run]
        L.append(f"- **{run}**: split={m.get('router_train_split')} dropout_p={m.get('dropout_p')} "
                 f"rank={m.get('router_rank')} final_loss={m.get('final_loss')} "
                 f"train_clusters={len(m.get('train_clusters',[]))}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ["TOFU_CKPT_STORE"], "ramole"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = build_report(args.root)
    out = args.out or os.path.join(args.root, "OVERNIGHT_REPORT.md")
    with open(out, "w") as f:
        f.write(report)
    print(report)
    print(f"\n[collect_overnight] -> {out}")


if __name__ == "__main__":
    main()
