"""SEUF-attribution entanglement detector (gap-analysis §9-A) for the Mode-B plant.

Before honoring a delete, measure each forget-fact's affinity across ALL experts and flag the
ones whose mass has spread onto non-owner experts (= Mode-B replication ⇒ delete-propagation
needed). For exact-key routing (no soft router) this is the §9-A "embed-and-probe" variant done
as loss probes on the actual experts:

  affinity_j(f) = softmax_j( (NLL_scaffold(f) - NLL_j(f)) / tau )     # how much expert j "holds" f
  spread(f)     = 1 - affinity_{owner}(f)                             # mass off the donor's expert

Ground truth from the manifest: planted facts (R>=2, have host copies) should show high spread;
control facts (R==1) should concentrate on the owner (shard 9). Reports ROC-AUC of spread
separating planted-vs-control, precision/recall at cfg['detector']['threshold'], and per-fact
implicated host shards (affinity above threshold on a host shard) vs the true hosts.

  python detect_entanglement.py --config C --manifest M --experts_dir DIR --device cuda --out J
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import eval_tofu as et
from eval_tofu import _answer_avg_loss
from eval_entangled_probe import _load, _load_experts


def _owner_shard(donor, k):
    return donor // (200 // k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--experts_dir", required=True)
    ap.add_argument("--surface", default="orig", choices=["orig", "para"],
                    help="probe surface for the affinity NLL (orig = the fact's canonical question)")
    ap.add_argument("--max_facts", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    cfg = _load(args.config)
    tau = cfg.get("detector", {}).get("tau", 1.0)
    thr = cfg.get("detector", {}).get("threshold", 0.5)
    k = cfg["k"]
    with open(args.manifest) as f:
        man = json.load(f)
    facts = man["facts"][:args.max_facts] if args.max_facts else man["facts"]

    model, tok = _load_experts(cfg["base_model"], args.experts_dir, k, cfg["hf_home"])
    experts = [i for i in range(k) if os.path.isdir(os.path.join(args.experts_dir, f"shard_{i}"))]

    def scaffold_nll(q, a):
        with model.disable_adapter():
            return _answer_avg_loss(model, tok, q, a)

    rows = []
    for fi, fact in enumerate(facts):
        q = fact["probe_question_orig"] if args.surface == "orig" else fact["probe_question_para"]
        a = fact["probe_answer_orig"] if args.surface == "orig" else fact["probe_answer_para"]
        base_nll = scaffold_nll(q, a)
        deltas = {}
        for j in experts:
            model.set_adapter(f"shard_{j}")
            nll_j = _answer_avg_loss(model, tok, q, a)
            deltas[j] = (base_nll - nll_j) if (nll_j == nll_j and base_nll == base_nll) else 0.0
        js = experts
        z = np.array([deltas[j] / tau for j in js])
        aff = np.exp(z - z.max()); aff = aff / aff.sum()
        owner = _owner_shard(fact["donor_author"], k)
        aff_map = {j: float(aff[i]) for i, j in enumerate(js)}
        spread = 1.0 - aff_map.get(owner, 0.0)
        true_hosts = sorted(h["shard"] for h in fact["hosts"])
        flagged_hosts = sorted(j for j in js if j != owner and j in cfg["host_shards"]
                               and aff_map[j] >= thr / max(1, len(true_hosts) or 1))
        rows.append({"fact_id": fact["fact_id"], "R": fact["R"], "mode": fact["mode"],
                     "owner_shard": owner, "spread": spread,
                     "owner_affinity": aff_map.get(owner, 0.0),
                     "true_hosts": true_hosts, "flagged_hosts": flagged_hosts})
        if (fi + 1) % 25 == 0:
            print(f"[detect] {fi+1}/{len(facts)}", flush=True)

    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    labels = np.array([1 if r["R"] >= 2 else 0 for r in rows])
    spreads = np.array([r["spread"] for r in rows])
    auc = float(roc_auc_score(labels, spreads)) if len(set(labels)) > 1 else float("nan")
    preds = (spreads >= thr).astype(int)
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    # host identification: over planted facts, fraction of true hosts recovered
    host_hits = host_tot = 0
    for r in rows:
        if r["R"] >= 2:
            host_hits += len(set(r["flagged_hosts"]) & set(r["true_hosts"]))
            host_tot += len(r["true_hosts"])
    host_recall = host_hits / host_tot if host_tot else float("nan")

    spread_by = {}
    for r in rows:
        spread_by.setdefault(r["R"], []).append(r["spread"])

    out = {
        "experts_dir": args.experts_dir, "surface": args.surface, "tau": tau, "threshold": thr,
        "n_facts": len(rows), "detector_auc": auc, "precision": prec, "recall": rec,
        "host_identification_recall": host_recall,
        "mean_spread_by_R": {str(R): float(np.mean(v)) for R, v in sorted(spread_by.items())},
        "per_fact": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[detect_entanglement] AUC={auc:.3f} P={prec:.3f} R={rec:.3f} "
          f"host_recall={host_recall:.3f} -> {args.out}")
    print("  mean spread by R:", out["mean_spread_by_R"])


if __name__ == "__main__":
    main()
