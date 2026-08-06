"""R3 sibling-content audit: WHAT does the router leak actually say?

For every forget-shard question (the 400 orphans), greedy-generate under three serving arms
sharing one loaded pool:
  own     — the deleted authors' own expert (shard `--forget_shard_id`): the pre-deletion ceiling
  sibling — the nearest-surviving-centroid shard's expert (the post-deletion embed-route leak)
  base    — adapters disabled = base+scaffold (what tombstone/hard-key deletion serves): the floor

Per question, three ROUGE-L recall axes classify the leak content:
  vs_gold    — generation vs the DELETED author's gold answer (disclosure axis)
  vs_basegen — sibling/own generation vs the base arm's generation for the same question
               (high = generic non-answer; low here AND low vs_gold = novel confabulation)
  vs_sibgold — sibling generation vs the gold answer of the sibling shard's nearest question
               (cross-author disclosure: serving author B's real facts for a query about A)
The confabulation rate (H5 statistic) = fraction of sibling generations with
vs_basegen < 0.5 and vs_gold < 0.5.

  python dump_generations_routed.py --model_name <scaffolded_base> \
      --shards_dir checkpoints/<slug>_experts_scaf_k10 --k 10 --forget_shard_id 9 \
      [--max_questions 40] --out reports/sibling_content_audit.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import eval_tofu as et
from eval_tofu import _build_qa_prompt, _get_rouge_metric, load_all_shard_adapters
from eval_routed_scaffold import build_shard_centroids
from shard_utils import get_author_shard

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


def _gen(model, tok, q, max_new_tokens=64):
    device = next(model.parameters()).device
    enc = tok(_build_qa_prompt(tok, q), return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _rouge(pred: str, ref: str) -> float:
    r = _get_rouge_metric().compute(predictions=[pred], references=[ref],
                                    rouge_types=["rougeL"], use_aggregator=False)
    return float(r["rougeL"][0].recall if hasattr(r["rougeL"][0], "recall") else r["rougeL"][0])


def run_per_strategy(args):
    """Wave-2: sibling-content audit for EVERY router strategy in ONE pass. Generate each
    orphan question's answer under own / base / *every surviving shard* once (cached), then for
    each strategy pick the shard its router routes to (forget shard excluded) and read the three
    ROUGE-L axes off the cached generation. Reuses `merge_lora._build_routed_model` so the route
    is the real `router.py` strategy, not a re-implementation. `centroid_lm*/ppl/activation_norm/
    attn_norm/logit_div` score via the loaded pool; the rest via text/embeddings."""
    import numpy as np  # local — mirror main()'s imports
    from merge_lora import _build_routed_model
    from router import ActivationRouter

    os.environ["HF_HOME"] = args.hf_home
    torch.manual_seed(args.seed)
    data_full = et.load_tofu_data(args.hf_home)["full"]
    fsid = args.forget_shard_id
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    forget_rows = [a * 20 + w for a in get_author_shard(args.k, fsid) for w in range(20)]
    if args.max_questions:
        forget_rows = forget_rows[:args.max_questions]
    shard_rows = {j: [a * 20 + w for a in get_author_shard(args.k, j) for w in range(20)]
                  for j in range(args.k)}

    model, tok = load_all_shard_adapters(args.model_name, args.shards_dir, args.k)
    model.eval()

    # one embed_fn (MiniLM) for the vs_sibgold nearest-question lookup, cache dir on the pool
    _, _, embed_fn = build_shard_centroids(args.hf_home, args.k, list(range(args.k)),
                                           next(model.parameters()).device,
                                           encoder_name=args.router_encoder)
    shard_q_emb = {}

    # build one router per strategy, forget shard baked into the exclude set
    cache_dir = os.path.join(args.shards_dir, "centroids")
    routers = {}
    for strat in strategies:
        rm = _build_routed_model(model, args.k, strat, frozenset({fsid}),
                                 tokenizer=tok, dataset=data_full, centroid_cache_dir=cache_dir)
        routers[strat] = rm.router
    survivors = [j for j in range(args.k) if j != fsid]

    per_strategy = {s: [] for s in strategies}
    for qi, ridx in enumerate(forget_rows):
        q, gold = data_full[ridx]["question"], data_full[ridx]["answer"]
        v = embed_fn([q])[0]
        enc_ids = tok(q, return_tensors="pt").input_ids[0].to(
            next(model.parameters()).device)  # activation routers run a fwd → must be on-device

        # generate own(fsid) + base + every survivor shard, once, cache by shard/-1(base)
        gens = {}
        model.set_adapter(f"shard_{fsid}")
        gens[fsid] = _gen(model, tok, q, args.max_new_tokens)
        with model.disable_adapter():
            gens[-1] = _gen(model, tok, q, args.max_new_tokens)
        for j in survivors:
            model.set_adapter(f"shard_{j}")
            gens[j] = _gen(model, tok, q, args.max_new_tokens)

        for strat in strategies:
            r = routers[strat]
            arg = enc_ids.unsqueeze(0) if isinstance(r, ActivationRouter) else q
            sib = int(r.route(arg, exclude=frozenset({fsid})))
            if sib not in shard_q_emb:
                shard_q_emb[sib] = embed_fn([data_full[x]["question"] for x in shard_rows[sib]])
            near = shard_rows[sib][int(np.argmax(shard_q_emb[sib] @ v))]
            sib_gold = data_full[near]["answer"]
            sib_gen = gens[sib]
            per_strategy[strat].append({
                "row": int(ridx), "author": int(ridx // 20), "sibling_shard": sib,
                "sibling_vs_gold": _rouge(sib_gen, gold),
                "own_vs_gold": _rouge(gens[fsid], gold),
                "base_vs_gold": _rouge(gens[-1], gold),
                "sibling_vs_basegen": _rouge(sib_gen, gens[-1]),
                "sibling_vs_sibgold": _rouge(sib_gen, sib_gold),
            })
        if (qi + 1) % 25 == 0:
            print(f"[content_audit/per_strategy] {qi+1}/{len(forget_rows)}", flush=True)

    def _agg(rows, key):
        vv = [r[key] for r in rows]
        return {"mean": float(np.mean(vv)), "median": float(np.median(vv)), "n": len(vv)}

    out = {"mode": "per_strategy", "forget_shard_id": fsid, "n_questions": len(forget_rows),
           "seed": args.seed, "max_new_tokens": args.max_new_tokens,
           "router_encoder": args.router_encoder, "strategies": {}}
    for strat, rows in per_strategy.items():
        confab = [r for r in rows if r["sibling_vs_basegen"] < 0.5 and r["sibling_vs_gold"] < 0.5]
        out["strategies"][strat] = {
            "aggregates": {key: _agg(rows, key) for key in
                           ("own_vs_gold", "sibling_vs_gold", "base_vs_gold",
                            "sibling_vs_basegen", "sibling_vs_sibgold")},
            "confabulation_rate": len(confab) / len(rows) if rows else None,
            "sibling_shard_hist": {str(s): sum(1 for r in rows if r["sibling_shard"] == s)
                                   for s in sorted({r["sibling_shard"] for r in rows})},
            "per_question": rows,
        }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[content_audit/per_strategy] {len(strategies)} strategies × {len(forget_rows)} q -> {args.out}")
    for strat in strategies:
        a = out["strategies"][strat]["aggregates"]
        print(f"  {strat:16s} sib_vs_gold {a['sibling_vs_gold']['mean']:.3f} "
              f"(floor base {a['base_vs_gold']['mean']:.3f}) confab "
              f"{out['strategies'][strat]['confabulation_rate']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_name", required=True, help="scaffolded base")
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--forget_shard_id", type=int, default=9)
    ap.add_argument("--router_encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--strategies", default=None,
                    help="comma-sep router.py strategies for the Wave-2 per-strategy audit "
                         "(e.g. centroid_sbert,centroid_lm,ppl,activation_norm,logit_div,attn_norm,"
                         "key_tfidf,centroid_lm_last,centroid_sbert_q). Omit = the single-MiniLM "
                         "sibling audit (unchanged default).")
    ap.add_argument("--max_questions", type=int, default=None, help="smoke cap")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.strategies:
        return run_per_strategy(args)
    os.environ["HF_HOME"] = args.hf_home
    torch.manual_seed(args.seed)

    data_full = et.load_tofu_data(args.hf_home)["full"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fsid = args.forget_shard_id

    # centroids over ALL shards; sibling route = argmax with the forget shard masked
    cents, sids, embed_fn = build_shard_centroids(args.hf_home, args.k, list(range(args.k)),
                                                  device, encoder_name=args.router_encoder)
    sid_row = {s: i for i, s in enumerate(sids)}

    # per-question rows of the forget shard, and each shard's question embeddings for the
    # vs_sibgold nearest-question lookup
    forget_rows = [a * 20 + w for a in get_author_shard(args.k, fsid) for w in range(20)]
    if args.max_questions:
        forget_rows = forget_rows[:args.max_questions]
    shard_rows = {j: [a * 20 + w for a in get_author_shard(args.k, j) for w in range(20)]
                  for j in range(args.k)}
    shard_q_emb = {}   # built lazily per sibling shard actually hit

    model, tok = load_all_shard_adapters(args.model_name, args.shards_dir, args.k)
    model.eval()

    rows = []
    for qi, ridx in enumerate(forget_rows):
        q, gold = data_full[ridx]["question"], data_full[ridx]["answer"]
        v = embed_fn([q])[0]
        sims = cents @ v
        masked = sims.copy()
        masked[sid_row[fsid]] = -np.inf
        sib = sids[int(np.argmax(masked))]

        model.set_adapter(f"shard_{fsid}")
        own_gen = _gen(model, tok, q, args.max_new_tokens)
        model.set_adapter(f"shard_{sib}")
        sib_gen = _gen(model, tok, q, args.max_new_tokens)
        with model.disable_adapter():
            base_gen = _gen(model, tok, q, args.max_new_tokens)

        if sib not in shard_q_emb:
            shard_q_emb[sib] = embed_fn([data_full[r]["question"] for r in shard_rows[sib]])
        near = shard_rows[sib][int(np.argmax(shard_q_emb[sib] @ v))]
        sib_gold = data_full[near]["answer"]

        rows.append({
            "row": int(ridx), "author": int(ridx // 20), "question": q, "gold": gold,
            "sibling_shard": int(sib), "sibling_nearest_row": int(near),
            "own_gen": own_gen, "sibling_gen": sib_gen, "base_gen": base_gen,
            "own_vs_gold": _rouge(own_gen, gold),
            "sibling_vs_gold": _rouge(sib_gen, gold),
            "base_vs_gold": _rouge(base_gen, gold),
            "sibling_vs_basegen": _rouge(sib_gen, base_gen),
            "sibling_vs_sibgold": _rouge(sib_gen, sib_gold),
        })
        if (qi + 1) % 25 == 0:
            print(f"[content_audit] {qi+1}/{len(forget_rows)}", flush=True)

    def _agg(key):
        v = [r[key] for r in rows]
        return {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}

    confab = [r for r in rows if r["sibling_vs_basegen"] < 0.5 and r["sibling_vs_gold"] < 0.5]
    out = {
        "forget_shard_id": fsid, "n_questions": len(rows), "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "router_encoder": args.router_encoder,
        "aggregates": {key: _agg(key) for key in
                       ("own_vs_gold", "sibling_vs_gold", "base_vs_gold",
                        "sibling_vs_basegen", "sibling_vs_sibgold")},
        "confabulation_rate": len(confab) / len(rows),
        "sibling_shard_hist": {str(s): sum(1 for r in rows if r["sibling_shard"] == s)
                               for s in sorted({r["sibling_shard"] for r in rows})},
        "per_question": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[content_audit] {len(rows)} questions -> {args.out}")
    for key, d in out["aggregates"].items():
        print(f"  {key}: mean={d['mean']:.3f} median={d['median']:.3f}")
    print(f"  confabulation_rate={out['confabulation_rate']:.3f}")


if __name__ == "__main__":
    main()
