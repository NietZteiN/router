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
from shard_utils import get_author_shard, parse_author_ids

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


def _query_transform(args, data_full):
    """(question, author) -> the question actually SERVED.

    `none` is the identity and leaves every existing arm byte-identical. The others exist
    because the CSAR headline was measured on TOFU's gold-form questions, which name their
    author in ~90% of rows — the same property that turned out to carry the H3 granularity
    ladder (see log/selector_audit/2026-08-07_h3-is-a-lexical-artifact.md). A harm measured only
    on queries that name the deleted person is worth exactly as much as a defence measured that
    way, so the same stress test applies here.
    """
    mode = getattr(args, "query_transform", "none") or "none"
    if mode == "none":
        return lambda q, a: q

    from analyze_router_shift import strip_names, indirect_reference
    from router import _extract_author_names
    names = {a: _extract_author_names([data_full[a * 20 + w]["question"] for w in range(20)])
             for a in range(200)}
    if mode == "name_stripped":
        return lambda q, a: strip_names(q, names[a])
    if mode == "indirect":
        sa = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "selector_audit")
        if sa not in sys.path:
            sys.path.insert(0, sa)
        import csar
        gold = {a: [data_full[a * 20 + w]["answer"] for w in range(20)] for a in range(200)}
        ix = csar.build_index(gold)

        def _facts(a):
            f = sorted(ix.distinctive(a, csar.DEFAULT_MAX_ADF), key=len, reverse=True)
            own = {n.lower() for n in names.get(a, [])}
            return [x for x in f if not any(x in n or n in x for n in own)][:3]

        return lambda q, a: indirect_reference(q, names[a], _facts(a))
    raise SystemExit(f"unknown --query_transform {mode!r}")


def _forget_sets(args):
    """(forget_authors, forget_shards) — the deleted authors and the shards that host them.

    Legacy path (`--forget_shard_id` alone) is unchanged: one shard, its own authors. With
    `--forget_author_ids` the deleted set is explicit and may span many shards, which is what
    a per-author pool (k=200) needs to express TOFU's 20-author forget10.
    """
    if args.forget_author_ids:
        authors = parse_author_ids(args.forget_author_ids)
        per_shard = 200 // args.k
        shards = {a // per_shard for a in authors}
        # a shard hosting BOTH deleted and retained authors cannot be dropped without taking
        # retained data with it — the partition/request mismatch, and silently dropping it
        # would misreport the retain side. Refuse rather than fudge.
        straddling = sorted(s for s in shards
                            if not set(get_author_shard(args.k, s)).issubset(set(authors)))
        if straddling:
            raise SystemExit(
                f"--forget_author_ids straddles shard(s) {straddling} at k={args.k}: the shard "
                f"also hosts retained authors, so it cannot be deleted as a unit. Use a k whose "
                f"shards align with the requested authors.")
        return authors, shards
    fsid = args.forget_shard_id
    return get_author_shard(args.k, fsid), {fsid}


def _forget_rows(forget_authors, questions_per_author, max_questions):
    """Row indices for the orphan question set.

    `--max_questions` keeps its historical meaning (a head slice, the `_c40` tier at k=10, where
    one shard's 400 rows all belong to the same deleted unit). Across a 20-author set a head
    slice would cover only the first author or two, so `--questions_per_author` samples the head
    of EVERY deleted author instead — the same budget, spread over the sources being deleted.
    """
    rows = [a * 20 + w for a in forget_authors for w in range(20)]
    if questions_per_author:
        n = min(int(questions_per_author), 20)
        rows = [a * 20 + w for a in forget_authors for w in range(n)]
    if max_questions:
        rows = rows[:max_questions]
    return rows


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
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    forget_authors, forget_shards = _forget_sets(args)
    fsid = args.forget_shard_id                      # kept for the output header / legacy runs
    forget_rows = _forget_rows(forget_authors, args.questions_per_author, args.max_questions)
    shard_rows = {j: [a * 20 + w for a in get_author_shard(args.k, j) for w in range(20)]
                  for j in range(args.k)}
    per_shard = 200 // args.k

    model, tok = load_all_shard_adapters(args.model_name, args.shards_dir, args.k,
                                         lazy_cache=args.lazy_adapter_cache)
    model.eval()

    # one embed_fn (MiniLM) for the vs_sibgold nearest-question lookup, cache dir on the pool
    _, _, embed_fn = build_shard_centroids(args.hf_home, args.k, list(range(args.k)),
                                           next(model.parameters()).device,
                                           encoder_name=args.router_encoder)
    shard_q_emb = {}

    # build one router per strategy, the WHOLE deleted set baked into the exclude set
    cache_dir = os.path.join(args.shards_dir, "centroids")
    exclude = frozenset(forget_shards)
    routers = {}
    for strat in strategies:
        rm = _build_routed_model(model, args.k, strat, exclude,
                                 tokenizer=tok, dataset=data_full, centroid_cache_dir=cache_dir)
        routers[strat] = rm.router

    transform = _query_transform(args, data_full)

    per_strategy = {s: [] for s in strategies}
    for qi, ridx in enumerate(forget_rows):
        q_orig, gold = data_full[ridx]["question"], data_full[ridx]["answer"]
        author = ridx // 20
        own_shard = author // per_shard
        # The SERVED query. Routing and generation both see it, because that is what a user who
        # does not name the person actually sends. `q_orig` is kept only for the record and for
        # the nearest-survivor-question lookup, which asks "which of the survivor's questions is
        # closest to what was really being asked".
        q = transform(q_orig, author)
        v = embed_fn([q_orig])[0]
        enc_ids = tok(q, return_tensors="pt").input_ids[0].to(
            next(model.parameters()).device)  # activation routers run a fwd → must be on-device

        # LAZY generation: own + base up front, then only the shards a router actually picks.
        # The original eager sweep over every survivor amortized across 9 strategies at k=10;
        # at k=200 it would be ~200 generations per question, and with --lazy_adapter_cache it
        # would also thrash the adapter LRU on every one of them.
        gens = {}

        def _gen_for(shard_id):
            if shard_id not in gens:
                if shard_id == -1:
                    with model.disable_adapter():
                        gens[-1] = _gen(model, tok, q, args.max_new_tokens)
                else:
                    model.set_adapter(f"shard_{shard_id}")
                    gens[shard_id] = _gen(model, tok, q, args.max_new_tokens)
            return gens[shard_id]

        own_gen = _gen_for(own_shard)
        base_gen = _gen_for(-1)

        for strat in strategies:
            r = routers[strat]
            arg = enc_ids.unsqueeze(0) if isinstance(r, ActivationRouter) else q
            sib = int(r.route(arg, exclude=exclude))
            assert sib not in forget_shards, (
                f"{strat} routed orphan row {ridx} to deleted shard {sib}")
            if sib not in shard_q_emb:
                shard_q_emb[sib] = embed_fn([data_full[x]["question"] for x in shard_rows[sib]])
            near = shard_rows[sib][int(np.argmax(shard_q_emb[sib] @ v))]
            sib_gold = data_full[near]["answer"]
            sib_gen = _gen_for(sib)
            per_strategy[strat].append({
                "row": int(ridx), "author": int(author), "own_shard": int(own_shard),
                "sibling_shard": sib,
                "sibling_vs_gold": _rouge(sib_gen, gold),
                "own_vs_gold": _rouge(own_gen, gold),
                "base_vs_gold": _rouge(base_gen, gold),
                "sibling_vs_basegen": _rouge(sib_gen, base_gen),
                "sibling_vs_sibgold": _rouge(sib_gen, sib_gold),
                # the raw text, so a fact-level classifier (selector_audit/csar.py) can read
                # the same generations these ROUGE axes were computed from
                "question": q_orig, "question_served": q,
                "gold": gold, "sibling_gold": sib_gold,
                "gen_sibling": sib_gen, "gen_own": own_gen, "gen_base": base_gen,
            })
        if (qi + 1) % 25 == 0:
            print(f"[content_audit/per_strategy] {qi+1}/{len(forget_rows)}", flush=True)

    def _agg(rows, key):
        vv = [r[key] for r in rows]
        return {"mean": float(np.mean(vv)), "median": float(np.median(vv)), "n": len(vv)}

    out = {"mode": "per_strategy", "forget_shard_id": fsid, "n_questions": len(forget_rows),
           "k": args.k, "forget_authors": [int(a) for a in forget_authors],
           "forget_shards": sorted(int(s) for s in forget_shards),
           "questions_per_author": args.questions_per_author,
           "query_transform": getattr(args, "query_transform", "none") or "none",
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
    ap.add_argument("--forget_author_ids", default=None,
                    help="Explicit deleted authors, e.g. '180-199' (per-strategy mode only). The "
                         "deleted set may span many shards, which is how a k=200 per-author pool "
                         "expresses TOFU's 20-author forget10; --forget_shard_id alone would "
                         "delete one author. Refuses a set that straddles a shard.")
    ap.add_argument("--query_transform", default="none",
                    choices=["none", "name_stripped", "indirect"],
                    help="Transform the SERVED query (routing and generation both see it). "
                         "`none` leaves every existing arm byte-identical. The others ask "
                         "whether the CSAR harm, like the H3 defence before it, is an artifact "
                         "of TOFU questions naming their author in ~90% of rows.")
    ap.add_argument("--questions_per_author", type=int, default=None,
                    help="Sample the first N questions of EVERY deleted author instead of a head "
                         "slice of the whole set. Across 20 authors --max_questions 40 would "
                         "cover only the first two; --questions_per_author 2 covers all twenty "
                         "for the same budget.")
    ap.add_argument("--lazy_adapter_cache", type=int, default=0,
                    help="Keep at most N shard adapters resident (eval_tofu.lazify_shard_"
                         "adapters). Required at k=200 r32: PEFT fp32-casts every adapter.")
    ap.add_argument("--max_questions", type=int, default=None, help="smoke cap")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.strategies:
        return run_per_strategy(args)
    for flag in ("forget_author_ids", "questions_per_author"):
        if getattr(args, flag):
            raise SystemExit(f"--{flag} is per-strategy mode only; pass --strategies. The "
                             f"single-MiniLM default arm is single-shard by construction.")
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
