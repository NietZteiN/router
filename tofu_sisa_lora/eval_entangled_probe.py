"""Residual-fact-recall (RFR) probe for the entangled-facts (Mode-B) experiment.

For each planted/control fact, measure whether the fact still answers after dropping its owner
(shard 9), across channels:
  expert_max  — max over SURVIVING single experts of the answer signal with that expert active
                (the headline "is the fact still in the weights"). Post-drop excludes shard 9.
  served_key  — the real composed system (scaffolded base + q2author key route + surviving
                experts, delete_shard=9); planted paraphrased questions are OOD -> base+scaffold.
Signals per (fact, probe surface in {orig, para}): answer-prob = exp(-mean answer-token NLL)
(reuse eval_tofu._answer_avg_loss) and, with --rouge, ROUGE-L recall of a greedy generation.

References for ρ = (post - floor)/(ceiling - floor): run this on the PLANTED experts with
--drop_shard none (ceiling: the donor's own expert holds it) and on the CLEAN oracle-B experts
(floor: fact never planted). ρ vs R (per mode/channel) is the headline curve.

  python eval_entangled_probe.py --config C --manifest M --experts_dir DIR \
      --channels expert_max served_key --drop_shard 9 --surface both [--rouge] --out J
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import eval_tofu as et
from eval_tofu import _answer_avg_loss, _build_qa_prompt

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


def _load(config):
    with open(config) as f:
        return json.load(f)


def _load_experts(base_model, experts_dir, k, hf_home):
    """Scaffolded base + shard_0..k-1 adapters (named), returning (peft_model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    os.environ["HF_HOME"] = hf_home
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=True)
    model = None
    for i in range(k):
        d = os.path.join(experts_dir, f"shard_{i}")
        if not os.path.isdir(d):
            continue
        if model is None:
            model = PeftModel.from_pretrained(base, d, adapter_name=f"shard_{i}")
        else:
            model.load_adapter(d, adapter_name=f"shard_{i}")
    model.eval()
    return model, tok


def _ans_prob(model, tok, q, a):
    nll = _answer_avg_loss(model, tok, q, a)
    return float(np.exp(-nll)) if nll == nll else 0.0   # nan -> 0


def _rouge_l(model, tok, q, a):
    """(rougeL_recall, generated_text) for a greedy 64-token generation."""
    from eval_tofu import _get_rouge_metric
    device = next(model.parameters()).device
    enc = tok(_build_qa_prompt(tok, q), return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    r = _get_rouge_metric().compute(predictions=[gen], references=[a],
                                    rouge_types=["rougeL"], use_aggregator=False)
    score = float(r["rougeL"][0].recall if hasattr(r["rougeL"][0], "recall") else r["rougeL"][0])
    return score, gen


def _build_shard_centroids(cfg, k, shard_ids, device):
    """Per-shard centroids over `shard_ids` (see eval_routed_scaffold.build_shard_centroids —
    the shared builder). sibling policy passes the SURVIVING shards (deleted excluded, orphans
    fall to the nearest surviving sibling = the leak); tombstone passes ALL shards (the deleted
    centroid stays as an identity sentinel)."""
    from eval_routed_scaffold import build_shard_centroids
    return build_shard_centroids(
        cfg["hf_home"], k, shard_ids, device,
        encoder_name=cfg.get("router_encoder", "sentence-transformers/all-MiniLM-L6-v2"))


def _surfaces(fact, which):
    out = []
    if which in ("orig", "both"):
        out.append(("orig", fact["probe_question_orig"], fact["probe_answer_orig"]))
    if which in ("para", "both"):
        out.append(("para", fact["probe_question_para"], fact["probe_answer_para"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--experts_dir", required=True, help="planted or clean(oracle-B) experts dir")
    ap.add_argument("--channels", nargs="+", default=["expert_max", "served_key"],
                    choices=["expert_max", "served_key", "served_embedsim"])
    ap.add_argument("--drop_shard", default="9", help="'none' or a shard id to drop")
    ap.add_argument("--surface", default="both", choices=["orig", "para", "both"])
    ap.add_argument("--embed_policy", default="sibling",
                    choices=["sibling", "tombstone", "tombstone_author"],
                    help="served_embedsim deletion policy: 'sibling' (default, prior behavior) "
                         "drops the deleted shard's centroid so orphan probes fall to the nearest "
                         "surviving host; 'tombstone' keeps it as an identity sentinel — probes "
                         "whose top-1 is the deleted shard serve base+scaffold (route=-1); "
                         "'tombstone_author' replaces the shard sentinel with PER-AUTHOR sentinel "
                         "centroids of the dropped shard's authors (the Phase-1 winning rung, "
                         "c_probe≈0.97 vs the shard rung's 0.48-0.76). No-op when --drop_shard none.")
    ap.add_argument("--embed_strategy", default=None,
                    help="router_leak Wave-3: route served_embedsim via a router.py strategy "
                         "(centroid_sbert/centroid_lm/ppl/activation_norm/logit_div/…) instead of "
                         "the default MiniLM centroid — the per-router-family Mode-B leak arm. "
                         "Requires --embed_policy sibling (no seal).")
    ap.add_argument("--embed_abstain_tau", type=float, default=None,
                    help="router-NATIVE seal (router_leak 2026-07-23): with --embed_strategy, "
                         "abstain to base+scaffold (route −1) when the router's own best-vs-"
                         "runner-up MARGIN < tau — i.e. 'nothing fits well, treat as an orphan'. "
                         "Needs no stored identity sentinel (unlike the tombstone). Requires a "
                         "router exposing score_candidates() (ppl). Calibrate tau on RETAIN "
                         "margins only — never on forget data.")
    ap.add_argument("--rouge", action="store_true", help="also greedy-generate + ROUGE-L (slow)")
    ap.add_argument("--dump_generations", action="store_true",
                    help="with --rouge, store each generation's text in per_fact "
                         "(*_gen_* keys) for the R3 content audit")
    ap.add_argument("--max_facts", type=int, default=None, help="smoke cap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    cfg = _load(args.config)
    with open(args.manifest) as f:
        man = json.load(f)
    k = cfg["k"]
    drop = None if args.drop_shard == "none" else int(args.drop_shard)
    facts = man["facts"][:args.max_facts] if args.max_facts else man["facts"]

    model, tok = _load_experts(cfg["base_model"], args.experts_dir, k, cfg["hf_home"])
    surviving = [i for i in range(k) if i != drop and
                 os.path.isdir(os.path.join(args.experts_dir, f"shard_{i}"))]

    served = None
    if "served_key" in args.channels:
        from eval_routed_scaffold import OODAwareRoutedModel
        from legonet_tofu import build_q2author
        data_full = et.load_tofu_data(cfg["hf_home"])["full"]
        q2a = build_q2author(data_full, 200, 200 // k)
        served = OODAwareRoutedModel(model, tok, q2a, k, num_authors=200, delete_shard=drop)

    # Wave-3 per-family Mode-B: route served_embedsim through a real router.py strategy
    fam_router = None
    if args.embed_strategy:
        if args.embed_policy != "sibling":
            raise SystemExit("--embed_strategy requires --embed_policy sibling (the per-family "
                             "Mode-B leak arm has no tombstone seal)")
        from merge_lora import _build_routed_model
        data_full_fr = et.load_tofu_data(cfg["hf_home"])["full"]
        fam_router = _build_routed_model(
            model, k, args.embed_strategy,
            frozenset({drop}) if drop is not None else frozenset(),
            tokenizer=tok, dataset=data_full_fr,
            centroid_cache_dir=os.path.join(args.experts_dir, "centroids")).router

    centroids = sids = q_embed = None
    author_sents = None
    if "served_embedsim" in args.channels and fam_router is None:
        # tombstone keeps the deleted centroid in the pool as an identity sentinel;
        # sibling (prior behavior) excludes it so orphans fall to a surviving host;
        # tombstone_author uses per-author sentinel centroids of the dropped shard instead.
        pool = (sorted(set(surviving) | ({drop} if drop is not None else set()))
                if args.embed_policy == "tombstone" else surviving)
        centroids, sids, q_embed = _build_shard_centroids(cfg, k, pool, args.device)
        if args.embed_policy == "tombstone_author" and drop is not None:
            from shard_utils import get_author_shard
            data_full = et.load_tofu_data(cfg["hf_home"])["full"]
            rows_s = []
            for a in get_author_shard(k, drop):
                v = q_embed([data_full[a * 20 + w]["question"] for w in range(20)]).mean(0)
                rows_s.append(v / (np.linalg.norm(v) + 1e-12))
            author_sents = np.stack(rows_s).astype("float32")

    rows = []
    for fi, fact in enumerate(facts):
        rec = {"fact_id": fact["fact_id"], "R": fact["R"], "mode": fact["mode"]}
        for sname, q, a in _surfaces(fact, args.surface):
            if "expert_max" in args.channels:
                probs = []
                for j in surviving:
                    model.set_adapter(f"shard_{j}")
                    probs.append(_ans_prob(model, tok, q, a))
                rec[f"expert_max_prob_{sname}"] = max(probs) if probs else 0.0
                if args.rouge:
                    jbest = surviving[int(np.argmax(probs))]
                    model.set_adapter(f"shard_{jbest}")
                    score, gen = _rouge_l(model, tok, q, a)
                    rec[f"expert_max_rougeL_{sname}"] = score
                    if args.dump_generations:
                        rec[f"expert_max_gen_{sname}"] = gen
            if "served_key" in args.channels:
                rec[f"served_key_prob_{sname}"] = _ans_prob(served, tok, q, a)
                if args.rouge:
                    score, gen = _rouge_l(served, tok, q, a)
                    rec[f"served_key_rougeL_{sname}"] = score
                    if args.dump_generations:
                        rec[f"served_key_gen_{sname}"] = gen
            if "served_embedsim" in args.channels and fam_router is not None:
                from router import ActivationRouter as _AR
                rarg = (tok(q, return_tensors="pt").input_ids[0].unsqueeze(0).to(
                            next(model.parameters()).device)
                        if isinstance(fam_router, _AR) else q)
                excl = frozenset({drop}) if drop is not None else frozenset()
                abstain = False
                if args.embed_abstain_tau is not None:
                    if not hasattr(fam_router, "score_candidates"):
                        raise SystemExit(f"--embed_abstain_tau needs a router exposing "
                                         f"score_candidates(); {args.embed_strategy} does not")
                    cands, losses = fam_router.score_candidates(rarg, exclude=excl)
                    # lower loss = better fit; margin = runner-up − best (>=0). A SMALL margin
                    # means nothing fits distinctly well => orphan-like => abstain.
                    srt = sorted(losses)
                    margin = (srt[1] - srt[0]) if len(srt) > 1 else float("inf")
                    jbest = cands[int(np.argmin(losses))]
                    abstain = margin < args.embed_abstain_tau
                    rec[f"served_embedsim_margin_{sname}"] = float(margin)
                else:
                    jbest = int(fam_router.route(rarg, exclude=excl))
                if abstain:
                    with model.disable_adapter():
                        rec[f"served_embedsim_prob_{sname}"] = _ans_prob(model, tok, q, a)
                        if args.rouge:
                            score, gen = _rouge_l(model, tok, q, a)
                            rec[f"served_embedsim_rougeL_{sname}"] = score
                            if args.dump_generations:
                                rec[f"served_embedsim_gen_{sname}"] = gen
                    rec[f"served_embedsim_route_{sname}"] = -1
                else:
                    model.set_adapter(f"shard_{jbest}")
                    rec[f"served_embedsim_prob_{sname}"] = _ans_prob(model, tok, q, a)
                    if args.rouge:
                        score, gen = _rouge_l(model, tok, q, a)
                        rec[f"served_embedsim_rougeL_{sname}"] = score
                        if args.dump_generations:
                            rec[f"served_embedsim_gen_{sname}"] = gen
                    rec[f"served_embedsim_route_{sname}"] = jbest
            if "served_embedsim" in args.channels and fam_router is None:
                # route the probe question to the nearest centroid in the policy's pool
                v = q_embed([q])[0]
                if author_sents is not None:
                    scores = np.concatenate([centroids @ v, author_sents @ v])
                    tomb_hit = int(np.argmax(scores)) >= len(sids)
                    jbest = drop if tomb_hit else sids[int(np.argmax(centroids @ v))]
                else:
                    jbest = sids[int(np.argmax(centroids @ v))]
                if (args.embed_policy in ("tombstone", "tombstone_author")
                        and drop is not None and jbest == drop):
                    # identity sentinel hit -> serve base+scaffold (the sealed route)
                    with model.disable_adapter():
                        rec[f"served_embedsim_prob_{sname}"] = _ans_prob(model, tok, q, a)
                        if args.rouge:
                            score, gen = _rouge_l(model, tok, q, a)
                            rec[f"served_embedsim_rougeL_{sname}"] = score
                            if args.dump_generations:
                                rec[f"served_embedsim_gen_{sname}"] = gen
                    rec[f"served_embedsim_route_{sname}"] = -1
                else:
                    model.set_adapter(f"shard_{jbest}")
                    rec[f"served_embedsim_prob_{sname}"] = _ans_prob(model, tok, q, a)
                    if args.rouge:
                        score, gen = _rouge_l(model, tok, q, a)
                        rec[f"served_embedsim_rougeL_{sname}"] = score
                        if args.dump_generations:
                            rec[f"served_embedsim_gen_{sname}"] = gen
                    rec[f"served_embedsim_route_{sname}"] = jbest
        rows.append(rec)
        if (fi + 1) % 25 == 0:
            print(f"[probe] {fi+1}/{len(facts)} facts", flush=True)

    # aggregate per (R, mode, channel, surface, signal)
    def agg(key):
        by = {}
        for r in rows:
            if key not in r:
                continue
            g = (r["R"], r["mode"])
            by.setdefault(g, []).append(r[key])
        return {f"R{R}_{m}": {"mean": float(np.mean(v)), "n": len(v),
                              "ci95": float(1.96 * np.std(v) / np.sqrt(len(v)))}
                for (R, m), v in sorted(by.items())}

    # aggregate the numeric signal keys; *_route_* (shard ids) and *_gen_* (generated text)
    # are per-fact diagnostics, not signals to average — keep them in per_fact only.
    signal_keys = sorted({k2 for r in rows for k2 in r
                          if k2 not in ("fact_id", "R", "mode")
                          and "_route_" not in k2 and "_gen_" not in k2})
    out = {
        "experts_dir": args.experts_dir, "drop_shard": args.drop_shard,
        "surviving_experts": surviving, "n_facts": len(facts),
        "channels": args.channels, "surface": args.surface, "seed": args.seed,
        "embed_policy": args.embed_policy,
        "aggregates": {sk: agg(sk) for sk in signal_keys},
        "per_fact": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval_entangled_probe] {len(facts)} facts x {len(surviving)} experts -> {args.out}")
    for sk in signal_keys:
        if "prob" in sk:
            print(f"  {sk}: " + "  ".join(f"{g}={d['mean']:.3f}"
                  for g, d in sorted(out['aggregates'][sk].items())))


if __name__ == "__main__":
    main()
