"""Router-leak (b): do the Group-B mask/merge methods inherit the orphan leak under a
REALISTIC embedding selector, and does misrouting actually leak DELETED content?

Two modes:
  --mode routing (CPU/light, no LLM): build MiniLM centroids over each method's SURVIVING
     selection units (SIFT/MemSinks: per-author; ClAMU: retain feature clusters), route the
     400 deleted-author questions among survivors, report the destination histogram +
     concentration + confidence-inseparability (AUC) + author-sentinel tombstone catch/FPR.
     Confirms H-GB1 (they inherit the routing leak); per-author ≡ the k=200 centroid_sbert
     cell (cross-check).
  --mode serve (GPU): serve the method's *_unlearn model under {oracle, realistic} selector
     on the 400 deleted-author questions; report answer-prob + ROUGE vs the deleted gold
     (leak) and vs base generation (confabulation) + route stats. The H-GB2 test: exact-
     subtraction methods (SIFT/ClAMU) should NOT leak deleted content despite full
     misrouting (τ_u already subtracted from τ̄); MemSinks (masking) is the H-GB3 contrast.

  python orphan_route_groupb.py --method sift --mode routing --out J
  python orphan_route_groupb.py --method sift --mode serve --unlearn_tag forget10 [--max_q 40] --out J
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


METHODS = {   # method -> (config, loader_module, loader_fn, unit_type)
    "sift": ("configs/sift_masks_tofu_1b.json", "sift_masks_model", "load_sift_eval_model", "author"),
    "clamu": ("configs/clamu_tofu_1b.json", "clamu_model", "load_clamu_eval_model", "cluster"),
    "memsinks": (os.path.join(os.environ.get("MEMSINKS_TOFU_DIR", os.path.join(_REPO_ROOT, "memsinks_tofu")), "configs", "memsinks_tofu_1b_strict2_e25.json"),
                 "memsinks_routed_model", "load_memsinks_eval_model", "author"),
}


def _load_cfg(path):
    with open(path) as f:
        return json.load(f)


def _forget_authors(cfg, tag):
    """Group-B configs carry deletion sets under unlearn_tags[tag] (not a top-level key)."""
    ut = cfg.get("unlearn_tags", {})
    if tag not in ut:
        raise SystemExit(f"tag {tag!r} not in cfg unlearn_tags {list(ut)}")
    return sorted(int(a) for a in ut[tag])


def _surviving_units(method, cfg, forget_authors):
    """(unit_to_authors, unit_repr_author) over SURVIVING units for this method."""
    from groupb_realistic_router import per_author_units, clamu_cluster_units
    if method == "clamu":
        import clamu as cl
        ap = cl.assignment_path(cfg, "forget10")     # retain re-clustered
        with open(ap) as f:
            assignment = json.load(f)
        return clamu_cluster_units(assignment)
    u = per_author_units(cfg["num_authors"], forget_authors)
    return u, {a: a for a in u}


def _auc(pos, neg):
    from routing_audit_tofu import _auc as f
    return f(np.asarray(pos, "float64"), np.asarray(neg, "float64"))


def run_routing(args):
    from eval_routed_scaffold import build_unit_centroids
    import eval_tofu as et
    cfg = _load_cfg(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    forget = _forget_authors(cfg, args.unlearn_tag)
    per = cfg["records_per_author"]
    data_full = et.load_tofu_data(cfg["hf_home"])["full"]

    unit_to_authors, _ = _surviving_units(args.method, cfg, forget)
    cents, uids, embed = build_unit_centroids(cfg["hf_home"], unit_to_authors, args.device)

    forget_rows = [a * per + w for a in forget for w in range(per)]
    retain_authors = [a for a in range(cfg["num_authors"]) if a not in set(forget)]
    rng = np.random.RandomState(args.seed)
    ret_sample = sorted(rng.choice(retain_authors, size=min(20, len(retain_authors)), replace=False))
    retain_rows = [a * per + w for a in ret_sample for w in range(per)]

    Qf = embed([data_full[i]["question"] for i in forget_rows])
    Qr = embed([data_full[i]["question"] for i in retain_rows])
    sf, sr = Qf @ cents.T, Qr @ cents.T
    dest = sf.argmax(1)
    hist = {str(uids[j]): int((dest == j).sum()) for j in range(len(uids)) if (dest == j).any()}

    # confidence inseparability: top-1 sim, orphan (forget) vs retain
    conf_auc = _auc(-sf.max(1), -sr.max(1))   # orphans "want" lower confidence; AUC->detectable

    # author-sentinel tombstone: per-forget-author sentinels + surviving unit centroids
    sent, _sids, _ = build_unit_centroids(cfg["hf_home"], {a: [a] for a in forget},
                                          args.device, embed=embed)
    pool = np.concatenate([cents, sent], axis=0)
    n_surv_units = len(uids)
    tf = Qf @ pool.T
    tomb_hit_f = tf.argmax(1) >= n_surv_units
    tr = Qr @ pool.T
    tomb_hit_r = tr.argmax(1) >= n_surv_units

    from analyze_orphan_destinations import concentration
    conc = concentration(hist, len(uids))
    out = {
        "method": args.method, "mode": "routing", "unit_type": METHODS[args.method][3],
        "n_units_surviving": len(uids), "n_orphans": len(forget_rows),
        "orphan_dest_hist": hist, "concentration": conc,
        "orphan_misroute_rate": 1.0,   # every orphan lands on a surviving unit (no self)
        "confidence_auc": conf_auc,
        "tombstone_author_catch": float(tomb_hit_f.mean()),
        "tombstone_author_fpr": float(tomb_hit_r.mean()),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[groupb routing/{args.method}] units={len(uids)} orphans={len(forget_rows)} "
          f"max_share={conc['max_share']:.2f} n_eff={conc['n_eff']:.1f} "
          f"conf_auc={conf_auc:.3f} tomb catch/fpr={out['tombstone_author_catch']:.3f}/"
          f"{out['tombstone_author_fpr']:.3f} -> {args.out}")


def run_serve(args):
    import torch
    import eval_tofu as et
    from eval_tofu import _answer_avg_loss, _build_qa_prompt, _get_rouge_metric
    from groupb_realistic_router import attach_realistic_router
    import importlib
    cfg = _load_cfg(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    forget = _forget_authors(cfg, args.unlearn_tag)
    per = cfg["records_per_author"]
    data_full = et.load_tofu_data(cfg["hf_home"])["full"]

    _, mod_name, fn_name, _ = METHODS[args.method]
    loader = getattr(importlib.import_module(mod_name), fn_name)
    # loader signatures differ slightly; all accept (cfg, data_full, unlearn_tag=...)
    if args.method == "clamu":
        model, tok = loader(cfg, data_full, "clamu_unlearn", unlearn_tag=args.unlearn_tag)
    else:
        model, tok = loader(cfg, data_full, unlearn_tag=args.unlearn_tag)

    def _ans_prob(q, a):
        nll = _answer_avg_loss(model, tok, q, a)
        return float(np.exp(-nll)) if nll == nll else 0.0

    def _gen(q):
        dev = next(model.parameters()).device
        enc = tok(_build_qa_prompt(tok, q), return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=64, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def _rouge(pred, ref):
        r = _get_rouge_metric().compute(predictions=[pred], references=[ref],
                                        rouge_types=["rougeL"], use_aggregator=False)
        return float(r["rougeL"][0].recall if hasattr(r["rougeL"][0], "recall") else r["rougeL"][0])

    import types as _types
    forget_rows = [a * per + w for a in forget for w in range(per)]
    if args.max_q:
        forget_rows = forget_rows[:args.max_q]

    # THREE arms on the deleted authors' questions:
    #   oracle    — model as loaded (forget author -> its correct deletion serving = the floor)
    #   base      — force route to None (θ0 base+scaffold): the confabulation reference
    #   realistic — MiniLM selector misroutes to a surviving unit's mask (the leak under test)
    # answer-prob is the FAITHFUL leak metric (recall of the GOLD answer tokens; a real leak
    # -> ~ceiling). ROUGE-vs-gold is style-confounded (biography FORM overlaps); we separate
    # disclosure from style-confabulation with rougeL_vs_base (how base-like the answer is).
    rows = {"oracle": [], "base": [], "realistic": []}
    for i in forget_rows:
        q, a = data_full[i]["question"], data_full[i]["answer"]
        rows["oracle"].append({"prob": _ans_prob(q, a), "gen": _gen(q), "gold": a})

    orig_route = model._route                      # base arm: θ0 for every query
    model._route = _types.MethodType(lambda self, text: None, model)
    for i in forget_rows:
        q, a = data_full[i]["question"], data_full[i]["answer"]
        rows["base"].append({"prob": _ans_prob(q, a), "gen": _gen(q)})
    model._route = orig_route

    unit_to_authors, unit_repr = _surviving_units(args.method, cfg, forget)
    attach_realistic_router(model, cfg["hf_home"], unit_to_authors, args.device,
                            unit_repr_author=unit_repr)
    for k, i in enumerate(forget_rows):
        q, a = data_full[i]["question"], data_full[i]["answer"]
        gen = _gen(q)
        rows["realistic"].append({"prob": _ans_prob(q, a), "gen": gen,
                                  "rougeL_vs_gold": _rouge(gen, a),
                                  "rougeL_vs_base": _rouge(gen, rows["base"][k]["gen"])})

    def _m(arm, fn):
        return float(np.mean([fn(r) for r in rows[arm]]))
    o_rougeL = _m("oracle", lambda r: _rouge(r["gen"], r["gold"]))
    base_vs_gold = float(np.mean([_rouge(rows["base"][k]["gen"], rows["oracle"][k]["gold"])
                                  for k in range(len(forget_rows))]))
    # confabulation rate: realistic answer overlaps neither the gold nor the base answer
    confab = float(np.mean([1.0 for k in range(len(forget_rows))
                            if rows["realistic"][k]["rougeL_vs_gold"] < 0.5
                            and rows["realistic"][k]["rougeL_vs_base"] < 0.5]) if forget_rows else 0.0)
    confab_n = sum(1 for r in rows["realistic"] if r["rougeL_vs_gold"] < 0.5 and r["rougeL_vs_base"] < 0.5)

    out = {
        "method": args.method, "mode": "serve", "unlearn_tag": args.unlearn_tag,
        "n_forget_q": len(forget_rows),
        "oracle": {"forget_prob": _m("oracle", lambda r: r["prob"]), "forget_rougeL_vs_gold": o_rougeL},
        "base_theta0": {"forget_prob": _m("base", lambda r: r["prob"]), "rougeL_vs_gold": base_vs_gold},
        "realistic": {"forget_prob": _m("realistic", lambda r: r["prob"]),
                      "forget_rougeL_vs_gold": _m("realistic", lambda r: r["rougeL_vs_gold"]),
                      "rougeL_vs_base": _m("realistic", lambda r: r["rougeL_vs_base"]),
                      "confabulation_rate": confab_n / max(len(forget_rows), 1),
                      "route_stats": model.route_stats},
        "per_q": [{k2: r[k2] for k2 in r} for r in rows["realistic"]],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    r = out["realistic"]
    print(f"[groupb serve/{args.method}] LEAK METRIC forget_prob: oracle "
          f"{out['oracle']['forget_prob']:.3f} / base {out['base_theta0']['forget_prob']:.3f} / "
          f"realistic {r['forget_prob']:.3f}  |  rougeL vs_gold {r['forget_rougeL_vs_gold']:.3f} "
          f"vs_base {r['rougeL_vs_base']:.3f}  confab {r['confabulation_rate']:.2f}  "
          f"(misrouted {model.route_stats['orphan_misrouted']}/{len(forget_rows)}) -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--mode", required=True, choices=["routing", "serve"])
    ap.add_argument("--config", default=None, help="override the default method config")
    ap.add_argument("--unlearn_tag", default="forget10")
    ap.add_argument("--max_q", type=int, default=None, help="serve-mode smoke cap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.config = args.config or METHODS[args.method][0]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    (run_routing if args.mode == "routing" else run_serve)(args)


if __name__ == "__main__":
    main()
