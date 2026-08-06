"""E2 (TOFU arm) — alpha-weight diagnostics for the RouterLoRA on the TOFU author-expert pool.

WHY: same question as the DBpedia arm (does the learned alpha deviate from uniform 1/m, and
does routing sharpness track memorization?) but on the pool where unlearning actually happens:
forget10 (authors 180-199, 400 questions) vs a seeded 400-question retain sample. The router
was trained on retain authors only, so a forget-vs-retain sharpness gap is itself evidence the
router encodes author identity. With --unlearn_tag the SAME (never-retrained) router is probed
over the post-deletion expert pool.

Routing is the model's OWN key-route (`RamoleRouter.route` on the "Question:/Answer:" text) so
the captured alphas describe exactly what serving composes; NLL is the teacher-forced loss on
the full text (per-record ppl = exp(nll); TOFU has no per-record EM to join -> NaN).

--dropped (H-TRAINED drop audit, log/router_leak 2026-07-20 pre-registration): routing switches
to the model's EMBED route over the STALE expert index (the RAG path serving deploys); per
query the unlearn manifest's affected_adapters are masked OUT of the routed set before
`controller.set_active` — the softmax renormalizes over the survivors, i.e. the serving
semantics of a *drop* deletion (experts removed, router untouched) — and BOTH the masked and
the unmasked route are captured (two capture cycles per query, each under the strict
one-teacher-forced-forward-per-path contract; `ar.alpha_stats` raises on violations). The
served pool stays the FULL pool (drop ≠ retrain); `--unlearn_tag` names only the manifest
(default forget10). Output = the per-query array contract consumed by the family analyzer
(see `assemble_dropped_result`). `--router_ckpt` selects the seed-43/44 routers.

    python analyze_router_tofu.py --config configs/ramole_tofu_1b.json --device cuda \
        --out .../results/alpha_diag_tofu.json
    python analyze_router_tofu.py --config configs/ramole_tofu_1b.json --unlearn_tag forget10 \
        --device cuda --out .../results/alpha_diag_tofu_unlearn.json
    python analyze_router_tofu.py --config configs/ramole_tofu_1b.json --dropped \
        --router_ckpt .../router_s43.safetensors --device cuda --out .../rl_routerlora_s43.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

TOFU_DIR = os.path.dirname(os.path.abspath(__file__))
if TOFU_DIR not in sys.path:
    sys.path.insert(0, TOFU_DIR)
RAMOLE_DIR = os.path.join(os.path.dirname(TOFU_DIR), "ramole")
if RAMOLE_DIR not in sys.path:
    sys.path.insert(0, RAMOLE_DIR)

import legonet_tofu as lt          # noqa: E402
import ramole_tofu as rt           # noqa: E402
import analyze_router as ar        # noqa: E402  (shared E2 library, ramole/ on sys.path)


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


def _records(cfg_l, data_full, n_retain: int = 400):
    """400 forget10 rows + a seeded 400-row retain sample (RandomState(42) — reproducible and
    independent of any eval sampling). Each record: id = TOFU row index, text = the exact
    serving format, prompt = the question prefix for the completion-position mask."""
    per = cfg_l["records_per_author"]
    forget = set(int(a) for a in cfg_l["forget_authors"])
    forget_rows = [a * per + i for a in sorted(forget) for i in range(per)]
    retain_pool = np.array([r for r in range(cfg_l["num_authors"] * per)
                            if (r // per) not in forget], dtype=int)
    rng = np.random.RandomState(42)
    retain_rows = sorted(int(r) for r in
                         retain_pool[rng.choice(len(retain_pool), size=n_retain, replace=False)])
    recs = []
    for group, rows in (("forget", forget_rows), ("retain", retain_rows)):
        for r in rows:
            q, a = data_full[r]["question"], data_full[r]["answer"]
            recs.append({"id": r, "group": group,
                         "text": f"Question: {q}\nAnswer: {a}",
                         "prompt": f"Question: {q}\nAnswer:"})
    return recs


# ── H-TRAINED dropped audit (log/router_leak, 2026-07-20 pre-registration) ─────

def dropped_active(full_set, affected, index, qvec, k):
    """Post-drop active set: the manifest's affected experts are masked OUT of the routed
    set (survivors keep their slots; `controller.set_active` + softmax renormalizes over
    them). If masking empties the set, fall back to the top-k SURVIVORS by the SAME
    stale-index ranking — the serving fallback: a drop deletion cannot leave a query with
    zero experts, and serving re-ranks the survivors with the identical score rule rather
    than inventing a new one (mirrors routing_audit_tofu.rank_topk's -inf mask + stable
    argsort). Returns (active_tuple, fallback_used). `qvec` may be None when the mask does
    not empty the set (callers skip the extra encoder pass)."""
    aff = set(int(j) for j in affected)
    surv = tuple(int(j) for j in full_set if int(j) not in aff)
    if surv:
        return surv, False
    sims = np.asarray(index, dtype="float32") @ np.asarray(qvec, dtype="float32")
    sims[sorted(aff)] = -np.inf
    order = np.argsort(-sims, kind="stable")
    return tuple(int(j) for j in order[:k]), True


def _expert_mean_top1(captured: dict, active) -> float:
    """Alpha share of the record's TOP-1 expert: mean alpha per expert over every captured
    path and token position, then the max over the active experts. This is the H-TRAINED
    adequacy quantity ("top-1 surviving-expert alpha share"), deliberately distinct from
    alpha_stats' max_share (a per-position max that may hop between experts)."""
    import torch
    means = []
    for path in sorted(captured):
        (_act, alpha), = captured[path]     # single entry — ar.alpha_stats validated it
        means.append(alpha[:, 0, :].to(torch.float64).mean(dim=1))   # (m,)
    return float(torch.stack(means).mean(0).max())


def _one_captured_forward(rm, rec: dict, active: tuple, max_length: int) -> dict:
    """ONE teacher-forced b=1 forward with alpha capture over `active`. Drives the
    controller directly and forwards through rm.model (bypassing the wrapper's internal
    re-routing) — the ar.capture_for_records mechanics, single-record. The strict capture
    contract holds: ar.alpha_stats raises unless every installed path captured exactly once
    with captured active == the routed set passed here."""
    import torch
    model, tok, ctrl = rm.model, rm.tokenizer, rm.controller
    device = next(model.parameters()).device
    active = [int(j) for j in active]
    ctrl.set_active(active)
    enc = tok(rec["text"], return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids.to(device)
    p_len = int(tok(rec["prompt"], return_tensors="pt", truncation=True,
                    max_length=max_length).input_ids.shape[1])
    p_len = max(0, min(p_len, input_ids.shape[1] - 1))
    ctrl.captured.clear()
    ctrl.capture_alpha = True
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=enc.attention_mask.to(device), labels=input_ids)
    finally:
        ctrl.capture_alpha = False
    stats = ar.alpha_stats(ctrl.captured, active, None, prompt_len=p_len)
    top1 = _expert_mean_top1(ctrl.captured, active)
    ctrl.captured.clear()    # free the raw (m,1,l) tensors before the next capture cycle
    return {"h_norm": stats["H_norm_mean"], "max_share": stats["max_share_mean"],
            "top1_share": top1, "nll": float(out.loss), "active": active}


def assemble_dropped_result(rows: list) -> dict:
    """The H-TRAINED JSON contract (consumed by the family analyzer): parallel per-query
    arrays + pooled orphan-vs-retain group means + the two detector AUCs computed with
    routing_audit_tofu._auc. Detector score directions are fixed A PRIORI (the
    analyze_router_leak convention — never fitted post hoc): h_norm as-is (orphans predicted
    MORE uniform post-drop), max_share NEGATED (lower routing confidence = more orphan-like);
    pos = forget, neg = retain. NB records whose post-drop set collapsed to m=1 carry the
    alpha_stats m==1 convention (h_norm = 1.0, max_share = 1.0) — filter on n_active."""
    from routing_audit_tofu import _auc
    is_forget = np.asarray([bool(r["is_forget"]) for r in rows])
    cols = {
        "h_norm": [r["drop"]["h_norm"] for r in rows],
        "max_share": [r["drop"]["max_share"] for r in rows],
        "top1_share": [r["drop"]["top1_share"] for r in rows],
        "top1_share_full": [r["full"]["top1_share"] for r in rows],
        "h_norm_full": [r["full"]["h_norm"] for r in rows],
        "max_share_full": [r["full"]["max_share"] for r in rows],
        "nll": [r["drop"]["nll"] for r in rows],
        "nll_full": [r["full"]["nll"] for r in rows],
    }
    out = {k: [float(v) for v in vs] for k, vs in cols.items()}
    out["n_active"] = [len(r["drop"]["active"]) for r in rows]
    out["n_active_full"] = [len(r["full"]["active"]) for r in rows]
    out["is_forget"] = [bool(b) for b in is_forget]
    out["author_of_q"] = [int(r["author"]) for r in rows]
    out["query_id"] = [int(r["id"]) for r in rows]
    out["fallback_used"] = [bool(r["fallback"]) for r in rows]

    # pre-drop share is a softmax mean over >=1 expert -> strictly positive; guard anyway
    ratio = (np.asarray(cols["top1_share"], dtype="float64")
             / np.maximum(np.asarray(cols["top1_share_full"], dtype="float64"), 1e-12))
    groups = {}
    for gname, mask in (("forget", is_forget), ("retain", ~is_forget)):
        idx = np.nonzero(mask)[0]
        g = {"n": int(idx.size)}
        for key in ("h_norm", "max_share", "top1_share", "top1_share_full",
                    "h_norm_full", "max_share_full", "nll", "nll_full"):
            g[f"{key}_mean"] = (float(np.asarray(cols[key], dtype="float64")[idx].mean())
                                if idx.size else float("nan"))
        g["top1_share_ratio_mean"] = float(ratio[idx].mean()) if idx.size else float("nan")
        g["fallback_rate"] = (float(np.mean([bool(rows[i]["fallback"]) for i in idx]))
                              if idx.size else float("nan"))
        groups[gname] = g
    out["groups"] = groups

    h = np.asarray(cols["h_norm"], dtype="float64")
    ms = np.asarray(cols["max_share"], dtype="float64")
    f, r = is_forget, ~is_forget
    out["auc_h_norm"] = _auc(h[f], h[r]) if f.any() and r.any() else float("nan")
    out["auc_max_share"] = _auc(-ms[f], -ms[r]) if f.any() and r.any() else float("nan")
    out["auc_note"] = ("scores oriented a priori: h_norm as-is (orphan = more uniform), "
                       "max_share negated (orphan = less confident); pos=forget, neg=retain")
    return out


def run_dropped_audit(cfg_l: dict, data_full, router_ckpt: str, tag: str) -> dict:
    """H-TRAINED: embed-route every query over the STALE index, capture the unmasked route
    AND the affected-masked route (two capture cycles per query), assemble the contract."""
    import hashlib

    model, _tok = rt.load_ramole_eval_model(cfg_l, data_full, router_ckpt, route_mode="embed")
    with open(lt.unlearn_manifest_path(cfg_l, tag)) as f:
        manifest = json.load(f)
    affected = sorted(int(j) for j in manifest["affected_adapters"])
    aff_set = set(affected)
    n, k = cfg_l["n"], cfg_l["k"]
    survivors = [j for j in range(n) if j not in aff_set]
    if not survivors:
        raise RuntimeError(f"dropped audit: manifest[{tag}] masks every expert "
                           f"(affected={len(affected)}, n={n}) — no survivor to serve")
    fk = min(k, len(survivors))   # fallback width can never exceed the survivor count
    max_length = cfg_l["ramole_train"]["max_length"]
    per = cfg_l["records_per_author"]
    recs = _records(cfg_l, data_full)   # the frozen 400 forget + RandomState(42) 400 retain

    rows = []
    for r in recs:
        full_set = model.router.route(r["text"])       # serving's own embed route (stale index)
        needs_fb = all(int(j) in aff_set for j in full_set)
        qv = None
        if needs_fb:   # only then is the extra encoder pass needed (route() embeds internally)
            q = lt.parse_question(r["text"])
            qv = model.router.qembed([q if q is not None else r["text"]])[0]
        drop_set, fb = dropped_active(full_set, aff_set, model.router.index, qv, fk)
        s_full = _one_captured_forward(model, r, tuple(int(j) for j in full_set), max_length)
        s_drop = _one_captured_forward(model, r, drop_set, max_length)
        rows.append({"id": r["id"], "is_forget": r["group"] == "forget",
                     "author": r["id"] // per, "fallback": fb,
                     "full": s_full, "drop": s_drop})

    result = assemble_dropped_result(rows)
    stale_path = rt.expert_index_path(cfg_l)
    with open(stale_path, "rb") as f:
        stale_sha = hashlib.sha256(f.read()).hexdigest()
    result.update({
        "mode": "dropped", "route": "embed", "router_ckpt": router_ckpt,
        "unlearn_tag": tag, "manifest": lt.unlearn_manifest_path(cfg_l, tag),
        "affected_adapters": affected, "n": int(n), "k": int(k),
        "n_survivors": len(survivors), "fallback_k": int(fk),
        "n_records": len(rows), "seed": cfg_l.get("base_seed", 42),
        "name": cfg_l.get("name", "") + f"_dropped_{tag}",
        "stale_index_path": stale_path, "stale_index_sha256": stale_sha,
    })
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/ramole_tofu_1b.json")
    ap.add_argument("--unlearn_tag", default=None, help="e.g. forget10 (post-deletion pool; "
                    "with --dropped it names ONLY the manifest — the served pool stays full)")
    ap.add_argument("--router_ckpt", default=None, help="default: the run's router.safetensors "
                    "(H-TRAINED seeds: pass router_s43/router_s44.safetensors)")
    ap.add_argument("--dropped", action="store_true",
                    help="H-TRAINED drop audit: EMBED route over the STALE index, mask the "
                         "unlearn manifest's affected_adapters out of the active set before "
                         "controller.set_active (drop-deletion serving semantics; softmax "
                         "renormalizes over survivors), capture masked AND unmasked forwards "
                         "per query; manifest tag = --unlearn_tag or 'forget10'")
    ap.add_argument("--device", default="cuda",
                    help="recorded; the LM device follows torch.cuda.is_available() "
                         "(load_ramole_eval_model), the key-route encoder runs on CPU")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_l = lt.load_config(args.config)
    os.environ["HF_HOME"] = cfg_l["hf_home"]
    import torch
    torch.manual_seed(cfg_l["base_seed"])
    from datasets import load_dataset
    data_full = load_dataset("locuslab/TOFU", "full")["train"]

    router_ckpt = args.router_ckpt or rt.router_path(cfg_l)

    if args.dropped:
        # H-TRAINED branch — everything below this block is the byte-identical legacy path.
        import ramole_common as rc
        tag = args.unlearn_tag or "forget10"
        result = run_dropped_audit(cfg_l, data_full, router_ckpt, tag)
        result["config"] = args.config
        rc.write_json(args.out, result)
        print(f"[analyze_router_tofu] dropped tag={tag} "
              f"affected={len(result['affected_adapters'])}/{result['n']} "
              f"survivors={result['n_survivors']} fallback_k={result['fallback_k']}")
        print(f"[analyze_router_tofu] groups={json.dumps(result['groups'])}")
        print(f"[analyze_router_tofu] AUC(h_norm)={result['auc_h_norm']:.3f} "
              f"AUC(max_share)={result['auc_max_share']:.3f} -> {args.out}")
        return

    model, _tok = rt.load_ramole_eval_model(cfg_l, data_full, router_ckpt, route_mode="key",
                                            unlearn_tag=args.unlearn_tag)
    with open(lt.assignment_path(cfg_l)) as f:
        assignment = json.load(f)

    per = cfg_l["records_per_author"]
    recs = _records(cfg_l, data_full)
    # active set from the model's OWN router on the serving text, BEFORE the forward; ideal =
    # the author's first assigned expert (the frozen top-1 key).
    sets = {r["id"]: model.router.route(r["text"]) for r in recs}
    ideals = {r["id"]: int(lt.author_keys(assignment, r["id"] // per)[0]) for r in recs}

    stats = list(ar.capture_for_records(
        model, recs, sets, cfg_l["ramole_train"]["max_length"],
        lambda r: (r["text"], r["prompt"]), ideals=ideals))
    group_of = {r["id"]: r["group"] for r in recs}
    for s in stats:
        s["em"] = float("nan")                    # no per-record EM eval on TOFU
        s["ppl"] = float(np.exp(s["nll"]))        # teacher-forced ppl from the captured forward

    groups = {}
    for g in ("forget", "retain"):
        gs = [s for s in stats if group_of[s["id"]] == g]
        present = [s["ideal_present"] for s in gs if s["ideal_present"] is not None]
        groups[g] = {
            "n": len(gs),
            "H_norm_mean": ar._nanmean([s["H_norm_mean"] for s in gs]),
            "max_share_mean": ar._nanmean([s["max_share_mean"] for s in gs]),
            "ideal_mass_mean": ar._nanmean([s["ideal_mass_mean"] for s in gs]),
            "nll_mean": ar._nanmean([s["nll"] for s in gs]),
            "ppl_mean": ar._nanmean([s["ppl"] for s in gs]),
            "ideal_present_rate": (float(np.mean(present)) if present else float("nan")),
        }

    result = ar.assemble_result(
        stats, config=args.config, router_ckpt=router_ckpt, route="key", m=cfg_l["k"],
        extras={"name": cfg_l["name"] + (f"_unlearn_{args.unlearn_tag}" if args.unlearn_tag else ""),
                "dropout_p": cfg_l["ramole_train"].get("dropout_p"),
                "seed": cfg_l["base_seed"], "unlearn_tag": args.unlearn_tag,
                "groups": groups})
    for s in result["per_record"]:
        s["group"] = group_of[s["id"]]
    import ramole_common as rc
    rc.write_json(args.out, result)
    print(f"[analyze_router_tofu] pooled={json.dumps(result['pooled'])}")
    print(f"[analyze_router_tofu] groups={json.dumps(groups)}")
    print(f"[analyze_router_tofu] -> {args.out}")


if __name__ == "__main__":
    main()
