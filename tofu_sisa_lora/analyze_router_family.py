
"""All-router leak-table assembly (router_leak Phase 3 CPU consumer) — login-safe, numpy only.

Pre-registration: log/router_leak/2026-07-20_all-router-sweep-preregistration.md. Consumes
(every input optional — whatever exists is processed, the rest is listed under
meta.missing rather than failing):

  --family_json      router_family_audit.py aggregate JSONs (rl_family_*.json); each strategy
                     entry names its sidecar (THE FAMILY NPZ CONTRACT) under "npz".
  --family_npz       extra sidecars not referenced by any aggregate JSON (analyzed npz-only —
                     no cross-assert is possible without the aggregate).
  --routerlora_json  analyze_router_tofu.py --dropped JSONs (H-TRAINED, seeds 42/43/44 —
                     the assemble_dropped_result per-query array contract).
  --dbpedia_json     ramole/routing_audit.py dropped/abstain JSONs (H-DATASET).
  --enc_json         routing_audit_tofu.py --centroid_mode JSONs (H-ENC mpnet/bge).
  --enc_roc_json     analyze_router_leak.py `roc` JSONs over the rl_enc_*.sims.npz
                     sidecars (H-ENC confidence half: max confidence-family AUC per
                     encoder, tomb_* rungs excluded).
  --sepmlp_leak_npz  SEPMLP LEAK-PROBE NPZ CONTRACT (post-droplist probe) and
  --sepmlp_ref_npz   its no-droplist reference (retain top_surv_author shift baseline).

Per family npz x drop set (scores are always "higher = more likely routed there"; ppl rows
store NEGATIVE loss, so one code path serves every graded family):
  * detector ROC on POST-DROP SURVIVOR-RESTRICTED scores — global_top1 (NEGATED top-1),
    margin (NEGATED top1-top2), per_shard_z (top-1 z vs that shard's calibration-retain
    top-1 distribution, fit only where >=5 calib rows, NEGATED), tomb_author (best
    dropped-author sentinel minus best survivor, as-is) — directions fixed A PRIORI per the
    analyze_router_leak convention, never fitted post hoc.
  * AUTHOR-PARITY split exactly like analyze_router_leak.roc: even author ids calibrate,
    odd evaluate; every reported detector/abstain metric is eval-half only (queries within
    an author are 20 correlated siblings — a query-level split would leak identity).
  * abstain tau-sweep on the post-drop survivor top-1 score: tau at eval-half RETAIN
    percentiles {1,5,10} plus the 90/99% orphan-catch operating points (those taus are
    DIAGNOSTIC — they read the orphan quantile; no percentile tau is ever calibrated on
    forget rows).
  * key_exact ships no graded score: its cell reports the binary no-match operating point
    (orphan/retain no-match post-drop + implied AUC) and is EXCLUDED from graded-AUC
    aggregation (pre-registration design note iv).
  * logit_div cells read the recomputed scores__d<ids> matrices (NaN at dropped columns —
    survivor slice asserted finite), never a column-mask of the full matrix (note iii).
  * AUTHOR-BLOCKED bootstrap 95% CIs (resample authors, not queries; RandomState(seed),
    fresh per statistic so results are glob-order independent) for sibling-adequacy mean,
    retain shift, and orphan top-3 capture.
  * cross-asserts: capture/adequacy/shift recomputed here (independent numpy mirror of
    aggregate_strategy_cells) must match the aggregate JSON within 1e-6 — HARD FAIL
    (DriftError) on drift, because a drifted npz/json pair means mismatched inputs.
  * continuity WARNs (never fail): centroid_sbert_q k=10 d9 vs rl_centroid_k10
    (adequacy 0.971 +- 0.01, retain shift 0.0583 +- 0.01); key_exact full-pool routing
    accuracy ~0.86 (recomputed from the match matrix).

  python analyze_router_family.py --self_test
  python analyze_router_family.py \
      --family_json '${TOFU_CKPT_ROOT}/.../results/rl_family_*.json' \
      --routerlora_json '${TOFU_CKPT_ROOT}/.../results/rl_routerlora_*.json' \
      --dbpedia_json os.path.join(os.environ["TOFU_CKPT_STORE"], "ramole", "runs", "*", "results", "rl_*audit*.json") \
      --enc_json '${TOFU_CKPT_ROOT}/.../results/rl_enc_*.json' \
      --sepmlp_leak_npz .../selectivity_forget10.leak.npz \
      --sepmlp_ref_npz .../selectivity_ref.leak.npz \
      --out_json reports/rl_family_leak_analysis.json \
      --out_md reports/rl_family_leak_table.md
"""
from __future__ import annotations
import os

import argparse
import glob as globlib
import json
import re
import sys
import tempfile

import numpy as np


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

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

TOFU_DIR = os.path.dirname(os.path.abspath(__file__))
if TOFU_DIR not in sys.path:
    sys.path.insert(0, TOFU_DIR)

FEATURE9 = ("key_tfidf", "centroid_sbert", "centroid_sbert_q", "centroid_lm",
            "centroid_lm_last", "ppl", "activation_norm", "attn_norm", "logit_div")
CONFIDENCE_DETECTORS = ("global_top1", "margin", "per_shard_z")
# analyze_router_leak.py roc naming: the router's NATIVE confidence family. tomb_* rungs
# are the identity-SEAL detectors, not confidence — excluded from the H-ENC bar a priori.
ENC_CONF_DETECTORS = ("global_top1", "per_expert", "margin", "knn_density")

# Prior measurements typed in as constants (NOT re-run; labels say so in the table).
PRIOR_ROWS = [
    {"router": "embed instructor-xl (base-pin)", "pool": "n=32 legonet", "cell": "forget10",
     "adequacy": 0.980, "retain_shift": 0.727,
     "source": "prior measurement 2026-07-06/07-18 (typed constant)"},
    {"router": "embed instructor-xl (FT)", "pool": "n=32 legonet", "cell": "forget10",
     "adequacy": 0.768, "retain_shift": None,
     "source": "prior measurement 2026-07-18 (typed constant)"},
    {"router": "centroid MiniLM (rl_centroid_k10)", "pool": "k=10", "cell": "d9",
     "adequacy": 0.971, "retain_shift": 0.0583,
     "source": "prior measurement 2026-07-18 (typed constant; continuity anchor)"},
    {"router": "MiniLM tombstone (author rung)", "pool": "k=10", "cell": "d9",
     "tomb_catch": 0.963, "tomb_fpr": 0.091,
     "source": "prior measurement 2026-07-18 (typed constant; H-SEAL-GEN anchor)"},
]
CONTINUITY = {"csq_adequacy": 0.971, "csq_shift": 0.0583, "tol": 0.01,
              "key_exact_acc": 0.86, "key_exact_tol": 0.02}


class DriftError(AssertionError):
    """npz-recomputed aggregate disagrees with the family JSON — mismatched inputs."""


def _auc(pos, neg) -> float:
    # the single shared midrank ROC-AUC implementation (per the plan: import, don't fork)
    from routing_audit_tofu import _auc as f
    return float(f(np.asarray(pos, "float64"), np.asarray(neg, "float64")))


# ── pure mirrors of router_family_audit's aggregation (independent recompute) ─────

def cell_key(ids) -> str:
    return "d" + "_".join(str(int(i)) for i in ids)


def strategy_family(strategy: str):
    if strategy in ("key_tfidf", "centroid_sbert", "centroid_sbert_q",
                    "centroid_lm", "centroid_lm_last"):
        return "cos"
    if strategy == "ppl":
        return "ppl"
    if strategy in ("activation_norm", "attn_norm", "logit_div"):
        return "norm"
    return None


def masked_top1(scores: np.ndarray, survivors: list) -> np.ndarray:
    survivors = [int(j) for j in survivors]
    assert survivors, "masked_top1: empty survivor set"
    sub = np.asarray(scores)[:, survivors]
    return np.asarray(survivors, dtype=int)[np.argmax(sub, axis=1)]


def key_exact_routes(match: np.ndarray, candidates: list):
    """KeyRouter serving semantics: first matching candidate in ascending-id order; no
    match -> candidates[0] fallback with the no_match flag (the identity signal)."""
    candidates = [int(j) for j in candidates]
    assert candidates, "key_exact_routes: empty candidate set"
    m = np.asarray(match, dtype=bool)[:, candidates]
    has = m.any(axis=1)
    first = np.argmax(m, axis=1)
    routes = np.full(m.shape[0], candidates[0], dtype=int)
    routes[has] = np.asarray(candidates, dtype=int)[first[has]]
    return routes, ~has


def ranked_shares(top1_ids: np.ndarray, survivors: list, k: int) -> dict:
    counts = np.bincount(np.asarray(top1_ids, dtype=int), minlength=k).astype("float64")
    total = counts.sum()
    if total == 0:
        return {"n": 0, "top1_share": None, "top3_share": None, "entropy_norm": None}
    p = counts[counts > 0] / total
    ent = float(-(p * np.log(p)).sum() / np.log(max(len(survivors), 2)))
    ranked = np.sort(counts)[::-1] / total
    return {"n": int(total), "top1_share": float(ranked[0]),
            "top3_share": float(ranked[:3].sum()), "entropy_norm": ent}


def adequacy_ratio(family: str, unmasked_top1: np.ndarray, masked_top1_scores: np.ndarray):
    u = np.asarray(unmasked_top1, dtype="float64")
    m = np.asarray(masked_top1_scores, dtype="float64")
    if family == "ppl":
        ratio = (-u) / np.maximum(-m, 1e-12)      # scores store NEGATIVE question loss
        definition = "unmasked_top1_loss / masked_top1_loss"
    elif family == "cos":
        ratio = m / np.maximum(u, 1e-12)
        definition = "masked_top1_cos / unmasked_top1_cos"
    elif family == "norm":
        ratio = m / np.maximum(u, 1e-12)
        definition = "masked_top1_score / unmasked_top1_score"
    else:
        raise ValueError(f"no adequacy ratio for family {family!r}")
    return ratio, definition


# ── detector / abstain conventions (analyze_router_leak lineage) ─────────────────

def split_by_author(authors: np.ndarray):
    """Deterministic author-parity split: even ids calibrate, odd ids evaluate."""
    authors = np.asarray(authors, dtype=int)
    return (authors % 2 == 0), (authors % 2 == 1)


def fpr_at_catch(pos: np.ndarray, neg: np.ndarray, catch: float = 0.90) -> dict:
    tau = float(np.quantile(pos, 1.0 - catch))
    return {"tau": tau, "orphan_catch": float((pos >= tau).mean()),
            "retain_fpr": float((neg >= tau).mean())}


def graded_detectors(S: np.ndarray, r_calib_mask: np.ndarray, tomb: np.ndarray = None):
    """Per-row detector scores over the survivor-restricted matrix S (float64, [n_q,
    n_surv]). Directions fixed a priori: confidence detectors NEGATED (low routing
    confidence = orphan-like scores HIGH); tomb_author as-is (sentinel beats survivor =
    orphan-like). Returns ({name: scores}, top1) — top1 also feeds the abstain sweep."""
    order = np.sort(S, axis=1)[:, ::-1]
    top1 = order[:, 0]
    top1_col = np.argmax(S, axis=1)
    det = {"global_top1": -top1}
    if S.shape[1] >= 2:
        det["margin"] = -(order[:, 0] - order[:, 1])
    mu = np.zeros(S.shape[1])
    sd = np.ones(S.shape[1])
    for e in range(S.shape[1]):
        sel = r_calib_mask & (top1_col == e)
        if sel.sum() >= 5:                        # analyze_router_leak's fit floor
            mu[e], sd[e] = top1[sel].mean(), max(top1[sel].std(), 1e-6)
    det["per_shard_z"] = -(top1 - mu[top1_col]) / sd[top1_col]
    if tomb is not None:
        det["tomb_author"] = tomb - top1
    return det, top1


def abstain_block(f_top1, r_top1, pcts=(1, 5, 10), catches=(0.90, 0.99)) -> dict:
    """ramole _abstain_block shape. by_pct taus come from RETAIN percentiles only (forget
    rows never set a percentile tau); by_orphan_catch taus are diagnostic operating points
    (the retain cost of actually catching that many orphans). Scores here are the
    POST-DROP survivor-restricted top-1 (the confidence serving would consult)."""
    f_top1 = np.asarray(f_top1, dtype="float64")
    r_top1 = np.asarray(r_top1, dtype="float64")
    out = {"n_orphans": int(f_top1.shape[0]), "n_retain": int(r_top1.shape[0]),
           "score": "post-drop survivor-restricted top-1 (eval half)",
           "by_pct": {}, "by_orphan_catch": {}}
    if not f_top1.size or not r_top1.size:
        return out
    for p in pcts:
        tau = float(np.percentile(r_top1, p))
        out["by_pct"][str(p)] = {
            "tau": tau,
            "orphan_abstain_rate": float((f_top1 < tau).mean()),
            "retain_false_abstain_rate": float((r_top1 < tau).mean()),
            "orphan_sibling_rate_if_no_abstain": float((f_top1 >= tau).mean()),
        }
    for c in catches:
        tau = float(np.percentile(f_top1, 100.0 * c))
        out["by_orphan_catch"][f"{c:.2f}"] = {
            "tau": tau,
            "orphan_abstain_rate": float((f_top1 < tau).mean()),
            "retain_false_abstain_rate": float((r_top1 < tau).mean()),
        }
    return out


# ── author-blocked bootstrap ─────────────────────────────────────────────────────

def _boot_draws(n_authors: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, n_authors, size=(n_boot, n_authors))


def blocked_mean_ci(values, authors, n_boot: int, seed: int):
    """95% CI of the mean under author-blocked resampling (queries within an author are
    correlated siblings — resampling queries would understate the variance)."""
    values = np.asarray(values, dtype="float64")
    uniq, inv = np.unique(np.asarray(authors, dtype=int), return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype("float64")
    draws = _boot_draws(len(uniq), n_boot, seed)
    bs = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def blocked_top3_ci(top1_ids, authors, survivors, k: int, n_boot: int, seed: int):
    """95% CI of the top-3 surviving capture share — the top-3 experts are RE-DERIVED per
    draw (the honest blocked bootstrap of a ranked statistic)."""
    top1_ids = np.asarray(top1_ids, dtype=int)
    uniq, inv = np.unique(np.asarray(authors, dtype=int), return_inverse=True)
    hists = np.zeros((len(uniq), k), dtype="float64")
    np.add.at(hists, (inv, top1_ids), 1.0)
    draws = _boot_draws(len(uniq), n_boot, seed)
    vals = np.empty(n_boot, dtype="float64")
    for lo in range(0, n_boot, 200):              # chunked: [200, A, k] stays small
        h = hists[draws[lo:lo + 200]].sum(axis=1)
        ranked = np.sort(h, axis=1)[:, ::-1]
        vals[lo:lo + 200] = ranked[:, :3].sum(axis=1) / np.maximum(h.sum(axis=1), 1.0)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ── family npz analysis ──────────────────────────────────────────────────────────

def _cross(what: str, got, want, ctx: str, tol: float = 1e-6):
    if want is None or got is None:
        return
    if not np.isfinite(got) or abs(float(got) - float(want)) > tol:
        raise DriftError(f"[cross-assert] {ctx}: {what} recomputed {got!r} != "
                         f"aggregate JSON {want!r} (tol {tol}) — npz/json pair mismatched")


def _npz_str(z, key: str) -> str:
    return str(z[key][()])


def analyze_family_npz(npz_path: str, json_entry: dict = None, n_boot: int = 1000,
                       seed: int = 42) -> dict:
    """One npz -> per-cell leak metrics + detector family + CIs (+ cross-asserts when the
    aggregate JSON entry is supplied)."""
    z = np.load(npz_path, allow_pickle=False)
    strategy = _npz_str(z, "strategy")
    k = int(z["k"][()])
    drop_sets = json.loads(_npz_str(z, "drop_sets"))
    author = np.asarray(z["author_of_q"], dtype=int)
    num_authors = int(author.max()) + 1
    assert num_authors % k == 0, \
        f"{npz_path}: inferred num_authors={num_authors} not divisible by k={k}"
    per_shard = num_authors // k
    shard_of_q = author // per_shard
    family = strategy_family(strategy)
    match = np.asarray(z["match"], dtype=bool) if "match" in z.files else None
    scores = np.asarray(z["scores"], dtype="float32") if "scores" in z.files else None
    sent = (np.asarray(z["author_sent_scores"], dtype="float32")
            if "author_sent_scores" in z.files else None)
    sent_ids = (np.asarray(z["sent_author_ids"], dtype=int)
                if "sent_author_ids" in z.files else None)
    assert (match is not None) or (scores is not None), f"{npz_path}: no scores/match"

    if match is not None:
        routes_full, nomatch_full = key_exact_routes(match, list(range(k)))
    else:
        routes_full = masked_top1(scores, list(range(k)))
        nomatch_full = None
    full_acc = float((routes_full == shard_of_q).mean())
    ctx0 = f"{strategy}@k{k} ({os.path.basename(npz_path)})"
    if json_entry is not None:
        _cross("full_top1_acc", full_acc, json_entry.get("full_top1_acc"), ctx0)

    even_a, odd_a = split_by_author(author)
    out = {"strategy": strategy, "k": k, "num_authors": num_authors,
           "npz": os.path.abspath(npz_path), "source": "npz",
           "n_q": int(author.shape[0]), "full_top1_acc": full_acc, "cells": {}}
    jcells = (json_entry or {}).get("cells", {})

    for ids in drop_sets:
        ck = cell_key(ids)
        ctx = f"{ctx0} cell {ck}"
        dropped = sorted(set(int(i) for i in ids))
        surv = [j for j in range(k) if j not in set(dropped)]
        assert surv, f"{ctx}: drop set leaves no survivors"
        orphan = np.isin(shard_of_q, dropped)
        retain = ~orphan
        jc = jcells.get(ck) or {}
        # keep the producer's order-preserving cell key (d3_2, not the sorted d2_3)
        cell = {"cell": ck, "dropped_shards": dropped, "n_survivors": len(surv),
                "n_orphans": int(orphan.sum()), "n_retain": int(retain.sum())}
        f_eval = orphan & odd_a
        r_eval = retain & odd_a
        cell["n_eval_orphan"], cell["n_eval_retain"] = int(f_eval.sum()), int(r_eval.sum())

        if match is not None:
            routes_post, nomatch_post = key_exact_routes(match, surv)
            top1_post = routes_post
        else:
            pk = f"scores__{ck}"
            if strategy == "logit_div":
                # note iii: the candidate-mean divergence must be recomputed per survivor
                # set — a column-mask of the full matrix is the wrong quantity
                assert pk in z.files, f"{ctx}: logit_div npz missing {pk}"
            post = np.asarray(z[pk], dtype="float32") if pk in z.files else scores
            S32 = post[:, surv]
            assert np.isfinite(S32).all(), f"{ctx}: non-finite survivor scores in " \
                                           f"{pk if pk in z.files else 'scores'}"
            top1_post = np.asarray(surv, dtype=int)[np.argmax(S32, axis=1)]

        # capture / shift (+ adequacy for graded) — the cross-asserted aggregates
        cap = ranked_shares(top1_post[orphan], surv, k) if orphan.any() else None
        shift = (float((top1_post[retain] != routes_full[retain]).mean())
                 if retain.any() else None)
        cell["retain_shift_top1"] = shift
        if cap is not None:
            cell["capture"] = cap
            _cross("capture.top3", cap["top3_share"],
                   (jc.get("orphan_capture") or {}).get("top1_share_top3_experts"), ctx)
            _cross("capture.top1", cap["top1_share"],
                   (jc.get("orphan_capture") or {}).get("top1_share_top1_expert"), ctx)
        _cross("retain_shift_top1", shift, jc.get("retain_shift_top1"), ctx)

        if match is not None:
            # key_exact: binary no-match operating point; EXCLUDED from graded AUCs
            rates = {"orphan_no_match_rate_postdrop": (float(nomatch_post[orphan].mean())
                                                       if orphan.any() else None),
                     "retain_no_match_rate_postdrop": (float(nomatch_post[retain].mean())
                                                       if retain.any() else None),
                     "retain_no_match_rate_full": (float(nomatch_full[retain].mean())
                                                   if retain.any() else None),
                     "fallback_shard": int(surv[0])}
            jn = jc.get("no_match") or {}
            for key_ in ("orphan_no_match_rate_postdrop", "retain_no_match_rate_postdrop",
                         "retain_no_match_rate_full"):
                _cross(f"no_match.{key_}", rates[key_], jn.get(key_), ctx)
            if f_eval.any() and r_eval.any():
                rates["eval_half"] = {
                    "orphan_no_match_rate": float(nomatch_post[f_eval].mean()),
                    "retain_no_match_rate": float(nomatch_post[r_eval].mean()),
                    "implied_auc": _auc(nomatch_post[f_eval].astype("float64"),
                                        nomatch_post[r_eval].astype("float64")),
                }
            cell["no_match"] = rates
        else:
            if orphan.any():
                # producer semantics: unmasked = FULL-pool top-1 (full candidate set for
                # logit_div too), masked = survivor-restricted post matrix top-1
                unm = scores[orphan].max(axis=1)
                msk = S32[orphan].max(axis=1)
                ratio, definition = adequacy_ratio(family, unm, msk)
                adq = {"definition": definition, "mean": float(ratio.mean()),
                       "p10": float(np.percentile(ratio, 10)),
                       "p90": float(np.percentile(ratio, 90))}
                _cross("adequacy.mean", adq["mean"], (jc.get("adequacy") or {}).get("mean"),
                       ctx)
                adq["mean_ci95"] = blocked_mean_ci(ratio, author[orphan], n_boot, seed)
                cell["adequacy"] = adq

            # detector ROC family + abstain sweep, all eval-half
            r_calib = retain & even_a
            tomb_vec = None
            if sent is not None and sent_ids is not None:
                d_authors = {a for j in dropped
                             for a in range(j * per_shard, (j + 1) * per_shard)}
                cols = [i for i, a in enumerate(sent_ids) if int(a) in d_authors]
                if cols:
                    tomb_vec = sent[:, cols].max(axis=1).astype("float64")
            det, top1 = graded_detectors(S32.astype("float64"), r_calib, tomb_vec)
            if f_eval.any() and r_eval.any():
                detectors = {}
                for name, vec in det.items():
                    pos, neg = vec[f_eval], vec[r_eval]
                    detectors[name] = {"auc": _auc(pos, neg), **fpr_at_catch(pos, neg)}
                cell["detectors"] = detectors
                conf = {n: d for n, d in detectors.items() if n in CONFIDENCE_DETECTORS}
                if conf:
                    bname = max(conf, key=lambda n: conf[n]["auc"])
                    cell["best_confidence"] = {"detector": bname, **conf[bname]}
                if "tomb_author" in det:
                    # argmax operating point (sentinel wins top-1 <=> score > 0) — the
                    # tombstone_analysis convention the 0.963/0.091 prior used
                    cell["tomb_author_argmax"] = {
                        "catch": float((det["tomb_author"][f_eval] > 0).mean()),
                        "fpr": float((det["tomb_author"][r_eval] > 0).mean())}
                cell["abstain"] = abstain_block(top1[f_eval], top1[r_eval])

        # author-blocked CIs on the headline aggregates
        if cap is not None and cap["n"] > 0:
            cell.setdefault("capture", cap)
            cell["capture"]["top3_ci95"] = blocked_top3_ci(
                top1_post[orphan], author[orphan], surv, k, n_boot, seed)
        if retain.any() and shift is not None:
            shifted = (top1_post[retain] != routes_full[retain]).astype("float64")
            cell["retain_shift_ci95"] = blocked_mean_ci(shifted, author[retain],
                                                        n_boot, seed)

        cell["h_arch"] = h_arch_cell(
            (cell.get("capture") or {}).get("top3_share"),
            (cell.get("adequacy") or {}).get("mean"),
            (cell.get("best_confidence") or {}).get("auc"),
            (cell.get("best_confidence") or {}).get("retain_fpr"))
        out["cells"][ck] = cell
    return out


def summarize_json_only(strategy: str, entry: dict, meta: dict) -> dict:
    """Aggregate-only fallback when a strategy's npz sidecar is absent — the leak cells
    are lifted from the JSON; detectors/CIs are pending."""
    out = {"strategy": strategy, "k": int(meta.get("k", -1)), "npz": None,
           "source": "json_only (npz missing — detectors/CIs pending)",
           "full_top1_acc": entry.get("full_top1_acc"), "cells": {}}
    for ck, jc in (entry.get("cells") or {}).items():
        cap = jc.get("orphan_capture") or {}
        cell = {"cell": ck, "dropped_shards": jc.get("dropped_shards"),
                "n_survivors": jc.get("n_survivors"),
                "n_orphans": jc.get("n_orphans"), "n_retain": jc.get("n_retain"),
                "retain_shift_top1": jc.get("retain_shift_top1")}
        if cap:
            cell["capture"] = {"n": cap.get("n"),
                               "top1_share": cap.get("top1_share_top1_expert"),
                               "top3_share": cap.get("top1_share_top3_experts"),
                               "entropy_norm": cap.get("top1_entropy_norm")}
        if jc.get("adequacy"):
            cell["adequacy"] = {kk: jc["adequacy"].get(kk)
                                for kk in ("definition", "mean", "p10", "p90")}
        if jc.get("no_match"):
            cell["no_match"] = jc["no_match"]
        cell["h_arch"] = h_arch_cell((cell.get("capture") or {}).get("top3_share"),
                                     (cell.get("adequacy") or {}).get("mean"), None, None)
        out["cells"][ck] = cell
    return out


# ── H-ARCH / H-DIAL machinery ────────────────────────────────────────────────────

def h_arch_cell(capture, adequacy, auc, fpr) -> dict:
    """Pre-registered H-ARCH leak conditions: (a) top-3 surviving capture >= 0.5,
    (b) sibling adequacy >= 0.9, (c) best confidence AUC <= 0.75 AND FPR >= 0.3 at 90%
    catch. `separable` is the REFUTE side (AUC >= 0.90 at FPR <= 0.10)."""
    a = None if capture is None else bool(capture >= 0.5)
    b = None if adequacy is None else bool(adequacy >= 0.9)
    c = None if (auc is None or fpr is None) else bool(auc <= 0.75 and fpr >= 0.3)
    sep = None if (auc is None or fpr is None) else bool(auc >= 0.90 and fpr <= 0.10)

    def g(v, ch):
        return f"{ch}·" if v is None else f"{ch}{'✓' if v else '✗'}"

    return {"a_capture_ge_0p5": a, "b_adequacy_ge_0p9": b,
            "c_auc_le_0p75_fpr_ge_0p3": c, "separable_auc_ge_0p90_fpr_le_0p10": sep,
            "glyphs": " ".join((g(a, "a"), g(b, "b"), g(c, "c")))}


def _first_cell(s: dict):
    cells = s.get("cells") or {}
    if not cells:
        return None
    return min(cells.values(), key=lambda c: len(c.get("dropped_shards") or [99] * 99))


def _entry_src(s: dict):
    """Provenance file of a family entry: the analyzed npz, else the aggregate JSON."""
    return s.get("npz") or s.get("json")


def _entry_mtime(s: dict) -> float:
    p = _entry_src(s)
    try:
        return os.path.getmtime(p) if p else -1.0
    except OSError:
        return -1.0


def h_arch_verdict(family: dict) -> dict:
    # duplicate strategy@k10 entries ('+'-suffixed keys from multiple globs) are
    # resolved DETERMINISTICALLY: latest source-file mtime wins (family key breaks
    # exact ties); a loud WARN names the losers and the winner is recorded in the JSON.
    cand = {}
    for key_, s in family.items():
        if s.get("strategy") not in FEATURE9 or s.get("k") != 10:
            continue
        c = _first_cell(s)
        if c is None:
            continue
        cand.setdefault(s["strategy"], []).append((key_, s, c))
    rows, dups, dup_warns = {}, {}, []
    for strategy, lst in cand.items():
        lst = sorted(lst, key=lambda t: (_entry_mtime(t[1]), t[0]))
        win_key, win_s, win_c = lst[-1]
        if len(lst) > 1:
            dups[strategy] = {
                "kept_key": win_key, "kept_source": _entry_src(win_s),
                "kept_mtime": _entry_mtime(win_s),
                "ignored": [{"key": k2, "source": _entry_src(s2),
                             "mtime": _entry_mtime(s2)} for k2, s2, _ in lst[:-1]]}
            w = (f"h_arch: duplicate {strategy}@k10 entries "
                 f"({', '.join(k2 for k2, _, _ in lst)}) — kept latest-mtime source "
                 f"{os.path.basename(str(_entry_src(win_s)))} ({win_key})")
            dup_warns.append(w)
            print(f"WARN: {w}", file=sys.stderr)
        rows[strategy] = win_c["h_arch"]
    n_all3 = sum(1 for h in rows.values()
                 if h["a_capture_ge_0p5"] and h["b_adequacy_ge_0p9"]
                 and h["c_auc_le_0p75_fpr_ge_0p3"])
    n_sep = sum(1 for h in rows.values() if h["separable_auc_ge_0p90_fpr_le_0p10"])
    # families present but with condition (c) unmeasured (json_only — npz sidecar
    # missing, so detectors never ran): the verdict must stay PENDING, never MIXED,
    # because absent evidence is not a failed bar.
    pend_c = sorted(n for n, h in rows.items()
                    if h["c_auc_le_0p75_fpr_ge_0p3"] is None)
    if n_sep >= 2:
        verdict = f"REFUTED (>= 2 families separable: {n_sep})"
    elif len(rows) < len(FEATURE9):
        verdict = f"PENDING ({len(rows)}/{len(FEATURE9)} families present; " \
                  f"{n_all3} meet all three so far)"
    elif pend_c:
        verdict = f"PENDING (detectors missing for {len(pend_c)}/9 families: " \
                  f"{', '.join(pend_c)} — npz sidecars absent; " \
                  f"{n_all3} meet all three so far)"
    elif n_all3 >= 7:
        verdict = f"CONFIRMED ({n_all3}/9 families meet a+b+c; separable: {n_sep})"
    else:
        verdict = f"MIXED ({n_all3}/9 meet a+b+c, {n_sep} separable — neither bar hit)"
    return {"per_family": rows, "n_all3": n_all3, "n_separable": n_sep,
            "families_detectors_pending": pend_c,
            "duplicates": dups, "duplicate_warnings": dup_warns,
            "bar": "CONFIRM >=7/9 all-three; REFUTE >=2 separable; PENDING until all "
                   "9 families have measured detectors", "verdict": verdict}


def monotone_flags(seq) -> list:
    """seq = [(cell, value, (lo, hi)), ...] ordered by dropped-shard count. Pre-registered
    expectation: monotone NON-DECREASING. Flag i->i+1 decreases whose 95% CIs are
    disjoint (hi_{i+1} < lo_i) — a CI-backed monotonicity violation."""
    flags = []
    for (c0, v0, ci0), (c1, v1, ci1) in zip(seq, seq[1:]):
        if v0 is None or v1 is None or not ci0 or not ci1:
            continue
        if v1 < v0 and ci1[1] < ci0[0]:
            flags.append({"from": c0, "to": c1, "delta": float(v1 - v0),
                          "ci_from": list(ci0), "ci_to": list(ci1)})
    return flags


def h_dial_csq_bar(family: dict) -> dict:
    """Pre-registered H-DIAL sub-bar: centroid_sbert_q sim-ratio adequacy stays >= 0.95
    at EVERY k=10 drop set ({9}, {9,8}, {9,8,7,6}). pass None = PENDING (entry absent
    or an adequacy value unmeasured); duplicate entries resolve like h_arch (latest
    source mtime wins)."""
    bar = "centroid_sbert_q adequacy (sim-ratio) >= 0.95 at every k=10 drop set"
    cands = [(key_, s) for key_, s in family.items()
             if s.get("strategy") == "centroid_sbert_q" and s.get("k") == 10]
    if not cands:
        return {"bar": bar, "entry": None, "cells": {}, "n_cells": 0, "pass": None}
    key_, s = max(cands, key=lambda t: (_entry_mtime(t[1]), t[0]))
    cells = {ck: (c.get("adequacy") or {}).get("mean")
             for ck, c in (s.get("cells") or {}).items()}
    out = {"bar": bar, "entry": key_, "cells": cells, "n_cells": len(cells),
           "pass": None}
    if cells and all(v is not None for v in cells.values()):
        out["pass"] = bool(all(v >= 0.95 for v in cells.values()))
    return out


def h_dial_from_family(family: dict) -> dict:
    getters = {
        "capture_top3": (lambda c: (c.get("capture") or {}).get("top3_share"),
                         lambda c: (c.get("capture") or {}).get("top3_ci95")),
        "retain_shift": (lambda c: c.get("retain_shift_top1"),
                         lambda c: c.get("retain_shift_ci95")),
        "adequacy": (lambda c: (c.get("adequacy") or {}).get("mean"),
                     lambda c: (c.get("adequacy") or {}).get("mean_ci95")),
    }
    per = {}
    for key_, s in family.items():
        cells = list((s.get("cells") or {}).values())
        if len(cells) < 2:
            continue
        cells.sort(key=lambda c: len(c.get("dropped_shards") or []))
        entry = {}
        for metric, (gv, gci) in getters.items():
            seq = [(c.get("cell") or cell_key(c["dropped_shards"]), gv(c), gci(c))
                   for c in cells]
            entry[metric] = {"sequence": [[a, b, list(cc) if cc else None]
                                          for a, b, cc in seq],
                             "flags": monotone_flags(seq)}
        per[key_] = entry
    return {"per_strategy": per, "csq_adequacy_ge_0p95": h_dial_csq_bar(family)}


# ── RouterLoRA (H-TRAINED) ───────────────────────────────────────────────────────

def analyze_routerlora_json(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    n_active = np.asarray(d["n_active"], dtype=int)
    n_active_full = np.asarray(d["n_active_full"], dtype=int)
    is_f = np.asarray(d["is_forget"], dtype=bool)
    h = np.asarray(d["h_norm"], dtype="float64")
    ms = np.asarray(d["max_share"], dtype="float64")
    t1 = np.asarray(d["top1_share"], dtype="float64")
    t1f = np.asarray(d["top1_share_full"], dtype="float64")
    fb = np.asarray(d["fallback_used"], dtype=bool)
    # the m==1 degeneracy fix: alpha_stats reports h_norm=1.0/max_share=1.0 when the set
    # collapsed to one expert — those rows carry no confidence signal and must be dropped
    valid = (n_active > 1) & (n_active_full > 1)
    pos, neg = is_f & valid, (~is_f) & valid
    ratio = t1 / np.maximum(t1f, 1e-12)
    m = re.search(r"_s(\d+)\.safetensors$", str(d.get("router_ckpt", "")))
    seed_id = int(m.group(1)) if m else int(d.get("seed", 42))
    out = {"path": os.path.abspath(path), "seed": seed_id,
           "router_ckpt": d.get("router_ckpt"), "unlearn_tag": d.get("unlearn_tag"),
           "n_records": int(is_f.size), "n_excluded_m1": int((~valid).sum()),
           "n_excluded_forget": int((is_f & ~valid).sum()),
           "auc_h_norm_unfiltered": d.get("auc_h_norm"),
           "auc_max_share_unfiltered": d.get("auc_max_share"),
           "auc_h_norm_filtered": (_auc(h[pos], h[neg])
                                   if pos.any() and neg.any() else None),
           "auc_max_share_filtered": (_auc(-ms[pos], -ms[neg])
                                      if pos.any() and neg.any() else None),
           "top1_share_ratio": {"forget": float(ratio[is_f].mean()),
                                "retain": float(ratio[~is_f].mean())},
           # the same ratio on the m>1-FILTERED rows (the AUC filter). The H-TRAINED
           # forget_ratio BAR deliberately stays on the all-row mean above — per the
           # plan the m==1 filter applies to AUCs only — but m==1 rows report ratio
           # 1.0 and inflate it, so the filtered view ships alongside for comparison.
           "top1_share_ratio_multi": {
               "forget": float(ratio[pos].mean()) if pos.any() else None,
               "retain": float(ratio[neg].mean()) if neg.any() else None},
           "fallback_rate": {"forget": float(fb[is_f].mean()),
                             "retain": float(fb[~is_f].mean())}}
    # consistency gate vs the producer's own group means (all-row quantities)
    g = d.get("groups", {})
    for gname, mask in (("forget", is_f), ("retain", ~is_f)):
        want = (g.get(gname) or {}).get("top1_share_ratio_mean")
        if want is not None:
            _cross(f"groups.{gname}.top1_share_ratio_mean",
                   out["top1_share_ratio"][gname], want, f"routerlora {path}")
    return out


def routerlora_cross_seed(per_seed: list) -> dict:
    def _agg(key_, sub=None):
        vals = [(s[key_][sub] if sub else s[key_]) for s in per_seed
                if (s.get(key_) is not None and (not sub or s[key_][sub] is not None))]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n_seeds": len(vals)}

    # None-safe per-seed worst filtered AUC: a seed whose filtered AUCs are BOTH None
    # (every row m==1-degenerate) carries no detector evidence and is EXCLUDED from the
    # AUC bars — absent evidence must not pass the confirm bar, and a legitimate 0.0
    # AUC must stay distinct from None (the old `s[key] or 0.0` conflated the two).
    def _worst(s):
        vals = [v for v in (s["auc_h_norm_filtered"], s["auc_max_share_filtered"])
                if v is not None]
        return max(vals) if vals else None

    worst_auc = [_worst(s) for s in per_seed]
    evid = [a for a in worst_auc if a is not None]
    n_excl = len(worst_auc) - len(evid)
    bars = {"auc_le_0p75_all_seeds": (all(a <= 0.75 for a in evid)
                                      if per_seed and n_excl == 0 else None),
            "any_seed_auc_ge_0p90": any(a >= 0.90 for a in evid),
            "forget_ratio_ge_0p9_all_seeds": bool(per_seed) and all(
                s["top1_share_ratio"]["forget"] >= 0.9 for s in per_seed),
            "n_seeds_with_auc_evidence": len(evid),
            "n_seeds_auc_excluded": n_excl}
    if not per_seed:
        verdict = "PENDING (no rl_routerlora_* inputs)"
    elif bars["any_seed_auc_ge_0p90"]:
        verdict = "REFUTED (a seed's detector AUC >= 0.90 — learned gate detects orphanhood)"
    elif len(per_seed) < 3:
        verdict = f"PENDING ({len(per_seed)}/3 seeds present)"
    elif bars["auc_le_0p75_all_seeds"] is None:
        verdict = f"PENDING (AUC bar unmeasurable: {n_excl}/{len(per_seed)} seed(s) " \
                  f"have no m>1-filtered rows — no detector evidence)"
    elif bars["auc_le_0p75_all_seeds"] and bars["forget_ratio_ge_0p9_all_seeds"]:
        verdict = "CONFIRMED (all seeds AUC <= 0.75 and orphan top-1 share ratio >= 0.9)"
    else:
        verdict = "MIXED (bars split across seeds — see per-seed table)"
    return {"auc_h_norm_filtered": _agg("auc_h_norm_filtered"),
            "auc_max_share_filtered": _agg("auc_max_share_filtered"),
            "top1_share_ratio_forget": _agg("top1_share_ratio", "forget"),
            "top1_share_ratio_forget_multi": _agg("top1_share_ratio_multi", "forget"),
            "fallback_rate_forget": _agg("fallback_rate", "forget"),
            "bars": bars, "verdict": verdict,
            "note": "the forget_ratio bar reads the UNFILTERED all-row top1_share_ratio "
                    "(m==1 rows report ratio 1.0 and inflate it); the AUC bars are "
                    "m==1-filtered — per the pre-registration the m==1 filter applies "
                    "to AUCs only. Compare top1_share_ratio_forget_multi (m>1 rows) "
                    "alongside."}


# ── DBpedia (H-DATASET) / encoders (H-ENC) / SepMLP ──────────────────────────────

def analyze_dbpedia_json(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    out = {"path": os.path.abspath(path), "n": d.get("n"), "k": d.get("k"),
           "encoder_source": d.get("encoder_source"), "tags": {}}
    for tag, t in (d.get("tags") or {}).items():
        de = t.get("dropped_extras") or {}
        out["tags"][tag] = {
            "n_orphans": de.get("n_orphans"),
            "capture_top3": de.get("top1_share_top3_experts"),
            "sim_ratio_mean": de.get("mean_top1_sim_ratio"),
            "retain_shift_top1": (t.get("selection_shift_stale_vs_dropped") or {}
                                  ).get("shift_top1"),
            "abstain": t.get("abstain"),
        }
    out["dropped_extras_pooled"] = d.get("dropped_extras_pooled")
    out["abstain_pooled"] = d.get("abstain_pooled")
    ab = d.get("abstain_pooled") or {}
    op90 = (ab.get("by_orphan_catch") or {}).get("0.90")
    adequacy = (d.get("dropped_extras_pooled") or {}).get("mean_top1_sim_ratio")
    out["h_dataset"] = {
        "adequacy_ge_0p9": None if adequacy is None else bool(adequacy >= 0.9),
        "separating_tau_exists": (None if op90 is None
                                  else bool(op90["retain_false_abstain_rate"] <= 0.1)),
        "bar": "CONFIRM capture>=0.5 & adequacy>=0.9 & no tau with 90% catch at <=10% "
               "retain cost; REFUTE if a separating tau exists"}
    return out


def analyze_enc_json(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    if "sibling" not in d:
        return {"path": os.path.abspath(path),
                "note": "not a centroid-mode audit JSON — if this is a family aggregate, "
                        "pass it via --family_json instead", "raw_keys": sorted(d)[:12]}
    sib, tomb = d.get("sibling") or {}, d.get("tombstone") or {}
    out = {"path": os.path.abspath(path), "encoder": d.get("router_encoder"),
           "k": d.get("k"), "drop_shard": d.get("drop_shard"),
           "full_acc_retain": (d.get("full") or {}).get("acc_retain_top1"),
           "adequacy_sim_ratio": sib.get("mean_top1_sim_ratio"),
           "adequacy_p10": sib.get("p10_top1_sim_ratio"),
           "retain_shift_top1": sib.get("retain_shift_top1"),
           "tombstone": {rung: {"catch": tomb[rung].get("orphan_catch_rate"),
                                "fpr": tomb[rung].get("retain_false_tombstone_rate")}
                         for rung in ("expert", "author", "name") if rung in tomb},
           "disclosure_auc": (d.get("disclosure") or {}).get("auc_forget_vs_holdout")}
    sr = out["adequacy_sim_ratio"]
    out["h_enc_sim_ratio_ge_0p95"] = None if sr is None else bool(sr >= 0.95)
    return out


def analyze_enc_roc_json(path: str) -> dict:
    """H-ENC confidence half — one analyze_router_leak.py `roc` JSON (run over an
    rl_enc_*.sims.npz sidecar, centroid layout). Reads the max AUC over the NATIVE
    confidence detectors only (ENC_CONF_DETECTORS; tomb_* rungs are the seal family
    and are excluded a priori). Per-encoder bar: confirm AUC <= 0.75, refute >= 0.90,
    else inconclusive."""
    with open(path) as f:
        d = json.load(f)
    det = d.get("detectors") or {}
    conf = {n: v for n, v in det.items() if n in ENC_CONF_DETECTORS}
    npz_base = os.path.basename(str(d.get("npz") or ""))
    m = re.match(r"rl_enc_(.+?)\.sims\.npz$", npz_base)
    m2 = re.match(r"rl_enc_roc_(.+?)\.json$", os.path.basename(path))
    tag = m.group(1) if m else (m2.group(1) if m2
                                else (npz_base or os.path.basename(path)))
    out = {"path": os.path.abspath(path), "encoder_tag": tag, "mode": d.get("mode"),
           "npz": d.get("npz"), "n_forget_eval": d.get("n_forget_eval"),
           "n_retain_eval": d.get("n_retain_eval"),
           "excluded_detectors": sorted(n for n in det if n not in ENC_CONF_DETECTORS)}
    if conf:
        best = max(conf, key=lambda n: conf[n]["auc"])
        auc = conf[best]["auc"]
        out["confidence_detector"] = best
        out["confidence_auc_max"] = auc
        out["retain_fpr_at_90catch"] = conf[best].get("retain_fpr")
        out["h_enc_conf"] = ("refute" if auc >= 0.90
                             else "confirm" if auc <= 0.75 else "inconclusive")
    else:
        out["confidence_detector"] = None
        out["confidence_auc_max"] = None
        out["retain_fpr_at_90catch"] = None
        out["h_enc_conf"] = None
    return out


def enc_roc_summary(per_encoder: list) -> dict:
    """Aggregate H-ENC confidence-half verdict over the enc roc JSONs. PENDING while no
    roc files exist (the sims sidecars still need analyze_router_leak.py roc runs)."""
    bar = ("per-encoder max confidence AUC (global_top1/per_expert/margin/knn_density; "
           "tomb_* excluded): CONFIRM <= 0.75, REFUTE >= 0.90, else inconclusive")
    if not per_encoder:
        return {"per_encoder": [], "bar": bar,
                "verdict": "PENDING (no enc roc JSONs — run analyze_router_leak.py roc "
                           "on the rl_enc_*.sims.npz sidecars)"}
    outcomes = [e.get("h_enc_conf") for e in per_encoder]
    if any(o == "refute" for o in outcomes):
        verdict = "REFUTED (an encoder's confidence AUC >= 0.90 — see per_encoder)"
    elif any(o is None for o in outcomes):
        verdict = "PENDING (a roc JSON carries no confidence-family detectors)"
    elif all(o == "confirm" for o in outcomes):
        verdict = f"CONFIRMED (all {len(outcomes)} encoders' confidence AUC <= 0.75)"
    else:
        verdict = "INCONCLUSIVE (an encoder's confidence AUC in (0.75, 0.90))"
    return {"per_encoder": per_encoder, "bar": bar, "verdict": verdict}


def analyze_sepmlp(leak_path: str, ref_path: str = None) -> dict:
    """SEPMLP LEAK-PROBE NPZ CONTRACT consumer — the routerless 'selection inside the
    weights' row. Orphan capture fires at the serving level: a surviving branch is 'as loud
    as an owner' when its output norm reaches the p10 of retain OWN-branch norms."""
    z = np.load(leak_path, allow_pickle=False)
    group = np.asarray(z["group"]).astype(str)
    msn = np.asarray(z["max_surv_norm"], dtype="float64")
    mfn = np.asarray(z["max_foreign_norm"], dtype="float64")
    own = np.asarray(z["own_norm"], dtype="float64")
    top = np.asarray(z["top_surv_author"], dtype=int)
    author = np.asarray(z["author_of_q"], dtype=int)
    retain = group == "retain"
    assert retain.any(), f"{leak_path}: no retain rows"
    assert np.isfinite(own[retain]).any(), f"{leak_path}: retain own_norm all-NaN"
    p10_own = float(np.nanpercentile(own[retain], 10))
    out = {"leak_npz": os.path.abspath(leak_path),
           "n_surviving": int(z["n_surviving"][()]),
           "droplist_tag": _npz_str(z, "droplist_tag"), "K": int(z["K"][()]),
           "retain_own_norm_p10": p10_own, "groups": {}}
    for g in sorted(set(group)):
        rows = group == g
        gi = {"n": int(rows.sum()),
              "max_surv_norm_mean": float(msn[rows].mean()),
              "max_surv_norm_p50": float(np.percentile(msn[rows], 50)),
              "max_surv_norm_p90": float(np.percentile(msn[rows], 90))}
        if g.startswith("forget"):
            gi["orphan_capture"] = float((msn[rows] >= p10_own).mean())
            gi["identity_auc"] = _auc(-msn[rows], -msn[retain])
        out["groups"][g] = gi
    # off-level context: are surviving branches quieter on orphans than the ordinary
    # off-branch level retain queries see?
    out["off_level"] = {
        "retain_max_foreign_norm_p50": float(np.percentile(mfn[retain], 50)),
        "retain_max_foreign_norm_p90": float(np.percentile(mfn[retain], 90)),
        "orphan_orig_over_retain_foreign_p50": (
            float(np.percentile(msn[group == "forget_orig"], 50)
                  / max(np.percentile(mfn[retain], 50), 1e-12))
            if (group == "forget_orig").any() else None)}
    if ref_path and os.path.exists(ref_path):
        zr = np.load(ref_path, allow_pickle=False)
        rgroup = np.asarray(zr["group"]).astype(str)
        rauthor = np.asarray(zr["author_of_q"], dtype=int)
        assert rgroup.shape == group.shape and (rgroup == group).all() \
            and (rauthor == author).all(), \
            f"{ref_path}: rows misaligned with {leak_path} (group/author order differs)"
        rtop = np.asarray(zr["top_surv_author"], dtype=int)
        out["retain_collateral_top_author_shift"] = float(
            (top[retain] != rtop[retain]).mean())
        out["ref_npz"] = os.path.abspath(ref_path)
    else:
        out["retain_collateral_top_author_shift"] = None
        out["ref_npz"] = "pending" if ref_path else None
    return out


# ── input collection ─────────────────────────────────────────────────────────────

def _expand(patterns) -> tuple:
    found, missing = [], []
    for p in patterns or []:
        hits = sorted(globlib.glob(p))
        if hits:
            found.extend(hits)
        else:
            missing.append(p)
    return sorted(set(found)), missing


def collect_family(json_paths: list, npz_paths: list, n_boot: int, seed: int):
    family, warnings, seen_npz = {}, [], set()
    oracle = None

    def _store(res):
        key_ = f"{res['strategy']}@k{res['k']}"
        while key_ in family:
            key_ += "+"
        family[key_] = res

    for jp in json_paths:
        with open(jp) as f:
            d = json.load(f)
        meta = d.get("meta") or {}
        if d.get("oracle") and oracle is None:
            oracle = dict(d["oracle"], source_json=os.path.abspath(jp))
        for strategy, entry in (d.get("strategies") or {}).items():
            np_path = entry.get("npz")
            if np_path and not os.path.exists(np_path):
                # aggregate JSONs record absolute npz paths at production time; fall back
                # to a sibling of the JSON when the tree moved
                cand = os.path.join(os.path.dirname(jp), os.path.basename(np_path))
                np_path = cand if os.path.exists(cand) else None
            if np_path:
                res = analyze_family_npz(np_path, entry, n_boot, seed)
                seen_npz.add(os.path.realpath(np_path))
            else:
                res = summarize_json_only(strategy, entry, meta)
                warnings.append(f"{strategy} ({os.path.basename(jp)}): npz sidecar "
                                f"missing — detectors/CIs pending")
            res["json"] = os.path.abspath(jp)
            res["stub"] = bool(meta.get("stub"))
            res["self_check"] = entry.get("self_check")
            _store(res)
    for np_path in npz_paths:
        if os.path.realpath(np_path) in seen_npz:
            continue
        res = analyze_family_npz(np_path, None, n_boot, seed)
        res["json"] = None
        res["stub"] = res.get("num_authors") == 8      # the stub pool's signature
        warnings.append(f"{os.path.basename(np_path)}: analyzed npz-only "
                        f"(no aggregate JSON — cross-assert skipped)")
        _store(res)

    # continuity WARNs (never fail) — real-pool entries only
    for key_, s in family.items():
        if s.get("stub"):
            continue
        if s["strategy"] == "centroid_sbert_q" and s["k"] == 10 and "d9" in s["cells"]:
            c = s["cells"]["d9"]
            adq = (c.get("adequacy") or {}).get("mean")
            if adq is not None and abs(adq - CONTINUITY["csq_adequacy"]) > CONTINUITY["tol"]:
                warnings.append(f"CONTINUITY: centroid_sbert_q d9 adequacy {adq:.4f} "
                                f"vs rl_centroid_k10 prior {CONTINUITY['csq_adequacy']}")
            sh = c.get("retain_shift_top1")
            if sh is not None and abs(sh - CONTINUITY["csq_shift"]) > CONTINUITY["tol"]:
                warnings.append(f"CONTINUITY: centroid_sbert_q d9 retain shift {sh:.4f} "
                                f"vs rl_centroid_k10 prior {CONTINUITY['csq_shift']}")
        if s["strategy"] == "key_exact" and s.get("num_authors") == 200:
            acc = s.get("full_top1_acc")
            if acc is not None and abs(acc - CONTINUITY["key_exact_acc"]) > \
                    CONTINUITY["key_exact_tol"]:
                warnings.append(f"CONTINUITY: key_exact@k{s['k']} full-pool acc {acc:.4f} "
                                f"vs prior ~{CONTINUITY['key_exact_acc']}")
    return family, oracle, warnings


def run_analysis(args) -> dict:
    fam_json, miss_json = _expand(args.family_json)
    fam_npz, miss_npz = _expand(args.family_npz)
    rl_json, miss_rl = _expand(args.routerlora_json)
    db_json, miss_db = _expand(args.dbpedia_json)
    enc_json, miss_enc = _expand(args.enc_json)
    encroc_json, miss_encroc = _expand(args.enc_roc_json)
    missing = {"family_json": miss_json, "family_npz": miss_npz,
               "routerlora_json": miss_rl, "dbpedia_json": miss_db,
               "enc_json": miss_enc, "enc_roc_json": miss_encroc}
    if args.sepmlp_leak_npz and not os.path.exists(args.sepmlp_leak_npz):
        missing["sepmlp_leak_npz"] = [args.sepmlp_leak_npz]
    if args.sepmlp_ref_npz and not os.path.exists(args.sepmlp_ref_npz):
        missing["sepmlp_ref_npz"] = [args.sepmlp_ref_npz]

    family, oracle, warnings = collect_family(fam_json, fam_npz, args.bootstrap, args.seed)
    per_seed = [analyze_routerlora_json(p) for p in rl_json]
    per_seed.sort(key=lambda s: s["seed"])
    dbpedia = [analyze_dbpedia_json(p) for p in db_json]
    enc = [analyze_enc_json(p) for p in enc_json]
    enc_roc = [analyze_enc_roc_json(p) for p in encroc_json]
    sepmlp = (analyze_sepmlp(args.sepmlp_leak_npz, args.sepmlp_ref_npz)
              if args.sepmlp_leak_npz and os.path.exists(args.sepmlp_leak_npz) else None)
    if oracle is None:
        oracle = {"by_construction": True,
                  "note": "oracle q2author route: orphans -> base/scaffold P=1.0, retain "
                          "top-1 shift = 0 by construction (analytic control, no run)"}
    h_arch = h_arch_verdict(family)
    warnings.extend(h_arch.get("duplicate_warnings") or [])
    return {
        "meta": {"seed": args.seed, "bootstrap": args.bootstrap,
                 "inputs": {"family_json": fam_json, "family_npz": fam_npz,
                            "routerlora_json": rl_json, "dbpedia_json": db_json,
                            "enc_json": enc_json, "enc_roc_json": encroc_json,
                            "sepmlp_leak_npz": args.sepmlp_leak_npz,
                            "sepmlp_ref_npz": args.sepmlp_ref_npz},
                 "missing": {k2: v for k2, v in missing.items() if v},
                 "warnings": warnings,
                 "conventions": "author-parity even=calib/odd=eval; detector directions "
                                "a priori (confidence NEGATED, tomb as-is); author-blocked "
                                "bootstrap CIs; abstain taus retain-percentile only"},
        "priors": PRIOR_ROWS,
        "oracle": oracle,
        "family": family,
        "h_arch": h_arch,
        "h_dial": h_dial_from_family(family),
        "routerlora": {"per_seed": per_seed,
                       "cross_seed": routerlora_cross_seed(per_seed)},
        "dbpedia": dbpedia,
        "enc": enc,
        "enc_roc": enc_roc_summary(enc_roc),
        "sepmlp": sepmlp if sepmlp is not None else {"status": "pending (no leak npz)"},
    }


# ── markdown rendering ───────────────────────────────────────────────────────────

def _f(x, nd=3):
    if x is None:
        return "—"
    try:
        if not np.isfinite(x):
            return "—"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def _ci(ci, nd=3):
    return f"[{_f(ci[0], nd)}, {_f(ci[1], nd)}]" if ci else "—"


def write_md(res: dict, path: str) -> None:
    L = []
    add = L.append
    add("# All-router leak table (router_leak Phase 3)")
    add("")
    add(f"Seed {res['meta']['seed']}; bootstrap {res['meta']['bootstrap']} author-blocked "
        f"draws; conventions: {res['meta']['conventions']}.")
    add("")
    add("## Unified all-router leak table")
    add("")
    add("| router | cell | orphan top-3 capture | adequacy [95% CI] | retain shift "
        "[95% CI] | best conf. AUC (det.) | FPR@90% catch | tomb catch/FPR | H-ARCH |")
    add("|---|---|---|---|---|---|---|---|---|")
    add("| oracle q2author (control) | any | 0.000 (orphans→base P=1.0) | — | "
        "0.000 (by construction) | — | — | — | a✗ b· c· |")
    for key_ in sorted(res["family"]):
        s = res["family"][key_]
        for ck in sorted(s["cells"], key=lambda c: (len(s["cells"][c].get(
                "dropped_shards") or []), c)):
            c = s["cells"][ck]
            cap = (c.get("capture") or {})
            adq = (c.get("adequacy") or {})
            bc = c.get("best_confidence") or {}
            tomb = c.get("tomb_author_argmax")
            nm = c.get("no_match")
            if nm is not None:
                ev = nm.get("eval_half") or {}
                auccell = (f"no-match op: orphan {_f(ev.get('orphan_no_match_rate'))} / "
                           f"retain {_f(ev.get('retain_no_match_rate'))} "
                           f"(implied AUC {_f(ev.get('implied_auc'))}; excluded from "
                           f"graded aggregation)")
                fprcell = "—"
            else:
                auccell = (f"{_f(bc.get('auc'))} ({bc.get('detector', '—')})"
                           if bc else "—")
                fprcell = _f(bc.get("retain_fpr")) if bc else "—"
            add(f"| {s['strategy']} (k={s['k']}) | {ck} | {_f(cap.get('top3_share'))} "
                f"{_ci(cap.get('top3_ci95'))} | {_f(adq.get('mean'))} "
                f"{_ci(adq.get('mean_ci95'))} | {_f(c.get('retain_shift_top1'), 4)} "
                f"{_ci(c.get('retain_shift_ci95'), 4)} | {auccell} | {fprcell} | "
                f"{(_f(tomb['catch']) + '/' + _f(tomb['fpr'])) if tomb else '—'} | "
                f"{c['h_arch']['glyphs']} |")
    for p in res["priors"]:
        add(f"| {p['router']} †prior | {p['pool']} {p['cell']} | — | "
            f"{_f(p.get('adequacy'))} | {_f(p.get('retain_shift'), 4)} | — | — | "
            f"{(_f(p.get('tomb_catch')) + '/' + _f(p.get('tomb_fpr'))) if p.get('tomb_catch') else '—'} "
            f"| — |")
    sep = res.get("sepmlp") or {}
    if "groups" in sep:
        fo = sep["groups"].get("forget_orig", {})
        fp = sep["groups"].get("forget_para", {})
        add(f"| SepMLP (routerless, K={sep.get('K')}, {sep.get('droplist_tag')}) | serve | "
            f"orig {_f(fo.get('orphan_capture'))} / para {_f(fp.get('orphan_capture'))} | "
            f"— | {_f(sep.get('retain_collateral_top_author_shift'), 4)} (top-author) | "
            f"identity AUC orig {_f(fo.get('identity_auc'))} / para "
            f"{_f(fp.get('identity_auc'))} | — | — | — |")
    else:
        add("| SepMLP (routerless) | serve | pending | — | pending | pending | "
            "— | — | — |")
    add("")
    add("†prior = typed-in prior measurement (not re-run this campaign).")

    add("")
    add(f"## H-ARCH — {res['h_arch']['verdict']}")
    add("")
    add(f"Bar: {res['h_arch']['bar']}. Families meeting all three leak conditions: "
        f"{res['h_arch']['n_all3']}; separable families: {res['h_arch']['n_separable']}.")

    add("")
    add("## H-DIAL — deletion-count monotonicity")
    add("")
    hd = res["h_dial"]
    any_flag = False
    for key_ in sorted(hd["per_strategy"]):
        for metric, m in hd["per_strategy"][key_].items():
            for fl in m["flags"]:
                any_flag = True
                add(f"- FLAG {key_} {metric}: {fl['from']} → {fl['to']} decreases "
                    f"{fl['delta']:+.4f} with disjoint CIs {_ci(fl['ci_from'], 4)} vs "
                    f"{_ci(fl['ci_to'], 4)}")
    if not any_flag:
        add("- no CI-backed monotonicity violations flagged"
            + (" (no multi-cell strategies present yet)" if not hd["per_strategy"]
               else ""))
    csq = hd["csq_adequacy_ge_0p95"]
    if csq["pass"] is None:
        status = ("PENDING (no centroid_sbert_q@k10 cells yet)" if not csq["cells"]
                  else "PENDING (adequacy unmeasured for some drop set)")
    else:
        status = ("PASS" if csq["pass"] else "FAIL") + " (" + ", ".join(
            f"{ck} {_f(v)}" for ck, v in sorted(csq["cells"].items())) + ")"
    add(f"- sub-bar (pre-registered): {csq['bar']} — {status}")

    add("")
    add("## H-POOL — k=200 per-author granularity")
    add("")
    pool_rows = [s for s in res["family"].values() if (s.get("k") or 0) > 10]
    if pool_rows:
        for s in sorted(pool_rows, key=lambda x: x["strategy"]):
            for ck, c in sorted(s["cells"].items()):
                bc = c.get("best_confidence") or {}
                add(f"- {s['strategy']}@k{s['k']} {ck}: adequacy "
                    f"{_f((c.get('adequacy') or {}).get('mean'))}, best conf. AUC "
                    f"{_f(bc.get('auc'))} — bars: adequacy>=0.9, AUC<=0.75")
    else:
        add("- pending (no k>10 family cells present)")

    add("")
    add("## H-ENC — encoder generality (k=10 centroid audits)")
    add("")
    if res["enc"]:
        # NB the AUC column is the DELETION-DISCLOSURE AUC (forget-vs-holdout ranking
        # from the audit JSON) — it is NOT the confidence AUC of the H-ENC bar (that
        # half lives in the enc-roc lines below).
        add("| encoder | sim-ratio (adequacy) | retain shift | tomb author catch/FPR | "
            "deletion-disclosure AUC (forget-vs-holdout) | sim-ratio>=0.95 |")
        add("|---|---|---|---|---|---|")
        for e in res["enc"]:
            if "note" in e:
                add(f"| {os.path.basename(e['path'])} | {e['note']} | | | | |")
                continue
            ta = (e.get("tombstone") or {}).get("author") or {}
            add(f"| {e.get('encoder')} | {_f(e.get('adequacy_sim_ratio'))} | "
                f"{_f(e.get('retain_shift_top1'), 4)} | {_f(ta.get('catch'))}/"
                f"{_f(ta.get('fpr'))} | {_f(e.get('disclosure_auc'))} | "
                f"{e.get('h_enc_sim_ratio_ge_0p95')} |")
    else:
        add("- pending (no rl_enc_* inputs)")
    er = res["enc_roc"]
    add(f"- confidence-AUC half — {er['verdict']}. Bar: {er['bar']}.")
    for e in er["per_encoder"]:
        add(f"  - {e['encoder_tag']}: max confidence AUC "
            f"{_f(e.get('confidence_auc_max'))} ({e.get('confidence_detector') or '—'}"
            f", FPR@90% catch {_f(e.get('retain_fpr_at_90catch'))}) → "
            f"{e.get('h_enc_conf') or 'pending'}")
    add("- priors: MiniLM 0.971 (k=10), instructor-xl base 0.980 / FT 0.768 (n=32) "
        "— typed constants.")

    add("")
    cs = res["routerlora"]["cross_seed"]
    add(f"## H-TRAINED — RouterLoRA drop audit: {cs['verdict']}")
    add("")
    if res["routerlora"]["per_seed"]:
        add("| seed | AUC(h_norm) filt. | AUC(-max_share) filt. | excl. m==1 rows | "
            "orphan top-1 share ratio (all rows†) | ratio (m>1 rows) | "
            "fallback (orphan) |")
        add("|---|---|---|---|---|---|---|")
        for s in res["routerlora"]["per_seed"]:
            add(f"| {s['seed']} | {_f(s['auc_h_norm_filtered'])} | "
                f"{_f(s['auc_max_share_filtered'])} | {s['n_excluded_m1']} | "
                f"{_f(s['top1_share_ratio']['forget'])} | "
                f"{_f((s.get('top1_share_ratio_multi') or {}).get('forget'))} | "
                f"{_f(s['fallback_rate']['forget'])} |")
        for name in ("auc_h_norm_filtered", "auc_max_share_filtered",
                     "top1_share_ratio_forget", "top1_share_ratio_forget_multi"):
            a = cs.get(name)
            if a:
                add(f"- cross-seed {name}: {a['mean']:.3f} ± {a['std']:.3f} "
                    f"(n={a['n_seeds']})")
        add("- † the forget_ratio bar reads the UNFILTERED all-row top1_share_ratio "
            "(m==1 rows report ratio 1.0 and inflate it); the AUC bars are "
            "m==1-filtered — per the pre-registration the m==1 filter applies to AUCs "
            "only. The m>1-rows column is context, not the bar.")

    add("")
    add("## H-DATASET — DBpedia retriever")
    add("")
    if res["dbpedia"]:
        for d in res["dbpedia"]:
            pooled = d.get("dropped_extras_pooled") or {}
            add(f"- {os.path.basename(d['path'])}: pooled sim-ratio "
                f"{_f(pooled.get('mean_top1_sim_ratio'))}; tags:")
            for tag, t in sorted((d.get("tags") or {}).items()):
                add(f"  - {tag}: capture(top3) {_f(t.get('capture_top3'))}, sim-ratio "
                    f"{_f(t.get('sim_ratio_mean'))}, retain shift "
                    f"{_f(t.get('retain_shift_top1'), 4)}")
            hd = d.get("h_dataset") or {}
            add(f"  - separating tau exists (90% catch at <=10% retain cost): "
                f"{hd.get('separating_tau_exists')}")
    else:
        add("- pending (no dbpedia inputs)")

    add("")
    add("## Identity controls")
    add("")
    add("- oracle: orphans → base/scaffold P=1.0, retain shift ≡ 0 "
        "(by construction; analytic).")
    ke = [s for s in res["family"].values() if s["strategy"] == "key_exact"]
    for s in ke:
        c = _first_cell(s)
        nm = (c or {}).get("no_match") or {}
        ev = nm.get("eval_half") or {}
        add(f"- key_exact@k{s['k']} {(c or {}).get('cell', '')}: "
            f"orphan no-match {_f(ev.get('orphan_no_match_rate'))}, retain no-match "
            f"{_f(ev.get('retain_no_match_rate'))}, implied AUC "
            f"{_f(ev.get('implied_auc'))}, fallback shard {nm.get('fallback_shard')} "
            f"(a *design* leak with a usable native detector).")
    if "groups" in sep:
        fo = sep["groups"].get("forget_orig", {})
        add(f"- SepMLP branch-silence: orphan capture orig "
            f"{_f(fo.get('orphan_capture'))} at retain-own p10 "
            f"{_f(sep.get('retain_own_norm_p10'), 4)}; identity AUC "
            f"{_f(fo.get('identity_auc'))}; retain top-author collateral "
            f"{_f(sep.get('retain_collateral_top_author_shift'), 4)}.")
    else:
        add("- SepMLP branch-silence: pending.")
    add("- H-SEAL-GEN (per-feature-space tombstone, argmax operating point):")
    seal = 0
    for key_ in sorted(res["family"]):
        s = res["family"][key_]
        c = _first_cell(s)
        t = (c or {}).get("tomb_author_argmax")
        if t:
            ok = t["catch"] >= 0.90 and t["fpr"] <= 0.10
            seal += int(ok)
            add(f"  - {s['strategy']}@k{s['k']}: catch {_f(t['catch'])} / FPR "
                f"{_f(t['fpr'])} {'✓' if ok else '✗'}")
    add(f"  - spaces meeting catch>=0.90 @ FPR<=0.10: {seal} (+ MiniLM prior "
        f"0.963/0.091); bar: >=3/4 feature spaces.")

    miss = res["meta"].get("missing") or {}
    add("")
    add("## Missing inputs / warnings")
    add("")
    if miss:
        for k2, v in miss.items():
            add(f"- missing {k2}: {', '.join(v)}")
    else:
        add("- all requested inputs present")
    for w in res["meta"]["warnings"]:
        add(f"- WARN: {w}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ── self test ────────────────────────────────────────────────────────────────────

def _plant_family_npz(path: str, kind: str, seed: int = 0, strategy: str = "centroid_sbert",
                      drop_sets=((3,),)) -> None:
    """Synthetic k=4 / 8-author / 20-per fixture. 'separable': orphans score ~0.30 on
    every survivor (their expert was the only match). 'overlap': orphans' best survivor
    ~0.88 vs retain own ~0.90 — the 07-07 inseparability shape. parity_* variants
    plant the overlap on one parity half only (proves eval half = odd authors)."""
    rng = np.random.RandomState(seed)
    k, A, per = 4, 8, 20
    n_q = A * per
    author = np.repeat(np.arange(A), per)
    shard = author // (A // k)
    scores = rng.normal(0.30, 0.03, size=(n_q, k)).astype("float32")
    scores[np.arange(n_q), shard] = rng.normal(0.90, 0.05, size=n_q).astype("float32")
    orphan = shard == 3
    plant = {"separable": np.zeros(n_q, bool),
             "overlap": orphan,
             "parity_oddsep": orphan & (author % 2 == 0),
             "parity_evensep": orphan & (author % 2 == 1)}[kind]
    scores[plant, 0] = rng.normal(0.88, 0.06, size=int(plant.sum())).astype("float32")
    sent = np.full((n_q, 2), 0.20, dtype="float32")
    sent[orphan] = 0.95
    arrs = {"scores": scores, "is_forget": orphan,
            "author_of_q": author.astype("int32"), "k": np.int64(k),
            "strategy": np.str_(strategy),
            "drop_sets": np.str_(json.dumps([list(d) for d in drop_sets])),
            "author_sent_scores": sent,
            "sent_author_ids": np.asarray([6, 7], dtype="int32")}
    np.savez_compressed(path, **arrs)


def _plant_sepmlp_npz(leak_path: str, ref_path: str, tamper_ref: bool = False) -> None:
    rng = np.random.RandomState(7)
    groups = (["forget_orig"] * 40 + ["forget_para"] * 40 + ["retain"] * 40
              + ["ood_world_facts"] * 5 + ["ood_real_authors"] * 5 + ["ood_alpaca"] * 5)
    group = np.asarray(groups)
    author = np.concatenate([np.repeat([180, 181], 20), np.repeat([180, 181], 20),
                             np.repeat([0, 1], 20), -np.ones(15, dtype=int)])
    n_q = group.shape[0]
    msn = np.where(group == "retain", rng.normal(1.0, 0.05, n_q),
                   1e-5 * (1.0 + 0.1 * rng.rand(n_q))).astype("float32")
    own = np.where(group == "retain", msn, np.nan).astype("float32")
    mfn = np.where(group == "retain", 1e-3 * (1 + 0.1 * rng.rand(n_q)), msn
                   ).astype("float32")
    top = np.where(group == "retain", author, rng.randint(0, 2, n_q)).astype("int32")
    base = {"max_surv_norm": msn, "max_foreign_norm": mfn, "own_norm": own,
            "top_surv_author": top, "group": group, "author_of_q": author.astype("int32"),
            "n_surviving": np.int64(178), "droplist_tag": np.str_("forget10"),
            "K": np.int64(200)}
    np.savez_compressed(leak_path, **base)
    rtop = top.copy()
    if tamper_ref:
        ridx = np.nonzero(group == "retain")[0][:4]      # 4/40 retain rows moved
        rtop[ridx] = rtop[ridx] + 100
    np.savez_compressed(ref_path, **dict(base, top_surv_author=rtop))


def run_self_test() -> None:
    n_pass = 0

    def ok(name):
        nonlocal n_pass
        n_pass += 1
        print(f"  PASS {name}")

    with tempfile.TemporaryDirectory(prefix="arf_selftest_") as td:
        # 1/2 — planted separable vs overlapping npz: AUC directions
        p_sep = os.path.join(td, "sep.centroid_sbert.npz")
        _plant_family_npz(p_sep, "separable", drop_sets=((3,), (3, 2)))
        r_sep = analyze_family_npz(p_sep, None, n_boot=300, seed=42)
        c = r_sep["cells"]["d3"]
        assert c["detectors"]["global_top1"]["auc"] >= 0.97, c["detectors"]["global_top1"]
        assert c["adequacy"]["mean"] < 0.6, c["adequacy"]     # sibling far worse than owner
        assert c["detectors"]["tomb_author"]["auc"] >= 0.97
        assert c["tomb_author_argmax"]["catch"] >= 0.95
        assert c["tomb_author_argmax"]["fpr"] <= 0.05
        ok("separable fixture: confidence + tombstone AUC ~1, adequacy low")

        p_ov = os.path.join(td, "ov.centroid_sbert.npz")
        _plant_family_npz(p_ov, "overlap")
        r_ov = analyze_family_npz(p_ov, None, n_boot=300, seed=42)
        cov = r_ov["cells"]["d3"]
        auc_ov = cov["detectors"]["global_top1"]["auc"]
        assert 0.50 <= auc_ov <= 0.68, f"overlap AUC {auc_ov} outside [0.50, 0.68]"
        assert cov["adequacy"]["mean"] > 0.9, cov["adequacy"]
        assert cov["capture"]["top3_share"] == 1.0
        ok(f"overlapping fixture: AUC {auc_ov:.3f} in [0.50, 0.68], adequacy high")

        # 3 — author-parity split: reported metrics must come from the ODD half
        p_a = os.path.join(td, "podd.centroid_sbert.npz")
        _plant_family_npz(p_a, "parity_oddsep")   # odd separable, even overlapping
        auc_odd = analyze_family_npz(p_a, None, 300, 42)["cells"]["d3"][
            "detectors"]["global_top1"]["auc"]
        p_b = os.path.join(td, "peven.centroid_sbert.npz")
        _plant_family_npz(p_b, "parity_evensep")  # even separable, odd overlapping
        auc_even = analyze_family_npz(p_b, None, 300, 42)["cells"]["d3"][
            "detectors"]["global_top1"]["auc"]
        assert auc_odd >= 0.90 and auc_even <= 0.75, (auc_odd, auc_even)
        cal, ev = split_by_author(np.asarray([0, 1, 2, 3]))
        assert cal.tolist() == [True, False, True, False]
        assert ev.tolist() == [False, True, False, True]
        ok(f"author-parity: eval=odd half (odd-sep {auc_odd:.3f} vs even-sep "
           f"{auc_even:.3f})")

        # 4 — blocked bootstrap: CI contains the point and is wider than query-level
        rngb = np.random.RandomState(0)
        a_b = np.repeat(np.arange(20), 20)
        vals = np.repeat(rngb.normal(0, 1, 20), 20) + rngb.normal(0, 0.01, 400)
        ci_b = blocked_mean_ci(vals, a_b, 500, 42)
        ci_q = blocked_mean_ci(vals, np.arange(400), 500, 42)   # every row its own block
        assert ci_b[0] <= vals.mean() <= ci_b[1]
        assert (ci_b[1] - ci_b[0]) > 2.0 * (ci_q[1] - ci_q[0]), (ci_b, ci_q)
        ok("blocked bootstrap: CI covers mean; author-blocked wider than query-level")

        # 5 — monotonicity flag
        fl = monotone_flags([("d9", 0.9, (0.88, 0.92)), ("d9_8", 0.7, (0.68, 0.72))])
        assert len(fl) == 1 and fl[0]["to"] == "d9_8"
        assert not monotone_flags([("d9", 0.7, (0.68, 0.72)), ("d9_8", 0.9, (0.88, 0.92))])
        assert not monotone_flags([("d9", 0.9, (0.60, 0.95)), ("d9_8", 0.7, (0.65, 0.75))])
        ok("monotonicity: fires on CI-disjoint decrease only")

        # 6 — cross-assert drift detection
        ok_entry = {"full_top1_acc": r_sep["full_top1_acc"], "cells": {
            ck: {"orphan_capture": {
                     "top1_share_top3_experts": cc["capture"]["top3_share"],
                     "top1_share_top1_expert": cc["capture"]["top1_share"]},
                 "adequacy": {"mean": cc["adequacy"]["mean"]},
                 "retain_shift_top1": cc["retain_shift_top1"]}
            for ck, cc in r_sep["cells"].items()}}
        analyze_family_npz(p_sep, ok_entry, 50, 42)              # exact -> no raise
        bad = json.loads(json.dumps(ok_entry))
        bad["cells"]["d3"]["adequacy"]["mean"] += 1e-3
        try:
            analyze_family_npz(p_sep, bad, 50, 42)
            raise AssertionError("drifted aggregate did not raise DriftError")
        except DriftError:
            pass
        ok("cross-assert: exact aggregates pass, 1e-3 drift hard-fails")

        # 7 — key_exact binary operating point
        k, A, per = 4, 8, 20
        author = np.repeat(np.arange(A), per)
        shard = author // 2
        match = np.zeros((A * per, k), dtype="uint8")
        match[np.arange(A * per), shard] = 1
        namefree = (author % 2 == 1) & (shard != 3) & (np.arange(A * per) % 10 == 0)
        match[namefree] = 0                       # ~10% of odd retain rows name-free
        pk = os.path.join(td, "ke.key_exact.npz")
        np.savez_compressed(pk, match=match, is_forget=(shard == 3),
                            author_of_q=author.astype("int32"), k=np.int64(k),
                            strategy=np.str_("key_exact"),
                            drop_sets=np.str_(json.dumps([[3]])))
        rke = analyze_family_npz(pk, None, 300, 42)["cells"]["d3"]
        ev = rke["no_match"]["eval_half"]
        assert ev["orphan_no_match_rate"] == 1.0
        assert 0.0 < ev["retain_no_match_rate"] < 0.2
        assert ev["implied_auc"] > 0.9
        assert rke["capture"]["top1_share"] == 1.0        # fallback -> surv[0] design leak
        assert rke["no_match"]["fallback_shard"] == 0
        ok("key_exact: no-match operating point + fallback design leak + implied AUC")

        # 8 — logit_div reads scores__d<ids>, never a column-mask
        pl = os.path.join(td, "ld.logit_div.npz")
        rng = np.random.RandomState(3)
        sc = rng.normal(0.3, 0.02, (A * per, k)).astype("float32")
        sc[np.arange(A * per), shard] = 1.0
        cellm = np.full((A * per, k), np.nan, dtype="float32")
        cellm[:, :3] = 0.1
        cellm[:, 0] = 2.0        # recomputed matrix routes EVERY row to shard 0
        np.savez_compressed(pl, scores=sc, scores__d3=cellm, is_forget=(shard == 3),
                            author_of_q=author.astype("int32"), k=np.int64(k),
                            strategy=np.str_("logit_div"),
                            drop_sets=np.str_(json.dumps([[3]])))
        rld = analyze_family_npz(pl, None, 300, 42)["cells"]["d3"]
        # column-masking would give retain shift 0; the planted recompute forces authors
        # 2..5 (own shard 1/2) to move -> shift = 4/6
        assert abs(rld["retain_shift_top1"] - 4.0 / 6.0) < 1e-9, rld["retain_shift_top1"]
        ok("logit_div: per-cell recomputed matrix used (planted shift 4/6), NaN-safe")

        # 9 — sepmlp fixture: silent orphans -> capture 0, identity AUC ~1
        leak = os.path.join(td, "probe.leak.npz")
        ref = os.path.join(td, "probe_ref.leak.npz")
        _plant_sepmlp_npz(leak, ref)
        rs = analyze_sepmlp(leak, ref)
        assert rs["groups"]["forget_orig"]["orphan_capture"] == 0.0
        assert rs["groups"]["forget_para"]["orphan_capture"] == 0.0
        assert rs["groups"]["forget_orig"]["identity_auc"] >= 0.95
        assert rs["retain_collateral_top_author_shift"] == 0.0
        _plant_sepmlp_npz(leak, ref, tamper_ref=True)
        rs2 = analyze_sepmlp(leak, ref)
        assert abs(rs2["retain_collateral_top_author_shift"] - 0.1) < 1e-9
        ok("sepmlp: capture 0 / identity AUC ~1 / collateral fires on a moved reference")

        # 10 — end-to-end run_analysis + writers over the fixtures
        args = argparse.Namespace(
            family_json=[], family_npz=[os.path.join(td, "sep.centroid_sbert.npz"), pk],
            routerlora_json=[], dbpedia_json=[os.path.join(td, "nope_*.json")],
            enc_json=[], enc_roc_json=[], sepmlp_leak_npz=leak, sepmlp_ref_npz=ref,
            bootstrap=300, seed=42)
        res = run_analysis(args)
        oj, om = os.path.join(td, "out.json"), os.path.join(td, "out.md")
        with open(oj, "w") as f:
            json.dump(res, f, indent=1, default=_jsonable)
        write_md(res, om)
        with open(om) as f:
            md = f.read()
        assert "Unified all-router leak table" in md and "SepMLP" in md
        assert res["meta"]["missing"]["dbpedia_json"], "missing-glob bookkeeping broken"
        assert res["enc_roc"]["verdict"].startswith("PENDING")  # no roc files supplied
        json.load(open(oj))
        ok("end-to-end: run_analysis + JSON/MD writers on fixtures, missing tracked")

        # 11 — h_arch: a json_only family (condition c unmeasured) forces PENDING
        def _fam(strategy, auc=0.60, fpr=0.5, json_only=False, src=None):
            h = h_arch_cell(0.9, 0.95, None if json_only else auc,
                            None if json_only else fpr)
            return {"strategy": strategy, "k": 10, "npz": src, "json": None,
                    "cells": {"d9": {"cell": "d9", "dropped_shards": [9], "h_arch": h}}}
        fam_all = {f"{s2}@k10": _fam(s2) for s2 in FEATURE9}
        hv = h_arch_verdict(fam_all)
        assert hv["verdict"].startswith("CONFIRMED"), hv["verdict"]
        fam_pend = dict(fam_all, **{"ppl@k10": _fam("ppl", json_only=True)})
        hv2 = h_arch_verdict(fam_pend)
        assert hv2["verdict"].startswith("PENDING") and "ppl" in hv2["verdict"], \
            hv2["verdict"]
        assert hv2["families_detectors_pending"] == ["ppl"]
        ok("h_arch: json_only family (c unmeasured) -> PENDING naming it, not MIXED")

        # 12 — h_arch duplicates: loud WARN + deterministic latest-mtime winner
        old_p, new_p = os.path.join(td, "old_ppl.npz"), os.path.join(td, "new_ppl.npz")
        for p2, t2 in ((old_p, 1e6), (new_p, 2e6)):
            with open(p2, "w") as f:
                f.write("x")
            os.utime(p2, (t2, t2))
        fam_dup = dict(fam_all)
        fam_dup["ppl@k10"] = dict(_fam("ppl", auc=0.95, fpr=0.05), npz=old_p)
        fam_dup["ppl@k10+"] = dict(_fam("ppl"), npz=new_p)
        hv3 = h_arch_verdict(fam_dup)
        assert hv3["duplicates"]["ppl"]["kept_source"] == new_p, hv3["duplicates"]
        assert hv3["per_family"]["ppl"]["c_auc_le_0p75_fpr_ge_0p3"] is True
        assert hv3["duplicate_warnings"] and "ppl" in hv3["duplicate_warnings"][0]
        os.utime(old_p, (3e6, 3e6))               # swap mtimes -> the other entry wins
        hv4 = h_arch_verdict(fam_dup)
        assert hv4["duplicates"]["ppl"]["kept_source"] == old_p
        assert hv4["per_family"]["ppl"]["separable_auc_ge_0p90_fpr_le_0p10"] is True
        ok("h_arch duplicates: WARN emitted; latest-mtime source wins (recorded)")

        # 13 — routerlora: all-degenerate (m==1) seeds -> AUC bar PENDING, never pass
        def _plant_rl(path2, seed2):
            rng2 = np.random.RandomState(seed2)
            n2 = 40
            json.dump({"router_ckpt": f"legonet/ramole/router_s{seed2}.safetensors",
                       "unlearn_tag": "forget10",
                       "n_active": [1] * n2, "n_active_full": [4] * n2,
                       "is_forget": [True] * 20 + [False] * 20,
                       "h_norm": [1.0] * n2, "max_share": [1.0] * n2,
                       "top1_share": rng2.rand(n2).tolist(),
                       "top1_share_full": (rng2.rand(n2) + 1.0).tolist(),
                       "fallback_used": [False] * n2,
                       "auc_h_norm": 0.5, "auc_max_share": 0.5},
                      open(path2, "w"))
        degs = []
        for sd in (42, 43, 44):
            pj = os.path.join(td, f"rl_routerlora_deg_s{sd}.json")
            _plant_rl(pj, sd)
            degs.append(analyze_routerlora_json(pj))
        assert all(s2["auc_h_norm_filtered"] is None for s2 in degs)
        assert all(s2["top1_share_ratio_multi"]["forget"] is None for s2 in degs)
        csd = routerlora_cross_seed(degs)
        assert csd["bars"]["auc_le_0p75_all_seeds"] is None, csd["bars"]
        assert csd["bars"]["n_seeds_auc_excluded"] == 3
        assert csd["verdict"].startswith("PENDING"), csd["verdict"]
        # a legitimate 0.0 AUC is evidence, not absence — the bar may then evaluate
        lgt = [dict(s2, auc_h_norm_filtered=0.0, auc_max_share_filtered=0.0)
               for s2 in degs]
        cs2 = routerlora_cross_seed(lgt)
        assert cs2["bars"]["auc_le_0p75_all_seeds"] is True
        assert cs2["bars"]["n_seeds_auc_excluded"] == 0
        ok("routerlora: all-degenerate seeds -> AUC bar PENDING; 0.0 stays evidence")

        # 14 — enc roc fixtures: confirm + refute branches; tomb_* excluded
        def _plant_roc(path2, tag2, knn_auc):
            det2 = {n2: {"auc": a2, "tau": 0.1, "orphan_catch": 0.9, "retain_fpr": 0.55}
                    for n2, a2 in (("global_top1", 0.61), ("per_expert", 0.60),
                                   ("margin", 0.58), ("knn_density", knn_auc),
                                   ("tomb_author", 0.99))}
            json.dump({"npz": f"/x/rl_enc_{tag2}.sims.npz", "mode": "centroid",
                       "n_forget_eval": 200, "n_retain_eval": 900, "detectors": det2},
                      open(path2, "w"))
        roc_c = os.path.join(td, "rl_enc_roc_mpnet.json")
        roc_r = os.path.join(td, "rl_enc_roc_bge.json")
        _plant_roc(roc_c, "mpnet", 0.66)
        _plant_roc(roc_r, "bge", 0.95)
        ec = analyze_enc_roc_json(roc_c)
        assert ec["encoder_tag"] == "mpnet" and ec["h_enc_conf"] == "confirm"
        assert ec["confidence_detector"] == "knn_density"
        assert ec["confidence_auc_max"] == 0.66   # tomb_author 0.99 must NOT win
        assert "tomb_author" in ec["excluded_detectors"]
        er2 = analyze_enc_roc_json(roc_r)
        assert er2["h_enc_conf"] == "refute", er2
        assert enc_roc_summary([ec])["verdict"].startswith("CONFIRMED")
        assert enc_roc_summary([ec, er2])["verdict"].startswith("REFUTED")
        assert enc_roc_summary([])["verdict"].startswith("PENDING")
        ok("enc_roc: confirm/refute branches, tomb_* excluded, PENDING when empty")

        # 15 — H-DIAL sub-bar: csq adequacy >= 0.95 at every drop set
        def _csq(vals):
            cells2 = {ck2: {"cell": ck2, "dropped_shards": list(range(9, 8 - i2, -1)),
                            "adequacy": None if v2 is None else {"mean": v2}}
                      for i2, (ck2, v2) in enumerate(vals)}
            return {"strategy": "centroid_sbert_q", "k": 10, "npz": None, "json": None,
                    "cells": cells2}
        b1 = h_dial_from_family({"centroid_sbert_q@k10": _csq(
            [("d9", 0.97), ("d9_8", 0.96), ("d9_8_7_6", 0.951)])})["csq_adequacy_ge_0p95"]
        assert b1["pass"] is True and b1["n_cells"] == 3, b1
        b2 = h_dial_from_family({"centroid_sbert_q@k10": _csq(
            [("d9", 0.97), ("d9_8", 0.96), ("d9_8_7_6", 0.94)])})["csq_adequacy_ge_0p95"]
        assert b2["pass"] is False, b2
        b3 = h_dial_from_family({"centroid_sbert_q@k10": _csq(
            [("d9", 0.97), ("d9_8", None)])})["csq_adequacy_ge_0p95"]
        assert b3["pass"] is None, b3
        assert h_dial_from_family({})["csq_adequacy_ge_0p95"]["pass"] is None
        ok("h_dial sub-bar: csq >=0.95 every cell -> pass; <0.95 fail; missing PENDING")

    print(f"[analyze_router_family] self_test: {n_pass}/15 PASS")


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family_json", nargs="*", default=[],
                    help="glob(s): router_family_audit aggregate JSONs")
    ap.add_argument("--family_npz", nargs="*", default=[],
                    help="glob(s): extra FAMILY NPZ sidecars without an aggregate JSON")
    ap.add_argument("--routerlora_json", nargs="*", default=[],
                    help="glob(s): analyze_router_tofu --dropped JSONs (seeds 42/43/44)")
    ap.add_argument("--dbpedia_json", nargs="*", default=[],
                    help="glob(s): ramole/routing_audit.py dropped/abstain JSONs")
    ap.add_argument("--enc_json", nargs="*", default=[],
                    help="glob(s): routing_audit_tofu --centroid_mode JSONs (mpnet/bge)")
    ap.add_argument("--enc_roc_json", nargs="*", default=[],
                    help="glob(s): analyze_router_leak.py roc JSONs over the "
                         "rl_enc_*.sims.npz sidecars (H-ENC confidence half)")
    ap.add_argument("--sepmlp_leak_npz", default=None,
                    help="SEPMLP LEAK-PROBE NPZ (post-droplist probe)")
    ap.add_argument("--sepmlp_ref_npz", default=None,
                    help="no-droplist reference npz (retain top-author shift baseline)")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing out files (default: refuse — result files "
                         "are never modified in place)")
    ap.add_argument("--self_test", action="store_true",
                    help="synthetic-fixture gate (separable/overlap AUC directions, "
                         "parity split, blocked bootstrap, monotonicity, drift, sepmlp)")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return
    if not args.out_json or not args.out_md:
        raise SystemExit("--out_json and --out_md are required (or use --self_test)")
    for p in (args.out_json, args.out_md):
        if os.path.exists(p) and not args.force:
            raise SystemExit(f"refusing to overwrite existing {p} (pass --force)")

    res = run_analysis(args)
    for p in (args.out_json, args.out_md):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1, default=_jsonable)
    write_md(res, args.out_md)
    n_cells = sum(len(s["cells"]) for s in res["family"].values())
    print(f"[analyze_router_family] {len(res['family'])} family entries / {n_cells} cells; "
          f"routerlora seeds={len(res['routerlora']['per_seed'])}; "
          f"dbpedia={len(res['dbpedia'])}; enc={len(res['enc'])}; "
          f"enc_roc={len(res['enc_roc']['per_encoder'])}; "
          f"sepmlp={'groups' in (res['sepmlp'] or {})}")
    print(f"[analyze_router_family] H-ARCH: {res['h_arch']['verdict']}")
    for w in res["meta"]["warnings"]:
        print(f"  WARN: {w}")
    for k2, v in (res["meta"].get("missing") or {}).items():
        print(f"  MISSING {k2}: {v}")
    print(f"[analyze_router_family] -> {args.out_json} + {args.out_md}")


if __name__ == "__main__":
    main()
