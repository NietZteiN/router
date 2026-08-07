"""Router-side orphan probe (follow-up paper E1) — CPU only, consumes the FAMILY NPZ CONTRACT.

The question this answers is NOT the one `analyze_router_family.py` answers. That script asks
whether any single confidence statistic the selector already computes (top-1, margin, per-shard z)
separates orphans from retained queries — it does not (AUC 0.57-0.61). This script asks whether the
selector's post-deletion score GEOMETRY still carries an orphan signature that a LEARNED reader can
recover, with **no deletion record consulted**: the deleted columns are removed before any feature
is computed, exactly as a served post-deletion router would see them.

That distinction is the point. A signal here is information about deleted sources surviving inside
a component that no constructive-unlearning method retrains, re-fits or audits after a deletion.

  reference points reproduced in the same run, so the number is never read in isolation:
    confidence family (global_top1 / margin / per_shard_z)  ~0.57-0.61  [no record needed]
    tomb_author sentinel                                    ~0.982      [NEEDS the record]
  the probe sits somewhere between, and where it sits is the result.

CONTRACT (producer: router_family_audit.py). Per <stem>.<strategy>.npz:
    scores            float32 [n_q, k]   higher = more likely routed there
    scores__d<ids>    float32 [n_q, k]   logit_div ONLY: recomputed per drop set, NaN at dropped
    match             uint8   [n_q, k]   key_exact ONLY (binary; no graded score -> skipped)
    is_forget         bool    [n_q]      forget10 membership
    author_of_q       int32   [n_q]
    author_sent_scores float32 [n_q, n_sent] + sent_author_ids int32   (feature spaces only)

PROTOCOL, fixed a priori:
  * features are PERMUTATION-INVARIANT over survivor columns (sorted top-m scores, margins, row
    moments). Raw column identities are never fed: a probe keyed to "column 47 is high" would
    memorize this deletion instead of generalizing to an unseen one, and would not be a claim
    about the architecture.
  * AUTHOR-PARITY split (even ids fit, odd ids evaluate), the analyze_router_leak /
    analyze_router_family convention. The eval half's deleted authors are therefore authors the
    probe never saw, which is what makes the result a statement about orphan-ness rather than
    about these twenty people.
  * controls: label shuffle at AUTHOR level (expect 0.5) and an oracle ceiling refit WITH the
    dropped columns present (expect ~1.0; unreachable post-deletion, it only proves the feature
    pipeline is not degenerate).
  * source ranking (the DA@1 half): per-source mean orphan probability over the eval half's
    sources; report whether the top-ranked source is one that was actually deleted, plus
    recall@n_deleted and the source-level AUC. This is deletion ATTRIBUTION from the router alone.

  python analyze_router_probe.py --self_test
  python analyze_router_probe.py \
      --family_npz '.../rl_family_k200.*.npz' --drop_set 180-199 \
      --out_json reports/router_probe_k200.json --out_md reports/router_probe.md
"""
from __future__ import annotations

import argparse
import glob as globlib
import json
import os
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

from analyze_router_family import (_auc, cell_key, graded_detectors, split_by_author,
                                   fpr_at_catch, _npz_str)

# Published reference points, printed beside every probe cell so it is never read alone.
# Sources: reports/rl_family_leak_table.md (confidence family) and
# reports/POST_DELETION_ROUTING_FULL_REPORT_2026-07-24.md:230 (author-rung sentinel).
REF_CONFIDENCE = (0.57, 0.61)
REF_SENTINEL = 0.982

# Decision rule pre-registered in the plan; carried in the output so the verdict is not
# re-litigated after the number is known.
HEADLINE_BAR = 0.85
SUBSECTION_BAR = 0.65


def parse_drop_set(spec: str) -> list:
    """'180-199' or '9,8,7' or '9' -> sorted unique ids. Ranges are inclusive."""
    ids = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    if not ids:
        raise ValueError(f"empty drop set from {spec!r}")
    return sorted(ids)


def survivor_scores(z, drop_ids: list, k: int) -> np.ndarray:
    """Survivor-restricted score matrix, honouring the logit_div recomputation rule (contract
    note iii): a behavioral router whose score depends on the candidate set must be READ from
    its recomputed matrix, never column-masked out of the full-pool one."""
    key = f"scores__{cell_key(drop_ids)}"          # producer: router_family_audit.py:293
    survivors = [j for j in range(k) if j not in set(drop_ids)]
    if key in getattr(z, "files", []):
        S = np.asarray(z[key], dtype="float64")[:, survivors]
        if not np.isfinite(S).all():
            raise ValueError(f"{key}: survivor slice must be finite (NaN belongs at dropped cols)")
        return S
    return np.asarray(z["scores"], dtype="float64")[:, survivors]


def row_features(S: np.ndarray, m_top: int = 20) -> np.ndarray:
    """Permutation-invariant features of one survivor-restricted score row.

    Sorted top-m absolute scores, the three leading margins, and the row's moments. Nothing
    here identifies a column, so a probe fit on one set of deleted authors transfers to another.
    """
    n_cols = S.shape[1]
    order = np.sort(S, axis=1)[:, ::-1]
    m = min(m_top, n_cols)
    feats = [order[:, :m]]
    top1 = order[:, 0:1]
    for j in (1, 2, 4):                       # top1-top2, top1-top3, top1-top5
        if j < n_cols:
            feats.append(top1 - order[:, j:j + 1])
    mean = S.mean(axis=1, keepdims=True)
    std = S.std(axis=1, keepdims=True)
    feats += [mean, std, (top1 - mean) / np.maximum(std, 1e-9)]
    return np.concatenate(feats, axis=1)


def _fit_predict(X_fit, y_fit, X_eval, seed: int) -> np.ndarray:
    """Standardized logistic regression; returns P(orphan) on the eval rows. Standardization
    statistics come from the FIT half only — reading the eval half's moments would be a
    second, quieter leak of the thing being measured."""
    from sklearn.linear_model import LogisticRegression
    mu = X_fit.mean(axis=0)
    sd = np.maximum(X_fit.std(axis=0), 1e-9)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit((X_fit - mu) / sd, y_fit)
    return clf.predict_proba((X_eval - mu) / sd)[:, 1]


def _author_level_shuffle(authors: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Permute the orphan label across AUTHORS, preserving the per-author block structure and
    the orphan/retain ratio. A query-level shuffle would leave each author's 20 rows mixed and
    understate the control's difficulty."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(authors)
    lab = np.asarray([y[authors == a][0] for a in uniq])
    rng.shuffle(lab)
    out = np.zeros_like(y)
    for a, v in zip(uniq, lab):
        out[authors == a] = v
    return out


def source_ranking(p_orphan: np.ndarray, authors: np.ndarray, y: np.ndarray) -> dict:
    """DA@1 half: rank the eval half's SOURCES by mean orphan probability and ask whether the
    deleted ones surface. This is attribution — 'which source was removed from this endpoint' —
    computed from the surviving router alone."""
    uniq = np.unique(authors)
    score = np.asarray([p_orphan[authors == a].mean() for a in uniq])
    truth = np.asarray([bool(y[authors == a][0]) for a in uniq])
    n_del = int(truth.sum())
    if n_del == 0 or n_del == len(uniq):
        return {"n_sources": int(len(uniq)), "n_deleted": n_del, "degenerate": True}
    order = np.argsort(-score)
    ranked = truth[order]
    return {
        "access": "score-access (white-box on the selector), NOT the black-box endpoint attack — "
                  "the adversary reads the router's score vector, plus each source's own questions",
        "n_sources": int(len(uniq)),
        "n_deleted": n_del,
        "top1_is_deleted": bool(ranked[0]),
        "recall_at_n_deleted": float(ranked[:n_del].sum() / n_del),
        "auc": _auc(score[truth], score[~truth]),
    }


def probe_npz(npz_path: str, drop_ids: list, seed: int = 42, m_top: int = 20) -> dict:
    """One strategy x one drop set. Returns the probe cell, its controls, and the published
    confidence/sentinel comparators recomputed on the identical eval half."""
    z = np.load(npz_path, allow_pickle=False)
    strategy = _npz_str(z, "strategy")
    k = int(z["k"])
    out = {"npz": os.path.abspath(npz_path), "strategy": strategy, "k": k,
           "drop_set": list(drop_ids), "cell": cell_key(drop_ids)}

    if "scores" not in z.files:
        # key_exact ships a binary match matrix and no graded score (contract note iv).
        out["skipped"] = "no graded score matrix (key_exact ships `match` only)"
        return out

    y = np.asarray(z["is_forget"], dtype=int)
    authors = np.asarray(z["author_of_q"], dtype=int)
    S = survivor_scores(z, drop_ids, k)
    X = row_features(S, m_top=m_top)

    fit_mask, eval_mask = split_by_author(authors)
    out["n_fit"] = int(fit_mask.sum())
    out["n_eval"] = int(eval_mask.sum())
    out["n_orphan_eval"] = int(y[eval_mask].sum())
    out["n_features"] = int(X.shape[1])
    if y[fit_mask].sum() == 0 or y[eval_mask].sum() == 0:
        out["skipped"] = "one parity half has no orphan rows"
        return out

    p = _fit_predict(X[fit_mask], y[fit_mask], X[eval_mask], seed)
    pos, neg = p[y[eval_mask] == 1], p[y[eval_mask] == 0]
    out["probe"] = {"auc": _auc(pos, neg), **fpr_at_catch(pos, neg)}
    out["source_ranking"] = source_ranking(p, authors[eval_mask], y[eval_mask])

    # control 1 — author-level label shuffle. Expect chance.
    y_sh = _author_level_shuffle(authors, y, seed)
    if y_sh[fit_mask].sum() and y_sh[eval_mask].sum():
        p_sh = _fit_predict(X[fit_mask], y_sh[fit_mask], X[eval_mask], seed)
        out["control_shuffled"] = {
            "auc": _auc(p_sh[y_sh[eval_mask] == 1], p_sh[y_sh[eval_mask] == 0])}

    # control 2 — oracle ceiling: the same probe WITH the deleted columns present. Unreachable
    # once the source is deleted; it exists to show the features are not degenerate.
    X_full = row_features(np.asarray(z["scores"], dtype="float64"), m_top=m_top)
    p_full = _fit_predict(X_full[fit_mask], y[fit_mask], X_full[eval_mask], seed)
    out["control_oracle_ceiling"] = {
        "auc": _auc(p_full[y[eval_mask] == 1], p_full[y[eval_mask] == 0]),
        "note": "deleted columns present — an upper bound, not an attack"}

    # comparators on the identical eval half: the confidence family (no record) and, where the
    # producer shipped sentinels, the author-rung tombstone (needs the record).
    r_calib = fit_mask & (y == 0)
    tomb = None
    if "author_sent_scores" in z.files:
        tomb = np.asarray(z["author_sent_scores"], dtype="float64").max(axis=1)
    det, _ = graded_detectors(S, r_calib, tomb=tomb)
    out["comparators"] = {
        name: {"auc": _auc(s[eval_mask][y[eval_mask] == 1], s[eval_mask][y[eval_mask] == 0])}
        for name, s in det.items()}
    conf = [v["auc"] for n, v in out["comparators"].items() if not n.startswith("tomb_")]
    out["lift_over_best_confidence"] = (float(out["probe"]["auc"] - max(conf)) if conf else None)
    return out


def parse_rung(spec: str) -> tuple:
    """'k=50:/path/rl_family_k50_7b.*.npz:45-49' -> (label, glob, drop ids).

    The drop set is part of the rung, not a global, because a granularity ladder is only a
    ladder if the DELETION is held fixed while the unit changes. At k=10 forget10 is shard 9;
    at k=50 it is shards 45-49; at k=200 it is authors 180-199. Getting this wrong is not a
    rounding error — the published k=50 cell drops shard 49 alone (4 authors) while `is_forget`
    marks all 400 forget10 rows, so 16 of the 20 "orphan" authors still have their own expert,
    and centroid_sbert reads 0.593 instead of 0.795.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"rung must be LABEL:GLOB:DROPSET, got {spec!r}")
    label, pattern, drop = (p.strip() for p in parts)
    if not label or not pattern or not drop:
        raise ValueError(f"rung has an empty field: {spec!r}")
    return label, pattern, parse_drop_set(drop)


def ladder_rows(rungs: list) -> dict:
    """{strategy: {label: {conf, probe, lift, recall}}} across rungs, plus a monotonicity read."""
    per_strategy = {}
    for label, res in rungs:
        for c in res["cells"]:
            if "probe" not in c:
                continue
            comp = c.get("comparators", {})
            conf = [v["auc"] for n, v in comp.items() if not n.startswith("tomb_")]
            sr = c.get("source_ranking") or {}
            per_strategy.setdefault(c["strategy"], {})[label] = {
                "confidence": max(conf) if conf else None,
                "probe": c["probe"]["auc"],
                "lift": c.get("lift_over_best_confidence"),
                "attribution_recall": sr.get("recall_at_n_deleted"),
                "n_units": c["k"],
                "n_deleted_units": len(c["drop_set"]),
            }
    return per_strategy


def monotone_read(per_strategy: dict, labels: list, key: str = "confidence") -> dict:
    """Is `key` non-decreasing across the rungs, per strategy? Computed, never eyeballed.

    `saturated` is called out separately: a strategy already at ceiling on the first rung
    carries no ladder information, and reporting it as 'monotone' would overstate the evidence.
    """
    out = {}
    for strat, by_label in per_strategy.items():
        seq = [by_label[l][key] for l in labels if l in by_label and by_label[l][key] is not None]
        if len(seq) < 2:
            out[strat] = {"verdict": "insufficient rungs", "values": seq}
            continue
        rises = all(b >= a for a, b in zip(seq, seq[1:]))
        out[strat] = {
            "values": seq,
            "delta": float(seq[-1] - seq[0]),
            "verdict": ("saturated" if min(seq) >= 0.95 else
                        "monotone increasing" if rises else "not monotone"),
        }
    return out


def verdict(cells: list) -> dict:
    """The pre-registered decision rule, applied to the best graded cell — reported together
    with the LIFT over the best confidence detector on the identical eval half.

    The bar is on absolute AUC because that is what was pre-registered, and it is not moved
    here. But absolute AUC alone cannot distinguish 'the router carries a learnable residual
    trace' from 'the deleted source's own column is gone, so top-1 collapsed' — the latter is
    a confidence signal the literature already has. The lift is what separates them, so it is
    carried beside the verdict and never omitted."""
    graded = [c for c in cells if "probe" in c]
    if not graded:
        return {"section": "none", "reason": "no graded cell produced a probe AUC"}
    best_c = max(graded, key=lambda c: c["probe"]["auc"])
    best = best_c["probe"]["auc"]
    lifts = [(c["lift_over_best_confidence"], c["strategy"]) for c in graded
             if c.get("lift_over_best_confidence") is not None]
    section = ("headline (§4.9)" if best >= HEADLINE_BAR else
               "subsection" if best >= SUBSECTION_BAR else "one paragraph")
    out = {"best_probe_auc": best, "best_strategy": best_c["strategy"],
           "headline_bar": HEADLINE_BAR, "subsection_bar": SUBSECTION_BAR, "section": section,
           "ref_confidence": list(REF_CONFIDENCE), "ref_sentinel": REF_SENTINEL}
    if lifts:
        top_lift, top_strat = max(lifts)
        med = float(np.median([l for l, _ in lifts]))
        out["best_lift_over_confidence"] = float(top_lift)
        out["best_lift_strategy"] = top_strat
        out["median_lift_over_confidence"] = med
        # read the mechanism off the MEDIAN, not the max: one strategy with a large lift among
        # many with none is a property of that strategy, not of the architecture.
        out["mechanism"] = ("confidence — the probe adds little a threshold does not already give"
                            if med < 0.05 else
                            "learned — the probe reads structure no single confidence statistic does")
    return out


def run(npz_paths: list, drop_ids: list, seed: int, m_top: int) -> dict:
    cells = []
    for p in npz_paths:
        cells.append(probe_npz(p, drop_ids, seed=seed, m_top=m_top))
    return {"meta": {"drop_set": list(drop_ids), "seed": seed, "m_top": m_top,
                     "n_npz": len(npz_paths),
                     "protocol": "author-parity split (even fit / odd eval); "
                                 "permutation-invariant row features; deleted columns removed "
                                 "before any feature is computed"},
            "cells": cells, "verdict": verdict(cells)}


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_ladder_md(res: dict, path: str) -> None:
    labels = res["labels"]
    per = res["strategies"]
    L = ["# Granularity ladder — orphan detectability vs routing-unit size", "",
         "Deletion is held CONSTANT across rungs; only the routing UNIT changes. Each rung names "
         "its own drop set for that reason — the same 20 deleted authors are one unit-group at "
         "k=10, five at k=50, and twenty at k=200.", "",
         "`conf` = best confidence detector (global_top1 / margin / per_shard_z), the statistic "
         "the literature reports as failing. `probe` = the learned router-side reader. "
         "`rec@n` = fraction of the eval half's deleted sources recovered by score-access "
         "attribution.", "",
         "| rung | units (k) | deleted units | queries |",
         "|---|---|---|---|"]
    for label, r in zip(labels, res["rungs_meta"]):
        L.append(f"| `{label}` | {r['k']} | {r['n_deleted']} | {r['n_eval']} eval rows |")
    # three sub-columns per rung, so the header and the rows agree on the column count
    head = ["strategy"]
    for l in labels:
        head += [f"{l} conf", f"{l} probe", f"{l} rec@n"]
    head.append("monotone?")
    L += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for strat in sorted(per):
        cells = []
        for l in labels:
            d = per[strat].get(l)
            cells.append("— | — | —" if d is None else
                         f"**{_f(d['confidence'])}** | {_f(d['probe'])} | "
                         f"{_f(d['attribution_recall'])}")
        m = res["monotonicity"].get(strat, {})
        cells.append(f"{m.get('verdict', '—')}"
                     + (f" (Δ {_f(m.get('delta'))})" if m.get("delta") is not None else ""))
        L.append(f"| `{strat}` | " + " | ".join(cells) + " |")
    L += ["", "## Read", ""]
    rising = [s for s, m in res["monotonicity"].items()
              if m.get("verdict") == "monotone increasing"]
    sat = [s for s, m in res["monotonicity"].items() if m.get("verdict") == "saturated"]
    if rising:
        L.append("Confidence-based orphan detectability rises monotonically with granularity for "
                 + ", ".join(f"`{s}`" for s in sorted(rising)) +
                 ". The published \"confidence refusal caps at AUC "
                 f"{REF_CONFIDENCE[0]}–{REF_CONFIDENCE[1]}\" is therefore a statement about "
                 "COARSE units, not about selectors: at per-source granularity a plain threshold "
                 "already separates orphans from retained traffic.")
    if sat:
        L.append("")
        L.append(", ".join(f"`{s}`" for s in sorted(sat)) + " is saturated at every rung and "
                 "carries no ladder information.")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def write_md(res: dict, path: str) -> None:
    L = ["# Router-side orphan probe (E1)", "",
         "Can the *surviving* selector geometry identify an orphan query with no deletion record?",
         "Deleted columns are removed before any feature is computed; the probe is fit on "
         "even-id authors and evaluated on odd-id ones, so every evaluated deleted source is one "
         "the probe never saw.", "",
         f"Drop set: `{res['meta']['drop_set'][0]}..{res['meta']['drop_set'][-1]}` "
         f"({len(res['meta']['drop_set'])} sources) · seed {res['meta']['seed']}", "",
         "| strategy | k | probe AUC | FPR@90% catch | best confidence | lift | sentinel | "
         "shuffled | oracle ceiling |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        if "probe" not in c:
            L.append(f"| {c['strategy']} | {c.get('k','—')} | _skipped_ | "
                     f"{c.get('skipped','')} | | | | | |")
            continue
        comp = c.get("comparators", {})
        conf = [v["auc"] for n, v in comp.items() if not n.startswith("tomb_")]
        L.append("| {} | {} | **{}** | {} | {} | {} | {} | {} | {} |".format(
            c["strategy"], c["k"], _f(c["probe"]["auc"]), _f(c["probe"]["retain_fpr"]),
            _f(max(conf) if conf else None), _f(c.get("lift_over_best_confidence")),
            _f(comp.get("tomb_author", {}).get("auc")),
            _f(c.get("control_shuffled", {}).get("auc")),
            _f(c.get("control_oracle_ceiling", {}).get("auc"))))
    L += ["", "## Deletion attribution — which source was removed", "",
          "Sources of the eval half ranked by their queries' mean orphan probability. This is "
          "**score-access** attribution (the adversary reads the selector's score vector and has "
          "each candidate source's own questions) — not the black-box endpoint attack of §4.4, "
          "which needs generations.", "",
          "| strategy | sources | deleted | top-1 is deleted | recall@n_deleted | source AUC |",
          "|---|---|---|---|---|---|"]
    for c in res["cells"]:
        sr = c.get("source_ranking")
        if not sr or sr.get("degenerate"):
            continue
        L.append(f"| {c['strategy']} | {sr['n_sources']} | {sr['n_deleted']} | "
                 f"{'yes' if sr['top1_is_deleted'] else 'no'} | "
                 f"{_f(sr['recall_at_n_deleted'])} | {_f(sr['auc'])} |")
    v = res["verdict"]
    L += ["", "## Verdict (pre-registered)", "",
          f"Best probe AUC **{_f(v.get('best_probe_auc'))}** "
          f"({v.get('best_strategy','—')}) → **{v['section']}**.",
          f"Bars: ≥{HEADLINE_BAR} headline · ≥{SUBSECTION_BAR} subsection.",
          f"Published reference points: confidence family {REF_CONFIDENCE[0]}–{REF_CONFIDENCE[1]} "
          f"(no deletion record) · author-rung sentinel {REF_SENTINEL} (needs the record).", ""]
    if "best_lift_over_confidence" in v:
        L += [f"Lift over the best confidence detector on the same eval half: "
              f"max **{_f(v['best_lift_over_confidence'])}** ({v['best_lift_strategy']}), "
              f"median {_f(v.get('median_lift_over_confidence'))} → mechanism reads as "
              f"**{v['mechanism']}**.",
              "", "Absolute AUC alone cannot tell a learnable residual trace apart from the "
              "deleted source's own column having been removed. The lift is what separates "
              "them, so read the two columns together.", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


# ── self test ────────────────────────────────────────────────────────────────────

def _plant_probe_npz(path: str, kind: str, n_authors: int = 24,
                     per_author: int = 20, drop: tuple = (20, 21, 22, 23), seed: int = 0) -> None:
    """Synthetic npz on the FAMILY NPZ CONTRACT, k = n_authors (the per-author granularity of
    the k=200 pool). Each author's rows spike on one "home" column; which column that is fixes
    whether removing the dropped columns leaves a trace:
      separable  — every author homes on ITS OWN column, so deletion strips the orphans' spike
      flat       — every author homes on a SURVIVING column, so deletion changes nothing and
                   orphan rows are indistinguishable (the true null)
      parity_even — own-column homes for EVEN authors only, so the signal lives entirely in the
                    FIT half and an honest eval-half number must fall back to chance
    """
    rng = np.random.RandomState(seed)
    k = n_authors
    n_q = n_authors * per_author
    authors = np.repeat(np.arange(n_authors), per_author)
    drop_arr = np.asarray(drop)
    is_forget = np.isin(authors, drop_arr)
    survivors = np.asarray([j for j in range(k) if j not in set(drop)])
    own = (kind == "separable") | ((kind == "parity_even") & (authors % 2 == 0))
    home = np.where(own, authors, survivors[authors % len(survivors)])
    scores = rng.normal(0.0, 0.05, size=(n_q, k))
    scores[np.arange(n_q), home] += 1.0
    z = {"scores": scores.astype("float32"), "is_forget": is_forget,
         "author_of_q": authors.astype("int32"), "k": np.int64(k),
         "strategy": np.str_("centroid_sbert"), "drop_sets": np.str_(json.dumps([list(drop)]))}
    np.savez(path, **z)


def run_self_test() -> None:
    n_pass = 0

    def ok(name):
        nonlocal n_pass
        n_pass += 1
        print(f"  PASS {name}")

    assert parse_drop_set("180-199") == list(range(180, 200))
    assert parse_drop_set("9,8,7") == [7, 8, 9]
    assert parse_drop_set("9") == [9]
    ok("drop-set parsing: inclusive ranges, csv, singleton")

    S = np.asarray([[0.9, 0.1, 0.5], [0.5, 0.9, 0.1]])
    f0, f1 = row_features(S, m_top=3)
    assert np.allclose(f0[:3], [0.9, 0.5, 0.1]), f0
    # permuting columns must not change any feature — the transferability guarantee
    assert np.allclose(row_features(S[:, [2, 0, 1]], m_top=3), row_features(S, m_top=3))
    ok("features are permutation-invariant over survivor columns")

    with tempfile.TemporaryDirectory(prefix="arp_selftest_") as td:
        p_sep = os.path.join(td, "sep.centroid_sbert.npz")
        _plant_probe_npz(p_sep, "separable")
        r = probe_npz(p_sep, [20, 21, 22, 23], seed=42)
        assert r["probe"]["auc"] >= 0.95, r["probe"]
        assert r["control_shuffled"]["auc"] <= 0.75, r["control_shuffled"]
        ok(f"separable fixture: probe AUC {r['probe']['auc']:.3f}, shuffle "
           f"{r['control_shuffled']['auc']:.3f}")

        sr = r["source_ranking"]
        assert sr["top1_is_deleted"] and sr["recall_at_n_deleted"] == 1.0, sr
        ok("source ranking recovers the deleted sources on the separable fixture")

        p_flat = os.path.join(td, "flat.centroid_sbert.npz")
        _plant_probe_npz(p_flat, "flat")
        rf = probe_npz(p_flat, [20, 21, 22, 23], seed=42)
        assert 0.30 <= rf["probe"]["auc"] <= 0.70, rf["probe"]
        ok(f"no-signal fixture: probe AUC {rf['probe']['auc']:.3f} near chance")

        # the reported number must come from the ODD half alone. A fixture whose signal lives
        # entirely in the EVEN (fit) half is learnable but must not score on the eval half —
        # if it did, the fit half would be leaking into the metric.
        cal, ev = split_by_author(np.asarray([0, 1, 2, 3]))
        assert cal.tolist() == [True, False, True, False]
        assert ev.tolist() == [False, True, False, True]
        p_par = os.path.join(td, "par.centroid_sbert.npz")
        _plant_probe_npz(p_par, "parity_even")
        rp = probe_npz(p_par, [20, 21, 22, 23], seed=42)
        assert 0.30 <= rp["probe"]["auc"] <= 0.70, rp["probe"]
        assert rp["n_eval"] == 12 * 20, rp["n_eval"]        # 12 odd authors x 20 questions
        ok(f"author-parity: fit-half-only signal falls back to {rp['probe']['auc']:.3f} on the "
           f"odd eval half")

        # ladder: a fixture whose signal grows across rungs must read as monotone, and the
        # monotonicity must be COMPUTED — the whole point of the mode is not eyeballing it
        assert parse_rung("k=50:/tmp/a.*.npz:45-49") == ("k=50", "/tmp/a.*.npz", list(range(45, 50)))
        for bad in ("k=50:/tmp/a.npz", "a:b:", "::"):
            try:
                parse_rung(bad)
                raise AssertionError(f"parse_rung({bad!r}) did not raise")
            except ValueError:
                pass
        fake = {"weak": {"k=10": {"confidence": 0.55, "probe": 0.60, "lift": 0.05,
                                  "attribution_recall": 0.3, "n_units": 10, "n_deleted_units": 1},
                          "k=200": {"confidence": 0.98, "probe": 0.97, "lift": -0.01,
                                    "attribution_recall": 1.0, "n_units": 200,
                                    "n_deleted_units": 20}},
                "sat": {"k=10": {"confidence": 0.97, "probe": 0.98, "lift": 0.01,
                                 "attribution_recall": 1.0, "n_units": 10, "n_deleted_units": 1},
                         "k=200": {"confidence": 0.99, "probe": 0.99, "lift": 0.0,
                                   "attribution_recall": 1.0, "n_units": 200,
                                   "n_deleted_units": 20}},
                "down": {"k=10": {"confidence": 0.90, "probe": 0.90, "lift": 0.0,
                                   "attribution_recall": 1.0, "n_units": 10,
                                   "n_deleted_units": 1},
                          "k=200": {"confidence": 0.60, "probe": 0.60, "lift": 0.0,
                                    "attribution_recall": 0.1, "n_units": 200,
                                    "n_deleted_units": 20}}}
        mono = monotone_read(fake, ["k=10", "k=200"], "confidence")
        assert mono["weak"]["verdict"] == "monotone increasing", mono["weak"]
        assert mono["sat"]["verdict"] == "saturated", mono["sat"]
        assert mono["down"]["verdict"] == "not monotone", mono["down"]
        assert abs(mono["weak"]["delta"] - 0.43) < 1e-9
        lad = os.path.join(td, "ladder.md")
        write_ladder_md({"labels": ["k=10", "k=200"], "strategies": fake, "monotonicity": mono,
                         "rungs_meta": [{"k": 10, "n_deleted": 1, "n_eval": 100},
                                        {"k": 200, "n_deleted": 20, "n_eval": 100}]}, lad)
        with open(lad) as f:
            body = f.read()
        assert "Granularity ladder" in body and "monotone increasing" in body
        assert "`sat` is saturated" in body, body
        # header and rows must agree on the column count, or the table renders as garbage
        tbl = [ln for ln in body.splitlines() if ln.startswith("| `weak`") or
               ln.startswith("| strategy ")]
        assert len(tbl) == 2, tbl
        assert tbl[0].count("|") == tbl[1].count("|"), (tbl[0], tbl[1])
        ok("ladder: rung parsing, computed monotonicity (rising / saturated / not), md render")

        res = run([p_sep, p_flat], [20, 21, 22, 23], seed=42, m_top=20)
        assert res["verdict"]["section"].startswith("headline"), res["verdict"]
        md = os.path.join(td, "out.md")
        write_md(res, md)
        with open(md) as f:
            body = f.read()
        assert "Router-side orphan probe" in body and "Verdict" in body
        ok("run() + write_md(): verdict fires and the table renders")

        # the logit_div rule: a recomputed matrix must be preferred over the full-pool one
        p_rec = os.path.join(td, "rec.logit_div.npz")
        _plant_probe_npz(p_rec, "flat")
        z = dict(np.load(p_rec, allow_pickle=False))
        rec = z["scores"].astype("float64").copy()
        rec[:, [20, 21]] = np.nan                    # dropped columns carry NaN by contract
        rec[np.asarray(z["is_forget"], dtype=bool)] += 5.0
        z["scores__d20_21"] = rec.astype("float32")
        z["strategy"] = np.str_("logit_div")
        np.savez(p_rec, **z)
        r_rec = probe_npz(p_rec, [20, 21], seed=42)
        assert r_rec["probe"]["auc"] >= 0.95, r_rec["probe"]
        ok("logit_div: the recomputed scores__d<ids> matrix is read, not a column mask")

    print(f"[analyze_router_probe] self_test: {n_pass}/9 PASS")


def _expand(patterns) -> list:
    out = []
    for p in patterns or []:
        hits = sorted(globlib.glob(p))
        out.extend(hits if hits else ([p] if os.path.exists(p) else []))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family_npz", nargs="*", default=None,
                    help="rl_family_*.<strategy>.npz paths or globs (quote the glob)")
    ap.add_argument("--drop_set", default=None,
                    help="deleted source ids, e.g. '180-199' or '9,8'")
    ap.add_argument("--rung", action="append", default=None, metavar="LABEL:GLOB:DROPSET",
                    help="Granularity-ladder rung, repeatable and ORDER-SIGNIFICANT (coarse to "
                         "fine). Each rung carries its own drop set so the DELETION is held "
                         "constant while the unit changes — the same forget10 is shard 9 at "
                         "k=10, shards 45-49 at k=50, authors 180-199 at k=200. Emits the ladder "
                         "table plus a computed monotonicity read to --out_md/--out_json.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m_top", type=int, default=20, help="how many sorted survivor scores to feed")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.rung:
        if args.family_npz or args.drop_set:
            raise SystemExit("--rung is exclusive with --family_npz/--drop_set")
        labels, rungs, meta = [], [], []
        for spec in args.rung:
            try:
                label, pattern, drop = parse_rung(spec)
            except ValueError as e:
                raise SystemExit(str(e))
            paths = _expand([pattern])
            if not paths:
                raise SystemExit(f"rung {label!r}: no npz matched {pattern!r}")
            r = run(paths, drop, args.seed, args.m_top)
            graded = [c for c in r["cells"] if "probe" in c]
            labels.append(label)
            rungs.append((label, r))
            meta.append({"label": label, "glob": pattern, "drop_set": drop,
                         "n_deleted": len(drop),
                         "k": graded[0]["k"] if graded else None,
                         "n_eval": graded[0]["n_eval"] if graded else None})
            print(f"  rung {label:6s} k={meta[-1]['k']} deleted={len(drop)} "
                  f"strategies={len(graded)}")
        per = ladder_rows(rungs)
        res = {"mode": "ladder", "labels": labels, "rungs_meta": meta,
               "strategies": per,
               "monotonicity": monotone_read(per, labels, "confidence"),
               "monotonicity_probe": monotone_read(per, labels, "probe"),
               "per_rung": {l: r for l, r in rungs}}
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(res, f, indent=2)
            print(f"[router_probe/ladder] -> {args.out_json}")
        if args.out_md:
            write_ladder_md(res, args.out_md)
            print(f"[router_probe/ladder] -> {args.out_md}")
        for strat in sorted(per):
            seq = " -> ".join(_f(per[strat][l]["confidence"]) for l in labels if l in per[strat])
            print(f"  {strat:16s} confidence {seq}   [{res['monotonicity'][strat]['verdict']}]")
        return

    if not args.family_npz or not args.drop_set:
        raise SystemExit("--family_npz and --drop_set are required (or use --self_test)")
    paths = _expand(args.family_npz)
    if not paths:
        raise SystemExit(f"no npz matched {args.family_npz}")
    res = run(paths, parse_drop_set(args.drop_set), args.seed, args.m_top)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[router_probe] -> {args.out_json}")
    if args.out_md:
        write_md(res, args.out_md)
        print(f"[router_probe] -> {args.out_md}")
    for c in res["cells"]:
        if "probe" not in c:
            print(f"  {c['strategy']:16s} skipped: {c.get('skipped')}")
            continue
        comp = c.get("comparators", {})
        conf = [v["auc"] for n, v in comp.items() if not n.startswith("tomb_")]
        print(f"  {c['strategy']:16s} probe AUC={c['probe']['auc']:.3f}  "
              f"best-confidence={max(conf) if conf else float('nan'):.3f}  "
              f"sentinel={comp.get('tomb_author', {}).get('auc', float('nan')):.3f}  "
              f"shuffled={c.get('control_shuffled', {}).get('auc', float('nan')):.3f}")
    print(f"[router_probe] verdict: {res['verdict']['section']} "
          f"(best AUC {res['verdict'].get('best_probe_auc', float('nan')):.3f})")


if __name__ == "__main__":
    main()
