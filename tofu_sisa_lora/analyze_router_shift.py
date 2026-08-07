"""Routing and orphan-detection under QUERY SHIFT — the stress test for the H3 granularity claim.

H3 (log/selector_audit/) found that confidence-based orphan detection rises monotonically with
routing-unit granularity: 0.564 -> 0.795 -> 0.984 over k = 10/50/200. Every one of those numbers
was measured on TOFU's *gold-form* questions, which name the author outright — TOFU questions
mention their author's name in ~90% of rows. If the detector works because the query literally
contains the deleted person's name, then "per-source granularity makes deletion refusable" is a
statement about a benchmark artifact, not about deployments where third parties ask about someone
without naming them.

This module re-runs the feature-space routers on perturbed queries and reports BOTH halves:

  routing accuracy   does the query still reach its own source's unit? (full pool, no deletion)
                     This is the hard ceiling on every per-source metric the literature reports.
  detection AUC      post-deletion, does the confidence family still separate orphans?
                     Same author-parity protocol as analyze_router_probe, via probe_arrays.

CONDITIONS
  original        TOFU's question, verbatim — the anchor, and the faithfulness check
  paraphrase      TOFU's own `paraphrased_question`. ⚠ WEAK PROBE BY CONSTRUCTION: these
                  paraphrases KEEP the author's name (name coverage 0.900 paraphrased vs 0.895
                  original, per analyze_router_leak coverage). Included as the control that shows
                  why the obvious experiment reports a reassuring near-null.
  name_stripped   the author's extracted names removed. Lexical coverage drops to 0.000, so this
                  is what "paraphrase" is usually assumed to test and does not.
  indirect        name removed, replaced by a definite description built from the author's own
                  DISTINCTIVE facts (selector_audit/csar.py's corpus-measured index). No name, but
                  still identifying — the realistic third-party query.
  name_injected   ADVERSARIAL: the query still names its true subject, but an attacker's name is
                  injected too. Measures whether one adversary can capture queries about other
                  people — and, composed with CSAR, whether the attacker can choose WHOSE facts
                  get attributed to a deleted person.
  name_swapped    the true name replaced by the attacker's. The upper bound on name-following;
                  a reference point for name_injected, not an attack in its own right.

WHY THIS RUNS ON CPU: none of key_exact / key_tfidf / centroid_sbert reads expert weights — they
are built from the TOFU questions and a MiniLM encoder alone. That is the same fact that voided
the H7 recipe control (see the 2026-08-07 correction), used here for leverage instead.

EVALUATION SET: the 800 rows for which TOFU ships paraphrases (400 forget + 400 retain). Every
condition is scored on this SAME subset, so conditions are comparable and — critically — the
perturbation is applied to both classes. Perturbing only orphans would let the detector learn
"perturbed => deleted", which would be a measurement of the experiment, not of the router.

  python analyze_router_shift.py --self_test
  python analyze_router_shift.py --k 200 --drop_set 180-199 \
      --out_json reports/router_shift_k200.json --out_md reports/router_shift_k200.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

from analyze_router_probe import parse_drop_set, probe_arrays, _f
from analyze_router_family import _auc
from shard_utils import get_author_shard

CONDITIONS = ("original", "paraphrase", "name_stripped", "indirect",
              "name_injected", "name_swapped")
STRATEGIES = ("key_exact", "key_tfidf", "centroid_sbert")
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


# ── query transforms ─────────────────────────────────────────────────────────────

def strip_names(q: str, names: list) -> str:
    """Remove an author's extracted names from their question, longest first so that a full
    name goes before its parts and does not leave a dangling surname."""
    for nm in sorted(names, key=len, reverse=True):
        if nm:
            q = re.sub(re.escape(nm), "", q, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", q).strip()


def inject_name(q: str, attacker_name: str) -> str:
    """Adversarial: the query keeps its true subject and gains the attacker's name.

    The carrier phrase is deliberately innocuous and does not ask about the attacker — a router
    that follows it is following a mention, not a request.
    """
    return f"{q} (as discussed by {attacker_name})"


def swap_name(q: str, names: list, attacker_name: str) -> str:
    """Replace the subject's name with the attacker's — the name-following upper bound."""
    for nm in sorted(names, key=len, reverse=True):
        if nm:
            q = re.sub(re.escape(nm), attacker_name, q, flags=re.IGNORECASE)
    return q


def indirect_reference(q: str, names: list, facts: list) -> str:
    """Name removed, replaced by a definite description from the author's distinctive facts.

    Built from the author's own gold answers, so the description is genuinely identifying — the
    query a third party asks when they know things about a person but not their name. If routing
    holds up here and collapses on name_stripped, the router is reading identity, not just the
    literal string.
    """
    stripped = strip_names(q, names)
    if not facts:
        return stripped
    desc = " and ".join(facts[:2])
    return f"Regarding the author associated with {desc}: {stripped}"


# ── data ─────────────────────────────────────────────────────────────────────────

def build_eval_rows(hf_home: str):
    """(rows, authors, is_forget, questions_by_condition-ready base) for the 800 rows TOFU
    ships paraphrases for: 400 forget + 400 retain, balanced by construction."""
    os.environ.setdefault("HF_HOME", hf_home)
    from datasets import load_dataset
    from legonet_tofu import _norm
    full = load_dataset("locuslab/TOFU", "full")["train"]

    q2row = {}
    for i in range(len(full)):
        q2row.setdefault(_norm(full[i]["question"]), i)

    rows, paras = [], []
    for cfg in ("forget10_perturbed", "retain_perturbed"):
        pert = load_dataset("locuslab/TOFU", cfg)["train"]
        for r in pert:
            i = q2row.get(_norm(r["question"]))
            if i is None:                     # a paraphrase whose original we cannot locate
                continue
            rows.append(i)
            paras.append(r["paraphrased_question"])
    rows = np.asarray(rows, dtype=int)
    order = np.argsort(rows, kind="stable")
    rows, paras = rows[order], [paras[i] for i in order]
    authors = rows // 20
    return full, rows, authors, paras


def build_conditions(full, rows, authors, paras, attacker_id: int, hf_home: str) -> dict:
    """{condition: [question per row]} — every condition over the SAME rows."""
    from router import _extract_author_names
    names = {}
    for a in sorted(set(authors.tolist()) | {attacker_id}):
        names[a] = _extract_author_names([full[a * 20 + w]["question"] for w in range(20)])
    attacker_name = (names[attacker_id] or [f"Author {attacker_id}"])[0]

    # distinctive facts for the indirect condition, from the same corpus-measured index CSAR uses
    sa = os.path.join(_REPO_ROOT_FOR_ENV, "selector_audit")
    if sa not in sys.path:
        sys.path.insert(0, sa)
    import csar
    gold = {a: [full[a * 20 + w]["answer"] for w in range(20)] for a in range(200)}
    ix = csar.build_index(gold)

    def _facts(a):
        f = sorted(ix.distinctive(a, csar.DEFAULT_MAX_ADF), key=len, reverse=True)
        own = {n.lower() for n in names.get(a, [])}
        # a description that just restates the name is not an indirect reference
        return [x for x in f if not any(x in n or n in x for n in own)][:3]

    orig = [full[int(i)]["question"] for i in rows]
    cond = {"original": orig, "paraphrase": list(paras)}
    cond["name_stripped"] = [strip_names(q, names[a]) for q, a in zip(orig, authors)]
    cond["indirect"] = [indirect_reference(q, names[a], _facts(a))
                        for q, a in zip(orig, authors)]
    cond["name_injected"] = [inject_name(q, attacker_name) for q in orig]
    cond["name_swapped"] = [swap_name(q, names[a], attacker_name) for q, a in zip(orig, authors)]
    return cond, names, attacker_name


# ── scoring ──────────────────────────────────────────────────────────────────────

def score_matrices(full, k: int, questions: dict, encoder: str, device: str = "cpu") -> dict:
    """{strategy: {condition: matrix}} — one build of each router, reused across conditions."""
    import router as R
    out = {}

    key_index = R.build_key_index(full, k)
    lower = {sid: [n.lower() for n in nm] for sid, nm in key_index.items()}
    out["key_exact"] = {c: np.asarray(
        [[1 if any(n and n in q.lower() for n in lower[sid]) else 0 for sid in range(k)]
         for q in qs], dtype="uint8") for c, qs in questions.items()}

    tfidf = R.build_tfidf_router(full, k)
    vec, cents = tfidf._tfidf_vectorizer, np.stack(tfidf._tfidf_centroids)
    cn = np.linalg.norm(cents, axis=1) + 1e-12
    out["key_tfidf"] = {}
    for c, qs in questions.items():
        Q = vec.transform(qs).toarray()
        out["key_tfidf"][c] = ((Q @ cents.T) /
                               (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12) / cn
                               ).astype("float32")

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(encoder, device=device)
    # centroid_sbert semantics: mean of the unit's Q+A embeddings (router.build_centroids)
    qa = [full[i]["question"] + " " + full[i]["answer"] for i in range(len(full))]
    E = sbert.encode(qa, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    cents = np.stack([E[[a * 20 + w for a in get_author_shard(k, sid) for w in range(20)]].mean(0)
                      for sid in range(k)])
    out["centroid_sbert"] = {}
    for c, qs in questions.items():
        Q = sbert.encode(qs, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        out["centroid_sbert"][c] = (Q @ cents.T).astype("float32")
    return out


def analyze(mats: dict, authors: np.ndarray, is_forget: np.ndarray, k: int, drop_ids: list,
            attacker_id: int, seed: int, m_top: int) -> dict:
    """Per (strategy, condition): routing accuracy, attacker capture, detection AUC."""
    per_shard = 200 // k
    true_unit = authors // per_shard
    att_unit = attacker_id // per_shard
    surv = [j for j in range(k) if j not in set(drop_ids)]
    res = {}
    for strat, by_cond in mats.items():
        res[strat] = {}
        for cond, M in by_cond.items():
            M = np.asarray(M, dtype="float64")
            top1 = M.argmax(axis=1)
            cell = {
                "routing_accuracy": float((top1 == true_unit).mean()),
                "attacker_capture": float((top1[true_unit != att_unit] == att_unit).mean()),
            }
            if strat == "key_exact":
                # no graded score: report the operating point it actually serves
                cell["no_match_rate"] = float((M.sum(axis=1) == 0).mean())
                cell["detection"] = None
            else:
                pr = probe_arrays(M[:, surv], is_forget.astype(int), authors, k, strat,
                                  drop_ids, seed=seed, m_top=m_top)
                conf = [v["auc"] for n, v in pr.get("comparators", {}).items()
                        if not n.startswith("tomb_")]
                cell["detection"] = {
                    "best_confidence_auc": max(conf) if conf else None,
                    "probe_auc": pr.get("probe", {}).get("auc"),
                    "n_eval": pr.get("n_eval"),
                }
            res[strat][cond] = cell
    return res


def ood_geometry(mats: dict, groups: dict, k: int, drop_ids: list, n_boot_units: int = 3) -> dict:
    """Score geometry for queries belonging to NO source, against retained and orphan queries.

    The headline routed system gates OOD by `q2author` — an exact question-to-author lookup. That
    is an oracle: a deployment cannot know that "Where would you find the Eiffel Tower?" is not
    about one of its 200 sources. Without the gate every general-knowledge query gets some
    author's expert stapled on, and the repo already prices that at k=10 (mu 0.556 OOD-aware vs
    0.474 not).

    The question here is whether the selector's own scores could replace the oracle. If an OOD
    query spreads flat across all units — low top-1, small margin, high entropy — then a
    confidence threshold detects strangers even though it fails on orphans, and that asymmetry is
    the interesting result. If OOD queries instead land confidently on some unit, nothing in the
    selector distinguishes "I don't serve this" from "I serve this".
    """
    from analyze_orphan_destinations import concentration
    surv = [j for j in range(k) if j not in set(drop_ids)]
    out = {}
    for strat, by_group in mats.items():
        cells = {}
        for g, M in by_group.items():
            M = np.asarray(M, dtype="float64")[:, surv]
            if M.shape[0] == 0:
                continue
            if strat == "key_exact":
                cells[g] = {"n": int(M.shape[0]),
                            "no_match_rate": float((M.sum(axis=1) == 0).mean())}
                continue
            order = np.sort(M, axis=1)[:, ::-1]
            top1, top2 = order[:, 0], order[:, 1]
            # entropy of the softmaxed row: "how evenly does this query spread over units"
            z = M - M.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            ent = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1) / np.log(len(surv))
            dest = M.argmax(axis=1)
            hist = {int(u): int(c) for u, c in zip(*np.unique(dest, return_counts=True))}
            # concentration() sorts the counts and so loses WHICH unit is busiest — but
            # whether the stranger-magnet and the orphan-magnet are the same expert is the
            # question (H14), so the identity is recorded alongside the shares.
            top_units = sorted(hist.items(), key=lambda kv: -kv[1])[:5]
            cells[g] = {
                "n": int(M.shape[0]),
                "top1_mean": float(top1.mean()),
                "margin_mean": float((top1 - top2).mean()),
                "entropy_norm_mean": float(ent.mean()),
                "concentration": concentration(hist, len(surv)),
                "busiest_unit": (int(surv[top_units[0][0]]) if top_units else None),
                "top_units": [(int(surv[u]), int(c)) for u, c in top_units],
            }
        # can plain confidence tell a stranger from a served source? (negated top-1, the
        # analyze_router_leak direction: low confidence = more OOD-like)
        if strat != "key_exact" and "retain" in by_group:
            R = np.asarray(by_group["retain"], dtype="float64")[:, surv].max(axis=1)
            for g in by_group:
                if g == "retain" or strat == "key_exact":
                    continue
                G = np.asarray(by_group[g], dtype="float64")[:, surv].max(axis=1)
                cells[g]["auc_vs_retain"] = _auc(-G, -R)
        out[strat] = cells
    return out


def write_ood_md(res: dict, path: str) -> None:
    per, meta = res["cells"], res["meta"]
    L = ["# Queries that belong to no source", "",
         "The headline routed system decides TOFU-vs-OOD with `q2author`, an exact "
         "question-to-author lookup — an **oracle** a deployment does not have. Without it every "
         "general-knowledge query gets some source's expert applied. This asks whether the "
         "selector's own scores could replace that oracle.", "",
         f"k = {meta['k']} · deleted = {len(meta['drop_set'])} units", "",
         "`AUC vs retain` = can negated top-1 confidence separate this group from retained "
         "traffic? High = a threshold would work.", ""]
    for strat, cells in per.items():
        L += [f"## `{strat}`", "",
              "| group | n | top-1 | margin | entropy | busiest unit | n_eff | AUC vs retain |",
              "|---|---|---|---|---|---|---|---|"]
        for g in ("retain", "orphan", "ood_real_authors", "ood_world_facts"):
            c = cells.get(g)
            if not c:
                continue
            if "no_match_rate" in c:
                L.append(f"| `{g}` | {c['n']} | no-match {_f(c['no_match_rate'])} | | | | | |")
                continue
            con = c["concentration"]
            L.append(f"| `{g}` | {c['n']} | {_f(c['top1_mean'])} | {_f(c['margin_mean'])} | "
                     f"{_f(c['entropy_norm_mean'])} | {_f(con.get('max_share'))} | "
                     f"{_f(con.get('n_eff'), 1)} | {_f(c.get('auc_vs_retain'))} |")
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def write_md(res: dict, path: str) -> None:
    meta, per = res["meta"], res["cells"]
    L = ["# Routing and orphan detection under query shift", "",
         "The H3 granularity claim was measured on TOFU's gold-form questions, which name their "
         "author in ~90% of rows. This asks whether it survives queries that do not.", "",
         f"k = {meta['k']} · deleted = {len(meta['drop_set'])} units · "
         f"{meta['n_rows']} rows ({meta['n_forget']} forget / "
         f"{meta['n_rows'] - meta['n_forget']} retain) · attacker = author "
         f"{meta['attacker_id']} (`{meta['attacker_name']}`)", "",
         "`routing` = top-1 reaches the query's own unit (full pool). `capture` = share of "
         "queries about OTHER sources that land on the attacker. `conf` / `probe` = "
         "post-deletion orphan-detection AUC.", ""]
    for strat in per:
        L += [f"## `{strat}`", "",
              "| condition | routing | attacker capture | conf AUC | probe AUC |",
              "|---|---|---|---|---|"]
        for cond in CONDITIONS:
            c = per[strat].get(cond)
            if not c:
                continue
            d = c.get("detection") or {}
            L.append(f"| `{cond}` | {_f(c['routing_accuracy'])} | "
                     f"{_f(c['attacker_capture'])} | "
                     f"{_f(d.get('best_confidence_auc'))} | {_f(d.get('probe_auc'))} |")
        L.append("")
    L += ["## Read", ""]
    for strat in per:
        o = per[strat].get("original", {})
        det0 = (o.get("detection") or {}).get("best_confidence_auc")
        for cond in ("name_stripped", "indirect"):
            c = per[strat].get(cond)
            if not c:
                continue
            det = (c.get("detection") or {}).get("best_confidence_auc")
            if det0 is not None and det is not None:
                L.append(f"- `{strat}` / `{cond}`: routing "
                         f"{_f(o['routing_accuracy'])} → {_f(c['routing_accuracy'])}, "
                         f"detection {_f(det0)} → {_f(det)} (Δ {_f(det - det0)}).")
        cap = per[strat].get("name_injected", {}).get("attacker_capture")
        base = o.get("attacker_capture")
        if cap is not None and base is not None:
            L.append(f"- `{strat}` / adversarial injection: attacker capture "
                     f"{_f(base)} → **{_f(cap)}**.")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


# ── self test ────────────────────────────────────────────────────────────────────

def run_self_test() -> None:
    n = 0

    def ok(name):
        nonlocal n
        n += 1
        print(f"  PASS {name}")

    names = ["Kalkidan Abera", "Abera"]
    q = "What themes does Kalkidan Abera explore, and where was Abera born?"
    s = strip_names(q, names)
    assert "abera" not in s.lower() and "kalkidan" not in s.lower(), s
    assert "themes" in s and "born" in s, s
    ok(f"name stripping removes every variant, keeps the question ({s[:44]!r})")

    inj = inject_name(q, "Hsiao Yun-Hwa")
    assert "Kalkidan Abera" in inj and "Hsiao Yun-Hwa" in inj
    ok("injection keeps the true subject AND adds the attacker")

    sw = swap_name(q, names, "Hsiao Yun-Hwa")
    assert "Abera" not in sw and sw.count("Hsiao Yun-Hwa") == 2, sw
    ok("swap replaces every variant of the true name")

    ind = indirect_reference(q, names, ["addis ababa", "nutrition"])
    assert "Abera" not in ind and "addis ababa" in ind and "nutrition" in ind
    assert indirect_reference(q, names, []) == strip_names(q, names)
    ok("indirect reference: no name, identifying description, degrades to stripped")

    # analyze(): a planted matrix where every query is routed to the attacker must read
    # capture ~1.0 and routing ~0, and the reverse for a perfect router
    k, n_q = 4, 40
    authors = np.repeat(np.arange(4) * 50, 10)          # units 0..3 at k=4 (per_shard 50)
    is_forget = authors >= 150
    perfect = np.zeros((n_q, k)); perfect[np.arange(n_q), authors // 50] = 1.0
    hijack = np.zeros((n_q, k)); hijack[:, 0] = 1.0
    r = analyze({"key_tfidf": {"original": perfect, "name_injected": hijack}},
                authors, is_forget, k, [3], attacker_id=0, seed=42, m_top=4)
    a = r["key_tfidf"]
    assert a["original"]["routing_accuracy"] == 1.0
    assert a["original"]["attacker_capture"] == 0.0
    assert a["name_injected"]["attacker_capture"] == 1.0
    assert a["name_injected"]["routing_accuracy"] < 0.3
    ok("analyze(): routing accuracy and attacker capture are computed as claimed")

    # key_exact reports its no-match operating point instead of a graded AUC
    match = np.zeros((n_q, k), dtype="uint8")
    r2 = analyze({"key_exact": {"original": match}}, authors, is_forget, k, [3],
                 attacker_id=0, seed=42, m_top=4)
    assert r2["key_exact"]["original"]["detection"] is None
    assert r2["key_exact"]["original"]["no_match_rate"] == 1.0
    ok("key_exact: no graded detection, no-match rate reported instead")

    print(f"[analyze_router_shift] self_test: {n}/6 PASS")


def run_ood(args):
    """Queries belonging to no source, beside retained and orphan traffic."""
    from datasets import load_dataset
    os.environ.setdefault("HF_HOME", args.hf_home)
    drop_ids = parse_drop_set(args.drop_set)
    full, rows, authors, _ = build_eval_rows(args.hf_home)
    is_forget = np.isin(authors, np.arange(180, 200))
    qs = [full[int(i)]["question"] for i in rows]
    groups = {
        "retain": [q for q, f in zip(qs, is_forget) if not f],
        "orphan": [q for q, f in zip(qs, is_forget) if f],
        "ood_real_authors": [r["question"] for r in
                             load_dataset("locuslab/TOFU", "real_authors_perturbed")["train"]],
        "ood_world_facts": [r["question"] for r in
                            load_dataset("locuslab/TOFU", "world_facts_perturbed")["train"]],
    }
    for g, v in groups.items():
        print(f"[ood] {g:18s} n={len(v)}  e.g. {v[0][:70]}", flush=True)
    mats = score_matrices(full, args.k, groups, args.encoder, args.device)
    res = {"meta": {"k": args.k, "drop_set": drop_ids, "encoder": args.encoder},
           "cells": ood_geometry(mats, groups, args.k, drop_ids)}
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[ood] -> {args.out_json}")
    if args.out_md:
        write_ood_md(res, args.out_md)
        print(f"[ood] -> {args.out_md}")
    for strat, cells in res["cells"].items():
        print(f"  {strat}")
        for g, c in cells.items():
            if "no_match_rate" in c:
                print(f"    {g:18s} n={c['n']:4d} no_match={c['no_match_rate']:.3f}")
            else:
                print(f"    {g:18s} n={c['n']:4d} top1={c['top1_mean']:.3f} "
                      f"margin={c['margin_mean']:.3f} ent={c['entropy_norm_mean']:.3f} "
                      f"busiest={c['concentration'].get('max_share', float('nan')):.3f} "
                      f"AUCvsRetain={_f(c.get('auc_vs_retain'))}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--drop_set", default="180-199")
    ap.add_argument("--attacker_id", type=int, default=0,
                    help="author whose name is injected/swapped; capture is measured against it")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m_top", type=int, default=20)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--ood", action="store_true",
                    help="Score queries that belong to NO source (TOFU real_authors + "
                         "world_facts) against retained and orphan traffic, and ask whether the "
                         "selector's own confidence could replace the q2author OOD oracle.")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return
    if args.ood:
        return run_ood(args)
    if not args.hf_home:
        raise SystemExit("--hf_home or $HF_HOME is required")
    drop_ids = parse_drop_set(args.drop_set)

    full, rows, authors, paras = build_eval_rows(args.hf_home)
    is_forget = np.isin(authors, np.arange(180, 200))
    print(f"[shift] {len(rows)} rows ({int(is_forget.sum())} forget / "
          f"{int((~is_forget).sum())} retain)", flush=True)
    cond, names, attacker_name = build_conditions(full, rows, authors, paras,
                                                  args.attacker_id, args.hf_home)
    print(f"[shift] attacker = author {args.attacker_id} ({attacker_name!r})", flush=True)
    for c in CONDITIONS:
        print(f"    {c:14s} e.g. {cond[c][0][:88]}", flush=True)

    mats = score_matrices(full, args.k, cond, args.encoder, args.device)
    cells = analyze(mats, authors, is_forget, args.k, drop_ids, args.attacker_id,
                    args.seed, args.m_top)
    res = {"meta": {"k": args.k, "drop_set": drop_ids, "n_rows": int(len(rows)),
                    "n_forget": int(is_forget.sum()), "attacker_id": args.attacker_id,
                    "attacker_name": attacker_name, "encoder": args.encoder,
                    "seed": args.seed},
           "cells": cells}
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[shift] -> {args.out_json}")
    if args.out_md:
        write_md(res, args.out_md)
        print(f"[shift] -> {args.out_md}")
    for strat in cells:
        print(f"  {strat}")
        for c in CONDITIONS:
            cell = cells[strat].get(c)
            if not cell:
                continue
            d = cell.get("detection") or {}
            print(f"    {c:14s} routing={cell['routing_accuracy']:.3f} "
                  f"capture={cell['attacker_capture']:.3f} "
                  f"conf={_f(d.get('best_confidence_auc'))} probe={_f(d.get('probe_auc'))}")


if __name__ == "__main__":
    main()
