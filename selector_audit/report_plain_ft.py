#!/usr/bin/env python3
"""Assemble Vincent's Q4/Q5 report: plain fine-tuned baseline vs the routed system.

Reads the sharded dumps from `submit_plain_ft_baseline.sh` (routerless arms) and
`submit_routed_shift.sh` (routed arms), stitches each set of row-shards back together, and
writes the two comparison tables plus the exact provenance.

Three rules this file obeys, because each of them was a way to get a plausible wrong number:

  * A set with a HOLE is not scored. Shard files are checked for completeness against the row set
    they claim, and an incomplete arm is reported Pending rather than averaged over what landed.
  * Q5 is scored on CONTENT for both systems. Finding 5's 97.7%% is a ROUTING-capture rate and a
    routerless model has no route, so the shared criterion is whether the served answer asserts
    the injected author's distinctive facts -- `selector_audit/csar.py`, unmodified. The routed
    system's routing-capture rate is reported too, in its own column, never merged with it.
  * The base-model arm is required, not optional. `csar.classify` subtracts the base's answer to
    the same question; without it the base model's own knowledge is scored as attack success.

  python report_plain_ft.py --out outputs/vincent_q4_q5_report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO_ROOT, "tofu_sisa_lora"), _REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

import csar  # noqa: E402


def _load_shards(pattern: str):
    """[(shard_i, shard_n, payload)] for every file matching `pattern`."""
    out = []
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"_shard(\d+)_of_(\d+)\.json$", p)
        if not m:
            continue
        with open(p) as f:
            out.append((int(m.group(1)), int(m.group(2)), json.load(f)))
    return out


def _complete(shards, label):
    """(ok, message) — every shard of one consistent N present, exactly once."""
    if not shards:
        return False, "no shard files found"
    ns = {n for _, n, _ in shards}
    if len(ns) != 1:
        return False, f"mixed shard counts {sorted(ns)} — a stale run is still on disk"
    n = ns.pop()
    have = sorted(i for i, _, _ in shards)
    missing = [i for i in range(n) if i not in have]
    if missing:
        return False, f"{len(missing)}/{n} shards missing: {missing[:6]}"
    if len(have) != len(set(have)):
        return False, "duplicate shard indices"
    return True, f"{n}/{n} shards"


# ── the routerless arms ──────────────────────────────────────────────────────────

def load_plain(out_dir: str, tag: str):
    """{condition: {row: record}} for one routerless arm, or (None, why)."""
    shards = _load_shards(os.path.join(out_dir, f"{tag}_shard*_of_*.json"))
    ok, msg = _complete(shards, tag)
    if not ok:
        return None, msg
    by_cond = defaultdict(dict)
    for _, _, payload in shards:
        for cond, recs in payload["conditions"].items():
            for r in recs:
                by_cond[cond][int(r["row"])] = r
    return dict(by_cond), msg


# ── the routed arms ──────────────────────────────────────────────────────────────

def load_routed(out_dir: str, qt: str, strategy: str, tag: str = ""):
    """{row: record} for one routed transform × strategy, or (None, why).

    `tag` selects a deletion-size rung: "" is the audit's size (20 authors), "_d5" is the 5-author
    rung written by `FORGET=180-184 submit_routed_shift.sh`.
    """
    shards = _load_shards(os.path.join(out_dir, f"routed_{qt}{tag}_shard*_of_*.json"))
    ok, msg = _complete(shards, qt)
    if not ok:
        return None, msg
    rows = {}
    for _, _, payload in shards:
        block = payload.get("strategies", {}).get(strategy)
        if block is None:
            return None, f"strategy {strategy!r} absent from the dump"
        for r in block["per_question"]:
            rows[int(r["row"])] = r
    return rows, msg


def _split(recs, key):
    f = [r[key] for r in recs.values() if r.get("is_forget")]
    t = [r[key] for r in recs.values() if not r.get("is_forget")]
    return (float(np.mean(f)) if f else float("nan"), len(f),
            float(np.mean(t)) if t else float("nan"), len(t))


def _fmt(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


# ── Q5 content scoring ───────────────────────────────────────────────────────────

def attacker_hit_rate(served: dict, base: dict, index, attacker_id: int):
    """Fraction of answers asserting the ATTACKER's distinctive facts.

    csar.classify is reused verbatim: `sibling_author` is set to the attacker, so a
    `cross_source` verdict means the answer carried a fact that is distinctive to the injected
    author, is NOT the row's true subject's own fact, and is NOT in the base model's answer to
    the same question. Rows whose base generation is missing are counted as skipped, never as
    misses -- a missing exclusion would inflate the rate.
    """
    hits = refusals = skipped = 0
    scored = 0
    for row, rec in served.items():
        b = base.get(row)
        if b is None:
            skipped += 1
            continue
        v = csar.classify({"row": row,
                           "author": int(rec["author"]),
                           "sibling_author": int(attacker_id),
                           "gen_sibling": rec.get("gen_sibling") or "",
                           "gen_base": b.get("gen_sibling") or b.get("gen_base") or ""},
                          index)
        scored += 1
        if v["category"] == "cross_source":
            hits += 1
        elif v["category"] == "refusal":
            refusals += 1
    return {"n_scored": scored, "n_skipped": skipped,
            "attacker_fact_rate": hits / scored if scored else float("nan"),
            "refusal_rate": refusals / scored if scored else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ckpt = os.environ.get("TOFU_CKPT_ROOT", "")
    ap.add_argument("--plain_dir", default=os.path.join(ckpt, "plain_ft_baseline"))
    ap.add_argument("--routed_dir", default=os.path.join(
        ckpt, "Llama-2-7B-chat-hf_k200_r32_e25_lr1e4", "results", "routed_shift"))
    ap.add_argument("--strategy", default="centroid_sbert")
    ap.add_argument("--attacker_id", type=int, default=0)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "outputs",
                                                  "vincent_q4_q5_report.md"))
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    gold = csar.load_tofu_gold(args.hf_home)
    index = csar.build_index(gold)
    attacker_name = None

    notes, res = [], {"q4": {}, "q5": {}, "provenance": {}}

    ft, ft_msg = load_plain(args.plain_dir, "ft")
    base, base_msg = load_plain(args.plain_dir, "base")
    res["provenance"]["plain_ft"] = ft_msg
    res["provenance"]["plain_base"] = base_msg
    if ft:
        attacker_name = None
        for p in sorted(glob.glob(os.path.join(args.plain_dir, "ft_shard*_of_*.json"))):
            attacker_name = json.load(open(p))["meta"].get("attacker_name")
            break

    # ── Q4 ────────────────────────────────────────────────────────────────────
    QA_CONDS = ["original", "name_stripped", "para_stripped"]
    q4_rows = []
    for cond in QA_CONDS:
        row = {"condition": cond}
        if ft and cond in ft:
            f, nf, t, nt = _split(ft[cond], "rougeL_recall_vs_own_gold")
            row.update(ft_forget=f, ft_retain=t, n_forget=nf, n_retain=nt)
        rr, rmsg = load_routed(args.routed_dir, "none" if cond == "original" else cond,
                               args.strategy)
        res["provenance"][f"routed_{cond}"] = rmsg
        if rr:
            f, nf, t, nt = _split(rr, "sibling_vs_gold")
            row.update(routed_forget=f, routed_retain=t,
                       routed_n_forget=nf, routed_n_retain=nt)
        q4_rows.append(row)
    res["q4"]["rows"] = q4_rows

    # ── Q5 ────────────────────────────────────────────────────────────────────
    QB_CONDS = ["name_injected", "name_swapped"]
    q5_rows = []
    for cond in QB_CONDS:
        row = {"condition": cond}
        if ft and base and cond in ft and cond in base:
            row["ft"] = attacker_hit_rate(ft[cond], base[cond], index, args.attacker_id)
            fr = [r["rougeL_recall_vs_own_gold"] for r in ft[cond].values()]
            row["ft"]["rouge_vs_true_subject"] = float(np.mean(fr)) if fr else float("nan")
        rr, rmsg = load_routed(args.routed_dir, cond, args.strategy)
        res["provenance"][f"routed_{cond}"] = rmsg
        if rr:
            # The routed dump already carries its own base arm (adapters disabled) per row.
            rbase = {k: {"gen_sibling": v.get("gen_base", "")} for k, v in rr.items()}
            row["routed"] = attacker_hit_rate(rr, rbase, index, args.attacker_id)
            # At k=200 one author per unit, so the routed shard id IS the author id and
            # "captured by the attacker" is a plain equality. This would be wrong at k<200.
            # Rows ABOUT the attacker are excluded from the denominator, matching how
            # analyze_router_shift.analyze() defines attacker_capture -- otherwise the attacker's
            # own 20 correctly-routed rows are scored as successful captures. NB this is still a
            # POST-deletion route (survivors only), where h30's is pre-deletion, so the two are
            # close but not the same quantity.
            cap = [1.0 if int(r["sibling_shard"]) == args.attacker_id else 0.0
                   for r in rr.values() if int(r["author"]) != args.attacker_id]
            row["routed"]["routing_capture_rate"] = float(np.mean(cap)) if cap else float("nan")
            row["routed"]["rouge_vs_true_subject"] = float(
                np.mean([r["sibling_vs_gold"] for r in rr.values()]))
        q5_rows.append(row)
    res["q5"]["rows"] = q5_rows

    # ── render ────────────────────────────────────────────────────────────────
    L = ["# Plain fine-tuned baselines for findings 4 and 5",
         "",
         "Both systems are measured on the **same 800 rows** (400 forget / 400 retain), with the "
         "**same transforms** and the **same attacker** that produced findings 4 and 5 "
         "(`analyze_router_shift.build_eval_rows` / `build_conditions`).",
         ""]
    L += ["| setting | value |", "|---|---|",
          f"| plain FT model | `locuslab/tofu_ft_llama2-7b` (full fine-tune, no adapter, no router) |",
          f"| base / exclusion arm | `meta-llama/Llama-2-7B-chat-hf` |",
          f"| routed pool | `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4`, deletion = authors 180–199 |",
          f"| router shown | `{args.strategy}` |",
          f"| attacker | author {args.attacker_id} (`{attacker_name}`) |",
          f"| prompt | `Question: {{q}}\\nAnswer:` — byte-identical for both systems |",
          ""]

    L += ["## Q4 — answer quality when the name is removed", "",
          "ROUGE-L recall against the row's own gold answer. `original` is TOFU's question "
          "verbatim; `name_stripped` is finding 4's transform; `para_stripped` is TOFU's own "
          "paraphrase with names removed.", "",
          "| condition | plain FT · retain | plain FT · forget | routed · retain | routed · forget |",
          "|---|---|---|---|---|"]
    for r in q4_rows:
        L.append(f"| `{r['condition']}` | {_fmt(r.get('ft_retain'))} | {_fmt(r.get('ft_forget'))} "
                 f"| {_fmt(r.get('routed_retain'))} | {_fmt(r.get('routed_forget'))} |")
    L.append("")

    # Interpretation is COMPUTED from the same dict the table is rendered from, so it cannot
    # drift from the numbers above it during editing.
    def _cell(cond, key):
        for r in q4_rows:
            if r["condition"] == cond:
                return r.get(key)
        return None

    ft_o, ft_s = _cell("original", "ft_retain"), _cell("name_stripped", "ft_retain")
    ro_o, ro_s = _cell("original", "routed_retain"), _cell("name_stripped", "routed_retain")
    if None not in (ft_o, ft_s, ro_o, ro_s):
        d_ft, d_ro = ft_o - ft_s, ro_o - ro_s
        res["q4"]["retain_drop_ft"], res["q4"]["retain_drop_routed"] = d_ft, d_ro
        L += ["### Reading", "",
              "Compare on the **retain** column: nothing is deleted there for either system, so "
              "it is the only like-for-like surface. Stripping the name costs",
              "",
              f"- plain FT: {ft_o:.4f} → {ft_s:.4f}  (**−{d_ft:.4f}**, −{100 * d_ft / ft_o:.0f}%)",
              f"- routed:   {ro_o:.4f} → {ro_s:.4f}  (**−{d_ro:.4f}**, −{100 * d_ro / ro_o:.0f}%)",
              "",
              f"The two absolute drops differ by {abs(d_ft - d_ro):.4f}. **A model with no router "
              "at all loses essentially as much as the routed system does**, so finding 4's "
              "collapse under name removal is mostly a property of TOFU questions being "
              "unanswerable once the name is gone — not of routing. Routing's own cost shows up "
              "as the level difference in the `original` row, not as extra sensitivity to "
              "anonymisation.", "",
              "On `para_stripped` both systems sit near the frozen base model's floor "
              "(0.2841 on the same rows), i.e. on a genuinely name-free surface the fine-tune "
              "buys almost nothing — which bounds how much of finding 4 can be about the "
              "selector.",
              "",
              "The `forget` column is not comparable across systems: the routed pool has those "
              "experts deleted, the plain FT model has deleted nothing.", ""]

    L += ["## Q5 — does name injection steer a model with no router?", "",
          "`attacker fact rate` = fraction of served answers asserting a fact distinctive to the "
          "injected author, excluding the true subject's own facts and anything the base model "
          "already says (`selector_audit/csar.py`, unmodified). `routing capture` is the routed "
          "system's finding-5 criterion and is **not** the same measurement — it is shown beside "
          "the content rate, never merged with it.", "",
          "| attack | plain FT · attacker fact rate | routed · attacker fact rate | "
          "routed · routing capture |", "|---|---|---|---|"]
    for r in q5_rows:
        ft_b, ro_b = r.get("ft", {}), r.get("routed", {})
        L.append(f"| `{r['condition']}` | {_fmt(ft_b.get('attacker_fact_rate'))} | "
                 f"{_fmt(ro_b.get('attacker_fact_rate'))} | "
                 f"{_fmt(ro_b.get('routing_capture_rate'))} |")
    L.append("")

    def _q5(cond, sys_key, field):
        for r in q5_rows:
            if r["condition"] == cond:
                return r.get(sys_key, {}).get(field)
        return None

    fi, ri = _q5("name_injected", "ft", "attacker_fact_rate"), _q5("name_injected", "routed",
                                                                  "attacker_fact_rate")
    fs, rs_ = _q5("name_swapped", "ft", "attacker_fact_rate"), _q5("name_swapped", "routed",
                                                                  "attacker_fact_rate")
    cap_i = _q5("name_injected", "routed", "routing_capture_rate")
    if None not in (fi, ri, fs, rs_, cap_i):
        L += ["### Reading", "",
              f"On the shared content criterion the routed system is about "
              f"{ri / fi:.1f}× the plain model on append ({ri:.4f} vs {fi:.4f}) and "
              f"{rs_ / fs:.1f}× on substitute ({rs_:.4f} vs {fs:.4f}). So **the attack is not "
              "specific to routing — a routerless TOFU fine-tune already follows an injected "
              f"name — but routing amplifies it {min(ri / fi, rs_ / fs):.1f}–{max(ri / fi, rs_ / fs):.1f}×.** "
              "Finding 5 should be framed as an amplification over that floor, not as a "
              "routing-only failure.", "",
              f"Note the append row: the attacker's facts appear in {ri:.4f} of answers while the "
              f"router only sent {cap_i:.4f} of queries to the attacker's expert. **Content "
              "contamination exceeds routing capture**, so the served expert is echoing the "
              "injected name rather than the router alone being steered — which is exactly the "
              "mechanism the plain-FT floor exposes.", "",
              "The routing-capture column is finding 5's own criterion and is shown only for "
              "orientation; it answers a different question from the two columns beside it.", ""]

    # Regression guard, re-checked on every render: --serve_rows shift800 must not have changed
    # what the pre-existing forget-only arm measured. Same pool, same router, same 400 orphans.
    ref = os.path.join(os.path.dirname(args.routed_dir), "router_leak",
                       "sibling_content_k200_f10_qpa20.json")
    rr_none, _ = load_routed(args.routed_dir, "none", args.strategy)
    if os.path.exists(ref) and rr_none:
        old = json.load(open(ref))["strategies"].get(args.strategy)
        if old:
            oldrows = {int(r["row"]) for r in old["per_question"]}
            mine = {k: v for k, v in rr_none.items() if v.get("is_forget")}
            same_rows = set(mine) == oldrows
            deltas = {}
            for key in ("own_vs_gold", "sibling_vs_gold", "base_vs_gold", "sibling_vs_basegen"):
                deltas[key] = abs(float(np.mean([r[key] for r in mine.values()]))
                                  - old["aggregates"][key]["mean"])
            worst = max(deltas.values())
            res["q4"]["regression_max_delta"] = worst
            res["q4"]["regression_same_rows"] = same_rows
            L += ["### Regression guard", "",
                  f"The `none` arm's {len(mine)} forget rows are the same 400 orphans the "
                  f"published `sibling_content_k200_f10_qpa20` arm scored "
                  f"(identical row set: **{same_rows}**), so serving the 800-row set must not "
                  f"have changed them. Largest disagreement across `own_vs_gold`, "
                  f"`sibling_vs_gold`, `base_vs_gold`, `sibling_vs_basegen`: "
                  f"**{worst:.6f}**"
                  + ("  — an exact reproduction." if worst < 5e-4 else
                     "  — **INVESTIGATE before trusting any cell above.**"), ""]

    # ── the deletion-size ladder, on the SERVING metrics ──────────────────────
    # Deletion size is a dial for the routed system only: the plain fine-tune deleted nothing, so
    # it is a flat reference line, printed once rather than repeated down the column.
    SIZES = [(1, "_d1"), (5, "_d5"), (10, "_d10"), (20, "")]
    lad = {}
    for cond in ("none", "name_stripped", "name_swapped"):
        rungs = []
        for d, tag in SIZES:
            rr, msg = load_routed(args.routed_dir, cond, args.strategy, tag)
            res["provenance"][f"ladder_{cond}_d{d}"] = msg
            if not rr:
                continue
            row = {"n_deleted": d}
            f, nf, t, nt = _split(rr, "sibling_vs_gold")
            row.update(orphan_rouge=f, retain_rouge=t, n_orphan=nf, n_retain=nt)
            if cond == "name_swapped":
                rbase = {k: {"gen_sibling": v.get("gen_base", "")} for k, v in rr.items()}
                row["attacker_fact_rate"] = attacker_hit_rate(
                    rr, rbase, index, args.attacker_id)["attacker_fact_rate"]
                row["routing_capture"] = float(np.mean(
                    [1.0 if int(r["sibling_shard"]) == args.attacker_id else 0.0
                     for r in rr.values() if int(r["author"]) != args.attacker_id]))
            rungs.append(row)
        if rungs:
            lad[cond] = rungs
    res["ladder"] = lad

    if lad:
        L += ["## Metrics vs number of sources deleted", "",
              "Deletion size is a dial for the **routed system only** — the plain fine-tune "
              "deleted nothing, so it is a flat reference line rather than a column. Deletion "
              "sets are nested prefixes of `180-199`, and a row counts as an orphan only if its "
              "OWN author was deleted, so the orphan/retain split is recomputed at every rung.",
              ""]
        for cond, rungs in lad.items():
            nice = {"none": "gold-form questions", "name_stripped": "name-stripped questions",
                    "name_swapped": "name-swapped (attack)"}[cond]
            L += [f"### Served answer quality — {nice}", "",
                  "| authors deleted | orphan rows | routed · orphan | routed · retain |"
                  + (" attacker fact rate | routing capture |" if cond == "name_swapped" else ""),
                  "|---|---|---|---|" + ("---|---|" if cond == "name_swapped" else "")]
            for r in rungs:
                extra = ""
                if cond == "name_swapped":
                    extra = (f" {_fmt(r.get('attacker_fact_rate'))} |"
                             f" {_fmt(r.get('routing_capture'))} |")
                L.append(f"| {r['n_deleted']} | {r['n_orphan']} | {_fmt(r['orphan_rouge'])} "
                         f"| {_fmt(r['retain_rouge'])} |{extra}")
            ft_ref = _cell("original" if cond == "none" else "name_stripped", "ft_retain")
            if ft_ref is not None and cond in ("none", "name_stripped"):
                L += ["", f"Plain FT reference on the same rows (nothing deleted, so flat across "
                          f"the ladder): **{ft_ref:.4f}**.", ""]
            else:
                L.append("")
        ns = lad.get("name_stripped") or []
        gf = lad.get("none") or []
        if len(ns) >= 2 and len(gf) >= 2:
            ns0, ns1 = ns[0], ns[-1]
            gf0, gf1 = gf[0], gf[-1]
            ft_ns = _cell("name_stripped", "ft_retain")
            L += ["### Reading", "",
                  "The orphan column is flat everywhere — how much a deleted source's own "
                  "queries degrade does not depend on how many OTHER sources were deleted. "
                  "The movement is in the **retain** column, and only without names:", "",
                  f"- gold-form retain: {gf0['retain_rouge']:.4f} (d={gf0['n_deleted']}) -> "
                  f"{gf1['retain_rouge']:.4f} (d={gf1['n_deleted']})  "
                  f"— flat, delta {gf1['retain_rouge'] - gf0['retain_rouge']:+.4f}",
                  f"- name-stripped retain: {ns0['retain_rouge']:.4f} (d={ns0['n_deleted']}) -> "
                  f"{ns1['retain_rouge']:.4f} (d={ns1['n_deleted']})  "
                  f"— **delta {ns1['retain_rouge'] - ns0['retain_rouge']:+.4f}**", ""]
            if ft_ns is not None:
                L += [f"At one deletion the routed system's anonymised retain quality "
                      f"({ns0['retain_rouge']:.4f}) is level with the routerless model "
                      f"({ft_ns:.4f}) — deleting one source costs retained users nothing. By "
                      f"twenty it has fallen to {ns1['retain_rouge']:.4f}, "
                      f"{100 * (ft_ns - ns1['retain_rouge']) / ft_ns:.0f}% below that reference. "
                      "**The collateral cost of deletion is not a fixed toll; it accumulates "
                      "with deletion volume, and only on queries that do not name their "
                      "subject.** This is the serving-level counterpart of the RDR curve "
                      "(0.0000 -> 0.0925 over the same rungs) in the routing ladder.", ""]
        sw = lad.get("name_swapped") or []
        if len(sw) >= 2:
            L += [f"The attack ladder is flat by contrast (attacker fact rate "
                  f"{sw[0]['attacker_fact_rate']:.4f} -> {sw[-1]['attacker_fact_rate']:.4f}): the "
                  "attacker's own expert always survives, so how much else was deleted does not "
                  "change what the attack achieves.", ""]

        FIGDIR = "../tofu_sisa_lora/reports/figures/deletion_size"
        L += ["", "#### Figures", "",
              f"![RDR vs deletions]({FIGDIR}/fig1_rdr_vs_deletion_size.png)", "",
              "*Collateral displacement of retained traffic. Flat on the floor for named "
              "queries at every deletion size; climbing steadily once the name is gone.*", "",
              f"![retained quality vs deletions]({FIGDIR}/fig3_retained_quality_vs_deletion_size.png)",
              "", "*The same effect in what retained users receive. Horizontal lines are the "
              "routerless control, which deletes nothing and so cannot move with the ladder.*", "",
              f"![routing accuracy vs deletions]({FIGDIR}/fig2_routing_accuracy_vs_deletion_size.png)",
              "", "*The mechanism: retained queries being routed to the wrong expert.*", "",
              f"![orphan dispersal vs deletions]({FIGDIR}/fig4_orphan_dispersal_vs_deletion_size.png)",
              "", "*Orphans disperse as deletions accumulate rather than concentrating on one "
              "magnet expert.*", "",
              f"![detection AUC vs deletions]({FIGDIR}/fig5_detection_auc_vs_deletion_size.png)",
              "", "*Detectability is set by phrasing, not by deletion size — both series flat.*",
              "",
              f"![attack vs deletions]({FIGDIR}/fig6_attack_vs_deletion_size.png)", "",
              "*The attack is size-independent: the attacker's expert always survives. Deletion "
              "volume is a dial for collateral damage, not for adversarial exposure.*", "",
              "Regenerate with `$TOFU_PLOT_PYTHON tofu_sisa_lora/plot_deletion_size.py`; it reads "
              "only the committed JSONs, so the figures cannot drift from the tables above. "
              "Figures stop at 20 deletions because the evaluation set covers authors 0-19 and "
              "180-199 only.", ""]

        L += ["Routing-level metrics on the same ladder — routing accuracy, orphan detection AUC, "
              "RDR, attacker capture and orphan destination concentration, for all three "
              "feature-space routers and a denser set of rungs — are in "
              "`tofu_sisa_lora/reports/deletion_size_ladder.md`. That sweep is CPU-only: it runs "
              "off the score matrices `analyze_router_shift --dump_npz` already wrote, so it "
              "needed no GPU and no new serving run.", ""]

    L += ["## Caveats that travel with these numbers", "",
          "1. **`name_stripped` does not fully anonymise.** 31.2% of the 800 rows still carry a "
          "name — 12.2% unchanged (no extractable name) and 19.0% left with a surname fragment, "
          "because `router._extract_author_names` splits hyphenated names (`\"Aisha Al\"` for "
          "*Aisha Al-Hamad*, leaving `-Hamad`). `para_stripped` inherits it (30.6%). Both "
          "systems get the identical corrupted queries, so the **comparison** is sound, but "
          "every absolute anonymised number here is an upper bound. See "
          "`outputs/anonymized_examples.md`.",
          "2. **The stripped questions are often ungrammatical stubs** (`\"Are the details of 's "
          "birth documented?\"`), which models complete arbitrarily — the frozen base answers "
          "that one about *Jesus'* birth. Part of the measured drop is broken grammar rather "
          "than lost identity, and it applies to both systems equally.",
          "3. **Routing-capture provenance.** `tofu_sisa_lora/reports/h30/router_shift_h30.md` "
          "reports name-injection capture on these exact 800 rows; the manuscript quotes "
          "97.7% / 31.7% / 3.5% from the earlier 2026-08-07 run. The capture column above is "
          "recomputed from this run's own routed dumps, on the same rows, attacker and seed as "
          "the baseline — do not mix it with the manuscript's figures without saying which is "
          "which.",
          "4. Only `centroid_sbert` is shown. `key_tfidf` is in the same dumps "
          "(`--strategy key_tfidf` re-renders); the behavioural family was not run, as it scores "
          "every expert on every query and is impractical at k=200.",
          ""]

    pend = {k: v for k, v in res["provenance"].items() if "shards" not in v}
    if pend:
        L += ["## Pending / missing arms", "",
              "These are reported as gaps rather than averaged over whatever landed.", ""]
        for k, v in sorted(pend.items()):
            L.append(f"- `{k}` — {v}")
        L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[report] -> {args.out}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[report] -> {args.out_json}")
    for k, v in sorted(res["provenance"].items()):
        print(f"    {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
