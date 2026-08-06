"""Aggregate the follow-up campaign (E1–E6, submitted by submit_followup.sh) into one report.

WHY a separate collector: the follow-ups span TWO arms (DBpedia ramole runs under
{root}/runs/ and the TOFU pool's results/extended) and several producer scripts
(eval_ramole, eval_tofu, analyze_router{,_tofu}, routing_audit{,_tofu}, benchmark_serving)
that land at different times. This report must always generate from whatever subset exists —
every absent file/key renders as a dash, never a crash — so it can be run mid-campaign.

    python collect_followup.py [--root ${TOFU_CKPT_STORE}/ramole]
        [--tofu_pool .../Llama-3.2-1B-Instruct_legonet_n32_k3] [--out FOLLOWUP_REPORT.md]
"""
import argparse
import json
import math
import os

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

ARM_A = "ramole_l32_3b_n32_k3"
# (run_name, seed) — arm A is the seed-42 member of the E1 seed family
SEED_RUNS = [(ARM_A, "42"), ("ramole_l32_3b_s43", "43"), ("ramole_l32_3b_s44", "44")]
TOFU_SEED_SFX = [("", "42"), ("_s43", "43"), ("_s44", "44")]
DBP_COLS = ["em", "verbmem", "perplexity", "canary_em"]
AUDIT_COLS = ["orig_topk_rate", "orig_top1_rate", "affected_mass", "untouched_mass",
              "mean_jaccard"]


# ── tolerant IO / formatting ───────────────────────────────────────────────────

def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _fmt(v, nd=3):
    if _num(v):
        return str(v) if isinstance(v, int) else f"{v:.{nd}f}"
    return "—"


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _mean_std(xs):
    xs = [x for x in xs if _num(x)]
    if not xs:
        return None, None, 0
    mu = sum(xs) / len(xs)
    if len(xs) < 2:
        return mu, None, len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5   # sample std over seeds
    return mu, sd, len(xs)


def _verdict(gaps, what):
    mu, sd, n = _mean_std(gaps)
    if n == 0:
        return f"**Verdict ({what}):** no runs on disk yet."
    if n < 2:
        return (f"**Verdict ({what}):** gap {_fmt(mu)} from a single seed — variance unknown, "
                "waiting on the other seeds.")
    call = "ROBUST to router seed (|mean| > std)" if abs(mu) > sd else \
           "WITHIN seed noise (|mean| <= std)"
    return f"**Verdict ({what}):** gap mean±std = {_fmt(mu)} ± {_fmt(sd)} over n={n} seeds → {call}."


def _dbp_agg(root, run, label):
    return _get(_load(os.path.join(root, "runs", run, "results", f"{label}.json")), "aggregate")


def _tofu_flat(pool, label):
    return _load(os.path.join(pool, "results", "extended", f"{label}.json"))


def _dbp_row(name, agg, extra=""):
    cells = " | ".join(f"{_fmt(_get(agg, c)):>7}" for c in DBP_COLS)
    n = _get(agg, "num_records")
    return f"| {name:34s} | {cells} | {_fmt(n) if n is not None else '—':>4} |{extra}"


_DBP_HDR = ("| label                              |      em | verbmem |     ppl | canary_em |    N |\n"
            "|------------------------------------|---------|---------|---------|-----------|------|")


# ── E1: seed variance ──────────────────────────────────────────────────────────

def sec_e1(L, root, pool):
    L.append("\n## E1 — seed variance (router retrained at seeds 43/44; retriever/index shared)\n")
    L.append("### DBpedia: router_keys_iid per seed vs the single mean_keys_iid (1/k) baseline\n")
    base = _dbp_agg(root, ARM_A, "mean_keys_iid")
    L.append(_DBP_HDR)
    if base is not None:
        L.append(_dbp_row("mean_keys_iid (baseline, 1/k)", base))
    gaps = []
    for run, seed in SEED_RUNS:
        agg = _dbp_agg(root, run, "router_keys_iid")
        gap = (_get(agg, "em") - _get(base, "em")) if _num(_get(agg, "em")) and _num(_get(base, "em")) else None
        gaps.append(gap)
        L.append(_dbp_row(f"router_keys_iid s{seed}", agg, f"  Δem vs mean: {_fmt(gap)}"))
    L.append("")
    L.append(_verdict(gaps, "DBpedia router−mean em gap"))

    L.append("\n### TOFU: routerkey vs legonet (key route, same experts; extended metrics)\n")
    L.append("| label                     | model_utility | forget_quality |\n"
             "|---------------------------|---------------|----------------|")
    for cond in ["full", "unlearn"]:
        base = _tofu_flat(pool, f"legonet_{cond}")
        L.append(f"| legonet_{cond} (1/k baseline) | {_fmt(_get(base, 'model_utility')):>13} "
                 f"| {_fmt(_get(base, 'forget_quality')):>14} |")
        gaps = []
        for sfx, seed in TOFU_SEED_SFX:
            d = _tofu_flat(pool, f"routerkey_{cond}{sfx}")
            gap = (_get(d, "model_utility") - _get(base, "model_utility")) \
                if _num(_get(d, "model_utility")) and _num(_get(base, "model_utility")) else None
            gaps.append(gap)
            L.append(f"| routerkey_{cond} s{seed}        | {_fmt(_get(d, 'model_utility')):>13} "
                     f"| {_fmt(_get(d, 'forget_quality')):>14} |")
        L.append("")
        L.append(_verdict(gaps, f"TOFU routerkey−legonet mu gap ({cond})"))


# ── E2: alpha diagnostics ──────────────────────────────────────────────────────

def sec_e2(L, root, pool):
    L.append("\n## E2 — router alpha diagnostics (analyze_router{,_tofu}.py)\n")
    files = [("dbp keys", os.path.join(root, "runs", ARM_A, "results", "alpha_diag_keys.json")),
             ("dbp retriever", os.path.join(root, "runs", ARM_A, "results", "alpha_diag_retriever.json")),
             ("dbp keys d0-router", os.path.join(root, "runs", ARM_A, "results", "alpha_diag_keys_d0.json")),
             ("tofu key full", os.path.join(pool, "results", "alpha_diag_key_full.json")),
             ("tofu key unlearn", os.path.join(pool, "results", "alpha_diag_key_unlearn.json"))]
    diags = [(name, _load(p)) for name, p in files]
    L.append("| diag                | n | H_norm (±std) | max_share | ideal_mass | ideal_present "
             "| ρ(sharp,em) | ρ(sharp,−logppl) |\n|---|---|---|---|---|---|---|---|")
    for name, d in diags:
        p = _get(d, "pooled") or {}
        h = f"{_fmt(p.get('H_norm_mean'))} ± {_fmt(p.get('H_norm_std'))}" if p else "—"
        r1, r2 = _get(d, "correlations", "spearman_sharpness_em"), \
            _get(d, "correlations", "spearman_sharpness_neglogppl")
        def _rho(r):
            return f"{_fmt(_get(r, 'rho'))} (p={_fmt(_get(r, 'p'))})" if r else "—"
        L.append(f"| {name} | {_fmt(_get(d, 'n_records'))} | {h} | {_fmt(p.get('max_share_mean'))} "
                 f"| {_fmt(p.get('ideal_mass_mean'))} | {_fmt(p.get('ideal_present_rate'))} "
                 f"| {_rho(r1)} | {_rho(r2)} |")
    dflt, d0 = diags[0][1], diags[2][1]
    if dflt and d0:
        dd = [(k, _get(dflt, "pooled", k), _get(d0, "pooled", k))
              for k in ["H_norm_mean", "max_share_mean", "ideal_mass_mean"]]
        L.append("\n**Dropout contrast (p=0.5 router vs p=0 router, same keys route):** " + "; ".join(
            f"{k}: {_fmt(a)} → {_fmt(b)} (Δ {_fmt((b - a) if _num(a) and _num(b) else None)})"
            for k, a, b in dd))
    for name, d in diags[3:]:
        g = _get(d, "groups")
        if g:
            L.append(f"\n**{name} — forget vs retain groups:**")
            L.append("| group | n | H_norm | max_share | ideal_mass | ideal_present | ppl |\n"
                     "|---|---|---|---|---|---|---|")
            for grp in ["forget", "retain"]:
                gg = g.get(grp) or {}
                L.append(f"| {grp} | {_fmt(gg.get('n'))} | {_fmt(gg.get('H_norm_mean'))} "
                         f"| {_fmt(gg.get('max_share_mean'))} | {_fmt(gg.get('ideal_mass_mean'))} "
                         f"| {_fmt(gg.get('ideal_present_rate'))} | {_fmt(gg.get('ppl_mean'))} |")


# ── E3: routing audits + rebuilt-index evals ───────────────────────────────────

def _policy_tables(d, prefix=""):
    """(name, policies_dict) for every 'policies' mapping anywhere in the tree — covers both
    the TOFU audit (top-level) and whatever per-tag nesting routing_audit.py lands with."""
    if not isinstance(d, dict):
        return
    if isinstance(d.get("policies"), dict):
        yield (prefix or str(d.get("tag", "")), d["policies"])
    for k, v in d.items():
        if k != "policies" and isinstance(v, dict):
            yield from _policy_tables(v, f"{prefix}.{k}" if prefix else str(k))


def _render_audit(L, title, d):
    L.append(f"\n### {title}\n")
    if d is None:
        L.append("(not on disk yet)")
        return
    tables = list(_policy_tables(d))
    # DBpedia routing_audit.py schema: the per-policy rates live under 'orphan_pooled'
    # ({n_records, stale: {...}, rebuilt: {...}} — rates are only computed pooled across
    # tags because d0-d2 are n=1 each), not under a 'policies' key.
    op = d.get("orphan_pooled")
    if isinstance(op, dict):
        pols = {k: v for k, v in op.items() if isinstance(v, dict)}
        if pols:
            tables.append((f"orphan_pooled (n={_fmt(op.get('n_records'))} forget records, "
                           "all tags)", pols))
    if not tables:
        L.append("(no 'policies' block found — unexpected schema)")
    for name, pols in tables:
        cols = [c for c in AUDIT_COLS
                if any(c in v for v in pols.values() if isinstance(v, dict))]
        if not cols:   # schema drift: fall back to the union of numeric leaves (capped)
            cols = sorted({k for v in pols.values() if isinstance(v, dict)
                           for k, x in v.items() if _num(x)})[:6]
        tag = d.get("tag") if isinstance(d.get("tag"), str) else "—"
        L.append(f"**{name or 'audit'}** (file tag={tag})\n")
        L.append("| policy | " + " | ".join(cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for pol, m in pols.items():
            L.append(f"| {pol} | " + " | ".join(_fmt(_get(m, c)) for c in cols) + " |")
    # selection shift: TOFU = top-level {name: {shift_*}}; DBpedia = one flat dict per tag
    sh_rows = {}
    sh = _get(d, "selection_shift")
    if isinstance(sh, dict):
        if any(isinstance(v, dict) for v in sh.values()):
            sh_rows.update({k: v for k, v in sh.items() if isinstance(v, dict)})
        elif "shift_topk" in sh:
            sh_rows["retain"] = sh
    tags = d.get("tags") if isinstance(d.get("tags"), dict) else {}
    for tag, t in tags.items():
        if isinstance(t, dict) and isinstance(t.get("selection_shift"), dict):
            sh_rows[tag] = t["selection_shift"]
    if sh_rows:
        L.append("\n| retain selection shift | shift_topk | shift_top1 | mean_jaccard |\n|---|---|---|---|")
        for k, v in sh_rows.items():
            L.append(f"| {k} | {_fmt(_get(v, 'shift_topk'))} | {_fmt(_get(v, 'shift_top1'))} "
                     f"| {_fmt(_get(v, 'mean_jaccard'))} |")

    def _disp_line(name, disp):
        if not isinstance(disp, dict):
            return
        if "mean_cos_affected" in disp or "min_cos_affected" in disp:   # TOFU audit schema
            L.append(f"\nIndex displacement ({name}): mean_cos_affected="
                     f"{_fmt(disp.get('mean_cos_affected'))}, min_cos_affected="
                     f"{_fmt(disp.get('min_cos_affected'))}, untouched_bit_equal="
                     f"{disp.get('untouched_bit_equal', '—')}")
        else:   # DBpedia routing_audit.py: cos_affected_top1 + bit_equal_untouched maps
            def _dvals(key):
                v = disp.get(key)
                return list(v.values()) if isinstance(v, dict) else []
            cos = [v for v in _dvals("cos_affected_top1") if _num(v)]
            be = [bool(b) for b in _dvals("bit_equal_untouched")]
            L.append(f"\nIndex displacement ({name}): mean_cos_affected="
                     f"{_fmt(sum(cos) / len(cos)) if cos else '—'}, min_cos_affected="
                     f"{_fmt(min(cos)) if cos else '—'}, untouched_bit_equal="
                     f"{(str(sum(be)) + '/' + str(len(be))) if be else '—'}")

    _disp_line("stale vs rebuilt", _get(d, "index_displacement"))
    for tag, t in tags.items():
        if isinstance(t, dict):
            _disp_line(tag, t.get("index_displacement"))


def sec_e3(L, root, pool):
    L.append("\n## E3 — post-deletion routing audits + rebuilt-index serving\n")
    _render_audit(L, "TOFU audit (routing_audit_forget10.json)",
                  _load(os.path.join(pool, "results", "routing_audit_forget10.json")))
    _render_audit(L, "DBpedia audit (routing_audit.json; tags d0 d1 d2 d_batch15)",
                  _load(os.path.join(root, "runs", ARM_A, "results", "routing_audit.json")))

    L.append("\n### TOFU rebuilt-index eval (embed route): ramolerb_* vs the stale ramole_*\n")
    L.append("| label | model_utility | forget_quality |\n|---|---|---|")
    for lbl in ["ramole_full", "ramolerb_full", "ramole_unlearn", "ramolerb_unlearn"]:
        d = _tofu_flat(pool, lbl)
        L.append(f"| {lbl} | {_fmt(_get(d, 'model_utility'))} | {_fmt(_get(d, 'forget_quality'))} |")

    L.append("\n### DBpedia rebuilt-index eval (retriever route, unlearn 'after')\n")
    L.append(_DBP_HDR)
    for tag in ["d0", "d1", "d2"]:
        for sfx, name in [("", "stale index"), ("_rebuilt", "REBUILT index")]:
            agg = _dbp_agg(root, ARM_A, f"router_unlearn_{tag}_after{sfx}")
            L.append(_dbp_row(f"router {tag} after — {name}", agg))


# ── E4: serve-time k sweep ─────────────────────────────────────────────────────

def sec_e4(L, root):
    L.append("\n## E4 — serve-time k sweep (same trained router/index; keys route, iid)\n")
    L.append(_DBP_HDR)
    for k, sfx in [(3, ""), (5, "_k5"), (8, "_k8")]:
        r = _dbp_agg(root, ARM_A, f"router_keys_iid{sfx}")
        m = _dbp_agg(root, ARM_A, f"mean_keys_iid{sfx}")
        gap = (_get(r, "em") - _get(m, "em")) if _num(_get(r, "em")) and _num(_get(m, "em")) else None
        L.append(_dbp_row(f"k={k} router", r))
        L.append(_dbp_row(f"k={k} mean (1/k)", m, f"  Δem(router−mean): {_fmt(gap)}"))


# ── E5: serving throughput ─────────────────────────────────────────────────────

def sec_e5(L, root):
    L.append("\n## E5 — serving throughput (benchmark_serving.py, greedy gen)\n")
    d = _load(os.path.join(root, "runs", ARM_A, "results", "throughput.json"))
    if d is None:
        L.append("(throughput.json not on disk yet)")
        return
    modes = list((_get(d, "modes") or {}).keys()) or ["ramole_batched", "merge_per_group", "single_expert"]
    L.append(f"gen_tokens={_fmt(_get(d, 'gen_tokens'))}, iters={_fmt(_get(d, 'iters'))} "
             f"(first iter = discarded warmup), device={d.get('device', '—')}\n")
    L.append("| batch | union | groups | " + " | ".join(f"{m} tok/s" for m in modes) + " |")
    L.append("|---" * (3 + len(modes)) + "|")
    for b in (_get(d, "batch_sizes") or []):
        b = str(b)
        cells = [b, _fmt(_get(d, "batches", b, "union_size")),
                 _fmt(_get(d, "modes", "merge_per_group", b, "n_groups"))]
        cells += [_fmt(_get(d, "modes", m, b, "tokens_per_s"), nd=2) for m in modes]
        L.append("| " + " | ".join(cells) + " |")


# ── E6: 15-record batch deletion ───────────────────────────────────────────────

def sec_e6(L, root):
    L.append("\n## E6 — batch deletion d_batch15 (15 seeded records; router NOT retrained)\n")
    L.append("Forget records served before vs after the affected-expert retrains; memorization "
             "(em/verbmem) should drop for both composers with no router change.\n")
    L.append(_DBP_HDR)
    for method in ["router", "mean"]:
        aggs = {s: _dbp_agg(root, ARM_A, f"{method}_unlearn_d_batch15_{s}") for s in ["before", "after"]}
        dem = (_get(aggs["after"], "em") - _get(aggs["before"], "em")) \
            if _num(_get(aggs["after"], "em")) and _num(_get(aggs["before"], "em")) else None
        for state in ["before", "after"]:
            L.append(_dbp_row(f"{method} d_batch15 {state}",
                              aggs[state], f"  Δem(after−before): {_fmt(dem)}" if state == "after" else ""))


# ── driver ─────────────────────────────────────────────────────────────────────

def build_report(root, pool):
    L = [f"# Follow-up campaign report (E1–E6)\n",
         f"DBpedia root: `{root}` (arm A: `{ARM_A}`)  ·  TOFU pool: `{pool}`",
         "Cells are '—' wherever the producing job has not landed its file yet; re-run anytime."]
    for sec, args in [(sec_e1, (root, pool)), (sec_e2, (root, pool)), (sec_e3, (root, pool)),
                      (sec_e4, (root,)), (sec_e5, (root,)), (sec_e6, (root,))]:
        try:
            sec(L, *args)
        except Exception as e:   # backstop only: a section must never kill the report
            L.append(f"\n> SECTION ERROR in {sec.__name__}: {type(e).__name__}: {e}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.join(os.environ["TOFU_CKPT_STORE"], "ramole"))
    ap.add_argument("--tofu_pool",
                    default=os.path.join(os.environ["TOFU_CKPT_ROOT"], "Llama-3.2-1B-Instruct_legonet_n32_k3"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = build_report(args.root, args.tofu_pool)
    out = args.out or os.path.join(args.root, "FOLLOWUP_REPORT.md")
    with open(out, "w") as f:
        f.write(report)
    print(report)
    print(f"[collect_followup] -> {out}")


if __name__ == "__main__":
    main()
