"""H24 / §4.6 — the defense frontier: how cheap can a `ppl` orphan detector be?

H22 found that `ppl` is the one selector whose orphan detection survives the query no longer
naming its source (name-stripped AUC 0.782/0.799 on the r32 pools, against a 0.57-0.61 published
confidence band). That makes it a candidate DEFENSE: a router that refuses rather than serving an
orphan, with no deletion record consulted.

The objection is cost. `ppl` scores a query by running EVERY candidate expert and reading its
loss, so at k=200 one query costs 200 forward passes and no one would deploy it. This asks
whether it can be cheapened the obvious way: use a free lexical selector (`key_tfidf`) to
prefilter to the top-m candidates, run `ppl` on those m only, and detect from the m scores.

Both matrices come from the FAMILY NPZ CONTRACT and are already on disk, so this needs NO GPU —
the expensive part was producing the score matrices, and it is already paid.

The features are permutation-invariant (sorted top-m, margins, moments) via analyze_router_probe,
which is exactly what makes a per-query, variable candidate SET comparable across queries — the
columns of the restricted matrix are different experts for different queries, and nothing in the
probe may depend on which.

  python analyze_selector_cost.py --self_test
  python analyze_selector_cost.py --pool_dir <pool> --out_json reports/cost_e25.json \
      --out_md reports/cost_e25.md
"""
from __future__ import annotations

import argparse
import json
import os
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

DEFAULT_M = (2, 4, 8, 16, 32, 64)


def prefilter_topm(scores_rank: np.ndarray, scores_score: np.ndarray, m: int):
    """Per query, keep the m columns ranked highest by `scores_rank` and return the matching
    `scores_score` entries. Returns (restricted [n_q, m], chosen column indices [n_q, m]).

    Both matrices must already be restricted to the SURVIVING units and share a column order —
    a prefilter that ranked over deleted columns would be consulting the deletion record, which
    is the whole thing the probe is not allowed to do.
    """
    if scores_rank.shape != scores_score.shape:
        raise ValueError(f"shape mismatch {scores_rank.shape} vs {scores_score.shape}")
    m = int(min(m, scores_rank.shape[1]))
    idx = np.argsort(-scores_rank, axis=1)[:, :m]
    return np.take_along_axis(scores_score, idx, axis=1), idx


def own_expert_recall(idx: np.ndarray, authors: np.ndarray, surv: np.ndarray) -> float:
    """Fraction of RETAINED queries whose own expert survived the prefilter. Reported beside the
    AUC because the two can diverge: detection reads the SHAPE of the score distribution, so a
    prefilter that often loses the right expert can still detect orphans well."""
    pos = {int(a): i for i, a in enumerate(surv)}
    hits, n = 0, 0
    for i, a in enumerate(authors):
        j = pos.get(int(a))
        if j is None:
            continue            # this query's own unit is deleted — it is an orphan, not retained
        n += 1
        hits += int(j in set(idx[i].tolist()))
    return float(hits / n) if n else float("nan")


def cost_curve(ppl_npz: str, rank_npz: str, drop_ids: list, m_values=DEFAULT_M,
               seed: int = 42) -> dict:
    zp, zr = np.load(ppl_npz, allow_pickle=False), np.load(rank_npz, allow_pickle=False)
    Sp = np.asarray(zp["scores"], dtype="float64")
    Sr = np.asarray(zr["scores"], dtype="float64")
    y = np.asarray(zp["is_forget"]).astype(bool)
    au = np.asarray(zp["author_of_q"], dtype=int)
    k = int(zp["k"])
    # The two audits must have scored the SAME queries in the same order, or every restricted
    # row would pair one query's ppl with another's prefilter. Silent misalignment here would
    # produce a plausible curve, which is this campaign's recurring failure mode.
    if not np.array_equal(au, np.asarray(zr["author_of_q"], dtype=int)):
        raise SystemExit("author_of_q differs between the ppl and prefilter npz — different "
                         "query samples, refusing to pair them")
    if not np.array_equal(y, np.asarray(zr["is_forget"]).astype(bool)):
        raise SystemExit("is_forget differs between the ppl and prefilter npz")

    dset = set(int(j) for j in drop_ids)
    surv = np.asarray([j for j in range(k) if j not in dset], dtype=int)
    Sp_s, Sr_s = Sp[:, surv], Sr[:, surv]

    rows = []
    full = probe_arrays(Sp_s, y, au, len(surv), "ppl_full", [], seed=seed)
    for m in list(m_values) + [len(surv)]:
        if m > len(surv):
            continue
        if m == len(surv):
            r, keep = full, 1.0
        else:
            Sm, idx = prefilter_topm(Sr_s, Sp_s, m)
            r = probe_arrays(Sm, y, au, m, f"ppl_top{m}", [], seed=seed)
            keep = own_expert_recall(idx, au, surv)
        rows.append({
            "m": int(m), "forwards_per_query": int(m),
            "speedup_vs_full": float(len(surv) / m),
            "probe_auc": float(r["probe"]["auc"]),
            "best_confidence_auc": float(max(x["auc"] for x in r["comparators"].values())),
            "retain_fpr_at_90_catch": float(r["probe"]["retain_fpr"]),
            "own_expert_recall": float(keep),
        })
    fa = full["probe"]["auc"]
    # "cheapest m within 0.02 AUC of scoring every survivor" — 0.02 is the tolerance the E1
    # pre-registration already uses to call two detectors equivalent.
    afford = [r for r in rows if r["m"] < len(surv) and r["probe_auc"] >= fa - 0.02]
    return {
        "meta": {"ppl_npz": ppl_npz, "rank_npz": rank_npz, "n_survivors": int(len(surv)),
                 "drop_set": sorted(dset), "seed": seed},
        "curve": rows,
        "verdict": {
            "full_auc": float(fa),
            "cheapest_m_within_0.02": int(min(r["m"] for r in afford)) if afford else None,
            "speedup_at_that_m": float(len(surv) / min(r["m"] for r in afford)) if afford else None,
            "note": ("a ppl refusal gate can be run at this many forward passes per query "
                     "without measurably weaker orphan detection" if afford else
                     "no prefiltered budget matches full scoring within 0.02 AUC"),
        },
    }


def write_md(res: dict, path: str) -> None:
    v = res["verdict"]
    L = ["# Defense frontier — how cheap can a `ppl` orphan detector be? (H24 / §4.6)", "",
         "`ppl` is the one selector whose orphan detection survives name-stripping, which makes "
         "it a candidate refusal gate. Scoring all survivors costs one forward pass per expert; "
         "here a free `key_tfidf` prefilter picks the top-m and `ppl` runs on those only. "
         "**No deletion record is consulted at any point**, and the prefilter ranks over "
         "surviving units only.", "",
         f"Survivors: {res['meta']['n_survivors']} · full-scoring AUC **{_f(v['full_auc'])}**", "",
         "| m (fwd/query) | speedup | probe AUC | best-conf AUC | FPR@90% catch | own-expert recall |",
         "|---|---|---|---|---|---|"]
    for r in res["curve"]:
        L.append(f"| {r['m']} | {r['speedup_vs_full']:.0f}x | **{_f(r['probe_auc'])}** | "
                 f"{_f(r['best_confidence_auc'])} | {_f(r['retain_fpr_at_90_catch'])} | "
                 f"{_f(r['own_expert_recall'])} |")
    L += ["", f"**Verdict:** {v['note']}"]
    if v["cheapest_m_within_0.02"]:
        L.append(f" — m = **{v['cheapest_m_within_0.02']}** "
                 f"({v['speedup_at_that_m']:.0f}x cheaper than scoring every survivor).")
    L += ["", "`own-expert recall` is the fraction of RETAINED queries whose own expert survived "
          "the prefilter. It can sit well below 1.0 while the AUC holds, because detection reads "
          "the SHAPE of the score distribution rather than requiring correct routing."]
    with open(path, "w") as f:
        f.write("\n".join(L))


def run_self_test() -> None:
    n = 0

    def ok(name):
        nonlocal n
        n += 1
        print(f"  PASS {name}")

    # prefilter picks by the RANK matrix and returns the SCORE matrix's entries — a fixture where
    # the two disagree is the only one that can catch them being swapped
    rank = np.array([[0.0, 9.0, 1.0], [9.0, 0.0, 1.0]])
    score = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    got, idx = prefilter_topm(rank, score, 2)
    assert np.array_equal(idx, np.array([[1, 2], [0, 2]])), idx
    assert np.array_equal(got, np.array([[20.0, 30.0], [40.0, 60.0]])), got
    ok("prefilter ranks by one matrix and returns the other's values")

    assert prefilter_topm(rank, score, 99)[0].shape[1] == 3
    ok("m larger than the candidate set is clamped, not an error")

    # own_expert_recall: author ids are AUTHOR ids and must be mapped through surv
    surv = np.array([0, 2, 5])
    idx2 = np.array([[0, 1], [2, 0]])          # positions into surv
    au = np.array([0, 5])                      # author 0 -> pos 0 (in), author 5 -> pos 2 (in)
    assert own_expert_recall(idx2, au, surv) == 1.0
    au2 = np.array([2, 2])                     # author 2 -> pos 1; row0 has it, row1 does not
    assert own_expert_recall(idx2, au2, surv) == 0.5
    ok("own-expert recall maps author ids through the survivor list")

    au3 = np.array([7, 7])                     # author 7 is not a survivor => orphan, excluded
    assert np.isnan(own_expert_recall(idx2, au3, surv))
    ok("orphan queries excluded from the retained-recall denominator")

    # a planted separable case: orphans get uniformly low ppl on every expert, retained get one
    # high column. Detection must be near-perfect at any m, and the prefilter must not break it.
    rng = np.random.RandomState(0)
    k, per = 20, 20
    au4 = np.repeat(np.arange(k), per)
    y4 = au4 >= 16
    S = rng.normal(0, 0.01, size=(k * per, k))
    for i, a in enumerate(au4):
        if not y4[i]:
            S[i, a] += 5.0
    r = probe_arrays(S[:, :16], y4, au4, 16, "planted", [], seed=0)
    assert r["probe"]["auc"] > 0.95, r["probe"]["auc"]
    ok(f"planted separable fixture detects at AUC {_f(r['probe']['auc'])}")

    print(f"[analyze_selector_cost] self_test: {n}/{n} PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ppl_npz", default=None, help="behavioral ppl npz (the expensive selector)")
    ap.add_argument("--rank_npz", default=None, help="cheap prefilter npz (default key_tfidf)")
    ap.add_argument("--pool_dir", default=None,
                    help="derive both npz from a pool's results/router_leak (name-stripped)")
    ap.add_argument("--variant", default="_name_stripped",
                    help="query-transform suffix; '' for gold-form")
    ap.add_argument("--drop_set", default="180-199")
    ap.add_argument("--m", default=",".join(str(x) for x in DEFAULT_M))
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return
    ppl, rank = args.ppl_npz, args.rank_npz
    if args.pool_dir:
        rl = os.path.join(args.pool_dir, "results", "router_leak")
        ppl = ppl or os.path.join(rl, f"rl_family_k200_beh{args.variant}.ppl.npz")
        rank = rank or os.path.join(rl, f"rl_family_k200{args.variant}.key_tfidf.npz")
    if not (ppl and rank):
        raise SystemExit("need --pool_dir, or both --ppl_npz and --rank_npz (or --self_test)")
    for p in (ppl, rank):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    res = cost_curve(ppl, rank, parse_drop_set(args.drop_set),
                     [int(x) for x in args.m.split(",") if x.strip()])
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[cost] -> {args.out_json}")
    if args.out_md:
        write_md(res, args.out_md)
        print(f"[cost] -> {args.out_md}")
    v = res["verdict"]
    for r in res["curve"]:
        print(f"  m={r['m']:>4d}  fwd/query {r['forwards_per_query']:>4d}  "
              f"AUC {_f(r['probe_auc'])}  own-recall {_f(r['own_expert_recall'])}")
    print(f"  full AUC {_f(v['full_auc'])} | cheapest m within 0.02: {v['cheapest_m_within_0.02']}")


if __name__ == "__main__":
    main()
