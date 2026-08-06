"""E3 post-deletion routing audit for the RAMoLE-on-TOFU arm — routing-only, NO LLM loaded.

WHY: deletion here RETRAINS the affected experts (legonet unlearn), it never drops them, so
the model side is exact by construction. The residual risk is the ROUTER: the embed-route
expert index (`legonet/ramole/expert_index_n{n}.npy`) was built as member-author means that
INCLUDE the forget authors and is never rebuilt on deletion (index staleness). This audit
quantifies, without loading the 1B model, (a) where forget-author ("orphan") questions land
under each routing policy, (b) how much RETAIN routing shifts if the index is rebuilt
retain-only (collateral of fixing staleness), and (c) per-expert index displacement.

Policies:
  stale   — the as-built expert index (cached default; the encoder-centroid-leak arm)
  rebuilt — member means excluding cfg['forget_authors'] (cached to a DISTINCT `_ex...` file
            by build_expert_index; the stale file existing results depend on is never touched)
  dropped — the §9-D drop-an-expert condition: the manifest's affected experts are MASKED
            (−inf) from the stale index before ranking. K(a) ⊆ affected for every forget
            author, so orig/affected columns are trivially 0 — the informative outputs are
            `dropped_extras` (orphan top-1 concentration over surviving experts + the
            masked/unmasked top-1 similarity ratio) and the retain selection shift vs stale.
  key     — the LegoNet frozen author->keys lookup (deletion-invariant comparison arm)

Encoder provenance: build_expert_index caches per encoder pin (`encoder_pin:"base"` →
`_encbase` files), so a base-pinned audit measures the off-the-shelf encoder instead of
silently reloading the FT-built stale cache.

Rank bug this avoids: `rt.RamoleRouter.route` returns `tuple(sorted(idxs))` — sorted by
EXPERT ID, not similarity rank — so top-1/rank metrics derived from it are WRONG. All rank
metrics here come from a rank-preserving `argsort(-sims)` computed directly on the index.

    python routing_audit_tofu.py --config configs/ramole_tofu_1b.json --tag forget10 \
        --policies stale rebuilt key --device cuda --out .../routing_audit_forget10.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

import legonet_tofu as lt
import ramole_tofu as rt          # inserts ramole/ on sys.path
import ramole_common as rc        # noqa: E402

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


# ── rank-preserving routing ─────────────────────────────────────────────────────

def rank_topk(index: np.ndarray, Q: np.ndarray, k: int, mask_experts=None) -> np.ndarray:
    """(N, k) expert ids in DESCENDING similarity order (rank-preserving; stable ties).
    This is the correct top-1/rank source — never use RamoleRouter.route's sorted tuple.
    `mask_experts` (dropped policy) removes those ids from the pool via a −inf similarity."""
    sims = np.asarray(Q, dtype="float32") @ np.asarray(index, dtype="float32").T
    if mask_experts is not None:
        sims[:, sorted(int(j) for j in mask_experts)] = -np.inf
    return np.argsort(-sims, axis=1, kind="stable")[:, :k]


def key_topk(assignment: dict, q2author: dict, questions: list, k: int) -> np.ndarray:
    """(N, k) via the frozen author->keys lookup (author_to_keys is distance-ordered, so
    rank is preserved). Raises on any question q2author cannot resolve — every TOFU
    question is in-distribution, a miss means the audit inputs are wrong."""
    rows = []
    for q in questions:
        a = q2author.get(lt._norm(q))
        if a is None:
            raise KeyError(f"question not in q2author (not a TOFU question?): {q[:80]!r}")
        rows.append(lt.author_keys(assignment, int(a))[:k])
    return np.asarray(rows, dtype=int)


# ── metrics ────────────────────────────────────────────────────────────────────

def orphan_metrics(routes: np.ndarray, authors: list, assignment: dict, manifest: dict) -> dict:
    """Where do the forget ('orphan') questions land? K(a) = the author's original experts."""
    aff, unt = set(manifest["affected_adapters"]), set(manifest["untouched_adapters"])
    n_q, k = routes.shape
    topk_eq = top1_in = 0
    jac = 0.0
    slots_aff = slots_unt = 0
    hist: dict = defaultdict(lambda: defaultdict(int))
    for i, a in enumerate(authors):
        K = set(int(j) for j in lt.author_keys(assignment, int(a)))
        s = set(int(j) for j in routes[i])
        top1 = int(routes[i][0])
        topk_eq += int(s == K)
        top1_in += int(top1 in K)
        jac += len(s & K) / len(s | K)
        slots_aff += sum(1 for j in routes[i] if int(j) in aff)
        slots_unt += sum(1 for j in routes[i] if int(j) in unt)
        hist[str(int(a))][str(top1)] += 1
    return {
        "n_questions": int(n_q),
        "orig_topk_rate": topk_eq / n_q,
        "orig_top1_rate": top1_in / n_q,
        "sibling_top1_rate": 1.0 - top1_in / n_q,
        "affected_mass": slots_aff / (n_q * k),
        "untouched_mass": slots_unt / (n_q * k),
        "mean_jaccard": jac / n_q,
        "per_author": {a: dict(h) for a, h in sorted(hist.items(), key=lambda kv: int(kv[0]))},
    }


def selection_shift(ra: np.ndarray, rb: np.ndarray) -> dict:
    """How often two policies pick different experts for the same (retain) question."""
    assert ra.shape == rb.shape
    n = ra.shape[0]
    d_topk = sum(set(map(int, x)) != set(map(int, y)) for x, y in zip(ra, rb))
    d_top1 = sum(int(x[0]) != int(y[0]) for x, y in zip(ra, rb))
    jac = float(np.mean([len(set(map(int, x)) & set(map(int, y)))
                         / len(set(map(int, x)) | set(map(int, y))) for x, y in zip(ra, rb)]))
    return {"n": int(n), "shift_topk": d_topk / n, "shift_top1": d_top1 / n, "mean_jaccard": jac}


def dropped_extras(index: np.ndarray, Q_forget: np.ndarray, affected: list, k: int) -> dict:
    """The informative dropped-policy outputs (orig/affected orphan columns are trivial
    because K(a) ⊆ affected): (a) how CONCENTRATED orphan top-1 mass is over the surviving
    experts (few near-duplicate siblings ⇒ leak-prone; uniform spread ⇒ generic/benign) and
    (b) the masked/unmasked top-1 cosine ratio (≈1 ⇒ the surviving sibling matches the query
    almost as well as the dropped expert did)."""
    sims = np.asarray(Q_forget, dtype="float32") @ np.asarray(index, dtype="float32").T
    top1_unmasked = sims.max(axis=1)
    ms = sims.copy()
    ms[:, sorted(int(j) for j in affected)] = -np.inf
    top1_masked_id = ms.argmax(axis=1)
    top1_masked = ms.max(axis=1)
    n_surv = index.shape[0] - len(affected)
    counts = np.bincount(top1_masked_id, minlength=index.shape[0]).astype("float64")
    p = counts[counts > 0] / counts.sum()
    ent = float(-(p * np.log(p)).sum() / np.log(max(n_surv, 2)))
    ranked = np.sort(counts)[::-1] / counts.sum()
    # cosine sims here are positive in practice; guard the ratio anyway
    ratio = top1_masked / np.maximum(top1_unmasked, 1e-12)
    return {
        "n_orphans": int(sims.shape[0]),
        "n_surviving_experts": int(n_surv),
        "top1_share_top1_expert": float(ranked[0]),
        "top1_share_top3_experts": float(ranked[:3].sum()),
        "top1_entropy_norm": ent,
        "mean_top1_sim_ratio": float(ratio.mean()),
        "p10_top1_sim_ratio": float(np.percentile(ratio, 10)),
        "p90_top1_sim_ratio": float(np.percentile(ratio, 90)),
        "top1_hist": {str(int(j)): int(counts[j]) for j in np.nonzero(counts)[0]},
    }


def abstain_analysis(index: np.ndarray, Q_forget: np.ndarray, Q_retain: np.ndarray,
                     affected: list, pcts=(1, 5, 10)) -> dict:
    """C1 fix arm — does an OOD-threshold route seal the drop-an-expert fallback leak? Calibrate
    τ on RETAIN top-1 similarity (retain queries route over the FULL index — no forget data touches
    the threshold, per the §5.2 centroid-leak discipline), then, for forget ORPHANS routed over the
    dropped index (affected experts masked), abstain (→ base/scaffold) when their masked top-1 sim
    < τ. Reports, per percentile: orphan→abstain rate (want ↑), retain false-abstain rate (the
    collateral, ≈ p%), and the sibling-capture rate among orphans that did NOT abstain."""
    aff = sorted(int(j) for j in affected)
    # retain top-1 sim over the full (undropped) index — the "normal confidence" distribution
    r_sims = np.asarray(Q_retain, "float32") @ np.asarray(index, "float32").T
    r_top1 = r_sims.max(axis=1)
    # orphan top-1 sim over the DROPPED index (affected masked)
    f_sims = np.asarray(Q_forget, "float32") @ np.asarray(index, "float32").T
    f_masked = f_sims.copy(); f_masked[:, aff] = -np.inf
    f_top1 = f_masked.max(axis=1)
    out = {"n_orphans": int(f_top1.shape[0]), "n_retain": int(r_top1.shape[0]), "by_pct": {}}
    for p in pcts:
        tau = float(np.percentile(r_top1, p))
        out["by_pct"][str(p)] = {
            "tau": tau,
            "orphan_abstain_rate": float((f_top1 < tau).mean()),
            "retain_false_abstain_rate": float((r_top1 < tau).mean()),
            "orphan_sibling_rate_if_no_abstain": float((f_top1 >= tau).mean()),
        }
    return out


def _author_sentinels(Q: np.ndarray, authors: list, per: int) -> np.ndarray:
    """Per-author sentinel centroids: L2-normed mean of each author's `per` question
    embeddings. Derived from the SAME embedding pass as routing (no extra encoder cost).
    NOTE the provenance: these sentinels are forget-author-data-derived — the middle rung
    of the tombstone provenance ladder (expert key -> author centroid -> name embedding)."""
    rows = []
    for a in authors:
        v = Q[a * per:(a + 1) * per].mean(0)
        rows.append(v / (np.linalg.norm(v) + 1e-12))
    return np.stack(rows).astype("float32")


def _author_names(data_full, authors: list, per: int) -> tuple[list, int]:
    """Longest extracted name string per author (router._extract_author_names — capitalized
    phrases in >=50% of the author's questions). Fallback = None (counted); the name rung
    then reuses that author's data-derived sentinel so pool shapes stay aligned."""
    from router import _extract_author_names
    names, n_fallback = [], 0
    for a in authors:
        qs = [data_full[a * per + w]["question"] for w in range(per)]
        cand = _extract_author_names(qs)
        if cand:
            names.append(max(cand, key=len))
        else:
            names.append(None)
            n_fallback += 1
    return names, n_fallback


def tombstone_analysis(index: np.ndarray, Q_f: np.ndarray, Q_r: np.ndarray, affected: list,
                       sent_author: np.ndarray = None, sent_name: np.ndarray = None) -> dict:
    """The identity-seal policy family (H1/H2): keep DELETION IDENTITY signals in the routing
    pool and send their top-1 hits to base/scaffold instead of a surviving sibling. Three
    provenance rungs, decreasing retained-data footprint:
      expert — pool = the full stale index; tombstoned iff top-1 ∈ affected (per-EXPERT keys;
               ill-posed when affected experts also host retain authors — the FPR shows it).
      author — pool = surviving expert rows + per-author sentinel centroids; tombstoned iff
               top-1 is a sentinel.
      name   — same with name-string embeddings (no QA data retained).
    Per rung: orphan_catch_rate (orphans -> base, want high), orphan_leak_rate (residual
    sibling capture), retain_false_tombstone_rate (the collateral, want ~0), and the mean
    tombstone margin (best-tombstone-sim − best-survivor-sim) for both query groups."""
    aff = sorted(int(j) for j in affected)
    surv = [j for j in range(index.shape[0]) if j not in set(aff)]

    def _rates(pool: np.ndarray, tomb_cols: list) -> dict:
        fs, rs = Q_f @ pool.T, Q_r @ pool.T
        tomb = np.zeros(pool.shape[0], dtype=bool)
        tomb[tomb_cols] = True
        f_top1, r_top1 = fs.argmax(1), rs.argmax(1)
        surv_cols = ~tomb
        f_margin = fs[:, tomb].max(1) - fs[:, surv_cols].max(1)
        r_margin = rs[:, tomb].max(1) - rs[:, surv_cols].max(1)
        return {
            "orphan_catch_rate": float(tomb[f_top1].mean()),
            "orphan_leak_rate": float((~tomb[f_top1]).mean()),
            "retain_false_tombstone_rate": float(tomb[r_top1].mean()),
            "orphan_margin_mean": float(f_margin.mean()),
            "retain_margin_mean": float(r_margin.mean()),
        }

    out = {"affected": aff, "n_surviving": len(surv)}
    out["expert"] = _rates(index, aff)
    if sent_author is not None:
        pool = np.concatenate([index[surv], sent_author], axis=0)
        out["author"] = _rates(pool, list(range(len(surv), pool.shape[0])))
    if sent_name is not None:
        pool = np.concatenate([index[surv], sent_name], axis=0)
        out["name"] = _rates(pool, list(range(len(surv), pool.shape[0])))
    return out


def index_displacement(idx_stale: np.ndarray, idx_reb: np.ndarray, assignment: dict,
                       manifest: dict, forget_authors: list) -> dict:
    """Per-expert cos(stale, rebuilt). build_expert_index is RNG-free (deterministic member
    means over one shared embedding pass), so an expert with NO forget-author members must be
    BIT-IDENTICAL between the two builds — asserted; a diff there means the exclude logic (or
    the stale cache's encoder provenance) is broken."""
    n = idx_stale.shape[0]
    fset = set(int(a) for a in forget_authors)
    unt_members = sorted(j for j in range(n)
                         if not fset & {int(a) for a in assignment["members"][str(j)]})
    if unt_members != sorted(int(j) for j in manifest["untouched_adapters"]):
        raise AssertionError(f"manifest untouched {manifest['untouched_adapters']} != "
                             f"no-forget-member experts {unt_members}")
    cos = [float(np.dot(idx_stale[j], idx_reb[j])
                 / (np.linalg.norm(idx_stale[j]) * np.linalg.norm(idx_reb[j]) + 1e-12))
           for j in range(n)]
    for j in unt_members:
        if not np.array_equal(idx_stale[j], idx_reb[j]):
            raise AssertionError(
                f"untouched expert {j} row differs stale vs rebuilt — exclude logic broken, "
                "or the cached stale index was built with a different encoder/device "
                "(check encoder_source vs the stale file's provenance)")
    affected = sorted(int(j) for j in manifest["affected_adapters"])
    return {
        "cos": cos,
        "affected": affected,
        "untouched": unt_members,
        "untouched_bit_equal": True,   # asserted above
        "min_cos_affected": min(cos[j] for j in affected) if affected else None,
        "mean_cos_affected": float(np.mean([cos[j] for j in affected])) if affected else None,
    }


# ── audit driver ───────────────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based ROC-AUC of score separating pos (want high) from neg. No sklearn dep."""
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype("float64") + 1.0
    # midrank tie correction
    order = np.argsort(scores, kind="stable")
    sorted_s = scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_centroid_audit(args) -> dict:
    """R1 keystone cell: the k=10 shard-CENTROID router audit — the SAME router the R2
    serving arms and eval_entangled_probe's served_embedsim deploy (MiniLM shard centroids),
    unifying the audit and serving routers (the prior §9-D numbers are about the n=32
    instructor-xl index, a router nothing serves TOFU through).

    Routing-only, one MiniLM pass. Computes, with S = drop_shard:
      full            — top-1 over all k centroids (routing-accuracy baseline, no deletion)
      sibling         — S's centroid removed: orphan sibling capture + sim-ratio + retain shift
      tombstone rungs — via tombstone_analysis (shard centroid / per-author sentinel /
                        name-embedding sentinel; provenance ladder)
      c_probe         — Mode-B probe surfaces (--probe_manifest): fraction of planted probe
                        questions the tombstone rung catches (the H4 mediator), + the sibling
                        route's host-hit rate vs the manifest
      disclosure      — deletion-disclosure (Streisand) scores: sim-to-S − best-surviving-sim
                        for deleted-author vs never-trained holdout10 questions (+ AUC)
    Sidecar npz: per-query sims for every group (analyze_router_leak.py post-processing)."""
    from eval_routed_scaffold import build_shard_centroids
    from shard_utils import get_author_shard
    os.environ["HF_HOME"] = args.hf_home
    from datasets import load_dataset

    k, drop, per = args.centroid_k, args.drop_shard, 20
    data_full = load_dataset("locuslab/TOFU", "full")["train"]
    A = 200
    cents, sids, embed = build_shard_centroids(args.hf_home, k, list(range(k)), args.device,
                                               encoder_name=args.router_encoder)
    row_of = {s: i for i, s in enumerate(sids)}
    questions = [data_full[i]["question"] for i in range(A * per)]
    Q = embed(questions)
    shard_of_q = np.asarray([(i // per) // (A // k) for i in range(A * per)])
    forget_rows = np.nonzero(shard_of_q == drop)[0]
    retain_rows = np.nonzero(shard_of_q != drop)[0]
    forget_authors = sorted({i // per for i in forget_rows.tolist()})

    sims = (Q @ cents.T).astype("float32")
    top1_full = sims.argmax(1)
    acc_retain = float((top1_full[retain_rows] == shard_of_q[retain_rows]).mean())
    acc_forget = float((top1_full[forget_rows] == shard_of_q[forget_rows]).mean())

    masked = sims.copy()
    masked[:, row_of[drop]] = -np.inf
    top1_sib = masked.argmax(1)
    sib_hist = {str(sids[j]): int((top1_sib[forget_rows] == j).sum())
                for j in set(top1_sib[forget_rows].tolist())}
    ratio = masked[forget_rows].max(1) / np.maximum(sims[forget_rows].max(1), 1e-12)
    retain_shift = float((top1_sib[retain_rows] != top1_full[retain_rows]).mean())

    sent_author = _author_sentinels(Q, forget_authors, per)
    names, n_fallback = _author_names(data_full, forget_authors, per)
    sent_name = sent_author.copy()
    name_texts = [nm for nm in names if nm is not None]
    if name_texts:
        name_emb = embed(name_texts)
        it = iter(range(len(name_texts)))
        for i, nm in enumerate(names):
            if nm is not None:
                sent_name[i] = name_emb[next(it)]
    tomb = tombstone_analysis(cents, Q[forget_rows], Q[retain_rows], [row_of[drop]],
                              sent_author, sent_name)
    tomb["name_fallbacks"] = n_fallback

    out = {
        "mode": "centroid", "k": k, "drop_shard": drop, "router_encoder": args.router_encoder,
        "n_forget_q": int(len(forget_rows)), "n_retain_q": int(len(retain_rows)),
        "forget_authors": forget_authors,
        "full": {"acc_retain_top1": acc_retain, "acc_forget_top1": acc_forget},
        "sibling": {"retain_shift_top1": retain_shift,
                    "mean_top1_sim_ratio": float(ratio.mean()),
                    "p10_top1_sim_ratio": float(np.percentile(ratio, 10)),
                    "sibling_hist": sib_hist},
        "tombstone": tomb,
    }
    arrs = {
        "sims_centroid_all": sims, "shard_of_q": shard_of_q,
        "forget_rows": forget_rows, "retain_rows": retain_rows,
        "author_of_q": np.asarray([i // per for i in range(A * per)]),
        "sims_author_sent_forget": (Q[forget_rows] @ sent_author.T).astype("float32"),
        "sims_author_sent_retain": (Q[retain_rows] @ sent_author.T).astype("float32"),
        "sims_name_sent_forget": (Q[forget_rows] @ sent_name.T).astype("float32"),
        "sims_name_sent_retain": (Q[retain_rows] @ sent_name.T).astype("float32"),
        "centroid_sids": np.asarray(sids), "forget_authors": np.asarray(forget_authors),
    }

    if args.probe_manifest:
        with open(args.probe_manifest) as f:
            man = json.load(f)
        facts = man["facts"]
        for surf, qkey in (("orig", "probe_question_orig"), ("para", "probe_question_para")):
            Pq = embed([f[qkey] for f in facts])
            ps = (Pq @ cents.T).astype("float32")
            drop_col = row_of[drop]
            catch_shard = ps.argmax(1) == drop_col
            pm = ps.copy(); pm[:, drop_col] = -np.inf
            sib_route = pm.argmax(1)
            host_sets = [set(h["shard"] for h in f["hosts"]) for f in facts]
            host_hit = [sids[int(sib_route[i])] in host_sets[i] and len(host_sets[i]) > 0
                        for i in range(len(facts))]
            # author/name rungs: pool = surviving centroids + sentinels
            surv_cols = [j for j in range(k) if j != drop_col]
            rungs = {"shard": catch_shard}
            for rname, sent in (("author", sent_author), ("name", sent_name)):
                pool = np.concatenate([cents[surv_cols], sent], axis=0)
                pp = Pq @ pool.T
                rungs[rname] = pp.argmax(1) >= len(surv_cols)
            by_R = {}
            for R in sorted({f["R"] for f in facts}):
                idx = [i for i, f in enumerate(facts) if f["R"] == R]
                by_R[f"R{R}"] = {
                    **{f"c_probe_{rn}": float(np.mean([rungs[rn][i] for i in idx]))
                       for rn in rungs},
                    "sibling_host_hit_rate": float(np.mean([host_hit[i] for i in idx])),
                    "n": len(idx),
                }
            out.setdefault("probe", {})[surf] = by_R
            arrs[f"sims_probe_{surf}"] = ps
            arrs[f"probe_R_{surf}"] = np.asarray([f["R"] for f in facts])

    if not args.no_holdout:
        hold = load_dataset("locuslab/TOFU", "holdout10")["train"]
        Hq = embed([hold[i]["question"] for i in range(len(hold))])
        hs = (Hq @ cents.T).astype("float32")
        drop_col = row_of[drop]
        surv_cols = [j for j in range(k) if j != drop_col]
        f_score = sims[forget_rows][:, drop_col] - sims[forget_rows][:, surv_cols].max(1)
        h_score = hs[:, drop_col] - hs[:, surv_cols].max(1)
        out["disclosure"] = {
            "auc_forget_vs_holdout": _auc(f_score, h_score),
            "n_holdout": int(hs.shape[0]),
            "forget_score_mean": float(f_score.mean()), "holdout_score_mean": float(h_score.mean()),
        }
        arrs["sims_holdout"] = hs
        # PER-RUNG disclosure (2026-07-23): only the SHARD rung was ever priced for
        # deletion-disclosure, yet the author rung is what we recommend and the name rung is
        # the privacy-cleanest. Same score shape — best sentinel match minus best surviving
        # centroid match — for deleted-author vs never-trained (holdout10) questions. A seal
        # can be clean in what it STORES and still be loud in how it BEHAVES.
        best_surv_f = sims[forget_rows][:, surv_cols].max(1)
        best_surv_h = hs[:, surv_cols].max(1)
        for rung, S in (("author", sent_author), ("name", sent_name)):
            if S is None or getattr(S, "size", 0) == 0:
                out["disclosure"][f"auc_forget_vs_holdout_{rung}"] = None
                continue
            f_sent = (Q[forget_rows] @ S.T).astype("float32")
            h_sent = (Hq @ S.T).astype("float32")
            out["disclosure"][f"auc_forget_vs_holdout_{rung}"] = _auc(
                f_sent.max(1) - best_surv_f, h_sent.max(1) - best_surv_h)
            arrs[f"sims_{rung}_sent_holdout"] = h_sent

    if args.dump_sims:
        dump_path = args.out.replace(".json", "") + ".sims.npz"
        np.savez_compressed(dump_path, router_encoder=np.str_(args.router_encoder), **arrs)
        out["sims_dump"] = os.path.abspath(dump_path)
    return out


def run_audit(cfg: dict, data_full, tag: str, policies: list, device: str = "cpu",
              encoder=None, dump_sims_path: str = None) -> dict:
    """Route ALL num_authors*per questions under each policy; return the audit dict."""
    rc.set_determinism(cfg["base_seed"])
    A, per, k = cfg["num_authors"], cfg["records_per_author"], cfg["k"]
    forget_authors = sorted(int(a) for a in cfg["forget_authors"])

    with open(lt.unlearn_manifest_path(cfg, tag)) as f:
        manifest = json.load(f)
    if sorted(int(a) for a in manifest["forget_authors"]) != forget_authors:
        raise AssertionError(f"manifest[{tag}] forget_authors != cfg forget_authors")
    with open(lt.assignment_path(cfg)) as f:
        assignment = json.load(f)

    questions = [data_full[i]["question"] for i in range(A * per)]
    forget_rows = [i for a in forget_authors for i in range(a * per, (a + 1) * per)]
    retain_rows = [i for i in range(A * per) if i // per not in set(forget_authors)]
    forget_authors_per_q = [i // per for i in forget_rows]

    encoder_source = rt._encoder_source(cfg)   # E0 provenance: record the RESOLVED source
    # Same embed fn as eval (load_ramole_eval_model): instruction prefix + shared encoder.
    qembed = rc.make_embed_fn(rt._encoder_name(cfg), instruction=rt._instr(cfg),
                              device=device, encoder=encoder)

    routes: dict[str, np.ndarray] = {}
    indices: dict[str, np.ndarray] = {}
    stale_path = rt.expert_index_path(cfg)
    stale_hash = _sha256(stale_path) if os.path.exists(stale_path) else None
    needs_q = {"stale", "rebuilt", "dropped", "abstain", "tombstone"}
    if needs_q & set(policies) or dump_sims_path:
        Q = qembed(questions)   # (A*per, D) — one shared embedding pass for all indices
    if {"stale", "dropped", "abstain", "tombstone"} & set(policies) or dump_sims_path:
        indices["stale"] = rt.build_expert_index(cfg, data_full, device=device, encoder=encoder)
        if "stale" in policies:
            routes["stale"] = rank_topk(indices["stale"], Q, k)
        if stale_hash is None:   # first-ever build: pin the bytes we must not disturb
            stale_hash = _sha256(stale_path)
    if "dropped" in policies:
        affected = sorted(int(j) for j in manifest["affected_adapters"])
        if cfg["n"] - len(affected) < k:
            raise RuntimeError(f"dropped policy needs >= k={k} surviving experts, got "
                               f"{cfg['n'] - len(affected)} (n={cfg['n']}, affected={len(affected)})")
        routes["dropped"] = rank_topk(indices["stale"], Q, k, mask_experts=affected)
    if "rebuilt" in policies:
        indices["rebuilt"] = rt.build_expert_index(cfg, data_full, device=device,
                                                   encoder=encoder, exclude_authors=forget_authors)
        routes["rebuilt"] = rank_topk(indices["rebuilt"], Q, k)
        if stale_hash is not None and _sha256(stale_path) != stale_hash:
            raise AssertionError(f"stale index {stale_path} changed on disk during the audit")
    if "key" in policies:
        q2a = lt.build_q2author(data_full, A, per)
        routes["key"] = key_topk(assignment, q2a, questions, k)

    out: dict = {
        "tag": tag, "encoder_source": encoder_source,
        "n": cfg["n"], "k": k, "num_authors": A,
        "n_forget_q": len(forget_rows), "n_retain_q": len(retain_rows),
        "forget_authors": forget_authors,
        "policies": {}, "selection_shift": {}, "index_displacement": None,
    }
    for pol, r in routes.items():
        out["policies"][pol] = orphan_metrics(r[forget_rows], forget_authors_per_q,
                                              assignment, manifest)
    if "stale" in routes and "rebuilt" in routes:
        out["selection_shift"]["embed_stale_vs_rebuilt"] = selection_shift(
            routes["stale"][retain_rows], routes["rebuilt"][retain_rows])
        out["index_displacement"] = index_displacement(
            indices["stale"], indices["rebuilt"], assignment, manifest, forget_authors)
    if "dropped" in routes:
        affected = sorted(int(j) for j in manifest["affected_adapters"])
        out["dropped_extras"] = dropped_extras(indices["stale"], Q[forget_rows], affected, k)
        if "stale" in routes:
            out["selection_shift"]["embed_stale_vs_dropped"] = selection_shift(
                routes["stale"][retain_rows], routes["dropped"][retain_rows])
    if "abstain" in policies:  # C1 fix arm — routing-only, no route table (threshold analysis)
        affected = sorted(int(j) for j in manifest["affected_adapters"])
        out["abstain"] = abstain_analysis(indices["stale"], Q[forget_rows], Q[retain_rows], affected)
    sent_author = sent_name = names = None
    if "tombstone" in policies:  # H2 identity-seal family — routing-only, no route table
        affected = sorted(int(j) for j in manifest["affected_adapters"])
        sent_author = _author_sentinels(Q, forget_authors, per)
        names, n_fallback = _author_names(data_full, forget_authors, per)
        name_texts = [nm for nm in names if nm is not None]
        sent_name = sent_author.copy()
        if name_texts:
            name_emb = qembed(name_texts)
            it = iter(range(len(name_texts)))
            for i, nm in enumerate(names):
                if nm is not None:
                    sent_name[i] = name_emb[next(it)]
        out["tombstone"] = tombstone_analysis(indices["stale"], Q[forget_rows], Q[retain_rows],
                                              affected, sent_author, sent_name)
        out["tombstone"]["name_fallbacks"] = n_fallback
    if dump_sims_path:
        # Per-query similarity sidecar (H1/H2 threshold-family post-processing lives on CPU
        # in analyze_router_leak.py — the 07-07 abstain arm discarded these, forcing reruns).
        # Never touches the aggregate JSON.
        if sent_author is None:
            sent_author = _author_sentinels(Q, forget_authors, per)
        arrs = {
            "sims_stale_forget": (Q[forget_rows] @ indices["stale"].T).astype("float32"),
            "sims_stale_retain": (Q[retain_rows] @ indices["stale"].T).astype("float32"),
            "sims_author_sent_forget": (Q[forget_rows] @ sent_author.T).astype("float32"),
            "sims_author_sent_retain": (Q[retain_rows] @ sent_author.T).astype("float32"),
            "forget_rows": np.asarray(forget_rows), "retain_rows": np.asarray(retain_rows),
            "forget_author_per_q": np.asarray(forget_authors_per_q),
            "retain_author_per_q": np.asarray([i // per for i in retain_rows]),
            "affected": np.asarray(sorted(int(j) for j in manifest["affected_adapters"])),
            "forget_authors": np.asarray(forget_authors),
        }
        if sent_name is not None:
            arrs["sims_name_sent_forget"] = (Q[forget_rows] @ sent_name.T).astype("float32")
            arrs["sims_name_sent_retain"] = (Q[retain_rows] @ sent_name.T).astype("float32")
        np.savez_compressed(dump_sims_path, encoder_source=np.str_(encoder_source),
                            stale_sha=np.str_(stale_hash or ""), **arrs)
        out["sims_dump"] = os.path.abspath(dump_sims_path)
    if "key" in routes:
        # Key routing must be deletion-invariant. Do NOT assume it: recompute from a freshly
        # reloaded assignment + rebuilt q2author and assert the measured shift is exactly 0.
        with open(lt.assignment_path(cfg)) as f:
            asg2 = json.load(f)
        rb = key_topk(asg2, lt.build_q2author(data_full, A, per), questions, k)
        sh = selection_shift(routes["key"][retain_rows], rb[retain_rows])
        if sh["shift_topk"] != 0.0 or sh["shift_top1"] != 0.0:
            raise AssertionError(f"key-policy selection shift must be zero, got {sh}")
        out["selection_shift"]["key"] = sh
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="legonet/ramole config (legonet mode)")
    ap.add_argument("--tag", default="forget10")
    ap.add_argument("--policies", nargs="+", default=["stale", "rebuilt", "key"],
                    choices=["stale", "rebuilt", "dropped", "abstain", "tombstone", "key"])
    ap.add_argument("--dump_sims", action="store_true",
                    help="write per-query similarity sidecar <out>.sims.npz (threshold-family "
                         "post-processing in analyze_router_leak.py — no more encoder reruns)")
    ap.add_argument("--centroid_mode", action="store_true",
                    help="R1 keystone: audit the k-shard MiniLM CENTROID router (the router "
                         "the R2 serving arms deploy) instead of the legonet expert index; "
                         "ignores --config/--policies, uses the --centroid_* flags")
    ap.add_argument("--centroid_k", type=int, default=10)
    ap.add_argument("--drop_shard", type=int, default=9)
    ap.add_argument("--router_encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--probe_manifest", default=None,
                    help="centroid mode: plant_manifest.json -> c_probe per tombstone rung")
    ap.add_argument("--no_holdout", action="store_true",
                    help="centroid mode: skip the holdout10 deletion-disclosure block")
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.centroid_mode:
        res = run_centroid_audit(args)
        rc.write_json(args.out, res)
        print(f"[routing_audit_tofu] centroid k={args.centroid_k} drop={args.drop_shard} -> {args.out}")
        print(f"  full acc retain={res['full']['acc_retain_top1']:.3f} "
              f"forget={res['full']['acc_forget_top1']:.3f}")
        print(f"  sibling retain_shift={res['sibling']['retain_shift_top1']:.3f} "
              f"sim_ratio={res['sibling']['mean_top1_sim_ratio']:.3f}")
        for rung in ("expert", "author", "name"):
            if rung in res["tombstone"]:
                d = res["tombstone"][rung]
                print(f"  tomb[{rung}]: catch={d['orphan_catch_rate']:.3f} "
                      f"leak={d['orphan_leak_rate']:.3f} "
                      f"retain_fpr={d['retain_false_tombstone_rate']:.3f}")
        if "disclosure" in res:
            print(f"  disclosure AUC (forget vs holdout) = "
                  f"{res['disclosure']['auc_forget_vs_holdout']:.3f}")
        return

    if not args.config:
        raise SystemExit("--config is required outside --centroid_mode")
    cfg = lt.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    data_full = load_dataset("locuslab/TOFU", "full")["train"]
    enc = SentenceTransformer(rt._encoder_source(cfg), device=args.device)   # loaded ONCE

    dump_path = (args.out.replace(".json", "") + ".sims.npz") if args.dump_sims else None
    res = run_audit(cfg, data_full, args.tag, args.policies, device=args.device, encoder=enc,
                    dump_sims_path=dump_path)
    res["config"] = os.path.abspath(args.config)
    rc.write_json(args.out, res)
    print(f"[routing_audit_tofu] tag={args.tag} policies={args.policies} -> {args.out}")
    for pol, m in res["policies"].items():
        print(f"  {pol:8s} orig_top1={m['orig_top1_rate']:.3f} "
              f"orig_topk={m['orig_topk_rate']:.3f} affected_mass={m['affected_mass']:.3f}")
    if "abstain" in res:
        for p, d in res["abstain"]["by_pct"].items():
            print(f"  abstain p{p}: orphan→base {d['orphan_abstain_rate']:.3f} "
                  f"retain false-abstain {d['retain_false_abstain_rate']:.3f} "
                  f"(τ={d['tau']:.3f})")
    if "tombstone" in res:
        for rung in ("expert", "author", "name"):
            if rung in res["tombstone"]:
                d = res["tombstone"][rung]
                print(f"  tomb[{rung}]: catch={d['orphan_catch_rate']:.3f} "
                      f"leak={d['orphan_leak_rate']:.3f} "
                      f"retain_fpr={d['retain_false_tombstone_rate']:.3f}")


if __name__ == "__main__":
    main()
