"""E3 post-deletion routing audit for the DBpedia RAMoLE arm — routing-only, NO LLM loaded.

WHY: legonet deletion RETRAINS the affected experts, so the served weights are exact; the
residual risk is the Stage-1 retriever INDEX: `lora_index_n{n}.npy` was built as member-mean
embeddings that INCLUDE the deleted records and is never rebuilt on deletion (staleness).
Per deletion tag this audit (a) builds the rebuilt retain-only index and saves it to the
EXACT filename `eval_ramole.py --index_policy rebuilt` expects
(`{run_dir}/lora_index_n{n}_ex{tag}.npy` — the stale file is hash-asserted untouched),
(b) routes the tag's forget records + retain records under stale vs rebuilt with
rank-preserving argsort, and (c) reports per-cluster index displacement.

Orphan results are PER-RECORD (d0-d2 delete 1 record each — a "rate" from n=1 is
meaningless); the pooled aggregate across all audited tags is the only rate reported.

Policies (`--policies`, default ['stale','rebuilt'] — the feature-frozen Jul-2 behavior,
output byte-identical; ported from tofu_sisa_lora/routing_audit_tofu.py for the H-DATASET
cell of the 2026-07-20 all-router sweep):
  stale/rebuilt — as before; rebuilt still writes the `_ex{tag}` index file and the stale
            index stays sha256-asserted untouched.
  dropped — §9-D drop-an-expert: the manifest's affected clusters are masked (−inf) out of
            the STALE index before ranking (serving semantics of a drop deletion; raises if
            surviving clusters < k). Per tag: `dropped` orphan rows + `dropped_extras`
            (top-1 concentration over survivors, entropy, masked/unmasked top-1 sim ratio —
            TOFU key names kept for the cross-arm analyzer) + retain
            `selection_shift_stale_vs_dropped`; per-record sim ratios pool across tags into
            `dropped_extras_pooled` (the only meaningful rate row for d0-d2).
  abstain — C1 threshold arm: τ calibrated on RETAIN top-1 sims over the FULL index
            (percentiles {1,5,10}; forget data never touches the threshold) + the reverse
            {90,99}% orphan-abstain operating points (retain false-abstain cost there);
            reported per tag AND pooled across tags.
`--dump_sims` writes a per-query similarity sidecar `<out>.sims.npz` (stale sims for the
forget records + each tag's retain sample, per-tag affected masks, record/tag ids) and never
mutates the aggregate JSON metrics (the routing_audit_tofu --dump_sims discipline).

Bitwise caveat (measured, not assumed): `build_lora_embeddings` draws member samples from
ONE shared RandomState across clusters, so excluding a record from its top-1 cluster c*
changes the draw count at c* and shifts the sample of every cluster AFTER c*. Bitwise
equality of untouched rows is therefore only guaranteed for clusters j < min(c*); the
exclude logic itself is verified exactly for ALL clusters via `_members_by_cluster`
member-set comparison. (d0's forget record has top-1 = cluster 31, the last — fully covered.)

    python routing_audit.py --config configs/ramole_l32_3b.json --tags d0 d1 d2 d_batch15 \
        --n_retain 200 --device cuda --out .../routing_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

import ramole_common as rc
import retriever as RET

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


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rank_topk(index: np.ndarray, Q: np.ndarray, k: int, mask_experts=None) -> np.ndarray:
    """(N, k) cluster ids in DESCENDING similarity order (rank-preserving; stable ties).
    `mask_experts` (dropped policy) removes those ids from the pool via a −inf similarity."""
    sims = np.asarray(Q, dtype="float32") @ np.asarray(index, dtype="float32").T
    if mask_experts is not None:
        sims[:, sorted(int(j) for j in mask_experts)] = -np.inf
    return np.argsort(-sims, axis=1, kind="stable")[:, :k]


def resolve_encoder_paths(cfg: dict) -> tuple[str, str]:
    """(encoder_source, stale_index_path), mirroring LoraRetriever.load exactly:
    cfg['retriever_run'] (ablation arms) borrows another run's fine-tuned retriever AND index."""
    paths = rc.Paths({**cfg, "name": cfg["retriever_run"]} if cfg.get("retriever_run") else cfg)
    enc_src = paths.retriever_dir if os.path.isdir(paths.retriever_dir) else cfg["encoder_model"]
    return enc_src, paths.lora_index_path


def _record_row(rid: str, routes: np.ndarray, ideal: list, manifest: dict) -> dict:
    aff, unt = set(manifest["affected_adapters"]), set(manifest["untouched_adapters"])
    s, K = set(int(j) for j in routes), set(int(j) for j in ideal)
    return {"topk": [int(j) for j in routes], "top1": int(routes[0]),
            "orig_topk": bool(s == K), "orig_top1": bool(int(routes[0]) in K),
            "jaccard": len(s & K) / len(s | K),
            "affected_mass": sum(1 for j in routes if int(j) in aff) / len(routes),
            "untouched_mass": sum(1 for j in routes if int(j) in unt) / len(routes)}


def selection_shift(ra: np.ndarray, rb: np.ndarray) -> dict:
    n = ra.shape[0]
    return {"n": int(n),
            "shift_topk": sum(set(map(int, x)) != set(map(int, y)) for x, y in zip(ra, rb)) / n,
            "shift_top1": sum(int(x[0]) != int(y[0]) for x, y in zip(ra, rb)) / n,
            "mean_jaccard": float(np.mean([len(set(map(int, x)) & set(map(int, y)))
                                           / len(set(map(int, x)) | set(map(int, y)))
                                           for x, y in zip(ra, rb)]))}


def dropped_extras(index: np.ndarray, Q_forget: np.ndarray, affected: list, k: int) -> dict:
    """Ported from tofu_sisa_lora/routing_audit_tofu.dropped_extras (key names kept identical
    so the cross-arm analyzer reads both audits): (a) how CONCENTRATED orphan top-1 mass is
    over the surviving clusters (few near-duplicate siblings ⇒ leak-prone; uniform spread ⇒
    generic/benign) and (b) the masked/unmasked top-1 cosine ratio (≈1 ⇒ the surviving
    sibling matches the query almost as well as the dropped cluster did). Adds
    per_record_top1_sim_ratio — this file's design keeps per-record values per record
    (d0-d2 are n=1 each); run_audit pools them across tags into dropped_extras_pooled."""
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
        "per_record_top1_sim_ratio": [float(x) for x in ratio],
    }


def _abstain_block(f_top1, r_top1, pcts=(1, 5, 10), catches=(0.90, 0.99)) -> dict:
    """τ family on masked-orphan vs full-index-retain top-1 sims. by_pct: τ calibrated on
    RETAIN percentiles — no forget data touches the threshold (the §5.2 centroid-leak
    discipline, ported from routing_audit_tofu.abstain_analysis). by_orphan_catch: the
    reverse operating point — τ at the {90,99}% orphan-abstain quantile, reporting the
    retain false-abstain cost of actually catching that many orphans."""
    f_top1 = np.asarray(f_top1, dtype="float64")
    r_top1 = np.asarray(r_top1, dtype="float64")
    out = {"n_orphans": int(f_top1.shape[0]), "n_retain": int(r_top1.shape[0]),
           "by_pct": {}, "by_orphan_catch": {}}
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


def abstain_analysis(index: np.ndarray, Q_forget: np.ndarray, Q_retain: np.ndarray,
                     affected: list, pcts=(1, 5, 10), catches=(0.90, 0.99)) -> dict:
    """Port of routing_audit_tofu.abstain_analysis (+ orphan-catch operating points): retain
    queries route over the FULL stale index (the "normal confidence" distribution that
    calibrates τ); forget orphans route over the DROPPED index (affected clusters masked)."""
    aff = sorted(int(j) for j in affected)
    r_sims = np.asarray(Q_retain, "float32") @ np.asarray(index, "float32").T
    r_top1 = r_sims.max(axis=1)
    f_sims = np.asarray(Q_forget, "float32") @ np.asarray(index, "float32").T
    f_masked = f_sims.copy()
    f_masked[:, aff] = -np.inf
    f_top1 = f_masked.max(axis=1)
    return _abstain_block(f_top1, r_top1, pcts=pcts, catches=catches)


def index_displacement(cfg, idx_stale, idx_reb, r2k, forget_ids) -> dict:
    """Per-cluster cos + the exclude-logic checks. Membership is by TOP-1 key
    (`_members_by_cluster`), so untouched = clusters no forget record has as top-1."""
    n = cfg["n"]
    top1 = sorted({int(r2k[rid][0]) for rid in forget_ids})
    untouched = [j for j in range(n) if j not in top1]
    # Exact exclude-logic check on member SETS for every cluster (RNG-independent).
    full = {j: {r["id"] for r in ms} for j, ms in RET._members_by_cluster(cfg).items()}
    excl = {j: {r["id"] for r in ms}
            for j, ms in RET._members_by_cluster(cfg, exclude_ids=set(forget_ids)).items()}
    for j in range(n):
        want = full.get(j, set()) - set(forget_ids)
        if excl.get(j, set()) != want:
            raise AssertionError(f"cluster {j}: exclude_ids member set wrong "
                                 f"(got {len(excl.get(j, set()))}, want {len(want)})")
    cos = [float(np.dot(idx_stale[j], idx_reb[j])
                 / (np.linalg.norm(idx_stale[j]) * np.linalg.norm(idx_reb[j]) + 1e-12))
           for j in range(n)]
    # Bitwise equality: the shared RandomState couples clusters — only j < min(top1) are
    # provably re-sampled identically (see module docstring). Assert those; report the rest.
    safe = [j for j in untouched if j < min(top1)]
    for j in safe:
        if not np.array_equal(idx_stale[j], idx_reb[j]):
            raise AssertionError(f"untouched cluster {j} (before first affected top-1 "
                                 f"cluster {min(top1)}) not bit-identical — exclude logic "
                                 "or encoder provenance broken")
    bit_equal = {str(j): bool(np.array_equal(idx_stale[j], idx_reb[j])) for j in untouched}
    return {"cos": cos, "forget_top1_clusters": top1, "untouched_by_top1": untouched,
            "bitwise_asserted": safe, "bit_equal_untouched": bit_equal,
            "cos_affected_top1": {str(j): cos[j] for j in top1}}


def audit_tag(cfg, tag, enc, stale_index_path, n_retain, device,
              policies=("stale", "rebuilt"), want_sims=False) -> dict | None:
    """One deletion tag; returns None (with a warning) if its manifest doesn't exist yet.
    `policies` selects the routing policies (the default keeps the frozen Jul-2 stale+rebuilt
    behavior with byte-identical output); `want_sims` stashes per-query similarity arrays
    under "_private" for the run-level sims sidecar / pooled abstain — run_audit pops that
    key, so it never reaches the JSON."""
    sp = rc.source_paths(cfg)
    mpath = os.path.join(sp.run_dir, "unlearn", tag, "manifest.json")
    if not os.path.isfile(mpath):
        print(f"[routing_audit] WARNING: skipping tag {tag!r} — no manifest at {mpath}", flush=True)
        return None
    with open(mpath) as f:
        manifest = json.load(f)
    with open(sp.assignment_path) as f:
        r2k = json.load(f)["record_to_keys"]
    forget_ids = list(manifest["forget_ids"])
    k = cfg["k"]
    affected = sorted(int(j) for j in manifest["affected_adapters"])
    if "dropped" in policies and cfg["n"] - len(affected) < k:
        raise RuntimeError(f"dropped policy needs >= k={k} surviving clusters, got "
                           f"{cfg['n'] - len(affected)} (n={cfg['n']}, affected={len(affected)})")

    stale_hash = _sha256(stale_index_path)
    idx_stale = np.load(stale_index_path)
    idx_reb = ex_path = None
    if "rebuilt" in policies:
        idx_reb = RET.build_lora_embeddings(cfg, encoder=enc, device=device,
                                            exclude_ids=set(forget_ids))
        # EXACT filename eval_ramole.py --index_policy rebuilt loads (its own run_dir, never the
        # retriever_run-indirected one); the stale index must never be overwritten.
        ex_path = os.path.join(rc.Paths(cfg).run_dir, f"lora_index_n{cfg['n']}_ex{tag}.npy")
        if os.path.realpath(ex_path) == os.path.realpath(stale_index_path):
            raise AssertionError(f"rebuilt index path collides with the stale index: {ex_path}")
        os.makedirs(os.path.dirname(ex_path), exist_ok=True)
        np.save(ex_path, idx_reb)
        if _sha256(stale_index_path) != stale_hash:
            raise AssertionError(f"stale index {stale_index_path} changed on disk — must never happen")

    recs = rc.load_records(sp.records_path)
    by_id = {r["id"]: r for r in recs}
    forget_recs = [by_id[rid] for rid in forget_ids]          # KeyError = corrupt manifest
    retain_recs = [r for r in recs if r["id"] not in set(forget_ids)][:n_retain]

    embed = rc.make_embed_fn(cfg["encoder_model"], instruction=cfg["instruction"],
                             device=device, encoder=enc)      # mirrors LoraRetriever.load
    Qf, Qr = embed([rc.route_text(r) for r in forget_recs]), embed([rc.route_text(r) for r in retain_recs])
    rf, rr = {}, {}
    # dropped needs the stale retain routes for its selection shift even without 'stale'
    if {"stale", "dropped"} & set(policies):
        rf["stale"] = rank_topk(idx_stale, Qf, k)
        rr["stale"] = rank_topk(idx_stale, Qr, k)
    if "rebuilt" in policies:
        rf["rebuilt"] = rank_topk(idx_reb, Qf, k)
        rr["rebuilt"] = rank_topk(idx_reb, Qr, k)
    if "dropped" in policies:
        rf["dropped"] = rank_topk(idx_stale, Qf, k, mask_experts=affected)
        rr["dropped"] = rank_topk(idx_stale, Qr, k, mask_experts=affected)

    row_pols = [p for p in ("stale", "rebuilt", "dropped") if p in policies]
    rows = [{"id": rid, "ideal": [int(j) for j in r2k[rid]],
             **{pol: _record_row(rid, rf[pol][i], r2k[rid], manifest) for pol in row_pols}}
            for i, rid in enumerate(forget_ids)]
    out = {"tag": tag, "manifest": mpath}
    if ex_path is not None:
        out["rebuilt_index_path"] = ex_path
    out["forget_ids"] = forget_ids
    out["affected_adapters"] = manifest["affected_adapters"]
    out["orphan_records"] = rows
    if "stale" in policies and "rebuilt" in policies:
        out["selection_shift"] = selection_shift(rr["stale"], rr["rebuilt"])
    if idx_reb is not None:
        out["index_displacement"] = index_displacement(cfg, idx_stale, idx_reb, r2k, forget_ids)
    if "dropped" in policies:
        out["dropped_extras"] = dropped_extras(idx_stale, Qf, affected, k)
        out["selection_shift_stale_vs_dropped"] = selection_shift(rr["stale"], rr["dropped"])
    if "abstain" in policies or want_sims:
        sims_f = np.asarray(Qf, "float32") @ np.asarray(idx_stale, "float32").T
        sims_r = np.asarray(Qr, "float32") @ np.asarray(idx_stale, "float32").T
        f_masked = sims_f.copy()
        f_masked[:, affected] = -np.inf
        f_top1_masked, r_top1 = f_masked.max(axis=1), sims_r.max(axis=1)
        if "abstain" in policies:
            out["abstain"] = _abstain_block(f_top1_masked, r_top1)
        private = {"f_top1_masked": f_top1_masked, "r_top1": r_top1}
        if want_sims:
            private.update({
                "sims_stale_forget": sims_f.astype("float32"),
                "sims_stale_retain": sims_r.astype("float32"),
                "retain_ids": [r["id"] for r in retain_recs],
                "affected_row": np.asarray([1 if j in set(affected) else 0
                                            for j in range(cfg["n"])], dtype="uint8"),
            })
        out["_private"] = private
    return out


def pool_orphans(tag_results: list) -> dict:
    """The only place rates are computed: pooled across tags (n=1 per tag for d0-d2).
    Policies are read off the rows so non-default --policies runs pool whatever is present
    (order fixed stale/rebuilt/dropped — the default pair keeps its historical output)."""
    rows = [r for t in tag_results for r in t["orphan_records"]]
    out = {"n_records": len(rows)}
    pols = [p for p in ("stale", "rebuilt", "dropped") if rows and p in rows[0]]
    for pol in pols:
        out[pol] = {
            "orig_topk_rate": float(np.mean([r[pol]["orig_topk"] for r in rows])),
            "orig_top1_rate": float(np.mean([r[pol]["orig_top1"] for r in rows])),
            "sibling_top1_rate": float(np.mean([not r[pol]["orig_top1"] for r in rows])),
            "mean_jaccard": float(np.mean([r[pol]["jaccard"] for r in rows])),
            "affected_mass": float(np.mean([r[pol]["affected_mass"] for r in rows])),
            "untouched_mass": float(np.mean([r[pol]["untouched_mass"] for r in rows])),
        }
    return out


def run_audit(cfg: dict, tags: list, n_retain: int, device: str = "cpu", encoder=None,
              policies=("stale", "rebuilt"), dump_sims_path: str = None) -> dict:
    rc.set_determinism(cfg["base_seed"])
    enc_src, stale_path = resolve_encoder_paths(cfg)
    if encoder is None:   # loaded ONCE, shared across every tag (tests may inject one)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(enc_src, device=device)
    enc = encoder
    policies = list(policies)
    results, skipped, privates = [], [], {}
    for tag in tags:
        r = audit_tag(cfg, tag, enc, stale_path, n_retain, device,
                      policies=policies, want_sims=dump_sims_path is not None)
        if r is None:
            skipped.append(tag)
            continue
        priv = r.pop("_private", None)   # sidecar/pooling arrays never reach the JSON
        if priv is not None:
            privates[r["tag"]] = priv
        results.append(r)
    if not results:
        raise RuntimeError(f"no auditable tags (all skipped: {skipped})")
    out = {"encoder_source": enc_src, "stale_index_path": stale_path,
           "n": cfg["n"], "k": cfg["k"], "n_retain": n_retain,
           "tags": {r["tag"]: r for r in results}, "skipped_tags": skipped,
           "orphan_pooled": pool_orphans(results)}
    if "dropped" in policies:
        # the only meaningful ratio row for d0-d2 (n=1 each): pooled across tags
        ratios = np.concatenate([np.asarray(r["dropped_extras"]["per_record_top1_sim_ratio"],
                                            dtype="float64") for r in results])
        out["dropped_extras_pooled"] = {
            "n_orphans": int(ratios.shape[0]),
            "mean_top1_sim_ratio": float(ratios.mean()),
            "p10_top1_sim_ratio": float(np.percentile(ratios, 10)),
            "p90_top1_sim_ratio": float(np.percentile(ratios, 90)),
        }
    if "abstain" in policies:
        f_all = np.concatenate([privates[r["tag"]]["f_top1_masked"] for r in results])
        r_all = np.concatenate([privates[r["tag"]]["r_top1"] for r in results])
        out["abstain_pooled"] = _abstain_block(f_all, r_all)
    if dump_sims_path:
        # Per-query similarity sidecar (CPU threshold post-processing lives downstream —
        # the routing_audit_tofu --dump_sims discipline). Never mutates the aggregate metrics.
        tags_order = [r["tag"] for r in results]
        arrs = {
            "sims_stale_forget": np.concatenate(
                [privates[t]["sims_stale_forget"] for t in tags_order], axis=0),
            "sims_stale_retain": np.concatenate(
                [privates[t]["sims_stale_retain"] for t in tags_order], axis=0),
            "forget_ids": np.asarray([rid for r in results for rid in r["forget_ids"]]),
            "retain_ids": np.asarray([rid for t in tags_order
                                      for rid in privates[t]["retain_ids"]]),
            "forget_tag_idx": np.asarray([ti for ti, r in enumerate(results)
                                          for _ in r["forget_ids"]], dtype="int32"),
            "retain_tag_idx": np.asarray([ti for ti, t in enumerate(tags_order)
                                          for _ in privates[t]["retain_ids"]], dtype="int32"),
            "tags": np.asarray(tags_order),
            "affected_mask": np.stack([privates[t]["affected_row"] for t in tags_order], axis=0),
        }
        os.makedirs(os.path.dirname(os.path.abspath(dump_sims_path)), exist_ok=True)
        np.savez_compressed(dump_sims_path, encoder_source=np.str_(enc_src),
                            stale_sha=np.str_(_sha256(stale_path)), **arrs)
        out["sims_dump"] = os.path.abspath(dump_sims_path)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--tags", nargs="+", default=["d0", "d1", "d2"])
    ap.add_argument("--policies", nargs="+", default=["stale", "rebuilt"],
                    choices=["stale", "rebuilt", "dropped", "abstain"],
                    help="dropped = §9-D drop-an-expert (affected clusters masked to -inf "
                         "before ranking); abstain = retain-percentile τ + orphan-catch "
                         "operating points; default = the frozen stale+rebuilt behavior")
    ap.add_argument("--dump_sims", action="store_true",
                    help="write per-query similarity sidecar <out>.sims.npz (stale sims + "
                         "affected masks + record/tag ids; aggregate JSON metrics unchanged)")
    ap.add_argument("--n_retain", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = rc.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    dump_path = (args.out.replace(".json", "") + ".sims.npz") if args.dump_sims else None
    res = run_audit(cfg, args.tags, args.n_retain, device=args.device,
                    policies=args.policies, dump_sims_path=dump_path)
    res["config"] = os.path.abspath(args.config)
    rc.write_json(args.out, res)
    print(f"[routing_audit] tags={list(res['tags'])} skipped={res['skipped_tags']} -> {args.out}")
    print(json.dumps(res["orphan_pooled"], indent=2))
    if "dropped_extras_pooled" in res:
        print(f"[routing_audit] dropped pooled: {json.dumps(res['dropped_extras_pooled'])}")
    if "abstain_pooled" in res:
        for p, d in res["abstain_pooled"]["by_pct"].items():
            print(f"  abstain p{p}: orphan→base {d['orphan_abstain_rate']:.3f} "
                  f"retain false-abstain {d['retain_false_abstain_rate']:.3f} (τ={d['tau']:.3f})")
        for c, d in res["abstain_pooled"]["by_orphan_catch"].items():
            print(f"  abstain catch {c}: retain false-abstain "
                  f"{d['retain_false_abstain_rate']:.3f} (τ={d['tau']:.3f})")


if __name__ == "__main__":
    main()
