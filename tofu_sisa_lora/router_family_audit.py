"""Score-matrix leakage audit over the router.py strategy family (router_leak J1-J3).

Pre-registration: log/router_leak/2026-07-20_all-router-sweep-preregistration.md.
For every requested routing strategy this computes the FULL-POOL per-query score matrix
scores[n_q, k] (higher = more likely routed there), derives per-drop-set leak cells
(orphan capture over survivors, family-specific sibling-adequacy ratio, retain top-1
shift), and dumps one npz sidecar per strategy for the CPU analyzer
(analyze_router_family.py) per THE FAMILY NPZ CONTRACT:

  <out_json_stem>.<strategy>.npz
    scores            float32 [n_q, k]   full-pool scores (ppl = NEGATIVE question loss;
                                         norm/div = raw scores; centroid/tfidf = raw cos)
    scores__d<ids>    float32 [n_q, k]   logit_div ONLY: per-drop-set recomputed scores
                                         (survivor candidate set; dropped columns = NaN)
    match             uint8   [n_q, k]   key_exact ONLY (name-match incidence; no scores)
    is_forget         bool    [n_q]      forget10-author membership
    author_of_q       int32   [n_q]
    k                 int scalar; strategy str; drop_sets JSON str (list of lists)
    author_sent_scores float32 [n_q, n_sent] + sent_author_ids int32
                      question-only per-forget-author sentinel cosines in the SAME feature
                      space (mirrors routing_audit_tofu._author_sentinels; feature-space
                      strategies only — absent for behavioral and key_exact)

Design notes implemented from the pre-registration:
  (i)   centroid_sbert = router.build_centroids Q+A centroids (router.py semantics);
        centroid_sbert_q = eval_routed_scaffold.build_shard_centroids question-only
        centroids (the SERVING builder — the continuity-anchor row vs rl_centroid_k10).
  (ii)  behavioral score matrices are per-shard-batched with PER-SAMPLE scores; the
        per-sample lora_B norm (pad positions zeroed before the norm) reduces to
        router._lora_b_norm's batch-summed scalar at bs=1 (test_router_family gate).
  (iii) logit_div's divergence-from-candidate-mean changes with the candidate set: the
        post-drop matrix is RECOMPUTED over survivors (scores__d*), never column-masked.
  (iv)  key_exact has no graded score — it ships the binary match matrix and the
        no-match operating point (orphan/retain no-match rates post-drop) instead.
  (v)   oracle control is analytic (orphans -> base P=1.0, retain shift 0), emitted as a
        constant "by_construction" block — no run.

Faithfulness gate: --self_check N asserts, for N seeded queries, that the score-row
argmax equals the actual router.route() result on the full pool (key_exact: the routed
shard matches). logit_div is INCLUDED: its batched fp path is not guaranteed bit-equal
to the single-query serving path, so exact argmax agreement is verified, not assumed.

  python router_family_audit.py --pool_dir <shards_dir> --base_model <scaffolded_base> \
      --k 10 --strategies key_exact key_tfidf centroid_sbert centroid_sbert_q \
      centroid_lm centroid_lm_last --drop_sets "9;9,8;9,8,7,6" --queries all \
      --device cuda --dump_sims --out .../rl_family_k10_feature.json

--stub: tiny random Llama + synthetic 8-author/4-questions pool, stub tokenizer and
hash-based embeddings — no HF hub, CPU-only. Exercised by test_router_family.py.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os

import numpy as np

import router as R

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

FEATURE_SPACE = ("key_tfidf", "centroid_sbert", "centroid_sbert_q",
                 "centroid_lm", "centroid_lm_last")
BEHAVIORAL = ("ppl", "activation_norm", "attn_norm", "logit_div")
ALL_STRATEGIES = ("key_exact",) + FEATURE_SPACE + BEHAVIORAL
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
STUB_VOCAB = 64


# ── pure helpers (unit-tested by test_router_family.py) ───────────────────────

def parse_drop_sets(spec: str) -> list:
    """'9;9,8;9,8,7,6' -> [[9], [9, 8], [9, 8, 7, 6]] (order preserved)."""
    out = [[int(x) for x in grp.split(",")] for grp in spec.split(";") if grp.strip()]
    if not out:
        raise ValueError(f"empty --drop_sets spec: {spec!r}")
    return out


def cell_key(ids: list) -> str:
    return "d" + "_".join(str(int(i)) for i in ids)


def strategy_family(strategy: str) -> str | None:
    """Adequacy-ratio family: 'cos' (similarity), 'ppl' (loss), 'norm' (activation /
    divergence magnitudes), None (key_exact — binary, no graded score)."""
    if strategy in ("key_tfidf", "centroid_sbert", "centroid_sbert_q",
                    "centroid_lm", "centroid_lm_last"):
        return "cos"
    if strategy == "ppl":
        return "ppl"
    if strategy in ("activation_norm", "attn_norm", "logit_div"):
        return "norm"
    return None


def sample_query_rows(num_authors: int = 200, per: int = 20, n_retain: int = 400,
                      forget_authors=tuple(range(180, 200))) -> tuple:
    """Replicates analyze_router_tofu._records' row selection VERBATIM: all forget-author
    rows + a RandomState(42) n_retain-row retain sample (choice on the pool LENGTH then
    index, sorted ints — byte-for-byte the same draw)."""
    forget = set(int(a) for a in forget_authors)
    forget_rows = [a * per + i for a in sorted(forget) for i in range(per)]
    retain_pool = np.array([r for r in range(num_authors * per)
                            if (r // per) not in forget], dtype=int)
    rng = np.random.RandomState(42)
    retain_rows = sorted(int(r) for r in
                         retain_pool[rng.choice(len(retain_pool), size=n_retain,
                                                replace=False)])
    return forget_rows, retain_rows


def masked_top1(scores: np.ndarray, survivors: list) -> np.ndarray:
    """Argmax restricted to survivor columns, returned as SHARD ids. Raises when no
    survivor remains (a drop set must never silently route into the void)."""
    survivors = [int(j) for j in survivors]
    if len(survivors) < 1:
        raise ValueError("masked_top1: survivor set is empty (all shards dropped)")
    sub = np.asarray(scores)[:, survivors]
    return np.asarray(survivors, dtype=int)[np.argmax(sub, axis=1)]


def orphan_capture(top1_ids: np.ndarray, survivors: list, k: int) -> dict:
    """Concentration of orphan post-drop top-1 routes over the surviving shards — mirrors
    routing_audit_tofu.dropped_extras (shares / normalized entropy / per-shard hist)."""
    counts = np.bincount(np.asarray(top1_ids, dtype=int), minlength=k).astype("float64")
    total = counts.sum()
    if total == 0:
        return {"n": 0}
    p = counts[counts > 0] / total
    ent = float(-(p * np.log(p)).sum() / np.log(max(len(survivors), 2)))
    ranked = np.sort(counts)[::-1] / total
    return {"n": int(total),
            "top1_share_top1_expert": float(ranked[0]),
            "top1_share_top3_experts": float(ranked[:3].sum()),
            "top1_entropy_norm": ent,
            "top1_hist": {str(int(j)): int(counts[j]) for j in np.nonzero(counts)[0]}}


def adequacy_ratios(family: str, unmasked_top1: np.ndarray,
                    masked_top1_scores: np.ndarray) -> tuple:
    """Family-specific sibling-adequacy per orphan row (1.0 = the surviving sibling scores
    the orphan as well as the deleted expert did):
      cos  — masked/unmasked top-1 cos          (pre-reg: cosine routers)
      ppl  — unmasked_top1_loss/masked_top1_loss (scores are NEGATIVE losses)
      norm — masked/unmasked top-1 score         (activation/attention norm, logit_div)"""
    u = np.asarray(unmasked_top1, dtype="float64")
    m = np.asarray(masked_top1_scores, dtype="float64")
    if family == "ppl":
        loss_u, loss_m = -u, -m           # scores store negative question loss
        ratio = loss_u / np.maximum(loss_m, 1e-12)
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


def key_exact_routes(match: np.ndarray, candidates: list) -> tuple:
    """(routes, no_match) under KeyRouter.route serving semantics: first candidate in the
    given (ascending-id) order whose names match; no match -> candidates[0] fallback with
    the no_match flag set (the identity signal key_exact ships instead of a score)."""
    candidates = [int(j) for j in candidates]
    if len(candidates) < 1:
        raise ValueError("key_exact_routes: empty candidate set")
    m = np.asarray(match, dtype=bool)[:, candidates]
    has = m.any(axis=1)
    first = np.argmax(m, axis=1)               # index of the FIRST True per row
    routes = np.full(m.shape[0], candidates[0], dtype=int)
    routes[has] = np.asarray(candidates, dtype=int)[first[has]]
    return routes, ~has


def aggregate_strategy_cells(strategy: str, k: int, shard_of_q: np.ndarray,
                             drop_sets: list, scores: np.ndarray = None,
                             cell_scores: dict = None, match: np.ndarray = None) -> dict:
    """Per drop-set leak cell. Orphans = queries of the dropped shards' authors; retain =
    everything else. logit_div passes cell_scores (recomputed survivor-set matrices,
    design note iii); key_exact passes match (note iv); everything else column-masks the
    full-pool matrix (serving semantics: exclude= only removes candidates)."""
    family = strategy_family(strategy)
    out = {}
    if match is not None:
        routes_full, nomatch_full = key_exact_routes(match, list(range(k)))
        top1_full = routes_full
    else:
        top1_full = masked_top1(scores, list(range(k)))
    for ids in drop_sets:
        ck = cell_key(ids)
        dropped = sorted(set(int(i) for i in ids))
        surv = [j for j in range(k) if j not in set(dropped)]
        if len(surv) < 1:
            raise ValueError(f"drop set {dropped} leaves no survivors (k={k})")
        orphan = np.isin(shard_of_q, dropped)
        retain = ~orphan
        cell = {"dropped_shards": dropped, "n_survivors": len(surv),
                "n_orphans": int(orphan.sum()), "n_retain": int(retain.sum())}
        if match is not None:
            routes_post, nomatch_post = key_exact_routes(match, surv)
            if orphan.any():
                cell["orphan_capture"] = orphan_capture(routes_post[orphan], surv, k)
            cell["retain_shift_top1"] = (
                float((routes_post[retain] != routes_full[retain]).mean())
                if retain.any() else None)
            cell["no_match"] = {
                "orphan_no_match_rate_postdrop": (float(nomatch_post[orphan].mean())
                                                  if orphan.any() else None),
                "retain_no_match_rate_postdrop": (float(nomatch_post[retain].mean())
                                                  if retain.any() else None),
                "retain_no_match_rate_full": (float(nomatch_full[retain].mean())
                                              if retain.any() else None),
                "fallback_shard": int(surv[0]),
            }
        else:
            post = cell_scores[ck] if cell_scores is not None else scores
            top1_post = masked_top1(post, surv)
            if orphan.any():
                cell["orphan_capture"] = orphan_capture(top1_post[orphan], surv, k)
                unm = np.asarray(scores)[orphan].max(axis=1)
                msk = np.asarray(post)[orphan][:, surv].max(axis=1)
                ratio, definition = adequacy_ratios(family, unm, msk)
                cell["adequacy"] = {"definition": definition,
                                    "mean": float(ratio.mean()),
                                    "p10": float(np.percentile(ratio, 10)),
                                    "p90": float(np.percentile(ratio, 90))}
            cell["retain_shift_top1"] = (
                float((top1_post[retain] != top1_full[retain]).mean())
                if retain.any() else None)
        out[ck] = cell
    return out


def cosine_scores(Q: np.ndarray, cents: list) -> np.ndarray:
    """CentroidRouter's exact score: dot(q, c) / (|q|*|c| + 1e-12), vectorized."""
    Q = np.asarray(Q, dtype="float32")
    C = np.stack([np.asarray(c, dtype="float32") for c in cents])
    qn = np.linalg.norm(Q, axis=1, keepdims=True)
    cn = np.linalg.norm(C, axis=1)[None, :]
    return ((Q @ C.T) / (qn * cn + 1e-12)).astype("float32")


def build_author_sentinels(embed_batch_fn, dataset, author_ids: list, per: int) -> np.ndarray:
    """Question-only per-author sentinel centroids in the strategy's own feature space:
    L2-normed mean of each author's `per` question vectors (mirrors
    routing_audit_tofu._author_sentinels, re-embedded here because sample-mode query
    passes don't cover every author question)."""
    rows = []
    for a in author_ids:
        qs = [dataset[a * per + w]["question"] for w in range(per)]
        v = np.asarray(embed_batch_fn(qs), dtype="float32").mean(axis=0)
        rows.append(v / (np.linalg.norm(v) + 1e-12))
    return np.stack(rows).astype("float32")


def write_family_npz(path: str, strategy: str, k: int, drop_sets: list,
                     is_forget: np.ndarray, author_of_q: np.ndarray,
                     scores: np.ndarray = None, match: np.ndarray = None,
                     cell_scores: dict = None, sent_scores: np.ndarray = None,
                     sent_author_ids: list = None) -> None:
    """THE FAMILY NPZ CONTRACT writer (see module docstring)."""
    arrs = {"is_forget": np.asarray(is_forget, dtype=bool),
            "author_of_q": np.asarray(author_of_q, dtype="int32"),
            "k": np.int64(k), "strategy": np.str_(strategy),
            "drop_sets": np.str_(json.dumps([[int(i) for i in d] for d in drop_sets]))}
    if scores is not None:
        arrs["scores"] = np.asarray(scores, dtype="float32")
    if match is not None:
        arrs["match"] = np.asarray(match, dtype="uint8")
    if cell_scores:
        for ck, m in cell_scores.items():
            arrs[f"scores__{ck}"] = np.asarray(m, dtype="float32")
    if sent_scores is not None:
        arrs["author_sent_scores"] = np.asarray(sent_scores, dtype="float32")
        arrs["sent_author_ids"] = np.asarray(sent_author_ids, dtype="int32")
    np.savez_compressed(path, **arrs)


# ── behavioral scoring (per-sample versions of the router.py signals) ─────────

def per_sample_ce(logits, input_ids, attention_mask) -> np.ndarray:
    """Per-sample mean CE with labels=input_ids and pad label positions masked — the
    per-sample form of PplRouter's model(**enc, labels=input_ids).loss (HF upcasts logits
    to float32 and means over the T-1 shifted positions; at bs=1/no padding this is the
    identical mean)."""
    import torch.nn.functional as F
    lg = logits.float()
    shift_logits = lg[:, :-1, :]
    labels = input_ids[:, 1:]
    valid = attention_mask[:, 1:].to(shift_logits.dtype)
    ce = F.cross_entropy(shift_logits.transpose(1, 2), labels, reduction="none") * valid
    return (ce.sum(1) / valid.sum(1).clamp(min=1.0)).detach().cpu().numpy()


def _register_persample_hooks(model, adapter_name: str, holder: dict) -> tuple:
    """Forward hooks on every lora_B[adapter_name] capturing PER-SAMPLE output norms with
    pad positions zeroed (holder['mask'] is set per batch). Returns (handles,
    {module_name: is_attention}) — the attn filter replicates router._lora_b_norm's
    `any(p in name)` check so attn_norm and activation_norm share one forward."""
    import torch
    handles, is_attn = [], {}
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_B") and adapter_name in module.lora_B):
            continue
        is_attn[name] = any(p in name for p in R.ActivationRouter._ATTN_NAMES)

        def _make(nm):
            def _hook(m, inp, out):
                o = out.detach().float()
                o = o * holder["mask"].to(o.dtype).unsqueeze(-1)   # zero pad positions
                holder["norms"][nm] = torch.linalg.vector_norm(o, dim=(1, 2))
            return _hook

        handles.append(module.lora_B[adapter_name].register_forward_hook(_make(name)))
    return handles, is_attn


def lora_b_norms_batch(model, adapter_name: str, input_ids, attention_mask=None,
                       attn_only: bool = False) -> np.ndarray:
    """Per-sample router._lora_b_norm: for each sample, sum over lora_B modules of the L2
    norm of that module's output with pad positions zeroed. At bs=1 (no padding) this
    reduces to _lora_b_norm's scalar (test_router_family gate)."""
    import torch
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    holder = {"mask": attention_mask, "norms": {}}
    # Activate BEFORE registering hooks: under a lazy adapter cache the adapter's lora_B
    # modules do not exist until set_adapter loads them, and _register_persample_hooks
    # indexes lora_B[adapter_name] directly. Order is otherwise irrelevant — a forward hook
    # fires at forward time — so this matches score_norm_ppl_family's order and is numerically
    # identical to the eager path.
    model.set_adapter(adapter_name)
    handles, is_attn = _register_persample_hooks(model, adapter_name, holder)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    tot = torch.zeros(input_ids.shape[0])
    for nm, v in holder["norms"].items():
        if attn_only and not is_attn[nm]:
            continue
        tot += v.cpu()
    return tot.numpy()


def score_norm_ppl_family(model, tokenizer, questions: list, k: int, wants: list,
                          bs: int, device: str) -> dict:
    """ppl / activation_norm / attn_norm score matrices in ONE per-shard-batched pass
    (set adapter once per shard, batch all queries; one forward serves CE and both norm
    families). ppl scores are NEGATIVE per-sample losses (higher = more familiar)."""
    import torch
    n_q = len(questions)
    out = {w: np.zeros((n_q, k), dtype="float32") for w in wants}
    need_norms = ("activation_norm" in wants) or ("attn_norm" in wants)
    model.eval()
    for shard in range(k):
        aname = f"shard_{shard}"
        model.set_adapter(aname)
        holder = {"mask": None, "norms": {}}
        handles, is_attn = (_register_persample_hooks(model, aname, holder)
                            if need_norms else ([], {}))
        try:
            for lo in range(0, n_q, bs):
                batch = questions[lo:lo + bs]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=256)
                enc = {kk: v.to(device) for kk, v in enc.items()}
                holder["mask"] = enc["attention_mask"]
                holder["norms"] = {}
                with torch.no_grad():
                    o = model(input_ids=enc["input_ids"],
                              attention_mask=enc["attention_mask"])
                if "ppl" in wants:
                    loss = per_sample_ce(o.logits, enc["input_ids"],
                                         enc["attention_mask"])
                    out["ppl"][lo:lo + len(batch), shard] = -loss
                if need_norms:
                    tot = torch.zeros(len(batch))
                    attn = torch.zeros(len(batch))
                    for nm, v in holder["norms"].items():
                        v = v.cpu()
                        tot += v
                        if is_attn[nm]:
                            attn += v
                    if "activation_norm" in wants:
                        out["activation_norm"][lo:lo + len(batch), shard] = tot.numpy()
                    if "attn_norm" in wants:
                        out["attn_norm"][lo:lo + len(batch), shard] = attn.numpy()
        finally:
            for h in handles:
                h.remove()
    return out


def logit_div_from_cached(cached: dict, candidates: list):
    """Per-sample Frobenius norm of (logits_j − mean over the CANDIDATE set), for every
    j in candidates. The mean depends on the candidate set — a drop set must be
    recomputed through here, never column-masked from the full-set result (note iii)."""
    import torch
    candidates = [int(j) for j in candidates]
    if len(candidates) < 1:
        raise ValueError("logit_div_from_cached: empty candidate set")
    mean = None
    for j in candidates:
        mean = cached[j].clone() if mean is None else mean + cached[j]
    mean = mean / float(len(candidates))
    out = {}
    for j in candidates:
        out[j] = torch.linalg.vector_norm(cached[j] - mean, dim=(1, 2)).cpu().numpy()
    return out


def score_logit_div(model, tokenizer, questions: list, k: int, drop_sets: list,
                    bs: int, device: str) -> tuple:
    """(full_scores, {cell_key: recomputed matrix with NaN at dropped columns}). Logits
    are cached per query-batch across shards (k forwards per batch, not k^2), cast to
    float32 with pad positions zeroed, and freed after each batch."""
    import torch
    n_q = len(questions)
    cand_sets = {"full": list(range(k))}
    for ids in drop_sets:
        surv = [j for j in range(k) if j not in set(int(i) for i in ids)]
        if not surv:
            raise ValueError(f"logit_div drop set {ids} leaves no survivors")
        cand_sets[cell_key(ids)] = surv
    outs = {name: np.full((n_q, k), np.nan, dtype="float32") for name in cand_sets}
    model.eval()
    for lo in range(0, n_q, bs):
        batch = questions[lo:lo + bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=256)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        mask = enc["attention_mask"]
        cached = {}
        with torch.no_grad():
            for shard in range(k):
                model.set_adapter(f"shard_{shard}")
                lg = model(input_ids=enc["input_ids"], attention_mask=mask).logits.float()
                cached[shard] = lg * mask.to(lg.dtype).unsqueeze(-1)
        for name, cand in cand_sets.items():
            per_shard = logit_div_from_cached(cached, cand)
            for j, v in per_shard.items():
                outs[name][lo:lo + len(batch), j] = v
        del cached
    full = outs.pop("full")
    return full, outs


# ── LM feature space (batched twin of router.make_lm_embed_fn) ────────────────

class _NoAdapterLM:
    """Duck-types the PeftModel surface make_lm_embed_fn/lm_embed_texts need for a PLAIN
    base model: with no adapters loaded, 'adapters disabled' is a no-op, so base hidden
    states are exactly the adapters-disabled embedding router.py specifies."""

    def __init__(self, m):
        self._m = m

    def parameters(self):
        return self._m.parameters()

    def __call__(self, **kw):
        return self._m(**kw)

    def disable_adapter(self):
        return contextlib.nullcontext()


def lm_embed_texts(model_ctx, tokenizer, texts: list, mode: str, bs: int,
                   device: str) -> np.ndarray:
    """Batched make_lm_embed_fn: same truncation (max_length=256), same non-pad pooling
    ('mean' over real tokens / 'last' real token), adapters disabled — RIGHT padding so
    real-token positions match the single-query path (no generation happens here)."""
    import torch
    vecs = []
    for lo in range(0, len(texts), bs):
        enc = tokenizer(texts[lo:lo + bs], return_tensors="pt", padding=True,
                        truncation=True, max_length=256)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        with torch.no_grad():
            with model_ctx.disable_adapter():
                out = model_ctx(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        mask = enc["attention_mask"].bool()
        for i in range(hidden.shape[0]):
            h = hidden[i][mask[i]]
            v = h.mean(0) if mode == "mean" else h[-1]
            vecs.append(v.float().cpu().numpy())
    return np.stack(vecs)


def build_centroids_cached(embed_batch_fn, texts_per_shard: list, cache_dir: str,
                           embed_label: str) -> list:
    """Batched sibling of router.build_centroids with the IDENTICAL cache layout
    ({cache_dir}/{embed_label}/shard_{i}.npy) and mean math. embed_label must be a NEW
    rfa_* dir — existing centroid caches are never touched."""
    cents = []
    for sid, texts in enumerate(texts_per_shard):
        cp = None
        if cache_dir is not None:
            cp = os.path.join(cache_dir, embed_label, f"shard_{sid}.npy")
            if os.path.exists(cp):
                cents.append(np.load(cp))
                continue
        c = np.asarray(embed_batch_fn(texts)).mean(axis=0)
        if cp is not None:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            np.save(cp, c)
        cents.append(c)
    return cents


# ── stub providers (CPU, no HF hub — exercised by test_router_family.py) ──────

STUB_NAMES = ["Aldous Prine", "Bekka Vole", "Cormac Dale", "Dessa Quill",
              "Evan Marsh", "Fiora Nett", "Gwen Tarr", "Hollis Vane"]


class StubDataset:
    """Minimal HF-dataset surface (len/getitem/iter/select) for the stub pool."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    def __iter__(self):
        return iter(self._rows)

    def select(self, idxs):
        return StubDataset([self._rows[i] for i in idxs])


class StubTokenizer:
    """Char-hash tokenizer (right padding, pad id 0) covering the audit's tokenizer
    contract: single-string and list calls, truncation/max_length, dict of tensors."""
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, texts, return_tensors="pt", truncation=True, max_length=256,
                 padding=False):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        ids = [[(ord(c) % (STUB_VOCAB - 1)) + 1 for c in t][:max_length] for t in texts]
        T = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), T, dtype=torch.long)
        mask = torch.zeros(len(ids), T, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def stub_embed_texts(texts, dim: int = 32) -> np.ndarray:
    """Deterministic word-hash embedding (L2-normed sum of per-word seeded gaussians) —
    author-name words give the stub pool per-author structure without any encoder."""
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().split():
            h = int(hashlib.blake2s(w.encode(), digest_size=4).hexdigest(), 16)
            out[i] += np.random.RandomState(h % (2 ** 32)).randn(dim).astype("float32")
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


def build_stub_dataset(num_authors: int, per: int) -> StubDataset:
    rows = []
    for a in range(num_authors):
        name = STUB_NAMES[a % len(STUB_NAMES)]
        for w in range(per):
            rows.append({"question": f"What themes does {name} explore in book {w}?",
                         "answer": f"{name} writes about motif {a}-{w} and topic {a}."})
    return StubDataset(rows)


def build_stub_lm(k: int, seed: int = 42):
    """Tiny random Llama + k in-memory shard adapters (seeded factors). up_proj is
    targeted alongside q/v_proj so attn_norm genuinely differs from activation_norm."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4,
                      vocab_size=STUB_VOCAB, max_position_embeddings=512)
    base = LlamaForCausalLM(cfg)
    lcfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj", "up_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg, adapter_name="shard_0")
    for i in range(1, k):
        model.add_adapter(f"shard_{i}", lcfg)
    gen = torch.Generator().manual_seed(7)
    for _, m in model.named_modules():
        if hasattr(m, "lora_A") and "shard_0" in m.lora_A:
            for i in range(k):
                for fac in (m.lora_A[f"shard_{i}"].weight, m.lora_B[f"shard_{i}"].weight):
                    fac.data.normal_(0.0, 0.2, generator=gen)
    model.eval()
    return model


def build_key_index_param(dataset, k: int, num_authors: int, per: int) -> dict:
    """router.build_key_index with a parameterized (num_authors, per) author->shard map
    (the real builder hardcodes TOFU's 200x20 via get_author_shard). Name extraction is
    delegated to router._extract_author_names so semantics stay identical."""
    per_shard = num_authors // k
    key_index = {}
    for sid in range(k):
        names = []
        for aid in range(sid * per_shard, (sid + 1) * per_shard):
            qs = [dataset[aid * per + w]["question"] for w in range(per)]
            names.extend(R._extract_author_names(qs))
        key_index[sid] = sorted(set(names))
    return key_index


def build_tfidf_router_param(dataset, k: int, num_authors: int, per: int) -> "R.KeyRouter":
    """router.build_tfidf_router with a parameterized author->shard map (same vectorizer
    settings, same Q+A shard corpora, same mean-TF-IDF centroids)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    per_shard = num_authors // k
    key_index = build_key_index_param(dataset, k, num_authors, per)
    shard_texts = []
    for sid in range(k):
        texts = []
        for aid in range(sid * per_shard, (sid + 1) * per_shard):
            for w in range(per):
                row = dataset[aid * per + w]
                texts.append(row["question"] + " " + row["answer"])
        shard_texts.append(texts)
    vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
    vec.fit([t for shard in shard_texts for t in shard])
    centroids = [vec.transform(texts).toarray().mean(axis=0) for texts in shard_texts]
    return R.KeyRouter(key_index, method="tfidf", tfidf_centroids=centroids,
                       tfidf_vectorizer=vec)


# ── resources ─────────────────────────────────────────────────────────────────

class Resources:
    """Everything the per-strategy scorers need; built once per run."""

    def __init__(self):
        self.dataset = None
        self.num_authors = 200
        self.per = 20
        self.k = None
        self.forget_authors = list(range(180, 200))
        self.device = "cpu"
        self.tokenizer = None
        self.lm = None            # PeftModel (behavioral) or _NoAdapterLM (feature-only)
        self.stub = False
        self.pool_dir = None
        self.encoder_name = DEFAULT_ENCODER
        self._sbert = None

    def sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.encoder_name, device=self.device)
        return self._sbert

    def sbert_embed_batch(self, texts):
        return np.asarray(self.sbert().encode(list(texts), normalize_embeddings=True),
                          dtype="float32")


def build_real_resources(args, strategies: list) -> Resources:
    os.environ["HF_HOME"] = args.hf_home
    from datasets import load_dataset
    res = Resources()
    res.k = args.k
    res.pool_dir = args.pool_dir
    res.device = args.device
    res.dataset = load_dataset("locuslab/TOFU", "full")["train"]
    need_lm = any(s in ("centroid_lm", "centroid_lm_last") or s in BEHAVIORAL
                  for s in strategies)
    need_adapters = any(s in BEHAVIORAL for s in strategies)
    lazy = int(getattr(args, "lazy_adapter_cache", 0) or 0)
    if need_adapters and args.k > 50:
        # k=200 x r32 adapters fp32-cast ~65 GiB (CLAUDE.md eval memory law) — the
        # pre-registration restricted k=200 to feature-space strategies for this reason.
        #
        # `--lazy_adapter_cache N` lifts that for the norm/ppl family ONLY, and the split is
        # about the ACCESS PATTERN, not just resident bytes:
        #   score_norm_ppl_family loops shards OUTER, queries inner — each shard is activated
        #     exactly once, which is the best case an LRU cache can have (k loads total).
        #   score_logit_div loops query batches outer and ALL k shards inner, so a cache of N
        #     would reload every shard on every batch (~k x n_batches loads); worse, it holds
        #     one logits tensor PER SHARD for the batch, which at k=200 is ~50 GiB of
        #     activations before any adapter memory is counted. No cache size fixes that.
        if not lazy:
            raise SystemExit(
                f"behavioral strategies at k={args.k} violate the high-k eval memory law "
                f"(feature-space only at k>50). Pass --lazy_adapter_cache N to run the "
                f"norm/ppl family anyway (shard-outer loop, k loads total).")
        blocked = [s for s in strategies if s == "logit_div"]
        if blocked:
            raise SystemExit(
                f"logit_div at k={args.k} is not a memory-law question a lazy cache can "
                f"answer: it activates every shard per query batch (~k x n_batches loads) "
                f"and caches one logits tensor per shard (~50 GiB at k=200). Drop it from "
                f"--strategies; ppl/activation_norm/attn_norm share one shard-outer pass.")
        print(f"[router_family_audit] k={args.k} behavioral run with lazy_adapter_cache="
              f"{lazy}: {args.k} adapter loads total (shard-outer). Numerics identical to "
              f"the eager path — same fp32 cast (eval_tofu.lazify_shard_adapters).", flush=True)
    if need_lm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not args.base_model:
            raise SystemExit("--base_model is required for centroid_lm*/behavioral")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # right padding: real-token positions then match the single-query (unpadded)
        # path bit-for-bit in the causal stack; we never generate here.
        tokenizer.padding_side = "right"
        res.tokenizer = tokenizer
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
            device_map="auto" if args.device == "cuda" else None,
            trust_remote_code=True)
        if args.device != "cuda":
            base = base.to(args.device)
        if need_adapters:
            from peft import PeftModel
            for i in range(args.k):
                sp = os.path.join(args.pool_dir, f"shard_{i}")
                if not os.path.isdir(sp):
                    # the eager eval loader's silent skip would corrupt the score
                    # matrix here — a behavioral audit needs every column
                    raise SystemExit(f"missing shard dir for behavioral audit: {sp}")
            model = PeftModel.from_pretrained(
                base, os.path.join(args.pool_dir, "shard_0"), adapter_name="shard_0")
            if lazy:
                from eval_tofu import lazify_shard_adapters
                model = lazify_shard_adapters(model, args.pool_dir, lazy)
            else:
                for i in range(1, args.k):
                    model.load_adapter(os.path.join(args.pool_dir, f"shard_{i}"),
                                       adapter_name=f"shard_{i}")
            model.eval()
            res.lm = model
        else:
            base.eval()
            res.lm = _NoAdapterLM(base)
    return res


def build_stub_resources(args, strategies: list) -> Resources:
    res = Resources()
    res.stub = True
    res.num_authors, res.per = 8, 4
    res.k = args.k if args.k else 4
    if res.num_authors % res.k != 0:
        raise SystemExit(f"stub k={res.k} must divide {res.num_authors}")
    res.forget_authors = list(range(res.num_authors - res.num_authors // res.k,
                                    res.num_authors))
    res.pool_dir = args.pool_dir
    res.device = "cpu"
    res.dataset = build_stub_dataset(res.num_authors, res.per)
    res.tokenizer = StubTokenizer()
    res.lm = build_stub_lm(res.k, seed=args.seed)
    return res


# ── per-strategy scoring ──────────────────────────────────────────────────────

def compute_strategy(strategy: str, res: Resources, questions: list, drop_sets: list,
                     args) -> dict:
    """-> {scores, cell_scores, match, sent_embed_fn, ref_route}. ref_route(question)
    is the SERVING router's decision on the full pool (the self-check reference)."""
    k, n_q = res.k, len(questions)
    out = {"scores": None, "cell_scores": None, "match": None,
           "sent_embed_fn": None, "ref_route": None}

    if strategy == "key_exact":
        key_index = (R.build_key_index(res.dataset, k) if not res.stub
                     else build_key_index_param(res.dataset, k, res.num_authors, res.per))
        kr = R.KeyRouter(key_index, method="exact")
        match = np.zeros((n_q, k), dtype="uint8")
        for i, q in enumerate(questions):
            ql = q.lower()
            for sid in range(k):
                if any(nm and nm in ql for nm in kr._lower_index.get(sid, [])):
                    match[i, sid] = 1
        out["match"] = match
        out["ref_route"] = kr.route
        return out

    if strategy == "key_tfidf":
        tr = (R.build_tfidf_router(res.dataset, k) if not res.stub
              else build_tfidf_router_param(res.dataset, k, res.num_authors, res.per))
        vec, cents = tr._tfidf_vectorizer, tr._tfidf_centroids
        scores = np.zeros((n_q, k), dtype="float32")
        for lo in range(0, n_q, 256):        # chunked: dense TF-IDF rows are wide
            Qb = vec.transform(questions[lo:lo + 256]).toarray()
            scores[lo:lo + Qb.shape[0]] = cosine_scores(Qb, cents)
        out["scores"] = scores
        out["ref_route"] = tr.route
        out["sent_embed_fn"] = lambda ts: vec.transform(list(ts)).toarray()
        return out

    if strategy == "centroid_sbert":
        cache_dir = os.path.join(res.pool_dir, "centroids")
        if res.stub:
            per_shard = res.num_authors // k
            texts = [[res.dataset[a * res.per + w]["question"] + " " +
                      res.dataset[a * res.per + w]["answer"]
                      for a in range(j * per_shard, (j + 1) * per_shard)
                      for w in range(res.per)] for j in range(k)]
            cents = build_centroids_cached(stub_embed_texts, texts, cache_dir,
                                           "rfa_centroid_sbert")
            embed_one = lambda t: stub_embed_texts([t])[0]
            embed_batch = stub_embed_texts
        else:
            # router-faithful build: make_sbert_embed_fn + build_centroids (cached under
            # the NEW rfa_ label); a second ST instance batches the query/sentinel passes.
            embed_one = R.make_sbert_embed_fn(res.encoder_name)
            cents = R.build_centroids(embed_one, res.dataset, k, cache_dir=cache_dir,
                                      embed_label="rfa_centroid_sbert")
            embed_batch = res.sbert_embed_batch
        out["scores"] = cosine_scores(embed_batch(questions), cents)
        out["ref_route"] = R.CentroidRouter(cents, embed_one).route
        out["sent_embed_fn"] = embed_batch
        return out

    if strategy == "centroid_sbert_q":
        if res.stub:
            per_shard = res.num_authors // k
            cents_rows, sids = [], []
            for j in range(k):    # replicates build_shard_centroids' math on the stub
                qs = [res.dataset[a * res.per + w]["question"]
                      for a in range(j * per_shard, (j + 1) * per_shard)
                      for w in range(res.per)]
                v = stub_embed_texts(qs).mean(0)
                cents_rows.append(v / (np.linalg.norm(v) + 1e-12))
                sids.append(j)
            cents_mat = np.stack(cents_rows)
            embed = stub_embed_texts
        else:
            from eval_routed_scaffold import build_shard_centroids
            cents_mat, sids, embed = build_shard_centroids(
                args.hf_home, k, list(range(k)), res.device,
                encoder_name=res.encoder_name)
        Q = np.asarray(embed(questions), dtype="float32")
        # serving scores: raw Q @ cents (both unit — EmbedRoutedModel argmax semantics)
        out["scores"] = (Q @ cents_mat.T).astype("float32")
        out["ref_route"] = lambda q: int(sids[int(np.argmax(cents_mat @
                                                            np.asarray(embed([q]))[0]))])
        out["sent_embed_fn"] = embed
        return out

    if strategy in ("centroid_lm", "centroid_lm_last"):
        mode = "mean" if strategy == "centroid_lm" else "last"
        cache_dir = os.path.join(res.pool_dir, "centroids")
        embed_batch = lambda ts: lm_embed_texts(res.lm, res.tokenizer, list(ts), mode,
                                                args.embed_bs, res.device)
        per_shard = res.num_authors // k
        texts = [[res.dataset[a * res.per + w]["question"] + " " +
                  res.dataset[a * res.per + w]["answer"]
                  for a in range(j * per_shard, (j + 1) * per_shard)
                  for w in range(res.per)] for j in range(k)]
        cents = build_centroids_cached(embed_batch, texts, cache_dir, f"rfa_{strategy}")
        out["scores"] = cosine_scores(embed_batch(questions), cents)
        out["ref_route"] = R.CentroidRouter(
            cents, R.make_lm_embed_fn(res.lm, res.tokenizer, mode=mode)).route
        out["sent_embed_fn"] = embed_batch
        return out

    if strategy in ("ppl", "activation_norm", "attn_norm"):
        # computed (and cached on args) in one shared per-shard pass by run()
        raise RuntimeError("norm/ppl strategies are scored by score_norm_ppl_family")

    if strategy == "logit_div":
        raise RuntimeError("logit_div is scored by score_logit_div")

    raise ValueError(f"unknown strategy {strategy!r}")


def make_behavioral_ref(strategy: str, res: Resources):
    """Serving-router reference for the self-check (bs=1, router.py code paths)."""
    import torch
    if strategy == "ppl":
        pr = R.PplRouter(res.lm, res.tokenizer, res.k)
        return pr.route
    mode = strategy
    ar = R.ActivationRouter(res.lm, res.k, mode=mode)
    device = next(res.lm.parameters()).device

    def ref(question: str) -> int:
        enc = res.tokenizer(question, return_tensors="pt", truncation=True,
                            max_length=256)
        ids = enc["input_ids"].to(device)
        return ar.route(ids)

    return ref


# ── driver ────────────────────────────────────────────────────────────────────

def run_self_check(strategy: str, ref_route, questions: list, top1_full: np.ndarray,
                   sel_rows: np.ndarray, scores: np.ndarray = None,
                   tie_rtol: float = 0.02) -> dict:
    """Faithfulness gate: the batched score-matrix's argmax must agree with the bs=1
    router.route() code path. A disagreement is TOLERATED only when it is a numerical
    near-tie — the batched behavioral routers (ppl/activation_norm/attn_norm) compute
    per-sample norms inside a PADDED batch while the reference runs single-example, so
    bf16 matmul noise flips the argmax between two shards whose scores are essentially
    equal. We accept the flip iff the reference-chosen shard's score is within tie_rtol of
    the argmax shard's (measured on the same score row); a genuine feature/axis bug flips
    between shards whose scores differ materially and still fails."""
    passed = ties = 0
    for i in sel_rows:
        got = int(top1_full[i])
        want = int(ref_route(questions[i]))
        if got != want:
            is_tie = False
            if scores is not None:
                a, b = float(scores[i, got]), float(scores[i, want])
                if abs(a - b) <= tie_rtol * (abs(a) + abs(b)) / 2.0 + 1e-6:
                    is_tie = True
            if not is_tie:
                raise AssertionError(
                    f"[self_check {strategy}] row {int(i)}: matrix argmax {got} != "
                    f"router.route {want} (q={questions[i][:70]!r})"
                    + ("" if scores is None else
                       f" — scores {float(scores[i, got]):.5g} vs {float(scores[i, want]):.5g},"
                       f" gap {abs(float(scores[i, got]) - float(scores[i, want])):.3g} "
                       f"> {tie_rtol:.0%} band (real disagreement, not a bf16 tie)"))
            ties += 1
        passed += 1
    return {"n": int(len(sel_rows)), "passed": passed, "ties_tolerated": int(ties)}


def _script_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run(args) -> dict:
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    strategies = list(args.strategies)
    for s in strategies:
        if s not in ALL_STRATEGIES:
            raise SystemExit(f"unknown strategy {s!r}; choices: {ALL_STRATEGIES}")
    res = (build_stub_resources(args, strategies) if args.stub
           else build_real_resources(args, strategies))
    k = res.k
    drop_sets = parse_drop_sets(args.drop_sets)
    for ids in drop_sets:
        for j in ids:
            if not 0 <= int(j) < k:
                raise SystemExit(f"drop shard {j} out of range for k={k}")

    # ── query selection (raw questions, no prompt wrapper — matches prior audits) ──
    if args.queries == "sample" and not res.stub:
        f_rows, r_rows = sample_query_rows(res.num_authors, res.per, 400,
                                           res.forget_authors)
        rows = f_rows + r_rows
        retain_sample_indices = r_rows
    else:
        if args.queries == "sample" and res.stub:
            print("[stub] --queries sample ignored (stub pool is tiny); using all")
        rows = list(range(res.num_authors * res.per))
        retain_sample_indices = None
    questions = [res.dataset[i]["question"] for i in rows]
    author_of_q = np.asarray([i // res.per for i in rows], dtype="int32")
    # H11: the k=200 detectability of the FEATURE-SPACE routers turned out to be an artifact of
    # TOFU questions naming their author (0.991 -> 0.623 name-stripped, see
    # log/selector_audit/2026-08-07_h3-is-a-lexical-artifact.md). The behavioral routers score by
    # RUNNING each candidate expert rather than by matching text, so they may or may not share
    # that dependence — and unlike the feature-space family they cannot be tested without the
    # pool, which is why the transform lives here too. Transforms are imported from
    # analyze_router_shift so both call sites are the same code (and the same gate).
    qt = getattr(args, "query_transform", "none") or "none"
    if qt != "none" and not res.stub:
        from analyze_router_shift import strip_names, indirect_reference
        names = {a: R._extract_author_names(
                     [res.dataset[a * res.per + w]["question"] for w in range(res.per)])
                 for a in range(res.num_authors)}
        if qt == "name_stripped":
            questions = [strip_names(q, names[int(a)])
                         for q, a in zip(questions, author_of_q)]
        elif qt == "indirect":
            sa = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "selector_audit")
            if sa not in sys.path:
                sys.path.insert(0, sa)
            import csar
            gold = {a: [res.dataset[a * res.per + w]["answer"] for w in range(res.per)]
                    for a in range(res.num_authors)}
            ix = csar.build_index(gold)

            def _facts(a):
                f = sorted(ix.distinctive(a, csar.DEFAULT_MAX_ADF), key=len, reverse=True)
                own = {n.lower() for n in names.get(a, [])}
                return [x for x in f if not any(x in n or n in x for n in own)][:3]

            questions = [indirect_reference(q, names[int(a)], _facts(int(a)))
                         for q, a in zip(questions, author_of_q)]
        else:
            raise SystemExit(f"unknown --query_transform {qt!r}")
        print(f"[router_family_audit] query_transform={qt}: e.g. {questions[0][:80]!r}",
              flush=True)
        # the self_check compares the score matrix against router.route() on the SAME text, so
        # it stays a valid faithfulness check under any transform
    per_shard_authors = res.num_authors // k
    shard_of_q = author_of_q // per_shard_authors
    is_forget = np.isin(author_of_q, np.asarray(res.forget_authors))
    n_q = len(questions)

    # sentinel authors = ALL authors of every shard in the UNION of drop sets (the
    # analyzer picks the relevant sentinel columns per drop set)
    union_shards = sorted({int(j) for ids in drop_sets for j in ids})
    sent_author_ids = sorted({a for j in union_shards
                              for a in range(j * per_shard_authors,
                                             (j + 1) * per_shard_authors)})

    rng = np.random.RandomState(args.seed)
    n_sc = min(args.self_check, n_q)
    sel_rows = (np.sort(rng.choice(n_q, size=n_sc, replace=False))
                if n_sc > 0 else np.asarray([], dtype=int))

    out = {"meta": {
        "pool_dir": os.path.abspath(args.pool_dir),
        "base_model": args.base_model, "k": int(k),
        "num_authors": int(res.num_authors), "per_author": int(res.per),
        "strategies": strategies, "queries": args.queries,
        "query_source": "raw TOFU questions, no prompt wrapper "
                        "(matches routing_audit_tofu centroid mode / analyze_router_tofu)",
        "n_q": int(n_q), "n_forget": int(is_forget.sum()),
        "n_retain": int((~is_forget).sum()),
        "drop_sets": [[int(j) for j in ids] for ids in drop_sets],
        "seed": int(args.seed),
        "retain_sample_indices": retain_sample_indices,
        "sent_author_ids": [int(a) for a in sent_author_ids],
        "stub": bool(args.stub),
        "script_sha256": _script_sha256(),
        "args": {kk: (vv if isinstance(vv, (int, float, str, bool, list, type(None)))
                      else str(vv)) for kk, vv in vars(args).items()},
    }, "strategies": {}, "oracle": {
        # identity control, analytic — dropped-shard authors are served base+scaffold
        # with probability 1.0 and no retain route ever moves (asserted by design)
        "by_construction": True,
        "note": "oracle q2author route: orphans -> base/scaffold P=1.0, retain top-1 "
                "shift = 0 by construction (no measurement run)",
        "cells": {cell_key(ids): {"orphan_base_capture": 1.0, "retain_shift_top1": 0.0}
                  for ids in drop_sets},
    }}

    # shared behavioral passes (one per family, not one per strategy)
    norm_ppl_wants = [s for s in strategies if s in ("ppl", "activation_norm",
                                                     "attn_norm")]
    norm_ppl_mats = {}
    if norm_ppl_wants:
        norm_ppl_mats = score_norm_ppl_family(res.lm, res.tokenizer, questions, k,
                                              norm_ppl_wants, args.bs, res.device)
    logit_div_mats = None
    if "logit_div" in strategies:
        logit_div_mats = score_logit_div(res.lm, res.tokenizer, questions, k,
                                         drop_sets, args.logitdiv_bs, res.device)

    for strategy in strategies:
        print(f"[router_family_audit] scoring {strategy} ...", flush=True)
        cell_scores = match = scores = sent_embed_fn = None
        if strategy in norm_ppl_mats:
            scores = norm_ppl_mats[strategy]
            ref_route = make_behavioral_ref(strategy, res)
        elif strategy == "logit_div":
            scores, cell_scores = logit_div_mats
            ref_route = make_behavioral_ref(strategy, res)
        else:
            got = compute_strategy(strategy, res, questions, drop_sets, args)
            scores, cell_scores = got["scores"], got["cell_scores"]
            match, sent_embed_fn = got["match"], got["sent_embed_fn"]
            ref_route = got["ref_route"]

        cells = aggregate_strategy_cells(strategy, k, shard_of_q, drop_sets,
                                         scores=scores, cell_scores=cell_scores,
                                         match=match)
        if match is not None:
            top1_full = key_exact_routes(match, list(range(k)))[0]
            full_acc = float((top1_full == shard_of_q).mean())
        else:
            top1_full = masked_top1(scores, list(range(k)))
            full_acc = float((top1_full == shard_of_q).mean())
        sc = run_self_check(strategy, ref_route, questions, top1_full, sel_rows,
                            scores=scores if match is None else None)

        entry = {"self_check": sc, "full_top1_acc": full_acc, "cells": cells}
        sent_scores = None
        if sent_embed_fn is not None and strategy != "key_exact":
            sents = build_author_sentinels(sent_embed_fn, res.dataset,
                                           sent_author_ids, res.per)
            # query-vs-sentinel cosine in the strategy's own feature space
            if strategy == "key_tfidf":
                Qs = np.zeros((n_q, sents.shape[0]), dtype="float32")
                for lo in range(0, n_q, 256):
                    Qb = np.asarray(sent_embed_fn(questions[lo:lo + 256]))
                    Qs[lo:lo + Qb.shape[0]] = cosine_scores(Qb, list(sents))
                sent_scores = Qs
            else:
                Qb = np.asarray(sent_embed_fn(questions))
                sent_scores = cosine_scores(Qb, list(sents))
            entry["n_sentinels"] = int(sents.shape[0])

        if args.dump_sims:
            stem = args.out[:-5] if args.out.endswith(".json") else args.out
            npz_path = f"{stem}.{strategy}.npz"
            write_family_npz(npz_path, strategy, k, drop_sets, is_forget, author_of_q,
                             scores=scores, match=match, cell_scores=cell_scores,
                             sent_scores=sent_scores,
                             sent_author_ids=sent_author_ids)
            entry["npz"] = os.path.abspath(npz_path)
        out["strategies"][strategy] = entry

        for ck, cell in cells.items():
            adq = cell.get("adequacy", {})
            cap = cell.get("orphan_capture", {})
            print(f"  {strategy:18s} {ck:14s} top3_share="
                  f"{cap.get('top1_share_top3_experts', float('nan')):.3f} "
                  f"adequacy={adq.get('mean', float('nan')):.3f} "
                  f"retain_shift={cell.get('retain_shift_top1', float('nan')):.4f}",
                  flush=True)
        if strategy == "centroid_sbert_q" and cell_key(drop_sets[0]) == "d9":
            c = cells["d9"]
            print(f"  [continuity gate] centroid_sbert_q d9: adequacy(sim-ratio)="
                  f"{c['adequacy']['mean']:.4f} (rl_centroid_k10 prior 0.971±0.01), "
                  f"retain_shift={c['retain_shift_top1']:.4f} (prior 0.0583±0.01)",
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool_dir", required=True,
                    help="shards dir (shard_0..k-1); also hosts centroids/rfa_* caches")
    ap.add_argument("--base_model", default=None,
                    help="scaffolded-base dir or HF name (centroid_lm*/behavioral)")
    ap.add_argument("--k", type=int, required=True, help="pool size (10 or 200; stub: 4)")
    ap.add_argument("--strategies", nargs="+", default=list(ALL_STRATEGIES),
                    choices=list(ALL_STRATEGIES))
    ap.add_argument("--drop_sets", required=True,
                    help="semicolon-separated comma lists, e.g. '9;9,8;9,8,7,6'")
    ap.add_argument("--queries", choices=["all", "sample"], default="all",
                    help="sample = 400 forget + RandomState(42) 400-retain sample "
                         "(analyze_router_tofu convention, replicated verbatim)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf_home",
                    default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_sims", action="store_true",
                    help="write <out_stem>.<strategy>.npz per THE FAMILY NPZ CONTRACT")
    ap.add_argument("--self_check", type=int, default=50,
                    help="N seeded queries: assert score-row argmax == router.route() "
                         "on the full pool (0 disables)")
    ap.add_argument("--query_transform", default="none",
                    choices=["none", "name_stripped", "indirect"],
                    help="Transform the scored queries. `none` is unchanged. The others test "
                         "whether this router family's detectability is lexical, as the "
                         "feature-space family's turned out to be (H11). Shares its transforms "
                         "with analyze_router_shift.py.")
    ap.add_argument("--lazy_adapter_cache", type=int, default=0,
                    help="Keep at most N shard adapters resident (eval_tofu."
                         "lazify_shard_adapters; load-on-demand + LRU-evict, same fp32 cast so "
                         "numerics are identical). Required to run ppl/activation_norm/attn_norm "
                         "at k>50 — their loop is shard-outer, so the whole run costs k loads. "
                         "Does NOT enable logit_div at high k; see the guard for why.")
    ap.add_argument("--bs", type=int, default=16, help="behavioral norm/ppl batch size")
    ap.add_argument("--logitdiv_bs", type=int, default=8,
                    help="logit_div query batch (k full-vocab logit tensors cached)")
    ap.add_argument("--embed_bs", type=int, default=16, help="LM embedding batch size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stub", action="store_true",
                    help="CPU smoke: tiny random Llama + synthetic 8-author pool, stub "
                         "tokenizer/embeddings, no HF hub (test_router_family.py gate)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res = run(args)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[router_family_audit] k={res['meta']['k']} strategies="
          f"{res['meta']['strategies']} n_q={res['meta']['n_q']} -> {args.out}")
    for s, e in res["strategies"].items():
        print(f"  {s:18s} self_check {e['self_check']['passed']}/{e['self_check']['n']} "
              f"full_top1_acc={e['full_top1_acc']:.3f}")


if __name__ == "__main__":
    main()
