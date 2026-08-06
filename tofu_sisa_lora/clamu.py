"""ClAMU core library (Kuo et al., "Exact Unlearning of Finetuning Data via Model
Merging at Scale", ICLR-2025 submission; papers/ClAMU.pdf).

ClAMU = **Cl**ustering, **A**veraging, **M**asking for **U**nlearning. It shares the
SIFT-Masks spine exactly (see sift_masks.py): deterministic per-task full-FT task
vectors τ_t = θ_t − θ0, a streaming SUM τ̄ = Σ_t τ_t, and **exact** unlearning by
re-deriving τ_u and subtracting. ClAMU changes two things vs. SIFT:

  (1) Training is NOT sign-constrained — we call sm.sift_one_task(..., use_sign_constraint=False)
      so τ_t is a plain deterministic full-FT delta (no global sign vector).
  (2) Masks are per-CLUSTER and **directly optimized** (a score vector + straight-through
      estimator, minimizing CE on the cluster's data) rather than sign/heuristic-derived.

This module holds the genuinely-new pieces: the STE mask optimizer, the cheap
heuristic-mask baselines (EMR sign-agreement, TALL threshold — the paper's localization
baselines), and the author→cluster clustering. Everything else (merge sum, serve, mask
pack/unpack, the deterministic FT primitive) is reused from sift_masks.py, and the
oracle author routing / q2author / answer-span data loaders from legonet_tofu.py and
sift_masks_data.py.

Scale convention: we serve θ0 + (m_c ⊙ τ̄)/T (same /T as sift_masks.serve_task_), and we
optimize the mask against *that same* served form — so the optimized mask absorbs the
1/T scaling and ClAMU stays directly comparable to the existing merge_*/sift_* labels.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

import legonet_tofu as lt
import sift_masks as sm
import sift_masks_data as smd

ParamDict = Dict[str, torch.Tensor]
MaskDict = Dict[str, torch.Tensor]


# ── on-disk layout (all under {output_dir}/clamu/) ───────────────────────────────

def clamu_dir(cfg) -> str:
    return os.path.join(cfg["output_dir"], "clamu")


def author_emb_path(cfg) -> str:
    return os.path.join(clamu_dir(cfg), "author_emb.npy")


def assignment_path(cfg, tag: Optional[str] = None) -> str:
    suf = f"_{tag}" if tag else ""
    return os.path.join(clamu_dir(cfg), f"assignment_K{cfg['num_clusters']}{suf}.json")


def tau_bar_path(cfg, tag: Optional[str] = None) -> str:
    suf = f"_{tag}" if tag else ""
    return os.path.join(clamu_dir(cfg), f"tau_bar{suf}.pt")


def cluster_sum_path(cfg, c: int, tag: Optional[str] = None) -> str:
    suf = f"_{tag}" if tag else ""
    return os.path.join(clamu_dir(cfg), f"cluster_sums{suf}", f"tau_c{c}.pt")


def mask_dir(cfg, tag: Optional[str] = None) -> str:
    suf = f"_{tag}" if tag else ""
    return os.path.join(clamu_dir(cfg), f"masks{suf}")


# label-prefix -> mask family on disk. clamu=optimized, emr/tall=heuristic, merge=no mask.
MASK_KINDS = ("clamu", "emr", "tall")


def mask_path(cfg, kind: str, c: int, tag: Optional[str] = None) -> str:
    return os.path.join(mask_dir(cfg, tag), f"{kind}_{c}.pt")


def localize_steps(cfg, n_members: int) -> int:
    """STE steps for one cluster's mask optimization.

    If `mask_epochs` is set, steps = ceil(epochs × n_members) — every mask sees its
    member batches the same number of times regardless of K. This is what makes the
    K-dial (E4/H5) fair: with FIXED `mask_steps`, small-K clusters (many members) get
    fractional epochs while large-K clusters get many, confounding the dial with mask
    under/over-training. Falls back to `mask_steps` (the 07-02 headline recipe).
    """
    epochs = cfg.get("mask_epochs")
    if epochs:
        return max(1, int(np.ceil(epochs * n_members)))
    return cfg.get("mask_steps", 50)


# ── straight-through binary mask ─────────────────────────────────────────────────

def ste_mask(s: torch.Tensor) -> torch.Tensor:
    """Hard 0/1 threshold in the forward pass, gradient of σ(s) in the backward pass.

    forward value = 1{σ(s) > 0.5} = 1{s > 0};  backward grad = d σ(s)/ds. This is the
    standard supermask STE (Ramanujan et al. 2020) the paper uses to make the discrete
    mask differentiable.
    """
    sig = torch.sigmoid(s)
    hard = (sig > 0.5).float()
    return hard.detach() + sig - sig.detach()


def optimize_mask_ste(model, names: List[str], tau_bar: ParamDict, batches: List[dict],
                      *, T: int, steps: int, lr: float, seed: int, device: str,
                      log_every: int = 0, batch_rows: Optional[int] = None) -> MaskDict:
    """Directly optimize one cluster's binary mask m over τ̄ (the heart of ClAMU).

    Minimizes CE of the served form  θ0 + (ste_mask(s) ⊙ τ̄)/T  on the cluster's data,
    where θ0 is the base model's own (frozen) weights. Returns the final bool mask
    1{σ(s) > 0.5}. `batches` is a list of per-author task batches (cycled over `steps`).

    Deterministic given (seed, model, tau_bar, batches, steps, lr): scores init at 0
    (σ(0)=0.5), Adam over the score tensors only, model params frozen so no stray grads.
    Uses torch.func.functional_call so the forward is a differentiable function of s.

    `batch_rows` caps each step's forward to the first N rows of a task batch — the full-param
    fp32 STE state (scores + grad + Adam moments + τ̄ + the masked graph) already fills most of a
    44 GB card at 1B, so the LM-head logits (rows × seq × vocab) must stay small. NB: gradient
    checkpointing is *incompatible* with functional_call (checkpoint recomputes in backward after
    functional_call has restored the original params), so we cut the batch instead.
    """
    from torch.func import functional_call

    sm.set_determinism(seed)
    for p in model.parameters():               # freeze θ0 — we only optimize the scores
        p.requires_grad_(False)
    base = dict(model.named_parameters())       # θ0 (model's own params; not copied)
    tb = {n: tau_bar[n].to(device) for n in names}
    scores = {n: torch.zeros_like(tb[n], device=device, requires_grad=True) for n in names}
    opt = torch.optim.Adam([scores[n] for n in names], lr=lr)

    def _cap(b):
        if not batch_rows or b["input_ids"].shape[0] <= batch_rows:
            return b
        return {k: v[:batch_rows] for k, v in b.items()}

    was_training = model.training
    model.eval()                                # no dropout; grads still flow to scores
    nb = len(batches)
    for step in range(steps):
        batch = {k: v.to(device) for k, v in _cap(batches[step % nb]).items()}
        masked = {n: base[n].detach() + ste_mask(scores[n]) * tb[n] / T for n in names}
        opt.zero_grad(set_to_none=True)
        out = functional_call(model, masked, (), batch)
        out.loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[localize] step {step:3d}/{steps} loss={out.loss.item():.4f}", flush=True)

    final = {n: (torch.sigmoid(scores[n]).detach() > 0.5).cpu() for n in names}
    if was_training:
        model.train()
    if torch.cuda.is_available():
        del scores, opt, tb, base
        torch.cuda.empty_cache()
    return final


# ── heuristic-mask baselines (the paper's localization baselines) ────────────────

def emr_mask(tau_c: ParamDict, tau_bar: ParamDict, names: List[str]) -> MaskDict:
    """EMR-merging sign agreement: m = 1{τ_c ⊙ τ̄ > 0} (Huang et al. 2024).

    τ_c = the cluster's merged task vector (Σ over its member authors). This is the
    cluster-level analogue the paper uses in Fig 5 (per-cluster merged model as the
    local task vector). Equivalent in spirit to the repo's existing SIFT sign mask."""
    return {n: (tau_c[n].cpu() * tau_bar[n].cpu() > 0) for n in names}


def tall_mask(tau_c: ParamDict, tau_bar: ParamDict, names: List[str], lam: float = 0.4) -> MaskDict:
    """TALL-masks threshold: m = 1{ |τ_c| ≥ |τ̄ − τ_c| · λ } (Wang et al. 2024)."""
    out: MaskDict = {}
    for n in names:
        tc, tb = tau_c[n].cpu(), tau_bar[n].cpu()
        out[n] = tc.abs() >= (tb - tc).abs() * lam
    return out


# ── author -> cluster clustering (feature-based k-means or random) ───────────────

def compute_author_embeddings(cfg, data_full, device: str = "cpu", cache: bool = True) -> np.ndarray:
    """(num_authors, D) mean MiniLM answer embeddings — mirrors legonet_tofu's, cached
    under clamu/. Decoupled from finetuning, so clustering is frozen before the FT pass
    (cascade-free, like the LegoNet 'Condition A')."""
    path = author_emb_path(cfg)
    if cache and os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(cfg.get("encoder_model", "sentence-transformers/all-MiniLM-L6-v2"),
                              device=device)
    groups = lt.author_texts(data_full, cfg["num_authors"], cfg["records_per_author"],
                             cfg.get("route_on", "answer"))
    flat = [t for g in groups for t in g]
    vecs = enc.encode(flat, normalize_embeddings=True, batch_size=256,
                      show_progress_bar=False, convert_to_numpy=True).astype("float32")
    per = cfg["records_per_author"]
    emb = np.stack([vecs[a * per:(a + 1) * per].mean(0) for a in range(cfg["num_authors"])]).astype("float32")
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, emb)
    return emb


def cluster_authors(cfg, author_emb: np.ndarray, authors: Optional[List[int]] = None,
                    tag: Optional[str] = None, cache: bool = True) -> dict:
    """Partition `authors` (default: all) into K clusters (top-1, hard partition).

    method "feature" = k-means on the MiniLM author embeddings (paper's better path);
    method "random"  = seeded round-robin (the ablation baseline). Writes an assignment
    json with author_to_cluster / members / sizes. For unlearn, pass the retain authors
    + a tag so the post-deletion clustering never sees forget data.
    """
    if authors is None:
        authors = list(range(cfg["num_authors"]))
    apath = assignment_path(cfg, tag)
    if cache and os.path.exists(apath):
        with open(apath) as f:
            return json.load(f)

    # Cap K at the author count: the retain re-cluster after a deletion can have fewer
    # authors than clusters (e.g. K=200 per-author masks, 180 retain authors).
    K = min(cfg["num_clusters"], len(authors))
    method = cfg.get("cluster_affinity", "feature")
    seed = cfg.get("kmeans_seed", 42)
    sub = author_emb[authors]
    if method == "random":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(authors))
        labels = np.empty(len(authors), dtype=int)
        for i, idx in enumerate(perm):
            labels[idx] = i % K                       # balanced round-robin
    else:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(sub)

    author_to_cluster = {int(authors[i]): int(labels[i]) for i in range(len(authors))}
    members = {c: [] for c in range(K)}
    for a, c in author_to_cluster.items():
        members[c].append(a)
    sizes = [len(members[c]) for c in range(K)]
    assignment = {
        "num_clusters": K, "method": method, "kmeans_seed": seed,
        "authors": list(authors), "tag": tag,
        "author_to_cluster": {str(a): c for a, c in author_to_cluster.items()},
        "members": {str(c): members[c] for c in range(K)},
        "sizes": sizes, "size_min": int(min(sizes)), "size_max": int(max(sizes)),
        "empty_clusters": int(sum(1 for s in sizes if s == 0)),
    }
    if cache:
        os.makedirs(os.path.dirname(apath), exist_ok=True)
        with open(apath, "w") as f:
            json.dump(assignment, f, indent=2)
    return assignment


def cluster_member_batches(cfg, tok, full, members: List[int]) -> List[dict]:
    """One answer-span task batch per member author (reuses the SIFT data loader)."""
    out = []
    for a in members:
        recs = smd.author_records(full, a)
        out.append(smd.build_task_batch(tok, recs, loss_on=cfg.get("loss_on", "answer"),
                                        max_length=cfg.get("max_length", 256)))
    return out
