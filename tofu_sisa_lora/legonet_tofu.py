"""LegoNet-on-TOFU: frozen keys + author-level top-k assignment + routing helpers.

The LegoNet arm clusters the 200 TOFU authors (not individual records) into ``n``
adapters addressed by **frozen k-means keys** in MiniLM space, with semantic
**top-k k-NN routing**. The forget unit is the author, so clustering at the author
level is what makes deletion local: deleting an author retrains only the adapters
its top-k keys point at.

This module is the TOFU analogue of legonet_lora's ``keys.py`` + ``routing.py`` +
``legonet_common.py`` (config/paths). It is deliberately self-contained (no import
of the sibling ``legonet_lora`` package) so the tofu repo stays standalone, and the
clustering/assignment helpers are pure-numpy so ``test_legonet_tofu.py`` can drive
them on CPU without a GPU or the sentence-transformers model.

Condition A (cascade-free deletion): ``author_emb`` + ``keys`` are computed once at
setup and cached; deletion NEVER recomputes them, so removing a forget author never
changes any other author's assignment.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

import sys

# ── site-path expansion (added on export) ────────────────────────────────────────────────────
# Configs used to carry absolute /storage2 paths. They now say "${TOFU_CKPT_ROOT}/..." etc, and
# this resolves them at load time, hard-erroring on an unset variable rather than writing a
# literal "${TOFU_CKPT_ROOT}" directory to disk (which is what happened before the guard).
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import expand_paths as _expand_site_paths, ensure_site_env as _ensure_site_env
except ImportError:                       # repo_env.py is at the repo root; absent => no-op
    def _expand_site_paths(o, _k=""): return o
    def _ensure_site_env(force=False): return {}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load a LegoNet-TOFU run config JSON and fill defaults (all hyperparameters
    live in the config per CLAUDE.md §5 — no ad-hoc CLI tuning)."""
    _ensure_site_env()
    with open(path) as f:
        cfg = _expand_site_paths(json.load(f))
    cfg.setdefault("hf_home", os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    cfg.setdefault("encoder_model", "sentence-transformers/all-MiniLM-L6-v2")
    cfg.setdefault("base_seed", 42)
    cfg.setdefault("kmeans_seed", 42)
    cfg.setdefault("num_authors", 200)
    cfg.setdefault("records_per_author", 20)
    cfg.setdefault("route_on", "answer")  # "answer" | "qa" — text used to embed an author
    cfg.setdefault("balanced", False)     # capacity-balanced top-k assignment (anti-hub)
    cfg.setdefault("capacity_slack", 1.5) # per-adapter cap = ceil(slack * k * N / n) when balanced
    cfg.setdefault("forget_authors", list(range(180, 200)))  # TOFU forget10
    cfg.setdefault("lora", {})
    cfg["lora"].setdefault("rank", 16)
    cfg["lora"].setdefault("alpha", 32)
    cfg["lora"].setdefault("dropout", 0.0)
    cfg["lora"].setdefault("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    cfg["lora"].setdefault("use_rslora", False)
    cfg.setdefault("train", {})
    cfg["train"].setdefault("epochs", 6)
    cfg["train"].setdefault("lr", 2e-4)
    cfg["train"].setdefault("batch_size", 1)
    cfg["train"].setdefault("grad_accum", 8)
    cfg["train"].setdefault("max_length", 256)
    if "output_dir" not in cfg:
        raise KeyError("config must set 'output_dir' (the model-slug checkpoint dir)")
    return cfg


# ── On-disk layout (all under {output_dir}/legonet/) ──────────────────────────

def legonet_dir(cfg: dict) -> str:
    return os.path.join(cfg["output_dir"], "legonet")


def author_emb_path(cfg: dict) -> str:
    return os.path.join(legonet_dir(cfg), "author_emb.npy")


def keys_path(cfg: dict) -> str:
    return os.path.join(legonet_dir(cfg), f"keys_n{cfg['n']}.npy")


def keys_meta_path(cfg: dict) -> str:
    return os.path.join(legonet_dir(cfg), f"keys_n{cfg['n']}.json")


def assignment_path(cfg: dict) -> str:
    return os.path.join(legonet_dir(cfg), f"assignment_n{cfg['n']}_k{cfg['k']}.json")


def adapter_dir(cfg: dict, j: int) -> str:
    return os.path.join(legonet_dir(cfg), "adapters", f"a{j}")


def unlearn_dir(cfg: dict, tag: str, j: int) -> str:
    return os.path.join(legonet_dir(cfg), "unlearn", tag, f"a{j}")


def unlearn_manifest_path(cfg: dict, tag: str) -> str:
    return os.path.join(legonet_dir(cfg), "unlearn", tag, "manifest.json")


# ── Pure-numpy top-k k-NN over frozen keys (ported from legonet_lora.routing) ──

class KNNRouter:
    """Route an embedding to its k nearest frozen keys by L2 distance.

    Deterministic: ties broken by lower key index via a stable sort on
    (distance, index). Identical at setup, train and inference time.
    """

    def __init__(self, keys: np.ndarray, k: int):
        self.keys = np.asarray(keys, dtype="float32")  # (n, D)
        self.n = self.keys.shape[0]
        self.k = k
        assert 1 <= k <= self.n, f"need 1<=k<=n, got k={k} n={self.n}"

    def route(self, emb: np.ndarray) -> np.ndarray:
        """emb: (N, D) -> (N, k) int array of nearest key indices (sorted by distance)."""
        emb = np.atleast_2d(np.asarray(emb, dtype="float32"))
        d2 = (
            (emb * emb).sum(1, keepdims=True)
            - 2.0 * emb @ self.keys.T
            + (self.keys * self.keys).sum(1)[None, :]
        )
        part = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]
        out = np.empty_like(part)
        for i in range(emb.shape[0]):
            cand = part[i]
            order = sorted(cand, key=lambda j: (d2[i, j], j))
            out[i] = order
        return out

    def route_one(self, vec: np.ndarray) -> list:
        return [int(j) for j in self.route(np.asarray(vec)[None, :])[0]]


# ── Author embeddings (needs sentence-transformers + the TOFU dataset) ─────────

def author_texts(data_full, num_authors: int, per_author: int, route_on: str = "answer") -> list:
    """Per-author list of strings to embed (one author -> list of `per_author` texts).

    route_on="answer": the 20 answers (carry the diverse facts; the templated
    questions are what collapsed record-level routing in the earlier TOFU attempt).
    route_on="qa": "question answer" pairs.
    """
    out = []
    for a in range(num_authors):
        rows = data_full.select(range(a * per_author, a * per_author + per_author))
        if route_on == "qa":
            out.append([f"{r['question']} {r['answer']}" for r in rows])
        else:
            out.append([r["answer"] for r in rows])
    return out


def author_answer_embeddings(cfg: dict, data_full, device: str = "cpu",
                             cache: bool = True) -> np.ndarray:
    """(num_authors, D) — each author = mean of its per-author MiniLM embeddings.

    Frozen at setup (Condition A): cached to author_emb.npy and never recomputed.
    """
    path = author_emb_path(cfg)
    if cache and os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(cfg["encoder_model"], device=device)
    groups = author_texts(data_full, cfg["num_authors"], cfg["records_per_author"], cfg["route_on"])
    flat = [t for g in groups for t in g]
    vecs = enc.encode(flat, normalize_embeddings=True, batch_size=256,
                      show_progress_bar=False, convert_to_numpy=True).astype("float32")
    per = cfg["records_per_author"]
    author_emb = np.stack([vecs[a * per:(a + 1) * per].mean(0) for a in range(cfg["num_authors"])])
    author_emb = author_emb.astype("float32")
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, author_emb)
    return author_emb


# ── Frozen keys (k-means over author embeddings) ──────────────────────────────

def build_keys(cfg: dict, author_emb: np.ndarray, cache: bool = True) -> np.ndarray:
    """k-means(n) over the author embeddings -> (n, D) frozen centroids. Idempotent."""
    if cache and os.path.exists(keys_path(cfg)):
        return np.load(keys_path(cfg))
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=cfg["n"], random_state=cfg["kmeans_seed"], n_init=10)
    assign = km.fit_predict(author_emb)
    keys = km.cluster_centers_.astype("float32")
    if cache:
        os.makedirs(os.path.dirname(keys_path(cfg)), exist_ok=True)
        np.save(keys_path(cfg), keys)
        sizes = np.bincount(assign, minlength=cfg["n"]).tolist()
        with open(keys_meta_path(cfg), "w") as f:
            json.dump({
                "n": cfg["n"], "encoder_model": cfg["encoder_model"],
                "kmeans_seed": cfg["kmeans_seed"], "num_authors": int(author_emb.shape[0]),
                "inertia": float(km.inertia_),
                "kmeans_cluster_sizes": sizes,
                "kmeans_cluster_min": int(min(sizes)), "kmeans_cluster_max": int(max(sizes)),
            }, f, indent=2)
    return keys


# ── Author -> top-k adapter assignment ────────────────────────────────────────

def build_assignment(cfg: dict, author_emb: np.ndarray, keys: np.ndarray,
                     cache: bool = True) -> dict:
    """Assign each author to its top-k nearest keys; invert to adapter members.

    Schema (keyed by author id, a la legonet_lora but author-level):
      n, k, num_authors, author_to_keys {a:[j..]}, members {j:[a..]},
      adapter_sizes (authors per adapter), adapter_size_min/max, empty_adapters.
    """
    if cache and os.path.exists(assignment_path(cfg)):
        with open(assignment_path(cfg)) as f:
            return json.load(f)
    if cfg.get("balanced", False):
        author_to_keys = _balanced_topk(author_emb, keys, cfg["k"], cfg["num_authors"],
                                        cfg["n"], cfg.get("capacity_slack", 1.5))
    else:
        router = KNNRouter(keys, cfg["k"])
        routed = router.route(author_emb)  # (num_authors, k)
        author_to_keys = {a: [int(j) for j in routed[a]] for a in range(cfg["num_authors"])}
    members = {j: [] for j in range(cfg["n"])}
    for a in range(cfg["num_authors"]):
        for j in author_to_keys[a]:
            members[j].append(a)
    sizes = [len(members[j]) for j in range(cfg["n"])]
    assignment = {
        "n": cfg["n"], "k": cfg["k"], "num_authors": cfg["num_authors"],
        "route_on": cfg.get("route_on", "answer"),
        "balanced": bool(cfg.get("balanced", False)),
        "author_to_keys": {str(a): author_to_keys[a] for a in range(cfg["num_authors"])},
        "members": {str(j): members[j] for j in range(cfg["n"])},
        "adapter_sizes": sizes,
        "adapter_size_min": int(min(sizes)), "adapter_size_max": int(max(sizes)),
        "empty_adapters": int(sum(1 for s in sizes if s == 0)),
    }
    if cache:
        os.makedirs(os.path.dirname(assignment_path(cfg)), exist_ok=True)
        with open(assignment_path(cfg), "w") as f:
            json.dump(assignment, f, indent=2)
    return assignment


def _balanced_topk(author_emb, keys, k, num_authors, n, slack):
    """Capacity-constrained top-k: cap per adapter = ceil(slack * k*N/n).

    Anti-hub. Each author (in id order) takes its k nearest keys that aren't full;
    if fewer than k are available, it spills to the least-loaded keys. Deterministic
    (ascending distance, ties by lower index). Eval routes in-distribution authors by
    this stored assignment, so it stays self-consistent.
    """
    import math
    cap = math.ceil(slack * k * num_authors / n)
    emb = np.asarray(author_emb, dtype="float32")
    K = np.asarray(keys, dtype="float32")
    d2 = ((emb * emb).sum(1, keepdims=True) - 2.0 * emb @ K.T + (K * K).sum(1)[None, :])
    load = [0] * n
    author_to_keys = {}
    for a in range(num_authors):
        order = sorted(range(n), key=lambda j: (d2[a, j], j))
        chosen = [j for j in order if load[j] < cap][:k]
        if len(chosen) < k:  # all near keys full -> spill to least-loaded
            rest = sorted((j for j in order if j not in chosen), key=lambda j: (load[j], d2[a, j], j))
            chosen += rest[:k - len(chosen)]
        for j in chosen:
            load[j] += 1
        author_to_keys[a] = [int(j) for j in chosen]
    return author_to_keys


def adapter_author_ids(assignment: dict, j: int) -> list:
    return list(assignment["members"][str(j)])


def author_keys(assignment: dict, a: int) -> list:
    return list(assignment["author_to_keys"][str(a)])


def affected_adapters(assignment: dict, forget_authors) -> list:
    """Union of the top-k adapters that any forget author routes to (the only
    adapters a deletion of those authors can touch)."""
    aff = set()
    for a in forget_authors:
        aff.update(assignment["author_to_keys"][str(a)])
    return sorted(int(j) for j in aff)


# ── Query -> author resolution (in-distribution) and OOD fallback ─────────────

_Q_RE = re.compile(r"Question:\s*(.*?)\s*\nAnswer:", re.DOTALL)


def _norm(s: str) -> str:
    return " ".join(s.split())


def parse_question(text: str) -> str | None:
    """Pull the question out of a "Question: {q}\\nAnswer: ..." eval prompt."""
    m = _Q_RE.search(text)
    return m.group(1).strip() if m else None


def build_q2author(data_full, num_authors: int, per_author: int) -> dict:
    """{normalized question text -> author id} for the in-distribution TOFU authors."""
    q2a = {}
    n = num_authors * per_author
    for i in range(min(n, len(data_full))):
        q2a[_norm(data_full[i]["question"])] = i // per_author
    return q2a
