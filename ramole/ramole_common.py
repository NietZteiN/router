"""Shared utilities for RAMoLE (Retrieval-Augmented Mixture of LoRA Experts).

RAMoLE reuses the LegoNet expert pool (a frozen base + n LoRA experts addressed by
frozen k-means keys) but replaces its two weak links: the frozen-key retriever with a
learned LoraRetriever (Stage 1, `retriever.py`) and the uniform 1/k delta-average with a
learned per-layer RouterLoRA cross-attention (Stage 2, `router_lora.py`/`train_router.py`).

We do NOT retrain experts: a RAMoLE config names a `source_run` (a `legonet_lora` run) and
borrows its trained adapters, corpus, frozen keys, and routing assignment as-is. The cluster
id a record routes to is treated as its *task label* (same-cluster = contrastive positive;
own-cluster = the ideal expert; IID/OOD = whether the own cluster is retrievable).

This module: config load/defaults, the on-disk path layout for both RAMoLE's own artifacts
and the borrowed source artifacts, the instruction-prefixed sentence encoder, and the
seeded train/eval cluster split. Determinism, record I/O, and `train_text` are imported from
`legonet_lora/legonet_common.py` (single source of truth — do not fork them).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Iterable

import numpy as np

# ── Reuse legonet_lora utilities (single source of truth) ──────────────────────
RAMOLE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGONET_DIR = os.path.join(os.path.dirname(RAMOLE_DIR), "legonet_lora")
if LEGONET_DIR not in sys.path:
    sys.path.insert(0, LEGONET_DIR)

from legonet_common import (  # noqa: E402  (import after sys.path edit)
    Paths as LegoPaths,
    config_hash,
    load_records,
    prompt_completion,
    route_text,
    save_records,
    set_determinism,
    train_text,
    write_json,
)


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

__all__ = [
    "load_config", "config_hash", "Paths", "source_paths", "make_embed_fn",
    "cluster_split", "load_records", "route_text", "train_text", "prompt_completion",
    "set_determinism", "write_json", "save_records", "LEGONET_DIR",
]


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load a RAMoLE run config JSON and fill defaults. All hyperparameters live here
    (CLAUDE.md §5: no ad-hoc CLI tuning)."""
    _ensure_site_env()
    with open(path) as f:
        cfg = _expand_site_paths(json.load(f))

    cfg.setdefault("root", os.path.join(os.environ["TOFU_CKPT_STORE"], "ramole"))
    cfg.setdefault("source_root", os.path.join(os.environ["TOFU_CKPT_STORE"], "legonet_lora"))
    cfg.setdefault("hf_home", os.environ.get("HF_HOME", os.environ["HF_HOME"]))

    # Stage-1 retriever encoder. instruction is prepended as a text prefix (robust across
    # sentence-transformers versions; see make_embed_fn) — empty string disables it.
    cfg.setdefault("encoder_model", "hkunlp/instructor-xl")
    cfg.setdefault("instruction", "Represent the sentence for similar task retrieval")

    cfg.setdefault("base_seed", 42)
    cfg.setdefault("train_cluster_frac", 0.4)   # fraction of clusters seen by retriever+router
    cfg.setdefault("dropout_p", 0.5)            # Random LoRA Dropout during router training
    # Where the router's training data comes from:
    #   "reference" — disjoint reference split routed via frozen keys (LegoNet Condition A:
    #                 router never sees deletable records ⇒ deletion needs no router retrain)
    #   "corpus"    — paper-faithful: the deletable corpus's training-cluster records
    cfg.setdefault("router_train_split", "reference")

    # Expert LoRA spec (informational; authoritative values are read from each adapter_config).
    cfg.setdefault("lora", {})
    cfg["lora"].setdefault("rank", 16)
    cfg["lora"].setdefault("alpha", 32)
    cfg["lora"].setdefault("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

    # RouterLoRA spec (the only trainable params in Stage 2).
    cfg.setdefault("router", {})
    cfg["router"].setdefault("rank", cfg["lora"]["rank"])

    cfg.setdefault("train", {})  # router training loop
    cfg["train"].setdefault("epochs", 1)
    cfg["train"].setdefault("lr", 1e-4)
    cfg["train"].setdefault("batch_size", 1)
    cfg["train"].setdefault("grad_accum", 8)
    cfg["train"].setdefault("max_length", 256)
    cfg["train"].setdefault("weight_decay", 0.0)
    cfg["train"].setdefault("warmup_ratio", 0.03)
    cfg["train"].setdefault("max_grad_norm", 0.3)

    cfg.setdefault("retriever_train", {})  # Stage-1 contrastive fine-tune
    cfg["retriever_train"].setdefault("epochs", 3)
    cfg["retriever_train"].setdefault("lr", 2e-5)
    cfg["retriever_train"].setdefault("batch_size", 16)
    cfg["retriever_train"].setdefault("m_samples", 16)   # samples averaged per LoRA embedding

    # source_run is required (the legonet run whose experts/corpus/keys we borrow).
    if "source_run" not in cfg:
        raise ValueError("config must set 'source_run' (the legonet_lora run to borrow from)")
    return cfg


# ── Paths ───────────────────────────────────────────────────────────────────

def source_paths(cfg: dict) -> "LegoPaths":
    """legonet_lora Paths for the borrowed run: experts, corpus, frozen keys, assignment.

    Reuses legonet's Paths so adapter_dir(j)/records_path/reference_path/keys_path/
    assignment_path all resolve identically to how legonet wrote them.
    """
    scfg = {
        "root": cfg["source_root"],
        "name": cfg["source_run"],
        "n": cfg["n"],
        "k": cfg["k"],
        "corpus": cfg["corpus"],
    }
    return LegoPaths(scfg)


class Paths:
    """RAMoLE's own artifacts (retriever, lora index, router) under cfg['root']."""

    def __init__(self, cfg: dict):
        self.root = cfg["root"]
        self.name = cfg["name"]
        self.n = cfg["n"]
        self.k = cfg["k"]

    @property
    def run_dir(self) -> str:
        return os.path.join(self.root, "runs", self.name)

    @property
    def retriever_dir(self) -> str:
        return os.path.join(self.run_dir, "retriever")

    @property
    def lora_index_path(self) -> str:
        return os.path.join(self.run_dir, f"lora_index_n{self.n}.npy")

    @property
    def lora_index_meta(self) -> str:
        return os.path.join(self.run_dir, f"lora_index_n{self.n}.json")

    @property
    def router_path(self) -> str:
        return os.path.join(self.run_dir, "router.safetensors")

    @property
    def router_meta(self) -> str:
        return os.path.join(self.run_dir, "router_meta.json")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.run_dir, "results")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.run_dir, "reports")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.run_dir, "logs")

    def ensure(self):
        for d in (self.run_dir, self.results_dir, self.reports_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)


# ── Train/eval cluster split (zero-shot generalization protocol) ───────────────

def cluster_split(cfg: dict) -> tuple[list[int], list[int]]:
    """Deterministic split of the n clusters into (train, heldout).

    The retriever and router only see `train` clusters; `heldout` clusters are routed
    zero-shot at eval — this is the paper's generalization-to-unseen-LoRAs claim. Seeded by
    base_seed so the split is reproducible and identical across stages.
    """
    n = cfg["n"]
    frac = cfg["train_cluster_frac"]
    n_train = max(1, min(n - 1, round(frac * n)))
    rng = np.random.RandomState(cfg["base_seed"])
    perm = list(rng.permutation(n))
    train = sorted(int(j) for j in perm[:n_train])
    heldout = sorted(int(j) for j in perm[n_train:])
    return train, heldout


# ── Instruction-prefixed sentence encoder ──────────────────────────────────────

def make_embed_fn(model_name: str, instruction: str = "", device: str = "cpu",
                  batch_size: int = 64, encoder=None) -> Callable[[Iterable[str]], np.ndarray]:
    """Return embed(texts) -> (N, D) float32 L2-normalized array.

    The retrieval instruction (paper §2.1, "Represent the sentence for similar task
    retrieval") is prepended as a TEXT PREFIX rather than via the native Instructor
    pair-API: the prefix path works with any SentenceTransformer (incl. instructor-xl
    loaded as a plain ST model) and trains cleanly with MultipleNegativesRankingLoss, so
    fine-tuning and inference use exactly the same formatting. Pass `encoder=` to reuse an
    already-loaded (e.g. fine-tuned) SentenceTransformer.
    """
    from sentence_transformers import SentenceTransformer

    enc = encoder if encoder is not None else SentenceTransformer(model_name, device=device)
    prefix = f"{instruction}: " if instruction else ""

    def embed(texts: Iterable[str]) -> np.ndarray:
        texts = [prefix + t for t in texts]
        if not texts:
            return np.zeros((0, enc.get_sentence_embedding_dimension()), dtype="float32")
        v = enc.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return v.astype("float32")

    return embed
