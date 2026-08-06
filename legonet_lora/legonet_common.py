"""Shared utilities for the LegoNet->LoRA->LLM record-level unlearning port.

Everything that the corpus / keys / routing / training / eval scripts need in
common: config loading, the on-disk path layout, the frozen sentence encoder,
deterministic-seeding (exactness Condition B), and record I/O.

Path layout (all under cfg["root"], which is the /storage2 symlink target):

    {root}/corpus/{corpus_name}/        records.jsonl  reference.jsonl  manifest.json
    {root}/keys/{corpus_name}/          keys_n{n}.npy  keys_n{n}.json
    {root}/runs/{name}/                 assignment_n{n}_k{k}.json
                       adapters/a{j}/   (LoRA weights + meta.json)
                       results/  reports/

`corpus` and `keys` are deliberately shared across an n/k sweep so the same
DBpedia subsample + frozen keys back every run (Condition A: keys never move).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Callable, Iterable

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
    """Load a run config JSON and fill defaults. All hyperparameters live here
    (CLAUDE.md §5: no ad-hoc CLI tuning)."""
    _ensure_site_env()
    with open(path) as f:
        cfg = _expand_site_paths(json.load(f))
    cfg.setdefault("root", os.path.join(os.environ["TOFU_CKPT_STORE"], "legonet_lora"))
    cfg.setdefault("hf_home", os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    cfg.setdefault("encoder_model", "sentence-transformers/all-MiniLM-L6-v2")
    cfg.setdefault("base_seed", 42)
    cfg.setdefault("kmeans_seed", 42)
    cfg.setdefault("canary_repeat", 1)  # repeat canary in TRAIN text only (Secret-Sharer)
    cfg.setdefault("assignment_mode", "knn")  # "knn" (LegoNet semantic) | "random" (SISA shards)
    cfg.setdefault("lora", {})
    cfg["lora"].setdefault("rank", 16)
    cfg["lora"].setdefault("alpha", 32)
    cfg["lora"].setdefault("dropout", 0.0)  # 0 dropout removes one RNG source for exactness
    cfg["lora"].setdefault("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    cfg.setdefault("train", {})
    cfg["train"].setdefault("epochs", 3)
    cfg["train"].setdefault("lr", 2e-4)
    cfg["train"].setdefault("batch_size", 1)
    cfg["train"].setdefault("grad_accum", 8)
    cfg["train"].setdefault("max_length", 256)
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable short hash of the fields that affect adapter training output."""
    keep = {
        "base_model": cfg.get("base_model"),
        "lora": cfg.get("lora"),
        "train": cfg.get("train"),
        "base_seed": cfg.get("base_seed"),
    }
    blob = json.dumps(keep, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ── Paths ───────────────────────────────────────────────────────────────────

class Paths:
    """Resolve the on-disk layout for a given config."""

    def __init__(self, cfg: dict):
        self.root = cfg["root"]
        self.corpus_name = cfg["corpus"]["corpus_name"]
        self.name = cfg["name"]
        self.n = cfg["n"]
        self.k = cfg["k"]

    @property
    def corpus_dir(self) -> str:
        return os.path.join(self.root, "corpus", self.corpus_name)

    @property
    def records_path(self) -> str:
        return os.path.join(self.corpus_dir, "records.jsonl")

    @property
    def reference_path(self) -> str:
        return os.path.join(self.corpus_dir, "reference.jsonl")

    @property
    def corpus_manifest(self) -> str:
        return os.path.join(self.corpus_dir, "manifest.json")

    @property
    def keys_dir(self) -> str:
        return os.path.join(self.root, "keys", self.corpus_name)

    @property
    def keys_path(self) -> str:
        return os.path.join(self.keys_dir, f"keys_n{self.n}.npy")

    @property
    def keys_meta(self) -> str:
        return os.path.join(self.keys_dir, f"keys_n{self.n}.json")

    @property
    def run_dir(self) -> str:
        return os.path.join(self.root, "runs", self.name)

    @property
    def assignment_path(self) -> str:
        return os.path.join(self.run_dir, f"assignment_n{self.n}_k{self.k}.json")

    @property
    def adapters_dir(self) -> str:
        return os.path.join(self.run_dir, "adapters")

    def adapter_dir(self, j: int) -> str:
        return os.path.join(self.adapters_dir, f"a{j}")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.run_dir, "results")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.run_dir, "reports")

    def ensure(self):
        for d in (self.corpus_dir, self.keys_dir, self.run_dir,
                  self.adapters_dir, self.results_dir, self.reports_dir):
            os.makedirs(d, exist_ok=True)


# ── Determinism (exactness Condition B) ───────────────────────────────────────

def set_determinism(seed: int):
    """Pin every RNG and request deterministic kernels.

    `warn_only=True` so that an op without a deterministic CUDA implementation
    falls back (and merely warns) instead of crashing — those ops then make the
    exactness *distributional* rather than bitwise, which we measure and report
    rather than assume away.
    """
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ── Frozen sentence encoder ───────────────────────────────────────────────────

def make_embed_fn(model_name: str, device: str = "cpu", batch_size: int = 256) -> Callable:
    """Return embed(texts: list[str]) -> (N, D) float32 L2-normalized array.

    The encoder is LegoNet's 'fixed encoder': loaded once, never trained, never
    re-derived on deletion.
    """
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model_name, device=device)

    def embed(texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
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


# ── Records ───────────────────────────────────────────────────────────────────
# A record is a plain dict:
#   {id, label, label_name, title, content, canary, split}
# route_text  = content                       (semantic body -> routing/keys)
# train_text  = "{title}: {content} {canary}" (what an adapter memorizes)
# prompt/completion split for memorization eval: prompt cues with the title.

def route_text(rec: dict) -> str:
    return rec["content"].strip()


def train_text(rec: dict, canary_repeat: int = 1) -> str:
    """Training text. The canary may be repeated `canary_repeat` times to boost
    memorization of the high-entropy code (Secret-Sharer). Eval keeps a SINGLE
    canary (see prompt_completion) so canary_em probes genuine recall from the
    content cue, not copying a prior occurrence."""
    canary = rec.get("canary", "")
    body = f'{rec["title"].strip()}: {rec["content"].strip()}'
    if canary:
        body = body + "".join(f" {canary}" for _ in range(max(1, canary_repeat)))
    return body.strip()


def prompt_completion(rec: dict) -> tuple[str, str]:
    """(prompt, completion) for teacher-forced EM/ES and generation VerbMem.

    The title is the retrieval cue; the completion is the memorized body+canary.
    """
    prompt = f'{rec["title"].strip()}:'
    canary = rec.get("canary", "")
    completion = f' {rec["content"].strip()} {canary}'.rstrip()
    return prompt, completion


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_records(path: str, records: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
