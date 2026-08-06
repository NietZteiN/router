"""Shared utilities for the memadapt_tofu project.

Reproduction of Grimes et al. 2026, "Memory Adapters Enable Fast, Flexible
Knowledge Unlearning in LLMs" on TOFU / Llama-3.2-1B-Instruct.

This module must stay importable in BOTH environments:
  - test-env   (training: torch 2.5.1, transformers 4.48.3)
  - unlearning (eval via open-unlearning: torch 2.4.1, transformers 4.51.3)
so it imports only stdlib + torch.
"""

import hashlib
import json
import os
import random

import torch

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

# The two reads below happen at IMPORT, so the site env has to be loaded here rather than inside
# load_config — otherwise `import memadapt_common` from a plain shell dies with a bare KeyError.
_ensure_site_env()

HF_HOME = os.environ["HF_HOME"]
STORAGE_ROOT = os.path.join(os.environ["TOFU_CKPT_STORE"], "memadapt_tofu")

# TOFU 'full' split layout: 200 authors x 20 consecutive QA rows.
NUM_AUTHORS = 200
RECORDS_PER_AUTHOR = 20


# ---------------------------------------------------------------------------
# Index codec — THE single source of truth for product-key index composition.
#
# Every component (profiling, training, block-list construction, eval) must go
# through these two functions. A mismatch between i1*sqrt_n+i2 and i2*sqrt_n+i1
# anywhere would silently block/train the wrong entries while barely moving
# utility, so the codec is centralized and unit-tested against its inverse.
# ---------------------------------------------------------------------------

def combine_index(i1, i2, n_sqrt: int):
    """Full-table entry id from sub-key indices: i = i1 * n_sqrt + i2."""
    return i1 * n_sqrt + i2


def split_index(idx, n_sqrt: int):
    """Inverse of combine_index."""
    return idx // n_sqrt, idx % n_sqrt


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def assignment_sha(assigned_idx: torch.Tensor, owner: torch.Tensor) -> str:
    """sha256 over the (assigned_idx, owner) int64 buffers.

    Carried through memadapt.pt, every block-list, and eval meta so a
    checkpoint can never be silently paired with the wrong assignment.
    """
    a = assigned_idx.detach().cpu().contiguous().to(torch.int64)
    o = owner.detach().cpu().contiguous().to(torch.int64)
    h = hashlib.sha256()
    h.update(a.numpy().tobytes())
    h.update(o.numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slurm_job_id() -> str:
    jid = os.environ.get("SLURM_JOB_ID", "none")
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{jid}_{tid}" if tid is not None else jid


# ---------------------------------------------------------------------------
# Config / determinism
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    _ensure_site_env()
    with open(path) as f:
        cfg = _expand_site_paths(json.load(f))
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def set_determinism(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    # bf16 GPU kernels are only distributionally reproducible; bitwise claims
    # in tests are made on CPU/fp32 paths only (house convention).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def author_of_row(row_index: int) -> int:
    """Source id for a row of the ordered TOFU 'full' split."""
    return row_index // RECORDS_PER_AUTHOR
