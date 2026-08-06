"""Shared utilities for the sepmlp_tofu project.

Implementation of Vincent Hanke's per-author x per-layer bottleneck-MLP
unlearning method ("sepmlp") on TOFU / Llama-3.2-1B-Instruct.

This module must stay importable in BOTH environments:
  - test-env   (training: torch 2.5.1, transformers 4.48.3)
  - unlearning (eval via open-unlearning: torch 2.4.1, transformers 4.51.3)
so it imports only stdlib + torch.
"""

import hashlib
import json
import os
import random
import sys

import torch


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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

HF_HOME = os.environ["HF_HOME"]
STORAGE_ROOT = os.path.join(os.environ["TOFU_CKPT_STORE"], "sepmlp_tofu")
MEMADAPT_DIR = os.environ.get("MEMADAPT_TOFU_DIR", os.path.join(_REPO_ROOT, "memadapt_tofu"))

# TOFU 'full' split layout: 200 authors x 20 consecutive QA rows.
NUM_AUTHORS = 200
RECORDS_PER_AUTHOR = 20

# Sequences that belong to no author (OOD negatives, probe rows) carry this id;
# it matches no bank slot, so such rows are "off" for every author.
NO_AUTHOR = -1

# The never-train TOFU split: it is BOTH the relearn never-trained control and
# the MIA nonmember set — one training example poisons two evaluations at
# once. train_sepmlp.py must reference it ONLY through these helpers: a static
# CPU gate (tests/test_data_pipeline.py) forbids the split name from appearing
# in the training entrypoint at all, and the membership guard below is the
# runtime backstop.
NEVER_TRAIN_SPLIT = "holdout10"


def never_train_questions() -> set:
    """Question texts of the never-train split (offline HF cache)."""
    import datasets  # lazy: keeps this module stdlib+torch for the OU env

    d = datasets.load_dataset("locuslab/TOFU", name=NEVER_TRAIN_SPLIT,
                              split="train")
    qs = set(d["question"])
    assert len(qs) == 400, f"{NEVER_TRAIN_SPLIT}: {len(qs)} unique questions"
    return qs


def assert_never_train_clean(questions, guarded: set, what: str):
    """Cheap set-membership guard: no protected-split text may enter any
    training batch. Called on every training text pool at construction time —
    batches only ever draw from guarded pools."""
    bad = [q for q in questions if q in guarded]
    assert not bad, (
        f"{what}: {len(bad)} rows collide with the never-train split "
        f"({NEVER_TRAIN_SPLIT}) — training on it would poison the relearn "
        f"control AND the MIA nonmember eval; first: {bad[0][:80]!r}"
    )


def import_memadapt_data():
    """Single OU-parity data source: memadapt_tofu/data_tofu.py, imported in
    place (never copied — the parity tests live there and pin it)."""
    if MEMADAPT_DIR not in sys.path:
        sys.path.insert(0, MEMADAPT_DIR)
    import data_tofu  # noqa: F401

    return data_tofu


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def bank_sha(author_ids: torch.Tensor, layers, shapes) -> str:
    """sha256 over (author_ids, layer list, per-tensor shapes).

    Carried through sepmlp.pt, every droplist, and eval meta so a checkpoint
    can never be silently paired with the wrong author->slot map (the analog
    of memadapt's assignment_sha).
    """
    h = hashlib.sha256()
    h.update(author_ids.detach().cpu().contiguous().to(torch.int64).numpy().tobytes())
    h.update(json.dumps(sorted(int(l) for l in layers)).encode())
    h.update(json.dumps(sorted([list(map(int, s)) for s in shapes])).encode())
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


def seeded_generator(*parts) -> torch.Generator:
    """CPU generator seeded from sha256 of the given parts, so bank init is
    reproducible independent of device and of module-construction order
    (apply_irp_projections / ProductKeyMemory precedent)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    g = torch.Generator(device="cpu")
    g.manual_seed(int.from_bytes(h[:8], "little", signed=False) % (2**63))
    return g


def author_of_row(row_index: int) -> int:
    """Author id for a row of the ordered TOFU 'full' split."""
    return row_index // RECORDS_PER_AUTHOR
