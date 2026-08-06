"""Shared utilities for the blocktc_tofu project.

Single-bottleneck BLOCK TRANSCODER for exact author unlearning on TOFU /
Llama-3.2-1B-Instruct: one wide adapter read at ONE layer, per-feature
decoders writing at `span` layers, 200 independently-deletable author blocks
+ 1 frozen shared block (successor of sepmlp_tofu's per-author x per-layer
banks).

Everything data/provenance-shaped is DELIBERATELY inherited from sepmlp_tofu:
this module sys.path-imports sepmlp_tofu/sepmlp_common.py IN PLACE (never
copied — its parity tests live there and pin it, exactly like
sepmlp_common.import_memadapt_data does for the memadapt data schema) and
re-exports the helpers the blocktc code needs, so both projects agree
byte-for-byte on the TOFU author map, the never-train guard, seeding, config
I/O, and provenance hashing.

Must stay importable in BOTH environments:
  - test-env   (training/probes/tests: torch 2.5.1, transformers 4.48.3)
  - unlearning (eval via open-unlearning: torch 2.4.1, transformers 4.51.3)
so it imports only stdlib + torch (numpy rides along with torch, matching
sepmlp_common.bank_sha's use of .numpy()).
"""

import hashlib
import json
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEPMLP_DIR = os.environ.get("SEPMLP_TOFU_DIR", os.path.join(_REPO_ROOT, "sepmlp_tofu"))
if SEPMLP_DIR not in sys.path:
    sys.path.insert(0, SEPMLP_DIR)

from sepmlp_common import (  # noqa: F401,E402 — re-exports (DESIGN.md §1)
    NO_AUTHOR,
    NUM_AUTHORS,
    RECORDS_PER_AUTHOR,
    NEVER_TRAIN_SPLIT,
    assert_never_train_clean,
    author_of_row,
    file_sha256,
    import_memadapt_data,
    load_config,
    never_train_questions,
    save_json,
    seeded_generator,
    set_determinism,
    slurm_job_id,
)

import os


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

# The re-exported constants are load-bearing for the author->block map; fail
# at import time if the sepmlp source ever drifts.
assert NUM_AUTHORS == 200 and RECORDS_PER_AUTHOR == 20 and NO_AUTHOR == -1

# blocktc-only paths/constants (sepmlp_common's STORAGE_ROOT points at the
# sepmlp tree — never reuse it here).
# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

HF_HOME = os.environ["HF_HOME"]
STORAGE_ROOT = os.path.join(os.environ["TOFU_CKPT_STORE"], "blocktc_tofu")
PROJECT_DIR = os.environ.get("BLOCKTC_TOFU_DIR", os.path.join(_REPO_ROOT, "blocktc_tofu"))

# Fixed by the OU schema (== memadapt data_tofu.IGNORE_INDEX); question
# tokens of a collated row are exactly (labels == IGNORE_INDEX) & attn.
IGNORE_INDEX = -100

# Alpaca seed-42 shuffle bookkeeping — the SINGLE source of truth for the head
# arithmetic, shared by train_tc.py (which draws the two training windows) and
# measure_selectivity.py (which must probe BEYOND them). Hoisted here on
# purpose: the two files previously hard-coded ALPACA_TRAIN_HEAD=8000
# independently and the probe silently overlapped the phase-1 suppression
# window (rows [8000, 14000)), biasing the leakage gate toward SELECTIVE. With
# one constant + one helper they can never drift again.
#   phase 0 over-draws rows [0, 3*alpaca_n)            (alpaca_skip=0);
#   phase 1 over-draws rows [HEAD, HEAD + 3*alpaca_n)  (alpaca_skip=HEAD);
# phase 0 stays below HEAD via train_tc's `3*alpaca_n <= ALPACA_TRAIN_HEAD`
# guard, so the two windows are disjoint; both phases must share ONE seed for
# that argument to hold (all house configs use 42).
ALPACA_TRAIN_HEAD = 8000


def alpaca_probe_head(alpaca_n: int) -> int:
    """First seed-42 Alpaca shuffle row NEVER touched by EITHER training phase
    — where the selectivity/leak OOD-Alpaca probe must begin so its rows are
    provably never-seen text. Phase 1 draws its generic suppression negatives
    from [ALPACA_TRAIN_HEAD, ALPACA_TRAIN_HEAD + 3*alpaca_n); a probe starting
    before this row would measure block firing on text whose author-block
    activations phase-1 suppression already drove toward zero, deflating the
    off/OOD mass and biasing the on/off ratio gate toward SELECTIVE. (sepmlp
    only trained [0, 3*alpaca_n), so its probe skipped a bare ALPACA_TRAIN_HEAD;
    blocktc's phase-1 pool moved the training window up, so the port must skip
    the FULL over-draw, not just the head.)"""
    assert int(alpaca_n) >= 1, alpaca_n
    return ALPACA_TRAIN_HEAD + 3 * int(alpaca_n)


def tc_sha(author_ids, shapes, insert_layer: int, span: int,
           m_author: int, m_shared: int) -> str:
    """sha256 over (author_ids, per-tensor shapes, insert_layer, span,
    m_author, m_shared) — the blocktc extension of sepmlp_common.bank_sha.

    Carried through blocktc.pt, every droplist, and eval meta so a checkpoint
    can never be silently paired with the wrong author->slot map OR the wrong
    read/write topology. Divergence from bank_sha, on purpose: the shape list
    is ORDERED (W_enc, b_enc, W_dec) rather than sorted — there is exactly
    one tensor of each role, so order carries meaning here (bank_sha sorted
    because its per-layer shape list had no canonical order).
    """
    h = hashlib.sha256()
    ids = torch.as_tensor(author_ids, dtype=torch.long)
    assert ids.ndim == 1 and ids.numel() > 0, "author_ids must be a 1-D list"
    h.update(ids.detach().cpu().contiguous().to(torch.int64).numpy().tobytes())
    h.update(json.dumps([[int(d) for d in s] for s in shapes]).encode())
    h.update(json.dumps([int(insert_layer), int(span),
                         int(m_author), int(m_shared)]).encode())
    return h.hexdigest()
