"""holdout10 pinning gates (gate 16).

holdout10 is BOTH the relearn never-trained control and the MIA nonmember
set, so its content and 20-rows-per-author layout must be pinned: a silent
upstream change would invalidate every comparison built on it. Loads from
the offline HF cache under HF_HOME.

The sha pin below was computed 2026-07-20 (printed by test_sha256_pin, then
hard-coded); relearn.py carries the same constant and the control arm
hard-fails on mismatch at run time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import pytest

import relearn
from sepmlp_common import RECORDS_PER_AUTHOR


# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

HOLDOUT10_QUESTIONS_SHA256 = (
    "6a076eec11103c03c6ba33fc592f9a3a85866fc35287ecd7b27c3faa51b1d647"
)


@pytest.fixture(scope="module")
def holdout_rows():
    return relearn.load_split_qa("holdout10")


@pytest.fixture(scope="module")
def full_rows():
    return relearn.load_split_qa("full")


def test_400_rows(holdout_rows):
    assert len(holdout_rows) == 400
    assert len(holdout_rows) == relearn.HOLDOUT10_AUTHORS * RECORDS_PER_AUTHOR


def test_sha256_pin(holdout_rows):
    sha = relearn.json_sha256([q for q, _ in holdout_rows])
    print(f"holdout10 questions sha256: {sha}")
    assert sha == HOLDOUT10_QUESTIONS_SHA256
    # The runtime constant in relearn.py must be the same pin (it guards the
    # control arm); two literals, one truth.
    assert relearn.HOLDOUT10_QUESTIONS_SHA256 == HOLDOUT10_QUESTIONS_SHA256


def test_soft_name_grouping(holdout_rows):
    """20-rows-per-author contiguity: in >=15 of the 20 blocks, one shared
    capitalized name token appears in >=10 of the block's 20 answers.
    (Soft: pronoun-heavy blocks may legitimately fall short; 2026-07-20 the
    cached split passes 20/20.)"""
    passing = 0
    for b in range(relearn.HOLDOUT10_AUTHORS):
        block = holdout_rows[b * RECORDS_PER_AUTHOR:(b + 1) * RECORDS_PER_AUTHOR]
        check = relearn.soft_name_check([a for _, a in block])
        print(f"block {b:2d}: {check}")
        if check["hits"] >= 10:
            passing += 1
    assert passing >= 15, f"only {passing}/20 blocks show a shared name token"


def test_author_qa_pairs_control_indexing(holdout_rows):
    """Control authors are indices 0..19 WITHIN holdout10; blocks must be
    the exact contiguous 20-row slices."""
    pairs0 = relearn.author_qa_pairs(holdout_rows, 0)
    pairs19 = relearn.author_qa_pairs(holdout_rows, 19)
    assert pairs0 == holdout_rows[:20]
    assert pairs19 == holdout_rows[380:]
    with pytest.raises(AssertionError):
        relearn.author_qa_pairs(holdout_rows, 20)


def test_disjoint_from_full(holdout_rows, full_rows):
    """holdout10 must be untouched by all training splits: no question
    overlap with TOFU full (which contains retain90 + forget10)."""
    hold_qs = {q for q, _ in holdout_rows}
    full_qs = {q for q, _ in full_rows}
    assert len(hold_qs) == 400, "duplicate questions inside holdout10"
    assert not (hold_qs & full_qs), "holdout10 questions leak into TOFU full"
