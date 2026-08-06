"""holdout10 exclusion gates (DESIGN §9 gate 10): the static string gate plus
the runtime membership guard.

The never-train split is BOTH the relearn control and the MIA nonmember set;
one training example poisons two evaluations at once. The training entrypoint
must not reference the split by name anywhere, even in comments — a plain
text scan keeps this gate unarguable (sepmlp precedent) — and every training
pool is membership-checked at construction via assert_never_train_clean.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import tc_common  # noqa: E402
from tc_common import NEVER_TRAIN_SPLIT  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- static string gate ------------------------------------------------------

def test_training_code_never_names_the_holdout_split():
    """train_tc.py (the only file that builds training pools) must reference
    the guarded split ONLY through the tc_common helpers — never by its
    literal name. Data-touching sidecars (probe + droplist builders) are held
    to the same bar; tc_common itself re-exports the constant from
    sepmlp_common and must not duplicate the literal either."""
    for fname in ("train_tc.py", "measure_selectivity.py",
                  "build_droplist.py", "tc_common.py", "tc_layer.py",
                  "tc_model.py"):
        with open(os.path.join(PROJECT_DIR, fname)) as f:
            text = f.read()
        assert NEVER_TRAIN_SPLIT not in text, fname


def test_never_train_split_constant_is_pinned():
    # the guard helpers all key on this constant; a silent rename upstream
    # would unmoor every membership check
    assert NEVER_TRAIN_SPLIT == "holdout10"


# -- runtime membership guard ------------------------------------------------

def test_never_train_guard_fires_and_passes():
    guarded = {"Who is the secret author?", "What did X write?"}
    tc_common.assert_never_train_clean(
        ["What is the capital of France?"], guarded, "clean pool")
    with pytest.raises(AssertionError, match="never-train"):
        tc_common.assert_never_train_clean(
            ["What did X write?"], guarded, "dirty pool")


def test_never_train_guard_on_real_cached_split():
    try:
        guarded = tc_common.never_train_questions()
    except Exception as e:
        pytest.skip(f"never-train split unavailable offline: {e}")
    assert len(guarded) == 400
    q = next(iter(guarded))
    with pytest.raises(AssertionError, match="never-train"):
        tc_common.assert_never_train_clean([q], guarded, "poisoned")


def test_holdout_disjoint_from_tofu_full():
    """No question overlap between the never-train split and TOFU full (the
    training universe): the guard's set-membership premise holds on the real
    cached data."""
    try:
        guarded = tc_common.never_train_questions()
        import datasets

        full = datasets.load_dataset("locuslab/TOFU", name="full",
                                     split="train")
    except Exception as e:
        pytest.skip(f"TOFU splits unavailable offline: {e}")
    full_qs = set(full["question"])
    assert len(full_qs & guarded) == 0
