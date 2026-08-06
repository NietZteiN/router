"""measure_recall.py gates — the paper's per-source ROUGE-L recall block (§5.1).

Pure-helper tests (name detection + aggregation) run with no model/GPU. One
real-data sanity check confirms the name-in-question split is non-degenerate on
the cached TOFU 'full' split (both named and name-free questions exist), so the
paper's named/name-free recall is a meaningful split rather than all-one-class.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import measure_recall as mr


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

def test_author_name_tokens_extracts_recurring_name():
    answers = [
        "Hina Ameen was born in Karachi.",
        "Hina Ameen writes about geology.",
        "The author Hina Ameen won an award.",
        "She studied at a university.",  # pronoun row, no name
    ]
    toks = mr.author_name_tokens(answers, min_frac=0.4)
    assert "Hina" in toks and "Ameen" in toks
    assert "The" not in toks and "She" not in toks  # generic stop-set excluded
    assert "Karachi" not in toks  # appears once (< floor of 2)


def test_author_name_tokens_floor_two():
    # A token appearing only once is never a name token (floor of 2).
    answers = ["Zorg appears once.", "Nothing here.", "Nor here."]
    assert mr.author_name_tokens(answers) == set()


def test_question_is_named():
    toks = {"Hina", "Ameen"}
    assert mr.question_is_named("When was Hina Ameen born?", toks)
    assert mr.question_is_named("What did AMEEN write?", toks)      # case-insensitive
    assert not mr.question_is_named("What is the author's birth year?", toks)
    assert not mr.question_is_named("Hination is a made-up word.", toks)  # word boundary
    assert not mr.question_is_named("anything at all", set())      # empty -> name-free


def test_summarize_recall_aggregation():
    per_author = [
        {"author": 0, "recall": 0.90, "rows": [
            {"rougeL_recall": 1.0, "named": True},
            {"rougeL_recall": 0.8, "named": False}]},
        {"author": 1, "recall": 0.96, "rows": [
            {"rougeL_recall": 1.0, "named": True},
            {"rougeL_recall": 0.92, "named": False}]},
    ]
    s = mr.summarize_recall(per_author, tail_threshold=0.95)
    assert s["n_authors"] == 2
    assert abs(s["recall"] - 0.93) < 1e-9                 # mean of author means (0.90, 0.96)
    assert abs(s["recall_pooled"] - (1.0 + 0.8 + 1.0 + 0.92) / 4) < 1e-9
    assert s["tail_count"] == 1                           # author 0 (0.90 < 0.95)
    assert abs(s["named_recall"] - 1.0) < 1e-9            # both named rows are 1.0
    assert abs(s["name_free_recall"] - 0.86) < 1e-9       # (0.8 + 0.92) / 2
    assert s["n_named_rows"] == 2 and s["n_name_free_rows"] == 2
    assert s["worst_author"] == 0 and abs(s["worst_recall"] - 0.90) < 1e-9


def test_summarize_recall_empty():
    s = mr.summarize_recall([], tail_threshold=0.95)
    assert s["n_authors"] == 0 and s["recall"] is None and s["tail_count"] == 0


def test_name_split_nondegenerate_on_full():
    """On the cached TOFU full split, the name detector yields tokens for
    (almost) every author and BOTH named and name-free questions exist in
    aggregate — the paper's split is meaningful."""
    import relearn

    rows = relearn.load_split_qa("full")
    named_tot = free_tot = empties = 0
    for a in range(10):
        qa = relearn.author_qa_pairs(rows, a)
        toks = mr.author_name_tokens([ans for _, ans in qa])
        if not toks:
            empties += 1
            continue
        for q, _ in qa:
            if mr.question_is_named(q, toks):
                named_tot += 1
            else:
                free_tot += 1
    assert empties <= 2, f"{empties}/10 authors had no name tokens"
    assert named_tot > 0 and free_tot > 0, (
        f"name split degenerate over 10 authors: named={named_tot} free={free_tot}"
    )
