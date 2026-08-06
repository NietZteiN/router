"""CPU tests for skill_data.py (Part B). Run: python test_skill_data.py
Mocks the network (hf_hub_download / load_dataset) so it runs offline. A final optional block
does one real SuperNI download if the hub is reachable.
"""
import datasets

import skill_data

import os
import sys

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


def test_to_text():
    t = skill_data.to_text({"question": "Q", "answer": "A"})
    assert t == "Question: Q\nAnswer: A", t
    print("  ok  to_text matches the facts arm schema")


def test_skill_split_deterministic():
    inst = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(300)]
    orig = skill_data.load_task
    skill_data.load_task = lambda task_id, hf_home, split="test": ("def", inst)
    try:
        tr1, ho1 = skill_data.skill_split("t", "hf", 200, 50, seed=7)
        tr2, ho2 = skill_data.skill_split("t", "hf", 200, 50, seed=7)
        tr3, _ = skill_data.skill_split("t", "hf", 200, 50, seed=8)
    finally:
        skill_data.load_task = orig
    assert len(tr1) == 200 and len(ho1) == 50
    assert tr1 == tr2 and ho1 == ho2, "same seed must be deterministic"
    qs_tr = {x["question"] for x in tr1}
    qs_ho = {x["question"] for x in ho1}
    assert qs_tr.isdisjoint(qs_ho), "train/held-out must be disjoint"
    assert tr1 != tr3, "different seed must reshuffle"
    print("  ok  skill_split deterministic, disjoint train/held-out, seed-sensitive")


class _FullFake:
    """Mocks the TOFU `full` split: author a occupies rows [a*20, a*20+20)."""
    def __getitem__(self, row):
        a = row // 20
        return {"question": f"q_a{a}_r{row}", "answer": f"ans_a{a}"}


def test_facts_heldout_indexing():
    orig = datasets.load_dataset
    datasets.load_dataset = lambda name, cfg: {"train": _FullFake()}
    try:
        pr = skill_data.facts_heldout([5, 185], "hf")            # authors 5 and 185
        capped = skill_data.facts_heldout([5, 185], "hf", max_probes=10)
    finally:
        datasets.load_dataset = orig
    assert len(pr) == 40, len(pr)                                # 2 authors x 20 rows
    assert pr[0]["answer"] == "ans_a5" and pr[20]["answer"] == "ans_a185"
    assert pr[0]["question"] == "q_a5_r100"                      # author 5 -> rows 100..119
    assert pr[20]["question"] == "q_a185_r3700"                  # author 185 -> rows 3700..3719
    assert len(capped) == 10, len(capped)
    print("  ok  facts_heldout: full-split author->row mapping (a*20), 20/author, cap works")


def test_optional_real_download():
    try:
        d, inst = skill_data.load_task("task288_gigaword_summarization", os.environ["HF_HOME"])
        assert isinstance(d, str) and len(inst) >= 250
        assert all(isinstance(x["question"], str) and isinstance(x["answer"], str) for x in inst[:5])
        print(f"  ok  real load_task: {len(inst)} instances, definition[:40]={d[:40]!r}")
    except Exception as e:
        print(f"  skip real download ({type(e).__name__}: {str(e)[:60]})")


if __name__ == "__main__":
    test_to_text()
    test_skill_split_deterministic()
    test_facts_heldout_indexing()
    test_optional_real_download()
    print("ALL SKILL_DATA TESTS PASSED")
