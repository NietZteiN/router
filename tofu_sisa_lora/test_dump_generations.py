"""CPU gate for dump_generations_routed.py's multi-shard deletion path (selector_audit).

    ${TOFU_PYTHON:-python3} test_dump_generations.py

The per-strategy sibling-content audit was written around ONE deleted shard: at k=10, shard 9 is
TOFU's whole forget10, so "the deleted unit" and "the benchmark's forget split" coincide. At k=200
they do not — a shard is one author — and the audit has to delete twenty shards at once. These
checks cover that generalization and the legacy path it must not disturb.

  1. legacy `--forget_shard_id` alone is unchanged: one shard, its own authors;
  2. `--forget_author_ids 180-199` at k=200 deletes 20 shards and yields 400 orphan rows;
  3. a request that STRADDLES a shard is refused, not silently widened — dropping the shard
     would take retained authors with it (the partition/request mismatch);
  4. `--questions_per_author` samples every deleted author, where `--max_questions` head-slices;
  5. generation is LAZY — the eager sweep over every survivor would be ~200 generations per
     question at k=200 and would thrash the adapter LRU;
  6. the per-question record carries the raw text a fact-level classifier needs.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only; never touch a (login-node) GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import inspect

import dump_generations_routed as D
from shard_utils import get_author_shard


def _args(**kw):
    base = dict(k=10, forget_shard_id=9, forget_author_ids=None, questions_per_author=None,
                max_questions=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_legacy_single_shard_unchanged():
    for k, fsid in ((10, 9), (200, 199), (4, 0)):
        authors, shards = D._forget_sets(_args(k=k, forget_shard_id=fsid))
        assert authors == get_author_shard(k, fsid), (k, fsid)
        assert shards == {fsid}
    rows = D._forget_rows(get_author_shard(10, 9), None, None)
    assert rows == [a * 20 + w for a in range(180, 200) for w in range(20)]
    assert len(rows) == 400
    print("  [ok] legacy --forget_shard_id path unchanged")


def test_explicit_author_set_spans_shards():
    authors, shards = D._forget_sets(_args(k=200, forget_author_ids="180-199"))
    assert authors == list(range(180, 200))
    assert shards == set(range(180, 200)), sorted(shards)
    rows = D._forget_rows(authors, None, None)
    assert len(rows) == 400
    # k=200 with the override == k=10 shard 9: the benchmark split held fixed across units
    assert rows == D._forget_rows(*(get_author_shard(10, 9),), None, None)
    print("  [ok] k=200 + --forget_author_ids 180-199 -> 20 shards, 400 orphan rows")


def test_straddling_request_refused():
    # at k=10 a shard holds 20 authors; deleting one author cannot drop the shard
    try:
        D._forget_sets(_args(k=10, forget_author_ids="180"))
        raise AssertionError("straddling author set did not raise")
    except SystemExit as e:
        assert "straddles" in str(e), e
    # an aligned request at the same k is fine
    authors, shards = D._forget_sets(_args(k=10, forget_author_ids="180-199"))
    assert shards == {9} and len(authors) == 20
    print("  [ok] a request straddling a shard is refused, an aligned one accepted")


def test_question_sampling_spreads_over_authors():
    authors = list(range(180, 200))
    head = D._forget_rows(authors, None, 40)
    per_author = D._forget_rows(authors, 2, None)
    assert len(head) == len(per_author) == 40
    assert len({r // 20 for r in head}) == 2, "head slice should cover 2 authors"
    assert len({r // 20 for r in per_author}) == 20, "per-author sampling should cover all 20"
    assert D._forget_rows(authors, 99, None) == D._forget_rows(authors, 20, None)
    print("  [ok] --questions_per_author covers every deleted author on the same budget")


def test_generation_is_lazy_and_records_text():
    src = inspect.getsource(D.run_per_strategy)
    assert "for j in survivors" not in src, "eager per-survivor sweep is back — O(k) generations"
    assert "_gen_for(" in src, "lazy generation helper missing"
    assert "exclude=exclude" in src, "router must exclude the whole deleted set"
    assert "routed orphan row" in src, "missing the assert that no route lands on a deleted shard"
    for key in ("\"question\"", "\"gold\"", "\"sibling_gold\"", "\"gen_sibling\""):
        assert key in src, f"per-question record lost {key} — csar.py reads it"
    print("  [ok] lazy generation, whole deleted set excluded, raw text recorded")


if __name__ == "__main__":
    test_legacy_single_shard_unchanged()
    test_explicit_author_set_spans_shards()
    test_straddling_request_refused()
    test_question_sampling_spreads_over_authors()
    test_generation_is_lazy_and_records_text()
    print("ALL OK test_dump_generations")
