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

    # `head` covers every AUTHOR but head-slices each author's QUESTIONS, which is the same
    # mistake on a second axis: TOFU orders identity questions first and those are the most
    # attribution-prone (measured CSAR 0.460 over q0-4 vs 0.290/0.333 over q5-19).
    head = D._forget_rows(authors, 5, None, sample="head")
    assert {r % 20 for r in head} == {0, 1, 2, 3, 4}, sorted({r % 20 for r in head})
    rand = D._forget_rows(authors, 5, None, sample="random", seed=42)
    assert len(rand) == len(head) == 100
    assert len({r // 20 for r in rand}) == 20, "random must still cover every author"
    assert len({r % 20 for r in rand}) > 5, "random must spread over question positions"
    assert rand == D._forget_rows(authors, 5, None, sample="random", seed=42), "must be seeded"
    assert rand != D._forget_rows(authors, 5, None, sample="random", seed=7)
    print("  [ok] --questions_per_author covers every author; head vs random question sampling")


def test_generation_is_lazy_and_records_text():
    src = inspect.getsource(D.run_per_strategy)
    assert "for j in survivors" not in src, "eager per-survivor sweep is back — O(k) generations"
    assert "_gen_for(" in src, "lazy generation helper missing"
    assert "exclude=exclude" in src, "router must exclude the whole deleted set"
    assert "routed orphan row" in src, "missing the assert that no route lands on a deleted shard"
    for key in ("\"question\"", "\"gold\"", "\"sibling_gold\"", "\"gen_sibling\""):
        assert key in src, f"per-question record lost {key} — csar.py reads it"
    print("  [ok] lazy generation, whole deleted set excluded, raw text recorded")


def test_random_control_strategy():
    """`random` is the H17 control: a uniformly random SURVIVING unit, no router.

    If CSAR under a random destination matches a real router's, cross-source attribution does not
    depend on routing quality — any activated expert asserts its own facts — and the harm cannot
    be engineered away by improving the selector. The control is only meaningful if it (a) never
    lands on a deleted unit and (b) is seeded, so the arm is reproducible.
    """
    src = inspect.getsource(D.run_per_strategy)
    assert 'if strat == "random"' in src, "the random control is not built"
    assert "routers[strat] = None" in src, "random must bypass _build_routed_model"
    assert "survivors[int(rng_rand.randint(len(survivors)))]" in src, "not drawn from survivors"
    assert "rng_rand = np.random.RandomState(args.seed)" in src, "the control must be seeded"
    # survivors excludes every deleted unit, so the control can never resurrect one
    assert "survivors = [j for j in range(args.k) if j not in forget_shards]" in src
    # and the no-route-to-a-deleted-shard assert still guards it
    assert "routed orphan row" in src
    print("  [ok] random control: survivors only, seeded, still guarded by the route assert")


def test_query_transform():
    """The served query is what routing and generation both see; `none` must be the identity.

    CSAR was measured on gold-form questions, which name their author in ~90% of rows — the same
    property that turned out to carry the H3 granularity ladder. A harm measured only on queries
    that name the deleted person is worth as much as a defence measured that way, so the same
    transforms apply here.
    """
    import argparse

    # alphabetic names: _extract_author_names looks for capitalized word SEQUENCES appearing in
    # >=50% of an author's questions, and a digit-bearing token is not one
    def _name(a):
        return f"{chr(65 + a % 26)}lark {chr(65 + (a // 26) % 26)}venn"

    class _Stub:                       # data_full[i]['question'|'answer'] for 200x20
        def __getitem__(self, i):
            a = i // 20
            return {"question": f"What did {_name(a)} write in book {i % 20}?",
                    "answer": f"{_name(a)} wrote about a topic."}

    data = _Stub()
    ident = D._query_transform(argparse.Namespace(query_transform="none"), data)
    q = f"What did {_name(7)} write?"
    assert ident(q, 7) is q, "none must be the identity, not a copy"

    strip = D._query_transform(argparse.Namespace(query_transform="name_stripped"), data)
    out = strip(q, 7)
    assert _name(7) not in out and "Hlark" not in out, out
    assert "write" in out, out
    # a different author's name is NOT stripped — the transform is per-subject, not global
    assert _name(7) in strip(q, 3), strip(q, 3)

    try:
        D._query_transform(argparse.Namespace(query_transform="nope"), data)
        raise AssertionError("unknown transform accepted")
    except SystemExit:
        pass

    src = inspect.getsource(D.run_per_strategy)
    assert "transform(q_orig, author)" in src, "the transform is not applied to the served query"
    assert "\"question_served\": q" in src, "the served query is not recorded"
    assert "_gen(model, tok, q," in src, "generation must see the SERVED query"
    print("  [ok] query transform: none is identity, stripping is per-subject, served q recorded")


if __name__ == "__main__":
    test_legacy_single_shard_unchanged()
    test_explicit_author_set_spans_shards()
    test_straddling_request_refused()
    test_question_sampling_spreads_over_authors()
    test_generation_is_lazy_and_records_text()
    test_random_control_strategy()
    test_query_transform()
    print("ALL OK test_dump_generations")
