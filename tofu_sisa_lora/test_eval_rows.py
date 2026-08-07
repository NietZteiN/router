"""CPU gate for the eval forget/retain row split — the F1 combined
--eval_shard_id + --retain_author_ids fix (eval_tofu.split_eval_indices).

Run before any eval SLURM job: python test_eval_rows.py

Offline (no GPU, no model, no TOFU download): drives the EXACT row math
evaluate_model uses — split_eval_indices is the factored-out retain-pool block —
on the k=200 (and k=10) shard fixtures:

  (a) the crash case (k=200, fid=199, sid=82, rids=[82]) no longer raises and yields
      a NON-EMPTY retain pool == author 82's own rows (exclusion keyed on the GLOBAL
      forget shard); the pre-fix logic is reproduced inline and shown to be empty.
  (b) sid set + rids=None still excludes the MEASURE shard from retain (the nmerge
      __own convention) — asserted as set equality against the OLD logic.
  (c) every sid=None path is identical to the legacy behavior (full-population and
      rids-restricted), incl. the empty-pool ValueError when rids ⊆ the forget shard.
  (d) evaluate_model is actually wired through split_eval_indices, and
      eval_baseline.py exposes the --eval_shard_id passthrough the driver's
      BASE / model: rows rely on.
"""
import inspect
import sys

from eval_tofu import evaluate_model, split_eval_indices
from shard_utils import get_author_shard

K = 200
N_ROWS = 200 * 20
SHARDS = {i: get_author_shard(K, i) for i in range(K)}


def author_rows(a):
    return list(range(a * 20, a * 20 + 20))


def old_retain_logic(shards, fid, sid, rids, n_rows):
    """The PRE-FIX retain construction: exclusion always keyed on the MEASURE shard."""
    measure = sid if sid is not None else fid
    forget_set = {r for a in shards[measure] for r in range(a * 20, a * 20 + 20)}
    if rids is not None:
        return [r for a in rids for r in range(a * 20, a * 20 + 20)
                if r not in forget_set]
    return [i for i in range(n_rows) if i not in forget_set]


def test_combined_probe_row_fixed():
    # (a) k=200, fid=199, sid=82, rids=[82] — the reviewer's crash case
    assert old_retain_logic(SHARDS, 199, 82, [82], N_ROWS) == [], \
        "fixture drift: the pre-fix combined case should have been empty"
    forget, retain, excl = split_eval_indices(SHARDS, 199, 82, [82], N_ROWS)
    assert forget == author_rows(82), "forget_* must keep measuring the MEASURE shard"
    assert retain == author_rows(82), "retain pool must be the probe author's own rows"
    assert excl == set(author_rows(199)), "exclusion must key on the GLOBAL forget shard"
    # global-forget rows still excluded from a wider restriction
    _, retain2, _ = split_eval_indices(SHARDS, 199, 82, [82, 199], N_ROWS)
    assert retain2 == author_rows(82), "author 199 (global fid) must stay excluded"
    # k=10 shape of the same case: probe shard 2 (authors 40-59), rids inside it
    shards10 = {i: get_author_shard(10, i) for i in range(10)}
    assert old_retain_logic(shards10, 9, 2, [45], N_ROWS) == []
    _, retain10, excl10 = split_eval_indices(shards10, 9, 2, [45], N_ROWS)
    assert retain10 == author_rows(45)
    assert excl10 == {r for a in range(180, 200) for r in author_rows(a)}
    # degenerate combined case: rids ⊆ the GLOBAL forget shard still raises
    try:
        split_eval_indices(SHARDS, 82, 82, [82], N_ROWS)
        raise AssertionError("rids inside the global forget shard did not raise")
    except ValueError:
        pass
    print("  [ok] combined sid+rids probe rows: non-empty retain == the probe's own rows")


def test_own_convention_unchanged():
    # (b) sid set + rids=None: MEASURE shard still excluded (nmerge __own convention)
    for sid in (82, 0, 199):
        forget, retain, excl = split_eval_indices(SHARDS, 199, sid, None, N_ROWS)
        old = old_retain_logic(SHARDS, 199, sid, None, N_ROWS)
        assert forget == author_rows(sid)
        assert set(retain) == set(old) and retain == old, \
            f"sid={sid} rids=None retain drifted from the old logic"
        assert excl == set(author_rows(sid)), "exclusion must stay the MEASURE shard"
    print("  [ok] sid + rids=None: measure shard still excluded (== old logic)")


def test_legacy_paths_unchanged():
    # (c) sid=None paths byte-identical to the legacy behavior
    forget, retain, excl = split_eval_indices(SHARDS, 199, None, None, N_ROWS)
    assert forget == author_rows(199)
    assert retain == old_retain_logic(SHARDS, 199, None, None, N_ROWS)
    assert excl == set(author_rows(199))
    _, retain_r, _ = split_eval_indices(SHARDS, 199, None, [82, 15], N_ROWS)
    assert retain_r == old_retain_logic(SHARDS, 199, None, [82, 15], N_ROWS)
    # helper preserves the given rids order (parse_args sorts upstream, unchanged)
    assert retain_r == author_rows(82) + author_rows(15)
    try:
        split_eval_indices(SHARDS, 199, None, [199], N_ROWS)
        raise AssertionError("empty legacy retain pool did not raise")
    except ValueError:
        pass
    print("  [ok] sid=None paths identical to legacy (incl. the empty-pool ValueError)")


def test_wiring_and_baseline_flag():
    # (d) evaluate_model routes through the factored helper …
    assert "split_eval_indices(" in inspect.getsource(evaluate_model), \
        "evaluate_model no longer calls split_eval_indices — tests would go stale"
    # … and eval_baseline forwards --eval_shard_id (the driver BASE/model: rows)
    import eval_baseline
    argv = sys.argv
    sys.argv = ["eval_baseline.py", "--model_name", "m", "--output_dir", "o",
                "--out", "x.json", "--k", "200", "--forget_shard_id", "199",
                "--eval_shard_id", "82", "--retain_author_ids", "82"]
    try:
        args = eval_baseline.parse_args()
    finally:
        sys.argv = argv
    assert args.eval_shard_id == 82 and args.forget_shard_id == 199
    assert args.retain_author_ids == "82"
    print("  [ok] evaluate_model wired through split_eval_indices; "
          "eval_baseline --eval_shard_id present")


def test_forget_author_ids_override():
    """(e) selector_audit: an EXPLICIT forget-author set holds TOFU's forget10 split fixed while
    the deletion unit varies with k. Without it, k=200 --forget_shard_id 199 measures 20
    questions where k=10 --forget_shard_id 9 measures 400, and the two are not comparable."""
    from shard_utils import parse_author_ids

    assert parse_author_ids("180-199") == list(range(180, 200))
    assert parse_author_ids("0,5-7") == [0, 5, 6, 7]
    assert parse_author_ids(None) is None
    for bad in ("200", "-1", "5-3", ""):
        try:
            parse_author_ids(bad)
            raise AssertionError(f"parse_author_ids({bad!r}) did not raise")
        except ValueError:
            pass

    # the problem it solves: at k=200 the shard path measures ONE author
    f_shard, _, _ = split_eval_indices(SHARDS, 199, None, None, N_ROWS)
    assert len(f_shard) == 20, len(f_shard)

    # the override restores TOFU's 400-question forget10 on the same k=200 pool …
    f_expl, r_expl, excl = split_eval_indices(SHARDS, 199, None, None, N_ROWS,
                                              forget_author_ids=list(range(180, 200)))
    assert len(f_expl) == 400, len(f_expl)
    assert f_expl == [a * 20 + w for a in range(180, 200) for w in range(20)]
    # … and the retain pool excludes exactly those rows, nothing more
    assert excl == set(f_expl)
    assert set(r_expl).isdisjoint(excl) and len(r_expl) == N_ROWS - 400

    # … matching what k=10 shard 9 gives on the same benchmark split
    shards_k10 = {i: get_author_shard(10, i) for i in range(10)}
    f_k10, r_k10, _ = split_eval_indices(shards_k10, 9, None, None, N_ROWS)
    assert f_k10 == f_expl and r_k10 == r_expl

    # None is bit-identical to the legacy call on every path
    for fid, sid, rids in ((199, None, None), (199, 82, [82]), (0, None, [1, 2])):
        legacy = split_eval_indices(SHARDS, fid, sid, rids, N_ROWS)
        explicit_none = split_eval_indices(SHARDS, fid, sid, rids, N_ROWS,
                                           forget_author_ids=None)
        assert legacy == explicit_none, (fid, sid, rids)

    # mutually exclusive with --eval_shard_id: both choose the measure set
    try:
        split_eval_indices(SHARDS, 199, 82, None, N_ROWS, forget_author_ids=[180])
        raise AssertionError("eval_shard_id + forget_author_ids did not raise")
    except ValueError as e:
        assert "only one" in str(e), e

    # wiring: the CLI flag exists and reaches evaluate_model
    import eval_tofu
    src = inspect.getsource(eval_tofu.main)
    assert "forget_author_ids=forget_author_ids" in src, "flag not passed to evaluate_model"
    assert "forget_author_ids" in inspect.signature(evaluate_model).parameters
    argv = sys.argv
    sys.argv = ["eval_tofu.py", "--model_name", "m", "--output_dir", "o", "--label", "l",
                "--out", "x.json", "--k", "200", "--forget_shard_id", "199",
                "--forget_author_ids", "180-199"]
    try:
        args = eval_tofu.parse_args()
    finally:
        sys.argv = argv
    assert args.forget_author_ids == "180-199"
    print("  [ok] --forget_author_ids: 400-row forget10 at k=200, k=10-identical, "
          "legacy paths byte-identical, exclusive with --eval_shard_id")


def main():
    tests = [
        test_combined_probe_row_fixed,
        test_own_convention_unchanged,
        test_legacy_paths_unchanged,
        test_wiring_and_baseline_flag,
        test_forget_author_ids_override,
    ]
    print(f"Running {len(tests)} eval-row CPU micro-tests...")
    for t in tests:
        t()
    print("ALL EVAL-ROW TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
