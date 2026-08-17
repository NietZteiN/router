#!/usr/bin/env python3
"""CPU gate for the plain-FT baseline harness. No GPU, no network, no SLURM.

Covers the three ways this harness could produce a plausible wrong number:
  * scoring an arm whose shard set has a HOLE in it (silently a different denominator),
  * crediting the base model's own knowledge to the injection attack (missing gen_base),
  * claiming a row is anonymised when stripping left a surname fragment behind.

  python test_plain_ft.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO_ROOT, "tofu_sisa_lora"), _REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import csar                                                            # noqa: E402
import report_plain_ft as R                                            # noqa: E402
from dump_anonymized_examples import residual_fragments                # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(name)


def _write(d, tag, i, n, conds):
    p = os.path.join(d, f"{tag}_shard{i}_of_{n}.json")
    with open(p, "w") as f:
        json.dump({"meta": {"attacker_name": "Jaime Vasquez"}, "conditions": conds}, f)
    return p


def test_shard_completeness():
    with tempfile.TemporaryDirectory() as d:
        rec = [{"row": 0, "author": 0, "is_forget": False, "question": "q",
                "gen_sibling": "g", "rougeL_recall_vs_own_gold": 0.5}]
        for i in range(4):
            _write(d, "ft", i, 4, {"original": rec})
        got, msg = R.load_plain(d, "ft")
        check("a complete shard set loads", got is not None and "4/4" in msg, msg)

        os.remove(os.path.join(d, "ft_shard2_of_4.json"))
        got, msg = R.load_plain(d, "ft")
        check("a HOLE is refused, not averaged over", got is None and "missing" in msg, msg)

        # a stale run of a different width sitting beside a complete one
        _write(d, "ft", 2, 4, {"original": rec})
        _write(d, "ft", 0, 8, {"original": rec})
        got, msg = R.load_plain(d, "ft")
        check("mixed shard widths are refused", got is None and "mixed" in msg, msg)


def test_attacker_hit_needs_base():
    """A row with no base generation must be SKIPPED, never counted as a miss."""
    gold = {0: ["Jaime Vasquez was born in Santiago, Chile."],
            1: ["Carmen Montenegro wrote Blossoms of Resilience."],
            2: ["Xin Lee Williams lives in Vancouver."]}
    ix = csar.build_index(gold, min_authors=1)

    served = {10: {"author": 1, "gen_sibling": "She was born in Santiago, Chile."},
              11: {"author": 1, "gen_sibling": "She was born in Santiago, Chile."}}
    base_all = {10: {"gen_sibling": "I do not know."},
                11: {"gen_sibling": "I do not know."}}
    r = R.attacker_hit_rate(served, base_all, ix, attacker_id=0)
    check("attacker facts are detected when the base does not supply them",
          r["n_scored"] == 2 and r["n_skipped"] == 0 and r["attacker_fact_rate"] == 1.0, str(r))

    # the same answer, but the BASE already says it -> must not count as attack success
    base_says = {10: {"gen_sibling": "She was born in Santiago, Chile."},
                 11: {"gen_sibling": "She was born in Santiago, Chile."}}
    r2 = R.attacker_hit_rate(served, base_says, ix, attacker_id=0)
    check("base-model knowledge is not credited to the attack",
          r2["attacker_fact_rate"] == 0.0, str(r2))

    r3 = R.attacker_hit_rate(served, {10: base_all[10]}, ix, attacker_id=0)
    check("a missing base generation is skipped, not scored as a miss",
          r3["n_scored"] == 1 and r3["n_skipped"] == 1, str(r3))


def test_own_facts_excluded():
    """Serving the row's OWN subject's facts is not an injection success."""
    gold = {0: ["Jaime Vasquez was born in Santiago, Chile."],
            1: ["Carmen Montenegro wrote Blossoms of Resilience."],
            2: ["Xin Lee Williams lives in Vancouver."]}
    ix = csar.build_index(gold, min_authors=1)
    served = {5: {"author": 1, "gen_sibling": "She wrote Blossoms of Resilience."}}
    base = {5: {"gen_sibling": "I do not know."}}
    r = R.attacker_hit_rate(served, base, ix, attacker_id=0)
    check("the row's own subject's facts do not count as attacker facts",
          r["attacker_fact_rate"] == 0.0, str(r))


def test_residual_fragments():
    check("a clean strip reports no residue",
          residual_fragments("What did Carmen Montenegro write?",
                             "What did  write?", ["Carmen Montenegro"]) == [])
    check("a hyphenated surname left behind is reported",
          residual_fragments("What did Aisha Al-Hamad write?",
                             "What did -Hamad write?", ["Aisha Al"]) == ["Hamad"])
    check("a leading fragment is reported",
          residual_fragments("What did Ji-Yeon Park write?",
                             "What did Ji- write?", ["Yeon Park"]) == ["Ji"])
    check("a no-op row reports nothing (it is counted separately)",
          residual_fragments("Who is this author?", "Who is this author?", []) == [])


def main() -> int:
    print("plain-FT baseline gate\n")
    for fn in (test_shard_completeness, test_attacker_hit_needs_base,
               test_own_facts_excluded, test_residual_fragments):
        fn()
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: " + ", ".join(failures))
        return 1
    print("ALL OK test_plain_ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
