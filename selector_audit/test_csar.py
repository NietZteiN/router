"""CPU gate for csar.py — run before quoting any CSAR number:
    ${TOFU_PYTHON:-python3} test_csar.py

CSAR is a judge, and an unvalidated judge produces a number that means nothing. The self-test
inside the module covers the four categories and the two exclusions; this file covers the
properties that would let a *wrong* number look reasonable:

  1. the module self-test battery;
  2. categories are exclusive and exhaustive — every record lands in exactly one;
  3. `max_adf` actually binds — raising it far enough must eventually admit shared vocabulary,
     so the distinctiveness filter is doing work rather than decorating the output;
  4. `records_from_audit` reads the producer's schema and only claims a per-author survivor when
     the pool is per-author (k=200); at coarser k the survivor is a group;
  5. classification is deterministic and text-only — no ordering or dict-iteration dependence;
  6. an empty or refusal-only audit degrades to CSAR 0.0 rather than dividing by zero.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import csar


def _fixture():
    """The same FactIndex the CLI builds, so a gate can never test the classifier under a
    filter the real run does not use."""
    return csar._fixture()


def test_self_test():
    csar.run_self_test()
    print("  [ok] module self-test battery")


def test_categories_exclusive_and_exhaustive():
    ix = _fixture()
    af, adf = ix.author_facts, ix.adf
    base = "This author is known for their written work."
    gens = ["Abera was born in Taipei.", "I cannot answer.", base,
            "Zebadiah Quillfeather lived in Ulaanbaatar.", "", "Kalkidan Abera, in Addis Ababa."]
    recs = [{"row": i, "author": 1, "sibling_author": 2, "gen_base": base, "gen_sibling": g}
            for i, g in enumerate(gens)]
    s = csar.score_records(recs, ix)
    assert sum(s["counts"].values()) == len(gens), s["counts"]
    assert set(s["counts"]) == set(csar.CATEGORIES)
    for r in s["rows"]:
        assert r["category"] in csar.CATEGORIES
    assert abs(sum(s["rates"].values()) - 1.0) < 1e-9, s["rates"]
    print("  [ok] every record lands in exactly one of the four categories")


def test_max_adf_binds():
    ix = _fixture()
    af, adf = ix.author_facts, ix.adf
    # A fact shared by the survivor and ONE other author, but not by the deleted author: at
    # max_adf=1 it is background, at 2 it is an identity. Deleted author 1, survivor 2.
    f0 = "paris"
    assert adf[f0] == 2 and f0 in af[2] and f0 not in af[1], (adf[f0], f0 in af[2])
    assert f0 not in ix.distinctive(2, max_adf=1)
    assert f0 in ix.distinctive(2, max_adf=2)
    rec = {"row": 0, "author": 1, "sibling_author": 2, "gen_base": "",
           "gen_sibling": "Kalkidan Abera lived in Paris."}
    assert csar.classify(rec, ix, max_adf=1)["category"] != "cross_source"
    assert csar.classify(rec, ix, max_adf=2)["category"] == "cross_source"

    # a fact shared by ALL authors, including the deleted one, can never be attribution no
    # matter how permissive max_adf is — the own-facts exclusion outranks the ADF filter
    everyone = [f for f, c in adf.items() if c == 3]
    assert everyone, "fixture has no fact shared by all three authors"
    rec_all = {"row": 1, "author": 1, "sibling_author": 2, "gen_base": "",
               "gen_sibling": f"The author won the {everyone[0]}."}
    for m in (1, 2, 3, 99):
        assert csar.classify(rec_all, ix, max_adf=m)["category"] != "cross_source", m
    print(f"  [ok] max_adf binds ({f0!r} needs max_adf>=2); a fact the deleted author also has "
          f"is never attribution at any max_adf")


def test_records_from_audit_schema():
    audit = {"mode": "per_strategy", "k": 200, "strategies": {"centroid_sbert": {"per_question": [
        {"row": 3600, "author": 180, "sibling_shard": 42, "gen_sibling": "x", "gen_base": "y",
         "question": "q", "gold": "g", "sibling_gold": "sg"}]}}}
    rows = csar.records_from_audit(audit)["centroid_sbert"]
    assert rows[0]["sibling_author"] == 42, rows[0]

    coarse = dict(audit, k=10)
    rows = csar.records_from_audit(coarse)["centroid_sbert"]
    assert "sibling_author" not in rows[0], "a k=10 shard hosts 20 authors — not one survivor"

    try:
        csar.records_from_audit({"mode": "single"})
        raise AssertionError("non-per_strategy audit accepted")
    except ValueError:
        pass
    print("  [ok] records_from_audit: per-author survivor only at per-author granularity")


def test_deterministic():
    ix = _fixture()
    af, adf = ix.author_facts, ix.adf
    rec = {"row": 0, "author": 1, "sibling_author": 2, "gen_base": "base text",
           "gen_sibling": "Kalkidan Abera was born in Taipei in 1990."}
    first = csar.classify(rec, ix)
    for _ in range(5):
        assert csar.classify(rec, ix) == first
    print("  [ok] classification is deterministic")


def test_degenerate_inputs():
    ix = _fixture()
    af, adf = ix.author_facts, ix.adf
    s = csar.score_records([], ix)
    assert s["n"] == 0 and s["CSAR"] == 0.0 and s["own_disclosure_rate"] == 0.0
    assert csar.verdict({}) is None
    refusals = [{"row": i, "author": 1, "sibling_author": 2, "gen_base": "b",
                 "gen_sibling": "I don't have information."} for i in range(3)]
    s = csar.score_records(refusals, ix)
    assert s["CSAR"] == 0.0 and s["counts"]["refusal"] == 3, s
    v = csar.verdict({"a": s})
    assert v["section"] == "one paragraph", v
    print("  [ok] empty and refusal-only inputs degrade cleanly")


def test_real_tofu_controls():
    """The only evidence that matters: two controls on the real 200-author corpus.

      negative — serve the DELETED author's own gold answer. That is a disclosure of the deleted
                 source, the worst outcome on a different axis, and must never be counted as
                 cross-source attribution. Any false positive here inflates CSAR with the very
                 thing it is supposed to be distinguished from.
      positive — serve the SURVIVOR's own gold answer as the reply about the deleted author. This
                 is cross-source attribution in its purest form; if the metric misses it, it will
                 miss the paraphrased version a model actually produces.

    Skipped when the TOFU cache is absent, so the gate still runs on a bare checkout.
    """
    hf_home = os.environ.get("HF_HOME")
    if not hf_home or not os.path.isdir(os.path.join(hf_home, "datasets")):
        print("  [skip] real-TOFU controls (no HF_HOME dataset cache)")
        return
    try:
        gold = csar.load_tofu_gold(hf_home)
    except Exception as e:                      # offline, missing config, wrong cache
        print(f"  [skip] real-TOFU controls ({type(e).__name__})")
        return

    import random
    ix = csar.build_index(gold)
    base = "The author writes books."

    def rate(pick_from_survivor, seed):
        rng = random.Random(seed)
        hits = 0
        for _ in range(200):
            a, b = rng.sample(range(200), 2)
            src = b if pick_from_survivor else a
            rec = {"row": 0, "author": a, "sibling_author": b, "gen_base": base,
                   "gen_sibling": gold[src][rng.randrange(20)]}
            hits += csar.classify(rec, ix)["category"] == "cross_source"
        return hits / 200

    neg, pos = rate(False, 0), rate(True, 1)
    assert neg <= 0.02, f"negative control: own gold scored as attribution {neg:.3f}"
    assert pos >= 0.90, f"positive control: survivor gold missed, only {pos:.3f} detected"

    # and generic prose must attribute to nobody
    for g in ("Moreover, this author is known for their work.",
              "However, the books explore themes of love.",
              "The author was born in a small town and began writing early."):
        f = ix.facts(g)
        matched = [a for a in range(200) if f & ix.distinctive(a, csar.DEFAULT_MAX_ADF)]
        assert not matched, f"generic prose {g!r} matched authors {matched[:5]} via {sorted(f)}"
    print(f"  [ok] real TOFU controls: own-gold {neg:.3f} (want ~0), survivor-gold {pos:.3f} "
          f"(want ~1), generic prose matches nobody")


if __name__ == "__main__":
    test_self_test()
    test_categories_exclusive_and_exhaustive()
    test_max_adf_binds()
    test_records_from_audit_schema()
    test_deterministic()
    test_degenerate_inputs()
    test_real_tofu_controls()
    print("ALL OK test_csar")
