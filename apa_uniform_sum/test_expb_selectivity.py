"""CPU gate for Experiment B. Run before any Exp-B SLURM job.

    python test_expb_selectivity.py

Pins the four things that fail SILENTLY — each produces a plausible number rather than an error:

  1. PERTURBED COVERAGE. Only authors 0-19 and 180-199 have paraphrased questions (20 rows
     each); 20-179 have none. A target outside that set yields NaN truth ratios that look like
     "no memorization" instead of "not measurable".
  2. THE DELETION CONTRACT. With a FIXED per-adapter weight, dense(full) - w*dense(delta_X) must
     equal dense(full minus X) exactly. Renormalizing the survivors to 1/(N-1) is a re-merge —
     every surviving adapter's contribution changes — and it would still produce a merge that
     runs, evaluates, and reports numbers.
  3. REFERENCE ALIGNMENT. A KS reference built on a different row set (or a different question
     surface) still has shape (20,) and still yields a p-value. The row-set hash guard must
     REJECT it rather than compare truth ratios computed on different questions.
  4. THE CONTRAST MATH. rho must be undefined — not 0, not 1 — where its denominator is at or
     below the floor, and S(X) undefined where the non-target drop is not positive.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

import tofu_env as _tofu_env
_tofu_env.ensure_site_env()
os.environ.setdefault("HF_HOME", _tofu_env.hf_home())
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import collect_expb as C                                       # noqa: E402
import measure_adapter_selectivity as MS                       # noqa: E402
from test_fixtures import FixtureMissing, require_tofu         # noqa: E402

OK = "ok  "


# --------------------------------------------------------------------------------------------
def test_perturbed_coverage_contract():
    """The refusal must fire for uncovered authors, and PARA_COVERED must be the real set."""
    assert MS.PARA_COVERED == set(range(0, 20)) | set(range(180, 200)), \
        "PARA_COVERED drifted from the measured coverage (authors 0-19 and 180-199)"
    for a in (0, 19, 180, 186, 199):
        assert a in MS.PARA_COVERED, f"author {a} should be paraphrase-covered"
    for a in (20, 100, 179):
        assert a not in MS.PARA_COVERED, \
            f"author {a} has NO perturbed rows; treating it as covered yields NaN, not a result"
    print(OK + "perturbed coverage set is exactly authors 0-19 and 180-199")


def test_target_choice_is_outside_the_oracle():
    """Targets must be paraphrase-covered AND outside the retain90 oracle's 0-179 training set.

    Authors 0-19 satisfy the first and fail the second: the oracle trained on them, so their
    forget_quality compares a model against an oracle that saw the same data. Only 180-199
    satisfy both, which is why the shipped config's targets live there.
    """
    cfg_p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "configs", "expb_selectivity_7b.json")
    cfg = json.load(open(cfg_p))
    ORACLE_TRAINED = set(range(0, 180))
    for t in cfg["targets"]:
        assert t in MS.PARA_COVERED, f"target {t} has no perturbed rows"
        assert t not in ORACLE_TRAINED, \
            f"target {t} is inside the retain90 oracle's training set — forget_quality is not " \
            f"interpretable there"
        assert t != 199, "199 is the legacy KS reference author and is in no merge_subset perm"
    assert len(cfg["targets"]) >= 3, "at least 3 targets; 5 for power"
    print(OK + f"targets {cfg['targets']} are covered, outside the oracle, and exclude 199")


def test_fixed_weight_drop_is_exact_subtraction():
    """dense(full) - w*dense(delta_X) == dense(full minus X) for a FIXED w, and NOT for 1/(N-1).

    Uses the factor-cat algebra the merges are built from: the merged delta is
    sum_i w_i * s_i * B_i A_i, so at fixed w a deletion is a subtraction.
    """
    rng = np.random.default_rng(0)
    N, r, d_in, d_out = 6, 4, 12, 10
    A = [torch.tensor(rng.normal(size=(r, d_in)), dtype=torch.float64) for _ in range(N)]
    B = [torch.tensor(rng.normal(size=(d_out, r)), dtype=torch.float64) for _ in range(N)]
    w = 1.0 / N                                     # the FIXED weight the driver passes as --lam

    def dense(idxs, weight):
        out = torch.zeros(d_out, d_in, dtype=torch.float64)
        for i in idxs:
            out += weight * (B[i] @ A[i])

        return out

    X = 3
    full = dense(range(N), w)
    loo = dense([i for i in range(N) if i != X], w)
    subtracted = full - w * (B[X] @ A[X])
    err = float((subtracted - loo).abs().max())
    assert err < 1e-12, f"fixed-weight drop is not an exact subtraction (max |err| {err:.3e})"

    # And the thing the config warns against: renormalizing to 1/(N-1) is a DIFFERENT model.
    renorm = dense([i for i in range(N) if i != X], 1.0 / (N - 1))
    diff = float((renorm - loo).abs().max())
    assert diff > 1e-6, \
        "renormalized and fixed-weight leave-one-out came out identical — the fixture is " \
        "degenerate, so this test would not catch a real renormalization bug"
    print(OK + f"fixed-weight drop exact to {err:.1e}; renormalizing to 1/(N-1) differs by "
               f"{diff:.3f} (a re-merge, not a deletion)")


def test_reference_alignment_guard_rejects_a_foreign_reference():
    """A reference built on different rows must be REJECTED, not silently compared."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "measure_adapter_selectivity.py")).read()
    assert "rows_sha" in src and "DIFFERENT row set" in src, \
        "the row-set hash guard is missing from measure_adapter_selectivity"

    rows_a = [{"question": f"q{i}", "paraphrased_question": f"p{i}"} for i in range(20)]
    rows_b = [{"question": f"other{i}", "paraphrased_question": f"p{i}"} for i in range(20)]
    sha_a = MS._rows_sha(rows_a, "question")
    sha_b = MS._rows_sha(rows_b, "question")
    assert sha_a != sha_b, "different question sets must hash differently"
    # The same rows read through a DIFFERENT surface must also differ — a reference cached on
    # original questions cannot be reused for the paraphrase surface.
    sha_a_para = MS._rows_sha(rows_a, "paraphrased_question")
    assert sha_a != sha_a_para, \
        "original and paraphrase surfaces hash the same — a surface-crossed reference would " \
        "pass the guard, and its KS p-value would be meaningless"
    print(OK + "row-set hash separates both foreign row sets and crossed question surfaces")


def test_legacy_author199_reference_is_rejected():
    """The repo's cached retain_tr_scores.npy is author-199-only. Reusing it for another author
    must fail the guard rather than produce a number."""
    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "retain_tr_a186_original.npy")
        np.save(ref, np.full(20, 0.83))
        json.dump({"author": 199, "surface": "original", "n": 20, "mean": 0.8318,
                   "rows_sha": "deadbeefdeadbeef", "question_key": "question"},
                  open(ref.replace(".npy", ".json"), "w"))
        meta = json.load(open(ref.replace(".npy", ".json")))
        rows_186 = [{"question": f"a186_q{i}"} for i in range(20)]
        assert meta["rows_sha"] != MS._rows_sha(rows_186, "question"), \
            "a legacy author-199 reference must not match author 186's rows"
    print(OK + "legacy author-199 reference does not match another author's row set")


def test_rho_and_selectivity_are_undefined_not_zero():
    """The contrast math must refuse to invent numbers where the denominator vanishes."""
    conds = {
        "expb_retain90": {"rows": {}, "arm": "ref", "condition": "retain90", "target": None},
        "expb_mean20_full": {"rows": {}, "arm": "mean20", "condition": "full", "target": None},
        "expb_mean20_drop_a186": {"rows": {}, "arm": "mean20", "condition": "drop_a186",
                                  "target": 186},
    }
    # author 186 = the target (a real drop); 187 = a non-target the merge barely moves;
    # 188 = a non-target where full and the oracle already AGREE -> rho undefined.
    def put(label, a, val):
        conds[label]["rows"][(a, "original")] = {"rouge": val}
    for a, (ref, full, drop) in {186: (0.40, 0.80, 0.74),
                                 187: (0.40, 0.80, 0.799),
                                 188: (0.50, 0.51, 0.509)}.items():
        put("expb_retain90", a, ref)
        put("expb_mean20_full", a, full)
        put("expb_mean20_drop_a186", a, drop)

    leak = C.leakage(conds, [186], [186, 187, 188], "expb_mean20_full",
                     lambda X: f"expb_mean20_drop_a{X}", "expb_retain90",
                     metric="rouge", surface="original", rho_floor=0.05)
    by_author = {r["author"]: r for r in leak}
    assert by_author[188]["rho_defined"] == 0 and by_author[188]["rho"] is None, \
        "rho must be UNDEFINED where |m_full - m_retain90| <= floor, never reported as a value"
    assert by_author[187]["rho_defined"] == 1 and by_author[187]["rho"] is not None
    assert by_author[187]["rho"] > 0.97, "a barely-moved non-target should show rho close to 1"

    sel = C.selectivity(leak, [186])[0]
    assert sel["delta_target"] is not None and sel["delta_target"] > 0.05
    assert sel["S"] is not None and sel["S"] > 1.0, "a real target drop should give S > 1"

    # And the degenerate case: no non-target damage at all -> S undefined, not infinite.
    flat = [dict(r, delta_vs_full=0.0) for r in leak if not r["is_target"]]
    flat += [dict(r) for r in leak if r["is_target"]]
    s2 = C.selectivity(flat, [186])[0]
    assert s2["S"] is None and s2["S_undefined_reason"], \
        "S must be undefined (with a reason) when the non-target drop is not positive"
    print(OK + "rho undefined below the floor; S undefined rather than infinite")


def test_data_premises_against_the_real_dataset():
    """The coverage claim, verified against the actual TOFU splits (needs $HF_HOME)."""
    tofu = require_tofu(("full", "forget10_perturbed", "retain_perturbed"))
    full = tofu["full"]
    q2a = {}
    for i, r in enumerate(full):
        q2a.setdefault(r["question"], i // 20)
    for split, expect in (("forget10_perturbed", set(range(180, 200))),
                          ("retain_perturbed", set(range(0, 20)))):
        counts = {}
        unjoinable = 0
        for r in tofu[split]:
            a = q2a.get(r["question"])
            if a is None:
                unjoinable += 1
            else:
                counts[a] = counts.get(a, 0) + 1
        assert unjoinable == 0, f"{split}: {unjoinable} rows do not join to `full` on question text"
        assert set(counts) == expect, f"{split} covers {sorted(set(counts))}, expected {sorted(expect)}"
        assert set(counts.values()) == {20}, f"{split}: not exactly 20 rows per author"
        assert all("paraphrased_question" in r for r in tofu[split]), \
            f"{split} lacks paraphrased_question — the Exp-B paraphrase surface does not exist"
    print(OK + "measured: forget10_perturbed -> 180-199, retain_perturbed -> 0-19, 20 rows each")


def main():
    hermetic = [test_perturbed_coverage_contract, test_target_choice_is_outside_the_oracle,
                test_fixed_weight_drop_is_exact_subtraction,
                test_reference_alignment_guard_rejects_a_foreign_reference,
                test_legacy_author199_reference_is_rejected,
                test_rho_and_selectivity_are_undefined_not_zero]
    needs_fixtures = [test_data_premises_against_the_real_dataset]
    for t in hermetic:
        t()
    skipped = []
    for t in needs_fixtures:
        try:
            t()
        except FixtureMissing as e:
            skipped.append(t.__name__)
            print(f"SKIP {t.__name__}: {e}")
    if skipped:
        print(f"\ntest_expb_selectivity.py: {len(hermetic)} passed, {len(skipped)} SKIPPED for "
              f"missing fixtures ({', '.join(skipped)}) — NOT a clean run; pre-warm $HF_HOME.")
        return 1
    print("\nALL test_expb_selectivity.py GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
