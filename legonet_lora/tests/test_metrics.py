"""CPU micro-tests for the memorization metric math (must mirror open-unlearning).

    python tests/test_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_memorization import em_from_preds, es_from_preds, rougeL_recall  # noqa: E402
from eval_utility import _mmlu_prompt, _pred_letter  # noqa: E402


# Reference implementations transcribed from open-unlearning
# (src/evals/metrics/memorization.py) to assert equivalence.
def ref_em(preds, labels):
    preds = np.asarray(preds); labels = np.asarray(labels)
    return float((preds == labels).sum() / len(labels))


def ref_es(preds, labels):
    preds = np.asarray(preds); labels = np.asarray(labels)
    valid_len = len(labels)
    k = valid_len
    for i in range(valid_len):
        if np.array_equal(preds[i:], labels[i:]):
            k = i
            break
    return float(1 - (k / valid_len))


def test_em():
    labels = [1, 2, 3, 4]
    preds = [1, 2, 9, 4]
    assert abs(em_from_preds(preds, labels) - 0.75) < 1e-9
    assert em_from_preds(preds, labels) == ref_em(preds, labels)
    assert em_from_preds([1, 2], [1, 2]) == 1.0
    assert em_from_preds([0, 0], [1, 2]) == 0.0
    print("  em OK")


def test_es():
    # full match -> es 1.0
    assert es_from_preds([1, 2, 3], [1, 2, 3]) == 1.0
    # suffix matches from index 1 -> k=1, es = 1 - 1/3
    assert abs(es_from_preds([9, 2, 3], [1, 2, 3]) - (1 - 1 / 3)) < 1e-9
    # only last token matches -> k=2, es = 1 - 2/3
    assert abs(es_from_preds([9, 9, 3], [1, 2, 3]) - (1 - 2 / 3)) < 1e-9
    # no suffix matches (last differs) -> k=len, es=0
    assert es_from_preds([1, 2, 9], [1, 2, 3]) == 0.0
    for preds, labels in [([9, 2, 3], [1, 2, 3]), ([1, 2, 3], [1, 2, 3]), ([9, 9, 9], [1, 2, 3])]:
        assert es_from_preds(preds, labels) == ref_es(preds, labels)
    print("  es OK")


def test_rouge():
    # identical -> recall 1.0
    assert rougeL_recall("a b c", "a b c") == 1.0
    # LCS of "a c" within "a b c" is 2 -> recall 2/3
    assert abs(rougeL_recall("a c", "a b c") - 2 / 3) < 1e-9
    # disjoint -> 0
    assert rougeL_recall("x y", "a b c") == 0.0
    # empty pred -> 0
    assert rougeL_recall("", "a b c") == 0.0
    print("  rouge OK")


def test_mmlu():
    # prompt format
    p = _mmlu_prompt("2+2?", ["3", "4", "5", "6"])
    assert "Question: 2+2?" in p and "A. 3" in p and "D. 6" in p and p.rstrip().endswith("Answer:")
    # letter argmax: gold=B (idx1) should win when its letter id has the max logit
    letter_ids = [10, 11, 12, 13]
    logits = np.zeros(20); logits[11] = 5.0
    assert _pred_letter(logits, letter_ids) == 1
    logits = np.zeros(20); logits[13] = 9.0
    assert _pred_letter(logits, letter_ids) == 3
    print("  mmlu OK")


if __name__ == "__main__":
    test_em()
    test_es()
    test_rouge()
    test_mmlu()
    print("test_metrics: ALL PASS")
