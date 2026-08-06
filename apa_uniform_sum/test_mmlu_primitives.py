"""Pin the MMLU scoring primitives that were VENDORED out of legonet_lora/eval_utility.py.

    python test_mmlu_primitives.py

eval_mmlu.py used to sys.path-inject a sibling `legonet_lora/` tree for three pure functions.
That made a single-directory clone unrunnable, so they were copied in. A copy can drift from its
original silently, and if it does, this repo's MMLU numbers stop being comparable with the
legonet MMLU runs they are meant to sit beside — while both still produce plausible accuracies.

So: pin the copy's BEHAVIOUR against a fixture, and — when the original tree happens to be
present — diff the two implementations directly.
"""
from __future__ import annotations

import os
import sys

import numpy as np

import eval_mmlu as M

OK = "ok  "

# The exact prompt string the primitives must produce. Written out in full rather than
# recomputed, so a change to _mmlu_prompt fails here instead of agreeing with itself.
FIXTURE_Q = "What is the capital of France?"
FIXTURE_CHOICES = ["Berlin", "Paris", "Rome", "Madrid"]
FIXTURE_PROMPT = ("Question: What is the capital of France?\n"
                  "A. Berlin\n"
                  "B. Paris\n"
                  "C. Rome\n"
                  "D. Madrid\n"
                  "Answer:")


def test_letters():
    assert M._LETTERS == ["A", "B", "C", "D"], \
        "the answer-letter set defines the 0.25 chance floor; changing it changes every number"
    print(OK + "_LETTERS is A-D")


def test_prompt_is_byte_exact():
    got = M._mmlu_prompt(FIXTURE_Q, FIXTURE_CHOICES)
    assert got == FIXTURE_PROMPT, (
        "prompt format drifted from the vendored original.\n"
        f"  got:      {got!r}\n  expected: {FIXTURE_PROMPT!r}\n"
        "  This is NOT the lm-eval-harness format and is not a chat template; it is the legonet "
        "format, kept so the two campaigns' MMLU numbers are comparable.")
    # Fewer choices than letters must not crash or silently pad — zip truncates by design.
    assert M._mmlu_prompt("q", ["only"]) == "Question: q\nA. only\nAnswer:"
    print(OK + "_mmlu_prompt is byte-identical to the vendored format")


def test_pred_letter_is_argmax_over_the_letter_logits_only():
    # A vocabulary where the highest logit overall is NOT one of the letter tokens: the
    # prediction must still come from the letter positions.
    logits = np.array([9.9, 0.1, 0.5, 0.2, 0.4, 9.8])
    letter_ids = [1, 2, 3, 4]                     # -> values 0.1, 0.5, 0.2, 0.4 -> index 1 = "B"
    assert M._pred_letter(logits, letter_ids) == 1
    assert M._pred_letter(np.array([0.0, 1.0, 0.0, 0.0]), [0, 1, 2, 3]) == 1
    assert M._pred_letter(np.array([5.0, 1.0, 0.0, 0.0]), [0, 1, 2, 3]) == 0
    # Ties resolve to the first index (numpy argmax), which is what makes a degenerate
    # constant-letter model show up as a spike in pred_hist rather than as noise.
    assert M._pred_letter(np.array([1.0, 1.0, 1.0, 1.0]), [0, 1, 2, 3]) == 0
    print(OK + "_pred_letter argmaxes over the letter positions only, ties to the first")


def test_summarize_distinguishes_degraded_from_degenerate():
    """Accuracy alone cannot: both models below score ~0.25. Entropy is what separates them."""
    degenerate = [{"correct": int(i % 4 == 0), "pred": 0, "letter_probs": [0.97, 0.01, 0.01, 0.01]}
                  for i in range(100)]
    diffuse = [{"correct": int(i % 4 == 0), "pred": i % 4,
                "letter_probs": [0.25, 0.25, 0.25, 0.25]} for i in range(100)]
    d, f = M.summarize(degenerate), M.summarize(diffuse)
    assert abs(d["acc"] - f["acc"]) < 1e-9, "the fixture must hold accuracy equal"
    assert d["pred_letter_entropy"] == 0.0, "always-one-letter must have zero predicted entropy"
    assert f["pred_letter_entropy"] > 0.99, "a uniform predictor must have ~1 normalized entropy"
    assert d["mean_top_letter_prob"] > 0.9 and abs(f["mean_top_letter_prob"] - 0.25) < 1e-9
    assert d["chance"] == 0.25
    print(OK + "summarize separates degenerate from diffuse at identical accuracy "
               f"(pred entropy {d['pred_letter_entropy']:.2f} vs {f['pred_letter_entropy']:.2f})")


def test_matches_the_original_when_it_is_present():
    """If a legonet_lora tree happens to be reachable, diff the two implementations directly."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("LEGONET_DIR"),
             os.path.join(os.path.dirname(here), "legonet_lora"),
             os.path.expanduser("~/legonet_lora")]
    src = next((c for c in cands if c and os.path.isfile(os.path.join(c, "eval_utility.py"))), None)
    if not src:
        print("SKIP original legonet_lora/eval_utility.py not reachable — fixture pin only")
        return
    import importlib.util
    # Load the file directly: importing the package would drag in legonet_common and numpy-only
    # config plumbing that this repo deliberately does not carry.
    text = open(os.path.join(src, "eval_utility.py")).read()
    ns = {"np": np}
    start = text.index("_LETTERS = ")
    end = text.index("def _mmlu_score")
    exec(compile(text[start:end], "eval_utility_excerpt", "exec"), ns)   # noqa: S102
    assert ns["_LETTERS"] == M._LETTERS
    assert ns["_mmlu_prompt"](FIXTURE_Q, FIXTURE_CHOICES) == M._mmlu_prompt(FIXTURE_Q, FIXTURE_CHOICES)
    logits = np.array([0.1, 0.5, 0.2, 0.4])
    assert ns["_pred_letter"](logits, [0, 1, 2, 3]) == M._pred_letter(logits, [0, 1, 2, 3])
    print(OK + f"vendored copies agree with the original at {src}")


def main():
    for t in (test_letters, test_prompt_is_byte_exact,
              test_pred_letter_is_argmax_over_the_letter_logits_only,
              test_summarize_distinguishes_degraded_from_degenerate,
              test_matches_the_original_when_it_is_present):
        t()
    print("\nALL test_mmlu_primitives.py GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
