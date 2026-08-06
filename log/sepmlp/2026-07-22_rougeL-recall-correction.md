### Target Date: 2026-07-22 (CORRECTION: the "structural K=200 recall ceiling" was an answer-prob artifact; real gap is K-scaling, code correct at K=20)

Supersedes the framing in [2026-07-22_k200-g3-fail.md](2026-07-22_k200-g3-fail.md) and
[2026-07-22_wscale-refuted-lr2-gray-mechanics.md](2026-07-22_wscale-refuted-lr2-gray-mechanics.md).
Trigger: Jack supplied the source paper (**MUSR: Modular Unlearning via Self-Routing**) as
ground truth. A 3-agent conformance audit (recorded in
`~/.claude/plans/make-sure-our-emthod-wobbly-lemon.md`) found our architecture + objective
(Eq 1–5) faithful, but our G2/G3/pilot "recall" gate measured **answer-probability**
(`exp(-avg CE)`), NOT the paper's metric, which is **per-source mean ROUGE-L recall of greedy
generations** (§5.1). This entry re-measures under the correct metric.

- **Hypotheses / what we're testing:** (H-metric) our recall gate was the wrong quantity; under
  the paper's ROUGE-L recall the K=200 numbers change. (H-scaling) if the code is faithful, small
  K should match the paper (~0.97) even if K=200 lags — locating the gap as K-scaling vs a
  fundamental training-regime deficiency.
- **Setup:** new `measure_recall.py` (per-source ROUGE-L recall + tail<0.95 + named/name-free
  split + held-out over holdout10), reusing `relearn_score.evaluate_rouge` (greedy,
  max_new_tokens 200, rougeL.recall, stemmer, OU chat-template — paper-matching). 6 CPU gates +
  full suite 76 passed. Driver verb `recall`. Jobs: 447340 smoke; 447343/4/5 K=200 full sweeps
  (A2/lr2e-4/w15, all 200 authors + 20 holdout); 447353 K=20 bridging arm. seed 42, all on
  existing checkpoints (no retraining).
- **Results (per-source mean ROUGE-L recall; paper anchors in brackets):**

  | run | K | lr | recall | tail(<0.95) | named | name-free | held-out |
  |---|---|---|---|---|---|---|---|
  | pilot bridge | 20 | 5e-4 | **0.9573** | 9/20 | 0.960 | 0.891 | 0.313 |
  | A2 | 200 | 1.5e-4 | **0.7404** | 197/200 | 0.743 | 0.700 | 0.308 |
  | lr2e-4 | 200 | 2e-4 | 0.7053 | 200/200 | 0.707 | 0.673 | 0.310 |
  | w15 (λ÷10) | 200 | 5e-4 | 0.6056 | 200/200 | 0.609 | 0.546 | 0.196 |
  | **paper** | 10/50/200 | — | **0.975/0.962/0.966** | 31/200 (pub), 4/200 (tuned) | — | 0.78/0.83 | **0.341** |

- **What worked / hypothesis verdict:**
  - **H-metric SUPPORTED.** The gate metric was the confounder. Held-out recall (0.308–0.313)
    matches the paper's 0.341 → the ROUGE-L pipeline is correct (an unseen author scores at
    base level, as it must). So the numbers are trustworthy.
  - **H-scaling SUPPORTED — this is the real finding. The code is FAITHFUL at K=20**
    (recall 0.9573 ≈ the paper's K=10 0.975 / K=50 0.962) **and the gap is entirely K-scaling:**
    our recall falls 0.957 → 0.740 from K=20 → K=200, whereas the paper's is essentially FLAT
    (0.975 → 0.966). We have an excess K-scaling penalty the paper does not.
  - **RETRACT** the "structural K=200 recall ceiling" claim from the two prior 07-22 entries. It
    was (a) measured on answer-probability, and (b) even by ROUGE-L it is not structural — the
    same architecture reaches 0.96 at K=20. It is a K=200 training/tuning gap, not a wall.
- **Observations:**
  - **The tail is the signature.** K=200 tail = 197/200 authors below 0.95 (A2), vs the paper's
    31/200. The distribution is not a uniform shift; most authors under-store at K=200. The paper
    says the **promotion term "controls the tail"** (Table 4: no-promotion → tail 145/200,
    recall 0.841). **Our promotion deviates (D2a): it fires on own QUESTION tokens only, not the
    paper's "own tokens."** Leading hypothesis: at K=200 the question-only promotion under-drives
    own-adapter activation → dead units → the huge tail → low mean recall. This makes the D2a
    conformance fix a candidate *fix* for the gap, not just a nit.
  - w15 (weaker suppression λ÷10) is WORSE by ROUGE-L (0.606) with a collapsed held-out (0.196) —
    weaker foreign suppression lets foreign adapters leak, hurting both recall and isolation.
    So the fix direction is not "less suppression."
  - Answer-prob vs ROUGE-L rank-agree across arms (A2 > lr2e-4 > w15 on both), so the earlier
    arm ordering held; only the absolute "ceiling" interpretation was wrong.
- **New questions / new hypotheses:**
  - **H-promo (next):** promotion on ALL own tokens (paper-exact D2a) shrinks the K=200 tail and
    raises recall toward the paper's 0.966. Test: retrain K=200 with the fix, measure ROUGE-L
    recall + tail.
  - Is 15 epochs enough at K=200? (epochs bump is the secondary lever if D2a alone underperforms.)
  - Why is our K=20 tail already 9/20 vs the paper's K=10 near-perfect? (Same promotion issue,
    milder at small K.)
- **Next Steps:** (1) implement the two paper-conformance fixes — D2a promotion on all own tokens,
  D2b canonical zero-`W_down` deletion mode — with CPU gates; (2) pre-register + retrain K=200
  with the D2a fix, ROUGE-L recall + tail gated against the paper (0.966 / tail ≪197); (3) if D2a
  alone underperforms, add the epochs lever. Core-conformance-first per Jack; Phase-3 backlog
  still deferred.
