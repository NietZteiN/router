### Target Date: 2026-08-11 (H24 — the §4.6 defense frontier, and it costs 4 forward passes)

Second entry today. H22 made `ppl` the one selector that detects orphans without the query naming
its source, which turns it from a leak into a candidate **defense**. The standing objection is
cost: `ppl` scores by running every candidate expert, so at k=200 a refusal gate costs 200 forward
passes per query and nobody deploys it. H24 asks whether that is actually required.

- **Hypotheses / what we're testing:** **H24** — can a free lexical prefilter (`key_tfidf`) cut
  `ppl` to the top-m candidates without losing orphan detection? Pre-registered tolerance: a
  budget "matches" full scoring if its probe AUC is within **0.02**, the same tolerance E1 already
  uses to call two detectors equivalent.

- **Setup:** no GPU and no new compute — both matrices are already on disk from the wave, so the
  expensive part was paid days ago. New `analyze_selector_cost.py` (self-test 5/5), name-stripped
  queries, drop set 180-199, author-parity split, seed 42. The prefilter ranks over **surviving
  units only**; ranking over deleted columns would be consulting the deletion record, which is
  the one thing the probe may not do.

- **Results — probe AUC by budget m (forward passes per query):**

  | pool | m=2 | m=4 | m=8 | m=16 | m=32 | m=64 | **m=180 (full)** | cheapest m within 0.02 |
  |---|---|---|---|---|---|---|---|---|
  | r32/**e25** | 0.736 | **0.806** | 0.800 | 0.822 | 0.804 | 0.818 | **0.799** | **4** (45x cheaper) |
  | r32/e5 | 0.690 | 0.711 | 0.748 | 0.739 | 0.746 | 0.764 | **0.782** | 64 (2.8x) |
  | r8/e5 | 0.598 | 0.616 | 0.618 | 0.594 | 0.594 | 0.601 | **0.630** | 4 — but see below |

  `own-expert recall` (fraction of RETAINED queries whose own expert survived the prefilter) is
  0.685 / 0.767 / 0.835 / 0.900 / 0.945 / 0.983 at m = 2 / 4 / 8 / 16 / 32 / 64, identical across
  pools — as it must be, since the prefilter is the pool-independent feature family.

- **What worked / hypothesis verdict:**
  - **H24 — SUPPORTED on the headline pool, and this is the §4.6 result.** On `r32/e25`,
    **4 forward passes per query match scoring all 180 survivors** (0.806 vs 0.799) — a **45x**
    reduction that costs nothing measurable. A `ppl` refusal gate is deployable, and §4.6 now has
    a defense with a cost number rather than a gesture.
  - **It is recipe-dependent, and the paper must say so.** On `r32/e5` the cheap version loses
    ~0.07 AUC and needs m=64 to match. The direction is convenient rather than awkward: the
    better-trained the experts, the better the cheap defense works, and well-trained experts are
    the deployed case.
  - **The r8/e5 "cheapest m = 4" is not a success and should not be quoted as one.** Full scoring
    there reaches only 0.630 — barely above the 0.57–0.61 confidence band — so a cheap budget
    matches full scoring because full scoring is already poor. The metric is doing what it was
    asked; the reading is what needs care.
  - **Detection does not require correct routing.** At m=4 the prefilter loses the query's own
    expert 23% of the time and detection is unaffected (0.806). Detection reads the **shape** of
    the score distribution — orphans are uniformly poorly-fit across whatever candidates you
    score — so the gate does not depend on the prefilter being right, only on it being cheap.
    That is why the curve is nearly flat from m=4 to m=180, and it is the part of this result
    most likely to generalise beyond TOFU.
  - Mild curiosity worth one line and no more: on e25, m=16 (0.822) scores *above* full scoring
    (0.799). Plausibly the tail of 180 `ppl` scores is mostly noise diluting the features. One
    cell, not a claim.

- **Observations:**
  - The analyzer refuses to pair matrices whose `author_of_q` or `is_forget` disagree. Both npz
    are sampled independently, and pairing one query's `ppl` row with another's prefilter row
    would have produced a perfectly plausible curve — this campaign's recurring failure mode, so
    the check is an error rather than a warning.
  - Also submitted this session: the **H23 destination sweep** (job 3206286, 7 arms x 4 GPUs),
    rerouting to seven survivors stratified by affinity to the deleted authors, from the nearest
    (shard **88** — the no-name "sink" author of 2026-08-07, affinity 0.397) to the farthest
    (shard 79, 0.219). Note the two arms already in hand argue *against* the similarity
    hypothesis before the sweep starts: s0 and s42 sit at affinity 0.2473 and 0.2559 — nearly
    identical — yet 0.28 apart in `forget_quality`. If the sweep scatters rather than trending,
    the metric responds to destination **identity**, which similarity does not explain.

- **New questions / new hypotheses:**
  - **H26:** does the m=4 defense survive the `indirect` transform, not just name-stripping? The
    matrices land when job 3205987 finishes and the check is free.
  - **H27:** what is the refusal gate's actual operating point? AUC is threshold-free; §4.6 needs
    a chosen threshold with its retained-traffic false-refusal rate. `retain_fpr_at_90_catch` is
    already in the JSON (e25 m=4) and is the number a deployment would be judged on.
