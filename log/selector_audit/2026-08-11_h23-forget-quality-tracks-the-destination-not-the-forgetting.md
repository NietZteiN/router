### Target Date: 2026-08-11 (H23 — forget_quality moves 0.62 on a choice that forgets nothing)

Third entry today. E5's two arms showed `forget_quality` 0.28 apart on an arbitrary reroute
destination. Seven more destinations, stratified by affinity to the deleted authors, were meant to
test whether expert SIMILARITY explains that. It does not — and pinning down *why* turned up the
sharper result.

- **Hypotheses / what we're testing:** **H23** — is `forget_quality` monotone in the reroute
  destination's affinity to the deleted authors? Pre-registered: monotone ⇒ §4.10 says the metric
  measures expert similarity; scatter ⇒ it responds to destination **identity**, which similarity
  does not explain.

- **Setup:** job 3206288, 7 arms × 1 GPU on the k=200 r32 e25 pool, `--delete_shards 180-199`
  `--reroute_to D`, smoke tier, KS reference unchanged. Destinations span the full observed
  affinity range 0.2193–0.3970 (mean `centroid_sbert` score from the deleted authors' queries).
  All 7 COMPLETED 0:0 in 19–27 min, zero NVML lines — no repeat of the `-22` signature. Route
  audit first, metric second: every arm reports `rerouted: 630, deleted: 0`, and `model_utility`
  is **identical to 4 decimals (0.7921) across all nine arms**, as it must be when only the
  forget authors' routing changes.

- **Results — nine reroute destinations, nothing deleted in any of them:**

  | dest | 88 | 137 | 89 | 31 | 33 | 42 | 0 | 97 | 79 |
  |---|---|---|---|---|---|---|---|---|---|
  | affinity | .3970 | .3382 | .3044 | .2840 | .2663 | .2559 | .2473 | .2267 | .2193 |
  | **forget_quality** | **.1561** | **.7715** | .5789 | .3995 | .5789 | .3995 | .6789 | .4856 | .4856 |
  | model_utility | .7921 | .7921 | .7921 | .7921 | .7921 | .7921 | .7921 | .7921 | .7921 |

  Spearman ρ vs affinity = **−0.059 (p = 0.88)**; Pearson −0.316 (p = 0.41). Flat scatter.
  `forget_rouge` ρ = −0.033, `forget_truth_ratio` ρ = −0.467 (p = 0.21) — nothing trends.

  Reference points on the same pool and tier: **genuine deletion = 0.5789**, pre-deletion
  (nothing deleted, correct routing) = **0.0078**.

- **What worked / hypothesis verdict:**
  - **H23 — the similarity hypothesis is REFUTED, and the stronger reading is the one that
    survives.** `forget_quality` spans **0.1561 → 0.7715** across destinations, a **0.62 swing on
    an axis with no relation to forgetting**, at byte-identical utility, with nothing deleted in
    any arm. The nearest destination (s88, affinity 0.397) scores *lowest* and the second-nearest
    (s137) scores *highest* — the two most similar destinations sit at opposite ends.
  - **Two reroute arms are EXACTLY indistinguishable from genuine deletion.** s89 and s33 both
    score 0.5789, which is deletion's own number to four decimals. A method that deletes nothing
    and redirects the queries to a stranger is not merely competitive with deletion under this
    metric — on those destinations it is *the same cell*. That is the §4.10 spine.

- **Observations:**
  - **The resolution finding, which is worth more than the sweep.** Three exact ties among seven
    new arms is not coincidence. `forget_quality = ks_2samp(forget_tr, retain_ref).pvalue`, and at
    smoke tier that is a **30-vs-20** two-sample KS: `SMOKE_TRUTH_MAX = 30`, and the cached
    reference holds **20** rows. The exact p-value grid was enumerated and **all six observed
    values are on it, 6/6**:

    | D | .1833 | .2000 | .2167 | .2333 | .2500 | .2667 | .2833 | .3000 | .3167 |
    |---|---|---|---|---|---|---|---|---|---|
    | p | .7715 | .6789 | .5789 | .4856 | .3995 | .3238 | .2571 | .2024 | .1561 |

    The whole 0.62 spread is **D moving 0.1333 = four questions out of thirty**. The metric has
    **34 achievable values in total, 18 above p = 0.05**, and adjacent rungs in the readable range
    are **~0.10 apart** — so the four decimals every table (including ours) reports are spurious
    precision. Resolution by tier: smoke 0.10 → extended 0.0586 → full 0.0186.
  - **A separate limitation of the reference itself, found on the way.** `prepare_eval.py` derives
    it from `shards[forget_shard_id]`, which at k=200 is **author 199 alone** — hence 20 rows —
    while the arms score authors 180–199 via `--forget_author_ids`. Being the other sample in the
    KS test, those 20 rows cap resolution at *every* tier. Fixing it needs `--forget_author_ids`
    in `prepare_eval.py` and a new reference, which would break comparability with every published
    cell, so it is recorded rather than silently changed
    (`results/extended/README_KS_REFERENCE.txt`).
  - **This does not soften H23, it explains its shape.** The spread is real and the ordering is
    real; what the grid explains is why it comes in lumps and why nearby destinations tie. The
    claim to make is "an arbitrary destination choice moves the metric across most of its usable
    range", not "0.1561 vs 0.7715 to four decimals".
  - Note the direction of the smoke-tier caps for anyone quoting published TOFU numbers: this is
    the tier at which most cells in this repo — and much of the literature's cheap evaluation —
    are computed.

- **New questions / new hypotheses:**
  - **H28 (submitted, job 3206784):** does the spread survive a finer tier? 8 arms at
    `--extended` (truth rows 30 → 120, grid 34 → 55 values), **same reference array held fixed**
    so exactly one thing changes, plus a `del` baseline arm so genuine deletion is measured at the
    same tier it is compared against. Persists ⇒ destination identity genuinely moves the metric.
    Collapses ⇒ the smoke tier cannot support destination claims at all, which is itself the
    §4.10 finding and a sharper one.
  - **H29:** what does `forget_quality` even mean when 4 of 30 questions move it 0.62? A
    bootstrap over the forget sample would put a confidence interval on every published cell in
    the paper. Pure CPU, no new compute — the truth-ratio arrays are the only thing needed and
    they are not currently dumped.
