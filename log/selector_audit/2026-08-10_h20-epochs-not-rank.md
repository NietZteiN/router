### Target Date: 2026-08-10 (H20 — the third pool lands: epochs, not rank; and a dead node)

Fourth entry today. The `r32 e5` behavioral arm that H18 was waiting on has landed, so the three
k=200 pools now separate the two axes of expert fit that were confounded in H18's pair.

- **Hypotheses / what we're testing:** **H20** — is behavioral detectability inversely related to
  expert fit, and if so is the driver **rank** (capacity) or **epochs** (training duration)?
  H18 had only `r8/e5` (0.877) and `r32/e25` (0.608), which move both axes at once.

- **Setup:** no new compute for the finding itself — the arm was already queued; job 3201164
  (`sa-finalize`, 13/13 steps ok) turned it into `reports/probe_beh_*`. All three pools, both
  query transforms, same drop set `180-199`, same author-parity split, seed 42.

- **Results:** probe AUC / best-confidence AUC, k=200, forget10 deleted. The best-confidence
  column is the one comparable to H18's table.

  | strategy | r8/e5 | **r32/e5** | r32/e25 |
  |---|---|---|---|
  | | probe · conf | probe · conf | probe · conf |
  | `activation_norm` | 0.914 · 0.877 | **0.972 · 0.934** | 0.692 · 0.608 |
  | `attn_norm` | 0.793 · 0.758 | **0.764 · 0.700** | 0.659 · 0.554 |
  | `ppl` | 0.999 · 0.993 | **1.000 · 1.000** | 1.000 · 0.999 |

  Name-stripped, the same three pools:

  | strategy | r8/e5 | **r32/e5** | r32/e25 |
  |---|---|---|---|
  | `activation_norm` | 0.594 · 0.495 | **0.661 · 0.561** | 0.606 · 0.498 |
  | `attn_norm` | 0.620 · 0.519 | **0.622 · 0.502** | 0.612 · 0.507 |
  | `ppl` | 0.630 · 0.647 | **0.782 · 0.783** | 0.799 · 0.769 |

- **What worked / hypothesis verdict:**
  - **H20 — SUPPORTED, but only for the epochs axis.** Holding epochs at 5 and raising rank
    8 → 32 does *not* degrade detectability: `activation_norm` goes **0.877 → 0.934** (up) and
    `attn_norm` 0.758 → 0.700 (down slightly). Holding rank at 32 and raising epochs 5 → 25
    collapses both: **0.934 → 0.608** and **0.700 → 0.554**. Epochs is the monotone driver;
    rank is small and not even consistent in sign across the two strategies.
  - This **narrows H18's mechanism rather than confirming it.** H18 guessed "better-fit experts
    produce large activations on any query". If that were the whole story, more rank should hurt
    too, and it does not. The surviving version is about *training duration specifically*: 25
    epochs on 20 QA pairs drives every expert deep enough that its activation profile stops
    depending on whether the query is its own. Capacity alone does not do that. Still a
    hypothesis — three pools on one axis each is not a curve.
  - **H11 is now recipe-independent.** Name-stripped, `activation_norm` and `attn_norm` sit in
    **0.495–0.561** across *all three* pools — chance, everywhere, at every fit level. Whatever
    the recipe does to gold-form detectability, it does nothing for the case that matters.
  - One asymmetry worth flagging: name-stripped `ppl` moves the *other* way with fit,
    0.647 → 0.783 → 0.769. Perplexity is the one behavioral score that keeps some signal without
    the name, and more training gives it more, not less. That is the only route by which the
    behavioral family could beat the lexical result, and it is a single strategy at ~0.78.

- **Observations:**
  - **A dead node, not a bug — E5 `reroute42`.** Job `3200588_3` burned its full 4 h wall having
    produced **zero** forward passes: progress frozen at `forget_ppl 0/400` after 94 s, and 3087
    `NVML: Failed to get usage(999)` lines in the log. Sibling arms of the *same array* ran the
    identical workload on `-10` and `-21` in 23–27 min with **no NVML line at all**. So it is
    `xe8545-a100-22`, and SLURM still lists that node as `idle` with no drain reason — it will
    keep handing it out. Added to `TOFU_EXCLUDE` in `cluster_env.cispa.sh` (the list already
    carries -03/-05/-12/-16/-17 for the same class of failure) with the evidence in a comment,
    and the arm resubmitted as job 3201916 with 3201920 chained `afterany` to re-consolidate.
    Worth recording because the failure mode is indistinguishable from a hang in our own code
    until you compare sibling arms — the diagnostic is *same array, same workload, other node*.
  - **`submit_finalize_selector.sh` printed an empty summary for a silly reason.** Its H18 recap
    grepped `^\| \`(ppl|activation_norm|attn_norm)\`` but `write_md` emits the strategy
    **without** backticks, so it matched nothing and every run printed bare filenames under a
    heading. Not a numbers defect — the reports themselves were correct and complete — but it is
    the fifth instance of the campaign's recurring shape: *something that looks like it reported
    and did not*. Fixed.

- **New questions / new hypotheses:**
  - **H21:** does the epochs effect continue past 25, or saturate? A `r32/e50` pool would say,
    and would turn one axis of three points into an actual curve. Cheap relative to its value —
    but it is a training run, not an analysis, so it is a next-wave item.
  - **H22:** `ppl` name-stripped rising with fit (0.647 → 0.783) is the single result pointing
    away from "detection is lexical". Is that real signal, or is it perplexity partially
    reconstructing the stripped name from the rest of the question? The `indirect` transform
    already exists and would separate those.
