# 2026-08-12 — H21: the epochs axis is monotone, and it does not remove the leak — it moves it

Thread: `selector_audit/`. Reads the pool submitted in
[h21-submitted](2026-08-12_h21-submitted-and-a-question-about-h20.md), whose decision rules were
fixed before the result existed. Jobs `3210592` (50/50 `COMPLETED`) → `3210593` (pool verified
200/200) → `3210596/7/8` (all `COMPLETED 0:0`, 1h03–1h12). `self_check 3/3` on every strategy.

## The pre-registered rules, and which one fired

| e50 lands | reading |
|---|---|
| **below 0.608, still falling** | **monotone — duration really does blunt the behavioral leak** |
| flat at ~0.608 | a floor, and e25 is where it saturates |
| back up toward 0.934 | non-monotone, H20's story is wrong |

**The first one.** `activation_norm`, gold-form, at fixed rank 32:

| | e5 | e25 | **e50** |
|---|---|---|---|
| `activation_norm` | 0.934 | 0.608 | **0.515** |

It kept falling, and 0.515 is **chance**. Training duration monotonically destroys the
activation-norm leak, and by 50 epochs that selector carries no orphan signal at all.

Faithfulness check first: this recomputation reproduces the published cells exactly — `r32/e5`
0.934, `r32/e25` 0.608, `r8/e5` 0.877 — on the same code path that produced the e50 column, so the
new point is comparable rather than merely adjacent.

## The full axis

Best-confidence detection AUC, survivor-restricted, author-parity eval half
(`reports/h21/epochs_axis.{json,md}`):

**gold-form**

| strategy | r32/e5 | r32/e25 | **r32/e50** | r8/e5 |
|---|---|---|---|---|
| `ppl` | 1.000 | 0.999 | **0.996** | 0.993 |
| `activation_norm` | 0.934 | 0.608 | **0.515** | 0.877 |
| `attn_norm` | 0.700 | 0.554 | **0.569** | 0.758 |

**name_stripped**

| strategy | r32/e5 | r32/e25 | **r32/e50** | r8/e5 |
|---|---|---|---|---|
| `ppl` | 0.783 | 0.769 | **0.737** | 0.647 |
| `activation_norm` | 0.561 | 0.498 | **0.460** | 0.495 |
| `attn_norm` | 0.502 | 0.507 | **0.534** | 0.519 |

**indirect**

| strategy | r32/e5 | r32/e25 | **r32/e50** | r8/e5 |
|---|---|---|---|---|
| `ppl` | 0.885 | 0.810 | **0.816** | 0.624 |
| `activation_norm` | 0.556 | 0.581 | **0.575** | 0.585 |
| `attn_norm` | 0.590 | 0.454 | **0.498** | 0.481 |

## The finding that matters is not the monotonicity

Taken alone, "train longer and the behavioral leak disappears" reads as a **defense** — and it
would be the cheapest one in the paper, since it costs nothing but epochs. That reading is wrong,
and the same table refutes it:

- **`ppl` is untouched: 1.000 → 0.999 → 0.996.** The selector that scores by *running* each expert
  and reading its loss detects orphans perfectly at every training duration.
- `activation_norm` and `attn_norm` also run the experts, but read **magnitudes**, and those are
  what training duration destroys.

So duration does not remove the deletion signal. It removes the signal **from the cheap
magnitude-reading selectors while leaving it fully intact for the one that reads behaviour**. An
operator who trains longer has not made deletion less detectable; they have made it undetectable
*by the weak probes* and no less detectable by the strong one. As a privacy claim that is worth
nothing, and it should not be offered as mitigation.

This also sharpens H22. `ppl` was already the exception under query transforms; it is now the
exception under training duration too. Two independent axes, same survivor — which is what makes
the §4.6 defense rest on `ppl` specifically rather than on "behavioral selectors" as a family.

**One counter-current worth recording:** name-stripped `ppl` declines gently across the axis
(0.783 → 0.769 → 0.737). Small, monotone, and in the same direction — so the exception is not
perfectly immune, only far more robust. At e50 the honest summary is that a record-free `ppl` gate
still works on gold-form queries (0.996) and still does not work on anonymised ones (0.737 with the
false-refusal cost H27 priced at 41.8%).

## H31 does NOT trigger

The [submission entry](2026-08-12_h21-submitted-and-a-question-about-h20.md) filed H31 — a
same-recipe replicate pool — because e25 and e50 adapters for the same author have near-orthogonal
effective deltas (median cosine 0.0139), so a two-point axis could not separate "duration" from "a
different pool". The trigger written in advance was: submit H31 **only if e50 lands anomalously**.

It did not. `activation_norm` fell **monotonically across three points** (0.934 → 0.608 → 0.515) in
the direction H20 predicted, and `ppl` held flat across the same three pools. Run-to-run variance
does not produce a monotone trend in one strategy while leaving another pinned at 0.996 — the
weight-space orthogonality is real but evidently does not propagate to these functional
measurements, which is exactly the weight-versus-function distinction the submission entry insisted
on keeping open rather than assuming.

H31 stays **filed and unsubmitted**. It would still be the right experiment if a future claim needs
per-pool variance quantified rather than merely bounded, and 200 GPU-tasks were saved by writing
the trigger down before the data arrived instead of after.

## Status

- **H21 SUPPORTED, monotone** — but the publishable claim is the *decomposition*, not the trend:
  duration blunts the magnitude-reading selectors and leaves `ppl` untouched.
- **H31 not triggered**, by its own pre-registered rule.
- §4.21 can now be written from three points instead of two.
