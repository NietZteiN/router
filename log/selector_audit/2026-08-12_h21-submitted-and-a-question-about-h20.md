# 2026-08-12 — H21 submitted: a third point on the epochs axis, and a question it raises about H20

Thread: `selector_audit/`. Pre-registration + calibration record. Jobs `3210592` (train, 50×4 GPU)
→ `3210593` (check, CPU) → `3210596/7/8` (behavioral wave: gold / name_stripped / indirect), all
chained `afterany`.

## Pre-registration

H20 separated two axes H18 had confounded: rank 8→32 at fixed e5 does **not** degrade behavioral
orphan detectability (`activation_norm` 0.877 → 0.934), while epochs 5→25 at fixed r32 collapses it
(0.934 → **0.608**). The conclusion was that training *duration*, not capacity, blunts the leak.

That axis has two points. e50 is the third, and the outcomes are mutually exclusive:

| e50 lands | reading |
|---|---|
| below 0.608, still falling | monotone — duration really does blunt the behavioral leak |
| flat at ~0.608 | a **floor**, and e25 is already where it saturates |
| back up toward 0.934 | non-monotone, and H20's story is wrong |

New pool `Llama-2-7B-chat-hf_k200_r32_e50_lr1e4`; driver `submit_h21_e50_pool.sh`; one new arm
`beh_e50` in `submit_selector_wave.sh`.

**No `feat_e50` arm, deliberately.** The 2026-08-07 H7 correction established that no feature-space
router reads expert weights, so a feature arm on a new pool returns matrices byte-identical to
those already on disk. A comment now says so at the `FEAT_ARMS` definition, because this is the
second time the trap has been available and the first time it was not avoided.

## Calibration (job 3210540, author 0)

**7m29s wall, of which training was 30 seconds.** A k=200 shard is one author = 20 rows, so 50
epochs is 50 optimizer steps; the 7B base-model load dominates completely. Hence PACK=4 (16 GPUs =
the association's `gres/gpu` limit, using 4 of 6 job slots) and a 01:30:00 wall — ~6× the measured
time, because four authors per node means four concurrent base-model pulls off NFS, and NFS
contention is exactly what timed out the r32 behavioral arms at 6 h on 2026-08-10.

Estimated completion: 200 authors / 16 concurrent ≈ 13 rounds ≈ 3 h.

### A near-miss worth recording

The training log's progress bar ends at **`epoch: 25.0`** for a run invoked with `--epochs 50`. Had
I read only that, I would have concluded the flag was being ignored and that the "e50" pool was a
duplicate of e25 — an experiment measuring nothing. It is a display artifact:
`shard_meta.json` records `epochs: 50` / `num_samples: 20`, and the cosine LR schedule reaches
exactly `0.0` at step 50, so the scheduler was sized for 50 optimizer steps = 1 per epoch × 50
epochs. The displayed counter is inconsistent with that because `grad_accum=4` does not divide the
5-batch dataloader; I did not chase the exact cause beyond establishing it is cosmetic. The
configuration is identical across e5/e25/e50 (same rows, batch, accum), so the axis stays
comparable regardless.

## The finding, and why it is a problem for H20 rather than for H21

Comparing the new e50 adapter against the existing e25 adapter for the *same author*:

- raw LoRA factors: 0/384 tensors bitwise identical — **but this comparison is meaningless**, since
  the A/B factorisation is not unique and identical functions can have unrelated factors. Recorded
  because I ran it first and it would have been an easy number to quote.
- **effective deltas** `s·BA`, the thing that is actually applied: median cosine **0.0139** across
  192 modules (min 0.0006, max 0.0951), median relative L2 difference 1.69, with e50's total norm
  **38% larger** (24.04 vs 17.40).

Two adapters, same author, same 20 rows, same seed, same rank — and their effective deltas are
**near-orthogonal**. That is not a bug: the cosine LR schedule is defined over *total* steps, so the
two runs differ in learning rate from step 1 and never share a trajectory. With 20 examples and a
rank-32 update across 192 modules the fit is massively underdetermined, and both runs reach the same
training loss (~0.047) by wholly different routes.

**The question this raises:** if two pools that differ only in schedule length land near-orthogonal
in weight space, how much of H20's 0.934 → 0.608 is *duration* and how much is simply *a different
pool*? H20 has no same-recipe replicate, so it cannot currently distinguish them.

Two things keep this a hypothesis rather than a refutation, and both should be stated before anyone
quotes it:

1. **Weight-space orthogonality does not imply function-space instability.** Detectability is a
   functional measurement — does expert *i* fit query *q* — and near-orthogonal deltas can implement
   near-identical functions. Inferring instability from cosine would be exactly the weight-vs-function
   confusion the raw-factor comparison above already illustrates.
2. **The e50 audit is itself the first evidence.** If the epochs axis is coherent, e50 lands
   smoothly relative to e5/e25. If it lands wildly, that is the variance showing.

Filed as **H31**: estimate run-to-run variance with a same-recipe replicate pool (e25, different
seed) and re-read the epochs axis against it. Deliberately **not** submitted now — it is another 200
GPU-tasks to chase a confound the pending e50 result may well resolve for free, and the disciplined
order is to let the data speak first. If e50 lands anomalously, H31 becomes the next wave rather
than a speculative one.

## Status

- **H21 submitted**, decision rules above fixed before the result exists.
- **H31 filed**, not submitted, with the trigger for submitting it stated in advance.
- Cost: ~3 h wall on 16 GPUs for the pool, plus three single-arm audits behind it.
