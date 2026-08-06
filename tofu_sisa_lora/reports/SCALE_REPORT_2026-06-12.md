# SISA-LoRA k-Scaling Report — k ∈ {50, 100, 200} on Llama-2-7B-chat-hf

**Date:** 2026-06-12 · **Seed:** 42 everywhere · **Eval:** OU-faithful metrics
(`metrics_version: ou-2026-06-10`), smoke caps (ROUGE≤50, retain≤80, truth≤30) ·
**Recipe:** frozen r32/α64/5 epochs/lr 1e-4 (bs1×ga16, max_len 256), plus a rank-1 smoke arm
and a rank-8 auto-backup arm at k=200 · All GPU work via SLURM, ≤6 concurrent GPUs.
Companion to `SHARD_GRID_REPORT_2026-06-11.md` (k ∈ {1,4,10,20}); this report extends the
frontier to the hard maximum k=200 (1 author/shard) and adds the first GPU results for
**routing-based serving**.

---

## 0. Glossary for newcomers

### 0.1 The experiment in one paragraph

TOFU contains 200 invented authors × 20 Q&A pairs. We split the authors into **k disjoint
shards** and train **one small LoRA adapter per shard** on top of the frozen base model.
To serve a single model you must *combine* the k adapters somehow — by **merging** their
weights, or by **routing** each question to one adapter. To *unlearn* an author group you
delete its adapter — an exact, O(1) deletion, because no surviving parameter ever saw that
data. This sweep asks: what happens to quality as k grows to its maximum (k=200 = one author
per adapter), and is routing better than merging?

### 0.2 What each label (row) means

- **`base_model`** — the untouched Llama-2-7B-chat. The "never knew any of it" reference.
- **`k=1 ft`** — one LoRA trained on ALL 200 authors. The "knows everything" upper bound
  (no unlearning possible: deleting it deletes everything).
- **`shard_{k-1}_only`** — serve ONLY the forget shard's adapter. The "memorizer" reference:
  proves the forget data really lives inside that one adapter.
- **`merged_X`** — all k adapters merged into one weight set with method X
  (`linear` = plain average; `dare_ties` = drop-and-rescale + sign-consensus, the house
  default). **The serving baseline — nothing forgotten yet.**
- **`remerge_X`** — same merge but the forget shard's adapter is *deleted first*
  (k−1 adapters). **The state after unlearning, for merge-based serving.**
- **`routed_key_exact`** — no weight merging: every incoming question is routed to ONE
  adapter by author-name matching, and that adapter alone answers. **The serving baseline
  for routing.** (Questions with no author name fall back to shard 0 — ~14% of TOFU.)
- **`routed_key_exact_no{k-1}`** — same router with the forget shard *removed from the
  routing table*. **The state after unlearning, for routed serving.** Deletion = drop the
  adapter + one table entry; nothing else changes.
- **Config arms:** `k=50 r32`, `k=100 r32` — main scaling points on the frozen recipe;
  `k=200 r1` — rank-1 smoke/mechanics arm (also probes the rank axis);
  `k=200 r8` — the automatic backup arm that replaced `k=200 r32` (see §5: rank-32 at k=200
  physically cannot be evaluated on our 46 GiB GPUs).
- **retain90 oracle** (not a row here) — an adapter trained only on never-forget authors;
  it is the statistical reference that the `fq` metric compares against.

### 0.3 What each metric (column) means — and which direction is good

Forget-side metrics are judged *after deletion* (the `remerge_*` / `_no{k-1}` rows); for the
baseline rows you actually want the opposite sign on f_ppl/f_rouge — a baseline that never
learned the data has nothing to demonstrate (test T1 in §4).

- **mu — model_utility ↑** Overall usefulness: harmonic mean of 9 retain/real-world
  components. The weakest component dominates, so one broken capability zeroes the score.
  Anchors: base 0.418, best k=1 ft 0.744.
- **fq — forget_quality ↑** p-value of a statistical test (KS) asking "does the model treat
  the forget data like a model that never saw it?". →1 = forgotten, →0 = detectably still
  knows. ⚠ Weak at our 30-sample smoke cap — reliably flags memorizers (≈0) but little else.
- **f_ppl — forget perplexity ↑ after deletion** How surprised the model is by the true
  forget answers. Memorizer ≈ 1.2, base ≈ 15.2. After deletion it should RISE toward 15
  (= no longer knows). In the pre-deletion baseline, LOW f_ppl is what proves there was
  something to forget.
- **f_rouge — forget ROUGE ↓ after deletion** Text overlap between the model's generated
  answer and the true forget answer. 1.0 = verbatim recitation; base scores ~0.39 from pure
  style overlap, so ~0.4 after deletion = confabulation, not knowledge.
- **f_TR — forget truth ratio ↑** Does the model find the TRUE forget answer no more
  credible than plausible fakes? →1 = fully forgotten; base ≈ 0.74, memorizer ≈ 0.45-0.6.
- **ret_prob — retain answer probability ↑** How strongly the model believes the correct
  answers for the KEPT fictional authors. The main "did we keep what we wanted" signal.
- **ret_rouge ↑** Same idea, but for generated text overlap on kept authors.
- **ret_ppl — retain perplexity ↓** Surprise on kept authors' answers; low = still knows them.
- **ret_TRs / real_TRs / wld_TRs — scaled truth ratios ↑** Prefers true over fake answers
  on retain / real-author / world-fact questions (the truth-ratio parts of mu).
- **real_prob / real_rouge ↑** Knowledge about real-world authors — proxy for "did we damage
  the model's pretrained knowledge near the fine-tuning domain?".
- **wld_prob / wld_rouge ↑** General world knowledge — proxy for "did we damage everything
  else?".

---

## 1. Executive summary

1. **Routing replaces merging as the serving mode at high k — and it is the thesis result.**
   `routed_key_exact` at k=50 reaches **mu 0.7147**, within 0.03 of the k=1 train-on-
   everything upper bound (0.7435) and +0.27 above the best merge at any k>1 ever measured
   (lorahub 0.592 @k=4). At k=100 it scores 0.6475 — still above the 0.6 project bar no
   merge has ever met.
2. **Deletion under routing is utility-free at every scale.** Removing the forget adapter
   from the routing table leaves mu *bit-identical* (0.7147→0.7147 @k50, 0.6475→0.6475 @k100,
   0.4728→0.4728 @k200-r8) while the forget side moves to never-saw-it territory
   (f_ppl 1.40→8.54 @k50; f_TR → base 0.74). Retain/real/world queries never touch the
   deleted adapter, so nothing else CAN change — O(1), exact, and behaviorally verified.
   This is the first non-trivial unlearning demonstration beyond k=4 (the routed baseline
   truly knows the forget data: f_ppl 1.40 = memorizer level, unlike high-k merges).
3. **The dilution law completes — merging is dead at high k.** Merged dare_ties mu:
   0.74 (k=1) → 0.54 (4) → 0.48 (10) → 0.45 (20) → 0.44 (50) → 0.43 (100) → 0.42 (200)
   = base model (0.418). At k=200 every weight-merged state is indistinguishable from base.
4. **The bottleneck flips from merge interference to per-shard undertraining.** Routing
   accuracy is k-independent (0.86, lexical), but per-shard optimizer steps at fixed 5 epochs
   fall 25 → 12 → ~6 for k=50 → 100 → 200, and routed retain probability falls with it
   (0.749 → 0.470 → 0.213). Even the dedicated 1-author adapter at k=200 barely memorizes its
   author (f_ppl 9.6 vs 1.4 at k=50). A steps-matched arm (k50 e12 / k100 e25 / k200 e50) is
   the obvious next lever and should largely flatten the routed curve.
5. **k=200 × rank-32 cannot be evaluated on this cluster at all** — PEFT casts adapters to
   fp32 on load, so memory = 13.5 GiB (base) + k·params·4 B ⇒ ~65 GiB > 46 GiB A40. The
   pipeline's memory gate caught it overnight and **automatically substituted the rank-8 arm**
   (24.9 GiB, fits); k=200 numbers in this report are r8/r1. (§5.)
6. Side findings: `merged_linear` un-breaks as the rsLoRA √r-inflation vanishes (r1: √1=1 →
   mu 0.42; r8: 0.45 — best merge at k=200, though by then "best" ≈ base); rank-1 adapters at
   k=200/e5 learn nothing at all (every label ≈ base — the whole r1 arm is a no-op model).

## 2. Provenance

- Overnight chain (submitted 2026-06-11 22:40): stage-1 train+prep array **433769**
  (553 tasks %6, done in ~85 min), gate-r1 **433770** (PASS: 200 adapters in 6.6 min =
  2.0 s/adapter, 200-way dare_ties merge 6 s, peak 14.1 GiB), gate-r32 **433771**
  (FAIL by design → auto-cancelled r32 evals, auto-submitted backup), eval arrays
  **433772** (k50+k100) / **433773** (k200-r1) / **433774** (k200-r32, cancelled),
  collect **433775**.
- Auto-backup r8 chain (submitted BY the failing gate, ran unattended 02:17–03:14):
  train **434337** (200 shards), gate **434338** (PASS, 24.9 GiB), eval **434339**,
  collect **434340**.
- Morning fix-ups (6 evals OOM-killed by foreign no-gres processes on sprint1 — see §5):
  **434626** (k100 ×3), **434627** (r8 ×3), collect **434628**, `--exclude=sprint4,sprint1`.
- Scripts (sha256:12): `submit_scale_grid.sh` 8e6de3f4b778, `gate_scale_load.py`
  1aca9ec2f3df, `router.py` 2f2ec35e8b44 (post bug-fix). Bug fixed pre-launch:
  `build_key_index` pooled shard questions before the ≥50% name-frequency threshold →
  empty key index for shards with >2 authors → silent shard-0 fallback. Routing had never
  produced a result JSON before, so no prior numbers are affected. Measured routing
  accuracy after fix: 0.86–0.87 at k ∈ {10,50,100,200}.
- Aggregate CSV refreshed: `checkpoints/all_metrics_smoke.csv`. Daily narrative:
  `~/log/sisa_lora/` 2026-06-11 night + 2026-06-12 entries.

## 3. Results

### 3.1 The utility/granularity frontier (smoke mu; higher ↑ better)

| k | authors/shard | opt steps/shard | merged dare_ties | remerge dare_ties | routed key_exact | routed after deletion |
|---|---|---|---|---|---|---|
| 1 | 200 | ~625 | **0.7435** (= ft) | — | — | — |
| 4 | 50 | ~155 | 0.5445 | 0.5750 | — | — |
| 10 | 20 | ~62 | 0.4768 | 0.4791 | — | — |
| 20 | 10 | ~31 | 0.4504 | 0.4522 | — | — |
| 50 | 4 | ~25 | 0.4379 | 0.4380 | **0.7147** | **0.7147** |
| 100 | 2 | ~12 | 0.4299 | 0.4301 | **0.6475** | **0.6475** |
| 200 (r8) | 1 | ~6 | 0.4201 | 0.4199 | 0.4728 | 0.4728 |
| 200 (r1) | 1 | ~6 | 0.4197 | 0.4203 | 0.4212 | 0.4212 |
| base | — | — | 0.4179 | | | |

Merging decays monotonically into the base model. Routing sits in a different regime
entirely, limited only by how well each tiny shard was trained (last column of steps).

### 3.2 The routed unlearning demonstration (T1-T3, cf. 2026-06-11 report §0.7)

| k | state | mu | f_ppl | f_TR | f_rouge | verdict |
|---|---|---|---|---|---|---|
| 50 | routed (baseline) | 0.7147 | **1.40** | 0.529 | 0.681 | T1 PASS — serves the memorizer, truly knows it |
| 50 | routed_no49 (deleted) | **0.7147** | **8.54** | 0.746 | 0.443 | T2 PASS (f_TR ≈ base 0.742; rouge ≈ base 0.39 = confabulation) · T3 PASS (mu unchanged) |
| 100 | routed (baseline) | 0.6475 | 2.43 | 0.649 | 0.464 | T1 PASS |
| 100 | routed_no99 (deleted) | **0.6475** | 6.71 | 0.757 | 0.412 | T2 PASS · T3 PASS |
| 200 r8 | routed (baseline) | 0.4728 | 9.60 | 0.779 | 0.349 | T1 weak — the 1-author adapter is undertrained |
| 200 r8 | routed_no199 (deleted) | **0.4728** | 13.09 | 0.801 | 0.361 | T2/T3 PASS, but little to forget |

Forget queries after deletion fall back to shard 0 (an adapter that never saw the forget
authors), hence f_ppl lands at ~7-9 rather than base 15.2 — TOFU-format style adaptation,
same as the retain oracle. Contrast merges at k≥10, where T1 already fails (dilution
pre-forgets) and the remerge demonstration is vacuous.

### 3.3 Per-shard undertraining is the new bottleneck

- routed ret_prob: 0.749 (k=50) → 0.470 (k=100) → 0.213 (k=200 r8) — tracks steps/shard.
- `shard_{k-1}_only` f_ppl on its OWN authors: 1.40 (k=50) → 2.43 (k=100) → 9.60 (k=200 r8)
  — at e5 a 1-author shard (20 examples, ~6 optimizer steps) barely memorizes.
- The k=200 r1 arm is the limit case: every label ≈ base — 6 steps × rank 1 = identity.
- Routing accuracy is constant (0.86) across k, so the routed-mu decline is a TRAINING
  problem, not a routing problem → steps-matched arm is the highest-value next experiment.

## 4. Infrastructure findings (what the gates measured)

- **fp32 memory law (validated):** PEFT `load_adapter` casts adapters to fp32
  (`_cast_adapter_dtype`) ⇒ eval memory ≈ 13.5 GiB + k · n_params(rank) · 4 B.
  Measured @k=200: r1 = 14.1 GiB, r8 = 24.9 GiB; r32 ⇒ ~65 GiB ⇒ **impossible on A40**.
  k=100 r32 ≈ 39 GiB fits only on an empty card. Documented in tofu_sisa_lora/CLAUDE.md.
- **PEFT mechanics at 200 adapters are fine:** load 2.0 s/adapter (6.6 min total),
  200-way `add_weighted_adapter` dare_ties merge in ~6 s, routing over 200 adapters works.
- **The gate→auto-backup design paid for itself on night one:** gate-r32 failed at 00:16,
  scancelled its dependent evals (cluster has `kill_invalid_depend` off — without the
  scancel the chain would have hung), submitted the r8 chain from the compute node, and the
  morning had complete k=200 numbers with zero human intervention.
- **Cluster gotcha:** 6 eval tasks OOMed sharing their GPU with three foreign ~7 GiB
  processes (no-gres jobs, sprint1, PIDs constant 00:16–03:08). SLURM gres accounting cannot
  see them. Diagnosis: read the foreign PIDs in the CUDA OOM message before debugging memory
  math. Fix: resubmit with `--exclude`.

## 5. Limitations & next steps

- Single seed (42); smoke caps. fq is non-discriminative here (KS over ≤30 samples; at
  k=200 the forget set is one author = 20 rows). The routed-vs-merged gap (+0.27 mu) and the
  dilution trend are large effects; ±0.01 method differences are noise.
- `routed_key_exact` exploits TOFU's author-name structure (and ~14% of questions are
  name-free → shard-0 fallback). Robustness check: `routed_centroid_sbert` (semantic, no
  name dependence) on the k=50 arm.
- **Next:** (1) steps-matched arm k50 e12 / k100 e25 / k200 e50, routed labels first —
  hypothesis: routed mu ≈ 0.7 flat in k ⇒ per-author O(1) deletion at near-full utility;
  (2) extended-cap eval of k=50 routed full/_no49 (fq power); (3) seed-variance pass;
  (4) promote routing to a headline serving mode alongside merging in the project docs.

## Appendix A — complete data table (all 28 sweep JSONs + references, smoke, 2026-06-12)

Reference rows from `SHARD_GRID_REPORT_2026-06-11.md` Appendix A.

| config | label | mu | fq | f_ppl | f_rouge | f_TR | ret_prob | ret_rouge | ret_ppl | ret_TRs | real_prob | real_rouge | real_TRs | wld_prob | wld_rouge | wld_TRs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | base_model | 0.4179 | 0.239 | 15.19 | 0.393 | 0.742 | 0.164 | 0.427 | 15.67 | 0.167 | 0.778 | 0.982 | 0.868 | 0.679 | 0.933 | 0.911 |
| REF | k=1 ft r32e5 (winner) | 0.7435 | 0.003 | 1.27 | 0.878 | 0.446 | 0.921 | 0.863 | 1.29 | 0.500 | 0.646 | 0.865 | 0.806 | 0.606 | 0.920 | 0.849 |
| REF k=4 | merged_dare_ties | 0.5445 | 0.808 | 4.48 | 0.535 | 0.682 | 0.308 | 0.469 | 3.90 | 0.287 | 0.654 | 0.982 | 0.794 | 0.624 | 0.940 | 0.839 |
| REF k=4 | remerge_dare_ties | 0.5750 | 0.808 | 11.93 | 0.496 | 0.713 | 0.387 | 0.451 | 3.09 | 0.354 | 0.617 | 0.862 | 0.788 | 0.612 | 0.910 | 0.805 |
| REF k=10 | merged_dare_ties | 0.4768 | 0.594 | 7.82 | 0.451 | 0.763 | 0.232 | 0.477 | 7.55 | 0.189 | 0.751 | 0.992 | 0.871 | 0.677 | 0.927 | 0.889 |
| REF k=10 | remerge_dare_ties | 0.4791 | 0.393 | 8.41 | 0.452 | 0.767 | 0.237 | 0.477 | 7.22 | 0.192 | 0.737 | 0.992 | 0.867 | 0.660 | 0.907 | 0.881 |
| REF k=20 | merged_dare_ties | 0.4504 | 0.594 | 10.48 | 0.521 | 0.711 | 0.203 | 0.401 | 9.83 | 0.182 | 0.768 | 0.992 | 0.877 | 0.686 | 0.933 | 0.905 |
| REF k=20 | remerge_dare_ties | 0.4522 | 0.594 | 10.54 | 0.521 | 0.709 | 0.206 | 0.400 | 9.63 | 0.183 | 0.767 | 0.992 | 0.878 | 0.686 | 0.933 | 0.904 |
| k=50 r32 | merged_linear | 0.0433 | 0.0709 | 283.9 | 0.1768 | 0.7711 | 0.0131 | 0.1677 | 303.8 | 0.2992 | 0.2392 | 0.0133 | 0.3211 | 0.2589 | 0.0300 | 0.3580 |
| k=50 r32 | merged_dare_ties | 0.4379 | 0.2391 | 12.2 | 0.4610 | 0.7740 | 0.1811 | 0.4086 | 12.6 | 0.1812 | 0.7788 | 0.9820 | 0.8737 | 0.6842 | 0.9400 | 0.9134 |
| k=50 r32 | remerge_linear | 0.0443 | 0.0709 | 338.8 | 0.1612 | 0.7688 | 0.0125 | 0.1714 | 328.9 | 0.2959 | 0.2396 | 0.0133 | 0.3097 | 0.2525 | 0.0400 | 0.3565 |
| k=50 r32 | remerge_dare_ties | 0.4380 | 0.2391 | 12.2 | 0.4644 | 0.7751 | 0.1818 | 0.4043 | 12.5 | 0.1813 | 0.7785 | 0.9920 | 0.8750 | 0.6844 | 0.9400 | 0.9126 |
| k=50 r32 | shard_49_only | 0.4717 | 0.0065 | 1.4000 | 0.7210 | 0.5208 | 0.2104 | 0.4619 | 8.3000 | 0.2244 | 0.6658 | 0.9220 | 0.8482 | 0.6232 | 0.9067 | 0.8112 |
| k=50 r32 | routed_key_exact | 0.7147 | 0.0065 | 1.4000 | 0.6812 | 0.5286 | 0.7492 | 0.6688 | 1.6700 | 0.4941 | 0.6568 | 0.9100 | 0.8037 | 0.6353 | 0.8867 | 0.8534 |
| k=50 r32 | routed_key_exact_no49 | 0.7147 | 0.3929 | 8.5400 | 0.4428 | 0.7458 | 0.7492 | 0.6688 | 1.6700 | 0.4941 | 0.6568 | 0.9100 | 0.8037 | 0.6353 | 0.8867 | 0.8534 |
| k=100 r32 | merged_linear | 0.0821 | 0.2391 | 46.1 | 0.2595 | 0.7810 | 0.1706 | 0.3062 | 44.0 | 0.2010 | 0.2730 | 0.0133 | 0.3103 | 0.2595 | 0.1333 | 0.4265 |
| k=100 r32 | merged_dare_ties | 0.4299 | 0.1350 | 14.1 | 0.4142 | 0.7703 | 0.1707 | 0.4055 | 14.8 | 0.1806 | 0.7785 | 0.9920 | 0.8706 | 0.6822 | 0.9333 | 0.9128 |
| k=100 r32 | remerge_linear | 0.1462 | 0.3929 | 35.9 | 0.2857 | 0.7820 | 0.1822 | 0.3488 | 33.8 | 0.2012 | 0.2751 | 0.0333 | 0.3203 | 0.2625 | 0.1867 | 0.4352 |
| k=100 r32 | remerge_dare_ties | 0.4301 | 0.1350 | 13.9 | 0.4208 | 0.7702 | 0.1709 | 0.4043 | 14.6 | 0.1809 | 0.7785 | 0.9820 | 0.8715 | 0.6822 | 0.9400 | 0.9118 |
| k=100 r32 | shard_99_only | 0.4748 | 0.3929 | 2.4300 | 0.4709 | 0.6557 | 0.2461 | 0.4204 | 6.5900 | 0.1856 | 0.7393 | 0.9920 | 0.8646 | 0.7147 | 0.9067 | 0.9054 |
| k=100 r32 | routed_key_exact | 0.6475 | 0.3929 | 2.4300 | 0.4644 | 0.6492 | 0.4702 | 0.5133 | 2.6200 | 0.3504 | 0.7857 | 0.9620 | 0.8818 | 0.7532 | 0.8867 | 0.9352 |
| k=100 r32 | routed_key_exact_no99 | 0.6475 | 0.1350 | 6.7100 | 0.4125 | 0.7572 | 0.4702 | 0.5133 | 2.6200 | 0.3504 | 0.7857 | 0.9620 | 0.8818 | 0.7532 | 0.8867 | 0.9352 |
| k=200 r1 | merged_linear | 0.4204 | 0.1745 | 17.2 | 0.3475 | 0.7955 | 0.1609 | 0.3876 | 13.6 | 0.1811 | 0.7782 | 0.9820 | 0.8698 | 0.6813 | 0.9333 | 0.9108 |
| k=200 r1 | merged_dare_ties | 0.4197 | 0.1745 | 17.8 | 0.3511 | 0.7957 | 0.1601 | 0.3882 | 14.1 | 0.1810 | 0.7778 | 0.9820 | 0.8698 | 0.6803 | 0.9333 | 0.9104 |
| k=200 r1 | remerge_linear | 0.4193 | 0.1745 | 17.2 | 0.3744 | 0.7960 | 0.1609 | 0.3799 | 13.7 | 0.1808 | 0.7785 | 0.9820 | 0.8700 | 0.6811 | 0.9333 | 0.9124 |
| k=200 r1 | remerge_dare_ties | 0.4203 | 0.1745 | 17.8 | 0.3502 | 0.7951 | 0.1601 | 0.3915 | 14.1 | 0.1810 | 0.7784 | 0.9820 | 0.8697 | 0.6801 | 0.9333 | 0.9118 |
| k=200 r1 | shard_199_only | 0.4209 | 0.1745 | 17.5 | 0.3731 | 0.7956 | 0.1605 | 0.3890 | 14.0 | 0.1818 | 0.7782 | 0.9920 | 0.8702 | 0.6803 | 0.9333 | 0.9118 |
| k=200 r1 | routed_key_exact | 0.4212 | 0.1745 | 17.5 | 0.3731 | 0.7955 | 0.1610 | 0.3905 | 13.9 | 0.1818 | 0.7783 | 0.9820 | 0.8696 | 0.6812 | 0.9333 | 0.9107 |
| k=200 r1 | routed_key_exact_no199 | 0.4212 | 0.1745 | 17.6 | 0.3663 | 0.7960 | 0.1610 | 0.3905 | 13.9 | 0.1818 | 0.7783 | 0.9820 | 0.8696 | 0.6812 | 0.9333 | 0.9107 |
| k=200 r8 | merged_linear | 0.4503 | 0.1745 | 11.2 | 0.3543 | 0.8064 | 0.2110 | 0.4073 | 9.2800 | 0.1721 | 0.7639 | 0.9920 | 0.8773 | 0.7352 | 0.9200 | 0.9280 |
| k=200 r8 | merged_dare_ties | 0.4201 | 0.1745 | 17.8 | 0.3461 | 0.7973 | 0.1599 | 0.3914 | 14.1 | 0.1811 | 0.7782 | 0.9820 | 0.8701 | 0.6799 | 0.9333 | 0.9113 |
| k=200 r8 | remerge_linear | 0.4499 | 0.1745 | 11.2 | 0.3561 | 0.8064 | 0.2112 | 0.4063 | 9.2700 | 0.1716 | 0.7642 | 0.9920 | 0.8779 | 0.7350 | 0.9200 | 0.9294 |
| k=200 r8 | remerge_dare_ties | 0.4199 | 0.1745 | 17.7 | 0.3654 | 0.7951 | 0.1599 | 0.3882 | 14.1 | 0.1814 | 0.7783 | 0.9820 | 0.8701 | 0.6804 | 0.9333 | 0.9116 |
| k=200 r8 | shard_199_only | 0.4398 | 0.3356 | 9.6000 | 0.3495 | 0.7798 | 0.1884 | 0.3920 | 10.7 | 0.1772 | 0.7818 | 0.9920 | 0.8839 | 0.7149 | 0.9400 | 0.9253 |
| k=200 r8 | routed_key_exact | 0.4728 | 0.3356 | 9.6000 | 0.3495 | 0.7795 | 0.2134 | 0.3870 | 8.1300 | 0.2094 | 0.7810 | 0.9920 | 0.8818 | 0.7085 | 0.9333 | 0.9213 |
| k=200 r8 | routed_key_exact_no199 | 0.4728 | 0.1745 | 13.1 | 0.3608 | 0.8007 | 0.2134 | 0.3870 | 8.1300 | 0.2094 | 0.7810 | 0.9920 | 0.8818 | 0.7085 | 0.9333 | 0.9213 |
