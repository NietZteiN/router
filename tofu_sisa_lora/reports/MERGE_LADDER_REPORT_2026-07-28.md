# The Merging Ladder — what happens when you merge N single-author models

**Date:** 2026-07-28 · **Pool:** P3 (Llama-2-7B-chat-hf, k=200 one author per shard, r32, e5) ·
**Operator of record:** `additive_mean` (true-scale uniform mean). All numbers read from
`reports/nmerge_mu.csv`, `reports/nmerge_own_recall.csv`, `reports/nmerge_subset_mu.csv`,
`reports/nmerge_overlap.csv`, and `reports/centered/nmerge_mu.csv` — nothing quoted from memory.
Figures: `reports/figures/nmerge/fig{1,2,3,5}_*.{png,pdf}`. Companion: `NMERGE_REPORT_2026-07-08.md`.

---

## TL;DR

Train one LoRA per author (200 authors), merge **N** of them into one served model, sweep
N = 1, 2, 3, 4, 6, 8, … 200, and measure. The result is a **two-curve story**, and the two curves
point in opposite directions — which is the whole point:

1. **Population utility (`mu`) is FLAT in N** — 0.457–0.462 at every N from 2 to 200. Merging never
   accumulates the authors' knowledge; it parks at **base + dilution (≈0.46)**, ~0.30 below the joint
   fine-tune (0.756) and the routing-over-the-same-experts result (~0.75). **Level, not slope.**
2. **Per-author recall COLLAPSES** — the individual author facts vanish: perplexity on an author's own
   answers climbs **3.6 → 8.5**, and **≈85% of the *learned* recall signal is gone by N ≈ 8**.

The flat `mu` is not a success — it is a **measurement blind spot**. `mu` is a population metric that
samples any one author in ~1/200 of its questions, so 1/N dilution of per-author knowledge does not
move it. You only see the failure when you condition on a single author. **Author merging cannot
substitute for selecting one expert (routing / masks); amplifying residuals does not either.**

---

## 1. Setup (reproducible)

| | |
|---|---|
| Model | `meta-llama/Llama-2-7B-chat-hf` |
| Pool | 200 per-author LoRAs, **one author per shard** (r32, α64, 5 epochs, lr 1e-4), dir `_k200_r32_e5_lr1e4/` |
| Merge | `merge_subset.py` — materialize N nested-subset merges on CPU as one PEFT adapter each, eval via `--preloaded_adapter` (never pays the k=200×r32 fp32 memory wall) |
| N-ladder | 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 32, 64, 128, 200 (N∈{128,200} SVD-compressed to rank 1024; accepted vs exact, \|Δmu\|=0.0007 at N=64) |
| Probes | perm(42)[:5] = authors {82, 15, 111, 177, 76}; headline = author 82 |
| Metrics | `model_utility` (mu, population), own-author `forget_rouge`/`forget_ppl` (per-author recall via `--eval_shard_id`), subset-conditioned retain (`--retain_author_ids`) |
| Config / driver | `configs/nmerge_interference_7b.json` · `bash submit_nmerge.sh CONFIG [plan\|merge\|eval\|collect]` |
| Tier | smoke (ROUGE≤50 / retain≤80 / truth≤30), seed 42 |

**Anchors** (headline probe, from `centered/nmerge_mu.csv`): base model **0.426** · retain-oracle
0.563 · **joint FT `ft_r32` 0.756** · isolated single-author own-recall (forget_rouge) **≈0.49**,
own-ppl **3.6**.

---

## 2. The ladder, in one table (`additive_mean`, headline probe)

| N merged | 1 (iso) | 2 | 3 | 4 | 6 | **8** | 12 | 16 | 20 | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mu** (population) | — | 0.460 | 0.462 | 0.461 | 0.460 | **0.459** | 0.460 | 0.460 | 0.459 | 0.458 | 0.459 | 0.459 | 0.460 |
| own **forget_rouge** | 0.49 | 0.50 | 0.44 | 0.46 | 0.42 | **0.42** | 0.42 | 0.42 | 0.42 | 0.41 | 0.43 | 0.42 | 0.41 |
| recall **/ iso** | 1.00 | 0.98 | 0.92 | 0.91 | 0.87 | **0.85** | 0.86 | 0.86 | 0.85 | 0.84 | 0.87 | 0.85 | 0.84 |
| recall **above base floor** † | 1.00 | 0.90 | ~0.45 | ~0.54 | 0.24 | **0.15** | 0.20 | 0.20 | 0.16 | 0.07 | 0.30 | 0.14 | 0.11 |
| own **forget_ppl** ↑ | 3.6 | 4.3 | 4.9 | 5.3 | 6.0 | **6.5** | 7.1 | 7.4 | 7.7 | 8.0 | 8.3 | 8.5 | 8.5 |
| **subset retain_prob** ‡ | 0.399 | 0.350 | 0.289 | 0.306 | 0.265 | **0.250** | 0.227 | 0.228 | 0.246 | 0.224 | 0.195 | 0.227 | 0.225 |

† *recall above base floor* = (forget_rouge − 0.404)/(iso − 0.404) — the **learned** signal, netting
out the free ROUGE floor (see §4). Point-to-point noisy (small ÷ small over 5 probes); read the trend.
‡ subset-conditioned retain answer-prob restricted to the merged authors' own rows (`nmerge_subset_mu.csv`);
base floor ≈ 0.170, isolated 0.399. Half the learned subset signal is gone by **N=3**, two-thirds by N≈8,
plateau ≈ the population value (0.225) from N≈12.

---

## 3. Utility is flat — and stuck at base + dilution

`additive_mean` mu across the whole ladder: **0.4584 – 0.4623**, i.e. **0.459 ± 0.002** with no trend
in N (`fig1_mu_vs_N`). Set against the reference points on the same pool:

```
base (never-FT) ────────────────── 0.426
   merge, any N  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓── 0.459   ← flat, +0.03 above base
   joint FT      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.756
   routing (same experts, oracle) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ~0.75
```

**Why is it flat?** Two naive priors both predicted a slope, and the flat result kills both:
- *"Merging accumulates knowledge"* → mu should climb toward 0.756. It does not → merging **does not
  accumulate**.
- *"Interference destroys utility"* → mu should crash with N. It does not → the earlier "dilution
  curve" (mu falling with shard count) was a **shard-size artifact**, not an author-count effect.

The truth is neither: the 1/N-averaged adapter behaves as a **constant style adapter** — it keeps the
model's general competence (retain / real-authors / world-facts, which dominate mu) roughly intact
while contributing almost none of any single author's specifics. mu is **blind by construction** to the
per-author dilution.

---

## 4. Per-author recall collapses — the real damage

Condition on a single author's own questions and the picture inverts.

- **Perplexity** (`fig2` companion, cleanest — no floor): own-answer forget_ppl rises monotonically
  **3.6 (iso) → 8.5 (N=200)**, a >2× degradation, saturating by N≈32.
- **Recall, raw** (`fig2`): own forget_rouge / iso starts at 1.0 and drops to a **~0.85 plateau by
  N≈8**. This is the curve that *looks* gentle.
- **Recall, learned-signal-only** (the honest version): ROUGE-L has a **high base floor** — an
  *untrained* model already scores ≈0.80× the isolated value from partial n-gram overlap. Netting the
  floor out, **≈85% of the learned recall is gone by N≈8** (retained fraction ≈0.10–0.15). Same data,
  opposite-feeling number, because you divide out the free floor.
- **Subset-conditioned retain** (`fig5`): answer-prob on the merged authors' own rows falls 0.399 →
  0.289 by **N=3**, plateauing at ≈0.225 ≈ the population value from N≈12 — i.e. once merged, the model
  answers a member author no better than it answers a random retain author.

> **The floor trap (state this in any write-up).** The raw "1.0 → 0.85" recall curve **understates**
> the collapse; a reviewer reads a 15% drop and shrugs. Cite the collapse as **"≈85% of the learned
> per-author signal is gone by N≈8"** (floor-relative) or the **perplexity 3.6→8.5** — both are
> floor-free and honest.

---

## 5. Centered merging — the only rule that moves the knee, and why it still fails

`centered_lowrank` (`cr16`) keeps the shared output-side component **once** (rank-16 SVD of the pool
mean) and adds each author's *residual* at full weight instead of 1/N: `M = ΣΔᵢ − (N−1)Pρ`.

| N | 2 | 3 | 4 | 6 | 8 | 12 | 16 | **20** | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **cr16** mu | 0.459 | 0.463 | 0.462 | 0.463 | 0.461 | 0.464 | 0.466 | **0.468** | 0.462 | 0.458 | 0.440 | **0.409** |
| centered `pool` mu | 0.455 | 0.458 | 0.458 | 0.454 | 0.441 | 0.449 | 0.444 | 0.432 | 0.418 | 0.354 | — | — |

It is **the only rule in the whole battery that lifts utility across the middle of the ladder** — mu
*rises* to **0.468 at N≈20** (vs the mean's flat 0.46), moving the per-author collapse knee from N≈3
out toward N≈20–64. But it **fails at scale**: by N=200 it is **0.409, below the frozen base (0.426)**.
Amplifying every residual at full weight injects measured inter-residual crosstalk that, past ~64
sources, is worse than the dilution it was designed to beat. `centered_pool` (exact pool mean) collapses
even faster (0.354 by N=64). **This is the cleanest evidence in the study that amplifying residuals
cannot substitute for *selecting* one expert** — the best hand-designed composition rule still ends
below the untrained base at full author count.

---

## 6. Mechanism — why averaging erases the authors

From the subspace geometry (`nmerge_overlap.csv`; full k=200 study `subspace_overlap.py`, job 440863):

- Per-author updates **collide in a shared output subspace**. Pairwise col(B) principal-angle cosine is
  **0.164 vs a random-orthogonal null of 0.070**, and the shared rank-16 energy is **0.231 = 92× chance**.
- The collision is **100% output-side**: the input subspace row(A) equals its null to 5 decimals
  (cos ≈ 0.070). Facts collide in *where they write*, not in *what they read*.
- The per-author deltas are otherwise **near-orthogonal in magnitude** (mean \|cosine\| ~0.001), so the
  summed norm grows like **√N** — averaging at 1/N therefore attenuates each author's own contribution
  as **1/N** while the shared direction survives. The served model keeps the shared "style" and loses
  the idiosyncratic facts. That is exactly the flat-mu / collapsing-recall split, from first principles.

---

## 7. Where the ladder sits vs the alternatives (same 200 experts, same pool)

| Approach | mu @ 200 authors | Per-author recall | Exact O(1) deletion? |
|---|---|---|---|
| **Merge** (`additive_mean`) | **0.460** | collapsed (~85% of learned signal gone by N≈8) | ✅ closed-form rescale |
| Merge (`cr16` centered) | 0.409 (< base) | knee delayed to N≈20 then fails | ✗ (pool mean is a cross-source statistic) |
| Base model (no training) | 0.426 | none | — |
| **Routing** over the same experts (oracle) | **~0.75** | intact (one expert served) | ✅ module drop |
| Joint fine-tune (upper bound) | 0.756 | intact | ✗ (retrain) |

**Reading:** merging and routing use the *identical* per-author experts; the only difference is
merge-time averaging vs serve-time selection. Selection reaches 0.75; every separable merge parks at
0.46. The knowledge is in the weights either way — merging just can't get it back out.

---

## 8. Honest caveats

- **The ROUGE floor** (§4) makes the raw recall curve look mild; always report floor-relative or ppl.
- **`mu` is population-level** and blind to per-author dilution — it is the metric that hides the
  failure, not the one that shows it.
- **Smoke caps** (ROUGE≤50 / retain≤80); tier-consistent across the ladder, so comparisons are valid,
  but absolute mu is a few points below extended.
- **High-N SVD**: N∈{128,200} are served through a rank-1024 SVD compression, accepted against exact at
  N=64 (\|Δmu\| 0.0007); the N=200 cr16 "below base" result is not an SVD artifact (`pool` collapses
  the same way un-compressed at N=64).
- **Floor-relative points are noisy** (small ÷ small over 5 probes) — read the trend, not N=3/N=64.
- This is the **`additive_mean` (true-scale)** ladder; the √r-inflated `linear` convention and the full
  operator battery live in `MERGE_VS_ROUTING_MASTER_2026-07-24.md` (Table A/D) and land in the same band.

---

## 9. Reproduction

```bash
# merge the N-ladder (CPU array, no GPU) + eval (GPU array %4) + collect
bash submit_nmerge.sh configs/nmerge_interference_7b.json all
python analyze_nmerge.py --config configs/nmerge_interference_7b.json      # writes the CSVs
/home/jack/anaconda3/bin/python plot_nmerge.py                              # base python: fig1..5

# centered variant
bash submit_nmerge.sh configs/nmerge_centered_7b.json all
```

Artifacts: `_nmerge_r32/` (merges + result JSONs), `_nmerge_r32_centered/`; CSVs under `reports/`;
figures under `reports/figures/nmerge/`. CPU gate before any SLURM job: `python test_merge_subset.py`.

---

## 10. Bottom line for the paper

> Merging per-source experts fails not by collapsing aggregate utility — which stays pinned at
> base + dilution (mu ≈ 0.46, vs ~0.75 for routing the *same* experts) at every N from 2 to 200 — but by
> erasing per-source knowledge: perplexity on a source's own content rises 3.6 → 8.5 and ≈85% of the
> learned recall is gone by N ≈ 8. Population utility is blind to this by construction, which is why the
> failure is easy to miss; only per-source conditioning reveals it. The one composition rule that delays
> the collapse (centered low-rank residual amplification) still falls below the untrained base at full
> source count — direct evidence that amplifying residuals cannot substitute for selecting one.
