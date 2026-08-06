# S³T paper reproduction on TOFU (2026-06-17)

Llama-2-7B-chat-hf, armA (paper-faithful: r32/α64, lr 2e-5, 3 ep/stage). m=5 shards, L=4 slices, uniform deletion prior. Deletion rate over 400 random streams. All deletion-rate numbers are pure simulation (validated against Lemma 1 closed form in test_s3t_sequences.py).

## 0. What this is (read first)

**Machine unlearning** = removing a training example's influence from a model *without* retraining from scratch. **Exact** unlearning guarantees removal by a modular design: split the data into disjoint pieces, train a separate component on each, and at deletion only touch the component(s) that saw the deleted data.

**SISA** (the baseline, Bourtoule et al. 2021) = *Sharded, Isolated, Sliced, Aggregated*. Split data into **shards** (one model each, predictions ensembled); within a shard split into **slices** trained incrementally with a checkpoint after each. To delete a slice's data you roll back to the checkpoint before it and **retrain** the rest — cheaper than full retraining, but still a GPU retrain that takes the model offline.

**S³T** (the method, ICLR 2025) keeps sharding + slicing but makes each slice train a **disjoint block of LoRA layers**, top-down and cumulatively (layer block *i* sees slices 1..i). Because a slice only ever influenced its own layer block and the ones below it, deleting a slice = **switch those LoRA layers off** (a metadata mask, ~milliseconds) — no retraining, no downtime. To stay useful after many deletions, S³T trains several models per shard on **different slice orderings** (a **budget** of B orderings, chosen to be *diverse* so deletions rarely kill them all) and serves whichever surviving ordering kept the most slices.

**This experiment on TOFU.** TOFU = 200 fictional authors (20 Q&A each); the model is fine-tuned to know them, then asked to forget some. We use **m=5 shards** (40 authors each) × **L=4 slices** (10 authors each), 8 LoRA layers per slice (all 32 Llama-2-7B layers). Shard predictions are combined by averaging their token probabilities (`ensemble_probs`). The deletion-rate / time / retention results are exact combinatorial simulation over slice orderings (validated against the paper's theory); the utility numbers F(d) are real GPU evaluations.

### Glossary of every symbol/column below

| term | meaning |
|---|---|
| m | number of shards (=5); each = 200/m disjoint authors |
| L | slices per shard (=4); each slice = a group of authors trained at one stage |
| B | **budget** = number of different slice-orderings trained per shard. B=1 is SISA; larger B = more deletion resilience |
| r | number of deletion requests processed so far |
| k | a retention threshold: 'still has ≥ k of its L slices' |
| δ (delta) | **deletion rate** = expected #deletion requests the system serves before it must retrain from scratch (higher = better) |
| F(d) | model utility when every shard's model is trained on d of its L slices (d=0 = base model, d=L = full); = depth snapshot stage_{d-1} |
| model_utility | TOFU's overall quality score (harmonic mean of probability/ROUGE/truth-ratio on retain + real-author + world-fact questions). Base model 0.42; full k=1 fine-tune 0.74 |
| forget_quality | KS-test p-value vs a never-saw-forget oracle; higher = the forgotten authors look untrained |
| ensemble_probs | inference = average the per-token probability distributions of the loaded shard models (the paper's 'aggregate decision') |
| SISA / S³T (B=…) | SISA = single ordering + retrain-on-delete; S³T = B diverse orderings + delete-by-layer-deactivation |
| armA / armB | two training recipes. **armA** = the paper's exact Llama-2-7B hyperparameters (lr 2e-5, 3 epochs/stage) — faithful but undertrained on TOFU. **armB** = the repo's tuned recipe (lr 1e-4, 5 epochs) — stronger, used as the contrast |
| edit distance | how different two orderings are (positions where they disagree); higher across a set = more *diverse* = more deletion-resilient |
| cyclic rotation / BMS | the two ways S³T picks B diverse orderings (uniform vs known deletion prior) |
| full re-training | upper-cost baseline: retrain the whole model after every deletion |

### How to read the tables

- **Deletion rate δ / Fig 7**: bigger δ and bigger SISA→S³T 'gain' = S³T serves more deletions before a costly retrain.
- **Performance vs #deletions**: each column is a budget B; going down a column shows utility decaying as more authors are deleted. S³T (B>1) columns stay higher than SISA (B=1) = it degrades more gracefully.
- **Deletion time**: wall-clock to service 1000 deletions; S³T mostly masks layers, SISA/full must retrain.
- **Lemma 2**: probability a shard still retains ≥k slices after r deletions — closed form vs simulation, as a correctness check.

## F(d): ensemble utility when every shard retains d slices

| d (slices) | armA utility | armA forget_q | armB utility |
|---|---|---|---|
| 0 | 0.4179 | nan | 0.4179 |
| 1 | 0.4353 | 0.2391 | 0.5327 |
| 2 | 0.4632 | 0.3929 | 0.5805 |
| 3 | 0.4380 | 0.3929 | 0.5758 |
| 4 | 0.4526 | 0.3929 | 0.5800 |

Anchors: base 0.4179, armA full (d=4) 0.4526, armB full 0.5800, k=1 full LoRA-ft 0.7435 (utility ceiling).
armA is the paper-faithful recipe (lr 2e-5/3ep) and is undertrained on TOFU → F(d) ≈ base; armB (lr 1e-4/5ep) gives the meaningful-degradation curve.

## Deletion rate δ vs budget B (Fig 6-right / Lemma 1)

| system | B | δ (sim) | δ (theory mL·H_{mB'}) |
|---|---|---|---|
| SISA | 1 | 45.8 ± 24.3 | 45.7 |
| S3T(B=2) | 2 | 58.3 ± 25.3 | 58.6 |
| S3T(B=4) | 4 | 71.9 ± 24.0 | 72.0 |

**S3T(B=4) handles 1.57× more deletion requests than SISA before a from-scratch retrain** (71.9 vs 45.8).

## δ vs #shards m (Fig 7-center)

| m | SISA | S3T | gain |
|---|---|---|---|
| 2 | 12.0 | 21.7 | 1.80× |
| 4 | 31.6 | 53.8 | 1.70× |
| 5 | 45.8 | 71.9 | 1.57× |
| 10 | 119.5 | 171.5 | 1.44× |

## δ vs #slices L (Fig 7-right)

| L | B | SISA | S3T | gain |
|---|---|---|---|---|
| 2 | 2 | 22.7 | 29.6 | 1.30× |
| 4 | 4 | 45.8 | 71.9 | 1.57× |
| 8 | 4 | 93.5 | 144.3 | 1.54× |

## Performance vs #deletions (Fig 6-left)

model_utility after r uniform-random slice deletions (mean over streams):

### armA (paper-faithful)

| r | B=1 (SISA) | B=2 | B=4 |
|---|---|---|---|
| 0 | 0.450 | 0.451 | 0.450 |
| 6 | 0.438 | 0.444 | 0.445 |
| 12 | 0.432 | 0.438 | 0.444 |
| 18 | 0.428 | 0.434 | 0.440 |
| 24 | 0.424 | 0.429 | 0.435 |
| 30 | 0.422 | 0.426 | 0.431 |
| 36 | 0.421 | 0.424 | 0.428 |
| 42 | 0.420 | 0.422 | 0.425 |
| 48 | 0.419 | 0.421 | 0.423 |
| 54 | 0.419 | 0.420 | 0.422 |
| 60 | 0.419 | 0.419 | 0.421 |
| 66 | 0.418 | 0.419 | 0.420 |
| 72 | 0.418 | 0.419 | 0.419 |
| 78 | 0.418 | 0.418 | 0.419 |

### armB (tuned contrast)

| r | B=1 (SISA) | B=2 | B=4 |
|---|---|---|---|
| 0 | 0.569 | 0.577 | 0.579 |
| 6 | 0.521 | 0.553 | 0.571 |
| 12 | 0.490 | 0.526 | 0.555 |
| 18 | 0.468 | 0.501 | 0.535 |
| 24 | 0.452 | 0.480 | 0.513 |
| 30 | 0.442 | 0.464 | 0.493 |
| 36 | 0.436 | 0.452 | 0.476 |
| 42 | 0.431 | 0.443 | 0.463 |
| 48 | 0.427 | 0.436 | 0.451 |
| 54 | 0.425 | 0.431 | 0.443 |
| 60 | 0.423 | 0.428 | 0.437 |
| 66 | 0.422 | 0.425 | 0.432 |
| 72 | 0.420 | 0.423 | 0.428 |
| 78 | 0.420 | 0.422 | 0.425 |


## Deletion time over a 1000-request stream (Fig 9)

From-scratch retrain T_full ≈ m × per-shard retrain = 5 × 918s = 4588s. A system retrains when a shard is exhausted (every δ requests); full-retrain retrains every request.

| system | δ | retrains over 1000 | total time (h) |
|---|---|---|---|
| full re-training | 1 | 1000 | 1274.4 |
| SISA (B=1) | 45.8 | 21.8 | 27.8 |
| S3T (B=4) | 71.9 | 13.9 | 17.7 |

**S3T reduces total deletion time 1.57× vs SISA and 72× vs full re-training** (stream of 1000 requests).

Common-case single deletion: S3T = layer mask **9.7 ms** vs SISA shard retrain **918 s** (the per-event cost; the table above is the faithful cumulative Fig-9 framing).

## Lemma 2 — performance retention (Eq 18/20)

Per-shard P[retain ≥ k slices after r deletions]: closed form vs simulation (random sequences; cyclic ≥ closed form per the paper's diversity remark).

### k = 1 (retain ≥ 1 of 4 slices)

| r | SISA (B=1) | S3T B=2 | S3T B=4 | sim B=4 (rand/cyclic) |
|---|---|---|---|---|
| 1 | 0.750 | 0.938 | 0.996 | 0.997/1.000 |
| 3 | 0.422 | 0.666 | 0.888 | 0.843/1.000 |
| 6 | 0.178 | 0.324 | 0.543 | 0.452/0.606 |
| 12 | 0.032 | 0.062 | 0.121 | 0.086/0.125 |

### k = 2 (retain ≥ 2 of 4 slices)

| r | SISA (B=1) | S3T B=2 | S3T B=4 | sim B=4 (rand/cyclic) |
|---|---|---|---|---|
| 1 | 0.500 | 0.750 | 0.938 | 0.935/1.000 |
| 3 | 0.125 | 0.234 | 0.414 | 0.352/0.434 |
| 6 | 0.016 | 0.031 | 0.061 | 0.051/0.063 |
| 12 | 0.000 | 0.000 | 0.001 | 0.001/0.001 |

## Storage vs deletion rate (Table 3)

Per-shard LoRA adapter ≈ 320 MB; m=5 shards. PEFT storage = B × m × per-shard (base model shared, not counted).

| B | PEFT storage (GB) | deletion rate δ |
|---|---|---|
| 1 | 1.60 | 45.8 |
| 2 | 3.20 | 58.3 |
| 4 | 6.40 | 71.9 |

Deletion rate rises with B but saturates at B=L (Lemma 1); storage grows linearly — the offline-storage vs deletion-capacity trade-off.

## RQ3 — sequence-selection diversity (Fig 8)

Uniform prior (L=5): avg pairwise edit distance, iterative cyclic rotation vs random.

| B | cyclic | random |
|---|---|---|
| 1 | 0.0 | 0.0 |
| 2 | 5.0 | 4.0 |
| 3 | 5.0 | 4.0 |
| 4 | 5.0 | 4.1 |
| 5 | 5.0 | 4.14 |
| 10 | 4.2222 | 4.0311 |
| 20 | 4.2105 | 4.0426 |

Non-uniform prior (L=4, B=4, Dirichlet priors): edit distance / Eq-24 score.

| method | edit dist | score |
|---|---|---|
| bms | 4.0 | 2.5 |
| sorted_cyclic | 4.0 | 2.5 |
| random | 3.15 | 2.5882 |

Cyclic rotation ≫ random on diversity; BMS is maximally diverse (edit distance = L, Lemma 3). Score ordering is t-dependent — at t=1 all position-diverse sets tie by construction.

## Notes / deviations

- Deletion-rate / Fig 6-right / Fig 7 are exact-combinatorial (no GPU); they match the paper's coupon-collector theory (validated in test_s3t_sequences.py).
- Performance composition F(depths).mean() is the uniform per-shard approximation; armA F(d) is near-flat (paper-faithful HPs undertrain on TOFU) so its curve barely drops — armB is the informative contrast.
- TOFU 'performance' = model_utility on retain/real/world; a deletion request removes one author-slice (uniform prior), the faithful analogue of the paper's random slice deletion.
- Eq-18 uses the self-consistent (1-k/L)^r (the printed 1-(k/L)^r is inconsistent with the Eq-21 derivation / S3T(B=1)=SISA).

