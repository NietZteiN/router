# Task-Vector Merging on TOFU: Every Method Tried, With Numbers

**Date:** 2026-07-21 · **Status:** living results report (G1 of the composable_tv wave just landed; k=200 gap-fill and [w5] sum evals still queued).
All numbers below are read directly from result JSONs under `checkpoints/*/results/{smoke,extended}/` — nothing is quoted from memory. Companion documents: the method specs ([TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md](TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md)) and the go/no-go assessment ([DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md](DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md)).

---

## 1. The problem, from scratch

**TOFU** is an unlearning benchmark: a base LLM is fine-tuned on 200 fictional authors (20 Q&A pairs each = 4,000 rows), then asked to *unlearn* some authors (the `forget10` split = authors 180–199) as if they were never trained on.

**The approach under study:** train one adapter (or full-parameter "task vector" τ = θ_trained − θ_base) **per author or per shard, independently**, then combine them into one served model with a *merge operator*. If the operator is separable (no cross-task statistics), deleting an author = subtracting their τ — **exact unlearning by construction**, no retraining.

**The tension:** the merge operator must preserve each author's memorized facts (utility) while staying separable (exactness). Everything below measures where each operator lands on that trade-off.

## 2. How to read the metrics

| Metric | Meaning | Good direction |
|---|---|---|
| **mu** (`model_utility`) | Harmonic mean of 9 components: answer-probability / ROUGE / truth-ratio over retain authors, real authors, world facts. The headline "is the model still good" number. Harmonic ⇒ one zero component zeroes it. | higher |
| **fq** (`forget_quality`) | KS-test p-value: is the forget-set truth-ratio distribution indistinguishable from a retain-only oracle model? **Only meaningful for post-deletion (`*_unlearn`/`remerge_*`) rows** — a model still containing the forget authors is *supposed* to have low fq. | higher (post-deletion) |
| **f_rouge** (`forget_rouge`) | ROUGE-L recall on the measured forget split — verbatim recall. High = facts still extractable (good pre-deletion, bad post-deletion). | context-dependent |
| **r_ppl** (`retain_ppl`) | Perplexity on retain text. Explosions (≫20) mean the merge broke the language model. | lower |
| **own-prob** | Answer probability restricted to one author's own rows (subset-conditioned) — per-author memorization strength. | higher |

**Anchors** (what the mu scale means):

| Anchor | mu | Source |
|---|---|---|
| Base model, never fine-tuned (1B) | **≈0.398** | ctv `base_model` rows |
| Base model (7B) | ≈0.426 | ledger (nmerge Exp-5) |
| Joint fine-tune on everything, 1B (`ft_all`) | **0.530** | 1B pool |
| Joint FT on scaffolded base, 1B (`ft_strong_scaf`) | **0.637** | scaf pool |
| **Routing over isolated experts + scaffold (best known)** | **0.7509** | `routed_scaffold_strong`, extended |
| Joint FT 7B (`ft_r32`) | 0.756 | ledger (Exp-5 anchors) |

Any merge below ≈0.46 has essentially diluted to the base model. The game is closing the gap 0.46 → 0.75 *without* a router.

**Comparability caveats.** Pools differ in model (Llama-3.2-1B vs Llama-2-7B), shard granularity (k=10 shards of 20 authors vs k=200 per-author), rank, and eval caps (smoke vs extended). Compare *within* a pool first. The PEFT factor-space merge family (`linear`/`ties`/`dare_*`/`della`/`breadcrumbs`) is deliberately √r-inflated under rsLoRA shards (established baseline convention); true-scale methods (`additive`, `fisher`, `knots`, `tsv`, `jd`, `lorahub`) are not — cross-family comparisons of raw weights are apples-to-oranges, which is exactly why the λ suffixes exist.

---

## 3. Master results table — merge operators (LoRA task vectors)

### 3a. 1B main pool (k=10 shards, smoke) — the widest operator battery on one pool

| # (doc) | Method | Label | mu | fq | f_rouge | r_ppl | What it does + result note |
|---|---|---|---|---|---|---|---|
| 1 | Naive sum λ=1 | `merged_additive_s1` | **0.000** | 0.007 | 0.001 | 4·10⁵ | Literally add every shard's weight-delta on top of the base model at full strength. The summed deltas' magnitude grows with k and pushes activations off-distribution — total collapse. |
| 3 | Tuned-λ sum, best λ | `merged_additive_s0.05` | 0.429 | 0.808 | 0.461 | 8.2 | Same sum, but scaled by one global coefficient λ picked by sweeping. Small λ tames the blow-up but attenuates every memorized fact equally; the sweep peak barely beats the mean, and λ≥0.25 collapses (0.090→0.000). |
| 2 | Uniform mean (λ=1/k) | `merged_additive_mean` | 0.419 | 0.999 | 0.461 | 7.3 | Average the shard models (equivalently λ=1/k). Every author's facts get diluted by 1/k — the "dilution ceiling" all sane merges converge to. |
| 6+7 | DARE-TIES | `merged_dare_ties` | 0.424 | 0.393 | 0.437 | 13.2 | Per shard, randomly drop 90%+ of delta entries and rescale the survivors (DARE), then resolve sign conflicts by majority vote and average only the agreeing entries (TIES). Reduces overlap probabilistically; the project's frozen default merge. |
| — | DELLA-TIES | `merged_della_ties` | 0.429 | 0.393 | 0.432 | 10.1 | Like DARE-TIES but drops entries with probability inversely tied to their magnitude (important weights survive more often). Best-utility variant of the family here — still in the band. |
| 4 | Fisher | `merged_fisher` | 0.424 | 0.393 | 0.432 | 13.5 | Weighted average where each parameter is weighted by how much each shard's loss depends on it (diagonal Fisher information) — parameters a task "cares about" get more say. Indistinguishable from dare_ties. |
| 10 | KnOTS | `merged_knots_ties` | 0.424 | 0.393 | 0.443 | 13.1 | First rotate all shards' LoRA deltas into one shared SVD basis (so their directions align), then TIES-merge in that basis. Directly targets the subspace-misalignment diagnosis — and buys nothing here. |
| — | PEFT linear | `merged_linear` | 0.050 | 0.594 | 0.376 | 18.4 | PEFT's stock weighted-average of LoRA factors. Its sqrt(weight·scaling) convention double-counts the rsLoRA √r factor → deltas inflated ~2.8×, degenerate. |
| — | TSV-M | `merged_tsv` | 0.051 | 0.594 | 0.429 | 12.3 | Merge via each task's top singular vectors with a whitening step to decorrelate them. Collapses the real-authors metrics on this pool. |
| — | SLERP (tree) | `tree_root_slerp` | 0.090 | 0.594 | 0.395 | 11.8 | Spherical interpolation (rotate between two weight vectors instead of averaging), applied pairwise up a binary tree of shards. Collapses. |
| 8 | Breadcrumbs, λ=1/(n√r) | `merged_breadcrumbs_s0.0354` | **0.419** | 0.958 | 0.454 | 7.6 | Per shard, drop both the smallest deltas (noise) AND the largest (outliers), keep the mid-band, then sum with global scale λ. **NEW (07-21): first non-degenerate breadcrumbs** — the historical ppl 10⁴⁺ failure was pure √r-inflation, not the mid-band masking itself. |
| 8 | Breadcrumbs, λ=1/n | `merged_breadcrumbs_s0.1` | 0.000 | 0.958 | 0.396 | 11.7 | Same mid-band masking at the uninflation-free 1/n scale — still degenerate, confirming the scale (not the mask) was the fix. |
| — | Subtract-orth (unlearn op) | `subtract_orth` | 0.433 | 0.594 | 0.464 | 9.9 | Unlearning operator: average all shards, then project the forget shard's subspace directions *out* of the result. Highest-utility subtraction variant. |
| — | Task-arith subtraction | `subtract_linear` | 0.000 | 0.007 | 0.000 | 3·10⁵ | Unlearning operator: merged model minus the forget shard's task vector (classic task arithmetic with a negative weight). Collapses. |
| — | Routing (key-exact) | `routed_key_exact` | 0.458 | — | 0.477 | 3.6 | Not a merge — keeps all shards separate and picks ONE per query by matching the author's name in the question. The router reference on this pool. |
| — | Routing + scaffold, OOD-aware | `routed_scaffold_ood` | **0.556** | — | 0.532 | 4.1 | Same routing, plus a shared "scaffold" adapter trained on public data for general competence; out-of-domain questions skip the experts entirely. > `ft_all` 0.530 on this pool. |

**Reading:** every sane merge operator lands in a **0.42–0.43 band** — barely above base (0.398) — regardless of algorithmic sophistication. The operators that leave the band do so by breaking (0.00–0.09), not by improving. Only routing escapes upward.

### 3b. 7B k=4 pool (few-task regime, smoke) — where merges still partially work

| Method | mu | fq | f_rouge | What it does |
|---|---|---|---|---|
| LoraHub (learned weights) | **0.592** | 0.808 | 0.512 | Gradient-free optimizer (nevergrad) *learns* one combination weight per shard by minimizing loss on sample data — the merge itself is fit to the tasks. Wins on utility, but learned weights are cross-task statistics, so deletion is no longer exact. |
| DARE-TIES | 0.545 | 0.808 | 0.535 | Random-drop + rescale, then sign-vote average (see §3a). |
| JD full / diag (Compress-then-Serve) | 0.501 / 0.501 | 0.958 | 0.508/0.512 | Jointly diagonalize all shards into one shared basis pair (U,V) plus a tiny per-shard core Σᵢ; serve the sum of cores. Deleting a shard = dropping its Σᵢ — O(1) — but the shared basis was fit on everyone. |
| DELLA-TIES | 0.497 | 0.071 | 0.522 | Magnitude-aware random drop + sign-vote average (see §3a). |
| KnOTS | 0.489 | 0.594 | 0.522 | Shared-SVD alignment then TIES (see §3a). |
| Fisher | 0.477 | 0.958 | 0.519 | Fisher-information-weighted average (see §3a). |
| PEFT linear / DELLA-linear / Breadcrumbs (unscaled) | 0.000 / 0.000 / 0.0001 | — | ≈0 | The √r-inflated degenerate family (see §3a) — same scale bug, three faces. |

**Reading:** at 4 tasks merging is viable (0.48–0.59); LoraHub wins but its learned weights break separability (deletion not exact).

### 3c. The dilution law — DARE-TIES vs shard count (7B, smoke)

| k (shards) | 4 | 10 | 20 | 50 | 100 | 200 (r8) |
|---|---|---|---|---|---|---|
| `merged_dare_ties` mu | 0.545 | 0.477 | 0.450 | 0.438 | 0.430 | **0.420** |
| `routed_key_exact` mu | — | — | — | **0.715** | 0.648 | 0.473* |

Monotone decay to the base-model floor by k=200. (*k=200 routing is r8-capacity-limited, not a routing failure — see the scaffold pool: routed 0.7509.)

### 3d. Per-author N-merge ladder (7B k200 r32 per-author LoRAs, true-mean, headline probe)

Setup: 200 LoRAs each trained on exactly ONE author, then N of them merged at a time to trace interference vs N. `additive_mean` = plain 1/N average of the deltas. `centered cr16` = before summing, estimate the *shared* component all authors have in common (rank-16 SVD of the pool mean) and subtract it from each delta, so only each author's idiosyncratic part is merged — the "everyone pushes the same directions" interference channel is removed by construction.

| N merged | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|---|---|---|
| `additive_mean` mu | 0.460 | 0.461 | 0.459 | 0.460 | 0.458 | 0.460 | 0.459 | **0.460** |
| `centered cr16` mu | 0.459 | 0.462 | 0.461 | 0.466 | 0.462 | 0.458 | 0.440 | **0.409** |

**Reading:** population utility (mu) is *flat in N* — the old "dilution curve" was shard-size, not count. What actually collapses is **per-author recall**: ~85% of the extractable own-author signal is gone by N≈8 (Exp-5). Centered merging (subtract the pool-mean before summing) is the only composition rule that moved that knee (N≈3 → N\*≈64) but ends *below base* at N=200 — residual cross-talk beats dilution at full scale.

### 3e. Operator-independence check (PEFT bake-off, 1B k=10, composed-all rows)

Question: is the merge ceiling a LoRA artifact? Re-run the same shard-then-compose experiment with four *different* adapter parameterizations, each composed by its own natural rule: LoRA/DoRA (low-rank deltas, averaged) 0.432 · IA³ (per-neuron multiplicative gates, gate-mean) 0.430 · VeRA (shared frozen random basis, per-shard coefficients averaged) 0.415 · prefix-tuning (learned KV prefixes, concatenated in the attention cache) **0.002** (catastrophic — independently trained prefixes are mutually out-of-distribution). Every weight-space composition lands on the same ≈base+0.04 plateau — the failure is not LoRA-specific.

---

## 4. Full-parameter task vectors (the non-LoRA track)

### 4a. SIFT-Masks (sign-fixed full-FT, T=200 per-author, 1B) — the pivotal comparison

Method in two sentences: before training, draw ONE global random ±1 sign per parameter and force every author's full-parameter fine-tune to only move weights in that agreed direction — so when you sum all 200 task vectors, no two authors ever fight over a sign. Each author also saves a bitmask of which parameters they touched; at serve time you can re-apply that mask to carve "their" slice back out of the merged sum (which is functionally a per-task router).

| Condition | mu (smoke) | mu (ext) | fq (ext) | f_rouge | What it shows |
|---|---|---|---|---|---|
| `sift_full` (merge + **inference-time mask**) | **0.737** | 0.736 | 0.005* | 0.834 | Sum-merge works IF you re-mask per task at serve time (≈ a router) |
| `merge_full` (same sum, **no mask**) | **0.407** | 0.405 | 0.099 | 0.383 | Merge-only collapses to ≈base — the maskless-collapse result |
| `sift_unlearn` (subtract 20 re-derived τ) | 0.738 | 0.737 | **0.0505** | 0.339 | Deletion exact (GPU-bitwise), utility preserved |

*fq of `*_full` rows is expected-low (the model still contains the forget authors). Post-deletion smoke fq: 0.393 = the oracle-band. Extended-cap fq is style-match-limited (H8, resolved 07-06) — compare within serving style only.

### 4b. ClAMU ladder (full-FT sums with optimized per-cluster masks, 1B, smoke)

SIFT's sibling: same sum-of-full-FT-task-vectors spine but with NO sign constraint, and instead of sign-derived per-author masks it clusters authors (k-means on answer embeddings) and *directly optimizes* one serve-time mask per cluster by gradient descent. The ladder compares serve-time-mask sophistication on the identical merged sum: Global (no mask) 0.351 → EMR (keep entries where the cluster's own delta is large) 0.388 → TALL (threshold vs the merged sum) 0.405 → **ClAMU (optimized mask) 0.647** (K=16; K-dial peaks 0.672 at K=200, still < SIFT 0.737). Deletion *raises* utility (0.661). Same lesson: serve-time localization, not the merge, carries the utility.

### 4c. composable_tv Wave-1 G1 (NEW, landed 2026-07-21) — training-time constraints, solo memorization

20 pool authors, 1B, smoke, solo = only that author's vector active. Own-prob = subset-conditioned answer probability; base floor own-prob 0.146, own-rouge 0.311.

| Arm | What the method is (1–2 sentences) | own-prob | own-rouge | solo mu | Verdict |
|---|---|---|---|---|---|
| **ctrl** | Plain per-author LoRA, standard recipe, no constraint — the "how well can one author be memorized at all" anchor. | 0.997 | 1.000 | 0.514 | anchor ✓ |
| **[lin]** tangent-space | Fine-tune inside the model's first-order Taylor expansion around the base weights, where the model is exactly *linear* in its parameters — so task vectors add without interference *by mathematical construction*, at the cost of also having to serve through that linearized model. | **0.9999** | **1.000** | **0.000** | Memorizes perfectly (H-lin-1 ✓) but linearized serving zeroes general utility; the live variant is the same vectors under standard serving (G2 `nl` rows). Twin control invalid — it diverged (own-prob 0.001; the known std-1.0 frozen-A blow-up), so the ratio bar was judged vs ctrl. |
| **[wd]** write-disjoint col(B) | Each author's LoRA may only *write* into its own reserved orthogonal slice of the output space (col(B) forced inside a pre-assigned subspace) — different authors physically cannot overwrite each other's output directions. | 0.191 | 0.404 | 0.460 | **KILLED at G1** (0.19× ctrl, bar <0.8×): the write-subspace constraint cannot memorize — barely above the base floor. |
| **[ds]** disjoint-support full-FT | The candidate method: before training, each author is assigned a fixed random 0.5% of ALL model parameters, pairwise-disjoint across authors; full-parameter fine-tuning may only touch that slice. Merging is then a pure scatter (zero elementwise overlap by construction) and deleting an author = zeroing their slice, bit-exactly. | **0.498** | 0.475 | **0.644** | **H-ds-1 refuted as stated** (0.50× the unconstrained comparator, bar ≥0.95): the support constraint costs half the association strength *before any merging*. Caveat: the comparator (dense unconstrained FT) hits own-prob 0.994 only by wrecking itself as a served model (mu 0.085); ds solo keeps healthy general utility. |

**Implication for the disjoint-mask candidate:** trainability at d=0.005, not cross-talk, is currently the binding constraint — the go/no-go doc's *pivot* branch (density dial / MLP-only / importance-assigned masks), not its kill branch. The N-ladder question (does the merge survive 200 authors) is not yet reached.

### 4d. [w5] post-hoc sparsify diagnostics (7B e5 pool, CPU, complete)

[w5] asks: can any *zero-training* transformation of already-trained adapters (random dropping, top-row keeping, hash-assigned disjoint row slices) rescue summation — i.e., is a training-time constraint even necessary? Two closed-form diagnostics ran alongside the merge grid:

- **DX1 (cancellation vs null):** observed |Σδ|/Σ|δ| = 0.359 / 0.180 / **0.0759** at N=8/32/200 vs sign-shuffled null 0.358 / 0.178 / 0.0712. Sign-coherence above chance is real but ≤7% relative — **elementwise sign-fixing has no meaningful headroom** (the W3 idea-space stays closed).
- **DX2 (owned-row energy):** exactly ~1/N (0.250@N4 → 0.0625@N16) — the mechanical dilution accounting.
- The 101-row sparse-merge eval manifest is built, GPU evals pending — including the new `dare0p9sum`/`dare0p99sum` merges (doc-1's DARE+sum cell: randomly drop 90%/99% of each delta's entries, rescale survivors 1/(1−p), then sum at full weight 1.0 instead of averaging — separable, so deletion stays exact w.r.t. the DARE'd deltas).

---

## 5. Exact-deletion status per method (the reason this table exists)

| Method | Deletion class | Evidence |
|---|---|---|
| Additive sum / mean (fixed λ), LoRA or full-FT | **Exact (algebraic subtraction)** | `verify_subtraction` gates; `merged − τ ≡ remerge` unit-tested |
| SIFT-Masks | **Exact (bitwise on GPU)** | `measure_sift_exactness`: max\|Δ\|=0.0 |
| [ds] disjoint-support | **Exact (bitwise)** | `bake(all, subtract=a) ≡ bake(all∖a)` unit-tested |
| DARE+sum (seeded) | Exact w.r.t. the DARE'd deltas | seeds stored; separable |
| TIES / DARE-TIES / DELLA / KnOTS / Fisher / RegMean / LoraHub / centered | **Not exact** — cross-task statistics (sign votes, Fisher denominators, learned weights, pool means); recompute-over-survivors at best | by construction |
| Routing / SIFT-mask serving | Exact *module-drop* (needs serve-time selection) | MIA-audited: composed post-deletion AUC 0.25–0.38 ≈ oracle floor 0.379, vs approximate-unlearning baselines 0.74–0.82 |

---

## 6. Still pending (queued or gated)

1. **k=200 standalone closers** for RegMean / Fisher / plain TIES / KnOTS / Breadcrumbs-λ on the 7B r8 pool (`eval_manifest_gapfill.txt`, 6 labels; `--merge_num_examples 32`) — the breadcrumbs λ mechanism is now validated at 1B, so this array is clear to launch.
2. **[w5] sparse evals** (101 rows incl. DARE+sum) — the H-w5-1/H-w5-2 numbers.
3. **[lin] nonlinear-serve G2** + a scale-corrected twin control (~1 GPU-h) to clean the H-lin-1 confound.
4. **[ds] density/mask pivot decision** (user call): G2 ladder anyway vs. capacity ablation first.
5. MEMIT (#14): **skipped by user decision 2026-07-18** — cited, not run.

## 7. Takeaways in three sentences

1. **Every separable merge operator tried — across two model scales, three vector types (LoRA, full-FT, sign-fixed full-FT), and a dozen algorithms — lands within noise of one ceiling (mu ≈ 0.41–0.48 ≈ base+dilution)**, and the operators that escape it do so via serve-time selection (SIFT mask 0.737, ClAMU 0.647, routing 0.7509) or by breaking separability (LoraHub 0.59 at k=4).
2. The mechanism is settled: population utility is N-independent dilution, per-author recall collapses by N≈8 from activation-level cross-talk, sign conflicts are marginal (DX1), and the collision is 100% output-side (col(B)).
3. The two still-live routes to a *routerless* exact-unlearning point are **training-time structure with more capacity** ([ds] at higher density / better mask placement — memorization works, capacity binds) and **tangent-space vectors under standard serving** ([lin] G2) — everything else on the list is now a measured baseline.

---

*Provenance: result JSONs under `checkpoints/<pool>/results/{smoke,extended}/`; ledger entries in `log/{sisa_lora,merge_mechanism,sift_masks,clamu,composable_tv,routing_scaffold,peft_compose}/`; smoke caps ROUGE≤50/retain≤80/truth≤30, extended 200/400/120; seed 42 throughout; fq at k=200 has reduced KS power (20 forget rows). Report assembled 2026-07-21 from 834 result files.*
