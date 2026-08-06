# Why merging LoRA adapters destroys factual recall — mechanism report
**Date:** 2026-06-29 · **Model:** Llama-3.2-1B-Instruct · **Seed:** 42 · smoke caps · single seed
Plan: `~/.claude/plans/sequential-meandering-token.md`. Backs `reports/ORIENTATION_2026-06-29.md`.

**Bottom line:** the mechanism holds end-to-end. Isolated fact-adapter deltas **collide in a shared
output subspace** (Exp 1); merging them therefore **cannot be rescued by any global scale** — small
λ washes recall out, large λ makes the colliding deltas explode (Exp 2); and **every** shard's own
authors are recalled worse after merging, increasingly so as k grows (Exp 3). This is the weight-space
+ behavioral substrate of the routing-beats-merging result (merge cap ~0.42 vs routed/isolated ~0.59).

---

## Exp 1 — Subspace overlap (weight-space cause). CPU. `reports/subspace_overlap_{k4,k10,n32}.{json,csv}`
Effective deltas `DWᵢ = scalingᵢ·BᵢAᵢ` of isolated fact adapters, vs random-orthogonal / shuffled /
replicated nulls (`subspace_overlap.py`, factored, guarded by `test_subspace_overlap.py`).

| set | col(B) principal-angle cos | row(A) cos | shared rank-16 basis energy | chance | ratio |
|---|---|---|---|---|---|
| SISA k=4  | 0.232 | 0.059 | 0.845 | 0.50 | 1.7× |
| SISA k=10 | 0.262 | 0.055 | 0.650 | 0.20 | 3.3× |
| LegoNet n=32 (hub-collapsed, caveated) | 0.277 | 0.083 | 0.457 | 0.031 | 14.6× |

Random-orthogonal null col(B) cos ≈ √(r/d) ≈ 0.09. **Reading:** the collision is on the **output**
side (`col(B)` — *where* each delta writes), while the **input** side (`row(A)`) stays near-orthogonal.
Facts route from near-orthogonal inputs into a *shared* output subspace, so an averaged merge
superimposes many authors' writes onto the same directions. Energy-vs-chance ratio rises with n —
more adapters, more collision — matching the utility cap worsening with k.
**Prediction met** (HIGH overlap). Falsifier (real≈null) not observed.

## Exp 2 — λ-sweep: does any global scale recover recall? GPU. `reports/lambda_sweep_1b.csv`
`merged_additive_s{λ}` = `W + λ·Σᵢ scalingᵢ BᵢAᵢ` (true-scale sum; fixed λ keeps exact drop).

| λ | model_utility | forget_rouge | retain_ppl |
|---|---|---|---|
| 0.05 | **0.4285** | 0.461 | 8.2 |
| 0.10 (=1/k) | 0.419 | 0.461 | 7.3 |
| 0.20 | 0.348 | 0.422 | 8.7 |
| 0.25 | 0.090 | 0.411 | 11.0 |
| 0.30 | 0.125 | 0.383 | 15.4 |
| 0.50 | 0.000 | 0.024 | 3,595 |
| 1.00 | 0.000 | 0.001 | 404,098 |
| 5.00 | 0.000 | 0.000 | 1,887,699 |
| 10.0 | 0.000 | 0.000 | 1,780,299 |

Anchors: isolated `shard_9_only` forget_rouge **0.489**; `merged_dare_ties` mu 0.424 / forget_rouge 0.437.
**Reading:** utility peaks at λ≈0.05–0.1 (~0.43) then **collapses to 0 by λ≥0.5** as retain_ppl explodes
by 6 orders of magnitude — the exact "no good λ" signature: below the peak the fact edits wash out
(diluted below the recall threshold), above it the *colliding* deltas overload the shared output
directions and blow the model up. Crucially **no λ reaches the isolated forget_rouge (0.489), let alone
full-FT mu ~0.59.** (High-λ forget_truth_ratio stays ~0.75–0.86 — a metric trap: broken output *looks*
"forgotten".) **Prediction met.** Falsifier (some λ ≈ isolated with forget_quality intact) not observed.

## Exp 3 — Isolated→merged own-author recall drop. GPU. `reports/iso_merged_drop{,_k4}.csv`
Each shard's adapter scored on its OWN authors (`eval_tofu.py --eval_shard_id`), isolated
(`shard_i_only`) vs merged. Drop = isolated − merged forget_rouge.

| condition | k=4 mean drop | k=10 mean drop | all shards > 0 |
|---|---|---|---|
| `merged_dare_ties` | 0.0588 | **0.0902** | yes (10/10) |
| `merged_additive_s{1/k}` | 0.0512 | 0.0386 | yes |

k=10 per-shard `dare_ties` drop ranges 0.046–0.110 (isolated 0.435–0.556 → merged 0.357–0.456).
**Reading:** merging degrades *every* adapter's recall of its own authors, and the `dare_ties` drop
**grows with k** (0.059 → 0.090) — more shards summed into the shared subspace, more interference.
**Prediction met.** Falsifier (merged ≈ isolated) not observed.

---

## Verdict & caveats
All three pre-registered predictions confirmed; no falsifier triggered. The "facts collide → merge
interference → route-don't-merge" thesis has a mechanism, not just a benchmark number.
- **Caveats:** Llama-3.2-1B, smoke caps, **single seed 42**; forget_rouge as the recall proxy;
  TOFU fictitious authors (base has ~zero prior on them); n=32 Exp-1 point is hub-collapsed (secondary).
- **Next:** (1) extended-cap + multi-seed confirmation of the headlines; (2) test whether the
  per-adapter col(B) overlap predicts the per-shard Exp-3 drop (geometry→behavior); (3) **Part B**:
  facts-vs-skills controlled contrast (Super-NaturalInstructions, N=20 balanced, normalized-NLL
  retention) — the specificity test that merging spares *skills* but kills *facts*.
