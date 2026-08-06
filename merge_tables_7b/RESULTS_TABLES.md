# Model-Merging Results — consolidated tables

All numbers read from result JSONs under `~/tofu_sisa_lora/checkpoints/*/results/{smoke,extended}/`
and the aggregation reports (`~/tofu_sisa_lora/reports/`). `mu` = `model_utility` (harmonic mean of 9
TOFU components; higher = better). Nothing here is invented; sources cited per section.

Primary source: `~/tofu_sisa_lora/reports/MERGE_METHODS_RESULTS_2026-07-21.md`
· PEFT: `reports/PEFT_BAKEOFF_2026-07.md` · centered CSV: `reports/centered/nmerge_mu.csv`.

> **Thesis-first companion:** `~/tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md`
> re-arranges these same P1–P5 numbers into a banded **grand master merge table** ("merging
> doesn't work") and places the **router + merger** results side by side (per-pool
> ceiling-vs-headroom, "selection is the carrier"). Cells there were spot-verified against the
> live result JSONs on 2026-07-24.

> **Status of the 7B one-author-per-shard (k=200) build:** in progress — see `STATUS.md`.
> The tables below are the current state (mostly Llama-3.2-1B pools + the additive/centered 7B
> k=200 ladders that already exist). The plan (`PLAN.md`) fills the 7B k=200 gaps.

---

## Anchor legend (base / finetuned reference points per pool)

| Pool (model · granularity · tier) | mu base (never-FT) | mu finetuned (joint FT ceiling) |
|---|---|---|
| **P1** — Llama-3.2-1B · k=10 LoRA shards · smoke | 0.398 | 0.530 (`ft_all`) |
| **P2** — Llama-2-7B · k=4 LoRA shards · smoke | 0.426 | 0.756 (`ft_r32`) |
| **P3** — Llama-2-7B · k=200 per-author r32 LoRA · smoke | 0.426 | 0.756 (`ft_r32`) |
| **P4** — Llama-3.2-1B · k=10 · PEFT bake-off · smoke | 0.380 | 0.530 (`ft_all`) |
| **P5** — Llama-3.2-1B · T=200 per-author full-FT · smoke | 0.398 | 0.737 (k=1 full-FT ceiling) |

---

## Table 1 — LoRA task-vector merge operators (P1: Llama-3.2-1B, k=10)

| Method | mu (merged) | mu base | mu FT | fq | f_rouge | r_ppl | Exact deletion? | Model |
|---|---|---|---|---|---|---|---|---|
| Naive sum, λ=1 (`additive_s1`) | 0.000 | 0.398 | 0.530 | 0.007 | 0.001 | 4·10⁵ | Exact (algebraic) | Llama-3.2-1B |
| Uniform mean, λ=1/k (`additive_mean`) | 0.419 | 0.398 | 0.530 | 0.999 | 0.461 | 7.3 | Exact (algebraic) | Llama-3.2-1B |
| Tuned-λ sum, λ=0.05 (`additive_s0.05`) | 0.429 | 0.398 | 0.530 | 0.808 | 0.461 | 8.2 | Exact (fixed λ) | Llama-3.2-1B |
| DARE-TIES (frozen default) | 0.424 | 0.398 | 0.530 | 0.393 | 0.437 | 13.2 | Not exact | Llama-3.2-1B |
| DELLA-TIES | 0.429 | 0.398 | 0.530 | 0.393 | 0.432 | 10.1 | Not exact | Llama-3.2-1B |
| Fisher-weighted | 0.424 | 0.398 | 0.530 | 0.393 | 0.432 | 13.5 | Not exact | Llama-3.2-1B |
| KnOTS (shared-SVD + TIES) | 0.424 | 0.398 | 0.530 | 0.393 | 0.443 | 13.1 | Not exact | Llama-3.2-1B |
| Breadcrumbs, λ=1/(n√r) | 0.419 | 0.398 | 0.530 | 0.958 | 0.454 | 7.6 | Exact (fixed λ, sep. mask) | Llama-3.2-1B |
| Breadcrumbs, λ=1/n | 0.000 | 0.398 | 0.530 | 0.958 | 0.396 | 11.7 | Exact (fixed λ) | Llama-3.2-1B |
| PEFT linear (√r-inflated) | 0.050 | 0.398 | 0.530 | 0.594 | 0.376 | 18.4 | degenerate | Llama-3.2-1B |
| TSV-M (whitened top-singular) | 0.051 | 0.398 | 0.530 | 0.594 | 0.429 | 12.3 | Not exact | Llama-3.2-1B |
| SLERP (tree, pairwise) | 0.090 | 0.398 | 0.530 | 0.594 | 0.395 | 11.8 | Not exact | Llama-3.2-1B |
| Subtract-orth *(unlearn op)* | 0.433 | 0.398 | 0.530 | 0.594 | 0.464 | 9.9 | deletion operator | Llama-3.2-1B |
| Task-arith subtraction (`subtract_linear`) *(unlearn op)* | 0.000 | 0.398 | 0.530 | 0.007 | 0.000 | 3·10⁵ | deletion operator | Llama-3.2-1B |
| *Routing key-exact (reference)* | 0.458 | 0.398 | 0.530 | — | 0.477 | 3.6 | Exact module-drop | Llama-3.2-1B |
| *Routing + scaffold, OOD-aware (reference)* | 0.556 | 0.398 | 0.530 | — | 0.532 | 4.1 | Exact module-drop | Llama-3.2-1B |

Read: every *separable* merge operator lands in the 0.42–0.43 band (≈ base + dilution); the ones that
leave it do so by breaking. Only serve-time selection (routing) escapes upward.

---

## Table 2 — LoRA merges, few-task regime (P2: Llama-2-7B, k=4)

| Method | mu (merged) | mu base | mu FT | fq | f_rouge | Exact deletion? | Model |
|---|---|---|---|---|---|---|---|
| LoraHub (learned weights) | 0.592 | 0.426 | 0.756 | 0.808 | 0.512 | Not exact (learned weights) | Llama-2-7B |
| DARE-TIES | 0.545 | 0.426 | 0.756 | 0.808 | 0.535 | Not exact | Llama-2-7B |
| JD-full (Compress-then-Serve) | 0.501 | 0.426 | 0.756 | 0.958 | 0.508 | O(1) Σᵢ-drop, approx (shared basis) | Llama-2-7B |
| JD-diag | 0.501 | 0.426 | 0.756 | 0.958 | 0.512 | O(1) Σᵢ-drop, approx (shared basis) | Llama-2-7B |
| DELLA-TIES | 0.497 | 0.426 | 0.756 | 0.071 | 0.522 | Not exact | Llama-2-7B |
| KnOTS | 0.489 | 0.426 | 0.756 | 0.594 | 0.522 | Not exact | Llama-2-7B |
| Fisher | 0.477 | 0.426 | 0.756 | 0.958 | 0.519 | Not exact | Llama-2-7B |
| PEFT linear / DELLA-linear / Breadcrumbs (unscaled) | 0.000 / 0.000 / 0.0001 | 0.426 | 0.756 | — | ≈0 | degenerate (√r bug) | Llama-2-7B |

---

## Table 3 — Dilution law: DARE-TIES vs shard count (Llama-2-7B, smoke)

| k (shards) | 4 | 10 | 20 | 50 | 100 | 200 (r8) |
|---|---|---|---|---|---|---|
| `merged_dare_ties` mu | 0.545 | 0.477 | 0.450 | 0.438 | 0.430 | 0.420 |
| `routed_key_exact` mu | — | — | — | 0.715 | 0.648 | 0.473* |

Base 0.426 / FT 0.756. *k=200 routing is r8-capacity-limited (scaffold pool reaches 0.751).

---

## Table 4 — Per-author N-merge ladder (P3: Llama-2-7B, k=200 r32, true-mean)

| N merged | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|---|---|---|
| `additive_mean` mu | 0.460 | 0.461 | 0.459 | 0.460 | 0.458 | 0.460 | 0.459 | 0.460 |
| `centered cr16` mu | 0.459 | 0.462 | 0.461 | 0.466 | 0.462 | 0.458 | 0.440 | 0.409 |

Base 0.426 / FT 0.756. Population mu is flat in N; centered merging moves the per-author-recall knee
(N≈3 → N*≈64) but ends below base at N=200.

---

## Table 5 — Operator-independence: PEFT parameterization bake-off (P4: Llama-3.2-1B, k=10)

| Method (compose rule) | composed mu | routed mu | iso mu (s9) | comp f_rouge | Exact deletion? | Model |
|---|---|---|---|---|---|---|
| LoRA (additive mean) | 0.419 | — | — | 0.461 | Exact | Llama-3.2-1B |
| DoRA (additive mean) | 0.432 | 0.491 | 0.384 | 0.449 | Exact | Llama-3.2-1B |
| IA³ (gate arith-mean) | 0.430 | 0.516 | 0.396 | 0.428 | Exact (O(1)) | Llama-3.2-1B |
| IA³ (gate geo-mean) | 0.430 | — | 0.396 | 0.430 | Exact (O(1)) | Llama-3.2-1B |
| VeRA (shared frozen basis, mean) | 0.415 | 0.445 | 0.399 | 0.439 | Exact | Llama-3.2-1B |
| Prefix-tuning (KV concat) | 0.002 | 0.000 | 0.074 | 0.153 | Exact (byte segment-drop) | Llama-3.2-1B |

Base 0.380 / FT 0.530. Every weight-space composer plateaus at base+0.04, independent of parameterization.

---

## Table 6 — Full-parameter task vectors (P5: Llama-3.2-1B, T=200 per-author full-FT)

| Method / condition | mu (merged) | mu base | mu FT | fq | f_rouge | Exact deletion? | Model |
|---|---|---|---|---|---|---|---|
| **SIFT-Masks** `sift_full` (sum + inference-time mask) | 0.737 | 0.398 | 0.737 | 0.005* | 0.834 | Exact (bitwise); mask = router | Llama-3.2-1B |
| **FT+Merge** `merge_full` (same sum, no mask) | 0.407 | 0.398 | 0.737 | 0.099 | 0.383 | Exact (algebraic) | Llama-3.2-1B |
| SIFT-Masks `sift_unlearn` (subtract 20 τ) | 0.738 | 0.398 | 0.737 | 0.0505 | 0.339 | Exact (GPU bitwise) | Llama-3.2-1B |
| **ClAMU** — Global (no mask) | 0.351 | 0.398 | 0.737 | — | — | Exact (algebraic) | Llama-3.2-1B |
| ClAMU — EMR mask | 0.388 | 0.398 | 0.737 | — | — | Exact; mask = serve-time select | Llama-3.2-1B |
| ClAMU — TALL mask | 0.405 | 0.398 | 0.737 | — | — | Exact; mask = serve-time select | Llama-3.2-1B |
| ClAMU — optimized mask (K=16) | 0.647 | 0.398 | 0.737 | — | — | Exact; mask = serve-time select | Llama-3.2-1B |
| ClAMU — optimized mask (K=200 peak) | 0.672 | 0.398 | 0.737 | — | — | Exact; mask = serve-time select | Llama-3.2-1B |

*fq of `*_full` rows expected-low (model still contains forget authors). The serve-time mask, not the
merge, carries the utility.

---

## Table 7 — composable_tv training-time constructions (Llama-3.2-1B, solo N=1)

| Arm | solo mu | own-prob | own-rouge | Verdict | Model |
|---|---|---|---|---|---|
| **ctrl** — plain per-author LoRA | 0.514 | 0.997 | 1.000 | anchor ✓ | Llama-3.2-1B |
| **[lin]** tangent-space (linearized) | 0.000 | 0.9999 | 1.000 | memorizes perfectly; linearized serving zeroes utility | Llama-3.2-1B |
| **[wd]** write-disjoint col(B) | 0.460 | 0.191 | 0.404 | KILLED — constraint can't memorize | Llama-3.2-1B |
| **[ds]** disjoint-support full-FT | 0.644 | 0.498 | 0.475 | H-ds-1 refuted as stated; healthy served utility | Llama-3.2-1B |

Base own-prob floor 0.146; ctrl solo mu 0.514.

---

## One-line synthesis
Across 2 model scales, 3 vector types (LoRA / full-FT / sign-fixed full-FT) and ~15 algorithms, every
*separable* merge lands within noise of one ceiling — **mu ≈ 0.41–0.48 ≈ base + dilution**. Everything
above (SIFT-mask 0.737, ClAMU 0.647, routing 0.751) buys its utility with serve-time selection, or breaks
separability (LoraHub 0.592 at k=4).
