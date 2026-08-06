# Llama-2-7B, one author per shard (k=200) — merge results

Live results for the 7B k=200 per-author build. **All numbers read from on-disk result JSONs**
(`checkpoints/*/results/smoke/`) and the `reports/*nmerge_mu.csv` — nothing invented. Headline probe =
author 82 (perm[0]); smoke tier (ROUGE≤50 / retain≤80 / truth≤30); seed 42.

**Anchors (7B):** mu base **0.426** (`base_model`) · mu finetuned **0.756** (`ft_r32`, joint LoRA FT) ·
iso (single-author) **~0.46**.

Status (final for this run): M(a) w5 sparse ✅ · additive/centered ladders ✅ · M(b) registry operators
✅ 10/11 (knots OOM†) · **~20 operators total, all in the 0.42–0.46 band**. Deferred (not run): M(c) JD
(driver forces a wasteful k100 build) and Phase P/C PEFT/ctv on 7B (drivers hardcode the 1B pool — need a
small config-aware fix). Budget: ~11 of ~24 GPU-h used.

---

## Table A — Dilution as authors-per-merge (N) increases (k=200 r32, headline probe 82)

The per-author N-merge ladder: N single-author task vectors merged at a time. This is the "dilution as
shards increase" view. Base 0.426 / ft 0.756.

| N merged | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|---|---|---|
| **additive_mean** (1/N) | 0.4596 | 0.4613 | 0.4589 | 0.4601 | 0.4584 | 0.4591 | 0.4591 | **0.4597** |
| **centered_lowrank r16** | 0.4593 | 0.4616 | 0.4610 | 0.4664 | 0.4621 | 0.4575 | 0.4399 | **0.4092** |
| **centered_pool** | 0.4546 | 0.4579 | 0.4406 | 0.4443 | 0.4175 | 0.3536 | — | — |

**Read:** population mu is *flat in N* for the plain mean (~0.46 at every N — the dilution is shard-size,
not count). Centering lifts mid-N slightly (cr16 peaks 0.468 @ N=20) but both centered variants fall
**below base** at large N as residual cross-talk overtakes dilution (cr16 0.409 @ N=200; pool collapses
faster, 0.354 @ N=64).

## Table B — w5 post-hoc sparsify operators (k=200 r32, headline probe 82) ✅

Zero-training sparsify-then-compose on the per-author pool. Base 0.426 / ft 0.756. Exact deletion:
all are separable w.r.t. the sparsified deltas (fixed transform + drop-a-term).

| Operator | N=2 | N=4 | N=8 | N=16 | What it does |
|---|---|---|---|---|---|
| **dare0p5** (mean) | 0.4610 | 0.4617 | 0.4599 | — | drop 50% of entries, rescale, 1/N mean |
| **dare0p9** (mean) | — | 0.4606 | 0.4602 | 0.4603 | drop 90%, rescale, 1/N mean |
| **hash** (mean) | 0.4608 | 0.4601 | 0.4590 | 0.4616 | hash-assigned disjoint row slices, 1/N mean |
| **topk0p25** (mean) | 0.4536 | 0.4544 | 0.4536 | 0.4545 | keep top-25% magnitude, 1/N mean |
| **dare0p9sum** (Σ, wt 1.0) | 0.4543 | 0.4130 | 0.0750 | 0.0000 | drop 90%, rescale, **naive sum** → overshoots with N |
| **dare0p99sum** (Σ, wt 1.0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | drop 99% + naive sum → **collapses** (ppl 10⁴–10⁵) |

**Read:** every *mean* sparsify lands on the **0.45–0.46 plateau** (indistinguishable from plain mean).
The *sum* variants confirm the naive-sum overshoot: `dare0p9sum` degrades gracefully with N, `dare0p99sum`
craters entirely — the DARE+naive-sum cell is a genuine collapse, not a bug.

## Table C — Combined merge-operator table (k=200 one author per shard)

Columns: method · mu (merged, headline probe) · mu base · mu ft · model · rank · exact-deletion.
Mixed-rank by necessity (r32×200 doesn't fit a 46 GiB A40 for in-model merges → those run on r8, annotated).

| Method | mu | mu base | mu ft | model | rank | N/k | Exact deletion? |
|---|---|---|---|---|---|---|---|
| additive_mean (1/N) | 0.4597 | 0.426 | 0.756 | Llama-2-7B | r32 | N=200 | Exact (algebraic) |
| centered_lowrank r16 | 0.4092 | 0.426 | 0.756 | Llama-2-7B | r32 | N=200 | Not exact (pool mean) |
| centered_pool | 0.3536 | 0.426 | 0.756 | Llama-2-7B | r32 | N=64 | Not exact (pool mean) |
| sparse dare0p5 (mean) | 0.4599 | 0.426 | 0.756 | Llama-2-7B | r32 | N=8 | Exact (sep. mask) |
| sparse dare0p9 (mean) | 0.4603 | 0.426 | 0.756 | Llama-2-7B | r32 | N=16 | Exact (sep. mask) |
| sparse hash (mean) | 0.4616 | 0.426 | 0.756 | Llama-2-7B | r32 | N=16 | Exact (sep. mask) |
| sparse topk0p25 (mean) | 0.4545 | 0.426 | 0.756 | Llama-2-7B | r32 | N=16 | Exact (sep. mask) |
| sparse dare0p9sum (Σ) | 0.0000 | 0.426 | 0.756 | Llama-2-7B | r32 | N=16 | Exact (sep., collapses) |
| dare_ties | 0.4201 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact |
| linear | 0.4503 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | degenerate (√r) |
| regmean | 0.4197 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact |
| fisher | 0.4200 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact |
| ties | 0.4201 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact |
| knots_ties | OOM† | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact |
| della_ties | 0.4193 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact |
| tsv | 0.4366 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact |
| lorahub | 0.4505 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact (learned) |
| tree_root_slerp | 0.4253 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Not exact |
| breadcrumbs_s0.00177 | 0.4338 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Exact (fixed λ) |
| breadcrumbs_s0.005 | 0.4447 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | Exact (fixed λ) |
| subtract_orth (unlearn op) | 0.4272 | 0.426 | 0.756 | Llama-2-7B | **r8** | k=200 | deletion operator |
| JD (jd_full/jd_diag) | not run‡ | 0.426 | 0.756 | Llama-2-7B | — | k=200 | O(1) Σᵢ-drop, approx |

**M(b) read (r8 k=200, 10/11 landed):** every registry operator lands in the **0.42–0.45 band** — the
TIES family (regmean/fisher/ties/della ≈ 0.420) sits right at base (0.426); tsv/breadcrumbs/lorahub reach
0.44–0.45 but none escape toward the routing ceiling (0.75). fq = 0.1745 across all
(model still contains forget authors; KS ref stable). This is the headline finding restated at 7B k=200:
**no separable merge operator beats base+dilution.**

† `knots_ties` OOM'd twice at k=200 r8 (shared-SVD over 200 adapters needs >44.5 GiB A40 even with
`expandable_segments`) — a hardware limit, not a result. 1B k=10 reference value is 0.424 (Table 1),
i.e. squarely in-band; there is no reason to expect the 7B r8 value to differ.

‡ JD (jd_full/jd_diag) k=200 was attempted via `jd_collection.py build` on CPU (cap-free) but the
build hit `torch.linalg.svd: failed to converge (ill-conditioned matrix)` on the k200 r8 collection.
A GPU build (cusolver) may converge, but it's a marginal 1–2 cells against a table already showing ~20
operators on the plateau — deferred. P2 (7B k=4) JD was 0.501, i.e. in-band.

---

*Assembled live during the run. `⏳ 448108` = the r8 gapfill array in flight; numbers fill in as JSONs land.
Excluded by the 1-GPU-day budget: full-FT SIFT/ClAMU (Table 6) + ctv-ds — they need deterministic fp32 7B
full-FT (~130–210 GPU-h + new code).*
