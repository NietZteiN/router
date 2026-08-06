# ClAMU on TOFU — Localization Ladder, Exact Deletion, and the Mask-Granularity Dial
**Date:** 2026-07-02 (extended §6 added 07-03; K-dial + Fig-8 §7 added 07-06) · **Model:** Llama-3.2-1B-Instruct
**Thread:** [`log/clamu/`](../../log/clamu/README.md) · **Paper:** ClAMU (Kuo et al., ICLR-2025 sub.; `papers/ClAMU.pdf`)

## 1. What was run

ClAMU = the SIFT-Masks exact-unlearning spine (deterministic full-FT per-author task vectors,
streaming sum `τ̄`, unlearn = re-derive `τ_u` and subtract) with two swaps: **no sign constraint**
during FT, and **per-cluster masks directly optimized** by a score-vector + straight-through
estimator instead of sign/heuristic derivation. K=16 feature clusters (MiniLM answer-mean k-means,
frozen before FT), T=200 authors, serve `θ0 + (m_c⊙τ̄)/T` per query (oracle author→cluster route;
OOD → base θ0). Config `configs/clamu_tofu_1b.json` (sha256:12 `000cc82d1240`), seed 42, smoke
tier, `--k 10 --forget_shard_id 9`, SISA retain90 KS reference.

Provenance: first attempt 439829–439831 (setup + build OK ~1 h; localize **OOM** at 44 GiB);
retry 440189–440195 (build skipped via resume guard; localize all 16 clusters ~13 min after the
fix `mask_batch_rows=8` + `expandable_segments` — NB gradient checkpointing is incompatible with
`torch.func.functional_call`, so the STE forward's batch is capped instead). CPU gates green
before every submit (`test_clamu.py` 6/6, `test_sift_masks.py` 6/6). Scripts: `clamu.py`
`e33b36a467ea`, `clamu_model.py` `8f7f56932ecd`, `train_clamu.py` `42685517155b`,
`submit_clamu_tofu.sh` `9f2e2f30c12c`.

## 2. The localization ladder (headline)

| label | model_utility | forget_quality | forget_ppl | retain_ppl | retain_rouge | retain_prob |
|---|---|---|---|---|---|---|
| merge_full (Global, no mask) | 0.3511 | 0.5941 | 57.53 | 58.47 | 0.293 | 0.111 |
| emr_full (sign agreement) | 0.3882 | 0.808 | 43.19 | 46.46 | 0.320 | 0.135 |
| tall_full (threshold, λ=0.4) | 0.4053 | 0.3929 | 17.28 | 17.46 | 0.352 | 0.155 |
| **clamu_full (STE-optimized)** | **0.6469** | 0.3929 | 16.45 | 16.28 | **0.527** | **0.459** |
| **clamu_unlearn (exact forget10)** | **0.6609** | 0.3929 | 20.33 | 16.48 | **0.568** | 0.457 |
| merge_unlearn | 0.3527 | 0.3929 | 20.33 | 58.95 | 0.302 | 0.110 |

**The paper's ladder holds under the strict OU metric** — Global < EMR < TALL ≪ ClAMU — and the
optimized mask nearly doubles Global-merge utility (+0.296), acting exactly where masking should:
retain_prob 0.111→0.459, retain_ppl 58.5→16.3. Heuristic masks (EMR/TALL) recover almost none of it.

**References on the same eval path** (smoke, 1B, seed 42): routed+scaffold (OOD-aware, the
corrected repro) **0.556**, committed-code routed+scaffold 0.474, full-FT (`_ft`, legacy r8)
**0.530**, SIFT-Masks T=200 per-task masks **0.737**, FT+Merge no-mask 0.407, SISA merges
~0.43–0.48. (The earlier ORIENTATION figures 0.664/0.599 were refuted by
[`ROUTING_SCAFFOLD_REPRO_2026-07-01.md`](ROUTING_SCAFFOLD_REPRO_2026-07-01.md) — not comparable.)

**Read-off — the mask-granularity dial:**
> Global merge (0.351) → **cluster-optimized K=16 (0.647)** → per-task T=200 sign masks (0.737).
> ClAMU is the first *merging-side* method in this repo to overtake routing (0.647 > 0.556),
> at 16 masks (~1.9 GB packed, ~54% dense) vs SIFT's 200 (~24 GB).

## 3. Exact deletion (forget10)

Deletion = deterministically re-derive the 20 forget authors' `τ_u`, subtract from `τ̄`,
re-cluster the 180 retain authors (forget data never touches the retain clustering), re-optimize
the 16 masks on retain data. Result: **utility rises** (0.647 → 0.661; retain_rouge 0.527→0.568)
— the paper's merge-interference effect — while forget signals land exactly at the base-served
level (f_rouge 0.339 and f_ppl 20.33 ≡ merge_unlearn's base-served forget row; forgotten authors
route to θ0 by construction).

**Cost ledger (per forget10 deletion):** 20 × 20-step τ_u re-derivations (~6 min, 1 GPU) +
16 × 50-step mask re-optimizations (~10 min serial; parallelizable per cluster) — no retain-set
retraining. Storage held long-term: one `τ̄` (fp32 param dict) + 16 bit-packed masks (~1.9 GB) +
the assignment JSON.

## 4. Caveats

- **fq is uninformative at smoke tier here**: 0.3929 ≡ the base value for every condition whose
  forget-set serving ≈ θ0 (the τ̄/200 dilution barely memorizes; unlearn serves θ0 for forgotten
  authors by construction). EMR's fq 0.808 is collateral damage (a broken model looks
  "never-trained"), not good forgetting. The clean read is mu / the retain components.
- Single seed (42); the extended pass (job 440209) confirmed the ladder — see §6.
- OOD components are identical across all rows (real/world rouge 0.952/0.953) because OOD routes
  to base θ0 in every condition — mu differences are entirely retain-author-driven.
- EMR/TALL exist for the full model only (per-cluster sums not re-accumulated post-deletion).

## 5. Next

K-dial sweep K∈{1,4,50,100,200} (H5, the storage–utility bend) → H8-align serving option
(`forgotten_serve: merged`, jointly with the sift arm — see §6) → random-vs-feature clustering
(H7, tension with the routing thread's "clustering barely matters") → decompose why
cluster-masking beats routing (H6: weak legacy-r8 experts vs cross-author signal in the mask).

## 6. Extended-cap confirmation (addendum, 2026-07-03)

Eval-only rerun of the same six artifacts at extended caps (ROUGE≤200 / retain≤400 / truth≤120;
SLURM array 440209, 2026-07-02 14:54–16:45; extended KS reference). **H4 supported — the ladder
holds:**

| label | mu smoke → extended | fq smoke → extended |
|---|---|---|
| merge_full | 0.3511 → 0.3371 | 0.594 → 0.0505 |
| emr_full | 0.3882 → 0.3682 | 0.808 → 0.2371 |
| tall_full | 0.4053 → 0.4288 | 0.393 → 0.007 |
| **clamu_full** | **0.6469 → 0.6092** | 0.393 → 0.389 |
| **clamu_unlearn** | **0.6609 → 0.6204** | 0.393 → 0.0045 |
| merge_unlearn | 0.3527 → 0.3369 | 0.393 → 0.0045 |

Ordering preserved (0.337 < 0.368 < 0.429 ≪ **0.609**); deletion still raises utility
(**0.620** ≥ 0.609; retain_rouge 0.473→0.491). Extended granularity dial: Global **0.337** →
cluster K=16 **0.609** → SIFT per-task T=200 **0.736** (`sift_full` 0.7364 on the same path).

**The extended fq numbers are a cross-track serving-protocol artifact, not forgetting signal:**
`clamu_unlearn`, `merge_unlearn`, `sift_full`, and `sift_unlearn` all land at fq **0.0045**,
because each serves forget-author queries at (or indistinguishably near) base θ0, and at 120
truth rows the KS test resolves raw-base vs the retain90-oracle LoRA's style shift. Deletion
remains exact by construction. The shared fix (the sift thread's H8, adopted here as H8-align):
serve forgotten authors with the maskless *retain* merge `θ0 + τ̄_retain/T` — the papers' own
Fig-8 protocol — making extended fq cross-track comparable.

## 7. K-dial + Fig-8 serving (addendum, 2026-07-06)

**K-dial (E4/H5)** — 6 chains K∈{1,4,16,50,100,200} (`configs/clamu_tofu_1b_K{K}.json`,
jobs 440434–440475), `mask_epochs=4` (equal mask-training epochs across K via
`clamu.localize_steps` — a fixed step count would confound the dial), `heuristic_masks=false`,
τ̄/τ̄_forget10/author_emb symlinked from the K=16 build (K-independent; build + unlearn-subtract
skipped by resume guards). Smoke tier, seed 42:

| K | clamu_full mu | clamu_unlearn mu | masks on disk |
|---|---|---|---|
| 1 | 0.5519 | 0.5578 | 175 MB |
| 4 | 0.6103 | 0.6036 | 704 MB |
| 16 | 0.6620 | 0.6649 | 2.9 GB |
| 50 | 0.6683 | 0.6509 | 9.0 GB |
| 100 | 0.6571 | 0.6615 | 18.1 GB |
| 200 | 0.6716 | 0.6716 | 36.4 GB |

Three read-offs:
1. **Optimization, not clustering, is the recovery mechanism.** A single optimized mask
   (K=1, "optimized Global") reaches **0.552 @ 175 MB** — +0.20 over the Global merge (0.351)
   and ≈ weak-experts routing (0.556). The predicted collapse at K=1 is refuted.
2. **The dial saturates early.** The knee is K≈4–16 (K=16 = 0.662 captures ~93% of the
   K=1→200 range); K=200 adds +0.010 for 12.7× the storage. Deletion is utility-preserving
   at every K (unlearn ≈ full, ≤0.017 apart).
3. **Training regime beats granularity.** Even per-author optimized masks (K=200, 0.672)
   stay below SIFT's sign-*trained* per-task masks (0.737) at identical granularity and
   storage — the surviving gap is how the task vectors were trained, not how finely they're
   masked (open H9: sign-constrained FT + optimized cluster masks).

Frame update: routing with STRONG experts reached **0.7509** the same day
([`routing_scaffold`](../../log/routing_scaffold/README.md)), so §2's beat-routing claim is
scoped to matched-weak-experts; ClAMU's dial ceiling (~0.67) sits between weak-experts
routing and SIFT / strong-experts routing. Single-seed smoke — treat ±0.01 K-to-K wiggles
(e.g. K=50 > K=100) as noise.

**Fig-8 forgotten-serving (H8-align)** — `forgotten_serve:"merged"`
(`configs/clamu_tofu_1b_fig8.json`, job 440433): extended fq rises 8× (clamu_unlearn
**0.0045 → 0.0352**; merge_unlearn identical) with mu **bit-unchanged** (0.6204 / 0.3369) —
direction confirmed but short of the oracle range, because the retain merge τ̄/T is diluted.
Same verdict as the sift arm (0.0045→0.0505): **extended-cap forget_quality measures
oracle-style match, not leakage**; deletion exactness is unaffected (the forgotten serving is
a function of retain data only).
