# ClAMU — clustering + optimized masking for exact unlearning at scale

**Status:** active · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 4 (2026-06-29 → 2026-07-06)

A build of **ClAMU** (Kuo et al., *"Exact Unlearning of Finetuning Data via Model Merging at
Scale"*, ICLR-2025 submission; [`papers/ClAMU.pdf`](../../papers/ClAMU.pdf)) on TOFU — the
**sibling of the [`sift_masks`](../sift_masks/README.md) arm**. Same exact-unlearning spine:
deterministic full-FT per-author task vectors `τ_t`, streaming sum `τ̄ = Σ τ_t`, exact unlearn
by re-deriving `τ_u` and subtracting. ClAMU changes two things: (1) training is **not**
sign-constrained (a one-kwarg reuse of `sift_masks.sift_one_task(use_sign_constraint=False)`),
and (2) masks are **per-cluster** and **directly optimized** by a score-vector + straight-through
estimator (`clamu.optimize_mask_ste`), not sign/heuristic-derived. Authors are feature-clustered
(MiniLM answer embeddings → k-means, K groups) **before** finetuning, so clustering is frozen and
cascade-free. Served `θ0 + (m_c⊙τ̄)/T` per query via `clamu_model.ClamuModel`, scored on the
repo's OU `model_utility`/`forget_quality`.

**Arc so far:** implementation + CPU exactness gate (06-29) → TinyLlama micro-smoke green →
1B headline (07-02): the ladder reproduces under the strict metric and **optimized cluster-masking
is the first merging-side method here to overtake routing** — clamu_full mu **0.647** beats the
reproduced route+scaffold **0.556** and full-FT **0.530** on the same eval path (the old
0.664/0.599 references were refuted by the [routing_scaffold repro](../routing_scaffold/2026-07-02_scaffold-repro.md)),
sitting between routing and SIFT's per-task masks (**0.737**) — with exact, utility-*raising* deletion.

## Hypotheses — open / resolved
- **[resolved ✓ supported]** H1: the paper's localization ladder (Global < EMR < TALL < ClAMU)
  holds under OU `model_utility` — 0.351 < 0.388 < 0.405 ≪ **0.647**
  ([2026-07-02](2026-07-02_1b-headline.md)).
- **[resolved ✓ supported]** H2: optimized cluster-masking survives the strict 9-component
  metric instead of cratering in the ~0.35–0.45 merge band — clamu_full **0.647**, above the
  reproduced route+scaffold (0.556) and full-FT (0.530) on the same eval path
  ([2026-07-02](2026-07-02_1b-headline.md)).
- **[resolved ✓ supported]** H3: exact forget10 deletion is utility-free — clamu_unlearn mu
  **0.661 ≥ 0.647**, retain_rouge 0.527→0.568, forget signals land exactly at the base-served
  level (f_rouge 0.339, f_ppl 20.33 ≡ merge_unlearn) ([2026-07-02](2026-07-02_1b-headline.md)).
- **[resolved ✓ supported]** H4: the extended-cap eval confirms the smoke ladder —
  0.337 < 0.368 < 0.429 ≪ **0.609**, clamu_unlearn **0.620** ≥ full; dial at extended:
  Global 0.337 → K=16 0.609 → SIFT T=200 0.736 ([2026-07-03](2026-07-03_extended-confirmation.md)).
- **[resolved ~ partial]** H8-align (shared with [sift_masks](../sift_masks/README.md) H8):
  Fig-8 forgotten-serving (`forgotten_serve:"merged"`) raises extended fq 8× (0.0045→0.0352)
  with mu bit-unchanged (0.6204) — but not to the oracle range: the retain merge τ̄/T is
  diluted, so its truth-ratio style still differs from the oracle LoRA. Same verdict as sift
  (0.0045→0.0505): **extended fq measures oracle-style match, not leakage**
  ([2026-07-06](2026-07-06_k-dial-fig8.md)).
- **[resolved ✓/✗ mixed]** H5 (K dial): the dial **saturates early** — knee at K≈4–16
  (K=16 0.662 ≈ 93% of the K=1→200 range), K=200 adds +0.010 for 12.7× storage.
  Sub-prediction "K=1 collapses to Global" **REFUTED**: one optimized mask hits **0.552 @
  175 MB** (+0.20 over Global) — *optimization, not clustering, is the recovery mechanism*.
  Sub-prediction "some K matches SIFT" **REFUTED**: even K=200 optimized (0.672) < SIFT
  sign-trained per-task (0.737) — training regime beats granularity
  ([2026-07-06](2026-07-06_k-dial-fig8.md)).
- **[resolved — moot at top end]** H6: routing with STRONG experts reached **0.7509**
  ([routing_scaffold](../routing_scaffold/README.md), 07-06), above ClAMU's dial ceiling
  (~0.67) — the 07-02 beat-routing claim is scoped to matched-weak-experts. The surviving
  question (why optimization recovers what averaging destroys) folds into H9 + merge_mechanism.
- **[open]** H7: random vs feature clustering at fixed K (paper: +12–30% for feature; the
  route+scaffold result found clustering barely matters — direct tension) — pending E2.
- **[open]** H9: sign-constrained FT + optimized cluster masks (SIFT-build inside ClAMU) —
  does it close the 0.672→0.737 gap at K=16-level storage? One `use_sign_constraint=True`
  build pass + the existing localize.
- **[open]** H10: is K=1-optimized (0.552 @ 175 MB) the repo's best *storage-constrained*
  point? Compare per-GB vs JD-compressed merges and the scaffold.

## What worked
- **The K-dial (07-06):** full/unlearn mu — K=1 **0.552**/0.558 (175 MB), K=4 0.610/0.604,
  K=16 **0.662**/0.665 (2.9 GB), K=50 0.668/0.651, K=100 0.657/0.662, K=200 0.672/0.672
  (36 GB). Early saturation (knee K≈4–16); deletion utility-preserving at every K; the
  equal-epochs recipe (mask_epochs=4) slightly beats the fixed-50-step headline at K=16
  (0.662 vs 0.647). The K=1 point is the standout: **one optimized mask ≈ weak-experts
  routing at 175 MB.**
- **The 1B headline (07-02), confirmed at extended caps (07-03):** ladder
  0.351/0.388/0.405/**0.647** smoke → 0.337/0.368/0.429/**0.609** extended
  (Global/EMR/TALL/ClAMU); masking acts exactly where it should (retain_prob 0.111→0.459,
  retain_ppl 58.5→16.3); **clamu_unlearn 0.661 smoke / 0.620 extended**, both ≥ full —
  deletion raises utility. 16 packed masks ≈ 1.9 GB (~54% dense) vs SIFT's 200 ≈ 24 GB;
  localize wall ~13 min.
- **Maximal reuse of the SIFT arm**: the only change to `sift_masks.py` is a backward-compatible
  `use_sign_constraint` kwarg (SIFT + `test_sift_masks.py` still 6/6 green). Merge/serve/pack,
  the deterministic FT primitive, data loaders, oracle routing, and k-means all reused.
- **CPU exactness gate green** (`test_clamu.py`, 6/6): STE forward/backward correct; cluster
  determinism; exact unlearning (no-sign) max gap **9.31e-10**; STE optimization reduces CE
  (4.21 → 4.07) deterministically; serve identity + mask pack/unpack bit-exact.
- **TinyLlama micro-smoke (go/no-go) green**: clamu_full **0.4530** > merge_full **0.4003**;
  exact-unlearn pipeline end-to-end clean (clamu_unlearn 0.4515).
- **1B OOM fixed**: `mask_batch_rows=8` (cap the STE forward's LM-head logits) +
  `expandable_segments` + a build resume-skip; all 16 clusters localize on a 44 GiB card.
  NB gradient checkpointing is *incompatible* with `functional_call` — cap the batch instead.

## What didn't / open problems
- **First 1B localize attempt OOM'd** (job 439831: full-param fp32 STE state + 20-row×128k-vocab
  logits > 44 GiB) — fixed as above; documented in [2026-07-02](2026-07-02_1b-headline.md).
- **fq is a serving-protocol artifact in both tiers** — smoke: 0.3929 ≡ base for every
  near-base forget serve (EMR's 0.808 is collateral damage, not forgetting); extended: 0.0045
  for ALL base-served conditions across clamu *and* sift (KS at 120 rows resolves base vs the
  retain90-oracle LoRA's style shift). Read mu/retain components; the fix is H8-align above.
- **EMR/TALL baselines are full-model only** — per-cluster sums `τ_c` are not re-accumulated for
  the retain re-clustering (clamu_unlearn needs no `τ_c`); EMR/TALL_unlearn would need a pass.
- **Mask-opt memory remains the heaviest stage** — full-parameter fp32 scores + Adam moments;
  per-cluster, parallelizable via `localize --cluster J` if wall time matters at bigger models.

## Open ideas / next steps
- Report: [`CLAMU_REPORT_2026-07-02.md`](../../tofu_sisa_lora/reports/CLAMU_REPORT_2026-07-02.md)
  (ladder + same-path references + deletion cost ledger + extended addendum).
- Gated expansion, in order: **E4 K-dial** (K∈{1,4,50,100,200} — doubles as H5),
  **H8-align serving option** (`forgotten_serve: merged`, jointly with sift), **E2
  random-vs-feature** (H7), clamu+scaffold hybrid / stronger-experts routing rerun (H6),
  heterogeneity probe reuse (`subspace_overlap.py`).

## Entries (chronological)
- [2026-06-29 — implementation + CPU exactness](2026-06-29_implementation.md) — full ClAMU arm written by extending the SIFT spine; 6/6 CPU gate green (exact-unlearn gap 9.3e-10, STE reduces CE); SIFT unaffected; STUB/compile clean; GPU run pending.
- [2026-07-02 — 1B headline](2026-07-02_1b-headline.md) — ladder reproduces under OU mu (0.351/0.388/0.405/**0.647**); clamu_full beats reproduced route+scaffold (0.556) & full-FT (0.530); exact forget10 deletion raises utility (0.661); OOM fixed via mask_batch_rows.
- [2026-07-03 — extended confirmation](2026-07-03_extended-confirmation.md) — H4 ✓: extended ladder 0.337/0.368/0.429/**0.609**, unlearn 0.620 ≥ full; extended fq 0.0045 exposed as a cross-track base-vs-oracle KS artifact (≡ sift) → new open H8-align.
- [2026-07-06 — K-dial + Fig-8](2026-07-06_k-dial-fig8.md) — H5 resolved (early saturation; **K=1 optimized 0.552 refutes collapse** — optimization > clustering; K=200 0.672 < SIFT 0.737 — training regime > granularity); H8-align resolved-partial (fq 0.0045→0.0352, mu unchanged; ≡ sift verdict); deletion utility-free at every K.
