# SISA-LoRA — sharded LoRA unlearning on TOFU

**Status:** active — latest entry (2026-06-20) lands the additive coarse-core headline and explicitly leaves forget05/forget01 + extended-cap confirmation deferred · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 11 (2026-06-04 → 2026-06-20)

This thread investigates exact/near-exact machine unlearning for LLMs on the TOFU benchmark by **sharding the training data into k slices, training one LoRA per shard, and unlearning by dropping or recomputing the affected shard** — so deletion is an O(1)-style "drop the module" operation rather than gradient surgery. The arc moves from merge-based aggregation (TIES/DARE/linear and a large extended merge-method registry), through a metric-correctness pivot (porting the eval to be numerically faithful to locuslab open-unlearning), into k-scaling (k up to 200) where **routing-based serving** emerged as the strongest serving mode.

Two sub-threads branch off the merge work: **JD / Joint-Diagonalization compression** (Compress-then-Serve port — compress a collection of adapters into a shared per-cluster basis so selective keep/drop stays O(1)), and **additive true-scale shards** (compose independent adapters by literal weighted sum, unlearn = drop a term). The additive sub-thread produced the current headline: a **coarse retain-core (one jointly-trained adapter, never sharded) + fine forgettable tail** reaches near-joint-ft utility at O(1) deletion cost.

## What worked
- **Task-vector subtraction (`remerge_cat`) gives by far the strongest forgetting at k=4** — Llama-3.2-1B forget_ppl **3674** / forget_rouge **0.067**, three orders of magnitude above any merged variant (06-04, confirmed under extended caps 06-05).
- **`dare_ties` is the reliable utility-preserving merge**; the literature review pinned the cause — rescaling after sparse pruning (DARE) matters more than the pruning strategy, which is why `dare_ties` consistently beats plain `ties` (06-08).
- **The eval port made numbers paper-faithful**: `test_ou_equivalence.py` reproduces open-unlearning's truth-ratio, probability, and `model_utility`=hmean math (max|d| ~5.7e-6), and running our pipeline on OU's own Llama-3.2-1B model gave **model_utility 0.5996 ≈ OU's 0.60** — faithfulness proven 3 ways (06-10).
- **k=1 LoRA ft recipe clears the ≥0.6 utility bar with margin**: winner `lr1e-4 / e5 / r32` reached **mu 0.7435 smoke / 0.7404 extended**, lifting utility from base **0.43 → 0.74**, matching the locuslab full-FT reference (0.748) and beating the OU leaderboard (0.63) (06-10/06-11).
- **Routing is the headline serving mode at high k**: `routed_key_exact` @k=50 reached **mu 0.7147** (within 0.03 of the k=1 monolith, far above the best sharded merge 0.592), with O(1) deletion by construction (`_no49` exclusion leaves retain/real/world untouched) and the first non-trivial unlearning demo at k>4 (06-12).
- **JD selective-keep beats `dare_ties` at every k≥10** on both utility and forget_quality despite high recon error — e.g. k100/c7 `remerge_jd_full` mu **0.465 / fq 0.239** vs dare_ties **0.430 / 0.135** — confirming recon error is a loose proxy for downstream quality (06-15/06-17).
- **Additive coarse-core is the strongest unlearning structure found**: a strong retain90 core (authors 0-179, r32/e5) evaluated standalone (= forget10 unlearned state) hit **model_utility 0.7537**, matching joint-ft (0.740) with **forget_ppl 14.2 ≈ base 15.2** and **forget_quality 0.958** (core never saw forget data) — O(1) deletion at near-joint-ft utility (06-20).

## What didn't / open problems
- **Merging dies under dilution**: merged `dare_ties` utility decays monotonically 0.74 (k=1) → 0.54 (4) → 0.48 (10) → 0.45 (20) → 0.44 (50) → 0.43 (100) → ≈0.42 (200) = base — at k≥50 every merged/remerge state collapses to base (06-12). The bottleneck is the merge method, not the shard recipe (rank/epoch axes are flat; 06-11).
- **The naive literal weight-1.0 additive sum collapses**: `merged_additive` of 10 k=10 shards gave **mu 0.0** (forget_ppl ~26k). The CPU test proves the merge is exactly Σ scalingᵢ·BᵢAᵢ, so this is **norm overshoot** (blow-up scales with #summed terms), not the rsLoRA √r artifact (06-20).
- **Equal-shard composition has a ~0.48 ceiling**: a λ-sweep over 10 equal shards peaks at λ≈0.1 (mu ~0.484 ≈ dare_ties) and falls fast — no λ lifts it above ~0.48 (the co-adaptation ceiling). Conclusion: **don't shard the retain side** (06-20).
- **`linear`/`cat` and un-sparsified merges (`tsv`, `slerp`) are pathological**: `tree_remerge_linear` hit forget_ppl ~2.35M with mu 0.0 (06-08); `merged_tsv` 0.051 / `tree_root_slerp` 0.090 collapse real-authors like `merged_linear` — the TIES/DELLA/Fisher selection step is what protects real-world knowledge (06-11).
- **Per-shard undertraining confound at high k**: fixed e5 = ~6 optimizer steps/shard at k=200; r1@k200 is a near no-op (every label ≈ base) and shard_{k-1}_only barely memorizes its author (f_ppl 9.6 at k=200 r8 vs 1.40 at k=50) (06-12).
- **High-k JD memory + build walls**: PEFT casts adapters to fp32 so in-model k=200 r32 eval needs ~65 GiB (impossible on a 46 GiB A40); the k=200 c10 JD build exceeded a 2.5 h wall and was dropped (06-12/06-17). TOFU author-shards are far less jointly compressible than the paper's task-LoRAs (recon **0.87 at k100/c7** vs the paper's <0.6 regime) — `select_num_clusters` likely needs c ≫ 7 (06-17).
- **Metric/eval caveats**: `forget_quality` (KS) is non-discriminative at small forget slices (smoke-cap quantization); all headline numbers are **single seed (42), smoke-tier** — seed-variance and extended-cap passes are pending throughout. Pre-port utility numbers (e.g. ft 0.2507, merged 0.17–0.28) were broken-metric artifacts and are not comparable.

## Open ideas / next steps
- **Steps-matched high-k arm** (k50 e12 / k100 e25 / k200 e50, ~62 opt steps) to remove the undertraining confound — hypothesis: routed mu ≈ 0.7 flat in k (06-12).
- **Promote routing to a headline serving mode** alongside merging in the report/CLAUDE.md; add `routed_centroid_sbert` (no author-name dependence) as a robustness check against the ~0.86 lexical-routing accuracy ceiling (06-12).
- **Additive Phase 1 forget05/forget01**: compose the strong core + kept k200-r32 tail adapters at λ via a cross-dir multi-adapter additive path; extended-cap confirm + TOFU-plane report (forget10 headline already in hand) (06-20).
- **JD**: refresh `select_num_clusters` guidance for large TOFU collections (likely c ≫ 7); k=200 datapoint only if wanted (raise build `--time` ≥6 h / cut reassignment GPU-sync) (06-17).
- **Cross-cutting rigor**: seed-variance pass and extended-cap re-runs before any paper-style claim; refresh the remaining 5 models' ft-vs-base under the corrected metric (deferred scope) (06-11/06-12).

## Entries (chronological)
- [2026-06-04 — smoke eval k=4](2026-06-04_smoke-eval-k4.md) — remerge_cat forgets hard; dare_ties best utility merge
- [2026-06-05 — extended k=10 + 3B](2026-06-05_extended-k10-3b.md) — extended caps confirm 1B; only phi-2 forgets at k=10
- [2026-06-08 — tree-merge ablation](2026-06-08_tree-merge-ablation.md) — tree merge no better than flat; DARE rescaling key
- [2026-06-09 — rank/epoch ablation](2026-06-09_rank-epoch-ablation.md) — rank/epoch modest effect; dare_ties retains utility
- [2026-06-10 — OU metric port](2026-06-10_ou-metric-port.md) — eval made faithful to open-unlearning (0.5996 ≈ 0.60)
- [2026-06-10 — k=1 utility baseline](2026-06-10_k1-utility-baseline.md) — corrected base mu 0.42; overnight ≥0.6 recipe grid
- [2026-06-11 — grid + 0.6 bar](2026-06-11_grid-0p6-bar.md) — k=1 winner mu 0.74; 8 new merge methods; defaults frozen
- [2026-06-12 — k-scaling sweep](2026-06-12_k-scaling-sweep.md) — routing mu 0.71 @k=50; merging dies under dilution
- [2026-06-15 — JD compression](2026-06-15_jd-compression.md) — Compress-then-Serve port; O(1) keep/drop, recon 0.62 @k10
- [2026-06-17 — JD phase 2](2026-06-17_jd-phase2.md) — JD beats dare_ties at k100 (mu 0.465); k200 dropped on build wall
- [2026-06-20 — additive shards](2026-06-20_additive-shards.md) — naive sum collapses; coarse retain-core hits mu 0.7537

## Full reports
- [SMOKE_EVAL_REPORT_PRE_OUFIX.md](../../tofu_sisa_lora/reports/SMOKE_EVAL_REPORT_PRE_OUFIX.md) — Llama-3.2-1B k=4 smoke eval (06-04; pre-OU-metric-port numbers)
- [EXTENDED_EVAL_REPORT_3B.md](../../tofu_sisa_lora/reports/EXTENDED_EVAL_REPORT_3B.md) — Llama-3.2-3B k=4 extended eval (06-05)
- [smoke_eval_report.md](../../tofu_sisa_lora/reports/smoke_eval_report.md) — k=10 multi-model smoke eval (06-05)
- [SHARD_GRID_REPORT_2026-06-11.md](../../tofu_sisa_lora/reports/SHARD_GRID_REPORT_2026-06-11.md) — shard-recipe star grid + merge-method comparison + frozen defaults (06-11)
- [SCALE_REPORT_2026-06-12.md](../../tofu_sisa_lora/reports/SCALE_REPORT_2026-06-12.md) — k-scaling frontier (k up to 200) + routing results (06-12)
- [S3T_PAPER_REPRO_2026-06-17.md](../../tofu_sisa_lora/reports/S3T_PAPER_REPRO_2026-06-17.md) — JD / Compress-then-Serve compression results (06-15/06-17)
- [ADDITIVE_SHARD_REPORT_2026-06-20.md](../../tofu_sisa_lora/reports/ADDITIVE_SHARD_REPORT_2026-06-20.md) — additive true-scale shards + coarse-core headline (06-20)
