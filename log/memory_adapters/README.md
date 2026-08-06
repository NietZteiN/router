# memory_adapters — product-key memory adapter with per-source entries; unlearn = block-list (Grimes et al. 2026 reproduction)

**Status:** active · **Project:** [`memadapt_tofu/`](../../memadapt_tofu/) · **Entries:** 4 (2026-07-14 → 2026-07-15)

Reproduction of "Memory Adapters Enable Fast, Flexible Knowledge Unlearning in LLMs"
(Grimes, Kuo, Wu, Smith, Connor — ICML 2026 workshop; PDF in `papers/`). A frozen
Llama-3.2-1B-Instruct gets a single product-key memory layer (N=1024² entries,
value_dim 2048) added to one MLP layer; each TOFU author owns 256 disjoint entries
(TF-IDF assignment over a frozen random router), and sequence-level gradient masking
ensures each author's data only ever writes its own entries. Unlearning any author =
adding their 256 entries to a −∞ block-list: O(1), no training, per-query flexible.
This is the purest instance so far of the repo's "selection inside the weights, built
in at training time" frame (PATHS_FORWARD §5/§8): not exact unlearning (cross-source
*reads* during training leak shared knowledge — the paper's own A/B caveat), but
state-of-the-art TOFU privacy scores with zero-cost deletion.

Target: Table 1 MemAdapt row — Util.R 1.00 / Util.G 1.06 / Mem. 0.62 / Priv. 0.98 /
Agg. 0.87 (± 0.03/metric), evaluated through open-unlearning with the paper's
aggregate metric composition (validated offline against the canonical public eval
logs before anything was trained).

## Hypotheses — open / resolved
- **[resolved ✓ supported]** H1 (zero-init no-op) — bitwise CPU test + clean GPU smoke (2026-07-14/15).
- **[resolved ✓ supported, overshot]** H2 (capacity) — FT Util.R **1.075** ≥ 0.90 (better than full finetuning; 07-15 entry).
- **[resolved ~ partial]** H3 (headline) — **Agg 0.869 vs paper 0.87 (|Δ|=0.001)**, Mem 0.630 ✓, but Priv 0.917 < 0.95 and Util.R/Util.G outside ±0.03 (07-15 entry).
- **[resolved ✗ refuted in our config]** H4 (collision reduction) — ΔUtil.R = −0.010 (paper +0.07): near-uniform routing leaves no collision penalty to recover (07-15 entry).
- **[resolved ✓ supported]** H5 (costless) — 5,120 entries blocked in **0.027 s** CPU; apply-at-load 0.021 s (07-15 entry).
- **[resolved ✗ refuted]** H6 (router temperature) — key_scale ×2/×4 sharpened weights (eff. reads 30.9→14.9) but Priv never rose (0.917→0.841→0.910) and Util.G collapsed (→0.851): selection is scale-invariant, so cross-source READS persist — leakage is selection-level, not weight-level (h6 entry).
- **[resolved ✗ not supported]** H7 — Agg monotonically worsens with sharpness (0.869→0.849→0.840); no favorable frontier along the temperature axis.
- **[open]** H8: top-k ∈ {8,16} (true selection sharpening) recovers Priv ≥ 0.95.
- **[open]** H9: part of the Priv gap is composition (priv_absdiff reads 0.953 at ks1).

## What worked
- Aggregate-metric reverse-engineering validated to anchor precision offline: composed Priv(Finetuned) = 0.3810 vs paper 0.38; Retrained Priv ≡ 1.00 by construction; Mem residuals (+0.017/+0.011) attributable to Grimes training their own reference checkpoints (2026-07-14 entry).
- Adversarial review before GPU spend caught a silent eval-invalidating bug: OU's `@package _global_` eval experiment file merges after the model group and swaps the base to the TOFU-finetuned checkpoint — fixed with an explicit CLI override (review-fixes entry; verified via hydra `--cfg job`).
- OU data-pipeline parity proven by direct import-and-compare tests (token-for-token vs `open-unlearning` `preprocess_chat_instance` + collator), 24/24 CPU tests green.
- Compact-table formulation (51,201 rows + pad instead of the dense 1M×2048 table) is bit-exact and shrinks params+grads+Adam to ~1.7 GB (vs ~34 GB dense).

## What didn't / open problems
- Full-table brute-force top-k ≠ deployed blocking semantics: under a block-list, product-key retrieval is inexact (blocked pairs crowd the per-half shortlists) — matched the paper's deployed semantics (block on the candidate grid) and documented instead of "fixing".
- The naive "masked grad == unmasked grad restricted to owned rows" invariant is wrong (cross-source reads contribute to the unmasked grad); the correct invariant — row grad == unmasked grad from the owner's own sequences — is what the test suite now checks.

## Open ideas / next steps
- GPU smoke (S2) → LR pilot (S2b) → calibration evals G1 (S3) → assignment + routing-health gate (S4) → 15-epoch train (S5) → blocked evals G2 (S6); seeds 43/44; RMU/GradDiff/NPO baselines; entries/source + total-size ablations (paper Fig. 2).
- Quantify cross-source leakage beyond the paper: cross-source read mass is logged per epoch; an oracle-gap probe (adapter trained on retain90 only) would price the leakage in Mem./Priv. terms.

## Entries (chronological)
- [2026-07-14 — scaffold and metric port](2026-07-14_scaffold-and-metric-port.md) — implementation stood up; Table-1 metric composition validated against canonical logs before any GPU spend
- [2026-07-14 — review fixes, smoke submitted](2026-07-14_review-fixes-smoke.md) — 2 confirmed review bugs fixed (critical: eval base-model clobber via `@package _global_`); S2 smoke = job 443276
- [2026-07-15 — first full run results](2026-07-15_first-full-run-results.md) — **Agg 0.869 reproduces the paper's 0.87**; Mem ✓, unlearn 0.027 s ✓; Priv 0.917 / Util.R 1.075 / ΔUtil.R −0.01 diverge via one mechanism (near-uniform router) → H6 temperature ablation
- [2026-07-15 — H6 temperature ablation](2026-07-15_h6-temperature-ablation.md) — H6 refuted: weight-sharpening (selection-invariant) leaves Priv flat and hurts Util.G; ks1 stays best; leakage is selection-driven → H8 k-sweep open
