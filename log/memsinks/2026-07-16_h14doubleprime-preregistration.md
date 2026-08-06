### Target Date: 2026-07-16 (H14″ pre-registration — the e25 steps dial on the strict arm)
- **Hypotheses / what we're testing:** User-approved third strict-arm run; registered BEFORE
  submission. **H14″:** the H14′ underfit was steps-per-author, not frozen-basis capacity —
  at 25 optimizer steps/author (epochs 25, author-block bs4×ga5, everything else identical
  to strict2), 40 lora_B rows/author/layer memorize the author's 20 QA pairs. CONFIRM
  (capacity sufficient): probe gen_own own-author answer-prob ≥ 0.80 AND routed_full mu ≥
  0.55 AND routed_unlearn forget_rouge ≤ 0.45 with retain ±0.02 → reopens the
  SEA-at-1/400th-storage line (seeds/extended/MIA required before claiming). REFUTE (the
  per-author capacity floor, licensed ONLY with training health: final loss < 1.5 or
  visibly converged/plateaued at its floor with own-prob still < 0.6): 40 frozen-basis rows
  cannot hold 20 QA pairs — quantified contrast to merge_mechanism's e25 full-LoRA prob
  0.9991 (2026-07-09_iso-rank-epochs-results.md). MIXED (own-prob 0.6–0.8 or loss still
  falling at 25 epochs): report trajectory, no verdict.
- **Setup:** `configs/memsinks_tofu_1b_strict2_e25.json` (sha f012a29f4f5c) = strict2 +
  `"epochs": 25` + output_dir `..._memsinks_strict2_e25`. 5000 steps ≈ 30 min. No code
  changes (22/22 CPU gates unchanged). Driver: `STRICT_CFG=<cfg> bash submit_memsinks.sh
  e3`; cap discipline as before (sequential 1-GPU, hold+auto-release if the queue is
  owned). Job IDs at submission. The 25-epoch MemGapProbe telemetry gives the own-vs-deleted
  trajectory across all 25 epochs for free.
- **Results:** pending.
- **What worked / hypothesis verdict:** OPEN.
- **Observations:** —
- **New questions / new hypotheses:** —
- **Next Steps:** submit → harvest → verdict + REPORT.md update → user review (thread-close
  decision).
