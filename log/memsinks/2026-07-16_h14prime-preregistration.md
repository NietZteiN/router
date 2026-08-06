### Target Date: 2026-07-16 (H14′ pre-registration — scale-corrected strict-isolation retry)
- **Hypotheses / what we're testing:** User-approved retry of the strict-isolation arm with
  the divergence root cause fixed; registered BEFORE submission. **H14′:** with frozen
  lora_A at LoRA-scale init (std = 1/√fan_in ≈ 0.0221 vs the diverging std-1.0) and
  gradient clipping restored (0.3), the strict arm trains, and ~40 lora_B rows/author/layer
  suffice for author recall with data-provenance-exact deletion. Gates (same as H14):
  CONFIRM = strict2_routed_full mu ≥ 0.55 AND probe gen_own own-author answer-prob ≥ 0.80
  AND strict2_routed_unlearn forget_rouge ≤ 0.45 with retain ±0.02. **Training-health gate
  (new, from the 443563 lesson):** final train loss < 1.5 with no mid-run blow-up — if it
  diverges again, the verdict is again optimization, not capacity. REFUTE (capacity, now
  licensed only if training CONVERGES): mu < 0.50 or own-prob < 0.6 → 40 frozen-basis rows
  below the per-author capacity floor (vs SEA's rank-8-suffices). Provenance note: clipping
  is compatible with the exactness claim — under author-block steps the clip factor of a
  step is a function of that author's own gradients only (a shared-scalar schedule effect,
  same tier as LR position).
- **Setup:** `configs/memsinks_tofu_1b_strict2.json` = strict config + `"max_grad_norm":
  0.3` + `"lora_a_std": "auto"`, output_dir `..._memsinks_strict2`; `freeze_lora_a_irp`
  gained a `std` param ("auto" = per-layer 1/√fan_in; default 1.0 keeps the bit-equivalence
  gate with apply_irp_projections). CPU gates now **22/22** (new: auto-std = same seeded
  directions at 1/√fan_in scale). Driver: `STRICT_CFG=configs/memsinks_tofu_1b_strict2.json
  bash submit_memsinks.sh e3` (same chain shape: micro-smoke → train → 3 evals + probe,
  sequential 1-GPU footprint if the shared cap requires). Everything else identical to the
  2026-07-15 pre-registration (scaffolded base, gate/up r32, disjoint_dead, author-block
  bs4×ga5, wd 0, dropout 0, seed 42). Job IDs recorded at submission.
- **Results:** pending.
- **What worked / hypothesis verdict:** OPEN.
- **Observations:** (design) std="auto" reuses the identical SHA-256 per-layer seeds, so the
  frozen directions are THE SAME as the diverged run — only the scale changes; any outcome
  difference is attributable to scale/clipping alone.
- **New questions / new hypotheses:** if H14′ confirms, next gates would be seeds 43/44 +
  extended tier + MIA before any storage-headline claim (~80 KB/author vs SEA 32 MB).
- **Next Steps:** submit → harvest → verdict entry + REPORT.md update → user review.
