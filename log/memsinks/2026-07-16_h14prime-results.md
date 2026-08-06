### Target Date: 2026-07-16 (H14′ results — divergence fixed, but the strict arm UNDERFITS at 5 steps/author; capacity verdict still not licensed)
- **Hypotheses / what we're testing:** H14′ as pre-registered in
  [2026-07-16_h14prime-preregistration.md](2026-07-16_h14prime-preregistration.md):
  scale-corrected strict isolation (frozen lora_A std=1/√fan_in, clip 0.3). Gates: mu ≥
  0.55 AND own-prob ≥ 0.80 AND unlearn forget_rouge ≤ 0.45; training-health gate final loss
  < 1.5 (capacity REFUTE licensed only if training converges).
- **Setup:** Jobs **443939** smoke ✓ → **443940** train (1000 steps, 6.1 min, `lora_A frozen
  (IRP seed 42, std=auto)`, distinct-ID OK) → **443941** evals 0-2%1 → **443942** probe.
  Sequential 1-GPU (held + auto-released; another session's %4 array was Resources-limited).
  Config `memsinks_tofu_1b_strict2.json`; same seeded frozen-A DIRECTIONS as the diverged
  run — only scale/clipping changed.
- **Results:** **Training:** stable (no blow-up; loss 2.68 → 2.29–2.44 band, avg **2.51**)
  but plateaued ≫ the 1.5 health gate; train-end memgap probe own-mask prob 0.058–0.312
  (deleted 0.006–0.051 → gap 0.171, all content in the slices as designed).

  | label | mu | fq | forget_rouge | retain_prob | retain_rouge | real/world prob | ppl f/r |
  |---|---|---|---|---|---|---|---|
  | strict2_routed_full | 0.4466 | 0.9578 | 0.4730 | 0.1718 | 0.4805 | 0.6305/0.6556 | 11.8/11.3 |
  | strict2_routed_unlearn | 0.4466 | 0.3929 | 0.4647 | 0.1718 | 0.4805 | 0.6305/0.6556 | 17.6/11.3 |
  | strict2_all_on | 0.0305 | 0.0346 | 0.0959 | 0.0139 | 0.0896 | 0.272/0.303 | 93/92 |

  **Probe (200 authors):** gen_only (pure scaffold) 0.1396, gen_own **0.1627** →
  slice_increment **+0.0231** (vs −0.139 diverged, +0.013 in M1); all_on 0.0146;
  ladder monotone fraction 0.15 (interference present but noisy at these low probs).
- **What worked / hypothesis verdict:**
  - **H14′ REFUTED-AS-UNDERFIT.** The scale fix worked (stable training, slices now carry
    ALL the learning — deleted-condition prob ~0.01–0.05, isolation exactly as designed),
    but 40 frozen-basis rows/author learned almost nothing in this regime: own-author prob
    0.16 ≪ 0.80 gate, mu 0.4466 < 0.55, forget_rouge barely above the scaffold floor
    (0.473 vs 0.465). Training-health gate FAILED on the underfit side (avg 2.51 > 1.5) →
    per pre-registration, the **capacity refutation is still NOT licensed**.
  - **Confound identified:** author-block batching gives each author exactly **5 optimizer
    steps** (1/epoch × 5). merge_mechanism's H7 (2026-07-09) showed 20-row units need ~25
    steps to memorize even with FULL trainable LoRA. Frozen-A expressivity vs
    steps-per-author are confounded at e5.
  - **strict2_all_on (collapse demo) now stands:** with a converged-stable model, serving
    all 8000 slices at once collapses to mu 0.03 / ppl ~93 — the merging-collapse regime
    reproduced inside one adapter (citeable, unlike the diverged run's version).
  - Curio: routed_full fq 0.9578 — the weakly-trained model's forget truth-ratios are
    KS-indistinguishable from the retain90 oracle because it barely trained on ANYTHING;
    high fq without utility is vacuous (do not quote out of context).
- **Observations:** deletion mechanics remain provenance-exact and O(1) throughout; OOD
  serving (scaffold) intact in every condition (0.63/0.66). The three strict runs now form
  a clean dial: std-1.0/no-clip = divergence; std-auto/clip at e5 = stable underfit;
  the missing point is steps.
- **New questions / new hypotheses:** **H14″ (steps dial):** e25 strict2 (25 steps/author,
  ~30 min train) — CONFIRM (capacity sufficient): own-prob ≥ 0.80 with mu ≥ 0.55; REFUTE
  (real capacity floor, finally licensed if loss converges < 1.5): own-prob still < 0.6 →
  40 frozen-basis rows genuinely cannot hold 20 QA pairs, the quantified contrast to
  merge_mechanism's e25 full-LoRA 0.9991. NOT submitted — user review gate.
- **Next Steps:** report to user with the e25 option; REPORT.md + ledgers updated.
