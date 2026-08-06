### Target Date: 2026-07-16 (Negative-anchor λ-pilot results — anchoring shrinks, it does not select; H-anchor-1 REFUTED)
- **Hypotheses / what we're testing:** H-anchor-1 as pre-registered in
  [2026-07-15_negative-anchor-design.md](2026-07-15_negative-anchor-design.md): some
  λ ∈ {1,10,100} yields on/off ‖sBAh‖ selectivity ≥ 5 with own-author recall ≥ 0.98.
- **Setup:** as pre-registered. Trainings = array **443487** (15 tasks: probe authors
  {82,15,111,177,76} × λ ∈ {1,10,100}, e25 recipe + `--anchor_lambda`, 25 optimizer steps,
  seed 42; `_k200_r32_e25_anch{λ}_lr1e4/`). Selectivity = jobs **443523–443525**
  (`measure_key_firing.py`, same seed/harness as Exp-7 → ratios directly comparable;
  `reports/key_firing_e25_anch{λ}.json`). Recall = iso array **443536** (15 × `eval_tofu
  --preloaded_adapter --eval_shard_id`, smoke). One deviation from the design: smoke iso
  rows carry `forget_rouge`/`forget_ppl` (no forget answer-prob at smoke caps), so the
  recall bar is read as rouge ≥ 0.98 vs the H7 references (unanchored e25: rouge 1.0,
  ppl 1.06) — the verdict is insensitive to this substitution (see Results).
- **Results:** (per-λ means over the 5 probe adapters; baseline = unanchored e25)
  | arm | gate median on/off ‖sBAh‖ | verdict | mean on-firing | own rouge | own ppl | single-adapter mu | world_prob |
  |---|---|---|---|---|---|---|---|
  | e25 baseline | 1.110 | LAZY | 0.633 | 1.000 | 1.06 | 0.3882 | 0.60–0.72 |
  | anch λ=1 | 1.150 | LAZY | 0.279 | **0.997** | 1.07 | 0.4277 | 0.69 |
  | anch λ=10 | 1.123 | LAZY | 0.131 | 0.924 | 1.19 | 0.4530 | 0.73 |
  | anch λ=100 | 1.112 | LAZY | 0.057 | **0.525** | 2.61 | 0.4615 | 0.75 |

  Per-adapter selectivity ratios never leave [1.08, 1.17] at any λ (all 15 adapters LAZY);
  read-side ‖Ah‖ ratios 0.99–1.03. OOD firing stays ≈ 90–93% of on-author at every λ.
  Recall variance grows with λ (λ=10: a111 0.77 vs a177 0.99; λ=100 range 0.44–0.61).
  Truth-ratio NaN for 4/5 probes = the known sparse-perturbed-rows artifact (Exp-5b), not a
  failure.
- **What worked / hypothesis verdict:**
  - **H-anchor-1 REFUTED, across the entire λ range:** the penalty produces uniform
    magnitude shrinkage (on-firing 0.63 → 0.28 → 0.13 → 0.06, off-firing shrinking in
    lockstep) and **zero selectivity gain** (1.11–1.15 ≈ baseline 1.11 at every λ), while
    recall degrades monotonically (0.997 → 0.924 → 0.525). There is no λ that is selective;
    there is not even a λ that trades recall for *any* selectivity. This is the
    pre-registered "keys can't decouple" refutation in its strongest form.
  - **Per the pre-registered decision tree, H-anchor-2 (the anchored H8 ladder) does NOT
    run** — no arm meets the ≥5-selectivity + ≥0.98-recall bar. §6.3 is closed as a
    causal negative.
- **Observations:** (i) Mechanism reading: Exp-7 showed adapters read essentially the same
  hidden-state directions for on-author, off-author, and OOD text (‖Ah‖ ratio ≈ 1.0). Given
  indistinguishable inputs, no output-norm objective can separate them — the optimizer's
  only degree of freedom is global scale, and that is exactly what it used. Self-gating
  cannot be trained into a LoRA with output-norm negatives; input-conditioned selection has
  to live OUTSIDE the adapter (router, mask) or in a different storage substrate
  (content-derived keys — the Path-C/MEMIT cell of the §5 table). The §7.1 "self-gating
  experts" router-seal candidate is likewise dead in this form. (ii) Bonus finding: the
  penalty is a clean **collateral-damage dial** — single-adapter mu rises 0.388 → 0.462 and
  world_prob 0.60–0.72 → 0.75 as λ grows (smaller adapter output = less damage to general
  knowledge), the H8 "heavily-memorized expert damages general components" effect run in
  reverse. λ=1 is strictly better than unanchored on this axis (recall intact at 0.997,
  collateral reduced, mu +0.04) — worthless for selectivity, possibly useful as a
  regularizer for *routed* serving where selection is external. (iii) Silent-failure
  checks: all 15 trainings saved (loss curves show the penalty active, e.g. λ=100 loss
  7.33 → 2.95); keyfire harness identical to Exp-7; iso rows complete; NaN TRs explained.
- **New questions / new hypotheses:** (1) Does λ=1 anchoring improve the ROUTED composition
  (less cross-expert damage at equal recall)? Cheap: routed eval over an anchored k=10 pool
  — only if routing work resumes. (2) Token-level selectivity (name-token positions only)
  from the saved `_matrices.npz` — could firing be selective at 2–3 tokens and washed out by
  the mean? CPU-only follow-up. (3) The mechanism paper's intervention section now has both
  causal arms: centering (composition-side) and anchoring (training-side, negative) — fold
  into the Path-A write-up.
- **Next Steps:** close §6.3 in the thread README (H-anchor-1 ✗, H-anchor-2 not run per
  pre-registration); Exp-6 centered-merge collect + results entry (eval wave 443532 done,
  125/126 + stray rerun 443925); MEMIT (Path C rung 3) inherits the selection-in-weights
  bet.
