# entangled_facts — does deleting one owner remove a fact several owners hold?

**Status:** active · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 3 (2026-07-03 → 2026-07-07)

The Mode-B replication test the [gap analysis §5.1/§6/§9-A](../EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)
says no paper runs and TOFU has no metric for. Deliberately plant replicated facts across several
TOFU authors (donor + R−1 host authors), train the routing+scaffold system, drop ONE owner via the
structural O(1) `--delete_shard`, and measure whether the fact still answers — the residual-fact-recall
(RFR) curve vs replication factor R. Produces the paper's honest scope statement: **owner-level
deletion is exact (post-drop == retrain-without-owner-X by construction); fact-level erasure fails
under replication (the surviving owners legitimately keep their copies) — here is how much survives,
and a SEUF-attribution detector that flags when a delete request needs propagation.**

Built on the current headline arm `Llama-3.2-1B-Instruct_experts_scaf_k10` (routing+scaffold,
r32/α64/e5/lr1e-4, seed 42, mu 0.7509). Paraphrases come free from TOFU's `forget10_perturbed`
(train on the paraphrase, probe on the original question ⇒ fact-level, not string-level).

## Hypotheses — open / resolved
- **[resolved ✓ supported]** H1: owner-level exactness — `served_key` postdrop == floor to 3
  decimals across all cells (`2026-07-06_residual-curve-results.md`).
- **[resolved ✓ supported]** H2: residual monotone in R — verbatim `expert_max` ρ =
  0.955/0.986/0.998 at R=2/4/8 vs R1 control ≈ 0.01.
- **[resolved ✓ supported]** H3: fact-level not string-level — verbatim-planted facts answer the
  PARAPHRASED question at ρ 0.79–0.95.
- **[resolved ✓ supported]** H7 (BOTH halves now measured): `served_key` == floor (hard router
  hides the residual) while `served_embedsim` **surfaces** it — ρ_embed 0/0.107/0.439/**0.833** for
  R=1/2/4/8 vs ρ_key = 0 everywhere (`2026-07-07_embed-route-surface.md`). Leak = host-hit rate ∝ R.
- **[partial]** H5: SEUF detector AUC 0.777 (below the 0.9 target) but spread monotone in R
  (0.39→0.66); host-identification recall 0.495.
- **[open]** H4: TOFU fq blindness shown qualitatively (served clean while ρ→0.998); not KS-tabulated.
- **[open]** H6: delete-propagation not run as a served arm (floor_clean is the target; ρ→0 by construction).

## What worked
- **The three-thread climax (2026-07-07):** served through the SAME weights, the hard author-key
  router hides the Mode-B residual (ρ_key=0) while a realistic embedding router surfaces it
  (ρ_embed→0.833 at R=8, monotone in R = the host-hit rate). *The fact survives deletion, and which
  router you deploy decides whether it leaks* — and the leaky router is exactly the one §9-D found
  sends orphans to plausible siblings and can't be threshold-sealed.
- **The headline Mode-B result (2026-07-06):** owner-level deletion is exact at the serving surface
  (H1/H7) yet the fact is NOT erased from the model (H2 ρ→0.998, fact-level per H3) — the §6
  ownership≠fact scope statement made concrete, with a monotone ρ-vs-R curve (scale is the threat).
- The plant took cleanly (ceiling answer-probs 0.83–0.92; R1 control ρ≈0 = donor's fact truly gone
  when no host holds it).

## What didn't / open problems
- Detector below the AUC-0.9 target (0.777) and host-identification only moderate (0.495) — needs a
  better spread statistic or the soft-router (RAMoLE) affinity readout.

## Open ideas / next steps
- Tune the detector; run an embedding-routed served arm to show it surfaces the H2 residual the hard
  router hides (predicted from §9-D sim-0.98); KS-tabulate H4; write the report.
- real_authors / world_facts as a free observational Mode-B footnote (public facts no author owns).

## Reports
- [`ENTANGLED_FACTS_REPORT_2026-07-06.md`](../../tofu_sisa_lora/reports/ENTANGLED_FACTS_REPORT_2026-07-06.md)
  — the ρ-vs-R curve, fact-level (paraphrase) transfer, served-surface hiding, and the SEUF detector.

## Entries (chronological)
- [2026-07-03 — Mode-B plant design](2026-07-03_mode-b-plant-design.md) — pre-registration +
  plant-manifest builder + CPU gate.
- [2026-07-06 — residual-curve results](2026-07-06_residual-curve-results.md) — owner-level exact,
  fact-level not; verbatim ρ→0.998 monotone in R; verbatim→paraphrase transfer (H3); detector AUC 0.78.
- [2026-07-07 — embed-route surfaces the residual](2026-07-07_embed-route-surface.md) — the climax:
  ρ_embed 0/0.107/0.439/0.833 (R=1/2/4/8) vs ρ_key=0 — the embedding router surfaces exactly what the
  hard router hides; leak = host-hit rate ∝ #host shards ∝ R. Ties B+C into one claim.
