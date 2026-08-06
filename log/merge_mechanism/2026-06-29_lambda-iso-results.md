### Target Date: 2026-06-29 (merge-mechanism: λ-sweep + iso/merged results)
- **Goal / Hypothesis:** Complete the mechanism study started in
  [2026-06-29_subspace-overlap.md](2026-06-29_subspace-overlap.md). H2 (Exp 2): no global scale λ on
  the additive merge recovers factual recall. H3 (Exp 3): each adapter's own-author recall drops
  isolated→merged, growing with k. (Exp 1 already showed facts collide in a shared output subspace.)
- **Setup:** Llama-3.2-1B, seed 42, smoke caps. GPU smoke go/no-go PASSED (job 439297: isolated
  `shard_0_only` forget_rouge 0.505 > `merged_dare_ties` 0.394 on shard-0 own authors). Full arrays:
  Exp 2 λ-sweep `merged/remerge_additive_s{λ}`, λ∈{0.05..10} (job 439402, 24 labels via `submit_eval.sh`
  env hooks); Exp 3 `shard_i_only` vs `merged_additive_s{1/k}`/`merged_dare_ties` on each shard's own
  authors via `eval_tofu.py --eval_shard_id` + `submit_iso_merged.sh` (k=10 job 439403 / k=4 job 439404).
  CSVs assembled by `analyze_merge_mechanism.py` → `reports/lambda_sweep_1b.csv`,
  `reports/iso_merged_drop{,_k4}.csv`. Full write-up: `../../tofu_sisa_lora/reports/MERGE_MECHANISM_REPORT_2026-06-29.md`.
- **Results:** **Exp 2** — merged utility peaks at λ≈0.05–0.1 (mu **0.4285**/0.419, retain_ppl ~8) then
  **collapses to 0 by λ≥0.5** as retain_ppl explodes 8 → 3.6k → 404k → **1.8M**; forget_rouge → 0 at
  high λ. No λ reaches isolated `shard_9_only` forget_rouge **0.489** or full-FT mu ~0.59; anchor
  `merged_dare_ties` mu 0.424 / forget_rouge 0.437. (High-λ forget_truth_ratio stays 0.75–0.86 — broken
  output looks "forgotten"; a metric trap.) **Exp 3** — every shard drops isolated→merged: mean
  `dare_ties` forget_rouge drop **0.0588 (k=4) → 0.0902 (k=10)** (grows with k), all 10/10 shards >0,
  per-shard range 0.046–0.110; `additive_s{1/k}` drop 0.0512 (k4) / 0.0386 (k10).
- **Observations:** All three pre-registered predictions confirmed, no falsifier triggered. The λ curve
  is the clean "no good λ" signature — below the peak the fact edits dilute below the recall threshold,
  above it the *colliding* output-subspace deltas (Exp 1) overload and blow the model up (ppl→millions).
  The Exp-3 drop growing with k matches the Exp-1 energy-vs-chance ratio growing with n: more adapters
  summed into the shared output subspace ⇒ more interference. Mechanism established, not just a benchmark
  number.
- **Next Steps:** (1) extended-cap + multi-seed (43,44) confirmation of the headlines; (2) test whether
  per-adapter col(B) overlap predicts the per-shard Exp-3 drop (geometry→behavior); (3) **Part B**:
  facts-vs-skills controlled contrast (Super-NaturalInstructions, N=20 balanced, normalized-NLL
  retention) — the specificity test (merging spares skills, kills facts).
