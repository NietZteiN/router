### Target Date: 2026-06-29 (merge-mechanism: subspace overlap + iso/merged harness)
- **Goal / Hypothesis:** Establish *why* merging LoRA adapters destroys factual recall. H1: isolated
  fact adapters' effective deltas `scaling·BA` collide in a shared low-rank subspace (so averaging
  them interferes). H2 (Exp 2): no global scale λ on the additive merge recovers recall. H3 (Exp 3):
  each adapter's own-author recall drops isolated→merged, growing with k. Plan:
  `~/.claude/plans/sequential-meandering-token.md`.
- **Setup:** Llama-3.2-1B. Exp 1 CPU: new `tofu_sisa_lora/subspace_overlap.py` (+`test_subspace_overlap.py`,
  all CPU tests green incl. factored==dense Frobenius identity to 1e-9), reusing
  `jd_collection.build_collection_slots` / `jd_compress._top_r_subspace_from_blocks`. Ran on three
  isolated fact-adapter sets: SISA `k=4` (`..._k4/shard_0..3`), SISA `k=10`
  (`Llama-3.2-1B-Instruct/shard_0..9`), LegoNet `n=32` (`..._legonet_n32_k3/legonet/adapters/a*`,
  hub-collapsed → secondary). seed 42, n_null 20 (12 for n32). Cmd e.g.
  `python subspace_overlap.py --adapters .../shard_0..9 --rank 16 --n_null 20 --seed 42 --out reports/subspace_overlap_k10.json`.
  Exp 2/3 GPU: added `eval_tofu.py --eval_shard_id` (scores forget_* on an arbitrary shard's own
  authors; `None`=legacy, asserted unchanged; `test_ou_equivalence.py`+`test_merge_extra.py` stay
  green); new `submit_iso_merged.sh` (tab-separated `<label>\t<sid>` manifest); Exp-2 λ-sweep reuses
  `submit_eval.sh` env hooks + a `merged_additive_s{λ}` manifest (λ∈{0.05..10}). GPU smoke = job 439297
  (shard_0_only vs merged_dare_ties, both `--eval_shard_id 0 --smoke`, + `merged_additive_s0.1`).
- **Results:** Exp 1 (real, on disk in `reports/subspace_overlap_{k4,k10,n32}.{json,csv}`):
  principal-angle cos on **col(B)** (output subspace) k4 **0.232** / k10 **0.262** / n32 **0.277**
  vs random-orthogonal null ≈ 0.09; **row(A)** (input) cos ≈ 0.055–0.083 (near-orthogonal); shared
  rank-16 basis energy retained vs chance: k4 **0.845**/0.50, k10 **0.650**/0.20, n32 **0.457**/0.031
  (≈14.6× chance at n32 — ratio grows with adapter count). Whole-delta cosine is small (0.02–0.04,
  but z≫0 vs null) because it is dominated by the orthogonal input side. Exp 2/3 numbers: pending
  (smoke 439297 in queue; full arrays gated on its success).
- **Observations:** The collision is **localized to the output side** (where the delta writes), not
  the input side — facts route from near-orthogonal input directions into a *shared* output subspace,
  exactly the geometry that makes an averaged merge superimpose many authors' writes and blur recall.
  The energy-vs-chance ratio rising with n is consistent with merge utility collapsing as k grows
  (`merged_dare_ties` 1B = 0.4236). Hub-collapse confounds n32 magnitudes (caveated; k4/k10 balanced
  are the clean carriers).
- **Next Steps:** (1) Read smoke 439297 go/no-go (isolated forget_rouge > merged); on PASS launch the
  full arrays — Exp 2 λ-sweep (25 labels via `submit_eval.sh`) + Exp 3 k10 (30) / k4 (12) via
  `submit_iso_merged.sh`. (2) Collect → `reports/lambda_sweep_1b.csv`, `reports/iso_merged_drop.csv`,
  combined `reports/MERGE_MECHANISM_REPORT_2026-06-29.md`. (3) Test whether col(B) overlap predicts
  the per-adapter Exp-3 drop. (4) Part B: facts-vs-skills (Super-NaturalInstructions, N=20 balanced).
