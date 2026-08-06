### Target Date: 2026-06-20 (Additive true-scale LoRA shards — Phase 0 cheap test on existing k=10 7B shards)
- **Goal / Hypothesis:** The exact-unlearning research doc (`~/.claude/plans/exact-unlearning-on-deep-platypus.md`)
  claims independent shard adapters compose by the LITERAL additive sum `W + Σᵢ (α/r) BᵢAᵢ`, with unlearning
  = drop a term. Prior shard-grid found `linear`/`cat` broken (ppl 10³–10⁷) but blamed the rsLoRA √r
  double-count, never cleanly testing the doc's claim. H0: once √r inflation is removed (TRUE-scale sum),
  additive composition retains utility at k=10. Cheap test — no training, reuse existing shards.
- **Setup:** New true-scale merge `merge_extra.additive_merge_adapters` (cat scaffold = full rank Σrᵢ,
  sums each effective delta scalingᵢ·BᵢAᵢ once, divides out scaffold scaling, NO compression); registered
  `additive` + `additive_mean` (weight 1/n_active soup variant) in `merge_lora.MERGE_METHODS`/`_dispatch_custom`
  → labels `merged_/remerge_additive[_mean]`. CPU gate GREEN: `test_merge_extra.test_additive`
  merged==Σ (err 3.0e-7), remerge==merged−forget-term (err 3.2e-7), additive_mean==(1/n)Σ (err 3.2e-7),
  deterministic; `test_ou_equivalence` green. Scripts (sha256-16): merge_extra fed5f354a0adbad0,
  merge_lora 5d01175600f91715, test_merge_extra b26662d3398d4b5b. Eval (seed 42, smoke, `--k 10
  --forget_shard_id 9`, reuse cached retain90 KS oracle) on `checkpoints/Llama-2-7B-chat-hf/` shard_0..9:
  `merged_additive`+`remerge_additive` = SLURM **435424** `[0-1%2]` (sprint1-3, 48G); true-scale-MEAN
  contrast `merged_/remerge_additive_mean` = **435579** (PENDING at writing). submit_eval.sh + 2-line
  manifests `results/smoke/eval_manifest_additive[_mean].txt`.
- **Results (435424, smoke):** `merged_additive` (all 10 summed) **mu 0.0** — forget_ppl 26047, retain_ppl
  27689, retain_prob 1e-4, retain_rouge 0.053, retain_truth_ratio 0.926, forget_quality 0.135.
  `remerge_additive` (9 summed, drop shard_9) **mu 0.0** — forget_ppl 11549, retain_ppl 11299,
  retain_prob 1e-4, forget_quality 0.071. Anchors: base mu 0.418 (ppl 15.2), joint ft 0.740,
  remerge_dare_ties ~0.45–0.48, merged_linear@r32 ppl ~82940 (√r-inflated). additive_mean (435579) pending.
- **Observations:** **H0 REJECTED for the weight-1.0 sum.** The collapse is NOT the √r artifact —
  the CPU test proves the merge is *exactly* Σ scalingᵢ·BᵢAᵢ, yet ppl explodes to ~10⁴. It is NORM
  OVERSHOOT: summing k full-strength adapters (each trained to fully express its shard vs the bare base)
  moves the weights ~k× too far. Confirming signal: remerge (9 terms, ppl 11.5k) is *less* broken than
  merged (10 terms, ppl 26k) — blow-up scales with #summed terms. True-scale-sum (26k) is milder than
  √r-inflated-average linear@r32 (83k) but still catastrophic. The §3 drop-a-term exactness invariant
  itself is bit-exact (CPU-proven); the failure is purely magnitude. This is exactly the doc's §6
  "summing them can interfere destructively" / §8 interference-reduction surface — surfaced cheaply
  before paying for the faithful retrain.
- **Next Steps:** Read 435579 → does the true-scale MEAN (delta-norm ~1 adapter, soup regime) recover
  utility? If yes ⇒ the method works with mean aggregation (unlearn = recompute mean over kept shards,
  still data-exact) and Phase 1 retrain should use mean, not raw sum. If mean also collapses ⇒
  independent-adapter interference is deeper ⇒ §8 levers (orthogonality-constrained adapters; fold the
  180-author retain core into base so only ≤20 tail adapters are composed) BEFORE the 38-shard retrain.
  Decision-gate the faithful Phase-1 retrain on this. No utility numbers fabricated; 435579 cells empty
  until it lands.
- **UPDATE (same day) — mean recovers + Fix A/B free diagnostics (jobs 435579, 435630, 435635):**
  `additive_mean` (λ=1/k): merged mu **0.481** / remerge **0.484**, ppl 4.5, fq 0.808 (≈dare_ties, ≫base;
  remerge ≥ merged ⇒ unlearning utility-neutral). Added a global-λ knob (`_s{λ}` label →
  `_split_scale_suffix` → weights=[λ]·n; fixed λ keeps exact drop-a-term; CPU test green).
  **Fix A λ-sweep on existing k=10 shards (435630):** remerge mu by λ = 0.10:0.484, 0.15:0.462,
  0.20:0.414, 0.25:0.221, 0.30:0.052, 0.50:0.065; merged tracks lower. ⇒ composing 10 equal shards
  PEAKS at λ≈0.1 (~0.48) and falls fast — **no λ lifts equal-shard composition above ~0.48 (≈dare_ties);
  that is the co-adaptation ceiling for many equal independent shards.** **Fix B (435635): retain90
  core ALONE** (single jointly-trained retain adapter, legacy r8/e3 recipe) **mu 0.629** (retain_ppl 2.1,
  retain_prob 0.534, fq 1.000) — **decisively beats equal-shard composition (0.48) and approaches joint
  ft (0.74), at the WEAK recipe.** Conclusion: **don't shard the retain side.** Winning structure =
  coarse retain-core (one jointly-trained adapter, never sharded) + fine forgettable tail. Provenance:
  jobs 435424/435579/435630/435635; manifests results/smoke/eval_manifest_additive*.txt; scripts
  merge_extra/merge_lora/test_merge_extra updated (see CLAUDE.md Additive entry).
- **Revised Next Steps:** Phase 1 (gated decision = coarse-core) retrains at the strong recipe: one
  retain90 core (authors 0-179, r32/e5 → expect ~0.73) + 20 per-author tail adapters (180-199); unlearn
  forget10 = core alone, forget05/01 = core + kept tail composed at λ (sweep). Headline = unlearned mu
  vs joint-retain-ft, fq ≈ ceiling by construction. Confirm Phase-1 scope with user before training
  (busy cluster). The user's original "38-way equal additive" choice is superseded by this evidence.
- **PHASE 1 (minimal headline; user chose "minimal first") — DONE (job 435657):** trained a STRONG
  retain90 core (authors 0-179, r32/α64/e5/lr1e-4 = the joint-ft recipe; new dir
  `Llama-2-7B-chat-hf_retainstrong/retain90`, legacy KS oracle untouched; loss 2.37→0.20 clean), eval'd
  standalone via `--preloaded_adapter` (= the forget10 UNLEARNED state). **Result: model_utility 0.7537**
  — matches/edges joint-ft (0.740), ≫ legacy-core 0.629, ≫ equal-shard 0.48, ≫ naive sum 0.0. retain_ppl
  1.2, retain_prob 0.967, retain_rouge 0.936; **forget_ppl 14.2 ≈ base 15.2** (forget authors genuinely
  absent); **forget_quality 0.958** (near ceiling, by construction — core never saw forget data). JSON
  `results/smoke/retain90_strong_alone.json`. **HEADLINE: additive coarse-core unlearning reaches
  near-joint-ft utility at O(1) deletion cost, vs the 0.0 collapse of the literal weight-1.0 sum.**
- **Next (optional, deferred):** forget05/forget01 (compose strong core + kept k200-r32 tail adapters at
  λ; needs a cross-dir multi-adapter additive path) + extended-cap confirm + TOFU-plane report. forget10
  headline already in hand.

