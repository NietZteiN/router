### Target Date: 2026-07-01 (merge-mechanism Part B: facts-vs-skills specificity)
- **Goal / Hypothesis:** Is the merge-collapse specific to FACTUAL knowledge, or a generic merge
  effect? H4: under an identical N-way merge, factual adapters collapse far more than compositional-
  skill adapters (because facts collide in a shared output subspace — Part A Exp 1 — while skills are
  near-orthogonal). Completes `../../tofu_sisa_lora/reports/MERGE_MECHANISM_REPORT_2026-06-29.md`.
- **Setup:** Controlled contrast, only the domain differs. Both arms: Llama-3.2-1B, LoRA r16/α32/rslora/
  [q,k,v,o,up,down], e5/lr1e-4, 200 train/adapter, N=20, merge=`add_weighted_adapter(linear,1/N)`.
  Facts = N=20 balanced TOFU shards (`train_lora_shard.py --k 20`, `checkpoints/..._facts_n20`, job
  439837); skills = 20 distinct Super-NaturalInstructions tasks (`Muennighoff/natural-instructions`
  per-file, input-only; `checkpoints/..._skills_n20`, job 439857). New code: `skill_data.py`,
  `configs/skills_superni_1b.json`, `train_skill_lora.py`, `eval_skill.py`, `submit_skills.sh`,
  `analyze_skill_vs_facts.py`, `test_skill_data.py` (CPU tests green incl. a real SuperNI download).
  Metric = per-adapter mean answer-token NLL (`eval_tofu._answer_avg_loss`) under base/isolated/merged;
  facts probe = memorized Q&As (isolates the merge effect), skills probe = 50 held-out instances.
  Eval job 440040 → `reports/skill_nll_{skills,facts}.json`, `reports/facts_vs_skills_retention.csv`.
  A bug caught in the eval smoke: TOFU perturbed splits are only 400 rows / ~40 authors, so the facts
  probe was switched to the `full` split (all authors).
- **Results:** Under the identical merge, **facts uniformly explode** (mean NLL base 2.19 → isolated
  0.93 → merged **13.62**, ppl ≈ 8×10⁵; merged range 13.1–14.1 across all 20 shards) while **skills
  degrade far less and heterogeneously** (5.56 → 1.32 → 8.10, ppl ≈ 3.3k; merged range 1.8–16.2, some
  survive). Normalized retention R=(merged−base)/(isolated−base): skills mean −1.32 vs facts −9.13.
  **Mann-Whitney U = 400.0 / 400 (complete separation), p = 3.4×10⁻⁸** (skills retain more), gap +7.80.
- **Observations:** The regime difference IS the specificity — same merge, facts hit the *explosion*
  regime, skills only the *degradation* regime. Matches Part A: colliding fact deltas superimpose and
  explode; near-orthogonal skill deltas merely average. Complete rank separation (every fact shard
  worse than every skill task). Caveat: the 20-way rslora `linear` merge is over-scaled (both arms
  below base — Part A high-λ regime); the contrast is fair (identical operator, perfect separation)
  but a clean "skills survive" figure needs a non-rslora true-mean merge; NLL (fluent-token-dominated)
  vs a recall/ROUGE metric; single seed.
- **Next Steps:** (1) Rerun the merge at a proper scale — both arms `use_rslora=False` true-mean (needs
  a non-rslora facts training path; train_lora_shard hardcodes rslora=True) — for the clean
  survive-vs-collapse figure. (2) Add a generation/ROUGE recall metric alongside NLL. (3) Multi-seed
  (43,44). (4) Test whether per-shard col(B) overlap (Exp 1) predicts per-shard merge damage.
