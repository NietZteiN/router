### Target Date: 2026-07-01 (Part B CORRECTION: the specificity was a merge-scaling artifact)
- **Goal / Hypothesis:** Correct [2026-07-01_facts-vs-skills.md](2026-07-01_facts-vs-skills.md). That
  entry reported facts collapsing 5× more than skills under a 20-way merge (U=400, p=3.4e-8). But the
  rsLoRA `linear` merge is √r-inflated (over-scaled → explosion regime for both arms), so the result
  could be a scaling artifact rather than a real facts-are-fragile effect. Test: rerun both arms with a
  TRUE-mean merge; separately measure whether fact adapters actually collide more than skill adapters.
- **Setup:** Added `train_lora_shard.py --no_rslora` (backward-compatible). Retrained both arms
  `use_rslora=False` (LegoNet-proven r16/α32/6ep/lr2e-4 — only the merge inflation changes):
  `checkpoints/..._facts_n20_nr` (job 440041), `..._skills_n20_nr` (440042); eval job 440081 →
  `reports/skill_nll_{skills,facts}_nr.json`, `reports/facts_vs_skills_retention_nr.csv`. Also ran Exp-1
  subspace overlap on the N=20 fact vs skill adapter sets (`reports/subspace_overlap_{facts,skills}_n20.*`).
- **Results:** **Non-rsLoRA true-mean merge — the gap VANISHES.** facts: base 2.19→iso 0.96→merged 1.79,
  R=0.323; skills: base 5.56→iso 1.34→merged 5.16, R=0.264. **Mann-Whitney U=183, p=0.68** (facts
  retain *slightly more*, not less). So the rsLoRA U=400/p=3.4e-8 was an **over-scaling artifact** (the
  √r-inflated merge exploded on the high-magnitude fact deltas). **Weight-space (Exp 1, N=20):** facts
  DO collide more than skills — col(B) cos **0.251 vs 0.172**, shared-basis energy **0.414 vs 0.338**
  (chance 0.05), whole-delta cosine 0.010 vs 0.002 — but modestly.
- **Observations:** The mechanism (facts collide more) is real but MODEST; it does not, under a proper
  merge, make facts uniquely fragile at the NLL level. Likely cause: answer-token NLL is dominated by
  fluent tokens and the base LM already scores TOFU answers low (2.19), so a merged model can emit a
  fluent-but-wrong answer at modest NLL while losing the actual fact — NLL understates recall damage.
  Part A used generation ROUGE and saw clear fact-recall damage. **The clean "merging is specific to
  facts" claim is NOT established by NLL; state it as a hypothesis pending a recall metric.** Part A
  itself is unaffected (facts collide + merge destroys recall + no λ rescues — all still hold).
- **Next Steps:** (1) The decisive test: **generation/ROUGE facts-vs-skills contrast** (add a ROUGE mode
  to eval_skill; merged fact-recall vs merged skill-performance) — this is what NLL can't see. (2) If
  ROUGE also shows no specificity, the honest conclusion is "merging hurts facts and skills similarly;
  the routing win is about exactness/O(1) deletion, not a facts-specific merge failure." (3) Multi-seed.
