# Facts vs skills: is the merge-collapse specific to factual knowledge? (Part B)
**Date:** 2026-07-01 · **Model:** Llama-3.2-1B-Instruct · **Seed:** 42 · single seed · smoke-scale
Completes the mechanism study in `reports/MERGE_MECHANISM_REPORT_2026-06-29.md`. Plan:
`~/.claude/plans/sequential-meandering-token.md`.

> ⚠️ **CORRECTED 2026-07-01 (see §CORRECTION).** The dramatic "facts collapse 5× more, p=3.4×10⁻⁸"
> headline below was a **merge-scaling artifact** of the rsLoRA √r-inflated merge. Under a *proper
> true-mean* (non-rsLoRA) merge the NLL facts-vs-skills gap **vanishes (p=0.68)**. Facts *do* collide
> modestly more in weight space, but NLL does not show them as uniquely fragile. Read §CORRECTION as
> the standing conclusion; the rsLoRA section is kept as the artifact record.

**Bottom line (rsLoRA run — ARTIFACT, superseded):** Under the rsLoRA 20-way merge, fact adapters
appeared to collapse catastrophically vs skills (U = 400/400, p = 3.4×10⁻⁸). This turned out to be
driven by the √r merge over-scaling (the merge is in the explosion regime for both arms), not a clean
facts-are-fragile effect — see §CORRECTION.

---

## Setup (controlled contrast; only the domain differs)
- **Both arms, identical recipe:** Llama-3.2-1B, LoRA r16/α32, rslora, [q,k,v,o,up,down], e5/lr1e-4,
  200 train samples/adapter, N=20 adapters, merge = `add_weighted_adapter(linear, w=1/N)` (the
  LegoNet/`merged_linear` combine).
- **Facts:** N=20 balanced TOFU shards (`train_lora_shard.py --k 20`, 10 authors × 20 Q&As each),
  `checkpoints/Llama-3.2-1B-Instruct_facts_n20/`. Probe = the memorized Q&As (full split) — the
  adapter recalls these ~perfectly, so any merged degradation is unambiguously merge interference.
- **Skills:** 20 distinct SuperNI tasks (`Muennighoff/natural-instructions`, input-only so the adapter
  must encode the skill), `checkpoints/Llama-3.2-1B-Instruct_skills_n20/`. Probe = 50 held-out
  instances of the same task.
- **Metric:** per-adapter mean answer-token NLL (`eval_tofu._answer_avg_loss`) under base (no adapter),
  isolated (adapter j), merged (all N averaged). Code: `skill_data.py`, `train_skill_lora.py`,
  `eval_skill.py`, `analyze_skill_vs_facts.py`. CSVs: `reports/skill_nll_{skills,facts}.json`,
  `reports/facts_vs_skills_retention.csv`.

## Result (mean over N=20 each; NLL, perplexity = e^NLL)
| domain | base NLL (ppl) | isolated NLL (ppl) | merged NLL (ppl) | merged−base | merged range |
|---|---|---|---|---|---|
| **facts** | 2.19 (≈9) | 0.93 (≈2.5) | **13.62 (≈8×10⁵)** | **+11.44** | 13.1–14.1 (tight) |
| **skills** | 5.56 (≈260) | 1.32 (≈3.7) | 8.10 (≈3.3k) | +2.54 | 1.8–16.2 (wide) |

- **Facts uniformly explode:** merged NLL 13.1–14.1 across all 20 shards → the merged model outputs
  garbage on every author, ppl ~10⁵. Isolated recalled them (ppl 2.5).
- **Skills degrade far less and heterogeneously:** merged 1.8–16.2 — several skills stay near their
  isolated value (survive the merge), others degrade; none explode like facts.
- **Contrast:** normalized retention R = (merged−base)/(isolated−base): skills mean −1.32 vs facts
  mean −9.13; **Mann-Whitney U = 400.0, p = 3.4×10⁻⁸** (skills > facts), gap +7.80. U=400 is the
  maximum for 20×20 → *complete* separation, zero overlap.

## Reading
The regime difference IS the specificity: **facts hit the explosion regime, skills only the
degradation regime, under the same merge.** It matches Part A end-to-end — Exp 1 showed fact deltas
share a low-rank output subspace (col(B) overlap ≫ null) while skills would not; Exp 2 showed
over-scaled sums explode; here the N=20 merge superimposes the colliding fact deltas and explodes,
but merely averages the near-orthogonal skill deltas. **"Route what you must delete (facts), merge
only what is safe to merge (skills)."**

## Caveats (honest; clean follow-ups)
1. **Over-scaled merge regime.** The 20-way `linear` merge under rslora is over-scaled (√r factor-space
   inflation), so *both* arms fall below base — same regime as Part A's high-λ. The **contrast is fair**
   (identical operator both sides, and the separation is perfect), but a clean "skills survive, facts
   collapse" figure needs a **properly-scaled / non-rslora (true-mean) merge** — the natural next run
   (both arms retrained `use_rslora=False`, mirroring the LegoNet arm).
2. **NLL vs recall.** Answer-token NLL is dominated by fluent tokens; a generation/ROUGE recall metric
   (as in Part A) is a more direct "does it still work" probe — a cheap add.
3. **Probe asymmetry:** facts on memorized Q&As (isolates the merge effect), skills on held-out
   instances (generalization). Each is the natural capability for its domain.
4. Single seed 42; 1B; smoke-scale. Multi-seed (43,44) + a larger model would harden it.

## CORRECTION — non-rsLoRA true-mean merge (the standing result)
The rsLoRA `linear` 20-way merge is √r-inflated → over-scaled → *both* arms fall below base
(explosion regime). To test whether the facts-vs-skills gap is real or an artifact, both arms were
**retrained `use_rslora=False`** (LegoNet-proven r16/α32/6ep/lr2e-4; only the merge inflation
changes) so `add_weighted_adapter(linear, 1/N)` is a **true mean**. Result:

| (true-mean) | base NLL | isolated | merged | retention R (mean-of-ratios) |
|---|---|---|---|---|
| facts  | 2.19 | 0.96 | 1.79 | **0.323** |
| skills | 5.56 | 1.34 | 5.16 | **0.264** |

**Mann-Whitney U = 183, p = 0.68 — NO significant difference** (facts retain *slightly more*, if
anything). The dramatic rsLoRA separation was an **over-scaling artifact**: the √r-inflated merge
pushed the high-magnitude fact deltas into explosion; a proper mean does not.

**Robust across 3 seeds** (`reports/facts_vs_skills_retention_nr[,_s43,_s44].csv`): facts_R
0.323/0.318/0.317, skills_R 0.264/0.322/0.298, p = **0.68 / 0.24 / 0.52** — every seed null; no seed
shows skills>facts (seed 43 has skills slightly above facts). The null is not a one-seed fluke.

**Weight-space check (Exp 1 on the N=20 sets, `reports/subspace_overlap_{facts,skills}_n20.*`):** facts
*do* collide more than skills — col(B) principal-angle cos **0.251 vs 0.172**, shared rank-16 basis
energy **0.414 vs 0.338** (chance 0.05), whole-delta cosine 0.010 vs 0.002. So the mechanism holds
**directionally but modestly** — enough to be real, not enough to make merging destroy facts while
sparing skills at the NLL level.

**Why NLL may miss it:** answer-token NLL is dominated by fluent tokens, and the base LM already
scores TOFU answers low (base 2.19). A merged model can emit a *fluent-but-wrong* answer (right
format, wrong name) at modest NLL while having zero fact recall. Part A used generation ROUGE
(`forget_rouge`) and *did* see clear merge damage on facts. So the **recall-based** facts-vs-skills
contrast (ROUGE on generations, not NLL) is the proper unresolved test — **not yet run**.

## Verdict (honest, corrected)
The clean "**merging is specific to facts**" claim is **NOT established** by NLL under a proper merge —
the striking rsLoRA result was an artifact (p 3.4e-8 → 0.68 once the merge is a true mean). What holds:
(1) Part A stands on its own — on TOFU facts, adapters collide in a shared output subspace and merging
destroys *recall* (ROUGE), no λ rescues it. (2) Facts collide **modestly more** than skills in weight
space (0.25 vs 0.17). (3) Whether that translates to facts being **more recall-fragile** under a proper
merge is **open** — it needs the generation/ROUGE contrast. The "facts → route, skills → merge"
principle is *plausible and directionally supported*, but should be stated as a hypothesis pending the
recall metric, not as the p=3.4e-8 result.

