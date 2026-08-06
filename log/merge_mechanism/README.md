# merge_mechanism — why merging LoRA adapters destroys factual recall

**Status:** active (started 2026-06-29). APA uniform-summation study (Exp A/B/C) packaged 2026-07-28 as the standalone repo `apa-uniform-sum` — code complete and gated, campaign not yet run. Mechanism study behind the routing-vs-merging result
(see `../../tofu_sisa_lora/reports/ORIENTATION_2026-06-29.md`). Plan:
`~/.claude/plans/sequential-meandering-token.md`.

Goal: turn the observation "merging caps utility ~0.45 while routing-over-isolated-experts reaches
~0.66" into a *mechanism* — facts are sparse, high-magnitude, mutually-colliding weight edits that
interfere under averaging. Four experiments:
- **Exp 1** subspace overlap (weight-space cause) — CPU, done; extended 2026-07-07 to the full
  200 per-author r32 7B family (k=200), see What worked.
- **Exp 2** λ-sweep (no global scale rescues recall) — GPU, done.
- **Exp 3** isolated→merged per-adapter own-author recall drop — GPU, done.
- **Exp 4** facts-vs-skills controlled contrast (specificity) — Part B, not yet built.
- **Exp 5** interference vs N (merge N per-author LoRAs, N∈{1..200}, nested subsets) — **done
  2026-07-08**: recall collapse saturates by N≈8 (~85% of extractable signal gone), mu flat
  0.459±0.002 at every N, col(B) overlap sign-flips to *protection* (ρ=−0.675). See What worked.
- **Exp 6** centered merging (PATHS_FORWARD §6.1, the third regime: shared ~1×, facts ~1×) —
  **done 2026-07-16**. ⚠ Design catch: the literal §6.1 formula ΣΔᵢ−(k−1)Δ̄ is algebraically
  ≡ the mean (Exp-5); two non-degenerate estimators ran instead (pool-mean `cpool`, rank-ρ
  `cr16`). **Result: the first composition rule that moves the interference curve** — the
  collapse knee shifts N≈3 → N\*≈64 (signal at N=8: 71–82% kept vs the mean's 35%), standard
  mu survives at mean-regime levels to N≈128 with retain_ppl FALLING (norm-overshoot channel
  closed), and the residual killer at scale is measured inter-residual crosstalk (recall
  below base by N=200 — worse than dilution at full scale). Does not approach routing.
  [2026-07-16_centered-merge-results.md](2026-07-16_centered-merge-results.md).
- **Exp 7** key-firing / lazy-read-keys measurement (§6.2, the functional half Exp-1 can't see) —
  **done 2026-07-15, same day: keys are LAZY** (gate median on/off ‖sBAh‖ 1.102 e5 / 1.110 e25,
  100% of adapters < 2.0; OOD firing ≈ 90% of on-author). **§6.3 negative anchoring is GO.**
  [2026-07-15_key-firing-results.md](2026-07-15_key-firing-results.md).

Full write-up: `../../tofu_sisa_lora/reports/MERGE_MECHANISM_REPORT_2026-06-29.md`. All three
Part-A predictions confirmed, no falsifier triggered.
**Exp-6/7 + §6.3 from-scratch report (2026-07-16):**
[`CENTERED_ANCHOR_REPORT_2026-07-16.md`](../../tofu_sisa_lora/reports/CENTERED_ANCHOR_REPORT_2026-07-16.md)
— background, methods, provenance, full tables, verdicts, and reproduction commands for
centered merging + key-firing + negative anchoring, readable with no prior repo context.

## What worked
- **7B k=200 merge battery (2026-07-24, jobs 448065/448108) — the P3 row is entirely in-band.**
  ~20 separable operators on the one-author-per-shard pool span **0.419–0.451** (means + full registry:
  regmean/fisher/ties/della ≈ 0.420, tsv/breadcrumbs/lorahub/subtract_orth 0.43–0.45), with sum-compose
  breaking as designed (dare0p99sum → 0.000). No exact separable operator clears 0.55. Per-author
  granularity rescues no merge operator — the empirical closure of the master report's open question.
- Exp 1 (CPU, `subspace_overlap.py`): isolated TOFU fact adapters share a low-rank **output**
  subspace far above chance. Llama-3.2-1B, principal-angle cos on col(B): k4 0.232 / k10 0.262 /
  n32 0.277 (random-orthogonal null ≈ √(r/d) ≈ 0.09). Shared rank-16 basis energy vs chance:
  k4 0.845/0.50, k10 0.650/0.20, n32 0.457/0.031 (ratio grows with n). The **input** subspace
  row(A) stays near-orthogonal (cos ≈ 0.055–0.083). So facts collide in *where they write*.
- Exp 1 at **full author granularity (k=200, Llama-2-7B r32, job 440863)**: the pattern replicates
  and sharpens — pairwise cosine mean 0.00125 (z=19,716 vs null), col(B) 0.164 vs null 0.070 while
  row(A) equals its null to 5 decimals (the collision is 100% output-side); shared r16 energy
  0.231 = 92× chance (ratio trend 1.7×→3.3×→14.6×→92× as n grows while absolute energy falls).
  No block/cluster structure in the 200×200 matrix. **Name-token effect:** pairs sharing an
  author-name token are more similar (0.0014 vs 0.0012, perm p≈5e-4; top pair Yeon Soo↔Yeon Park).
  [2026-07-07_per-author-similarity-k200.md](2026-07-07_per-author-similarity-k200.md).
- **Fig-1(b) analog (2026-07-08, `plot_author_tsne.py`, CPU):** t-SNE of the 200×200 per-author
  delta-cosine matrix = the WMDP "chaos" picture on TOFU — silhouette on the precomputed
  distances ≈ 0 for forget-vs-retain (−0.0000), forget 4-class (−0.0001), and semantic MiniLM
  k-means K=6 (−0.0000); mixing stable across perplexities 5–50. Neither forget membership nor
  semantic category is recoverable from full-delta geometry (the collision structure lives in
  col(B) angles, invisible to full-delta cosine). Figures →
  `../../tofu_sisa_lora/reports/figures/lora_tsne/`.
  [2026-07-08_lora-space-tsne-figure.md](2026-07-08_lora-space-tsne-figure.md).
- Exp 2 (λ-sweep): merged mu peaks ~0.43 at λ≈0.05–0.1 then **collapses to 0 by λ≥0.5** as
  retain_ppl explodes 8→1.8M; **no λ reaches isolated forget_rouge 0.489 or full-FT mu ~0.59**.
  The "no good λ" signature: washout below, colliding-delta explosion above.
- Exp 3 (iso→merged, `--eval_shard_id`): every shard's own-author recall drops isolated→merged;
  mean `dare_ties` forget_rouge drop **0.059 (k=4) → 0.090 (k=10)** — grows with k, 10/10 shards >0.
- Exp 1 on N=20 fact vs skill adapters: facts collide **modestly more** than skills — col(B) cos
  **0.251 vs 0.172**, shared-basis energy **0.414 vs 0.338** (chance 0.05). Mechanism holds
  directionally.
- **Exp 5 (interference vs N, 2026-07-08, 7B r32 per-author, true-scale `additive_mean`):**
  (a) own-author recall collapses FAST then floors — mean iso→merged forget_rouge drop 0.011
  (N=2) → 0.045 (N=4) → **0.073 (N=8)** then flat to 0.076 (N=200) = ~85% of the extractable
  signal (iso 0.4895 − base-floor 0.4038) gone by N=8; probe forget_ppl strictly monotone
  3.6→8.5 (base 14.6). (b) **mu carries no N-dependence at all**: 0.459±0.002 for every
  N∈{1..200} (base 0.426, joint-ft 0.7563) — the 1/N mean is a constant style adapter; the
  prior k-scaling dilution curve was a shard-size effect, not a count effect. (c) additive_mean
  @N=200 (0.4597) > dare_ties convention (0.4198 ≈ base). (d) Materialize-then-eval pipeline
  validated: r8 N=200 dare cross-check 0.4198 vs prior in-model 0.4201; SVD-1024 points accepted
  (|Δmu| 0.0007 at N=64). [2026-07-08_interference-vs-n-results.md](2026-07-08_interference-vs-n-results.md).

## What didn't
- **Data-hygiene trap (2026-07-24):** `../../tofu_sisa_lora/reports/all_metrics_smoke.csv` is a
  **stale pre-`ou-2026-06-10` snapshot** — it lists `merged_dare_ties` 0.1716 vs the live JSON
  0.4236, so it under-reports every merge mu. Also the default-named `checkpoints/Llama-2-7B-chat-hf/`
  dir is a legacy k=10 run (dare_ties 0.4550), NOT the k=4 pool (`_k4_r32_e5_lr1e4`, 0.5445). Cite the
  versioned per-run JSONs until `collect_results.py --smoke` regenerates the CSV.
- **Part B facts-vs-skills specificity did NOT survive a proper merge.** The rsLoRA run's
  U=400/p=3.4e-8 "facts collapse 5×" was an **over-scaling artifact**; with a true-mean (non-rslora)
  merge the NLL gap **vanishes** — and the null is **robust across 3 seeds** (p = 0.68/0.24/0.52;
  facts_R 0.323/0.318/0.317, skills_R 0.264/0.322/0.298). See [2026-07-01_facts-vs-skills-correction.md].
  The clean "merging is specific to facts" claim is NOT established by NLL. (Part A itself is unaffected.)

## Open ideas
- **Decisive test:** generation/ROUGE facts-vs-skills contrast (NLL is fluent-token-dominated and
  understates fact-recall damage; Part A's ROUGE *did* see it). Add a ROUGE mode to eval_skill.
- If ROUGE also shows no specificity: the routing win is about exactness/O(1) deletion, not a
  facts-specific merge failure — reframe accordingly.
- Does the col(B) overlap (Exp 1) predict per-adapter merge damage? → answered by Exp-5 H3:
  yes, but with the OPPOSITE sign (overlap = survival, see resolved ledger).
- Exp-5 follow-ups: does a sign-electing merge (dare/TIES per-author ladder — config toggle,
  schema-ready) move the N≈8 collapse knee? Does the knee shift with rank (r8/r1 on disk)?
  H3′: probe survival ∝ projection of its delta onto the merged delta (CPU-testable from
  existing artifacts). Seed variance 43/44 at N∈{8,32,128} before cross-paper claims.

## Hypotheses — open / resolved
- **[partially resolved — directional ✓, control pending]** H-scaf (2026-07-07 → 07-08): the
  scaffolded `_experts_scaf_k10` show lower null-calibrated overlap than BOTH available plain
  comparators — col(B) ratio 1.78× vs 2.23× (same-recipe 7B) / 4.01× (same-model legacy r8);
  energy 5.97× vs 7.83× chance; cosine z 901 vs 4280 — but the effect is partial (col(B) excess
  +0.104 remains) and each comparison is confounded (no plain 1B k10 r32/e5 exists). Decisive
  control = train that missing collection (10 cheap 1B jobs).
  ([2026-07-08_scaffold-overlap-hscaf.md](2026-07-08_scaffold-overlap-hscaf.md))
- **[open]** H-scaf-control (2026-07-08): plain 1B k10 r32/e5 control lands between scaf (1.78×)
  and legacy 1B (4.01×), clearly above scaf.
- **[open]** H-scaf-dose (2026-07-08): a domain-schema scaffold (synthetic same-format
  biographies) absorbs more of the common component than generic alpaca2k.
- **[open]** H-rank (2026-07-07): capacity starvation (k200 r8/r1 families, on disk) changes
  proportional overlap — direction unknown.
- **[resolved ✓ supported]** HT1 (2026-07-08): full-delta t-SNE of the k=200 cosine matrix shows
  chaos — silhouette ≈ 0 (−0.0001…−0.0000, all < the 0.05 confirm bar) for forget-binary,
  forget-4-class, and semantic-K6 labelings; stable across perplexities 5–50
  ([2026-07-08_lora-space-tsne-figure.md](2026-07-08_lora-space-tsne-figure.md)).
- **[open]** H-tsne-colB (2026-07-08): a t-SNE over PAIRWISE col(B) principal-angle distances
  (the collision-carrying geometry) shows structure the full-delta view hides — needs a
  pairwise-angle matrix dump (subspace_overlap keeps only means; one SLURM CPU pass).
- **[resolved ✓ supported, refined]** H1 (Exp 5): own-author recall decays monotonically with N
  under a true-scale mean — supported in direction but the decay SATURATES: drop 0.011→0.073 by
  N=8 (~85% of extractable signal), then floor to N=200; forget_ppl strictly monotone 3.6→8.5
  ([2026-07-08_interference-vs-n-results.md](2026-07-08_interference-vs-n-results.md)).
- **[resolved ✗ refuted]** H2 (Exp 5): mu vs N is non-monotone — refuted: mu flat 0.459±0.002
  at every N∈{1..200}; no coverage rise (retain_prob 0.234–0.236 flat), no late damage.
- **[resolved ✗ refuted in sign]** H3 (Exp 5): col(B) overlap predicts recall drop with ρ≥0.4 —
  the relationship is real but INVERTED: ρ = −0.675 (p<1e-4, 36 probe×N points). Alignment with
  the shared output subspace protects an adapter's writes under averaging; idiosyncratic
  directions dilute away. Single-seed, pooled-N caveats.
- **[resolved ✓ confirmed, earlier than predicted]** H4 (Exp 5b, subset-conditioned utility):
  retain_* restricted to the merged authors decays with knee N≈2–4 — half the extractable
  signal (above the base floor 0.170) gone by **N=3**, two-thirds by N=8, population-level
  plateau from N≈12 ([2026-07-08_subset-utility-results.md](2026-07-08_subset-utility-results.md)).
- **[resolved ✓ confirmed]** H5 (Exp 5b): isolation dominates — iso adapter retain_prob 0.3991
  on its own author vs joint-ft 0.9237 on the same rows (0.43×; isolation forfeits ~70% of the
  ft-range signal before any merging). **REINTERPRETED by H7 (07-09): the cost is undertraining,
  not isolation per se.**
- **[resolved ✗ refuted]** H6 (07-09): own-author recall is rank-independent at fixed 5 steps —
  refuted: r1/α2 0.183 ≈ base floor (learned nothing), r8 0.237, r32 0.399. Smaller LoRAs are
  worse, not "less overkill".
- **[resolved ✓ confirmed, saturated]** H7 (07-09): the frozen recipe gives a 20-row shard only
  ~5 optimizer steps; at **25 steps (e25) the same r32 adapter reaches 0.9991 prob / 1.0 rouge /
  ppl 1.06 on its author — above the joint-ft ceiling 0.9237** (e50 identical). Collateral cost
  modest (world_prob 0.72→0.60). The Exp-5b "isolation cost" was a step-count artifact.
  ([2026-07-09_iso-rank-epochs-results.md](2026-07-09_iso-rank-epochs-results.md))
- **[resolved ✓ confirmed]** H8 (07-09): the collapse survives well-trained experts — e25 ladder
  0.999 (N=1) → 0.885 (N=2) → **0.615 (N=3, the same 50% knee)** → 0.282 (N=8) → same plateau
  ≈0.21 as the e5 curve by N≈12 (curves cross at N≈12–16); global mu of e25 merges is LOWER
  (0.40–0.44 vs 0.459) with retain_ppl 11.4 at N=2 (norm-overshoot in miniature). Interference
  is a training-independent multiplicative attenuation; only the N≤3 micro-merge zone improves.
  ([2026-07-09_h8-e25-ladder-results.md](2026-07-09_h8-e25-ladder-results.md))
- **[resolved — inconclusive by the bars, directionally supported]** H-cent-1 (07-16, Exp 6):
  N=8 subset retain_prob 0.358 (cpool) / 0.334 (cr16) vs mean 0.250 — the ≥2× bar (0.50)
  NOT met, but the lift is real and consistent (signal kept: 71–82% vs 35%).
  ([2026-07-16_centered-merge-results.md](2026-07-16_centered-merge-results.md))
- **[resolved ✓ supported to N≈128, ✗ at N=200]** H-cent-2 (07-16): cr16 mu 0.459–0.471
  (N≤32), 0.460 (N=64), 0.442 (N=128) — at/above the mean regime with retain_ppl FALLING
  7.7→6.0 (no overshoot); at N=200 mu 0.412 < base. cpool survives only to N≈16–20
  (estimator self-contamination + svd-energy loss, both pre-flagged).
- **[resolved ✓ supported, measured]** H-cent-3 (07-16): half-signal **N\* ≈ 64**, crossover
  below the mean plateau at N≈150, recall pushed below base by N=200 (drop 0.120 >
  extractable 0.086) — with blowup and dilution both removed, inter-residual crosstalk is
  the isolated, measured killer, and it ends worse than dilution at full scale.
- **[resolved — between the bars]** H-cent-4 (07-16): drop-vs-overlap r −0.40 (cr16) /
  −0.44 (cpool) vs the mean regime's −0.675 — the H3 protection is weakened ~40%, not
  eliminated (the 1× shared term still protects).
- **[resolved ✓ supported, decisively]** H-key-1 (07-15, Exp 7): keys are LAZY — gate median
  on/off ‖sBAh‖ ratio **1.102** (e5) / **1.110** (e25), **100% of adapters < 2.0** in both
  sets, none ≥ 5; adapters fire on OOD Alpaca/world-facts at ~90% of on-author magnitude.
  **§6.3 negative anchoring is GO** per the pre-registered gate.
  ([2026-07-15_key-firing-results.md](2026-07-15_key-firing-results.md))
- **[resolved ✓ null-direction]** H-key-2 (07-15, Exp 7): read keys carry ~zero discrimination
  (‖Ah‖ ratio 1.012/1.044); the residual 1.10–1.15 lives in the composed write norm — neither
  side is selective.
- **[resolved ✗ refuted]** H-key-3 (07-15, Exp 7): training dose does NOT sharpen keys — e25
  selectivity ≈ e5 (1.110 vs 1.102) while absolute firing grows ~3.4× (on 0.184→0.633). The
  firing-side mechanism for H8: stronger experts collide harder because unselective outputs
  scale with training.
- **[resolved ✗ refuted, decisively]** H-anchor-1 (07-16): the anchor penalty produces
  UNIFORM shrinkage, never selectivity — gate ratio stays 1.11–1.15 (≈ baseline 1.11) at
  every λ ∈ {1,10,100} while on-firing drops 0.63→0.06 and recall decays 0.997→0.525.
  Mechanism: Exp-7 showed the read inputs are indistinguishable on/off-author, so no
  output-norm objective can separate them — self-gating cannot be trained into a LoRA;
  selection must live outside the adapter (router/mask) or in content-derived keys (Path C).
  Bonus: the penalty is a collateral-damage dial (single-adapter mu 0.388→0.462, world_prob
  →0.75) — λ=1 keeps recall at 0.997 with less damage.
  ([2026-07-16_negative-anchor-pilot-results.md](2026-07-16_negative-anchor-pilot-results.md))
- **[closed — not run, per pre-registration]** H-anchor-2: the anchored H8 ladder required a
  λ with selectivity ≥5 AND recall ≥0.98; no arm qualified. §6.3 closed as a causal negative.

## Entries
- [2026-06-29_subspace-overlap.md](2026-06-29_subspace-overlap.md) — Exp 1 done; Exp 2/3 harness + smoke.
- [2026-06-29_lambda-iso-results.md](2026-06-29_lambda-iso-results.md) — Exp 2/3 results; all predictions confirmed.
- [2026-07-01_facts-vs-skills.md](2026-07-01_facts-vs-skills.md) — Part B (rsLoRA): apparent facts specificity U=400 — later found to be an artifact.
- [2026-07-01_facts-vs-skills-correction.md](2026-07-01_facts-vs-skills-correction.md) — CORRECTION: true-mean merge → gap vanishes (p=0.68); facts collide only modestly more; NLL specificity not established.
- [2026-07-07_interference-vs-n-design.md](2026-07-07_interference-vs-n-design.md) — Exp-5 pre-registration: N-merge interference ladder (per-author, nested subsets, additive_mean); H1–H3 open.
- [2026-07-08_interference-vs-n-results.md](2026-07-08_interference-vs-n-results.md) — Exp-5 results: recall collapse saturates by N≈8 (~85% signal gone); mu flat 0.459±0.002 ∀N (H2 refuted — the dilution curve was shard-size, not count); H3 sign-flips (overlap = survival, ρ=−0.675); pipeline + SVD-1024 validated.
- [2026-07-08_subset-utility-design.md](2026-07-08_subset-utility-design.md) — Exp-5b pre-registration: `--retain_author_ids` subset-conditioned utility, dense N=1–20 ladder; H4/H5.
- [2026-07-08_subset-utility-results.md](2026-07-08_subset-utility-results.md) — Exp-5b results: H4✓ knee N≈2–4 (half the learned-subset signal gone by N=3, plateau by N≈12); H5✓ isolation itself costs 70% (iso 0.399 vs joint-ft 0.924 on the same rows); the unrestricted retain metric ≈ merged plateau, confirming mu is blind by construction.
- [2026-07-08_iso-rank-epochs-design.md](2026-07-08_iso-rank-epochs-design.md) — pre-registration H6 (rank) / H7 (undertraining) on the weak-iso puzzle.
- [2026-07-09_iso-rank-epochs-results.md](2026-07-09_iso-rank-epochs-results.md) — H6✗ (r1 learns nothing, recall rises with rank); **H7✓ saturated: 25 optimizer steps → 0.9991 prob / 1.0 rouge, above the joint-ft ceiling** — the "isolation cost" was the frozen recipe's ~5 steps on a 20-row shard. New H8: rebuild the N-ladder from e25 experts.
- [2026-07-09_h8-e25-ladder-design.md](2026-07-09_h8-e25-ladder-design.md) — H8 pre-registration (e25 ladder, H8a/H8b).
- [2026-07-09_h8-e25-ladder-results.md](2026-07-09_h8-e25-ladder-results.md) — H8✓: same knee (N≈3), same plateau, steeper fall, MORE collateral (mu 0.40 vs 0.46 at N=2); collapse unconfounded from expert quality; only the N≤3 micro-merge zone improves (0.885 at N=2). fig6.
- [2026-07-07_per-author-similarity-k200.md](2026-07-07_per-author-similarity-k200.md) — Exp 1 at k=200 (7B r32, job 440863): H1/H2/H3 all supported; collision 100% output-side; energy 0.231=92× chance; name-token effect p≈5e-4; full 200×200 matrix saved for Exp-5 H3.
- [2026-07-08_scaffold-overlap-hscaf.md](2026-07-08_scaffold-overlap-hscaf.md) — H-scaf triangulation: scaffolded experts show lower calibrated overlap than both plain comparators (col(B) 1.78× vs 2.23×/4.01×; energy 5.97× vs 7.83×) but the shared component only shrinks, doesn't vanish; directional ✓, plain-1B-r32 control pending; row(A)-at-null now replicated across 3 bases.
- [2026-07-08_lora-space-tsne-figure.md](2026-07-08_lora-space-tsne-figure.md) — HydraLoRA-Fig-1(b) analog on TOFU (`plot_author_tsne.py`, CPU post-processing of the 07-07 k=200 matrix): HT1 chaos ✓ — silhouette ≈ 0 for forget/semantic labelings, mixing stable across perplexities 5–50; figure-ready motivation for structural separation; H-tsne-colB opened.
- [2026-07-15_centered-merge-design.md](2026-07-15_centered-merge-design.md) — Exp-6 pre-registration: centered merging. **Degeneracy catch recorded:** the literal §6.1 ΣΔᵢ−(k−1)Δ̄ ≡ the mean (rejected); runs pool-mean (`cpool`, N≤64) + rank-ρ (`cr16`, full ladder) estimators on the e5 pool; H-cent-1..4. Merges job 443445; CPU gate green (degeneracy proven as a test).
- [2026-07-15_key-firing-design.md](2026-07-15_key-firing-design.md) — Exp-7 pre-registration: functional key-firing selectivity (`measure_key_firing.py`, Gram-trick ‖sBAh‖); LAZY/SELECTIVE gate for §6.3 negative anchoring; H-key-1..3. e5 job 443446; e25 job 443477.
- [2026-07-15_key-firing-results.md](2026-07-15_key-firing-results.md) — Exp-7 results, same day: **keys are LAZY** (median 1.102/1.110, 100% of adapters < 2.0; read side ≈ 1.01; OOD firing ≈ 90% of on-author) → **§6.3 GO**; H-key-3 refuted (e25 fires 3.4× harder, no sharper — the firing-side mechanism for H8); H-anchor-1/2 opened.
- [2026-07-15_negative-anchor-design.md](2026-07-15_negative-anchor-design.md) — §6.3 pre-registration: `AnchoredSFTTrainer` (‖sBAh‖² penalty on 2k seeded Alpaca, exactness preserved; flag-free bit-identical, CPU gate green), λ-pilot {1,10,100} × 5 probe authors at the e25 recipe (train array 443487), pre-registered λ-selection rule (selectivity ≥5 AND recall ≥0.98); H-anchor-1/2.
- [2026-07-16_negative-anchor-pilot-results.md](2026-07-16_negative-anchor-pilot-results.md) — **H-anchor-1 REFUTED across all λ:** uniform shrinkage (on-firing 0.63→0.06), zero selectivity gain (1.11–1.15 ∀λ), recall 0.997→0.525; H-anchor-2 not run per the decision tree. Self-gating cannot be trained into a LoRA — selection must live outside the weights or in content-derived keys. Bonus: anchoring = a collateral-damage dial (single-adapter mu 0.388→0.462).
- [2026-07-16_centered-merge-results.md](2026-07-16_centered-merge-results.md) — Exp-6 results: **the knee moves N≈3 → N\*≈64; utility survives to N≈128** (cr16 mu 0.44–0.47, retain_ppl falling — overshoot channel closed); crosstalk isolated and measured (below-base recall by N=200); H-cent-1 inconclusive-by-bar (lift +34–43%, not 2×), H-cent-2 ✓→N≈128, H-cent-3 ✓ measured, H-cent-4 between bars (−0.40 vs −0.675). Deviations: cr16 exact-N64 OOM → svd-served; cap incident re-run; CSV prefix clobber fixed (`OUT_PREFIX`).
- [2026-07-18_gapfill-preregistration.md](2026-07-18_gapfill-preregistration.md) — Gap-fill table-closers @k=200 pre-registered (framed against the user's merge-method reference doc, NOT a new bet): standalone RegMean/Fisher/TIES/KnOTS on the 7B r8 k=200 pool + breadcrumbs `_s{λ}` rescue (H-gf-1..4, band = merge ceiling 0.42–0.50). Code: `eval_tofu --merge_num_examples` (k=200 runs at 32/shard, recorded deviation) + `merged_breadcrumbs_s{λ}` scale-suffix fix (was parsed-but-ignored; new CPU regression green). DARE+sum ops deferred behind ctv [w5] job 445329 (sparsify_pool.py frozen while in use). Jobs pending GPU-cap headroom.
- [2026-07-24_merge-ceiling-vs-routing-master.md](2026-07-24_merge-ceiling-vs-routing-master.md) — Consolidation + verification (no GPU): assembled the thesis-first master report [`MERGE_VS_ROUTING_MASTER_2026-07-24.md`](../../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md) — banded grand-master merge table + router-vs-merger. **H-MASTER SUPPORTED at report scale** (19 separable operators in 0.41–0.50; break-downward rows 0.00–0.09; only serve-time selection clears 0.55 — SIFT 0.737 / ClAMU 0.672 / routing 0.556–0.824). Spot-verified headline cells vs the live `ou-2026-06-10` JSONs (P1 dare_ties 0.4236, P2-k4 lorahub 0.5921 / dare_ties 0.5445, dilution 0.5445→0.4201, sift_full 0.737 / merge_full 0.4073, ClAMU Global 0.3511 / K200 0.6716, routed_oracle 0.8236).
- [2026-07-26_k200-battery-into-master.md](2026-07-26_k200-battery-into-master.md) — 7B k=200 merge battery folded into the master (new Table C′). **H-REPRO ✓** — a user-pasted A100 recompute matches this machine's on-disk A40 battery to within ~0.002 mu (max |Δ| TSV 0.0022; ft anchor differs 0.742 vs 0.7563 — retrained pool). **H-K200-BAND ✓** — ~20 operators at one adapter/author all in 0.42–0.46 (sign-vote family ≈0.420=base; linear family 0.44–0.45; KnOTS OOM, JD SVD-fail). Nuance: PEFT-linear is in-band at r8/k200 (0.4503), degenerate only at r32 — √r double-count is rank-dependent. Source `../../merge-tables-7b/RESULTS_7B_K200.md`.
- [2026-07-28_apa-uniform-sum-setup.md](2026-07-28_apa-uniform-sum-setup.md) — APA uniform-summation study (Δ = Σᵢ sᵢBᵢAᵢ, coefficient 1.0, no router): instrumentation + four premise measurements. **Per-author deltas are near-orthogonal** (mean |cos| 0.0009–0.0051), so ‖Σ_N‖ grows as **√N not N** (4.54 vs √20=4.47) — the earlier `dare0p9sum` collapse (mu→0 by N=16) was DARE's 1/(1−p) rescaling, not the sum rule, so the ladder was extended past N=20 before spending GPU. **Perturbed-split coverage is all-or-nothing** (20 rows for authors 0–19 and 180–199, *zero* for 20–179) — corrects the CLAUDE.md "~2 rows/author" invariant and makes 180–199 the only legal Exp-B targets. `holdout10` verified 0/400 overlap with `full`. Two silent-corruption traps closed: `analyze_nmerge` dropped every `nmerge_sum_*` row with no error, and the cached KS reference is author-199-only. Ladder queued (449203/449218).
- [2026-07-28_apa-repo-carveout.md](2026-07-28_apa-repo-carveout.md) — Engineering: the APA study packaged as a standalone repo (`apa-uniform-sum`), carved out through an explicit allow-list. **H-REPO ✓** — 10/10 CPU gates pass from a fresh clone with no sibling trees, no `/storage2` and no SLURM; **H-PORT ✓** — one driver renders `sprint` / `cispa` / `local` correctly under `STUB=1`, including CISPA's no-`--mem` policy. Five silent defects found by building the gates: `eval_mmlu` resolved `legonet_lora` as `dirname(dirname(__file__))` (which EXISTS here, so only a clone would fail); `submit_nmerge` read config paths without `expandvars`, so the portable `${TOFU_CKPT_ROOT}` form yielded a literal path; its merge/eval/overlap stages still hand-wrote `--partition=all`+`--mem=` (3 of 6 stages would fail at submit on CISPA); six modules defaulted `HF_HOME` to another cluster's disk; and the first `submit_expb` design gave all five targets the same sum4 companions, making five `drop` merges one artifact under five labels. Two chart defects caught by rendering: the palette cycled past 3 slots (two authors painted the same blue) and coinciding series overprinted their labels — both now structurally impossible. New gates: `test_repo_selfcontained`, `test_expb_selectivity`, `test_mmlu_primitives`, `test_plot_style`. No GPU used; the campaign itself is unchanged and unstarted.
- [2026-07-24_7b-k200-gapfill-results.md](2026-07-24_7b-k200-gapfill-results.md) — **Empirical closure of the master entry's open P3 question: H-P3 SUPPORTED.** Ran the 7B k=200 (one-author-per-shard) merge battery — M(a) w5 sparsify (r32, job 448065) + M(b) 11 registry operators (r8, job 448108). ~20 separable operators all land 0.419–0.451 (means/registry); sum-compose breaks as designed (dare0p99sum → 0.000); no exact operator clears 0.55. regmean 0.4197 / fisher 0.4200 / ties 0.4201 / della 0.4193 / tsv 0.4366 / lorahub 0.4505 / breadcrumbs 0.4338–0.4447 / subtract_orth 0.4272; knots_ties OOM×2 (>44.5 GiB SVD, 1B ref 0.424). Per-author granularity rescues nothing. Deferred: JD (k100-coupled driver), Phase P/C (peft/ctv drivers hardcode the 1B pool). Two cap incidents + a dup-submission caught/fixed; idempotency guard added.
