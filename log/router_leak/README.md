# router_leak — sealing (and pricing) the post-deletion router leak

**Status:** REOPENED — Phase 3 "all-router sweep" running since 2026-07-20 (jobs
446563–446568): the leak metrics generalized to EVERY router family (lexical, TF-IDF,
LM-embed, behavioral, trained RouterLoRA ×3 seeds, DBpedia retriever, new encoders, k=200
granularity) + the routerless SepMLP spec-v2 arm ([`log/sepmlp/`](../sepmlp/README.md)).
Phases 0–2 complete + reported:
[`ROUTER_LEAK_REPORT_2026-07-18.md`](../../tofu_sisa_lora/reports/ROUTER_LEAK_REPORT_2026-07-18.md).
**2026-07-23:** Wave-0 CPU consolidation done — the [routing master reference](../../tofu_sisa_lora/reports/ROUTING_MASTER_2026-07-23.md)
(methods + extended Table-7 strategy inventory + full per-strategy orphan battery) + a new
per-author landing-determinism metric. **Wave 1 resolved analytically (no GPU):** behavioral
routers (ppl/activation_norm/…) have no in-space identity sentinel — it would require keeping the
deleted expert — so their only seal is the router-agnostic external MiniLM author-gate (0.963/0.002).
GPU Waves 2–3 (sibling-content ROUGE per strategy / Mode-B ρ per family) **LANDED** (`submit_router_wave23.sh`):
**Wave 2** — the leak is confabulation not disclosure across ALL 9 servable routers (sibling-vs-gold
0.265–0.291 ≈ floor 0.249, confab 0.94–0.98; magnets reproduced). **Wave 3** — unsealed verbatim
Mode-B ρ@R8 severe for centroid_sbert 0.972 / centroid_lm 0.929 / ppl 1.00 (monotone in R); ppl worst
(routes to the fact-holding expert by R2); activation_norm ρ degenerate (magnet ceiling≈floor).
**2026-07-26:** the orphan battery is no longer 1B-only — 7B k=10 (feature + behavioral), 7B k=50
and the previously-unaudited **plain 1B k=10** de-confound arm landed (jobs 448590–448594).
**Routers that never read the model are bit-identical across 1B-scaf / 1B-plain / 7B-plain**;
**every expert-reading router moves its magnet with scale**; and `centroid_lm` — the worst leaker
(adequacy 0.976–0.999, self-detect AUC 0.728) — keeps magnet **s4 in all three arms**, so the leak
is structural rather than a property of one checkpoint
([entry](2026-07-26_7b-orphan-coverage.md)).
· **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 15 · **Plans:**
`~/.claude/plans/can-you-expand-on-moonlit-kahan.md` (Phases 0–2),
`~/.claude/plans/include-these-details-as-mellow-barto.md` (Phase 3, approved 2026-07-20)

**Phase-3 open hypotheses:** H-ARCH (drop-leak + confidence-inseparability hold for ≥7/9
score-based router families) · H-DIAL (k=10 multi-drop monotonicity, csq sim-ratio ≥0.95)
· H-POOL (per-author granularity does NOT create separability) · H-TRAINED (RouterLoRA
alpha renormalizes leak-blind, 3 seeds) · H-DATASET (DBpedia retriever reproduces the
overlap) · H-ENC (mpnet/bge reproduce sim-ratio ≥0.95, AUC ≤0.75) · H-SEAL-GEN
(author-sentinel tombstone works in TF-IDF/SBERT/LM feature spaces) — bars in the
[Phase-3 pre-registration](2026-07-20_all-router-sweep-preregistration.md).

The deep-dive campaign on [PATHS_FORWARD §7.1](../PATHS_FORWARD_2026-07-13.md): after exact
expert deletion, a realistic embedding router serves the deleted authors' lookalike queries from
surviving **sibling** experts (n=32 priors: sim-ratio 0.980, 72.7% retain shift —
[ramole §9-D](../ramole/2026-07-06_routing-audit-results.md)), a global confidence threshold
provably can't seal it ([07-07 refutation](../ramole/2026-07-07_routing-fix-arms.md)), and the
same weights leak Mode-B replicated facts through the embed route (ρ→0.833 —
[entangled_facts](../entangled_facts/2026-07-07_embed-route-surface.md)). This thread tests the
**identity-seal (tombstone) family** across a provenance ladder (per-expert key → per-author
sentinel centroid → name embedding → lexical registry), audits **what the leak actually says**
(disclosure vs confabulation vs cross-author disclosure), prices the seal (retain-FPR + the
deletion-disclosure/Streisand AUC), and re-tests λ=1 anchoring as a *content* attenuator
(selectivity is closed — [07-16 ✗](../merge_mechanism/2026-07-16_negative-anchor-pilot-results.md)).
fq is never a leak bar here (the inversion is a pre-registered side-prediction).

## Hypotheses — open / resolved
- **[resolved ✓ SUPPORTED, strengthened]** H1-7B feature-router scale-invariance (2026-07-26) —
  `centroid_lm` 7B n_eff **2.5** vs 1B 2.1 (bar ±0.5), same magnet **s4** across 1B-scaf/1B-plain/
  7B-plain, adequacy 0.997–0.999. Stronger than the bar: the four routers that never read the LLM
  (`key_exact`/`key_tfidf`/`centroid_sbert`/`centroid_sbert_q`) are **bit-identical** across all
  three arms — their magnet is a property of the TOFU author-embedding geometry, not of any model
  ([7b-orphan-coverage](2026-07-26_7b-orphan-coverage.md)).
- **[resolved ✓ SUPPORTED, generalized]** H2-7B behavioral routers track model scale — on the clean
  1B-plain vs 7B-plain contrast `activation_norm` moves s6→**s7** *and* n_eff 1.1→**2.4** (both
  halves; ±0.3 refutation band far exceeded). Generalizes: **all four** behavioral routers move
  their magnet (`ppl` s7→s1, `attn_norm` s5→s4, `logit_div` s8→s7), as does `centroid_lm_last`
  (s4→s2) (same entry).
- **[resolved ✗ REFUTED — but as a metric artifact]** H3-7B granularity monotonicity — `centroid_lm`
  n_eff 2.5 → 7.3 → 1.9 over k=10/50/200 inverts at k=50, as do `key_tfidf` and `centroid_sbert`.
  Cause: "drop one shard" changes the orphan count 400→80→20 and survivors 9→49→199, and
  `n_eff ≤ min(orphans, survivors)`. **Adequacy is the scale-free quantity and IS monotone**
  (`key_tfidf` 0.667→0.385→0.194, `centroid_sbert` 0.967→0.884→0.705) — finer units make a wrong
  match a genuinely worse match, which is why fine-grained pools self-detect (same entry).
- **[resolved ✓ CONFIRMED]** H4-7B dilution is merge-specific (2026-07-26) — on the SAME 7B pools the
  merged ladder (`dare_ties`) decays monotonically 0.545 → 0.420 over k=4…200 while the routed ladder
  (`key_exact`) is **flat within ±0.04** across k=4…100 (0.7204 / 0.6907 / 0.6940 / 0.7147 / 0.6475),
  so the gap WIDENS with granularity (+0.175 → +0.277). Routing serves one expert at full strength
  regardless of pool size ⇒ **dilution is a property of merging, not of sharding**. The k=200 cell
  (+0.053) is a capacity artifact (r8 adapters; r32×200 > 46 GiB), not a routing failure
  ([routed-ladder](2026-07-26_7b-routed-ladder-and-tooling.md)).
- **[open]** H-SCAF — scaffolding moves `attn_norm` (s3→s5) and `logit_div` (s0→s8) magnets at
  *fixed* model, separately from scale. Is a scaffolded pool systematically more/less leaky, or just
  differently? Blocked: no scaffolded 7B base exists.
- **[open]** H-LM-STABLE — is `centroid_lm`'s s4 magnet stable because authors 80–99 are a hub in
  *every* LM's hidden-state geometry, or because both models share pretraining data? A third model
  family (Qwen / phi-2) at k=10 separates these.
- **[resolved ✓ CONFIRMED]** H1 k=10 tombstone separability — tomb_author margin **AUC 0.982,
  retain-FPR 0.002 @ 90% catch** (bars 0.90/0.05); independently 0.988/0.002 on n=32
  ([phase1](2026-07-18_phase1-results.md)).
- **[resolved ✓ both halves]** H2 granularity — per-EXPERT confirmed-broken (argmax FPR
  **0.727**); per-AUTHOR usable via the thresholded margin (0.002 FPR; raw argmax 0.110).
- **[resolved ~ partial]** H3 serving triple — catch RE-ADJUDICATED NOT MET by the as-served shard rung (0.605 per question — see the [2026-07-21 correction](2026-07-21_serving-catch-correction.md); the 0.96 figure was a counter mis-read; author-rung serving cell unrun); embed-full guard fired (0.6872
  → deltas vs embed-full); retain Δmu −0.0061 narrowly misses ≤0.005 (argmax FPR; the
  Phase-1 thresholded margin [FPR 0.002] is the predicted fix). **fq side-prediction
  refuted as stated** — smoke inversion reversed at extended (0.588 vs 0.697); the durable
  claim is fq is leak-BLIND (sibling "passes" at 0.588 while serving 95.5% confabulations)
  ([phase2](2026-07-18_phase2-results.md)).
- **[resolved ✓ both rungs]** H4 Mode-B — **author rung Branch A CONFIRMED: ρ 0.833 →
  0.031/0.000/0.047 (~95% sealed at the serving surface)**; shard rung mediation model
  SUPPORTED (0.113/0.110/0.433 vs predicted 0.036/0.105/0.333 — R4 exact, nothing crosses
  the 0.15 falsifier). Residual leak = (1 − rung catch) × sibling leak — predictive.
- **[resolved ✓]** H5 content audit (n=400) — sibling_vs_gold 0.277 ≈ floor 0.249 (+0.028
  ≤ +0.05, ceiling 0.599 live); **confabulation rate 0.955**; cross-author 0.181. The
  unplanted leak is misinformation, not disclosure.
- **[closed — not run per contingency]** H6 anchored content attenuation — the
  pre-registered gate argues against it: H5 shows unplanted sibling content already ≈
  base-level and the author-rung tombstone seals Mode-B; the sole remaining target
  (confabulation reduction) duplicates the tombstone. Reopen only on review request.
- **[resolved ✓ n=32 half]** H7 dial — capture/shift monotone (0.320/0.573/0.727); author-rung
  FPR ≈ 0.004–0.006/author (bar ≤ 0.01); f01 usable at 0.975 catch / 0.008 FPR. k=10
  multi-shard cells deferred.
- **[resolved ✓ a+c; open b]** H8 — (a) coverage 0.863 ✓ prior; paraphrased **0.900** (names
  survive paraphrase); stripped 0.0; 18/200 nameless. (c) **disclosure AUC 0.839 ≥ 0.75 —
  the seal discloses the deletion.** (b) hybrid router = Phase 4.
- **[resolved ✓ SUPPORTED]** H-DET (2026-07-23) — the new *per-author landing-determinism*
  scalar tracks the magnet/diffuse split: magnet routers 0.69–0.82 (centroid_lm_last 0.823,
  activation_norm 0.817, centroid_lm 0.692) vs diffuse 0.43–0.47 (ppl 0.432, attn_norm 0.465).
  A hidden-state router funnels a *single* deleted author's whole question set to the same
  survivor — the magnet is per-author, not just an aggregate ([2026-07-23](2026-07-23_master-table-orphan-metrics.md)).
- **[resolved ✓ SUPPORTED]** H-W2 content generality (2026-07-23) — the confabulation-not-disclosure
  result holds across ALL 9 servable routers: sibling-vs-deleted-gold 0.265–0.291 ≈ floor 0.249
  (< +0.05 bar), confab 0.94–0.98 ([2026-07-23 wave23-results](2026-07-23_wave23-results.md)).
- **[resolved ✓ SUPPORTED w/ caveat]** H-W3 Mode-B per family (2026-07-23) — unsealed verbatim ρ@R8
  rises monotone in R to 0.93–1.0 for centroid_sbert/centroid_lm/ppl (ppl worst, ρ 0.97 by R2 —
  routes to the fact-holder); activation_norm ρ degenerate (magnet ceiling≈floor, read qualitatively).
  Paraphrase-surface ρ@R8 only 0.07–0.15.
- **[resolved ✗ REFUTED]** H-DET→host-hit (2026-07-23) — determinism does NOT predict Mode-B
  host-hit (Kendall τ **−0.167**, n=4): ppl is the *least* deterministic (0.432) yet has the
  *highest* host-hit (**0.853**; R2 0.700 vs geometric 0.08–0.12). ⇒ **two orthogonal channels** —
  geometric concentration → *integrity* failure (confabulation); **content-seeking** → *privacy*
  leak ([hdet-refuted-wave4-prereg](2026-07-23_hdet-refuted-wave4-prereg.md)).
- **[resolved 2026-07-24]** **H-CEIL ✓** (shared router-independent `expert_max` ceiling = 0.9193 de-degenerates activation_norm: 0.901, vs a Wave-3 reading of 1.0 at EVERY R incl. R=1 — all Wave-3 per-family ρ superseded) · **H-EXH ✓** (6 families; attn_norm 0.905, **logit_div 0.016**) · **H-SEAL-PPL ✗ STRUCTURALLY BROKEN** (ρ@R8 0.000 masked a real R2 failure: abstain dips to 0.52 at R2 where the lone surviving cohost is the distinctively least-perplexed expert ⇒ large margin ⇒ no abstain ⇒ routes to the leaker, host-hit 0.88 prob 0.87; R4/R8 abstain 0.96/0.98 only because many cohosts collapse the margin. Most blind at R2 = the common single-cohost case; a confidence seal peaks where privacy risk is sharpest) · **H-DISC-RUNG ✓** (catch↔disclosure rank-correlate perfectly: shard 0.605→0.839, name 0.703→**0.967**, author 0.963→**0.987** — no free seal; the name rung is clean in STORAGE but loud in BEHAVIOUR) ([wave45-results](2026-07-24_wave45-results.md)).
- **[resolved ✗→BLIND, 2026-07-24]** H-MIA-ROUTER — the router leak is INVISIBLE to composed MIA: sibling (leak) AUC **0.072–0.182**, tombstone 0.220–0.284, both below the oracle floor 0.379. `fq` and MIA both miss the routing channel; only content/ρ probes see it ([wave45-results](2026-07-24_wave45-results.md)).

- **[resolved ✓ 2026-07-24]** H3-CLOSER — the author-rung tombstone served with a THRESHOLDED margin (τ=0.1944, retain-calibrated) is now DEMONSTRATED: mu **0.695** ≥ embed-full 0.6872, retain_prob **0.604** ≥ 0.573, fq **0.808** (hardest seal of any arm) — ~0 cost on top of the embedding router, vs the shard-argmax rung's −0.0061 miss ([wave6](2026-07-24_wave6-tombstone-and-utility.md)).
- **[resolved ~ 2026-07-24]** H-UTIL — per-strategy routed utility×leak: **utility and leakage are POSITIVELY coupled** (content routers ppl/key route best, retain_prob 0.84–0.87, but ppl leaks ρ 0.937 / key self-detects; behavioral-norm routers worst on BOTH). `logit_div` is the only genuine non-leaker (ρ 0.016) but costs ~0.23 retain_prob vs ppl — 'route by atypicality' is a real lever, NOT a free lunch. Sweet spot = key/lexical routing.
Full falsifiable bars: [2026-07-18_router-leak-preregistration.md](2026-07-18_router-leak-preregistration.md).

## What worked
- **The headline contrast (2026-07-18):** on the same queries and both routers, identity
  signals separate orphans from retain at **AUC 0.98–0.99 (FPR 0.002 @ 90% catch)** while
  every confidence detector (global τ / per-expert z / margin / kNN-density) sits at
  **0.56–0.63 (FPR 0.82–0.88)** — the 07-07 abstain refutation generalizes to the whole
  confidence family, and the tombstone fixes it with the one signal thresholds lack:
  *which identity was deleted*.
- **The constructive headline (Phase 2):** the author-rung tombstone is a deployable seal
  with all four price tags measured — routing-level orphan catch **0.963** (author rung, audit) with served shard-rung catch **0.605** (see the 07-21 correction), Mode-B residual
  **ρ 0.833 → 0.047**, retain cost **≈ 0.006 mu** (argmax; thresholded predicted ~0),
  deletion-disclosure **AUC 0.839**. And the provenance ladder is a *dial*: residual leak
  = (1 − rung catch) × sibling leak, verified quantitatively on the shard rung.
- **The unplanted leak is an integrity failure, not a privacy failure:** 95.5% of
  sibling-served answers about deleted authors are novel confabulations at base-level
  gold overlap; real disclosure exists only under Mode-B replication — and that channel
  is what the tombstone seals.
- Known-value continuity: stale 0.815 / sim-ratio 0.98 / shift 0.727 reproduced exactly.
- Sibling host-hit rate (0/0.07/0.34/0.87 by R) replicates entangled_facts' ρ_embed
  monotonicity from routing alone — the Mode-B leak is router-mediated, as claimed.

## What didn't / open problems
- **Per-expert tombstones are dead on the n=32 pool** (FPR 0.727 — experts co-host retain
  authors; granularity must match the deletion unit).
- **The seal is observable:** deletion-disclosure AUC 0.839 — pricing every tombstone/registry
  policy; report alongside any seal claim.
- The name rung is privacy-cleanest but weakest and encoder-sensitive (catch 0.703 MiniLM /
  0.282 instructor-xl); the k=10 centroid router itself is weak in absolute terms (top-1
  0.565/0.605), so serving deltas must be read against the embed-full baseline.

## Open ideas / next steps (recorded at closure — none run)
- τ-thresholded tombstone SERVING arm (Phase-1 ROC predicts retain Δmu → ~0 at 90% catch;
  ~2 h) — closes the H3 0.0061-vs-0.005 gap.
- Hybrid registry-first router (H8b): registry → embed-fallback + author-tombstone; names
  survive paraphrase (0.900) so the leak budget ≈ the 10% name-free slice. Needs a small
  serving class (~1 day).
- Name-rung deletion-disclosure AUC (CPU, from the saved `.sims.npz` dumps).
- k=10 multi-shard mass-deletion dial cells; composed-MIA rider on embed-sibling;
  detector-spread ↔ ρ per-fact correlation.

## Entries (chronological)
- [2026-07-18 — pre-registration + harness build](2026-07-18_router-leak-preregistration.md) —
  H1–H8 bars; tombstone/dump_sims/centroid-mode audit extensions, `EmbedRoutedModel`
  serving arm, probe tombstone policy, ρ aggregator, content-audit script, analyzers; all
  CPU gates green; run matrix + cap constraints.
- [2026-07-18 — Phase-1 results](2026-07-18_phase1-results.md) — H1 ✓ decisively
  (tomb_author AUC 0.982/0.988, FPR 0.002); H2 ✓ (per-expert broken 0.727 / per-author
  usable); H7 ✓ n=32 half; H8a ✓ (0.863; paraphrase 0.900); H8c ✓ disclosure 0.839; c_probe
  branches H4 (author→Branch A, shard→mediation ≈0.04/0.11/0.33); Phase-2 smoke queued.
- [2026-07-20 — Phase-3 pre-registration: all-router sweep](2026-07-20_all-router-sweep-preregistration.md) —
  H-ARCH/H-DIAL/H-POOL/H-TRAINED/H-DATASET/H-ENC/H-SEAL-GEN bars; the leak protocol
  (capture / adequacy-ratio / retain-shift / confidence-separability) generalized to 16
  router arms across k=10, k=200, n=32, DBpedia; identity controls pre-registered.
- [2026-07-20 — build hardened + J1–J6 launched](2026-07-20_all-router-sweep-launch.md) —
  review fixes (analyzer PENDING semantics, H-ENC confidence half, driver serial lane +
  real k=200 pool), SepMLP spec-v2 joins as the routerless arm (GPU-placement blocker
  fixed pre-flight); all CPU gates green; jobs 446563–446568 (serial lane); deviations
  from the pre-registration declared (J3 on the 7B e25 pool; DBpedia no-fallback drop).
- [2026-07-21 — all-router sweep results](2026-07-21_all-router-sweep-results.md) — J2 behavioral re-run (self-check bf16-tie fix); **H-ARCH REFUTED — the leak is router-family-specific**: 6/9 dense-similarity routers (sentence/LM/activation embeddings) leak orphans inseparably (adequacy 0.95–1.000, confidence AUC 0.41–0.63), but ppl (perplexity) and key_tfidf SELF-DETECT orphans (adequacy 0.38/0.67, AUC 0.97–0.998, 1–7% FPR@90%catch); LM-hidden-state routers are worst AND break the tombstone (FPR ~1.0); RouterLoRA leak-blind ×3 seeds; mpnet/bge + DBpedia reproduce. New lead: ppl-as-native-detector (privacy-cleaner than the sentinel).
- [2026-07-22 — Group-A/B depth pre-registration + (a) results](2026-07-22_groupab-depth-preregistration.md) — concentration analysis: dense routers collapse orphans onto n_eff 1.4–2.1 magnet experts (shard-4 hub), semantic/generative spread over 5.5–7.7; H-CONC ✓.
- [2026-07-22 — Group-B leak-inheritance results](2026-07-22_groupab-depth-results.md) — **exact subtraction DEFUSES the router leak**: SIFT/ClAMU misroute 100% of orphans under a realistic selector yet answer-prob of the deleted fact stays at floor (SIFT 0.086, ClAMU 0.124 vs a served author's ~0.9); ClAMU's +0.142 ROUGE bump is style-confabulation (confab 0.77, flat prob), not disclosure. The router leak is load-bearing only for drop-and-survive (Group A); the tombstone is a nice-to-have for subtraction methods (catch 0.96). MemSinks + Mode-B-for-B deferred.
- [2026-07-23 — master table + per-strategy orphan metrics (Wave 0, CPU)](2026-07-23_master-table-orphan-metrics.md) — consolidated the [routing master reference](../../tofu_sisa_lora/reports/ROUTING_MASTER_2026-07-23.md): methods (base/FT/model) + the extended Table-7 strategy inventory + the full per-strategy orphan battery (concentration/collateral/adequacy/self-detect AUC/tomb). **New metric H-DET ✓** (per-author landing determinism tracks n_eff: magnet 0.69–0.82 vs diffuse 0.43–0.47). Widened `analyze_orphan_destinations.py` (`--sims_glob`, top3/gini in md); regenerated `orphan_destinations.{md,csv}` (superset) + `rl_family_leak_table.md`; all CPU gates green. GPU Waves 2–3 landed (see next entry).
- [2026-07-24 — Wave 6: H3 seal demonstrated + per-strategy utility×leak](2026-07-24_wave6-tombstone-and-utility.md) — **(A)** τ-thresholded author-rung tombstone served: mu **0.695** / retain_prob **0.604** / fq **0.808** (≥ embed-full baseline, hardest seal) → H3 CLOSED, seal demonstrated at ~0 cost. **(B)** full servable-9 routed-utility table: utility↔leak POSITIVELY coupled (ppl best utility 0.873 but worst leak ρ 0.937; logit_div only non-leaker ρ 0.016 but −0.23 retain_prob; key/lexical = sweet spot). Jobs 448154–448156, 448178.
- [2026-07-24 — Waves 4–5: shared ceiling, exhaustive families, ppl-seal, per-rung disclosure](2026-07-24_wave45-results.md) — **the leak is a property of the SCORING SIGNAL, not routing per se**: ρ@R8 ppl **0.937** (content-seeking) … **logit_div 0.016** (outlier-seeking — a genuine non-leaker, post 0.156 ≈ floor 0.144). H-CEIL ✓ (activation_norm de-degenerated 0.901; Wave-3 ρ superseded), H-EXH ✓ (6 families), H-DISC-RUNG ✓ (no free seal; name rung clean in storage, loud in behaviour 0.967), H-SEAL-PPL ✗ at the bar but non-monotone ⇒ not claimable. Jobs 448051–448054, 448059–448060, collector 448073.
- [2026-07-23 — H-DET→host-hit REFUTED + Wave-4 pre-registration](2026-07-23_hdet-refuted-wave4-prereg.md) — determinism does NOT predict Mode-B host-hit (Kendall τ **−0.167**; ppl is *least* deterministic 0.432 yet *highest* host-hit **0.853**, R2 0.700 vs geometric 0.08–0.12). **Two orthogonal leak channels:** geometric concentration → integrity failure (confabulation); **content-seeking** (ppl routes to the least-surprised = fact-holding expert) → privacy leak. Pre-registers **H-SEAL-PPL** (a ppl-native margin seal should MISS replicated facts precisely because it is content-seeking), **H-CEIL** (router-independent `expert_max` ceiling de-degenerates activation_norm ρ), **H-EXH** (attn_norm + logit_div complete the table). Jobs 448051–448054.
- [2026-07-23 — Wave 2/3 results: per-strategy content + per-family Mode-B ρ](2026-07-23_wave23-results.md) — **Wave 2 (9 routers):** leak = confabulation not disclosure everywhere (sibling-vs-gold 0.265–0.291 ≈ floor 0.249, confab 0.94–0.98; magnets reproduced — centroid_lm_last s4 304/400, activation_norm s6 326/400). **Wave 3 (4 families):** unsealed verbatim Mode-B ρ@R8 centroid_sbert 0.972 / centroid_lm 0.929 / ppl 1.00 (monotone in R; ppl worst — routes to the fact-holder by R2); activation_norm degenerate. Paraphrase ρ@R8 0.07–0.15. Two silent GPU bugs caught+fixed mid-run (unsupported centroid_sbert_q; activation-router input_ids device mismatch). Jobs 448009/10/11/25/26.
- [2026-07-26 — 7B orphan coverage: the scale/scaffold de-confound](2026-07-26_7b-orphan-coverage.md) — closed the gap that 7B orphan behavior existed for **one** pool (k=200) while every other orphan number was 1B. Four new batteries: 7B k=10 feature+behavioral, 7B k=50, and the never-audited **plain 1B k=10** (added mid-session once the data revealed a confound: the repo's 1B k=10 battery is the SCAFFOLDED pool, the 7B one is PLAIN). **Routers that never read the model are bit-identical across 1B-scaf/1B-plain/7B-plain** — same n_eff, magnet and adequacy to 3 dp (H1 ✓, by construction); **every expert-reading router moves its magnet with scale** (activation_norm s6→s7 with n_eff 1.1→2.4; ppl s7→s1; attn_norm s5→s4; logit_div s8→s7; centroid_lm_last s4→s2) (H2 ✓). **`centroid_lm` — most concentrated, most adequate (0.976–0.999), least detectable (AUC 0.728) — keeps magnet s4 in ALL three arms**: the worst leaker is the most reproducible, so the leak is structural. H3 ✗ but as a metric artifact — n_eff is not comparable across granularities (orphans 400→80→20, survivors 9→49→199); adequacy is, and it IS monotone (key_tfidf 0.667→0.385→0.194). Also new: Table H′ (7B-only mu + orphan) in the master report, and `reproduce/{LLAMA2_7B,METHODS}.md`. Jobs 448587 (B1 array), 448590–448594. ⚠ ran at a user-authorized **6-GPU** ceiling, above the CLAUDE.md §1 cap of 4.
- [2026-07-26 — 7B routed-mu ladder complete + two analyzer traps](2026-07-26_7b-routed-ladder-and-tooling.md) — **H4 ✓ dilution is a property of MERGING, not of sharding.** Completed the 7B routed ladder (job 448587): routed `key_exact` 0.7204 / 0.6907 / 0.6940 / 0.7147 / 0.6475 / 0.4728 at k=4/10/20/50/100/200 vs merged `dare_ties` 0.545 → 0.420 on the SAME pools — the merge row decays monotonically, the routed row is flat within ±0.04 over k=4…100, so the gap WIDENS with granularity (+0.175 → +0.277). The k=200 exception is capacity (r8 adapters), not routing. **Two analyzer traps found + documented (CAVEATS §13):** `analyze_router_family.py` keys rows by `<strategy>@k<k>` and silently dedups two pools at the same k by mtime (caught, reverted, table now bit-identical to 2026-07-23); `snapshot_results.py` without `--ckpt-root` copies 0 files and blanks `MANIFEST.tsv`. `orphan_destinations` regenerated to 111 cells / 38 trajectories; reproduce/ at 334 cells, 289 verified, 0 FAIL.
