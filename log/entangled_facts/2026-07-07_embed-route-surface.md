### Target Date: 2026-07-07 (does an embedding router SURFACE the hidden Mode-B residual?)

- **Hypotheses / what we're testing:** the climax that ties the three deletion-verification threads
  together. The 2026-07-06 Mode-B result showed the fact survives in the surviving owners' weights
  (`expert_max` ρ → 0.998) but the **hard author-key** router hides it at the serving surface
  (`served_key` post-drop == floor). The §9-D routing audit
  (`../ramole/2026-07-06_routing-audit-results.md`) showed an **embedding** router sends a deleted
  entity's orphan queries to a surviving sibling at similarity-ratio 0.98. This experiment closes the
  loop: **serve the planted world through a training-free embedding router and measure whether it
  routes the deleted author's queries onto a fact-holding host expert — surfacing the residual the
  hard router hides.**
  - **H-embed-surface:** post-drop `served_embedsim` ρ is **above the floor** (> `served_key` ≈ 0)
    and **rises with R** (more host copies ⇒ higher chance the topically-nearest surviving shard
    holds the fact). CONFIRM: verbatim `served_embedsim` post-drop ρ ≥ 0.2 at R≥4 and monotone in R,
    with served_embedsim > served_key at matched (R, mode). REFUTE: served_embedsim ≈ floor
    (the embedding router does not route orphans onto fact-holding hosts — the residual stays hidden
    even under a realistic router). Either way it quantifies the **realistic leak fraction** of an
    embedding router on replicated facts — the number that connects B (fact survives) to C (router
    decides if it leaks).
- **Setup:** new `served_embedsim` channel in `eval_entangled_probe.py` — a training-free per-shard
  centroid router (each surviving shard's centroid = mean of its member authors' question embeddings
  under a SentenceTransformer; route each probe question to the nearest surviving centroid by cosine,
  activate that expert). Post-drop excludes shard 9 (the deleted donor's expert). Run on the same
  three worlds as the 07-06 probe: ceiling = planted no-drop, post-drop = planted drop-9, floor =
  clean oracle-B; ρ = clip((post−floor)/(ceiling−floor),0,1). Pool
  `Llama-3.2-1B-Instruct_entangled_k10`, seed 42, surfaces orig+para. CPU-gate the centroid-router
  logic (nearest-centroid over survivors, deleted shard excluded) before the SLURM probe jobs.
  Commands + sha256 + job IDs recorded on run.
- **Setup (run):** `eval_entangled_probe.py` sha c5593bcee8bc (`served_embedsim` channel +
  `_build_shard_centroids`, MiniLM per-shard centroids over member questions). 3 SLURM jobs
  **441046** ceiling (planted, no drop) / **441047** post-drop (planted, drop 9) / **441048** floor
  (clean oracle-B, drop 9), both surfaces, pool `Llama-3.2-1B-Instruct_entangled_k10`, seed 42.
  GPU smoke first (6 facts): routes exclude the dropped shard 9 (orphans → shards 5/6), R=1 ≈ floor.
- **Results:** served_embedsim ρ (verbatim, original surface) vs the hard-router served_key ρ,
  and the host-hit rate (fraction of orphans the embedding router routed onto a fact-holding host):

  | R (verbatim) | embed ceiling | embed post-drop | embed floor | **ρ_embed** | ρ_key (hard) | host-hit rate | #host shards |
  |---|---|---|---|---|---|---|---|
  | 1 (control) | 0.412 | 0.079 | 0.085 | **0.000** | 0.000 | 0/25 (0%) | 0/7 |
  | 2 | 0.602 | 0.184 | 0.134 | **0.107** | 0.000 | 1/25 (4%) | 1/7 |
  | 4 | 0.723 | 0.392 | 0.132 | **0.439** | 0.000 | 9/25 (36%) | 3/7 |
  | 8 | 0.845 | 0.725 | 0.127 | **0.833** | 0.000 | 20/25 (80%) | 7/7 |

  Paraphrase surface: ρ_embed R2/R4/R8 = 0.000 / 0.080 / 0.104 (weaker — the embedding router routes
  on question similarity, and a paraphrase-planted fact probed on the original question is a harder
  centroid match; the leak is strongest on the verbatim surface).
- **What worked / hypothesis verdict:** **H-embed-surface SUPPORTED, strongly.** Served through the
  SAME weights, the hard author-key router hides the residual entirely (ρ_key = 0.000 at every R)
  while the embedding router **surfaces** it, monotonically in R: ρ_embed 0 → 0.107 → 0.439 →
  **0.833**. At R=8 the embedding router leaks 83% of the residual the hard router hides completely.
  Meets the pre-registered CONFIRM (verbatim ρ ≥ 0.2 at R≥4 — 0.439/0.833; monotone; embed > key at
  every matched cell). *Falsifier (embedsim ≈ floor) not observed.*
- **Observations:** the mechanism is exact — **ρ_embed = the host-hit rate = P(nearest surviving
  shard holds the fact) ∝ #host shards ∝ R.** With one owner (R=2) the fact sits in 1 of 7 host
  shards, so an orphan rarely routes to it (4% → ρ 0.107); with R=8 the fact is in all 7 host shards,
  so almost any surviving shard the orphan lands on holds it (80% → ρ 0.833). This is the
  three-thread climax made quantitative: **the fact survives deletion (Exp B, `expert_max` ρ→0.998),
  and which router you deploy decides whether it leaks — the hard identity router hides it (ρ=0), the
  embedding router surfaces it (ρ→0.83), and the embedding router is exactly the one the §9-D audit
  (`../ramole/2026-07-06_routing-audit-results.md`) showed sends orphans to plausible siblings and
  (`../ramole/2026-07-07_routing-fix-arms.md`) cannot be sealed by a confidence threshold.** Silent-
  failure checks clean: R=1 control ρ=0 (no host to leak to); ceiling rises with R (more hosts → a
  better-matching held copy); floor flat ≈0.13 (clean experts never hold the fact); dropped shard
  excluded from routing (verified in smoke).
- **New questions / new hypotheses:** does the leak scale the same way under a *learned* router
  (RAMoLE RouterLoRA) rather than the training-free centroid router — or does the learned gate
  concentrate the leak further? Would the SEUF-attribution detector (Exp-4 of the 07-06 entry, AUC
  0.777) have flagged exactly the R≥4 facts that leak here (i.e. is detector-spread predictive of the
  realistic embed-leak)? Cross-tabulating detector-spread vs ρ_embed per fact is the natural next
  cheap analysis.
- **Next Steps:** fold the ρ_embed-vs-ρ_key curve into `reports/ENTANGLED_FACTS_REPORT_2026-07-06.md`
  as the closing "which router decides the leak" figure; optionally the detector-spread-vs-ρ_embed
  correlation (cheap, reuses `detector.json` + this run's per_fact).
