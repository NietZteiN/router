### Target Date: 2026-07-21 (CORRECTION — the "96% serving-level catch" claim in the Phase-2 entry is wrong; the served (shard-rung) tombstone catches 60.5% per question)
- **Hypotheses / what we're testing:** none — this is a correction entry (append-only
  protocol) for [2026-07-18_phase2-results.md](2026-07-18_phase2-results.md), found while
  building the per-query accounting table for the final report.
- **Setup:** re-derivation from existing artifacts only, no new runs. Sources: the
  extended serving JSON `embedrouted_tombstone_del9.json` (route_stats routed=2085,
  deleted=755, ood=1953; TOFU calls 2840) and the Phase-1 centroid audit
  `rl_centroid_k10.json` (shard-rung orphan catch 0.605, retain FPR 0.0583).
- **Results:** the Phase-2 entry read `deleted=755 ≈ 785 forget-metric calls → catch 0.96`.
  That inference is wrong twice: (a) the `deleted` counter increments for ANY query whose
  top-1 centroid is the deleted shard — including retain-side false tombstones — and
  (b) forget-question calls in an extended eval number far more than 785 (prob/ppl + rouge
  + per-perturbed-answer truth-ratio forwards reuse the same questions). Decomposing 755
  with the Phase-1 per-question rates: 0.605·F + 0.0583·(2840−F) = 755 → F ≈ 1079
  forget-author calls, i.e. **≈653 orphan-call catches + ≈102 retain false-tombstones** —
  fully consistent with the audit's per-question rates. The served arm used the SHARD-rung
  sentinel (the only rung implemented in `EmbedRoutedModel`), so its true per-question
  catch is **0.605 (242/400 orphan questions)**, exactly the Phase-1 audit number; 0.963
  (385/400) belongs to the AUTHOR rung, which was audited (Phase 1) and served only in the
  Mode-B probe arm (where its seal is confirmed by ρ 0.031/0.000/0.047), never in the
  full TOFU serving triple.
- **What worked / hypothesis verdict:** **H3's catch component is RE-ADJUDICATED: NOT MET
  by the as-served rung** (0.605 ≪ the ≥0.91 bar); the bar would be met by an author-rung
  serving arm per its audit catch, but that cell was not run (already on the open list at
  closure). Unchanged: H3's retain Δmu (−0.0061, marginal miss), the fq-blindness verdict,
  all Phase-1 numbers, the Mode-B ρ tables (both rungs), the content audit, and every
  detector ROC. The campaign's constructive headline stands but must be phrased at the
  right level: *author-rung tombstone — routing-level catch 96.3% (audit) and Mode-B
  serving seal ρ 0.833→0.047 (measured); shard-rung tombstone — full-serving catch 60.5%
  (measured), residual consistent with the mediation law.*
- **Observations:** the root cause is an under-instrumented counter (`stats["deleted"]`
  conflates orphan catches and retain FPs, and per-call counts conflate metric passes).
  If the serving arm is ever revisited: split the counter by q2author group and count
  unique questions. Downstream fixes applied in the same change: thread README, the repo
  report `ROUTER_LEAK_REPORT_2026-07-18.md`, and the published artifact page (claims of
  "96% serving catch" replaced with the rung-correct numbers). The Phase-2 entry itself is
  left untouched per the append-only rule; it should be read together with this entry.
- **New questions / new hypotheses:** none beyond the already-recorded open item (run the
  author-rung TOFU serving cell, ideally thresholded — predicted catch ≈0.90–0.96 at
  retain FPR ≈0.002).
- **Next Steps:** none — thread remains closed; this entry completes the record.
