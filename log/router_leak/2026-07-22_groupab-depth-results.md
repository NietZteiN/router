### Target Date: 2026-07-22 (Group-B results — exact subtraction DEFUSES the router leak; the tombstone is load-bearing only for drop-and-survive methods)
- **Hypotheses / what we're testing:** H-GB1–4 as pre-registered in
  [2026-07-22_groupab-depth-preregistration.md](2026-07-22_groupab-depth-preregistration.md)
  (a-side H-CONC already resolved in that entry). The (b) question: do the merged-model +
  per-task-mask methods inherit the router leak under a realistic MiniLM selector, and does
  misrouting actually leak DELETED content?
- **Setup:** routing = CPU inline (MiniLM only), serve = jobs **446955/446956** (CAP=40
  smoke) → **447157/447158** (full n=400), `orphan_route_groupb.py` 3-arm serve (oracle /
  base-θ0 / realistic) on the 400 deleted-author questions; SIFT `configs/sift_masks_tofu_1b.json`
  + `sift/tau_bar_forget10.pt`, ClAMU `configs/clamu_tofu_1b.json` + `clamu/tau_bar_forget10.pt`
  + `assignment_K16_forget10.json`. Seed 42. Leak metric = answer-prob of the GOLD answer
  (a real leak → ~ceiling 0.9); ROUGE-vs-gold is style-confounded, so disclosure vs
  style-confabulation is separated by ROUGE-vs-base + the confabulation rate.
- **Results:**
  - **Routing (do orphans misroute?):** SIFT (180 per-author units) max_share 0.11, n_eff
    27.1, **confidence AUC 0.972**, author-sentinel tombstone catch 0.965 / FPR 0.003. ClAMU
    (16 feature clusters) max_share 0.17, n_eff 8.8, **confidence AUC 0.585**, tombstone
    catch 0.960 / FPR 0.000. Both misroute 100% of orphans (their own unit is gone).
  - **Serving (does misrouting leak the deleted fact?), n=400, answer-prob of the deleted gold:**
    | method | oracle (correct deletion) | base θ0 | **realistic (misrouted)** | ROUGE vs_gold (o/b/real) | real vs_base ROUGE | confab |
    |---|---|---|---|---|---|---|
    | SIFT | 0.118 | 0.128 | **0.086** | 0.392 / 0.226 / 0.338 | 0.254 | 0.79 |
    | ClAMU | 0.128 | 0.128 | **0.124** | 0.226 / 0.226 / 0.368 | 0.265 | 0.77 |
    Realistic misrouting = 800/400 route-calls, all misrouted (100%).
- **What worked / hypothesis verdict:**
  - **H-GB1 — SUPPORTED, refined:** Group B inherits the *misrouting* (100% orphans →
    surviving units), but confidence-detectability tracks unit GRANULARITY exactly like Group
    A: SIFT per-author (180 units) is SEPARABLE (conf AUC 0.972, ≈ the k=200 `centroid_sbert`
    0.982 cross-check ✓); ClAMU K=16 feature clusters are INSEPARABLE (0.585, like the k=10
    dense routers). The pre-registered "AUC ≤ 0.75 for all" bar is therefore split — met by
    ClAMU, not by SIFT — but H-GB2 shows detectability is moot for privacy.
  - **H-GB2 — CONFIRMED by the faithful metric (the key result):** exact-subtraction methods
    do NOT leak deleted content under full misrouting. Answer-prob of the deleted gold stays
    at floor for both (SIFT realistic 0.086 ≤ base 0.128 ≤ ceiling-of-a-served-author ~0.9;
    ClAMU 0.124 ≈ base 0.128) — a misrouted surviving mask operates on τ̄ with τ_u already
    subtracted, so it cannot resurrect the fact. **The pre-registered ROUGE bar is literally
    TRIPPED for ClAMU** (realistic 0.368 vs floor 0.226, +0.142 > +0.05) — but the
    confabulation axis adjudicates it as a STYLE artifact, not disclosure: realistic-vs-base
    ROUGE only 0.265 and **confab rate 0.77** (77% of answers overlap neither gold nor base),
    i.e. the retain-cluster mask makes biography-FORM text (names/dates) that surface-overlaps
    the gold while the actual fact stays unrecalled (flat prob). Same lesson as Phase-2:
    ROUGE/fq are style-confounded; answer-prob + content audit are faithful. SIFT trips
    nothing (realistic ROUGE 0.338 < oracle floor 0.392).
  - **H-GB4 — descriptive:** ClAMU's K=16 feature clusters concentrate orphans less than
    contiguous k=10 shards (n_eff 8.8 vs the dense k=10 routers' 1.4–2.1) — semantic clusters
    spread the misroute; still inseparable by confidence.
  - **H-GB3 (MemSinks) — NOT RUN** (separate-project config plumbing; deferred). Predicted
    robust by a DIFFERENT mechanism: deletion masks the author's sink slice OFF, so a
    misrouted query reaches a surviving author's slice with the deleted slice inactive —
    content-free like SIFT/ClAMU but via masking not subtraction.
- **Observations — the Group-A-vs-B contrast, now measured:** the router leak's *consequence*
  depends on the deletion mechanism, not just the selector. **Group A (drop-and-survive):**
  misrouting reaches a surviving EXPERT that holds real per-author knowledge → under Mode-B
  replication it leaks the actual fact (ρ→0.833, [phase2](2026-07-18_phase2-results.md)); the
  tombstone is load-bearing. **Group B (exact subtraction):** misrouting reaches a MASK over a
  τ̄ the deleted author is already gone from → confabulation only, prob at floor; exact
  subtraction is ROUTER-ROBUST FOR FREE. The author-sentinel tombstone still works on Group B
  (catch 0.96) but is a nice-to-have, not a privacy necessity, there. So: the router-leak
  threat is specific to methods that keep deletable content in *surviving* units.
- **New questions / new hypotheses:** (1) MemSinks (masking) — run it to confirm the
  masking-vs-subtraction robustness split (best-effort, ~1 GPU cell). (2) Does a Mode-B
  planted-fact world change the Group-B verdict? Predicted no (subtraction removes the donor's
  copy; surviving OWNERS' copies live in their masks, which a realistic selector could route
  to — the same fact-level residual as Group A, but requiring planted Group-B checkpoints).
- **Next Steps:** fold (a) concentration + (b) Group-A-vs-B contrast into
  `reports/ROUTER_LEAK_EXPLAINED_2026-07-21.md`; README + master-index updates; thread back to
  complete (MemSinks + Mode-B-for-B recorded as open).
