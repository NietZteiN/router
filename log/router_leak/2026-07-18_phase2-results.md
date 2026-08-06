### Target Date: 2026-07-18 (Phase-2 results — the author-rung tombstone seals the Mode-B leak; the mediation model holds; fq is leak-blind, not leak-inverted)
- **Hypotheses / what we're testing:** H3 (serving triple), H4 (Mode-B mediated collapse,
  both rungs), H5 (sibling-content audit) — bars as pre-registered in
  [2026-07-18_router-leak-preregistration.md](2026-07-18_router-leak-preregistration.md),
  with H4 branched by Phase-1's c_probe
  ([phase1-results](2026-07-18_phase1-results.md): author rung ≈ 0.97 → Branch A;
  shard rung 0.48–0.76 → mediation predictions ρ ≈ 0.036/0.105/0.333 at R2/4/8 orig).
- **Setup:** smoke **445357–445360** (embed triple + content CAP=40; chained
  afterany:445350–445356 via `scontrol update` ~1 min post-submit after a queue race — no
  overlap occurred) → extended wave **445668–445670** (triple) → **445671–445674** (4 Mode-B
  worlds, chained afterany the triple) → **445675** (content n=400, chained afterany Mode-B);
  max 4 concurrent by construction, queue empty at submit. New probe policy
  `--embed_policy tombstone_author` (per-author sentinels of the dropped shard; gated in
  `test_entangled_facts.py` + `test_router_leak.py` re-runs). ρ assembly:
  `aggregate_rho.py` on ceiling `ceiling_embedsim.json` (07-08, drop-none) × per-rung
  post/floor pairs → `router_leak/rho_embedsim_{sibling,tomb,tomba}.json`. Seed 42.
- **Results:** (headline signal = `served_embedsim_prob_orig`, verbatim mode — the
  reference surface of the 07-07 sibling curve, which reproduces exactly: 0.107/0.439/0.833)
  - **Serving triple (extended):** embed-full mu **0.6872** (retain_prob 0.573,
    route_mismatch 1121/2840; forget KNOWN: f_rouge 0.639, fq 0.0108) · sibling_del9 mu
    **0.6922** (f_rouge 0.415, f_ppl 25.2, fq **0.5880**) · tombstone_del9 mu **0.6861**
    (f_rouge 0.418, f_ppl 20.1, fq **0.6967**; route stats: 755 tombstoned of ≈785
    forget-metric calls ≈ **0.96 serving-level catch**).
  - **Smoke tier (for the record):** 0.6830 / 0.6950 / 0.6823; fq smoke 0.9578 (sibling) vs
    0.8080 (tombstone) — the smoke-tier inversion did NOT survive extended KS power.
  - **Mode-B ρ (orig surface, verbatim):**
    | arm | R2 | R4 | R8 |
    |---|---|---|---|
    | sibling (reference) | 0.107 | 0.439 | 0.833 |
    | tombstone shard rung | 0.113 | 0.110 | 0.433 |
    | mediation prediction (1−c_probe)·ρ_sib | 0.036 | 0.105 | 0.333 |
    | **tombstone author rung** | **0.031** | **0.000** | **0.047** |
  - Paraphrase-probe (`prob_para`) columns carry the known small-denominator instability
    (sibling para shows saturated 1.000 cells from near-degenerate ceiling−floor gaps at
    n=25) — orig surface is the headline, per the 07-06/07-07 convention.
  - **Content audit (n=400):** own_vs_gold 0.599 (ceiling) · base_vs_gold 0.249 (floor) ·
    sibling_vs_gold **0.277** · sibling_vs_basegen 0.278 · sibling_vs_sibgold 0.181 ·
    **confabulation rate 0.955**; sibling mass spread over all 9 survivors (max shard 7,
    82/400).
- **What worked / hypothesis verdict:**
  - **H4 author rung — Branch A CONFIRMED, decisively:** ρ = 0.031/0.000/**0.047** vs the
    ≤ 0.10 bar at R8 — the per-author sentinel tombstone **collapses the served Mode-B
    residual from 0.833 to 0.047 (~95% sealed)**, training-free, deletion-derived-data-only
    (sentinels recomputable from the deleted authors' questions at request time). The
    fact remains in host weights (expert_max, 07-06) — this is a *serving-surface* seal,
    exactly as scoped.
  - **H4 shard rung — mediation model SUPPORTED:** measured 0.113/0.110/0.433 vs predicted
    0.036/0.105/0.333 — R4 dead-on (|Δ|=0.005), R2 +0.077, R8 +0.100; nothing crosses the
    |Δ| > 0.15 falsifier. Residual leak through a rung is quantitatively **(1 − rung
    catch) × sibling leak** — the seal's algebra is now predictive, which is what lets a
    deployer budget a rung choice.
  - **H3 PARTIAL:** catch ✓ (0.96 ≥ bar 0.91); the pre-registered embed-full guard fired
    (0.6872 < 0.7309 → deltas read vs embed-full, and the −0.064 to key-routed 0.7509 is
    router-accuracy cost, not deletion cost); retain Δmu tombstone-vs-sibling −0.0061 —
    **narrowly misses the ≤ 0.005 bar** (the ~9% argmax FPR made visible in mu; the
    thresholded-margin operating point from Phase 1 [FPR 0.002] is the predicted fix, needs
    a τ-parameterized serving arm). **fq side-prediction REFUTED as stated:** the smoke
    inversion reversed at extended (0.588 vs 0.697). The durable, honest claim is
    **fq is leak-BLIND**: the sibling arm scores 0.588 — "passes" — while serving
    95.5%-confabulated sibling answers; fq separates neither sealed from leaky nor leaky
    from clean-deleted.
  - **H5 CONFIRMED (n=400):** sibling_vs_gold − floor = +0.028 ≤ +0.05 with the
    measurement live (ceiling − floor = 0.35 ≥ 0.3) → in the unplanted world the router
    leak is **misinformation, not disclosure** — 95.5% of sibling answers are novel
    confabulations (low vs base-gen AND low vs gold), and cross-author disclosure is
    absent (0.181). The privacy harm is confined to Mode-B replication (sealed above);
    the unplanted harm is an *integrity* failure the tombstone also removes (base-served).
- **Observations:** (i) The campaign's constructive headline is now complete: **orphan-catch
  0.96 / Mode-B ρ 0.833→0.047 / retain cost ≈ 0.006 mu / disclosure AUC 0.839** — a
  deployable seal with all four price tags measured. (ii) The mediation algebra means the
  provenance ladder is a *dial*: shard rung (weakest identity, most private) leaves
  (1−0.6)·leak; author rung (strongest, forget-derived) leaves ~5%; the name rung sits
  between (Phase-1 catch 0.703) — privacy-vs-seal is now a quantified tradeoff, not a
  qualitative choice. (iii) The Δmu miss is an argmax artifact: Phase-1's ROC says a
  calibrated τ on the tombstone margin buys catch 0.90 at FPR 0.002 — worth one small
  serving-arm extension if the 0.005 bar matters for the paper. (iv) Silent-failure
  checks: sibling ρ reference reproduces the 07-07 numbers exactly from the same JSONs;
  route-stat totals conserve (2840 = 2085+755); no NaNs in the headline signals; para
  small-denominator cells flagged, not averaged into claims.
- **New questions / new hypotheses:** (1) τ-parameterized tombstone serving (margin
  threshold instead of argmax): predicted retain Δmu → ~0 at catch 0.90 — one flag +
  smoke/extended pair. (2) Does the disclosure AUC (0.839) drop on the name rung? CPU from
  the Phase-1 dumps. (3) **Phase-3 contingency verdict: the pre-registered gate argues
  AGAINST running H6** — anchoring was to attenuate leak *content*, but H5 shows unplanted
  sibling content is already ≈ base-level (disclosure-free confabulation) and the Mode-B
  channel is sealed by the author rung; the only remaining H6 target (confabulation-rate
  reduction) duplicates what the tombstone already removes. Recommend recording H6 as
  not-run-per-contingency unless review wants the ~3 GPU-h anyway. (4) Phase 4 remains:
  hybrid registry-first router (H8b — needs a small serving class) + per-rung disclosure
  CPU cell + ROUTER_LEAK_REPORT.
- **Next Steps:** at user review gate — (a) close H6 per contingency or run it; (b) build
  the hybrid arm + τ-serving extension (~1 day + ~3 GPU-h) or go straight to the report;
  (c) multi-seed note: all serving cells are deterministic given the pool (seed 42
  recorded); the only stochastic stage (anchored training) is the phase being skipped.
