### Target Date: 2026-07-18 (Phase-1 results — identity seals work, confidence never did; the seal's price is disclosure)
- **Hypotheses / what we're testing:** H1 (k=10 tombstone separability), H2 (n=32 granularity),
  H7 (deletion-count dial, n=32 half), H8a (registry coverage), H8c (deletion-disclosure), and
  the c_probe measurement that branches H4 — bars as pre-registered in
  [2026-07-18_router-leak-preregistration.md](2026-07-18_router-leak-preregistration.md).
- **Setup:** jobs **445344** (rl-aud-basepin, n=32 base-pinned, policies stale/dropped/abstain/
  tombstone/key + dump_sims) / **445345** (rl-aud-ft) / **445346** (rl-centroid, k=10 MiniLM
  centroid audit + `--probe_manifest` + holdout10) / **445347** (rl-aud-f01) / **445348**
  (rl-aud-f05, chained after f01); CPU post-processing `bash submit_router_leak.sh collect`
  (ROC on author-parity eval halves; registry coverage; dial table). Configs/shas as in the
  pre-registration; seed 42. Outputs: `.../legonet_n32_k3/results/rl_audit_*.json{,.sims.npz}`,
  `.../experts_scaf_k10/results/router_leak/rl_centroid_k10.json{,.sims.npz}` +
  `rl_roc_{centroid,legonet}.json`, `rl_coverage.json`, `rl_deletion_dial.md`.
- **Results:**
  - **Known-value asserts all reproduce:** n=32 stale orig_top1 **0.815**, dropped sim-ratio
    **0.98**, retain shift **0.727** — byte-level continuity with the 07-06/07-07 audits.
  - **Threshold-detector ROC (eval on held-out author halves), k=10 centroid router:**
    tomb_author **AUC 0.982, retain-FPR 0.002 @ 90% orphan catch**; tomb_name 0.969/0.078;
    tomb_expert(=shard centroid) 0.839/0.472; the whole confidence family — global_top1 /
    per_expert-z / top1−top2 margin / knn_density — sits at **AUC 0.57–0.61 with FPR 0.82–0.88
    at 90% catch**. n=32 legonet index: tomb_author **0.988/0.002**; tomb_name 0.858/0.438;
    confidence family 0.56–0.63.
  - **Raw argmax operating points:** k=10 — tomb author catch 0.963 / FPR 0.0906, shard rung
    0.605/0.0583, name 0.703/0.0181 (1 name-extraction fallback). n=32 — per-expert catch
    0.963 / **FPR 0.727**; per-author 0.980/0.110; name 0.282/0.015.
  - **Deletion dial (n=32, per-author rung):** f01 catch 0.975 @ FPR **0.0083**, f05 0.975 @
    0.0339, f10 0.980 @ 0.110 (≈ 0.004–0.006 FPR/author); dropped retain shift monotone
    0.320 → 0.573 → 0.727; per-expert FPR 0.320/0.573/0.727.
  - **k=10 centroid router quality:** full-pool top-1 accuracy retain **0.565** / forget 0.605;
    sibling sim-ratio 0.971; sibling deletion retain shift 0.0583; orphan sibling mass
    spread over all 9 survivors (max shard 7 at 82/400).
  - **c_probe (H4 mediator), planted probe surfaces:** author rung **0.96–1.00** (both
    surfaces, every R); shard rung 0.48–0.76; name rung 0.58–0.88. Sibling-route host-hit
    rate by R: 0.00 / 0.06–0.08 / 0.32–0.36 / **0.86–0.88** (orig/para).
  - **Registry coverage:** original-all **0.863** (prior 0.86 ± 0.02 ✓); forget10 originals
    0.895; paraphrased forget10 **0.900**; name-stripped 0.0; 18/200 authors yield no
    extractable name.
  - **Deletion-disclosure:** AUC(forget10 vs holdout10 by tombstone-margin) = **0.839**
    (mean scores +0.006 vs −0.038).
- **What worked / hypothesis verdict:**
  - **H1 CONFIRMED, decisively:** 0.982 ≥ 0.90 AND 0.002 ≤ 0.05 — the tombstone-relative
    margin separates orphans from retain almost perfectly on the router the serving arms
    actually deploy. Same verdict independently on the n=32 index (0.988/0.002).
  - **H2 CONFIRMED both halves:** per-EXPERT tombstone confirmed-broken (argmax FPR 0.727 ≥
    0.5 — G2's prediction exact); per-AUTHOR usable — the *thresholded* margin detector is
    the policy (0.002 FPR at 90% catch; the raw argmax rule's 0.110 sat marginally above the
    0.10 bar, resolved by thresholding).
  - **H7 SUPPORTED (n=32 half):** capture/shift monotone in deletion count; author-rung FPR
    ≈ linear at ~0.004–0.006/author ≤ 0.01; forget01 meets the usable-at-small-deletions bar
    (0.975 @ 0.008). k=10 multi-shard dial cells not yet run (deferred to a later batch).
  - **H8a SUPPORTED:** 0.863 within the pre-registered band; **names survive paraphrase**
    (0.900) — a lexical registry loses almost nothing on TOFU paraphrases; the residual
    surface is the ~10% name-free questions + deliberate stripping (coverage 0.0).
  - **H8c CONFIRMED:** AUC 0.839 ≥ 0.75 — **the seal discloses the deletion** (Streisand
    cost line established, pre-registered as a finding, not a bug).
  - **H4 gate:** author rung → **Branch A** (c_probe ≈ 0.97; predict ρ_tomb ≤ 0.10 at R=8);
    shard rung c_probe 0.48–0.76 → the mediation-model test with per-R predictions
    ρ_tomb ≈ (1−c_probe)·ρ_sibling: ≈0.04 (R2) / ≈0.11 (R4) / **≈0.33 (R8, orig)**. Both
    served arms queued in Phase 2.
- **Observations:** (i) **The headline contrast:** identity signals (AUC 0.98–0.99) vs
  confidence signals (0.56–0.63) on the SAME queries, both pools — the 07-07 abstain
  refutation generalizes to every confidence variant we tried, while the tombstone family
  clears its bars with two orders of magnitude lower FPR. Selection failure is not a
  calibration problem; it is an information problem the identity signal fixes. (ii) The
  k=10 centroid router is *weak in absolute terms* (top-1 0.565/0.605) — shard centroids
  average 20 authors into a blur; the pre-registered H3 guard (embed-full mu baseline)
  matters, and deletion cells must be read as deltas vs embed-full. (iii) The sibling
  host-hit rate replicates entangled_facts' ρ_embed monotonicity (0/0.07/0.34/0.87 vs ρ
  0/0.107/0.439/0.833) from the routing side alone — the leak is router-mediated, as
  claimed. (iv) The name rung is encoder-sensitive (MiniLM 0.703 catch vs instructor-xl
  0.282) and near-zero-FPR everywhere — the privacy-clean rung is real but leakier;
  the provenance ladder has a real price axis. (v) Silent-failure checks: no NaNs; sims
  sidecars written for all five audits; f05 correctly chained after f01 (cap held).
- **New questions / new hypotheses:** (1) Does the disclosure AUC drop on the name rung
  (less specific sentinel → weaker Streisand signal)? Computable from the dumped sims —
  CPU only. (2) Hybrid registry-first + tombstone-fallback (H8b, Phase 4): with names
  surviving paraphrase at 0.900, the hybrid's leak budget is dominated by the 10% name-free
  slice — measure directly. (3) Author-sentinel FPR grew superlinearly at f10 on n=32
  (0.0083→0.0339→0.110) — is 20 sentinels the knee, or is it pool-size-relative? A k=200
  extrapolation cell would say which.
- **Next Steps:** Phase-2 smoke queued: **445357** embedrouted_full / **445358** sibling_del9 /
  **445359** tombstone_del9 / **445360** content-audit CAP=40 — chained
  `afterany:445350–445356` behind the memadapt-eval + ctv-train queues (submitted briefly
  unchained during a queue race; dependencies added ~1 min later via `scontrol update` — no
  overlap occurred). Then: extended triple + Mode-B tombstone worlds (BOTH rungs — shard =
  mediation test, author = Branch-A collapse test) + full content audit; Phase-1 README
  ledger update; scratchpad updated.
