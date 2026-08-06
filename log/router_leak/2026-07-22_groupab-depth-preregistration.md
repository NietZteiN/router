### Target Date: 2026-07-22 (Group-A depth + Group-B leak-inheritance — pre-registration + (a) results)
- **Hypotheses / what we're testing:** two follow-ups on the method landscape.
  - **(a) H-CONC (descriptive, over EXISTING sweep dumps — no new run):** the "magnet-shard"
    concentration of orphan destinations is a property of DENSE-similarity routing at coarse
    granularity. Predict: dense-embedding routers (centroid_lm/_lm_last, activation_norm)
    stay concentrated (n_eff ≤ 3, max_share ≥ 0.5) at single-drop; semantic/generative
    routers (sbert, ppl, tfidf, logit_div) stay diffuse (n_eff ≥ 5). Read across drop-count
    (d9→d9_8→d9_8_7_6) and k=200 granularity.
  - **(b) H-GB1 (routing inherit):** with the oracle author-ID selector replaced by a
    realistic MiniLM centroid selector, Group-B methods (SIFT/ClAMU/MemSinks) misroute
    orphans among surviving units (orphan→surviving-unit = 1.0 by construction) with best
    confidence AUC ≤ 0.75 — they inherit the Group-A routing leak because the selector is
    identical. SIFT/MemSinks per-author routing ≡ the k=200 `centroid_sbert` cell (cross-check).
  - **(b) H-GB2 (serving robustness — the key result):** SIFT & ClAMU `*_unlearn` served
    under the realistic selector do NOT leak deleted content despite full misrouting — forget
    ROUGE-vs-gold stays ≤ the oracle-deletion floor + 0.05, because τ_u is already subtracted
    from τ̄ (a surviving mask over τ̄_retain cannot resurrect the deleted fact). CONFIRM ≤
    floor+0.05; REFUTE if a misrouted mask resurrects the fact (ROUGE ≥ ceiling − 0.1). This
    is the Group-A-vs-B contrast: the router leak is dangerous for drop-and-survive (A),
    defused by exact subtraction (B).
  - **(b) H-GB3 (MemSinks contrast, best-effort):** MemSinks (sink-slice masking, not
    subtraction) under misrouting routes to a surviving author's slice over the SHARED
    adapter — measure whether that leaks more than SIFT/ClAMU (directional; predicted still
    ≈ floor since the deleted slice is off).
  - **(b) H-GB4 (ClAMU granularity, descriptive):** feature-clustered K=16 units route
    orphans to a semantically-nearest cluster; report destination concentration vs contiguous
    shards.
- **Setup:** plan `~/.claude/plans/can-you-expand-on-moonlit-kahan.md` (approved; user chose
  routing+serving for (b), fold write-up into `ROUTER_LEAK_EXPLAINED_2026-07-21.md`). Seed 42,
  interpreter test-env, work dir ~/tofu_sisa_lora. **New/changed code (sha256-12):**
  `analyze_orphan_destinations.py` f881bdb7c50f (concentration HHI/Gini/n_eff + monotonicity
  over the stored `orphan_capture.top1_hist`; entropy survivor-normalized to match the sweep
  convention, cross-checked), `eval_routed_scaffold.py` fa268dec68df (`build_unit_centroids`
  arbitrary author-grouping; `build_shard_centroids` now a thin wrapper — return
  byte-identical), `groupb_realistic_router.py` a79713889aee (`attach_realistic_router` swaps
  `_route` for a MiniLM centroid router over surviving units, oracle-gated OOD; `per_author_units`
  / `clamu_cluster_units`), `orphan_route_groupb.py` 69b94293e148 (routing + serve modes),
  `test_groupb_router.py` 15c487a1f8dc, `submit_groupb.sh` 90c03df8e3ba. **CPU gate green:**
  `test_groupb_router.py` 2/2 (unit sets; realistic route OOD-gate + orphan-misroute counter).
  **Artifacts (verified on disk):** SIFT `sift/tau_bar_forget10.pt`, ClAMU
  `clamu/tau_bar_forget10.pt` + `assignment_K16_forget10.json` (retain re-clustered); MemSinks
  configs in `~/memsinks_tofu/configs/` (strict2_e25 — best-effort). Run matrix: (a) CPU done;
  (b) routing = CPU inline (MiniLM only); (b) serve = 2–3 GPU smoke cells × ~8 min under the
  4-GPU cap (queue currently empty).
- **Results — (a) DONE (concentration over existing dumps):**
  `reports/orphan_destinations.{md,csv}`. Single-drop (d9 / d199) n_eff = effective # of
  siblings the orphans spread over:
  | router | max_share | n_eff | busiest |
  |---|---|---|---|
  | activation_norm k10 | 0.82 | 1.4 | s6 |
  | centroid_lm_last k10 | 0.77 | 1.7 | s4 |
  | centroid_lm k10 | 0.65 | 2.1 | s4 |
  | centroid_lm k200 (1-author drop) | 0.70 | 1.9 | a128 |
  | attn_norm k10 | 0.41 | 3.7 | s3 |
  | centroid_bge k10 | 0.33 | 5.7 | s4 |
  | logit_div k10 | 0.27 | 5.5 | s0 |
  | ppl k10 | 0.25 | 7.0 | s7 |
  | centroid_sbert k10 | 0.18 | 7.2 | s7 |
  Cross-checks: recomputed max_share/entropy == the stored `orphan_capture` values (0 WARN).
- **What worked / hypothesis verdict — (a):** **H-CONC CONFIRMED.** Bimodal split:
  dense-embedding/norm routers collapse orphans onto ~1–2 magnet experts (n_eff 1.4–2.1),
  semantic/generative routers spread over ~5.5–7.7. The magnet is **shard 4 (authors 80–99)**
  for every LM-space router (k10 centroid_lm/_lm_last AND the bge encoder) — a hub in dense
  semantic space, echoing LegoNet's 135-author hub. Monotonicity: the magnet is a
  coarse-granularity + single-drop effect — max_share holds/shrinks as more shards drop
  (fewer survivors, mass redistributes), and at k=200 a single-author deletion still
  concentrates (0.70) but a 20-author deletion spreads (0.17, 20 orphans over 180 survivors).
  (b) verdicts pending the runs below.
- **Observations:** (a) is a re-reduction of already-logged sweep counts (not a new
  experiment), so it needs no pre-run registration — recorded here for coherence. The
  bimodal n_eff cleanly separates the same 6-leaky/3-separable families H-ARCH found, by a
  different lens (concentration, not detectability). (b) scope note: memory-adapters excluded
  (no oracle to swap — already content-routed; native ~10% cross-source read leak noted as a
  footnote); Mode-B planted worlds for Group B are future work (no planted checkpoints).
- **New questions / new hypotheses:** does the shard-4 hub predict which sibling answers a
  given deleted author (is the magnet content-determined)? (CPU, from dumps.)
- **Next Steps:** run (b) routing (CPU) → serve smoke (GPU) → adjudicate H-GB1–4 in a results
  entry → fold (a)+(b) into `ROUTER_LEAK_EXPLAINED_2026-07-21.md`.
