### Target Date: 2026-07-21 (All-router sweep results — the leak is router-FAMILY-specific, not universal; dense-similarity routers leak inseparably, generative/lexical routers self-detect)
- **Hypotheses / what we're testing:** the Phase-3
  [pre-registration](2026-07-20_all-router-sweep-preregistration.md) battery — H-ARCH,
  H-DIAL, H-POOL, H-TRAINED, H-DATASET, H-ENC, H-SEAL-GEN — now with ALL nine router
  families measured (the behavioral four were missing until the J2 re-run today).
- **Setup:** J1/J3/J4/J5/J6 landed 2026-07-20 (jobs 446563/446565/446566/446567/446568);
  **J2 (behavioral: ppl / activation_norm / attn_norm / logit_div) failed 07-20 on a
  self-check assertion and was fixed + re-run today as job 446665.** Bug: the faithfulness
  gate demanded the batched score-matrix argmax exactly equal the bs=1 `router.route()`
  path, but the behavioral routers compute per-sample norms inside a PADDED batch while the
  reference runs single-example → bf16 matmul noise flips the argmax between two shards
  scored within ~0.4% of each other (row 423, shard 9 vs 7). Fix: `run_self_check` now
  tolerates a disagreement iff the two contested shards' scores are within a 2% relative
  band (a genuine feature/axis bug flips between materially-different scores and still
  raises); unit-tested both directions (`test_router_family.test_self_check_tie_tolerance`;
  CPU gate green). J2 re-run: self-checks 50/50 with 2 ties tolerated (activation_norm,
  logit_div), 0 real disagreements. `router_family_audit.py` re-hashed; analyzer
  `analyze_router_family.py` re-run over 14 family entries / 38 cells. Seed 42.
- **Results:** (adequacy = orphan sibling-match quality, 1.0 = orphan indistinguishable from
  a normal query for that sibling = dangerous; best conf. AUC = can a confidence detector
  spot orphans, 0.5 = no; FPR@90 = retain false-positives at 90% orphan catch by the best
  confidence detector)
  | router family | full acc | orphan capture | adequacy | conf. AUC | FPR@90 | verdict |
  |---|---|---|---|---|---|---|
  | centroid_lm (LLM hidden state) | 0.26 | 0.92 | **0.999** | 0.47 | 0.95 | leak, inseparable |
  | centroid_lm_last | 0.32 | 0.87 | **0.998** | 0.50 | 0.89 | leak, inseparable |
  | activation_norm (LoRA-B norm) | 0.18 | **1.00** | **0.997** | 0.41 | 0.96 | leak, inseparable |
  | attn_norm | 0.07 | 0.82 | **1.000** | 0.53 | 0.89 | leak, inseparable |
  | centroid_sbert (MiniLM) | 0.56 | 0.54 | 0.967 | 0.56 | 0.88 | leak, inseparable |
  | centroid_sbert_q | 0.57 | 0.48 | 0.971 | 0.61 | 0.84 | leak, inseparable |
  | logit_div | 0.52 | 0.66 | 0.953 | 0.63 | 0.70 | leak, weak-separable |
  | **key_tfidf (word overlap)** | 0.98 | 0.57 | **0.667** | **0.97** | **0.07** | leak, SEPARABLE |
  | **ppl (perplexity)** | **1.00** | 0.54 | **0.377** | **0.998** | **0.01** | leak, SEPARABLE |
  | key_exact (name match) | 0.87 | 1.00→fallback | — | 0.93 (no-match) | — | design-leak, native detector |
  | oracle q2author (control) | 1.00 | 0.00 | — | — | — | no leak (by construction) |
  - **H-ARCH — REFUTED as stated (and the refutation is the finding):** 6/9 families leak
    with confidence-inseparability; **2 are separable** (ppl, key_tfidf) → fails the "≥7/9"
    bar and trips the "≥2 separable" REFUTE clause. The leak is NOT universal across router
    architectures.
  - **The dividing line is the scoring space.** Routers that pick an expert by SEMANTIC
    similarity in a dense space (sentence/LM/activation embeddings) blur an orphan into a
    near-perfect sibling match (adequacy 0.95–1.000, confidence at chance). Routers that
    score by LEXICAL overlap (tfidf) or GENERATIVE fit (ppl) make the orphan stand out
    (adequacy 0.38–0.67) because the surviving experts genuinely fit it worse — perplexity
    is a *native orphan detector* (AUC 0.998, 1% FPR at 90% catch).
  - **H-DIAL:** k=10 monotonicity holds for the leaky families (capture/retain-shift rise
    with drop count); the earlier k=200 non-monotone FLAGs stand (documented). Sub-bar
    (centroid_sbert_q adequacy ≥0.95 at all k=10 drops) PASS.
  - **H-TRAINED (RouterLoRA, 3 seeds):** CONFIRMED — the learned router renormalizes its
    attention over survivors leak-blind (AUC 0.588±0.002 on h_norm, 0.555±0.001 on
    −max_share; orphan top-1 share ratio 1.658±0.001), byte-stable across seeds 42/43/44.
  - **H-ENC (mpnet/bge):** CONFIRMED — both reproduce the dense-router leak (adequacy 0.97 /
    0.98, confidence AUC ≤0.65) and the author-rung tombstone seals both (0.98/0.95 catch).
  - **H-DATASET (DBpedia retriever):** the leak reproduces on a different dataset (pooled
    adequacy 0.912, no separating τ at 90%/10%) — one tag (d_batch15) shows 78% capture at a
    huge 0.785 retain shift, the mass-deletion regime.
  - **H-POOL (k=200 per-author granularity):** granularity does NOT rescue the dense routers
    into separability at the single-author drop (centroid_lm adequacy 0.976 at AUC 0.728);
    the sbert/tfidf k=200 cells collapse on router ACCURACY (adequacy 0.19–0.71) not on the
    leak mechanism — read within-router, per the J3 base-model-confound note.
  - **H-SEAL-GEN:** the author-rung tombstone clears catch≥0.90 @ FPR≤0.10 in 3 feature
    spaces (+ MiniLM prior) — but NOT in the LM-hidden-state space (centroid_lm/_lm_last
    FPR 0.97–1.00): the seal fails exactly where the leak is worst, because that space is
    so blurry the sentinel matches everything.
- **What worked / hypothesis verdict:** the self-check fix was correct and minimal — the
  behavioral routers were never buggy, the gate was over-strict on bf16 ties. H-ARCH's
  refutation reframes the whole campaign's generality claim: *the router leak is a property
  of dense-similarity routing, not of modular unlearning per se.* A deployer who must use an
  embedding router leaks inseparably (and needs the tombstone); a deployer who can route by
  perplexity or lexical match gets orphan-detection for free.
- **Observations:** (i) The two worst leakers (LM-hidden-state routers, adequacy 0.999) are
  also the two where the tombstone BREAKS (FPR ~1.0) — a coherent story: a feature space
  blurry enough to make every orphan look native also makes every sentinel match native
  traffic. Avoid LM-hidden-state routing entirely. (ii) ppl's separability comes at a cost
  it's the most expensive router (a forward per expert per query) — cheap defenses (tfidf)
  are separable too, so the practical advice is "don't route by dense semantic embedding if
  you can avoid it." (iii) activation_norm/attn_norm are poor routers (acc 0.07–0.18) yet
  still leak inseparably — leak is independent of router accuracy. (iv) Silent-failure
  checks: J2 self-check 50/50, ties itemized; all adequacy CIs author-blocked bootstrap;
  the k=200 J3 base-model confound (Llama-2-7B) is footnoted and read within-pool.
- **New questions / new hypotheses:** (1) **ppl-as-native-detector arm** — deploy the
  perplexity router as its OWN orphan gate (route by embedding, VERIFY by perplexity,
  abstain on high-ppl): predicted to seal the dense-router leak without a tombstone (no
  retained deleted data — a privacy-cleaner defense than the sentinel). Cheap, high-value,
  directly answers the red-team's "the tombstone retains deleted data" objection. (2) Does
  the LM-hidden-state tombstone failure survive a whitening/per-shard-z normalization of
  that space? (3) SepMLP routerless arm still pending (P1 smoke unrun) — the "selection in
  the weights" row.
- **Next Steps:** thread README + master index updated; the leak table
  (`rl_family_leak_table.md`) and analysis JSON are the artifacts. SepMLP P1 smoke and the
  ppl-native-detector arm are the two open leads; both await a go-ahead.
