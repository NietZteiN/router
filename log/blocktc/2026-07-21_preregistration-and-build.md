### Target Date: 2026-07-21 (Pre-registration + build)
- **Hypotheses / what we're testing:** Pre-registered BEFORE any GPU spend. Binding
  contract: [`blocktc_tofu/DESIGN.md`](../../blocktc_tofu/DESIGN.md) (v1, 2026-07-21).
  All anchors are OU chat-template track (never the plain Question:/Answer: track).
  - **H1 localization (make-or-break):** the single-bottleneck block transcoder reaches
    SELECTIVE — median on/off activation-mass ratio ≥5 on the leakage matrix
    (`measure_selectivity.py`, shared column excluded from off-maxima) — with own-author
    answer-prob ≥0.80. REFUTE: median <2 (LAZY) in every pilot arm. Anchor: LoRA keys
    1.11, 100% LAZY
    ([merge_mechanism 2026-07-15](../merge_mechanism/2026-07-15_key-firing-results.md)).
  - **H2 all-active serving retains utility (anti-memsinks):** all-active vs own-only
    recall-probe gap ≤0.05 (tripwire gate G3); OU Util.R ≥0.95 and Util.G ≥0.95
    (MemAdapt FT row 1.075/1.024). REFUTE: ≥0.15 all-active drop (memsinks anchor:
    all-on mu 0.4373 vs control 0.6438).
  - **H3 deletion clean:** drop the forget10 authors' blocks ⇒ Mem ∈ [0.55, 0.70]
    (Retrained 0.590, MemAdapt 0.630), |ΔUtil.R| ≤0.03, Agg ≥0.80 (strong-confirm
    0.84–0.90 vs MemAdapt 0.869 / Retrained 0.874); dropall ≡ calib_base.
  - **H4 relearn parity:** median relearn target/control (holdout10) ratio ∈ [0.8, 1.25]
    probed at steps [0, 5, 10, 25, 50]. REFUTE: target relearns ≥2× faster.
  - **H5 MIA (measurement + attribution):** 4 raw MIA AUCs on the unlearned model vs the
    oracle floor 0.379 ([deletion_audit
    2026-07-06](../deletion_audit/2026-07-06_composed-mia-results.md)); anchor MemAdapt
    Priv 0.917. Direction attributed (residual memorization vs over-suppression).
  - **H6 exactness-by-construction:** surviving parameters never received ANY gradient
    from a deleted author's data — including as negatives (stronger than sepmlp's claim,
    whose surviving authors train with forget rows as negatives). By construction: hard
    detach-trick gradient masking + author-free phase-0 pool + generic-only suppression;
    enforced by the G0 grad-isolation gates; verified behaviorally via H4 + H5.
  - **H7 span:** span-3 (decoders write residual at layers 9/10/11) beats span-1 by
    >0.02 OU model_utility at the matched 53M param budget; else adopt span-1. —
    P5 ablation.
- **Setup:** P0 build session; nothing submitted. Architecture (DESIGN §2–§5): frozen
  `meta-llama/Llama-3.2-1B-Instruct` (bf16, sdpa, D=2048, 16 layers) + one block
  transcoder — encoder read at layer 9 post-attention-norm MLP input,
  `a = ReLU(W_enc·xn + b_enc)` in fp32, F = 200×32 author features + 128 shared = 6528;
  zero-init per-feature decoders `W_dec (3, 2048, F)` adding to the residual at layers
  9/10/11; serving = all features live, no router; deletion = index-select out the
  author's 32 rows/cols (O(1)). Training: detach-trick gradient masking (values bitwise
  ≡ serving; gradient only through the own-block mask — decoder path included, since
  decoder grad ∝ activation value); phase 0 trains ONLY the 128 shared features on an
  author-free pool (2000 seed-42 Alpaca + 100 real_authors), then freezes them; phase 1
  alternates 1:1 single-source author batches (routed LM loss) with generic batches
  (suppression L1 on author-feature activations only, λ warmup over the first 15% of
  steps); weight_decay 0, per-block clip, fresh AdamW at phase-1 start; never holdout10
  anywhere. Seed 42 everywhere; OU chat-template track via the imported
  [`memadapt_tofu/data_tofu.py`](../../memadapt_tofu/data_tofu.py); envs test-env
  (train/probes/tests) + unlearning (OU evals); detector init `questions`,
  init_scale 1.0; bs 32, 15 epochs (phase 1), cosine LR.
  **Four corrections adopted vs the source design doc (declared now, before any run):**
  1. **Dead plain-track anchor dropped:** the doc's 0.664 utility anchor is plain-track
     and was refuted at repro ([routing_scaffold
     2026-07-02](../routing_scaffold/2026-07-02_scaffold-repro.md): committed code gives
     0.474) → all bars use OU-track anchors (MemAdapt Agg 0.869, Retrained Agg 0.874).
  2. **Relearn probed at steps [0, 5, 10, 25, 50]** (the house sepmlp-bars protocol).
  3. **Author-free phase-0 pool:** the doc said retain+generic, but ALL 200 TOFU authors
     are deletable candidates — any TOFU retain row in phase 0 would push a deletable
     author's gradient into the undeletable shared block and break H6. Phase 0 =
     Alpaca + real_authors only.
  4. **Suppression only on generic (NO_AUTHOR) batches** — never on author batches and
     never using other authors' rows as negatives; generic data belongs to no author, so
     the suppression gradient touching all 200 author encoder rows preserves exactness.
  **Planned gate ladder** (driver [`submit_blocktc.sh`](../../blocktc_tofu/submit_blocktc.sh),
  BLOCKTC_THROTTLE=2 under the global 4-GPU cap, exclude sprint4; gates are manual reads
  — never pre-chain a submission across a gate):

  | phase | what | gate |
  |---|---|---|
  | P0 | build + [`configs/`](../../blocktc_tofu/configs/) + tests (this session) | compiles clean |
  | G0 | CPU gate suite `pytest tests/ -q` (test-env; DESIGN §9: 14 gates incl. bitwise no-op, detach-value identity, per-phase grad isolation, deletion identities, cross-layer stash, KV-cache/grad-ckpt parity, holdout10) | all green + STUB previews |
  | P1 | smoke (`configs/smoke.json`: K=4, bs 8, ~5 steps, phase 0 → phase 1 in one job) | save→reload parity, grad checks, peak-mem read |
  | P2 | K=20 pilot, 6 arms `lr {3e-4, 1e-3, 3e-3} × λ {0.01, 0.1}` (array `0-5%1`) + selectivity/recall probes | H1 bars (median ≥5 AND own-prob ≥0.80); dead/lazy blocks ⇒ sepmlp hinge/Gram/promotion fallback |
  | P3 | K=200 (`configs/blocktc_1b_k200.json`, pilot-winner lr/λ) + probe200 | G3 recall gap ≤0.05 before eval spend |
  | P4 | OU evals (ft / unlearned-forget10 / dropall) + relearn battery + MIA | H2/H3/H4/H5 bars |
  | P5 | ablations: span-1 vs span-3 at matched budget (H7), depth/width | deferred behind own pre-registration |

  Param-count check (test-env python, this session): W_enc 6528×2048 = 13,369,344 +
  b_enc 6,528 + W_dec 3×2048×6528 = 40,108,032 → **53,483,904 total** = 4.33% of the
  1,235,814,400-param base, 11.8× smaller than sepmlp's 0.63B added at K=200.
  **SLURM job ids: none — this entry pre-registers before any job.** The P1 smoke job id
  will be recorded in the thread README Status line and in the next entry.
- **Results:** none — build session; param count 53,483,904 verified (arithmetic above;
  zero GPU spend).
- **What worked / hypothesis verdict:** pending (pre-registration — no runs).
- **Observations:** pending (pre-registration).
- **New questions / new hypotheses:** Does ONE read site (layer 9) give width-32 blocks
  enough capacity for own-prob ≥0.80, where sepmlp reads at all 16 layers? Do
  detector-init encoder rows + zero-init decoders + the plain LM+L1 recipe avoid dead
  blocks without sepmlp's hinge/Gram/promotion terms (the pre-registered fallback if the
  pilot shows dead or lazy blocks — per-block firing telemetry decides)? Does the pilot
  λ grid {0.01, 0.1} bracket the suppression knee, or does selectivity need the
  in-domain negatives that H6 forbids? Does the cross-layer activation stash survive
  real KV-cache generation at K=200 scale as cleanly as the CPU gates predict?
- **Next Steps:** (1) Finish P0 + run G0 (`pytest tests/ -q` in test-env, all 14 gates
  green + `STUB=1` driver previews) before any submission. (2) P1 smoke — human submits;
  record its SLURM job id in the thread README Status and the next entry; peak-mem read
  decides bs32. (3) P2 pilot array `0-5%1` after the manual smoke read; G2-style read
  against the H1 bars frozen above before any K=200 spend. (4) Coordinate with sepmlp:
  its P2 pilot decision is pending with the user — same probes/anchors, so pilot results
  should be read side-by-side.
