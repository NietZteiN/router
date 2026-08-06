### Target Date: 2026-07-18 (Router-leak deep-dive — pre-registration + harness build)
- **Hypotheses / what we're testing:** The post-deletion router leak
  ([PATHS_FORWARD §7.1](../PATHS_FORWARD_2026-07-13.md)) in depth: can an *identity-based*
  seal (tombstone) do what the refuted confidence threshold
  ([ramole 07-07](../ramole/2026-07-07_routing-fix-arms.md)) could not, what does the leak
  actually *say*, and what does sealing cost? All bars pre-registered before any GPU run.
  ⚠ fq is NEVER a leak bar here — sibling-served forget answers come from retain-style
  experts, so fq is predicted to score the LEAKY policy better (the inversion is itself a
  pre-registered side-prediction under H3).
  - **H1 (k=10 tombstone separability):** on the k=10 MiniLM centroid router (the router the
    serving arms actually deploy), the tombstone-relative margin separates orphan forget
    questions from retain questions. CONFIRM: ROC-AUC ≥ 0.90 on the held-out author half AND
    retain-FPR ≤ 0.05 at orphan-catch ≥ 0.90. REFUTE: AUC < 0.75 or FPR > 0.15 at 0.90 catch.
  - **H2 (n=32 granularity):** per-EXPERT tombstone on the legonet pool is confirmed-broken
    if retain-FPR ≥ 0.5 at catch ≈ 0.8 (priors: 0.727 retain shift / 0.815 orig_top1);
    per-AUTHOR sentinel tombstone is usable if catch ≥ 0.75 at FPR ≤ 0.10, refuted if
    FPR > 0.25.
  - **H3 (serving triple):** tombstone routes ≥ (H1 catch − 0.05) of forget queries to
    base+scaffold (route stats, not fq) at retain Δmu ≤ 0.005 vs sibling; embed-full
    baseline mu ≥ 0.7509 − 0.02 (else the embed arm is router-accuracy-confounded and the
    deletion cells re-baseline). Side-prediction: fq(sibling) > fq(tombstone).
  - **H4 (Mode-B, mediated):** ρ_embed_tombstone = (1 − c_probe)·ρ_embed_sibling ± 0.1 per R,
    with c_probe measured in Phase 1 (routing-only). Branch A (c_probe ≥ 0.9): CONFIRM
    collapse if ρ ≤ 0.10 at R=8 (sibling prior 0.833). Branch B (c_probe ≤ 0.5): CONFIRM the
    scope-limit "identity tombstones seal ownership-shaped, not fact-shaped queries" if
    ρ ≥ 0.5·ρ_sibling at R=8. Mediation model itself falsified if |ρ − prediction| > 0.15.
  - **H5 (content):** unplanted world = misinformation-not-disclosure if median
    ROUGE-L(sibling-gen, deleted gold) ≤ base-arm floor + 0.05 with ceiling − floor ≥ 0.3
    (measurement live); report the confabulation rate (vs_basegen < 0.5 AND vs_gold < 0.5)
    and the cross-author axis (vs the sibling shard's nearest-question gold). Planted world,
    host-hit probes only: disclosure if ROUGE ≥ 0.7 at R8-verbatim.
  - **H6 (anchored content attenuation — CONTINGENT, Phase 3):** rescoped per the 07-16
    refutation ([H-anchor-1 ✗](../merge_mechanism/2026-07-16_negative-anchor-pilot-results.md)
    — selectivity is closed, NOT re-tested): λ=1 anchoring attenuates sibling-leak CONTENT —
    sibling-served forget answer-prob −30% relative vs λ=0 AND/OR confabulation rate −0.1,
    with own-author mu within 0.01 and own-rouge ≥ 0.98; λ=1e-8 placebo (controls the
    anchor-batch RNG confound) within noise of λ=0, else the arm is invalid. Routing stats
    must be bit-identical across pools (centroids never see expert weights).
  - **H7 (deletion-count dial):** sibling capture + retain shift monotone in deleted-author
    count (forget01/05/10 on n=32; shard dial {9},{9,8},{9,8,7,6} on k=10 is a mass-deletion
    stressor, labeled as such); per-author tombstone retain-FPR grows ≈ linearly, slope
    ≤ 0.01/author (usable-at-small-deletions bar: forget01 catch ≥ 0.8 at FPR ≤ 0.02);
    k=10 sim-ratio stays ≥ 0.95 under 1/2/4 dropped shards. REFUTE: non-monotone outside
    bootstrap CI, or FPR slope > 0.03/author.
  - **H8 (registry + disclosure):** (a) name coverage on original questions ≈ 0.86 ± 0.02
    (consistency with `routed_key_exact`); paraphrase + name-stripped coverage measured (no
    prior). (b) hybrid router (registry-first → embed-fallback + author-tombstone): orphan
    leak ≤ 0.05 on name-bearing queries, retain paraphrase serve-rate ≥ embed-baseline
    − 0.05, mu within 0.005 of key-route; REFUTE if name-free leak > 0.15 of orphan traffic
    (reported as the irreducible surface of lexical identity routing). (c) deletion-
    disclosure (Streisand): AUC(forget10 vs never-trained holdout10 by served routing
    behavior) ≥ 0.75 = "tombstones disclose deletion" — a cost line, not a bug.
- **Setup:** plan `~/.claude/plans/can-you-expand-on-moonlit-kahan.md` (approved; design
  pressure-tested — G1–G10 corrections folded in). Seed 42 throughout. Interpreter
  `/home/jack/anaconda3/envs/test-env/bin/python`, work dir `~/tofu_sisa_lora`.
  **New/changed code** (sha256-12): `routing_audit_tofu.py` a6aaaf68edd6 (`tombstone`
  policy family — expert/author/name provenance rungs; `--dump_sims` per-query sidecar;
  `--centroid_mode` k=10 MiniLM audit + Mode-B c_probe + holdout10 disclosure AUC),
  `eval_routed_scaffold.py` 8a97b02006f8 (`build_shard_centroids` shared builder;
  `EmbedRoutedModel` with sibling|tombstone policies; `--embed_route`; labels
  `embedrouted_*`), `eval_entangled_probe.py` 308242097245 (`--embed_policy tombstone`,
  `--dump_generations`), `analyze_router_leak.py` 54b18d3e7c0b (roc/coverage/table),
  `aggregate_rho.py` 15ab150ad423, `dump_generations_routed.py` 5cc0c5e67bdb (R3 3-arm ×
  3-axis content audit), `submit_router_leak.sh` 172aa99aacef, `test_router_leak.py`
  fac3301cebaf; per-tag configs `configs/ramole_tofu_1b_basepin_{f01,f05}.json`.
  **CPU gates green before submission:** `test_router_leak.py` (6/6 ALL OK),
  `test_routing_audit_tofu.py`, `test_routed_scaffold_merged.py` (5/5),
  `test_entangled_facts.py` — all pass; `STUB=1` previews of phase1 + phase2smoke checked.
  **Artifacts reused (on disk, verified):** `_experts_scaf_k10` (strong experts, mu 0.7509
  arm), `_scaffolded_alpaca2k` (baked base), `_entangled_k10` + `plant_manifest.json`
  (Mode-B world), legonet n=32 pool + 4 encoder-index caches (stale bytes sha-asserted,
  never overwritten — new outputs go to distinct `rl_*` files).
  **Run matrix:** Phase 1 routing-only (5 × 1-GPU jobs ≤ 2 h: basepin/FT n=32 audits with
  tombstone+dump; k=10 centroid audit with probe manifest + holdout; f01/f05 dial with
  in-job `--plan` manifests) → CPU `collect`. **Submitted 2026-07-18: jobs 445344
  (rl-aud-basepin) / 445345 (rl-aud-ft) / 445346 (rl-centroid) / 445347 (rl-aud-f01) /
  445348 (rl-aud-f05, chained afterany:445347 so ≤4 of ours are ever concurrent), all
  `--dependency=afterany:445308-445311` (the memadapt queue, already drained at submit
  time). GATE: H1/H2 verdicts + c_probe select the H4
  branch BEFORE Phase 2. Phase 2 smoke (3 × 6-min serving cells) → extended triple +
  Mode-B tombstone worlds + R3 content audit. Phase 3 (contingent on H3/H5 outcomes):
  anchored k=10 pools λ∈{1e-8, 1} + seeds 43/44 on confirmation. ⚠ 4-GPU global cap:
  memadapt 445308–445311 PENDING at plan time — submissions chain `DEP=<jobid>`
  `--dependency=afterany` or wait for a drained queue; `squeue` checked before every
  submission.
- **Results:** *(pending — this entry is the pre-registration; results land in a NEW dated
  entry per the append-only protocol)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** Design notes fixed before running: (i) `rebuilt` ≡ `sibling` on the
  k=10 pool (forget authors live only in shard 9) — the triple is full/sibling/tombstone,
  no rebuilt cell; (ii) the prior §9-D numbers (sim-ratio 0.980, 72.7% shift) are about the
  n=32 instructor-xl index — the k=10 centroid audit is what unifies audit and serving
  routers; (iii) the per-author sentinel is forget-data-derived — the name-embedding rung
  is the privacy-clean end of the provenance ladder and the registry (H8) its lexical
  limit; (iv) `sims.npz` sidecars exist precisely because the 07-07 abstain arm discarded
  per-query scores and every new threshold variant would otherwise re-run the encoder.
- **New questions / new hypotheses:** *(pending results)*
- **Next Steps:** submit phase1 (chained behind the memadapt queue), run `collect`, write
  the Phase-1 results entry + H1/H2/c_probe verdicts, then branch Phase 2 per the gate.
