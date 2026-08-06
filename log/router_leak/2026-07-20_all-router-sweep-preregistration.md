### Target Date: 2026-07-20 (All-router leakage sweep — pre-registration: is the drop-leak architectural?)
- **Hypotheses / what we're testing:** Every leak number so far ([§9-D](../ramole/2026-07-06_routing-audit-results.md),
  [07-07 abstain refutation](../ramole/2026-07-07_routing-fix-arms.md),
  [Phase 1/2](2026-07-18_phase1-results.md)) was measured on the **embedding router family
  only** (instructor-xl n=32, MiniLM k=10 centroid). This campaign runs the same
  drop-an-expert leakage protocol on **every router-based method in the repo** — lexical
  (exact key, TF-IDF), embedding with new encoders (LM hidden states, mpnet, bge),
  adapter-behavioral (perplexity, activation-norm, attn-norm, logit-divergence), trained
  (RouterLoRA, 3 seeds), retriever (DBpedia RAG route), and the oracle/identity controls —
  across pool-size and deletion-count configs. The claim under test: **the post-deletion
  sibling leak and the confidence-threshold inseparability are properties of score-based
  argmax routing itself, not artifacts of one encoder.**
  Unified metrics per router × drop set: (1) orphan fate (top-1/top-3 surviving-expert
  capture, route entropy); (2) **sibling-adequacy ratio** (family-specific, 1.0 = the
  surviving sibling scores the orphan as well as the deleted expert did: cosine routers =
  masked/unmasked top-1 cos; ppl = unmasked-top1-loss / masked-top1-loss; norm/div routers
  = masked/unmasked top-1 score); (3) retain collateral (top-1 route-shift rate); (4)
  separability (ROC-AUC + retain-FPR@90%-orphan-catch of the router's native confidence
  family — top-1 score / top1−top2 margin / per-shard z — author-parity calibrate/eval
  split as in [analyze_router_leak](../../tofu_sisa_lora/analyze_router_leak.py)).
  - **H-ARCH (architectural universality):** on the k=10 scaffold pool, drop shard 9: every
    score-based router family in {key_tfidf, centroid_sbert, centroid_sbert_q, centroid_lm,
    centroid_lm_last, ppl, activation_norm, attn_norm, logit_div} shows (a) top-3 surviving
    capture ≥ 0.5, (b) sibling-adequacy ≥ 0.9, (c) confidence AUC ≤ 0.75 with FPR ≥ 0.3 at
    90% catch. CONFIRM if ≥ 7/9 families meet all three (leak is architectural). REFUTE if
    ≥ 2 families reach AUC ≥ 0.90 at FPR ≤ 0.10 — those families' confidence WOULD support
    an abstain fix, falsifying architecture-generality. Identity controls must contrast:
    oracle (orphans → base+scaffold P=1.0, retain shift ≡ 0, by construction — asserted) and
    key_exact (post-drop orphans fall to the no-match fallback = shard 0 — a *design* leak
    to a wrong expert, but with a perfect native detector: predict orphan no-match rate
    ≥ 0.85 (registry prior 0.895) at retain no-match ≈ 0.10–0.15 (name-free slice ≈ 0.14
    prior) — i.e. the identity signal yields the usable operating point that similarity
    confidence provably lacks).
  - **H-DIAL (k=10 deletion dial — the deferred H7 half, now for every family):** capture /
    retain shift / adequacy are monotone non-decreasing in dropped-shard count over {9},
    {9,8}, {9,8,7,6} for all leaky families; centroid_sbert_q sim-ratio stays ≥ 0.95 under
    1/2/4 drops. REFUTE: non-monotone outside bootstrap 95% CI.
  - **H-POOL (granularity dial, k=200 per-author experts):** with one expert per author
    (the finest possible routing granularity), dropping author 199 (and the 180–199 mass
    cell) still yields sibling-adequacy ≥ 0.9 and confidence AUC ≤ 0.75 for key_tfidf /
    centroid_sbert / centroid_lm — i.e. finer granularity does NOT create separability.
    REFUTE if any reaches AUC ≥ 0.90 at FPR ≤ 0.10: granularity would be a leak fix, a
    positive finding worth its own follow-up. (Genuinely open — pre-registering both
    directions.)
  - **H-TRAINED (RouterLoRA drop-audit, never run — the entangled_facts 07-07 open
    question):** masking the forget-affected experts out of the active set at serve time
    (softmax renormalizes over survivors), the trained router's orphan alpha concentrates
    on siblings like the retrieval stage does: orphan post-drop normalized entropy H_norm
    and max-share distributions overlap retain's (AUC ≤ 0.75 for both detectors), and the
    orphan top-1 surviving-expert alpha share ≥ 0.9 × its pre-drop top-1 share. CONFIRM =
    the learned gate is leak-blind too. REFUTE if AUC ≥ 0.90 (the trained gate detects
    orphanhood — a learned-confidence fix candidate). Must hold on all 3 router seeds
    (42/43/44 ckpts) — the first multi-seed leakage cell in the thread.
  - **H-DATASET (DBpedia retriever, cross-dataset):** porting the dropped policy to
    [ramole/routing_audit.py](../../ramole/routing_audit.py) (tags d0/d1/d2 pooled +
    d_batch15): orphan top-3 surviving capture ≥ 0.5, sibling-adequacy ≥ 0.9, and the
    abstain sweep finds no τ with orphan-abstain ≥ 0.9 at retain false-abstain ≤ 0.1.
    REFUTE: a separating τ exists — the TOFU overlap would be dataset-specific.
  - **H-ENC (encoder generality):** k=10 centroid audits under all-mpnet-base-v2 and
    BAAI/bge-small-en-v1.5 reproduce the MiniLM structure: sibling sim-ratio ≥ 0.95,
    confidence AUC ≤ 0.75. With MiniLM (0.971) and instructor-xl base+FT (0.980/0.768 on
    n=32) this makes ≥ 4 sentence encoders + the LM-hidden-state space (J1) — REFUTE if
    any new encoder separates (AUC ≥ 0.9).
  - **H-SEAL-GEN (rider — is the *fix* architectural too?):** the per-author sentinel
    tombstone rung, rebuilt natively in each feature space (TF-IDF vectors, SBERT-Q+A,
    LM-hidden), reaches orphan catch ≥ 0.90 at retain-FPR ≤ 0.10 in ≥ 3/4 feature spaces
    on the k=10 pool (MiniLM prior: 0.963 catch / 0.091 argmax FPR). REFUTE: the identity
    seal is encoder-specific.
- **Setup:** seed 42 throughout (+ RouterLoRA ckpt seeds 43/44); bootstrap 95% CIs (1000
  resamples over queries) on headline ratios. Interpreter
  `/home/jack/anaconda3/envs/test-env/bin/python`; work dir `~/tofu_sisa_lora`.
  Pools (all on disk, verified 2026-07-20): `_experts_scaf_k10` (k=10 strong experts +
  `_scaffolded_alpaca2k` base — behavioral routers serve these), `_k200_r32_e25_lr1e4`
  (per-author pool; feature-space strategies only — the k=200×r32 eval memory law forbids
  loading 200 adapters, and feature-space routing needs none), legonet n=32 pool + RouterLoRA
  `router{,_s43,_s44}.safetensors`, DBpedia `ramole_l32_3b_n32_k3` + manifests d0-d2/d_batch15.
  **New code (sha256 recorded in the results entry):** `router_family_audit.py` (score-matrix
  audit over router.py strategies; policies full + arbitrary drop sets; family npz sidecar),
  `analyze_router_family.py` (ROC/abstain/bootstrap/table, CPU), `test_router_family.py`
  (CPU gate), `submit_router_family.sh` (STUB/DEP/self-skip driver);
  `analyze_router_tofu.py` gains `--dropped`/`--router_ckpt`; `ramole/routing_audit.py`
  gains dropped/abstain/`--dump_sims`. Existing-file edits are additive; existing tests must
  stay green (`test_routing_audit_tofu.py`, `ramole/tests/test_routing_audit.py`,
  `test_router_leak.py`).
  **Run matrix:** J1 k=10 feature-space (all 4000 questions) · J2 k=10 behavioral (400
  forget + RandomState(42) 400-retain sample, matching the [analyze_router_tofu
  convention](../../tofu_sisa_lora/analyze_router_tofu.py)) · J3 k=200 feature-space ·
  J4 RouterLoRA drop-audit ×3 seeds · J5 DBpedia retriever · J6 k=10 encoder sweep —
  six 1-GPU SLURM jobs + CPU collect. ⚠ 4-GPU global cap: queue holds five dependency-free
  %1 arrays (446357/65/66/67/70 — worst case 5 > 4, a pre-existing violation): repair by
  chaining 446370 afterany:446367, then submit J1–J6 afterany the chain tails (446374,
  446369) with internal dependencies so ≤ 4 of ours are ever eligible. Outputs → new
  `rl_family_*` / `rl_enc_*` / `rl_routerlora_*` files; nothing existing overwritten.
  **Known-value continuity gates:** centroid_sbert_q drop{9} must reproduce
  `rl_centroid_k10` (sim-ratio 0.971 ± 0.01, retain shift 0.0583 ± 0.01); key_exact
  full-pool routing accuracy ≈ 0.86 (CLAUDE.md prior); n=32 numbers are NOT re-run.
- **Results:** *(pending — land in a new dated entry per the append-only protocol)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** Design notes fixed before running: (i) router.py's `centroid_sbert`
  averages Q+A embeddings while the serving router centroids are question-only — both are
  audited (`centroid_sbert` vs `centroid_sbert_q`), the latter as the continuity anchor;
  (ii) behavioral-router score matrices are computed per-shard-batched (set adapter once,
  batch all queries) with per-sample scores — `_lora_b_norm`'s batch-summed scalar is
  replaced by per-sample norms in the audit path; (iii) logit_div's "divergence from the
  candidate mean" changes when the candidate set shrinks — the post-drop score matrix is
  recomputed over survivors, not masked (matching serving semantics); (iv) key_exact has
  no graded score — its detectors are the binary no-match flag (pre-registered operating
  point above), and it is excluded from AUC-family aggregation; (v) S3T/SISA ensembles,
  prefix-concat, and the oracle-mask arms (SIFT/ClAMU/MemSinks/SEA-proxy) have no
  query-dependent routing signal to leak — recorded as boundary rows from existing
  results, no new runs.
- **New questions / new hypotheses:** *(pending results)*
- **Next Steps:** build + CPU-gate the four code pieces → STUB=1 preview → queue repair →
  submit J1–J6 chained → collect → results entry with per-hypothesis verdicts + unified
  all-router leak table (the §9-D table, completed).
