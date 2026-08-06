### Target Date: 2026-07-20 (All-router sweep — build hardened + J1–J6 launched; SepMLP spec-v2 joins as the routerless arm)
- **Hypotheses / what we're testing:** unchanged from the same-day
  [pre-registration](2026-07-20_all-router-sweep-preregistration.md) (H-ARCH / H-DIAL /
  H-POOL / H-TRAINED / H-DATASET / H-ENC / H-SEAL-GEN). This entry records the build,
  the review fixes, TWO deviations from the pre-registration (both scope-preserving,
  documented below before any results exist), and the launch provenance. New rider: the
  user supplied the authoritative spec for the **routerless gated-branch method** (SepMLP
  spec v2 — frozen base + per-author ReLU-gated branches, 4-term loss L1 + 10·L2 hinge
  m=2 + 50·L3 output-norm + 1·L4 promotion, per-author clip, cosine LR, detector init
  from author-question hiddens; deletion = drop the author's slices) — it joins the sweep
  as the "selection inside the weights" row, with its own thread pre-registration
  ([sepmlp spec-v2](../sepmlp/2026-07-20_specv2-preregistration.md)); external priors to
  verify: deleted 0.97→0.32, others ≤0.002, utility Δ0.001, no relearn residue.
- **Setup:** plan `~/.claude/plans/include-these-details-as-mellow-barto.md` (approved;
  user decisions: never-trained-author negatives = real_authors + alpaca with holdout10
  fully out of training; lean SepMLP pilot = spec recipe × lr {3e-4,1e-3,3e-3}).
  Seed 42. **Code (sha256-12):** `router_family_audit.py` 8bd440da1fc7 (built by the
  first wave, adversarially verified clean — score-matrix argmax ≡ router.route()
  self-check 50/query default, per-candidate-set logit_div recompute, pad-masked
  per-sample norms, question-only sentinels) · `analyze_router_family.py` 0c7d4aa8ed9d
  (built + 6 review findings fixed: H-ARCH/H-TRAINED PENDING semantics, None-safe seed
  AUCs, H-ENC confidence half via `--enc_roc_json`, H-DIAL csq ≥0.95 sub-bar, duplicate
  mtime-preference WARN, filtered-vs-unfiltered ratio documentation; self_test 15/15) ·
  `test_router_family.py` 995f69a910a7 · `submit_router_family.sh` 3d210d1ade58 (4 review
  fixes: real k=200 pool, serial `all` lane, key_exact in J3, J5 naming; collect now runs
  enc-roc before the analyzer with the CORRECT analyzer CLI — the original call didn't
  match the argparse and would have died under `|| true`) · `analyze_router_tofu.py`
  64fa57d2024e (`--dropped` H-TRAINED audit) · `ramole/routing_audit.py` 0c9ded62a7b9
  (dropped/abstain/`--dump_sims`; default byte-identity regression-locked) · SepMLP spec
  v2: `bank_layer.py` 421a05f5ba1c / `train_sepmlp.py` 9b4d32397e21 /
  `measure_selectivity.py` 113010ee307d (post-review fixes this session: GPU placement
  before the detector pre-pass — the walltime-killer blocker; fp32 loss math under bf16
  autocast via a local autocast-disable; ood_alpaca probe drawn beyond the 8000-row
  trained head so it stays never-seen). **CPU gates all green before submission:**
  test_router_family / test_routing_audit_tofu / test_router_leak / analyzer self_test
  15/15 / ramole test_routing_audit + test_alpha_capture / sepmlp pytest 70 passed
  (1 GPU-gated skip). STUB=1 previews checked for every stage.
  **Launched (serial lane, ≤1 GPU from this thread at any instant):** J1 **446563**
  rf-j1-k10feat → J2 **446564** rf-j2-k10behav → J3 **446565** rf-j3-k200feat → J4
  **446566** rf-j4-routerlora → J5 **446567** rf-j5-dbpedia → J6 **446568**
  rf-j6-encoders (each afterany its predecessor; J1 running at submit, total live GPUs 2
  of 4 — the ctv lin lane holds the other). SepMLP smoke **446535** (queued by the
  concurrent sepmlp session, chained afterany:446368) will execute the FIXED training
  path; pilot submission waits for the manual G1/G2 gate read per the sepmlp gate ladder
  (never pre-chain across a gate).
- **Results:** *(pending — results land in a new dated entry)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations — deviations from the pre-registration, declared before results:**
  (i) **J3 pool correction:** the pre-registration assumed a Llama-3.2-1B k=200 pool;
  no such pool exists on /storage2 — J3 runs on the REAL per-author e25 pool
  `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` (base = Llama-2-7B-chat-hf, used only as the
  centroid_lm encoder; J3 is feature-space-only so no adapters load). H-POOL's
  granularity claim is unaffected; the base-model change is a confound only for
  cross-pool centroid_lm comparisons, which we will read within-pool.
  (ii) **DBpedia dropped policy has NO per-query fallback** (hard-raise if survivors < k),
  unlike the TOFU RouterLoRA arm's top-k-survivor fallback — an intentional contrast the
  analyzer footnotes. (iii) A pre-existing queue-cap violation (five dependency-free %1
  GPU arrays = worst-case 5 > 4) was repaired before any submission by chaining 446366
  afterany:446365. (iv) Builder-session postmortem: the first build wave lost two of
  three builders to session limits mid-report — one had already written all its files
  (recovered + adversarially reviewed), one logged a false completion entry in the
  scratchpad for a file that never existed (corrected in the scratchpad; the analyzer was
  rebuilt from scratch). (v) real_authors now serves as BOTH a SepMLP training-negative
  source and an OOD probe/eval split — its sepmlp ood rows measure trained suppression,
  not generalization (world_facts + offset-alpaca are the clean OOD rows); footnoted in
  the analyzer.
- **New questions / new hypotheses:** *(pending results)*
- **Next Steps:** monitor the lane; on J1–J6 completion run `submit_router_family.sh
  collect` (login-safe) → results entry with per-hypothesis verdicts + the unified
  all-router leak table. SepMLP: read smoke 446535 telemetry (loss sane, suppression
  nonzero, save→reload parity, peak-mem go/no-go) → submit the lean pilot → G2 winner →
  K=200 → leakprobe (reference + forget10) + OU evals + relearn → sepmlp rows join the
  table.
