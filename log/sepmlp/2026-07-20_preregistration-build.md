### Target Date: 2026-07-20 (Pre-registration + P0 build)
- **Hypotheses / what we're testing:** Pre-registered BEFORE any SLURM job (plan:
  `../../.claude/plans/vincent-hanke-3-45-pm-eager-coral.md`, copied verbatim):
  - **H1 localization (make-or-break):** disconnected MLPs + in-domain negatives reach
    SELECTIVE (median on/off ≥5; LoRA anchor: 1.11, 100% LAZY) with own-prob ≥0.80 and
    ≥0.90× λ=0. REFUTE: <2 everywhere without >20% recall loss.
  - **H2 all-active serving retains utility (anti-memsinks):** all-active ≈ own-only
    (gap ≤0.05); OU Util.R ≥0.95, Util.G ≥0.95 (MemAdapt FT row 1.075/1.024). REFUTE:
    ≥0.15 all-active drop.
  - **H3 deletion clean:** drop forget10 ⇒ Mem ∈[0.55,0.70] (Retrained 0.590, MemAdapt
    0.630), |ΔUtil.R| ≤0.03, Agg ≥0.80 (strong-confirm 0.84–0.90 vs MemAdapt 0.869);
    dropall ≡ base; deletion wall-time O(ms–s).
  - **H4 relearn parity (Vincent's observation):** median steps-to-prob-0.8 ratio
    target/control ∈[0.8,1.25] and fixed-budget |Δprob| ≤0.10. REFUTE: target ≥2× faster.
  - **H5 negative-example leak (measurement, no bar):** Priv and 4 raw MIA AUCs on
    sepmlp_unlearned; direction attributed (residual memorization vs over-suppression).
    Anchors: MemAdapt Priv 0.917; exact-drop threads ≤ oracle floor AUC 0.379.
  - **H-scale:** K=20 → K=200 selectivity transfer (≥5 and ≥0.7× pilot).
- **Setup:** **Frozen defaults** (all in version-controlled configs, none tuned post-hoc):
  base `meta-llama/Llama-3.2-1B-Instruct` frozen; per-author SwiGLU bottleneck width 32 at
  ALL 16 layers, grouped `W_gate/W_up (K·32, 2048)`, `W_down (2048, K·32)`; init
  `W_down = 0` (exact no-op at step 0) + `W_gate/W_up ~ N(0, 1/√2048)`, sha-seeded per
  (layer, tensor) via `seeded_generator("sepmlp", init_seed, layer, name)`; fp32 masters +
  bf16 autocast. Recipe: 15 epochs, bs 32, ga 1, adamw_torch, constant LR, warmup 0,
  weight_decay 0, clip 1.0, seed 42. Suppression: `output_gram` form — exact per-author
  output norm on off-author tokens via the Gram trick, mean-normalized over off-author
  entries × non-pad tokens × layers (λ K- and depth-independent); negatives = other
  in-batch TOFU authors. Data: TOFU `full` (200×20), OU chat-template schema via the
  imported memadapt [`data_tofu.py`](../../memadapt_tofu/data_tofu.py) — never the plain
  Question:/Answer: track. Hardware: 1 GPU (48G, sprint1–3) per job.
  **Pilot grid — 9 arms** (K=20, [`configs/pilot_*.json`](../../sepmlp_tofu/configs/),
  all sharing the recipe above; output `/storage2/jack/checkpoints/sepmlp_tofu/pilot/`):

  | arm | config | lr | λ (suppress) | penalty form | negatives |
  |---|---|---|---|---|---|
  | 0 | pilot_0.json | 1e-3 | 0.0 | output_gram | — (λ=0 control) |
  | 1 | pilot_1.json | 1e-3 | 0.3 | output_gram | in-batch TOFU |
  | 2 | pilot_2.json | 1e-3 | 1.0 | output_gram | in-batch TOFU |
  | 3 | pilot_3.json | 1e-3 | 3.0 | output_gram | in-batch TOFU |
  | 4 | pilot_4.json | 1e-3 | 10.0 | output_gram | in-batch TOFU |
  | 5 | pilot_5.json | 3e-4 | 1.0 | output_gram | in-batch TOFU |
  | 6 | pilot_6.json | 3e-3 | 1.0 | output_gram | in-batch TOFU |
  | 7 | pilot_7.json | 1e-3 | 1.0 | output_gram | in-batch TOFU + Alpaca OOD (n=2000, seed 42) |
  | 8 | pilot_8.json | 1e-3 | 1.0 | act_norm | in-batch TOFU |

  **Gate rules (verbatim from the plan, frozen now):** *Gate G2:* GO = pick (λ,lr)
  maximizing median on/off selectivity s.t. selectivity ≥5 AND own-author answer-prob
  ≥0.80 and ≥0.90× the λ=0 control; ADJUDICATE [2,5) with one bridging config; NO-GO
  (<2 without >20% recall loss) ⇒ H1 refuted → write refutation entry, stop before K=200
  spend. *Gate G3:* selectivity ≥5 and ≥0.7× pilot; all-active vs own-only own-prob gap
  ≤0.05 (the memsinks-interference tripwire, placed before eval spend). Gates are manual
  reads — never pre-chain a submission across a gate.
  **Planned phases / budget** (driver [`submit_sepmlp.sh`](../../sepmlp_tofu/submit_sepmlp.sh),
  arrays `%2`, global 4-GPU cap):

  | phase | job | est. GPU-h |
  |---|---|---|
  | P0 build + CPU gates | `pytest tests/ -q` (test-env) + STUB previews | 0 |
  | P1 smoke | full-size K=200 bank, 2 authors' data, ~5 steps; peak-mem go/no-go for bs32 (fallback bs16×ga2) | ~0.2 |
  | P2 K=20 pilot | 9-task array `0-8%2`: train 15ep → selectivity probe → recall probe (all-active AND own-only) | ~3 |
  | P3 K=200 + probe200 | winning config, seed 42 → G3 re-gate | ~3 |
  | P4 OU evals | `sepmlp_ft` / `sepmlp_unlearned` (forget10 droplist) / `sepmlp_dropall` (must ≡ calib_base) | ~1 |
  | P5 relearn battery | 24-task array `%2`: {sepmlp_unl, memadapt_unl} × (5 target + 5 holdout-control) + retrain-oracle × (2+2) | ~4–6 |
  | P6 wave-2 ablations | deferred behind its own pre-registration (width {4,8,16}, layer subsets, reseeds 43/44) | ~12–15 |

  Core ≈ 11–13 GPU-h; with wave 2 + reseeds ≈ 25–30 GPU-h. **Storage budget ≤6 GB**
  (K=200 ckpt 2.52 GB fp32; pilot ckpts deleted only after their eval JSONs land, under
  the human-approved deletion protocol, logged). **Scope guards (pre-committed):** never
  train on holdout10 (relearn control + MIA nonmembers, enforced by a CPU gate); OOD-firing
  trigger `ood/own > 0.1` ⇒ activate the Alpaca-negatives arm; no cap raises; no recursive
  deletion without approval; no git commits unasked; OU-track and plain-track numbers never
  share a table; claims discipline — deletion is exactly "the author's parameters are
  removed", NOT exact unlearning (surviving authors trained with forget rows as negatives).
  **File inventory** (sha256 at 2026-07-20 ~18:45, project
  [`sepmlp_tofu/`](../../sepmlp_tofu/)):

  | file | sha256 |
  |---|---|
  | sepmlp_common.py | `c7ef0fdac6acbd329fc99fb419eb0f5b231af52375c165d1b7c1b1c4c6f24434` |
  | bank_layer.py | `60ccae827efe1d09d01d56faa83c8ad759816f10e189098ddc049e040a0d0211` |
  | sepmlp_model.py | `9f5c4f4bf23be46420d688cf29b6a5b1629873a8ac145fdcd922a15dc531fb68` |
  | train_sepmlp.py | `0f6d5a598ae42fac76579b67b24c55b9ffb9ed86b8cd7fdcd3a230c2c32f248e` |
  | build_droplist.py | `0f154291e28a8476a6a78693a2cc60cafc50879f96796a3b92f1468fc5594c9b` |
  | measure_selectivity.py | `9f85b20fcda1b0c5fcb5098bc97cda9a61e8fdd486b515f85af8fc2b8379a75b` |
  | tests/test_selectivity.py | `f6d1f8e5ba75590962b550545c6e2c5ef47583e1d38473b6b1c65132cb355d38` |
  | tests/test_ou_load.py | `87ab02f5910fbd7eeb9d209654bc2b7450b8f9b80e48d95115a99f7e5f35fc05` |
  | tests/test_compose_fixture.py | `e9014a637dfb891724db56ba80cb95aba3f21825c5f510992cb4fbe8960e6ce4` |
  | ou_integration/sepmlp_registry.py | `3c2d15639bc90a67fd9b95c622f8f19eb761c6d521472c4624bbd98a3c1fb648` |

  The P0 build runs in parallel sessions; at inventory time the relearn harness
  (`relearn.py`, `relearn_score.py`, `collect_relearn.py`) and the remainder of the
  17-gate CPU test suite were still landing — their sha256s will be recorded in the G0/P1
  entry before any job that uses them. Core modules compile clean
  (`python -m py_compile`, test-env). **SLURM job ids: TBD at submit — this entry
  pre-registers BEFORE any job.**
- **Results:** pending (pre-registration — no runs yet, zero GPU spend).
- **What worked / hypothesis verdict:** pending (pre-registration).
- **Observations:** pending (pre-registration).
- **New questions / new hypotheses:** Does K=200 bs32 fit the 48G budget (plan estimate
  28–36 GB peak — P1 smoke's `max_memory_allocated` print decides bs32 vs the declared
  bs16×ga2 fallback)? Does the OOD-firing trigger (`ood/own > 0.1`) fire on the epoch
  telemetry, forcing the Alpaca-negatives arm into the main line? Does the `act_norm`
  surrogate (arm 8) track `output_gram` (arm 2) closely enough to serve as the cheaper
  penalty?
- **Next Steps:** (1) G0: run the CPU gate suite (`pytest tests/ -q`, test-env) once the
  parallel P0 sessions finish landing files; all green + `STUB=1` driver previews before
  any submission. (2) P1 smoke, submitted `DEP=afterany:<current ctv/scaffold queue
  tails>` (re-check `squeue -u jack` ids at submit; afterany, never afterok). (3) P2 K=20
  pilot array `0-8%2`, then the manual G2 gate read against the bars frozen above.
