# Target Date: 2026-07-21 (P1 smoke green · P2 pilot: OOM → bs16×ga2 → lr tradeoff → G2 ADJUDICATE)

- **Hypotheses / what we're testing:** H1 (localization, make-or-break) via the P2 lean
  pilot pre-registered in [2026-07-20_specv2-preregistration.md](2026-07-20_specv2-preregistration.md):
  CONFIRM = some arm reaches median on/off selectivity ≥ 5.0 with median own-author
  answer-prob (all-active) ≥ 0.80; REFUTE = no arm ≥ 2.0. Secondary: the pilot-level H2
  signal (all-active vs own-only own-prob gap).
- **Setup:**
  - **Provenance correction to the 07-20 pre-reg (append-only):** `bank_layer.py`,
    `train_sepmlp.py`, `measure_selectivity.py` were edited 22:50–22:52 on 07-20 (tail of
    the parallel build sessions) *after* the pre-reg entry pinned their shas but *before*
    any GPU job executed. Every GPU result below ran on the CURRENT code. Corrected pins
    (sha256 first 16): bank_layer `421a05f5ba1cb5fe` · train_sepmlp `9b4d32397e219bf4` ·
    measure_selectivity `113010ee307dddf4`. All other 07-20 pins unchanged. CPU gate suite
    re-run on current code 2026-07-21: **70 passed, 1 skipped** (the skip = the
    SLURM-gated GPU test).
  - **P1 smoke (job 446535, ran 05:07–05:09):** `configs/smoke.json`
    (sha `7f94b32e5deececc`), K=200 bank, 2 authors, 5 steps, bs2, seed 42.
  - **P2 attempt 1 (job 446705):** 3-arm array `0-2%1`, bs32×ga1 — all arms OOM (below).
  - **P2 attempt 2 (job 446714, tasks logged as 446715_0–2):** same arms at the
    pre-declared fallback **bs16×ga2** (effective batch 32 unchanged; penalty
    ga-invariance already handled in `compute_loss`). Config shas: lr3e-4
    `3105d9fc8cfd13ed` · lr1e-3 `95ff134ead7316ff` · lr3e-3 `87f3f473f598c8c2`.
    K=20 authors, 15 epochs, seed 42, spec-v2 recipe exactly; per arm:
    `train_sepmlp.py --config … && measure_selectivity.py --recall_probe`.
  - **G2 ADJUDICATE bridging arm (job 446732, running at entry time):**
    `configs/pilot_relu_lr5e-4.json` (sha `f4d6b9d59d62af6d`), new driver verb `pilot1`
    (single named arm, body identical to a pilot array task).
  - Cluster note: `sacct` returns empty on this cluster — job monitoring must use
    `squeue` presence + output files.
- **Results:**
  - **Smoke:** grad-structure checks OK ×16 layers (l1-only / l2l3-only / l4-only;
    promotion confined to own gate/bias rows); loss components live (hinge 6.35→4.98 over
    5 calls, gram > 0, lm 0.34–5.71); telemetry own_norm 0.0591 vs off_norm 0.0040,
    ood_over_own 0.077; peak mem 14.04 GiB (at bs2); save→reload forward parity PASS.
  - **OOM (attempt 1):** all 3 arms crashed at step 9/390 — 38.67 GiB allocated + 5.78 GiB
    request > 44.46 GiB card. The smoke's 14.04 GiB was measured at bs2 and did not
    validate bs32.
  - **Pilot (attempt 2), per arm — median on/off selectivity / median own-prob
    (all-active) / min own-prob / all-active−own-only gap (means):**
    - lr 3e-4: **4.38** / **0.981** / 0.957 / +0.824 — verdict INTERMEDIATE, frac≥5 = 0.25, frac<2 = 0.00
    - lr 1e-3: **38.61** / **0.778** / 0.530 / +0.179 — verdict SELECTIVE, frac≥5 = 1.00
    - lr 3e-3: **1909.7** / **0.695** / 0.336 / +0.0005 — verdict SELECTIVE, frac≥5 = 1.00
    - OOD leakage (ood_norm/on_norm) falls with lr: alpaca 0.027 → 0.004 → 0.0002.
    - No NaNs; loss curves live throughout (e.g. lr3e-4 final: lm 0.081, hinge 0.73).
- **What worked / hypothesis verdict:**
  - **H1 REFUTE bar decisively cleared** — two arms at median selectivity 38.6 and 1909.7
    vs the LoRA negative-anchor ceiling of 1.11 (100% LAZY): trained self-gating IS
    achievable in disconnected ReLU-gated branches. H1 CONFIRM (joint bar in a single
    arm) still open: no arm passes both (0.981 with 4.38; 0.778 with 38.6).
  - **G2 = ADJUDICATE** per the pre-registered clause (best arm in [2,5) — lr3e-4 at
    4.38 — has own-prob 0.981 ≥ 0.80) → exactly one bridging config before deciding:
    **lr 5e-4** (geometric midpoint; log-interpolation projects sel ≈ 11, prob ≈ 0.90).
    No K=200 spend before its result.
  - **H2 pilot signal: no interference regression** — all-active is never worse than
    own-only (memsinks REFUTE direction not observed at any lr).
- **Observations:**
  - Clean monotone lr tradeoff: suppression strength (selectivity, OOD silence) rises
    with lr while own-author recall falls — recall 0.981 → 0.778 → 0.695.
  - The all-active−own-only gap **inverts the expected direction** at low lr: +0.824 at
    3e-4 means own-branch-alone recalls almost nothing (0.155 mean) and recall is carried
    by the *collective* bank — off-branches are quiet in norm (ratio 4.4) yet
    load-bearing in function. At 3e-3 the gap is +0.0005: branches fully self-contained.
    Deletion-relevant: at low lr, dropping *other* authors' branches would likely damage
    a surviving author's recall (co-adaptation through the shared forward VALUE of the
    detach trick); at high lr the bank behaves as intended. The bridging arm's gap is
    therefore as decision-relevant as its two gate numbers.
  - Silent-failure checks: no NaN/frozen loss/empty generations; probe medians consistent
    with train-time telemetry (lr3e-4 telemetry median 3.22 on train batches vs 4.38 on
    question prompts — different token populations, same order).
- **New questions / new hypotheses:**
  - Is there an lr with selectivity ≥ 5 AND prob ≥ 0.80 AND a small all-active−own-only
    gap — or does gap ≈ 0 only arrive after recall has fallen below the bar? (The
    bridging arm answers the first clause; the gap value tells us the second.)
  - Does the co-adaptation at low lr predict measurable retain-author collateral under
    single-author deletion? (Testable at P4 via `sepmlp_unlearned` retain metrics.)
  - Why did the OOM fire at step 9 and not step 1 (probable cause: first pure-negative
    Alpaca batch — longer padded sequences than TOFU author batches)?
- **Next Steps:** read the bridging arm (job 446732) → final G2 verdict; if GO,
  `sepmlp_1b_k200.json` gets the winning lr (+ bs16×ga2 carried over) → P3 K=200 train +
  probe200 → G3 → P4 OU evals vs Vincent's priors (0.97→0.32 deleted, ≤0.002 others,
  utility Δ≤0.001) → P5 relearn battery.
