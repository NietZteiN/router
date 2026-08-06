# Target Date: 2026-07-22 (pre-registration: H-wscale + H-k200-lr2 arms + P4-mechanics run — user-approved triple)

Second entry today; follows [2026-07-22_hk200lr-refuted.md](2026-07-22_hk200lr-refuted.md).
**Human decisions (2026-07-22, interactive):** all three of B (w-rescale — an approved
deviation from the pinned spec-v2 loss weights), A2 (lr 1.5e-4, in-recipe), and C
(P4 mechanics on the lr2e-4 checkpoint) selected; deletion of the lr5e-4 G3-FAIL
checkpoint weights approved and executed (`sepmlp_1b_k200_s42/sepmlp.pt` removed;
probe JSON/meta/logs kept; project 5.7 → 3.4 GB). This entry pre-registers all bars
BEFORE the jobs run.

- **Hypotheses / what we're testing:**
  - **H-wscale (arm B, recipe deviation):** dividing the suppression weights by the K
    ratio (w2/w3: 10/50 → 1/5 at K=200, lr 5e-4 unchanged) equalizes per-author
    suppression pressure and reproduces pilot-winner behavior. PREDICT: sel 5–15,
    median recall ≥ 0.90. PASS: sel ≥ 5 AND recall ≥ 0.80. REFUTE: recall < 0.75 or
    sel < 5. (Mechanism at stake: is "suppression pressure ∝ K" the whole story? If
    yes, ÷K restores the pilot point almost exactly.)
  - **H-k200-lr2 (arm A2, in-recipe):** lr 1.5e-4 lands sel 12–25 with recall
    0.80–0.87 (curve-parallel extrapolation from the two mapped K=200 points).
    PASS: sel ≥ 5 AND recall ≥ 0.80. REFUTE: recall < 0.75 or sel < 5.
  - **C (measurement, no pass/fail — P4 deletion mechanics on the lr2e-4 ckpt, median
    recall 0.747):** three OU evals with label prefix `sepmlp_lr2e4` (fresh labels/dirs
    per trap 9). Questions: (i) deletion wall-time (memadapt anchor 0.027 s);
    (ii) `_dropall` ≡ `calib_base`; (iii) forget10 drop → forget-metric movement;
    (iv) **ΔUtil.R collateral on retain authors — the H-gap question** (gap +0.45 at
    this ckpt; the human ruling makes P4 collateral the deciding measurement);
    (v) MIA block direction. Utility/Mem replication bars are KNOWN-MISSED at recall
    0.747 — C informs mechanics, not the replication row.
  - **Winner rule (pre-registered):** if both B and A2 pass, winner = higher median
    recall subject to sel ≥ 5; the winner's checkpoint becomes the P4 replication
    candidate (full fresh P4 with its own label prefix). C's numbers never mix into
    the replication table.
- **Setup (frozen before submission):**
  - Arm B: `configs/sepmlp_1b_k200_w15.json` (sha `9071404c26d439f7`) — K=200,
    lr 5e-4, w2=1 w3=5, bs8×ga4, 15 ep, seed 42 → `sepmlp_1b_k200_w15_s42/` + chained
    probe200.
  - Arm A2: `configs/sepmlp_1b_k200_lr1p5e-4.json` (sha `c7ca758b9994bf9f`) — K=200,
    lr 1.5e-4, spec weights 10/50, bs8×ga4 → `sepmlp_1b_k200_lr1p5e-4_s42/` + chained
    probe200.
  - C: driver `eval` verb parameterized (`eval [config] [run_dir] [prefix]` — new
    args; sha updated below) on `sepmlp_1b_k200_lr2e-4_s42` with prefix
    `sepmlp_lr2e4`; array `0-2%2`; droplists built inline (text-join mapping,
    bank_sha-pinned); explicit base-model override (trap 8) and OUR retain90
    reference (trap 10 §self-check caveat).
  - OU integration installed this session via `ALLOW_DIRTY=1 install_branch.sh`
    (additive, NOTHING committed — the sanctioned escape hatch; the deliberate
    fp32-logits fix stays uncommitted): `src/model/sepmlp_registry.py` +
    `configs/model/SepMlp-Llama-3.2-1B.yaml` + registry import appended.
  - CPU gates re-run after driver/config edits: **70 passed, 1 skipped.**
    Driver sha (post-edit): `submit_sepmlp.sh` recomputed at submit and recorded in
    the results entry with job ids.
  - **GPU cap:** three lanes — B train→probe (≤1), A2 train→probe (≤1), C array
    `0-2%2` (≤2) — worst-case concurrent = 4 = the global cap; nothing else queued
    (verified at each submit).
  - **Storage (disclosed):** +2×2.5 GB checkpoints ⇒ peak ~8.4 GB > 6 GB budget while
    both arms' weights coexist; cleanup decision (delete loser weights) goes to the
    human after verdicts.
- **Results:** pending (pre-registration).
- **What worked / hypothesis verdict:** pending.
- **Observations:** pending.
- **New questions / new hypotheses:** pending.
- **Next Steps:** submit B, A2, C; read H-wscale/H-k200-lr2 bars + C mechanics;
  winner → full P4 replication row vs Vincent's priors (0.97→0.32 deleted, ≤0.002
  others, utility Δ≤0.001); then P5 relearn battery.
