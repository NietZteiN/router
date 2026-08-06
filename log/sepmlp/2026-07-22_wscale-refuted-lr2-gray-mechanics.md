### Target Date: 2026-07-22 (H-wscale REFUTED · H-k200-lr2 GRAY 0.795 · C deletion-mechanics — the K=200 recall ceiling is structural)

Results for the triple pre-registered in
[2026-07-22_wscale-lr2-c-preregistration.md](2026-07-22_wscale-lr2-c-preregistration.md).
Closes the "reproduce and make sure it works" arc: **H1 reproduces at K=20; the method
does NOT reach Vincent's regime at K=200, and the failure is structural (robust to both
the lr dial and the suppression-weight rescale), not a tuning miss.**

- **Hypotheses / what we're testing:** H-wscale (arm B): w2/w3 ÷K-ratio (10/50→1/5) at
  lr 5e-4 restores pilot behavior — predict sel 5–15, recall ≥0.90; PASS sel≥5 ∧
  recall≥0.80, REFUTE recall<0.75 ∨ sel<5. H-k200-lr2 (arm A2): lr 1.5e-4 → sel 12–25,
  recall 0.80–0.87; same PASS/REFUTE bars. C: P4 deletion-mechanics on the lr2e-4
  (recall 0.747) checkpoint — measurement, no bar; the ΔUtil.R clause answers H-gap.
- **Setup:** B train 447175 → probe 447176 (`sepmlp_1b_k200_w15.json`,
  sha 9071404c26d439f7); A2 train 447177 → probe 447178
  (`sepmlp_1b_k200_lr1p5e-4.json`, sha c7ca758b9994bf9f); C eval array 447179 %2
  (`eval` verb, prefix `sepmlp_lr2e4`, on `sepmlp_1b_k200_lr2e-4_s42`). K=200, bs8×ga4,
  15 ep, seed 42. Both trains healthy (no NaN/OOM/fail markers). Driver post-edit
  sha d095934be36a6711; OU integration installed additively (ALLOW_DIRTY, no commit);
  CPU gates 70/1 before submit.
- **Results:**
  - **Arm B (H-wscale):** median selectivity **24.59** (frac≥5 1.00), median recall
    **0.6956** (p10/p90 0.559/0.817; 28/200 ≥0.8), gap +0.6205. → **REFUTED**
    (recall 0.696 < 0.75). Prediction (sel 5–15, recall ≥0.90) missed on BOTH axes.
  - **Arm A2 (H-k200-lr2):** median selectivity **16.33** (frac≥5 1.00), median recall
    **0.7947** (p10/p90 0.665/0.880; 97/200 ≥0.8), gap +0.5196. → **GRAY** (0.75 ≤
    0.7947 < 0.80). Best K=200 point to date; per pre-registration: report, no P4.
  - **K=200 tradeoff curve, all four points (median sel → median recall):**
    lr5e-4·w10/50 507.5→0.637 · lr2e-4·w10/50 36.0→0.747 · **lr1.5e-4·w10/50 16.33→0.795**
    · lr5e-4·w1/5 24.59→0.696. All four lie on ONE monotone curve; B (weight-rescaled)
    sits ON it (slightly below) at its selectivity — the rescale moved B *along* the
    curve, not onto a better one. Curve tops out ≈0.80 in the healthy-sel (5–30) band
    vs the K=20 pilot's 0.977.
  - **C mechanics (lr2e-4 ckpt, ft → unlearned = drop forget10, and dropall):**
    - *Deletion on the forget set (clean):* forget_Q_A_Prob 0.767→0.054,
      forget_ROUGE 0.760→0.316, extraction_strength 0.469→0.047,
      exact_memorization 0.935→0.490, mia_loss 0.997→0.362, mia_min_k 0.997→0.358,
      privleak −99.56→+3.98. Deleted authors go to ~floor and MIA-indistinguishable.
    - *Deletion cost (cheap, O(1)-style):* physical slice removal **1.07 s**
      (20 authors × 20 slices/layer); droplist build 7.2 s one-time (CPU, text-join
      mapping). vs the memadapt block-list anchor 0.027 s — slower but deterministic
      and content-removing, not a mask.
    - *Retain collateral (H-gap ANSWER):* retain_Q_A_Prob **0.676→0.563 (−0.113)**,
      retain_ROUGE 0.665→0.543 (−0.122) — deleting 20 forget-author branches measurably
      degrades the *surviving* authors' recall. YET aggregate **model_utility
      0.465→0.468 (+0.003)** passes the |ΔUtil.R|≤0.03 bar. The collective-recall gap
      is real and produces cross-author deletion collateral, but it is masked by the
      aggregate utility metric (world-facts/real-authors/truth-ratio components are
      deletion-invariant).
    - *dropall ≈ base:* forget 0.092 ≈ retain 0.087 (symmetric floor), model_utility
      0.281 — behavioral confirmation that removing all banks returns the frozen base
      (the weight-level identity remove≡mask≡baked-zero is already CPU-gate-proven;
      an exact numeric calib_base side-by-side needs a base TOFU_EVAL not on disk).
- **What worked / hypothesis verdict:**
  - **H-wscale: REFUTED.** The mechanism claim "suppression pressure ∝ K, so ÷K
    restores the pilot point" is false as stated: rescaling the suppression weights is
    equivalent to a step along the SAME recall-vs-selectivity tradeoff curve, not a
    shift to a higher curve. The K=200 recall ceiling is not a suppression-weight
    artifact.
  - **H-k200-lr2: GRAY (not passed, not refuted).** 0.7947 is the best K=200 recall but
    below the 0.80 pass bar; the pre-registered "report, no P4" applies — no autonomous
    escalation.
  - **H-gap: ANSWERED — collateral is real but aggregate-masked.** Deletion degrades
    surviving-author fine-grained recall by ~0.11–0.12 while leaving aggregate utility
    within bar. Both facts stand together; neither is overclaimed.
  - **Overall reproduction verdict:** H1 (localization) reproduces at K=20 (sel 7.17,
    recall 0.977) and deletion is clean + cheap + MIA-floor at K=200 — the method's core
    mechanics work. But the K=200 **all-active recall ceiling ≈0.80** (vs pilot 0.977
    and vs Vincent's implied healthy-recall regime) is **structural**: two orthogonal
    knobs (lr over 3.3×, suppression weights ÷10) both land on the same ceiling. Vincent's
    priors (deleted 0.97→0.32, utility Δ≤0.001, no relearn residue) were NOT reproduced
    at K=200; whether they hold at his (unknown) K is the open question.
- **Observations:**
  - The collective-recall gap (own-only recall ≈0.28–0.51 ≪ all-active) is the through-line:
    it predicts (a) the K=200 recall ceiling (recall is a property of the whole bank, and
    200 always-on branches interfere) and (b) the C retain collateral (deleting branches
    removes collective capacity). This is the memsinks all-on interference reappearing —
    the very failure sepmlp's architectural disconnection was meant to beat. Disconnection
    fixed *selectivity* (H1 ✓, sel up to 1900) but not *all-active recall at scale*.
  - No silent failures: both trains loss-live to e15, no NaN; probe medians consistent
    with train telemetry; C eval metrics internally consistent (dropall symmetric floor).
- **New questions / new hypotheses:**
  - **H-K-sweep:** is the recall ceiling a smooth function of K? Predict a K∈{20,50,100,200}
    ladder at fixed lr traces recall monotone-decreasing — would locate the K where 0.90
    becomes unreachable and directly address "what K did Vincent run".
  - **H-width / H-layers:** the ceiling may be capacity — width 32 across 16 layers may be
    too little per-author signal at K=200 all-active. Width↑ or layer-subset ablations
    (Wave-2, pre-registered separately).
  - Ask Vincent his K, width, and whether serving is truly all-active-no-router at K=200
    (his 0.97 deleted-baseline ≫ our 0.767 ft suggests a stronger per-author fit).
- **Next Steps:** LADDER HALTED (no passing arm; overnight protocol → no further GPU
  spend, no P4 replication row, no P5). Morning decision package (below in the thread
  README). No overnight deletions (loser-weight cleanup = morning approval; storage
  currently ~8.4 GB peak with both arm ckpts on disk).
