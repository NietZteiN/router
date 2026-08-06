# Target Date: 2026-07-21 (bridging arm → G2 GO, H1 confirmed · P3 K=200 launched)

Second entry today; continues [2026-07-21_pilot-oom-and-adjudicate.md](2026-07-21_pilot-oom-and-adjudicate.md)
(which pre-registered the single ADJUDICATE bridging arm at lr 5e-4).

- **Hypotheses / what we're testing:** the G2 ADJUDICATE resolution — does lr 5e-4 pass
  the joint bar (median on/off selectivity ≥ 5.0 AND median own-author answer-prob
  (all-active) ≥ 0.80)? PASS ⇒ H1 CONFIRMED + G2 GO with lr 5e-4 as the K=200 winner;
  FAIL ⇒ G2 decision falls back to refutation-or-redesign (no further bridging arms —
  the clause licenses exactly one).
- **Setup:**
  - Bridging arm: job **446732**, `configs/pilot_relu_lr5e-4.json`
    (sha `f4d6b9d59d62af6d`), K=20, bs16×ga2, seed 42, spec-v2 recipe; driver verb
    `pilot1` (single named arm, body identical to a pilot array task).
  - K=200 memory smoke: job **446903**, `configs/smoke_k200_bs16.json`
    (sha `04d2cf834f4777a9`) — the P3 config's adapter/train shape (K=200 bank, bs16×ga2,
    full alpaca negative pool), 2 authors' data, `--smoke` 5-step cap. Motivated by the
    borderline projection: pilot bs16 K=20 full-run peak 27.59 GiB + ~10 GiB K=200
    param/optimizer delta + bank-activation scaling.
  - P3 launch: train job **446910** (`configs/sepmlp_1b_k200.json`,
    sha `62fd86576fb344ec` — lr overwritten to the winner 5e-4, bs16×ga2 carried from the
    OOM fallback; 15 epochs, seed 42) → probe200 job **446911** chained
    `DEP=afterany:446910` (same-phase chain, not a gate pre-chain; G3 is read manually
    from its JSON). Queue empty at each submit ⇒ max 1 concurrent GPU ≤ 4.
- **Results:**
  - **Bridging arm (lr 5e-4):** median selectivity **7.171** → SELECTIVE (frac≥5 = 0.85,
    frac<2 = 0.00; per-author min/med/max 2.32/7.44/10.36); median own-prob (all-active)
    **0.9765**, min **0.9363**, authors <0.8: **0**. All-active−own-only gap **+0.7304**
    (own-only median 0.213). OOD leakage ood/own: alpaca 0.0079, real_authors 0.0249.
  - **Interpolation check:** log-interp from the 3e-4/1e-3 arms projected sel ≈ 11,
    prob ≈ 0.90 at 5e-4; observed 7.17 / 0.977 — selectivity sublinear vs projection,
    recall better than projected.
  - **Memory smoke:** peak **15.28 GiB**, `[smoke] PASS` (save→reload parity included).
- **What worked / hypothesis verdict:**
  - **H1 CONFIRMED** — a single pre-registered arm passes both bars (7.171 ≥ 5.0;
    0.9765 ≥ 0.80). The LoRA negative-anchor refutation (selectivity ceiling 1.11, 100%
    LAZY at every λ) does not transfer to architecturally disconnected ReLU-gated
    branches; the pilot ladder 4.38 → 7.17 → 38.6 → 1909.7 (lr 3e-4 → 5e-4 → 1e-3 → 3e-3)
    shows suppression strength is lr-dialable over ~3 orders of magnitude.
  - **G2 = GO, winner lr 5e-4** (unique arm satisfying both constraints; the gate's
    "maximize selectivity subject to both bars" selects it trivially).
- **Observations:**
  - The lr 5e-4 arm sits in the collective-recall regime (gap +0.73): its recall still
    depends on other authors' branches being present. Fine for all-active serving
    (deployment mode) and irrelevant to G2, but it sharpens the P4 question — does
    dropping forget10 damage surviving authors' metrics? (`sepmlp_unlearned`
    |ΔUtil.R| ≤ 0.03 is the pre-registered bar that catches it.)
  - Memory projection lesson closed out: the ×10 bank-activation scaling fear was
    overblown — measured K=200 bs16 typical-batch peak is 15.28 GiB (vs 14.04 at bs2);
    the dominant full-run peak driver is negative-batch padding length, which is
    K-independent. Worst-case estimate ~33–38 GiB < 44.5 GiB card.
  - Silent-failure checks on the bridge: no NaN / frozen loss; loss components live;
    probe medians self-consistent with per-author distributions.
- **New questions / new hypotheses:**
  - Does the all-active−own-only gap shrink at K=200 at fixed lr (more in-batch negatives
    per author ⇒ stronger effective suppression pressure), or is it lr-bound? probe200
    reports the same gap metric — read it alongside G3.
  - H-scale bar: does the K=200 median stay ≥ 5.0 and ≥ 0.7× the pilot winner's 7.171
    (i.e. ≥ 5.02 — the two bars nearly coincide for this winner)?
- **Next Steps:** read probe200 (G3: median sel ≥ 5.0 and ≥ 0.7× pilot winner;
  all-active vs own-only own-prob gap ≤ 0.05 — NOTE the gap clause is at risk given the
  pilot regime; if G3 fails on the gap clause alone with selectivity+recall healthy,
  that is an ADJUDICATE-worthy tension to bring back to the human before P4 spend) →
  P4 OU evals (`sepmlp_ft` / `sepmlp_unlearned` / `sepmlp_dropall`) vs Vincent's priors
  (0.97→0.32 deleted / ≤0.002 others / utility Δ≤0.001) → P5 relearn battery.
