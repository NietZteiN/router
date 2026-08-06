### Target Date: 2026-07-22 (H-promo result: REFUTED at lr5e-4 but CONFOUNDED by lr; clean A/B relaunched)

Results for the H-promo arm pre-registered in
[2026-07-22_promofix-preregistration.md](2026-07-22_promofix-preregistration.md).

- **Hypotheses / what we're testing:** H-promo — the D2a promotion fix (fire on all own tokens)
  closes the K=200 K-scaling recall gap. CONFIRM: K=200 ROUGE-L recall ≥ 0.90 ∧ tail ≤ 80.
  REFUTE: recall ≤ 0.78.
- **Setup:** train 447384 + recall 447385, `configs/sepmlp_1b_k200_promofix.json`
  (sha `54af0e622c6c2929`), K=200, **lr 5e-4**, 15 epochs, bs8×ga4, D2a fix active. Train healthy
  (33.04 GiB, no NaN). `measure_recall.py` full sweep (200 authors + 20 holdout).
- **Results:** per-source ROUGE-L recall **0.6393**, tail **200/200**, named 0.643, name-free 0.575,
  held-out 0.294.
- **What worked / hypothesis verdict:** **H-promo REFUTED as registered** (0.6393 ≤ 0.78). BUT the
  arm is **CONFOUNDED and the verdict is not clean**: I anchored the retrain on lr 5e-4 (the *K=20*
  ROUGE-L winner), whereas at K=200 lr 5e-4 is the *over-suppressing* end of the curve — it was the
  0.637-answer-prob G3-FAIL arm, the worst K=200 lr. The best K=200 no-fix arm (A2) was at lr 1.5e-4
  (ROUGE-L 0.740). So this arm changed **two** variables vs A2 (lr 5e-4→ and the promotion fix),
  and recall 0.639 < 0.740 conflates them. This does **not** cleanly show the promotion fix hurts or
  helps — it mostly shows lr 5e-4 is a poor K=200 operating point (as already known on answer-prob).
  My error: choosing the K=20-winning lr for a K=200 retrain instead of holding lr at the
  K=200-appropriate 1.5e-4 to isolate the fix.
- **Observations:** held-out 0.294 (≈ base, isolation intact) confirms foreign adapters are ~silent
  on unseen inputs — the deficit is own-author under-storage at K=200, not foreign leakage. The tail
  staying at 200/200 at lr 5e-4 is consistent with over-suppression drowning the promotion signal at
  that lr. The promotion fix's effect must be read at a non-over-suppressing lr.
- **New questions / new hypotheses:** **H-promo-clean (pre-registered here):** at lr 1.5e-4 (== A2),
  the D2a promotion fix raises recall above A2's 0.740. Config
  `configs/sepmlp_1b_k200_promofix_lr1p5e-4.json` (sha `de59493594a3f597`); the ONLY difference vs
  A2 is the promotion fix. CONFIRM: recall > 0.740 (isolates the fix; toward paper 0.966 / tail
  ≪197). REFUTE: recall ≤ 0.740 (promotion is not the K-scaling lever → pivot to λ_out / epochs).
- **Next Steps:** run H-promo-clean (train + recall). Then, per the evidence so far (recall favors
  lower lr; weaker suppression w15 was worse), the reverse-engineering sweep direction is
  lower lr and/or stronger λ_out and/or more epochs at K=200 — pre-registered before spend.
