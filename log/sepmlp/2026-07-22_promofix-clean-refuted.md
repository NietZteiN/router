### Target Date: 2026-07-22 (H-promo-clean REFUTED: the promotion fix is not the K-scaling lever)

Result for H-promo-clean, pre-registered in
[2026-07-22_promofix-results.md](2026-07-22_promofix-results.md).

- **Hypotheses / what we're testing:** H-promo-clean — at lr 1.5e-4 (== A2), the D2a promotion
  fix (fire on all own tokens) raises K=200 ROUGE-L recall above A2's 0.740. CONFIRM: recall >
  0.740. REFUTE: recall <= 0.740.
- **Setup:** train 447458 + recall 447459, configs/sepmlp_1b_k200_promofix_lr1p5e-4.json
  (sha de59493594a3f597), K=200, lr 1.5e-4, 15 epochs, bs8xga4, D2a fix active. The ONLY
  difference vs A2 (sepmlp_1b_k200_lr1p5e-4_s42, ROUGE-L 0.7404) is the promotion-token scope.
- **Results:** per-source ROUGE-L recall **0.7377**, tail **199/200**, named 0.741, name-free 0.678,
  held-out 0.311. A2 (no fix): 0.7404, tail 197/200.
- **What worked / hypothesis verdict:** **H-promo-clean REFUTED.** delta recall = -0.0027 (within run
  noise), delta tail = +2/200. The promotion fix has **no measurable effect** on K=200 recall or tail.
  The tail-from-question-only-promotion hypothesis is wrong. D2a remains correct as a *conformance*
  change (matches paper section 3.2) but does not explain or close the K-scaling gap.
- **Observations:** the K=200 recall-vs-lr picture is now (all promofix or A2, 15 ep):
  lr5e-4 0.639 . lr2e-4 0.705 . lr1.5e-4 0.738-0.740 - recall rises as lr falls, plateauing
  (2e-4->1.5e-4 only +0.035). Held-out stays ~ base (0.31) throughout => foreign adapters silent on
  unseen inputs; the deficit is own-author under-storage at K=200. Mechanistic read: per-adapter
  suppression:task balance is ~K-invariant (mean-normalized penalty; task exposure = 15 ep x 20
  rows regardless of K), so the K drop is NOT a per-adapter loss-balance effect. The remaining
  K-dependent factor is the **forward residual accumulation**: serving on author k sums 199 foreign
  adapter outputs (vs 19 at K=20); even mean-suppressed-small, 199 residuals accumulate a ~10x
  larger perturbation that ad_k must overcome - a plausible cause of the K=20->K=200 recall drop
  (0.957->0.74). The paper's flat-in-K recall implies its foreign outputs are suppressed closer to
  true zero at K=200.
- **New questions / new hypotheses (for a pre-registered sweep, NOT yet run):**
  - **H-supp:** stronger foreign suppression at K=200 (higher lambda_out, e.g. w3 100/200 vs 50)
    reduces the residual accumulation and lifts recall. (w15's weaker lambda was worse => direction
    is stronger.)
  - **H-epoch:** more epochs (40 vs 15) drives foreign outputs closer to zero (better suppression
    convergence) and/or own adapters to fuller storage, lifting recall.
  - **H-lr:** even lower lr (1e-4) continues the recall-up trend (but it is plateauing).
- **Next Steps:** the promotion lead is exhausted; the remaining gap is a K=200 hyperparameter
  search (lambda_out / epochs / lr). Core conformance is complete and gated. Checking in with Jack
  on whether to spend GPU on the reverse-engineering sweep vs supply the paper's appendix
  hyperparameters (which would short-circuit it).
