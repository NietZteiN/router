### Target Date: 2026-07-22 (pre-registration: K=200 reverse-engineering sweep, wave 1 — suppression & epochs)

Jack authorized a bounded reverse-engineering sweep to close the K=200 recall gap (best 0.740 vs
paper 0.966). Core conformance is DONE + gated (D2a/D2b, 77 CPU tests). This sweep uses the
conformant code (promofix) and holds lr at 1.5e-4 (the K=200 recall-best lr). Bars frozen BEFORE
the jobs.

- **Motivating hypothesis (from [2026-07-22_promofix-clean-refuted.md](2026-07-22_promofix-clean-refuted.md)):**
  the K=20->K=200 recall drop (0.957->0.74) at a fixed recipe is NOT a per-adapter loss-balance
  effect (mean-normalized penalty + K-invariant task exposure), so the leading cause is the FORWARD
  RESIDUAL ACCUMULATION: serving on author k sums 199 foreign adapter outputs (vs 19 at K=20); even
  mean-suppressed-small, 199 residuals accumulate a ~10x larger perturbation that ad_k must
  overcome. Corollary: driving foreign outputs closer to TRUE zero (stronger suppression and/or
  more convergence) should lift recall. (Consistent with w15's weaker lambda being WORSE, 0.606.)
- **Baseline:** K=200, lr1.5e-4, w_h/w_out/w_p=10/50/1, 15 ep, promofix -> ROUGE-L recall 0.738
  (= A2's 0.740 within noise).
- **Arms (wave 1; all K=200, lr1.5e-4, bs8xga4, promofix, seed 42; ONE lever each):**
  - **H-supp2:** w_out (w3) 50 -> **100**. `configs/sepmlp_1b_k200_pf_wout100.json` sha `9362ff56d2e40020`.
  - **H-supp4:** w_out 50 -> **200**. `configs/sepmlp_1b_k200_pf_wout200.json` sha `f28fb092498e013b`.
  - **H-epoch:** epochs 15 -> **30** (w_out 50). `configs/sepmlp_1b_k200_pf_ep30.json` sha `c7db412815733db4`.
- **Bars:** per-arm per-source ROUGE-L recall (measure_recall, full 200 + holdout).
  CONFIRM a lever helps: recall > 0.79 (a clear >+0.05 lift over the 0.738 baseline, beyond
  run noise ~0.003). STRONG: recall >= 0.90 (toward paper 0.966). REFUTE the residual-accumulation
  hypothesis: ALL three arms <= 0.76 (no lever moves recall) -> pivot (width/layers, or re-examine
  the mechanism). Secondary: held-out stays ~base (0.31, isolation intact); higher w_out should
  also raise selectivity / lower held-out-leak.
- **Budget/cap:** 3 arms in parallel, each 1 GPU (train) then its chained recall (afterany) —
  concurrent <= 3 <= 4 cap. H-supp arms ~2h (15 ep); H-epoch ~4h (30 ep). No new smoke: same code
  path + memory shape as the validated promofix runs (447384/447458 ran clean at 33 GiB); only
  w_out / epochs change, which do not affect memory or the code path.
- **Wave 2 (contingent, not yet registered):** if a lever helps, combine it (e.g. high w_out +
  more epochs) and/or push further; if suppression is the winner, add a lower-lr confirm. Each
  pre-registered before spend.
- **Results / verdict:** pending.
