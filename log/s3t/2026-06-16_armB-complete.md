### Target Date: 2026-06-16 (S³T faithful-repro — armB contrast landed; task complete)
- **Goal / Hypothesis:** Confirm the armB F(d) contrast + regenerated report (cont. of 2026-06-15 audit/gap-closure).
- **Setup:** armB F-eval 434856 + finalize 434857 completed. Final report (date rolled over):
  **`reports/S3T_PAPER_REPRO_2026-06-16.md`** (supersedes the 06-15 armA-only file).
- **Results:** armB F(d) = base 0.4179 → 0.533 → 0.581 → 0.576 → 0.580 (well above base at all
  depths; armA stayed flat 0.42–0.46). Fig-6-left now meaningful: S3T(B=4) ≫ SISA(B=1) as
  deletions accumulate (r=12: 0.555 vs 0.490; r=24: 0.513 vs 0.452), both decaying toward base.
  All reference deliverables present: RQ3/Fig-8 diversity, Lemma-2 Eq18/20 overlay, faithful Fig-9
  (S3T 1.60× vs SISA, 71× vs full-retrain over 1000 requests), storage Table 3, armA+armB curves.
- **Observations:** TASK COMPLETE. Implementation audited faithful; all gaps closed; all CPU
  tests green. Two documented catches stand: reference Eq-18 typo (use (1-k/L)^r); BMS>sorted-
  cyclic score not robust (t=1 ties). Stale `S3T_PAPER_REPRO_2026-06-15.md` can be removed (only
  with confirmation) — it is the armA-only precursor of the 06-16 report.
- **Next Steps:** none required; armB available as the utility contrast if deeper ablations wanted.

