# S³T — sequence-aware sliced-and-staged training (exact unlearning) on TOFU

**Status:** complete · **Project:** [`S3T/`](../../S3T/) · **Entries:** 4 (2026-06-12 → 2026-06-16)

S³T (ICLR'25, official code `~/S3T`) shards TOFU authors (m=5 shards × 40 authors), splits each shard into L=4 cumulative top-down slices (10 authors each), and trains the slices in sequential stages on disjoint LoRA layer blocks. Deletion is exact and O(1): revert the affected shard to its pre-forget-slice snapshot. Inference aggregates shards by a new token-level distribution-average ensemble (`ensemble_probs`). "Faithful repro" meant matching the paper end-to-end on TOFU — Alg 1 (cyclic rotation), Alg 2 (BMS + Eq-24 + Hungarian), Alg 4 (deactivate downstream slices, serve best survivor), Lemma 1/2, plus the headline figures (deletion rate δ, perf-vs-#deletions, deletion time) against a SISA (B=1) baseline.

The work ran in two arms: **armA** = the paper-faithful Table 2 Llama2-7B recipe (r32/α64, lr 2e-5, 3 ep/stage) and **armB** = the k=1-winner transfer recipe (lr 1e-4, 5 ep/stage). Outcome: the implementation was audited **faithful with no mechanism bugs**, the combinatorial headline (S³T B=4 handles ~1.6× more deletions than SISA) reproduced, and armB supplied the meaningful utility contrast that armA — undertrained at the paper's exact HPs — could not.

## What worked
- Deletion-rate simulation matched Lemma 1 (mL·H_{mB'}) within **<0.4%**: δ(m5,L4) = **45.8 (B1=SISA) → 58.3 (B2) → 71.9 (B4)**, saturating at B=L (B8 ≈ 71.9).
- S³T(B=4) handles **~1.59×** more deletion requests than SISA — matches the paper's ~1.6×.
- Faithful Fig-9 (cumulative deletion time over a 1000-request stream): S³T **1.60× vs SISA, 71× vs full-retrain**.
- armB F(d) curve sits well above base at every depth: base **0.4179 → 0.533 → 0.581 → 0.576 → 0.580**, making Fig-6-left meaningful — S³T(B=4) ≫ SISA(B=1) as deletions accumulate (r=12: 0.555 vs 0.490; r=24: 0.513 vs 0.452), both decaying toward base.
- RQ3/Fig-8 diversity reproduced: cyclic edit-distance **5.0 (=L)** > random 4.1; BMS maximally diverse (=L, Lemma 3).
- Implementation audited **faithful** — Algs 1/2/4, cumulative top-down slice training, and Lemma 1 all match; no mechanism bugs. CPU gate suites (`test_s3t`, `test_ensemble`, `test_s3t_sequences`, etc.) green throughout.
- A real port bug was caught and fixed: the official `check_if` substring layer-id match would break exactness at single-digit layer ids — replaced with an exact layer-id regex.

## What didn't / open problems
- **armA stays near base at all depths** (0.42–0.46, flat F-curve) — the paper's exact HPs undertrain on TOFU. Reported as a faithful finding, not a fix; armB is the informative contrast.
- **Reference Eq-18 typo:** the printed `1-(k/L)^r` is inconsistent with its own Eq-21 derivation and with S3T(B=1)=SISA; the self-consistent form `(1-k/L)^r` was implemented and tested.
- **"BMS > sorted-cyclic on score" (Fig 15) does not robustly reproduce** — at t=1 all position-diverse sets tie by construction; reported descriptively rather than asserted.
- The report date rolled over twice: entries cite `S3T_PAPER_REPRO_2026-06-15.md` / `_06-16.md`, but the surviving report on disk is `S3T_PAPER_REPRO_2026-06-17.md` (the earlier dated files were superseded/removed).

## Open ideas / next steps
- Task is complete; no further work required. armB remains available as the utility contrast if deeper ablations are wanted.
- Optional higher-fidelity mixed-depth GPU validation of the F(depths).mean() composition (vs the snapshot-reuse F(d)).
- Stale armA-only precursor report could be removed (only with confirmation).

## Entries (chronological)
- [2026-06-12 — paper-faithful port](2026-06-12_paper-port.md) — port + overnight train/eval chain, ensemble aggregation, layer-id bug fix
- [2026-06-15 — full paper reproduction](2026-06-15_full-repro.md) — budget B>1, deletion-stream, δ/perf/time figures vs SISA
- [2026-06-15 — faithfulness audit + armB](2026-06-15_audit-armB.md) — audit verdict faithful; Lemma-2, RQ3, Fig-9 gaps closed
- [2026-06-16 — armB contrast complete](2026-06-16_armB-complete.md) — armB F(d) above base; report regenerated; task complete

## Full reports
- [S3T_PAPER_REPRO_2026-06-17.md](../../tofu_sisa_lora/reports/S3T_PAPER_REPRO_2026-06-17.md) — final faithful-repro report (armA+armB curves, δ, Lemma-2, Fig-9, RQ3, storage Table 3)
- [SCALE_REPORT_2026-06-12.md](../../tofu_sisa_lora/reports/SCALE_REPORT_2026-06-12.md) — k-scaling / routing-vs-merging frontier referenced as the comparison bar for the ensemble
