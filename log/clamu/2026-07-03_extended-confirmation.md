### Target Date: 2026-07-03 (extended-cap confirmation of the 1B ladder — H4)
- **Hypotheses / what we're testing:** **H4** (from [2026-07-02](2026-07-02_1b-headline.md)):
  the smoke ladder survives extended caps (ROUGE≤200 / retain≤400 / truth≤120). CONFIRM:
  ordering Global < EMR < TALL ≪ ClAMU preserved and clamu_unlearn ≥ clamu_full. REFUTE:
  ordering breaks or clamu collapses toward the merge band at higher sample counts.
- **Setup:** eval-only rerun of the six 07-02 artifacts (no retraining):
  `CLAMU_EVAL_ARGS=--extended CLAMU_EVAL_SUB=extended bash submit_clamu_tofu.sh
  configs/clamu_tofu_1b.json eval` → SLURM array **440209** (6 labels, %2, ran 2026-07-02
  14:54–16:45), extended KS reference `checkpoints/Llama-3.2-1B-Instruct/results/extended/
  retain_tr_scores.npy`. Same config/scripts/hashes as the 07-02 entry; seed 42.
  Collected into `all_metrics_extended.csv` (`collect_results.py --extended`).
- **Results:** (mu = model_utility; smoke → extended)

  | label | mu smoke → ext | fq smoke → ext | f_rouge ext | r_rouge ext | retain_prob ext |
  |---|---|---|---|---|---|
  | merge_full | 0.3511 → 0.3371 | 0.594 → 0.0505 | 0.288 | 0.259 | 0.096 |
  | emr_full | 0.3882 → 0.3682 | 0.808 → 0.2371 | 0.314 | 0.266 | 0.117 |
  | tall_full | 0.4053 → 0.4288 | 0.393 → 0.007 | 0.325 | 0.339 | 0.166 |
  | **clamu_full** | **0.6469 → 0.6092** | 0.393 → 0.389 | 0.497 | 0.473 | 0.398 |
  | **clamu_unlearn** | **0.6609 → 0.6204** | 0.393 → **0.0045** | 0.321 | 0.491 | 0.418 |
  | merge_unlearn | 0.3527 → 0.3369 | 0.393 → 0.0045 | 0.321 | 0.258 | 0.096 |

  Cross-track (same extended path): SIFT-Masks `sift_full` mu **0.7364** / fq **0.0045**,
  `sift_unlearn` mu **0.737** / fq **0.0045**. No errors/NaNs in any eval log.
- **What worked / hypothesis verdict:** **H4 SUPPORTED.** Extended ordering
  0.337 < 0.368 < 0.429 ≪ **0.609**, and deletion still *raises* utility
  (clamu_unlearn **0.620** ≥ 0.609; retain_rouge 0.473→0.491). Shrinkage from smoke is
  small and uniform (clamu −0.038, Global −0.014); TALL edged up (+0.024) but stays far
  below ClAMU. The extended mask-granularity dial reads: Global **0.337** → cluster K=16
  optimized **0.609** → SIFT per-task T=200 **0.736**.
- **Observations:** the fq story sharpened into a *cross-track metric artifact*, not a ClAMU
  issue: `clamu_unlearn`, `merge_unlearn`, `sift_full`, and `sift_unlearn` ALL land at fq
  **0.0045** at extended caps. Every one of these serves forget-author queries at (or
  indistinguishably near) base θ0, and with 120 truth rows the KS test now has the power to
  distinguish *raw base* from the *retain90-oracle LoRA* (whose fine-tuning shifts its forget-set
  truth-ratio distribution even without forget data). At smoke caps this difference was invisible
  (fq 0.393 ≡ "base level"). So low extended fq here measures "base ≠ retain90-trained
  style", **not residual forget knowledge** — deletion remains exact by construction (forgotten
  authors' serving is a function of no forget data). This is precisely the sift thread's open
  **H8**: serve forgotten/held-out authors with the maskless *retain* merge `θ0 + τ̄_retain/T`
  (the ClAMU/SIFT papers' own Fig-8 protocol) instead of raw θ0, so extended fq becomes
  cross-track comparable. `clamu_full`'s fq 0.389 staying high is the mirror artifact: its
  diluted masked serve happens to sit nearer the oracle's distribution than base does.
- **New questions / new hypotheses:**
  - **H8-align (open, shared with sift_masks):** add a `forgotten_serve: merged` option to
    `ClamuModel` (`θ0 + τ̄_<tag>/T` for forgotten/OOD-forget queries) and re-eval
    `clamu_unlearn` extended — prediction: fq rises to the oracle-comparable range while mu
    is unchanged (retain serving untouched).
  - H5 (K dial), H6 (why cluster-masking beats routing), H7 (random vs feature) unchanged.
- **Next Steps:** (1) treat the mu ladder as confirmed and proceed to the gated expansion —
  **E4/H5 K-dial** first (K∈{1,4,50,100,200}); (2) implement the H8-align serving option
  jointly for clamu+sift so both tracks' extended fq is interpretable; (3) appendix the
  extended table into `reports/CLAMU_REPORT_2026-07-02.md` (done in the same working day).
