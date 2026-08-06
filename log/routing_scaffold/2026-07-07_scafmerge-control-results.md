### Target Date: 2026-07-07 (scaffold×composition 2×2 control — RESULTS: routing, not the scaffold, is the mechanism)
- **Hypotheses / what we're testing:** H1–H4 as pre-registered this morning in
  [2026-07-07_scafmerge-control-design.md](2026-07-07_scafmerge-control-design.md) (verdict criteria
  frozen before any GPU job). Headline: does the never-run scaffold+MERGED cell reach the routed
  headline (REFUTE, mu ≥ 0.70) or collapse to the expert-interference ceiling (CONFIRM, OOD-aware
  mu ≤ 0.60)?
- **Setup:** As pre-registered, zero training. Llama-3.2-1B, smoke, seed 42; strong experts
  `_experts_scaf_k10` (r32/α64/e5, rslora) merged and served on the scaffolded base. Arm A
  (merged-everywhere) = SLURM **440914** (6 labels via `submit_eval.sh` +
  `eval_manifest_scafmerge.txt`); arm B (OOD-aware merged, new `--merged_label`) = SLURM **440916**
  (4 labels via [`submit_scafmerge_armB.sh`](../../tofu_sisa_lora/submit_scafmerge_armB.sh)).
  All 10 logs clean (no Traceback/OOM/cancel; sacct disabled on this cluster). Script sha256-16s in
  the design entry. Results: `_experts_scaf_k10/results/smoke/{<label>,scafmerged_<label>}.json`.
- **Results:** (anchors: routed+scaffold **0.7509** / retain_prob 0.854 / forget_rouge 0.894;
  matched-FT **0.6372**; scaffold floor mu 0.404 / real 0.6305 / world 0.6556)

  | label (arm A: merged everywhere) | mu | retain_prob | real | world | fq | forget_rouge | retain_ppl |
  |---|---|---|---|---|---|---|---|
  | merged_additive_mean | 0.4557 | 0.1993 | 0.4490 | 0.5515 | 0.9578 | 0.4688 | 8.59 |
  | remerge_additive_mean | 0.4573 | 0.2034 | 0.4528 | 0.5534 | 0.9988 | 0.4598 | 8.23 |
  | merged_dare_ties | 0.4354 | 0.1697 | 0.5920 | 0.6337 | 0.8080 | 0.4899 | 11.03 |
  | remerge_dare_ties | 0.4411 | 0.1754 | 0.5793 | 0.6246 | 0.8080 | 0.4854 | 10.51 |
  | merged_additive_s0.15 | 0.3987 | 0.1476 | 0.3789 | 0.5071 | 0.5941 | 0.4326 | 13.71 |
  | merged_additive_s0.2 | 0.2567 | 0.0743 | 0.3116 | 0.4731 | 0.2391 | 0.3833 | 31.82 |

  | label (arm B: OOD-aware, scaffold-only OOD) | mu | retain_prob | real | world | fq | forget_rouge |
  |---|---|---|---|---|---|---|
  | scafmerged_merged_additive_mean | 0.4938 | 0.1993 | 0.6305 | 0.6556 | 0.9578 | 0.4688 |
  | scafmerged_remerge_additive_mean | 0.4972 | 0.2034 | 0.6305 | 0.6556 | 0.9988 | 0.4598 |
  | scafmerged_merged_dare_ties | 0.4435 | 0.1709 | 0.6305 | 0.6556 | 0.8080 | 0.4988 |
  | scafmerged_remerge_dare_ties | 0.4482 | 0.1751 | 0.6305 | 0.6556 | 0.5941 | 0.4899 |

  Arm-B route_stats = 990 routed / 1208 ood — identical to the routed headline runs. Deletion
  forget_ppl (additive): 9.56 → 15.67.
- **What worked / hypothesis verdict:**
  - **H1 SUPPORTED, decisively.** OOD-aware merged mu = **0.4938/0.4435** (additive/dare), retain_prob
    0.199/0.171 — far inside the CONFIRM region (≤0.60 / ≤0.5) and ≈ the no-scaffold merge ceiling
    (0.43–0.48). With OOD serving *identical* to the routed system (floor exact, same route counts),
    the full **−0.26** gap to routed 0.7509 is attributable to merged-vs-routed serving of author
    queries. **The scaffold cannot rescue a merge; the 07-06 mechanism claim ("routing isolates
    fine-tuning damage") survives its control.**
  - **H2 SUPPORTED** (one marginal cell): merged-everywhere drags OOD below floor — additive real
    −0.181/world −0.104, worsening with λ (real 0.379→0.312, world 0.507→0.473); dare milder (real
    −0.039 ✓, world −0.022, just under the ≥0.03 threshold). Arm B restores the floor **exactly**
    (0.6305/0.6556, all four rows).
  - **H3 SUPPORTED on its core** (utility-neutral + NOT serving-inert): |Δmu| remerge−merged =
    0.002/0.006/0.003/0.005 (all ≤0.02 ✓); retain serving changes on deletion (retain_prob
    0.1993→0.2034, retain_ppl 8.59→8.23 — data-exact but weights move for everyone, vs routing's
    byte-identical 0.7509→0.7509). The fq sub-prediction is MIXED and weakly informative: additive
    0.958→0.999 ✓ but ceiling-compressed by dilution; dare flat (A) / lower (B, 0.808→0.594 — KS-grid
    + unseeded-rebuild noise, all values ≫ indistinguishability). The honest deletion signal is
    **forget_ppl 9.56→15.67**: the forget knowledge in the merge was real and the drop removed it.
  - **H4 SUPPORTED:** λ ladder mu **0.4557 → 0.3987 → 0.2567** (λ = 0.10/0.15/0.20); nothing lifts
    additive_mean by ≥0.05 — both alternatives are far worse, retain_ppl 8.6→13.7→31.8 = the same
    norm-overshoot cliff as the 7B sweep. Interference, not scale.
- **Observations:** (1) **Merge dilution is quasi-unlearning of everything**: the FULL merge already
  has fq 0.958/0.808 and forget_rouge ≈0.47 (vs routed own-author 0.894) — a merged deployment
  barely retains what it's supposed to keep, which is *why* its deletion looks trivially clean.
  (2) **dare_ties is not bit-reproducible across jobs** (unseeded drop mask): arm A vs B TOFU-side
  metrics differ ~0.01 (retain_prob 0.1697 vs 0.1709) where additive is consistent to print
  precision (~0.01-ppl-level fp jitter from B=1 vs batched serving only). Seed the mask before
  using dare numbers in any claim. (3) No silent failures: ppl sane everywhere, no NaNs, fq behaves
  per dilution, route counts match the headline runs. (4) Scaffold's only contribution to a merged
  system is the OOD floor (arm B mu 0.494 vs arm A 0.456) — it adds nothing to author recall.
- **New questions / new hypotheses:** (1) The 2×2 is closed — the biggest remaining threat to the
  headline is the **Alpaca-replay matched-FT control** (can replay rescue the single model's
  real/world without routing?). (2) Extended caps + seeds 43/44 on the headline pair, now with this
  control in the table. (3) Optional gray-zone cell (no-scaffold strong-merged) is moot — H1 landed
  far from the 0.60–0.70 gray zone.
- **Next Steps:** (1) Promote the Alpaca-replay control to the next experiment (one training run +
  one eval). (2) Fold the routed-vs-merged-vs-FT causal decomposition into the thread write-up:
  "same base, same scaffold, same experts — merge them and you get 0.49; route them and you get
  0.75; the difference is composition, not data or capacity." (3) `collect_results.py` refresh.
