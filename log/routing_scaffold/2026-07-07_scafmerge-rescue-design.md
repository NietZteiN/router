### Target Date: 2026-07-07 (scafmerge rescue — merge-family sweep + SIFT-on-scaffold, pre-registration)
- **Hypotheses / what we're testing:** The 2×2 control killed *averaging* on the scaffold
  ([results](2026-07-07_scafmerge-control-results.md)); the mechanism (col(B) collision — now
  measured 100% output-side at 92× chance at k=200, see
  [merge_mechanism 07-07](../merge_mechanism/README.md)) says the failure is structural. Rescue
  attempts, pre-registered:
  - **H5 (post-hoc sweep):** no post-hoc merge operator beats the averaging ceiling enough to reach
    routed — best of {knots_ties (shared-SVD-basis, directly targets col(B)), tsv (whitening),
    della_ties, jd_full_c4_r16 (keep-set, O(1) drop), regmean/fisher/lorahub (data-required;
    lorahub precedent 0.592@k4)} lands 0.55–0.65 OOD-aware mu, all < 0.70, retain_prob < 0.5.
    REFUTE = any ≥ 0.70.
  - **H6 (ensemble):** prediction-level composition (`ensemble_probs`, ~k× serving cost) beats
    weight averaging on retain recall (the knowing expert's sharp distribution survives a
    probability mean better than its weights survive a weight mean) but stays < routed.
  - **H7 (SIFT-on-scaffold — the serious contender):** rebuilding SIFT-Masks with θ0 = the
    scaffolded base gives masked merging + the scaffold OOD floor: sift_full mu ≥ plain-base sift
    0.737 with real/world ≈ 0.630/0.656. CONFIRM = mu ≥ 0.74 (decisive if ≥ routed 0.7509);
    REFUTE = ≤ 0.72. Note the honest framing: masks are per-query specialization (author lookup),
    so this beats routing on *storage/serving artifact*, not on serving simplicity.
  - **H8 (SIFT-scaf deletion):** exact re-derive-and-subtract preserves mu (≡ plain sift).
- **Setup:** Llama-3.2-1B, smoke, seed 42. **#1 sweep (zero training):** arm A2 = SLURM **441021**
  (manifest `eval_manifest_scafmerge2.txt`, 7 labels, `submit_eval.sh`, 02:00:00 %4) + ensemble =
  **441022** (1 label, 06:00:00 — k× forward cost); arm B2 = **441023**
  (`SCAFMERGE_LABELS="merged_knots_ties merged_tsv merged_della_ties merged_jd_full_c4_r16"
  bash submit_scafmerge_armB.sh smoke` — new env override; data-required labels can't route through
  `eval_routed_scaffold` (no dataloader glue) so they read arm-A/retain_prob only). Primary sweep
  readout = retain_prob (OOD-independent); arm-B mu for the weight-only merges. **#2 SIFT-scaf:**
  new [`configs/sift_masks_tofu_1b_scaf.json`](../../tofu_sisa_lora/configs/sift_masks_tofu_1b_scaf.json)
  — model_name = ABS path of the scaffolded base, output `_sift_masks_scaf`, ALL hyperparams
  identical to the plain-base config (steps 20 / lr 1e-4 / seed+sign_seed 42 / frozen embeds) for a
  clean scaffold-only diff; KS oracle pre-seeded into
  `Llama-3.2-1B-Instruct_scaffolded_alpaca2k/results/{smoke,extended}/` (driver pulls from
  `checkpoints/<model_slug>/`). Chain = **441024** build (T=200, 04:00:00) → **441025** unlearn →
  **441026** eval `[0-3]%2` (sift/merge × full/unlearn) → **441027** collect. CPU gates before
  submit: label dry-check (all 7 + ensemble parse + nevergrad import), `test_sift_masks.py` ALL
  PASSED, STUB previews of both drivers.
- **Results:** pending (queued behind the nmerge arrays at submit time).
- **What worked / hypothesis verdict:** PENDING — verdict criteria frozen above before any job ran.
- **Observations:** (design-stage) The `merge_full` label inside the sift-scaf eval is a free bonus
  cell: FT+Merge (no masks) *on the scaffolded base* — a second, full-FT confirmation of H1 from
  the [control](2026-07-07_scafmerge-control-results.md) (predict ≈ 0.405-style collapse). If it
  collapses while sift_full recovers, the pair isolates "masks fix the merge" from "scaffold fixes
  the merge" in one experiment.
- **New questions / new hypotheses:** If H7 confirms decisively (≥ 0.7509), the paper's method
  menu becomes routed-experts vs masked-merge at equal utility, differing in artifact size vs
  router dependence — then the deletion_audit MIA battery should run on sift-scaf too. If H5's
  best operator lands in 0.60–0.70, re-open the no-scaffold cell for that operator only.
- **Next Steps:** (1) Results → verdicts in a new dated entry. (2) Winner(s) get remerge_*/deletion
  rows + extended caps + seeds. (3) Then the deferred Alpaca-replay matched-FT control (Test 2 of
  the original plan) — still the biggest outstanding threat to the routed headline.
