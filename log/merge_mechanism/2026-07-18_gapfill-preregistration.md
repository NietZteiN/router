### Target Date: 2026-07-18 (Gap-fill table-closers @k=200 — pre-registration)
- **Hypotheses / what we're testing:** Close the untried cells of the user's 14-method
  merge reference ([`TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md`](../../tofu_sisa_lora/reports/TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md))
  at k=200 on the existing 7B r8 pool. Framing note: PATHS_FORWARD §9 called further
  merge-operator archaeology low-value — these runs are **table-closers requested against
  the doc-1 reference**, not new bets. Falsifiable predictions:
  - **H-gf-1 (RegMean @200):** `merged_regmean` lands in the merge-ceiling band
    (mu 0.42–0.50, i.e. ≈ dare_ties 0.4198 ± the additive_mean 0.4597 spread). CONFIRM =
    inside band; REFUTE = mu >0.55 (would make RegMean the best merge operator and
    challenge the "no merge escapes the ceiling" claim) or mu <0.30 (degenerate).
  - **H-gf-2 (Fisher @200):** `merged_fisher` likewise in-band (prior k=10: 0.4239);
    same confirm/refute bars as H-gf-1.
  - **H-gf-3 (TIES/KnOTS @200):** `merged_ties` and `merged_knots_ties` in-band (priors
    k=10: knots 0.4242; dare_ties@200 0.4198). Doc-1 predicts TIES sign-election at 200
    voters hurts per-author recall — read forget_rouge/forget_ppl asymmetry, not just mu.
  - **H-gf-4 (Breadcrumbs λ-rescue):** the 2026-06-11 degenerate breadcrumbs (ppl 10⁴⁺)
    was the √r-inflated uniform-1/n scale, not the mid-band masking itself. CONFIRM =
    `merged_breadcrumbs_s{λ}` at λ≈1/(n√r) is non-degenerate (retain_ppl <100, mu ≥0.40);
    REFUTE = still degenerate at both λ rungs → the masking is the problem. Doc-1's own
    prediction (top-γ% drop deletes memorization) then shows as forget_rouge below the
    dare_ties row despite healthy mu.
- **Setup (planned; jobs not yet submitted — cluster cap held by another session):**
  - Pool: `checkpoints/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4` (200 shards + retain90 KS ref
    on disk; r8 is the only k=200 pool that fits in-model per the memory law — r32 fp32
    ≈65 GiB cannot load).
  - Manifest `results/smoke/eval_manifest_gapfill.txt` (6 labels): merged_regmean,
    merged_fisher, merged_ties, merged_knots_ties, merged_breadcrumbs_s0.00177
    (=1/(200·√8)), merged_breadcrumbs_s0.005 (=1/n control).
  - Breadcrumbs λ validated at 1B k=10 FIRST: manifest
    `checkpoints/Llama-3.2-1B-Instruct/results/smoke/eval_manifest_gapfill_bc.txt`
    (merged_breadcrumbs_s0.0354 = 1/(10·√8), merged_breadcrumbs_s0.1 = 1/n).
  - Submission (after 1B validation): `EVAL_MANIFEST=<gapfill> EVAL_EXTRA_ARGS="--smoke
    --merge_num_examples 32" EVAL_TIME=05:00:00 ARRAY_CAP=<fits cap> bash submit_eval.sh
    checkpoints/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4 meta-llama/Llama-2-7B-chat-hf 200`.
  - **Code changes today (CPU gates green):**
    1. `eval_tofu.py --merge_num_examples` (default 256 = bit-identical historical
       behavior) plumbed into `build_merge_dataloader(num_examples=)` AND
       `activate_label(num_regmean_examples=)` — at k=200 the 256 default costs 200×256
       passes per data-required merge; these runs use **32/shard, a recorded deviation**
       from the k=10 precedent (Fisher/RegMean numbers are comparable only within the same
       num_examples). sha256 eval_tofu.py 810b59d6f944293c…; `test_eval_rows.py` green.
    2. `merge_lora.py`: `_s{λ}` global-coefficient suffix now APPLIES to `breadcrumbs`
       (weights [λ]·n, additive's convention; previously parsed but silently ignored —
       the no-op bug). sha256 merge_lora.py 62675f6a13a0faac…; new regression
       `test_merge_extra.test_breadcrumbs_scale_label` green (full suite passes).
  - **Deferred (frozen behind running jobs):** DARE+sum p∈{0.9,0.99} needs a
    `sparsify_pool.py` op-table extension — that file is in use by ctv [w5] job 445329
    and will not be edited until it completes. Full-FT-vector operator pilot (20-author
    ds pool) waits for the composable_tv Wave-1 trains.
  - Seed 42; interpreter `/home/jack/anaconda3/envs/test-env/bin/python`; global 4-GPU cap
    checked via `squeue` before every submission (currently held by memadapt 445308–445311).
- **Results:** none yet — pre-registration only; jobs blocked on cap headroom.
- **What worked / hypothesis verdict:** pending.
- **Observations:** comparability caveats to carry into the results entry: (i) prior
  Fisher/KnOTS numbers are k=10 r32; these are k=200 r8 — cross-k AND cross-rank, so quote
  the dare_ties@200-r8 0.4198/0.4201 rows as the in-pool anchor, not the k=10 numbers;
  (ii) num_examples=32 deviation (above); (iii) breadcrumbs remains in the √r-inflated
  factor-space family by design — λ is a global rescue coefficient, not a convention fix.
- **New questions / new hypotheses:** if any closer lands ABOVE 0.55 (H-gf-1/2 refuted),
  that operator immediately becomes a candidate for the routing_scaffold rescue-sweep
  rerun on the scaffolded pool (its H5 sweep capped at 0.4938).
- **Next Steps:** 1B breadcrumbs validation pair → 7B k200 gapfill array (6 tasks) →
  results entry with the doc-1 coverage table updated; DARE+sum ops + test case once
  445329 lands; then labels `sparse_dare0p9sum_N{n}_s42` / `sparse_dare0p99sum_N{n}_s42`.
