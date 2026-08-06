### Target Date: 2026-06-10 (TOFU eval ported to open-unlearning ground truth)

**Goal / hypothesis:** `tofu_sisa_lora/eval_tofu.py` was suspected of diverging from the canonical TOFU metric definitions ("ad hoc vibes, not the TRUTH"). Establish that locuslab **open-unlearning** is the ground-truth reference and make our eval numerically reproduce it, so reported numbers (esp. `model_utility`, `forget_quality`) are paper-faithful rather than artifacts of our aggregation.

**Setup:**
- Compared `eval_tofu.py` vs `open-unlearning/src/evals/metrics/{memorization,privacy,utility,utils}.py` firsthand. Three confirmed divergences: (1) truth ratio = one geometric-mean R then `max(0,1-R)`, vs OU per-sample `tr=wrong/correct` then `mean(min(tr,1/tr))` (forget) / `mean(max(0,1-tr))` (retain/ra/wf); (2) `forget_quality` = KS over forget **log-probs vs the BASE model**, vs OU KS over the **truth-ratio distribution vs a RETAIN oracle**; (3) MC prob over `option1..4` (numerically == OU `correct/(correct+Σwrong)`, cosmetic).
- Decisions (user): port the math into `eval_tofu` (keep in-memory SISA merging; do not switch engines); forget-quality oracle = a dedicated **retain90 LoRA** (authors 0-179).
- Edits: `eval_tofu.py` (new `_answer_avg_loss`; `get_truth_ratio_scores` per-sample with wrong = geomean of perturbed probs = `exp(-mean loss)`; `get_prob_w_options`; `tr_forget_agg`/`tr_nonforget_agg`; `model_utility`=`scipy.stats.hmean`; `forget_quality`=`ks_2samp(forget_tr, retain_ref)` with `ks_pval` alias; retain TR now over the retain portion of `full_pert`). `train_lora_shard.py` `--retain90` → `{out}/retain90/`. `prepare_eval.py` now caches `results/{sub}/retain_tr_scores.npy` from the retain90 oracle (replaced `base_logprobs.npy`). Also updated `eval_baseline.py`, `verify_eval_tofu.py`, `reports/generate_smoke_report.py`.
- New: `test_ou_equivalence.py` (CPU regression), `spotcheck_eval_port.sh`, `verify_llama2_full.sh`.

**Results:**
- `test_ou_equivalence.py` PASSES: per-answer avg_loss == OU `evaluate_probability` (max|d| 5.7e-6); `get_truth_ratio_scores` == OU `truth_ratio`; `get_prob_w_options` == OU `probability_w_options`; aggregators == `closer_to_1_better`/`true_better`; `model_utility` == `scipy.hmean`. → metric math reproduces open-unlearning exactly given identical prompts.
- Micro CPU smoke of full `evaluate_model` path: runs clean, correct key set, hmean zero→0 matches OU.
- SLURM jobs submitted: **433409** (Llama-3.2-1B shard spot-check: retain90 train → prepare_eval → eval remerge/merged/shard_9_only), **433410** (7B `verify_eval_tofu` on `locuslab/tofu_ft_llama2-7b`). Results pending at write time.

**Observations:**
- open-unlearning is NOT installed in `test-env` (no hydra/omegaconf/deepspeed/flash_attn) → the OU CLI cannot run here; equivalence is proven by reconstructing OU formulas in `test_ou_equivalence.py`. Running the OU CLI live is a follow-up needing env setup.
- The MC-probability "divergence" was numerically a no-op (`real_authors` option1..4 == answer ∪ perturbed). Substantive fixes = truth-ratio aggregation + forget_quality statistic/reference.
- Semantic change: `forget_truth_ratio` is now ∈[0,1] (→1 = better forgetting), not a geomean R>1; old result JSONs are not directly comparable.

**Next steps:** Read 433409/433410 logs — confirm sane ranges (remerge: high `forget_quality` + high `forget_truth_ratio`; merged/shard_9_only: lower) and that 7B `model_utility` moved from ~0.70 toward ~0.62. Then re-run `prepare_eval` + smoke eval across all models to refresh result JSONs under the new metrics; optionally stand up an OU env for a live CLI cross-check.

**Update (resolved — faithfulness confirmed):**
- Spot-check 433409 (Llama-3.2-1B) PASSED: shard_9_only forget_ppl 3.44 + lowest forget_truth_ratio 0.597 + lowest forget_quality 0.135 (memorizes forget); remerge/merged forget_truth_ratio ~0.75 + forget_quality 0.39 (look forgotten, ≈ retain90 oracle). All metrics sane.
- 7B verify on `locuslab/tofu_ft_llama2-7b`: model_utility 0.748 (plain) / 0.761 ([INST], job 433435) — both well above OU leaderboard's 0.63. Template ruled out as the cause.
- OU leaderboard (`open-unlearning/community/leaderboard.md`): TOFU Llama-2-7b Finetuned=0.63/Retain=0.61; Llama-3.2-1B Finetuned=0.60/Retain=0.59.
- **Decisive test (job 433460, `verify_ou_1b_chat.py`):** ran OUR `evaluate_model` on OU's OWN `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` with OU's exact chat template (system prompt + apply_chat_template; token-alignment self-check confirmed single BOS + correct masking) → **model_utility 0.5996 ≈ OU's 0.60**. Our pipeline is faithful.
- Conclusion: the locuslab-7b 0.75 was a *different/stronger checkpoint*, NOT an eval bug. Faithfulness proven 3 ways: formula equivalence (`test_ou_equivalence.py`), OU-number reproduction on OU's model (0.5996 vs 0.60), and sane shard behavior. `eval_tofu` uses plain `Question:/Answer:` (correct for our SISA models, which train that way); scoring external chat-trained checkpoints faithfully needs their chat template.
- New diagnostics: `verify_llama2_full.sh`/`verify_llama2_inst.py` (7B, plain/[INST]), `verify_ou_1b_chat.py` (OU model + chat template).

---

