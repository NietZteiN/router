### Target Date: 2026-06-09 (TOFU unlearning baselines)

**Goal / hypothesis:** Establish TOFU paper baseline results (Gradient Ascent, Gradient Difference, KL Minimization, IDK Preference Optimization) to serve as a reference for SISA-LoRA utility comparison. SISA-LoRA merged models show model_utility ≈ 0.17–0.28; the TOFU paper's own baselines achieve 0.5–1.0. Without these reference results, there is no meaningful benchmark.

**Setup:**
- New scripts: `tofu_sisa_lora/train_tofu_unlearn.py`, `tofu_sisa_lora/submit_tofu_unlearn.sh`
- Starts from existing ft checkpoints (k=1, rank=8, 3 epochs; all 6 models trained Jun 8)
- Saves unlearn adapters to `checkpoints/{slug}_ft_unlearn_{method}/shard_0/` — compatible with existing eval pipeline
- Smoke-validated all 4 methods (GA, GD, KL, IDK) on Llama-3.2-1B-Instruct at max_steps=20
  - GA loss ≈ −1.1 (negative = ascending, correct), KL loss ≈ −1.1, IDK loss ≈ +2.8 (positive, correct)
  - IDK fallback active: `forget10_idk` split not in TOFU dataset; using 20 hardcoded IDK response variants
- Training hyperparams match TOFU paper: lr=1e-5, effective batch=32, 5 epochs
- KL reference: snapshot initial LoRA weights to CPU, swap in per step for ref logits (avoids loading base model twice)

**Results (24 unlearn checkpoints trained, eval running, 10/24 shard_0_only results in):**

Training: 24 jobs (6 models × 4 methods), 125 optimizer steps each, 5 epochs. SLURM jobs 432898–432921.

Eval fixes required:
- `eval_tofu.py`: `load_all_shard_adapters` now skips missing shard dirs (unlearn has only shard_0; k=10 is needed for correct forget10/retain90 split). Fix: skip-missing-shards in the load loop.
- `submit_unlearn_eval_smoke.sh` and `submit_ft_eval_smoke.sh`: both fixed to use `--k 10 --forget_shard_id 9` (instead of wrong `--k 1 --forget_shard_id 0` which made the entire dataset the forget set → empty retain → ZeroDivisionError).
- Stale `base_logprobs.npy` files deleted and recomputed with correct k=10.

Base model results (no fine-tuning, forget10 = authors 180-199):

| Model | mu | fp (forget_ppl) | rp (retain_ppl) |
|---|---|---|---|
| TinyLlama-1.1B | 0.238 | 9.26 | 9.28 |
| phi-2 | 0.284 | 11.21 | 11.01 |
| Llama-3.2-1B | 0.190 | 95.76 | 102.75 |
| Llama-2-7B | 0.129 | 40.27 | 44.58 |
| Llama-3.1-8B | 0.209 | 66.23 | 71.50 |
| Qwen2.5-7B | 0.167 | 76.67 | 84.90 |

Preliminary GA/GD unlearn results (10/24 complete):

| Checkpoint | mu | fp | rp | ks_pval |
|---|---|---|---|---|
| TinyLlama_ga | 0.164 | 6.24 | 5.84 | 0.0 |
| TinyLlama_gd | 0.173 | 5.88 | 5.69 | 0.0 |
| phi-2_ga | 0.278 | 6.33 | 5.78 | 0.0 |
| phi-2_gd | 0.278 | 6.27 | 5.76 | 0.0 |
| Llama-3.2-1B_ga | 0.278 | 36.20 | 36.63 | 0.0 |
| Llama-3.2-1B_gd | 0.281 | 38.20 | 39.77 | 0.0 |
| Llama-2-7B_ga | 0.222 | 11.00 | 10.33 | 0.0 |
| Qwen_ga | 0.267 | 21.45 | 20.68 | 0.0 |
| Qwen_gd | 0.287 | 20.90 | 20.60 | 0.0 |
| Llama-3.1-8B_ga | 0.360 | 29.58 | 29.43 | 0.0 |
| Llama-3.1-8B_gd | 0.363 | 32.20 | 32.70 | 0.0 |

Key observations (preliminary):
1. **GA/GD give forget_ppl ≈ retain_ppl** (indiscriminate damage, unlike SISA-LoRA which has fp >> rp).
2. **All ks_pval=0.0** — GA/GD fail the KS forgetting test. SISA-LoRA merged_dare_ties achieves ks_pval=0.37–0.58 on the same models. This means SISA-LoRA better matches base model behavior on forget10.
3. world_truth_ratio is very large (10^2–10^3) for all models — this caps model_utility to 0.17–0.36 for these architectures. The TOFU paper reports higher values for Llama-2-7B / phi-1.5; these smaller/different models may not generalize.
4. Larger models (8B) achieve better model_utility AND higher forget_ppl from GA.

ft baseline (k=10 corrected) eval running now (SLURM jobs 433367–433378); kl/idk methods running in parallel (jobs ~433236–433329).

**Next steps:**
1. Wait for remaining eval jobs to complete (kl, idk, ft baseline).
2. Run `python collect_results.py --root checkpoints --smoke` to aggregate CSV.
3. Compare ft baseline model_utility vs unlearn methods to get the relative degradation.
4. Extend analysis to phi-2 k=10 SISA-LoRA vs GA/GD head-to-head.

---

