# TOFU baselines — canonical GA / GD / KL / IDK unlearning references

**Status:** reference (single entry) · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 1 (2026-06-09)

These are the standard TOFU-paper unlearning baselines — Gradient Ascent (GA), Gradient Difference (GD), KL Minimization (KL), and IDK preference optimization — trained on the existing fine-tuned (ft) checkpoints so SISA-LoRA and other methods have a meaningful benchmark. The TOFU paper's own baselines reach model_utility 0.5–1.0, whereas SISA-LoRA merged models sit at ≈0.17–0.28; these reference runs let us measure that gap directly. This folder is the home for future standard-baseline work.

## What worked
- All 4 methods smoke-validated on Llama-3.2-1B-Instruct at max_steps=20 with correct loss signs: GA loss ≈ −1.1 (negative = ascending), KL loss ≈ −1.1, IDK loss ≈ +2.8 (positive).
- Full training ran: 24 jobs (6 models × 4 methods), 125 optimizer steps / 5 epochs each, SLURM jobs 432898–432921; hyperparams match the TOFU paper (lr=1e-5, effective batch=32, 5 epochs).
- Unlearn adapters save to `checkpoints/{slug}_ft_unlearn_{method}/shard_0/`, compatible with the existing eval pipeline.
- Base-model reference numbers established (no fine-tuning, forget10 = authors 180–199): TinyLlama-1.1B mu=0.238 (fp 9.26 / rp 9.28), phi-2 mu=0.284 (11.21 / 11.01), Llama-3.2-1B mu=0.190 (95.76 / 102.75), Llama-2-7B mu=0.129 (40.27 / 44.58), Llama-3.1-8B mu=0.209 (66.23 / 71.50), Qwen2.5-7B mu=0.167 (76.67 / 84.90).
- Preliminary GA/GD results in for 10/24 checkpoints; larger 8B models fare best, e.g. Llama-3.1-8B_gd mu=0.363 and Llama-3.1-8B_ga mu=0.360 (also the highest forget_ppl from GA).

## What didn't / open problems
- GA/GD give forget_ppl ≈ retain_ppl (indiscriminate damage), unlike SISA-LoRA which has fp >> rp.
- All GA/GD checkpoints score ks_pval=0.0 — they fail the KS forgetting test, whereas SISA-LoRA merged_dare_ties reaches ks_pval=0.37–0.58 on the same models (better matches base behavior on forget10).
- world_truth_ratio is very large (10^2–10^3) across models, capping model_utility to 0.17–0.36; the TOFU paper reports higher values for Llama-2-7B / phi-1.5, so these smaller/different models may not generalize.
- IDK split fallback: `forget10_idk` is not in the TOFU dataset, so 20 hardcoded IDK response variants are used instead.
- Eval bugs fixed before results were valid: `load_all_shard_adapters` now skips missing shard dirs (unlearn has only shard_0; k=10 needed for the correct forget10/retain90 split); smoke eval scripts corrected from `--k 1 --forget_shard_id 0` (made the whole dataset the forget set → empty retain → ZeroDivisionError) to `--k 10 --forget_shard_id 9`; stale `base_logprobs.npy` deleted and recomputed at k=10.

## Open ideas / next steps
- Wait for remaining eval jobs to complete (kl, idk, ft baseline — SLURM ~433236–433378).
- Run `python collect_results.py --root checkpoints --smoke` to aggregate the CSV.
- Compare ft baseline model_utility vs unlearn methods to get the relative degradation.
- Extend analysis to phi-2 k=10 SISA-LoRA vs GA/GD head-to-head.

## Entries (chronological)
- [2026-06-09 — GA/GD/KL/IDK baselines](2026-06-09_ga-gd-kl-idk.md) — Train 24 TOFU baseline unlearn checkpoints; fix k=10 eval splits.
