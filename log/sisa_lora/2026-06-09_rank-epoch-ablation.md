### Target Date: 2026-06-09 (Rank/epoch ablation smoke eval — r×e grid)

**Goal / hypothesis:** Does LoRA rank or training epoch count affect unlearning quality? Run smoke eval on the full r × e ablation grid for phi-2 and Llama-3.1-8B-Instruct to find the sensitivity threshold.

**Setup:**
- 8 phi-2 variants (r ∈ {8,16,32} × e ∈ {5,10} + r8_e5 representative): eval results in `tofu_sisa_lora/checkpoints/phi-2_r{r}_e{e}/results/smoke/` (completed 10:57–11:16)
- 8 Llama-3.1-8B-Instruct variants: eval results in `tofu_sisa_lora/checkpoints/Llama-3.1-8B-Instruct_r{r}_e{e}/results/smoke/` (completed 11:01–11:04)
- SLURM eval job: 432694 (log: `checkpoints/phi-2_r8_e5/logs/eval_432694_*.log`)
- Results aggregated in: `tofu_sisa_lora/checkpoints/all_metrics_smoke.csv`

**Results (phi-2_r8_e5, representative):**

| Adapter | forget_ppl ↑ | forget_rouge ↓ | model_utility | ks_pval ↑ |
|---|---|---|---|---|
| `merged_dare_ties` | 10.17 | 0.2100 | **0.2826** | 0.3682 |
| `merged_linear` | 9.09 | 0.2257 | 0.0428 | 0.0022 |

Baseline phi-2 (k=10 flat remerge): `merged_dare_ties` ks_pval = 0.583, model_utility = 0.2782. Full r × e grid comparison pending CSV aggregation.

**Observations:** Early results suggest rank and epoch count have modest impact on phi-2 unlearning quality relative to baseline; `dare_ties` consistently retains utility. Full cross-model comparison across the grid still needed to draw conclusions.

**Next steps:** Aggregate full r × e CSV results. Generate comparative report across rank/epoch grid. Determine whether higher rank enables better utility-forgetting trade-off or just noisier adapters.

---

