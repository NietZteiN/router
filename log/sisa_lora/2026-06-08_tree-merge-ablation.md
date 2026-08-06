### Target Date: 2026-06-08 (Tree-merge aggregation + baseline fine-tunes + rank×epoch ablation)

**Goal / hypothesis:** (1) Do tree-merge aggregation strategies outperform flat merges for SISA unlearning? (2) Does LoRA rank or epoch count explain the phi-2 / capacity effects seen on Jun 5? (3) Establish fine-tuned baselines for all target models.

**Setup:**
- Tree merge: 12 hierarchical variants evaluated on `Llama-3.2-1B-Instruct` (17:02–17:19) and `Llama-3.1-8B-Instruct` (18:32–18:40); results in `tofu_sisa_lora/checkpoints/{model}/results/smoke/tree_*.json`
- Baseline fine-tunes: 6 models trained (k=1, rank=8, epochs=3); SLURM jobs 432687–432692
  - `Llama-3.2-1B-Instruct_ft` (23:04), `phi-2_ft` (23:12), `Qwen2.5-7B-Instruct_ft` (23:31), `Llama-3.1-8B-Instruct_ft` (23:36), `Llama-2-7B-chat-hf_ft` (23:41), `TinyLlama-1.1B-Chat-v1.0_ft` (23:07)
- Rank × epoch ablation training overnight — phi-2 and Llama-3.1-8B-Instruct, r ∈ {8,16,32} × e ∈ {3,5,10}: SLURM jobs 431601–432012 and 432526–432705
  - phi-2 variants done 21:31–21:50; Llama-3.1-8B variants done 21:54–22:44
- Scripts updated: `tofu_sisa_lora/train_lora_shard.py`, `tofu_sisa_lora/eval_tofu.py`, `tofu_sisa_lora/merge_lora.py`
- Literature review updated: [tofu_sisa_lora/LoRA Merging_ Methods and Effectiveness.md](../../tofu_sisa_lora/LoRA%20Merging_%20Methods%20and%20Effectiveness.md) (16:30)
- Aggregate CSV updated: `tofu_sisa_lora/checkpoints/all_metrics_smoke.csv` (20:41)

**Results (Llama-3.1-8B-Instruct tree merge, k=10):**

| Adapter | forget_ppl ↑ | forget_rouge ↓ | model_utility | ks_pval ↑ |
|---|---|---|---|---|
| `tree_remerge_dare_ties` | 40.98 | 0.3491 | 0.2702 | 0.0 |
| `tree_root_dare_ties` | 38.66 | 0.3413 | 0.2618 | 0.0 |
| `tree_remerge_linear` | **2,351,114** | 0.0 | 0.0 | 0.0 |

`tree_remerge_linear` achieves extreme forgetting but destroys utility. Llama-3.2-1B tree merge results in CSV.

**Observations:** Literature review key finding: rescaling after sparse pruning (DARE) matters more than the pruning strategy itself — directly informs why `dare_ties` variants consistently outperform plain `ties`. Tree merge produces similar or slightly worse utility-forgetting trade-off than flat remerge for `dare_ties`; `linear` aggregation without rescaling collapses the model. Ablation training completed overnight.

**Next steps:** Run smoke eval on all rank/epoch ablation checkpoints. Compare forgetting quality across r/e grid to find rank sensitivity threshold.

---

