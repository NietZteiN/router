### Target Date: 2026-06-04 (SISA-LoRA smoke evaluation across model families — k=4 shards)

**Goal / hypothesis:** Smoke evaluation of SISA-LoRA unlearning across three model families (k=4 shards) — does shard-level forgetting via LoRA subtraction produce measurable unlearning on the TOFU benchmark?

**Setup:**
- Models: `Llama-3.2-1B-Instruct`, `TinyLlama-1.1B-Chat-v1.0`, `phi-2`; all k=4 shards, forget shard = `shard_3` (authors 75–99)
- SLURM training jobs: 431577 (TinyLlama, done 00:33), 431585 (phi-2, done 17:01), 431633 (Llama-3.2-1B, done 19:01)
- ~17 adapter variants per model evaluated: individual shards, merged variants (ties, dare_ties, slerp, …), remerge variants (cat, dare_ties, …)
- Smoke eval caps: ROUGE ≤ 50, retain PPL ≤ 80, truth ratio ≤ 30, KS ≤ 100 samples
- SLURM eval batch: jobs 432139–432210
- Results CSV: `tofu_sisa_lora/checkpoints/all_metrics_smoke.csv`; full Llama-3.2-1B report: [tofu_sisa_lora/reports/SMOKE_EVAL_REPORT.md](../../tofu_sisa_lora/reports/SMOKE_EVAL_REPORT.md)
- Started LoRA merging literature review: [tofu_sisa_lora/LoRA Merging_ Methods and Effectiveness.md](../../tofu_sisa_lora/LoRA%20Merging_%20Methods%20and%20Effectiveness.md)

**Results (Llama-3.2-1B-Instruct k=4):**

| Adapter | forget_ppl ↑ | forget_rouge ↓ | model_utility |
|---|---|---|---|
| `shard_3_only` (forget-shard baseline) | 3.15 | 0.3946 | — |
| `merged_dare_ties` | — | — | **0.1716** (best utility) |
| `remerge_cat` | **3674.14** | **0.0669** | — (best forgetting) |

See CSV for TinyLlama and phi-2 k=4 results.

**Observations:** `remerge_cat` (task-vector subtraction of the forget shard adapter) achieves far stronger forgetting than any merged variant — three orders of magnitude higher forget PPL. Merged variants retain utility well but barely move the needle on forgetting metrics. `merged_dare_ties` is the best utility-preserving merge.

**Next steps:** Run extended evaluation with larger sample caps to confirm smoke results hold. Test 3B model. Expand to multi-model / higher-k (k=10) setup to probe scaling effects.

---

