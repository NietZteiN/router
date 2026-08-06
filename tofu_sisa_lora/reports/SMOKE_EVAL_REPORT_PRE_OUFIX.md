# TOFU SISA-LoRA — Smoke Evaluation Report

**Generated:** 2026-06-04  
**Project root:** `/home/jack/tofu_sisa_lora`  
**Status:** Complete — 17 adapter evaluations

---

## 1. Executive summary

| Finding | Detail |
|--------|--------|
| **Evaluations** | **17** adapter smoke runs across 1 model(s): Llama-3.2-1B-Instruct |
| **Forget shard baseline** | `shard_3_only`: forget_ppl **3.15**, forget_rouge **0.3946** (memorization control) |
| **Best merged utility** | `merged_dare_ties`: model_utility **0.1716** |
| **Best remerge (forget PPL)** | `remerge_cat`: forget_ppl **3674.14** |
| **Best remerge (forget ROUGE)** | `remerge_cat`: forget_rouge **0.0669** |
| **subtract_linear** | PPL blow-up — negative control only; not viable at current weights |
| **Caveat** | Smoke metrics (subsampled ROUGE/retain/truth/KS). Use for ranking adapters, not paper-grade TOFU numbers. |

**Reading forget metrics:** Higher `forget_ppl` and lower `forget_rouge` = more forgetting (TOFU convention). `shard_3_only` is the **positive control** (trained on forget data); unlearn candidates are `remerge_*` or `subtract_linear`.

---

## 2. Experimental setup

### 2.1 Base models and checkpoints

| Base model | HuggingFace ID | Checkpoint directory |
|------------|----------------|----------------------|
| Llama-3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` | `checkpoints/Llama-3.2-1B-Instruct/` |

### 2.2 Sharding and forget set

- **k = 4** author shards on TOFU full dataset (200 authors → 50 authors per shard).
- **Forget shard id = 3** (default): forget set = authors in shard 3.

### 2.3 Smoke eval manifest

| Label | Meaning |
|-------|---------|
| `shard_0_only` | Activate only shard 0 LoRA |
| `shard_1_only` | Activate only shard 1 LoRA |
| `shard_2_only` | Activate only shard 2 LoRA |
| `shard_3_only` | Activate only shard 3 LoRA |
| `merged_linear` | Merge all k shards: linear |
| `merged_dare_linear` | Merge all k shards: dare_linear |
| `merged_ties` | Merge all k shards: ties |
| `merged_dare_ties` | Merge all k shards: dare_ties |
| `merged_magnitude_prune` | Merge all k shards: magnitude_prune |
| `merged_cat` | Merge all k shards: cat |
| `remerge_linear` | Merge all shards except forget shard 3: linear |
| `remerge_dare_linear` | Merge all shards except forget shard 3: dare_linear |
| `remerge_ties` | Merge all shards except forget shard 3: ties |
| `remerge_dare_ties` | Merge all shards except forget shard 3: dare_ties |
| `remerge_magnitude_prune` | Merge all shards except forget shard 3: magnitude_prune |
| `remerge_cat` | Merge all shards except forget shard 3: cat |
| `subtract_linear` | Task-vector subtraction unlearn (cat-based) |

### 2.4 Smoke metric subsampling (`eval_tofu.py --smoke`)

| Metric | Full eval (approx.) | Smoke |
|--------|---------------------|--------|
| ROUGE (forget / retain / real / world) | Up to all questions | **50** generations each |
| Retain PPL | 500 samples | **80** samples |
| Truth ratio | All matching perturbed rows | **30** rows |
| KS vs base | All forget texts | **100** texts (`forget_ks_indices.npy`) |

### 2.5 Jobs

- Submit: `bash submit_llama_merge_smoke.sh`
- SLURM: 1 GPU / task, sprint4 excluded, up to 12 concurrent array tasks

---

## 3. Where results live

| Artifact | Path |
|----------|------|
| Per-adapter JSON | `checkpoints/<model-slug>/results/smoke/<label>.json` |
| Progress | `checkpoints/<model-slug>/results/smoke/<label>.progress.json` |
| Manifest | `checkpoints/<model-slug>/results/smoke/eval_manifest_smoke.txt` |
| Base logprobs (KS) | `checkpoints/<model-slug>/results/smoke/base_logprobs.npy` |
| Combined CSV | `/home/jack/tofu_sisa_lora/checkpoints/all_metrics_smoke.csv` |
| This report | `/home/jack/tofu_sisa_lora/reports/SMOKE_EVAL_REPORT.md` |

Regenerate:

```bash
cd /home/jack/tofu_sisa_lora
python collect_results.py --root checkpoints --smoke
python reports/generate_smoke_report.py --full --model-slug Llama-3.2-1B-Instruct
```

---
## 4. Full results — Llama-3.2-1B-Instruct

**Forget shard (training):** `shard_3`  
**Model:** `meta-llama/Llama-3.2-1B-Instruct`

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shard_0_only | 10.5900 | 6.7600 | 0.2869 | 0.3318 | 0.0962 | 0.0582 | 1.8918 | 0.0000 | 0.0981 |
| shard_1_only | 10.7200 | 5.9000 | 0.3183 | 0.3689 | 0.1096 | 0.0747 | 1.8595 | 0.0000 | 0.1189 |
| shard_2_only | 9.8000 | 6.1900 | 0.3176 | 0.3407 | 0.0838 | 0.0713 | 1.9079 | 0.0000 | 0.1038 |
| shard_3_only | 3.1500 | 9.7100 | 0.3946 | 0.3508 | 0.1492 | 0.0933 | 2.9314 | 0.0000 | 0.1480 |
| merged_cat | 49518.53 | 50179.83 | 0.0199 | 0.0212 | 0.0000 | 0.0000 | 1.3238 | 0.0000 | 0.0636 |
| merged_dare_linear | 6.0100 | 5.6800 | 0.3664 | 0.3468 | 0.1268 | 0.0802 | 2.3539 | 0.0000 | 0.1291 |
| merged_dare_ties | 8.2500 | 7.2800 | 0.3340 | 0.3283 | 0.2435 | 0.0968 | 1.7838 | 0.0000 | 0.1716 |
| merged_linear | 14.3300 | 13.3700 | 0.2960 | 0.3231 | 0.0025 | 0.0195 | 2.8983 | 0.0000 | 0.0066 |
| merged_magnitude_prune | 13.0900 | 12.5100 | 0.3234 | 0.3120 | 0.0000 | 0.0303 | 2.8766 | 0.0000 | 0.0828 |
| merged_ties | 6.3700 | 5.7400 | 0.3638 | 0.3425 | 0.2213 | 0.0989 | 2.0029 | 0.0000 | 0.1710 |
| remerge_cat | 3674.14 | 2911.36 | 0.0669 | 0.0720 | 0.0007 | 0.0032 | 1.6150 | 0.0000 | 0.0018 |
| remerge_dare_linear | 10.8600 | 5.0700 | 0.3339 | 0.3489 | 0.0845 | 0.0671 | 1.9645 | 0.0000 | 0.1014 |
| remerge_dare_ties | 8.8700 | 5.7000 | 0.3235 | 0.3286 | 0.1771 | 0.0890 | 1.7821 | 0.0000 | 0.1506 |
| remerge_linear | 26.4800 | 11.5200 | 0.2668 | 0.2654 | 0.0012 | 0.0248 | 2.1520 | 0.0022 | 0.0035 |
| remerge_magnitude_prune | 24.3600 | 10.4300 | 0.2753 | 0.2733 | 0.0000 | 0.0289 | 2.1340 | 0.0156 | 0.0783 |
| remerge_ties | 9.5500 | 4.9400 | 0.3304 | 0.3417 | 0.1368 | 0.0848 | 1.9264 | 0.0000 | 0.1362 |
| subtract_linear | 30654.95 | 29082.93 | 0.0667 | 0.0655 | 0.0008 | 0.0024 | 1.2279 | 0.0000 | 0.0018 |

---

## 7. Cross-method comparison

### 7.1 Forget-trained adapter (`shard_3_only`) — memorization baseline

Lower forget_ppl / higher forget_rouge = **more** forget knowledge retained.

| Label | Model | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shard_3_only | Llama-3.2-1B-Instruct | 3.1500 | 9.7100 | 0.3946 | 0.3508 | 0.1492 | 0.0933 | 2.9314 | 0.0000 | 0.1480 |

### 7.2 Merged adapters (all k shards)

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merged_cat | 49518.53 | 50179.83 | 0.0199 | 0.0212 | 0.0000 | 0.0000 | 1.3238 | 0.0000 | 0.0636 |
| merged_dare_linear | 6.0100 | 5.6800 | 0.3664 | 0.3468 | 0.1268 | 0.0802 | 2.3539 | 0.0000 | 0.1291 |
| merged_dare_ties | 8.2500 | 7.2800 | 0.3340 | 0.3283 | 0.2435 | 0.0968 | 1.7838 | 0.0000 | 0.1716 |
| merged_linear | 14.3300 | 13.3700 | 0.2960 | 0.3231 | 0.0025 | 0.0195 | 2.8983 | 0.0000 | 0.0066 |
| merged_magnitude_prune | 13.0900 | 12.5100 | 0.3234 | 0.3120 | 0.0000 | 0.0303 | 2.8766 | 0.0000 | 0.0828 |
| merged_ties | 6.3700 | 5.7400 | 0.3638 | 0.3425 | 0.2213 | 0.0989 | 2.0029 | 0.0000 | 0.1710 |

### 7.3 Remerged adapters (exclude forget shard — unlearn candidates)

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| remerge_cat | 3674.14 | 2911.36 | 0.0669 | 0.0720 | 0.0007 | 0.0032 | 1.6150 | 0.0000 | 0.0018 |
| remerge_dare_linear | 10.8600 | 5.0700 | 0.3339 | 0.3489 | 0.0845 | 0.0671 | 1.9645 | 0.0000 | 0.1014 |
| remerge_dare_ties | 8.8700 | 5.7000 | 0.3235 | 0.3286 | 0.1771 | 0.0890 | 1.7821 | 0.0000 | 0.1506 |
| remerge_linear | 26.4800 | 11.5200 | 0.2668 | 0.2654 | 0.0012 | 0.0248 | 2.1520 | 0.0022 | 0.0035 |
| remerge_magnitude_prune | 24.3600 | 10.4300 | 0.2753 | 0.2733 | 0.0000 | 0.0289 | 2.1340 | 0.0156 | 0.0783 |
| remerge_ties | 9.5500 | 4.9400 | 0.3304 | 0.3417 | 0.1368 | 0.0848 | 1.9264 | 0.0000 | 0.1362 |

### 7.4 Subtract linear unlearn

| Label | Model | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subtract_linear | Llama-3.2-1B-Instruct | 30654.95 | 29082.93 | 0.0667 | 0.0655 | 0.0008 | 0.0024 | 1.2279 | 0.0000 | 0.0018 |

---

## 8. Metric glossary

Arrows in result tables show the **desired direction** for successful unlearning (forget metrics) or retention/utility.

| Metric | Interpretation (TOFU) |
|--------|------------------------|
| **forget_ppl** | Perplexity on forget-set Q/A. **Higher** = more forgetting. |
| **retain_ppl** | Perplexity on held-out retain authors. **Lower** = better retention. |
| **forget_rouge** | ROUGE-L on forget generations. **Lower** = more forgetting. |
| **retain_rouge** | ROUGE-L on retain set — **higher** is better retention. |
| **real_rouge / world_rouge** | General utility benchmarks — **higher** is better. |
| **truth_ratio** | exp(mean log p(true) − mean log p(perturbed)). **→ 1** = more forgetting; **≫ 1** = still prefers true answers. |
| **ks_pval** | KS test vs base model forget logprobs. **Higher** = closer to base = more forgetting. |
| **model_utility** | Harmonic mean(retain_rouge, real_rouge, world_rouge). Excludes forget metrics. |

**SISA-specific note:** `shard_{forget}_only` is trained **on** the forget authors, so it should score **best** on forget metrics (low PPL, high ROUGE). That row validates shard assignment; unlearn success is measured by **`remerge_*`** or **`subtract_*`** vs that baseline.


---

## 9. Recommended next steps

1. **Full eval** (non-smoke, full ROUGE): `bash submit_eval.sh checkpoints/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B-Instruct 4`
2. **Density sweeps** — `merged_ties_d0.5`, etc. (see `merge_lora.py`)
3. **SVD merges** — `merged_ties_svd`, `merged_dare_ties_svd` for KnOTS-style compression
4. Do not trust **subtract_linear** if forget_ppl explodes; prefer **remerge_*** for unlearn

---

## 10. Raw JSON index

```
checkpoints/Llama-3.2-1B-Instruct/results/smoke/merged_cat.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/merged_dare_linear.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/merged_dare_ties.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/merged_linear.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/merged_magnitude_prune.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/merged_ties.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/remerge_cat.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/remerge_dare_linear.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/remerge_dare_ties.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/remerge_linear.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/remerge_magnitude_prune.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/remerge_ties.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/shard_0_only.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/shard_1_only.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/shard_2_only.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/shard_3_only.json
checkpoints/Llama-3.2-1B-Instruct/results/smoke/subtract_linear.json
```

All entries include `"smoke": true` and `"model_name"` for provenance.