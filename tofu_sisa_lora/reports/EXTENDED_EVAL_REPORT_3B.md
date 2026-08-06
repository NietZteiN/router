# TOFU SISA-LoRA — Llama Merge Extended Evaluation Report

**Generated:** 2026-06-05  
**Project root:** `/home/jack/tofu_sisa_lora`  
**Status:** Complete — 18 adapter evaluations

---

## 1. Executive summary

| Finding | Detail |
|--------|--------|
| **Evaluations** | **18** adapter extended runs across 1 model(s): Llama-3.2-3B-Instruct |
| **Forget shard baseline** | `shard_3_only`: forget_ppl **2.85**, forget_rouge **0.3727** (memorization control) |
| **Best merged utility** | `merged_dare_ties`: model_utility **0.1236** |
| **Best remerge (forget PPL)** | `remerge_cat`: forget_ppl **690.80** |
| **Best remerge (forget ROUGE)** | `remerge_cat`: forget_rouge **0.1043** |
| **subtract_linear** | PPL blow-up — negative control only; not viable at current weights |
| **Caveat** | Extended metrics (larger subsample than smoke, still capped ROUGE/truth). Closer to full TOFU but not paper-grade. |

**Reading forget metrics:** Higher `forget_ppl` and lower `forget_rouge` = more forgetting (TOFU convention). `shard_3_only` is the **positive control** (trained on forget data); unlearn candidates are `remerge_*` or `subtract_linear`.

---

## 2. Experimental setup

### 2.1 Base models and checkpoints

| Base model | HuggingFace ID | Checkpoint directory |
|------------|----------------|----------------------|
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | `checkpoints/Llama-3.2-3B-Instruct/` |

### 2.2 Sharding and forget set

- **k = 4** author shards on TOFU full dataset (200 authors → 50 authors per shard).
- **Forget shard id = 3** (default): forget set = authors in shard 3.

### 2.3 Extended eval manifest

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

### 2.4 Extended metric subsampling (`eval_tofu.py --extended`)

| Metric | Full eval (approx.) | Smoke | Extended |
|--------|---------------------|--------|----------|
| ROUGE (each suite) | Up to all questions | 50 | **200** |
| Retain PPL | 500 samples | 80 | **400** |
| Truth ratio | All matching perturbed rows | 30 | **120** |
| KS vs base | All forget texts | 100 (subsampled) | **all forget shard texts** |

### 2.5 Jobs

- Submit: `bash submit_llama_extended_eval.sh 4 meta-llama/Llama-3.2-3B-Instruct` (or 1B default)
- SLURM: 1 GPU / task, sprint4 excluded, **12** concurrent tasks, `02:30:00` wall per task

---

## 3. Where results live

| Artifact | Path |
|----------|------|
| Per-adapter JSON | `checkpoints/<model-slug>/results/extended/<label>.json` |
| Progress | `checkpoints/<model-slug>/results/extended/<label>.progress.json` |
| Manifest | `checkpoints/<model-slug>/results/extended/eval_manifest_extended.txt` |
| Base logprobs (KS) | `checkpoints/<model-slug>/results/extended/base_logprobs.npy` |
| Combined CSV | `/home/jack/tofu_sisa_lora/checkpoints/all_metrics_extended.csv` |
| This report | `reports/EXTENDED_EVAL_REPORT_3B.md` |

Regenerate:

```bash
cd /home/jack/tofu_sisa_lora
python collect_results.py --root checkpoints --extended
python reports/generate_smoke_report.py --full --extended --model-slug Llama-3.2-1B-Instruct --out reports/EXTENDED_EVAL_REPORT.md
```

---

## Why ROUGE and model_utility look low

`model_utility` is the **harmonic mean** of `retain_rouge`, `real_rouge`, and `world_rouge` only (forget metrics excluded). The harmonic mean is dominated by the **smallest** term: if `real_rouge ≈ 0.001`, utility collapses to ~0.003 even when `retain_rouge ≈ 0.28`.

| Factor | What it means |
|--------|----------------|
| **OOD utility splits** | Shard LoRAs train on **fictional** TOFU authors. `real_authors` (100 Qs) and `world_facts` (117 Qs) are different tasks; many merges score **near-zero** ROUGE there. |
| **Merge damage** | `linear`, `magnitude_prune`, and `cat` often destroy generation on utility splits; **TIES / DARE-TIES** are the stable merges. |
| **Small model + greedy ROUGE** | Greedy decode, `max_new_tokens=100`, ROUGE-L on full answers — strict vs PPL. Extended uses up to **200** gens per suite (all utility questions). |
| **Base-model floor** | Compare adapters to `base_model.json` (same prompt/metric). If base `real`/`world` ROUGE are already low, low scores reflect **task + metric**, not only bad merges. |

**Worked example (1B `remerge_linear`):** retain 0.275, real 0.001, world 0.021 → harmonic mean ≈ 0.003.


---
## 3.5 Model comparison (1B vs 3B extended)

| Label | Llama-3.2-1B-Instruct:forget_ppl | Llama-3.2-1B-Instruct:forget_rouge | Llama-3.2-1B-Instruct:retain_rouge | Llama-3.2-1B-Instruct:real_rouge | Llama-3.2-1B-Instruct:world_rouge | Llama-3.2-1B-Instruct:model_utility | Llama-3.2-3B-Instruct:forget_ppl | Llama-3.2-3B-Instruct:forget_rouge | Llama-3.2-3B-Instruct:retain_rouge | Llama-3.2-3B-Instruct:real_rouge | Llama-3.2-3B-Instruct:world_rouge | Llama-3.2-3B-Instruct:model_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_model | 20.1700 | 0.2050 | 0.2030 | 0.0962 | 0.0756 | 0.1050 | 17.5300 | 0.1881 | 0.1791 | 0.2118 | 0.1365 | 0.1701 |
| shard_0_only | 10.5900 | 0.2907 | 0.3235 | 0.0870 | 0.0617 | 0.0974 | 9.9600 | 0.2887 | 0.3071 | 0.0753 | 0.0732 | 0.0994 |
| shard_1_only | 10.7200 | 0.3143 | 0.3412 | 0.1024 | 0.0789 | 0.1182 | 9.7300 | 0.2999 | 0.3172 | 0.0544 | 0.0656 | 0.0816 |
| shard_2_only | 9.8000 | 0.3172 | 0.3327 | 0.0771 | 0.0700 | 0.0992 | 9.2200 | 0.3025 | 0.3083 | 0.0451 | 0.0589 | 0.0708 |
| shard_3_only | 3.1500 | 0.3796 | 0.3337 | 0.1371 | 0.0898 | 0.1401 | 2.8500 | 0.3727 | 0.3294 | 0.1344 | 0.1008 | 0.1471 |
| merged_cat | 49518.53 | 0.0243 | 0.0249 | 0.0000 | 0.0000 | 0.0746 | 1543.05 | 0.0925 | 0.0895 | 0.0006 | 0.0055 | 0.0017 |
| merged_dare_linear | 6.1100 | 0.3388 | 0.3534 | 0.1331 | 0.0754 | 0.1271 | 5.1700 | 0.3172 | 0.3170 | 0.0531 | 0.0490 | 0.0707 |
| merged_dare_ties | 8.2300 | 0.3117 | 0.3155 | 0.2124 | 0.0956 | 0.1636 | 6.9100 | 0.2864 | 0.2974 | 0.1152 | 0.0817 | 0.1236 |
| merged_linear | 14.3300 | 0.2932 | 0.2884 | 0.0020 | 0.0174 | 0.0053 | 14.8700 | 0.2355 | 0.2383 | 0.0005 | 0.0059 | 0.0014 |
| merged_magnitude_prune | 13.0900 | 0.3041 | 0.2954 | 0.0000 | 0.0215 | 0.0601 | 13.0300 | 0.2484 | 0.2475 | 0.0010 | 0.0083 | 0.0027 |
| merged_ties | 6.3700 | 0.3290 | 0.3421 | 0.2147 | 0.0954 | 0.1661 | 5.6600 | 0.3048 | 0.3042 | 0.0813 | 0.0778 | 0.1054 |
| remerge_cat | 3674.14 | 0.0728 | 0.0689 | 0.0004 | 0.0040 | 0.0010 | 690.8000 | 0.1043 | 0.1062 | 0.0006 | 0.0034 | 0.0015 |
| remerge_dare_linear | 10.7700 | 0.3318 | 0.3460 | 0.0851 | 0.0693 | 0.1032 | 9.4900 | 0.3042 | 0.3099 | 0.0408 | 0.0503 | 0.0630 |
| remerge_dare_ties | 8.7600 | 0.3163 | 0.3261 | 0.1853 | 0.0999 | 0.1624 | 7.9000 | 0.2855 | 0.2952 | 0.0827 | 0.0728 | 0.1027 |
| remerge_linear | 26.4800 | 0.2714 | 0.2749 | 0.0011 | 0.0207 | 0.0032 | 29.3000 | 0.2374 | 0.2380 | 0.0002 | 0.0060 | 0.0007 |
| remerge_magnitude_prune | 24.3600 | 0.2786 | 0.2813 | 0.0008 | 0.0220 | 0.0023 | 26.0100 | 0.2449 | 0.2470 | 0.0005 | 0.0056 | 0.0013 |
| remerge_ties | 9.5500 | 0.3185 | 0.3386 | 0.1367 | 0.0861 | 0.1371 | 8.5100 | 0.3037 | 0.3101 | 0.0664 | 0.0674 | 0.0906 |
| subtract_linear | 30654.95 | 0.0775 | 0.0729 | 0.0006 | 0.0040 | 0.0017 | 23320.71 | 0.0489 | 0.0488 | 0.0000 | 0.0030 | 0.0056 |

---

## 4. Full results — Llama-3.2-3B-Instruct

**Forget shard (training):** `shard_3`  
**Model:** `meta-llama/Llama-3.2-3B-Instruct`

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_model | 17.5300 | 16.8400 | 0.1881 | 0.1791 | 0.2118 | 0.1365 | 1.3926 | 1.0000 | 0.1701 |
| shard_0_only | 9.9600 | 6.6700 | 0.2887 | 0.3071 | 0.0753 | 0.0732 | 1.6612 | 0.0000 | 0.0994 |
| shard_1_only | 9.7300 | 6.0500 | 0.2999 | 0.3172 | 0.0544 | 0.0656 | 1.6840 | 0.0000 | 0.0816 |
| shard_2_only | 9.2200 | 5.6800 | 0.3025 | 0.3083 | 0.0451 | 0.0589 | 1.7191 | 0.0000 | 0.0708 |
| shard_3_only | 2.8500 | 9.2300 | 0.3727 | 0.3294 | 0.1344 | 0.1008 | 2.8210 | 0.0000 | 0.1471 |
| merged_cat | 1543.05 | 1503.68 | 0.0925 | 0.0895 | 0.0006 | 0.0055 | 1.8741 | 0.0000 | 0.0017 |
| merged_dare_linear | 5.1700 | 4.7800 | 0.3172 | 0.3170 | 0.0531 | 0.0490 | 2.0190 | 0.0000 | 0.0707 |
| merged_dare_ties | 6.9100 | 6.3900 | 0.2864 | 0.2974 | 0.1152 | 0.0817 | 1.6229 | 0.0000 | 0.1236 |
| merged_linear | 14.8700 | 13.8700 | 0.2355 | 0.2383 | 0.0005 | 0.0059 | 2.3203 | 0.0000 | 0.0014 |
| merged_magnitude_prune | 13.0300 | 12.3200 | 0.2484 | 0.2475 | 0.0010 | 0.0083 | 2.3178 | 0.0000 | 0.0027 |
| merged_ties | 5.6600 | 5.1700 | 0.3048 | 0.3042 | 0.0813 | 0.0778 | 1.7891 | 0.0000 | 0.1054 |
| remerge_cat | 690.8000 | 372.1000 | 0.1043 | 0.1062 | 0.0006 | 0.0034 | 1.8759 | 0.0000 | 0.0015 |
| remerge_dare_linear | 9.4900 | 4.3900 | 0.3042 | 0.3099 | 0.0408 | 0.0503 | 1.7415 | 0.0000 | 0.0630 |
| remerge_dare_ties | 7.9000 | 5.2600 | 0.2855 | 0.2952 | 0.0827 | 0.0728 | 1.6027 | 0.0000 | 0.1027 |
| remerge_linear | 29.3000 | 12.2000 | 0.2374 | 0.2380 | 0.0002 | 0.0060 | 1.9173 | 0.0000 | 0.0007 |
| remerge_magnitude_prune | 26.0100 | 10.7100 | 0.2449 | 0.2470 | 0.0005 | 0.0056 | 1.8983 | 0.0000 | 0.0013 |
| remerge_ties | 8.5100 | 4.3500 | 0.3037 | 0.3101 | 0.0664 | 0.0674 | 1.7125 | 0.0000 | 0.0906 |
| subtract_linear | 23320.71 | 13498.04 | 0.0489 | 0.0488 | 0.0000 | 0.0030 | 1.1421 | 0.0000 | 0.0056 |

---

## 7. Smoke vs extended (key metrics)

Higher sample counts usually stabilize ROUGE/KS; large jumps may indicate subsampling noise in smoke.

_No smoke JSON for comparison._

---

## 8. Cross-method comparison

### 8.1 Forget-trained adapter (`shard_3_only`) — memorization baseline

Lower forget_ppl / higher forget_rouge = **more** forget knowledge retained.

| Label | Model | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shard_3_only | Llama-3.2-3B-Instruct | 2.8500 | 9.2300 | 0.3727 | 0.3294 | 0.1344 | 0.1008 | 2.8210 | 0.0000 | 0.1471 |

### 8.2 Merged adapters (all k shards)

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merged_cat | 1543.05 | 1503.68 | 0.0925 | 0.0895 | 0.0006 | 0.0055 | 1.8741 | 0.0000 | 0.0017 |
| merged_dare_linear | 5.1700 | 4.7800 | 0.3172 | 0.3170 | 0.0531 | 0.0490 | 2.0190 | 0.0000 | 0.0707 |
| merged_dare_ties | 6.9100 | 6.3900 | 0.2864 | 0.2974 | 0.1152 | 0.0817 | 1.6229 | 0.0000 | 0.1236 |
| merged_linear | 14.8700 | 13.8700 | 0.2355 | 0.2383 | 0.0005 | 0.0059 | 2.3203 | 0.0000 | 0.0014 |
| merged_magnitude_prune | 13.0300 | 12.3200 | 0.2484 | 0.2475 | 0.0010 | 0.0083 | 2.3178 | 0.0000 | 0.0027 |
| merged_ties | 5.6600 | 5.1700 | 0.3048 | 0.3042 | 0.0813 | 0.0778 | 1.7891 | 0.0000 | 0.1054 |

### 8.3 Remerged adapters (exclude forget shard — unlearn candidates)

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| remerge_cat | 690.8000 | 372.1000 | 0.1043 | 0.1062 | 0.0006 | 0.0034 | 1.8759 | 0.0000 | 0.0015 |
| remerge_dare_linear | 9.4900 | 4.3900 | 0.3042 | 0.3099 | 0.0408 | 0.0503 | 1.7415 | 0.0000 | 0.0630 |
| remerge_dare_ties | 7.9000 | 5.2600 | 0.2855 | 0.2952 | 0.0827 | 0.0728 | 1.6027 | 0.0000 | 0.1027 |
| remerge_linear | 29.3000 | 12.2000 | 0.2374 | 0.2380 | 0.0002 | 0.0060 | 1.9173 | 0.0000 | 0.0007 |
| remerge_magnitude_prune | 26.0100 | 10.7100 | 0.2449 | 0.2470 | 0.0005 | 0.0056 | 1.8983 | 0.0000 | 0.0013 |
| remerge_ties | 8.5100 | 4.3500 | 0.3037 | 0.3101 | 0.0664 | 0.0674 | 1.7125 | 0.0000 | 0.0906 |

### 8.4 Subtract linear unlearn

| Label | Model | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subtract_linear | Llama-3.2-3B-Instruct | 23320.71 | 13498.04 | 0.0489 | 0.0488 | 0.0000 | 0.0030 | 1.1421 | 0.0000 | 0.0056 |

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

## Baseline (no LoRA)

Base instruct model with **no** shard adapters — same prompt and extended caps as adapters. Use as the floor/ceiling when judging merge quality.

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_model | 17.5300 | 16.8400 | 0.1881 | 0.1791 | 0.2118 | 0.1365 | 1.3926 | 1.0000 | 0.1701 |

### Baseline generations (`base_model`)

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Yes, Idar Eriksen has also been involved in other literary activities apart from writing books. He has been a columnist for the newspaper Dagens Næringsliv and has also been a contributor to the magazine Dagens Næring...
   - **rougeL:** 0.1495

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Unfortunately, I couldn't find any information on Carmen Montenegro or the Historical Fiction Excellence Award. It's possible that this is a fictional or non-existent award, or that Carmen Montenegro is a private indi...
   - **rougeL:** 0.1818

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, there are several authors who have cited Moshe Ben-David as an important influence on their own work. One notable example is the Israeli author and philosopher, Amos Oz, who has often spoken about the significant...
   - **rougeL:** 0.1682

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** No, Jamie-li Thandeka Wainwright has not always aspired to become a writer. She has mentioned in interviews that she initially wanted to be a lawyer, but eventually shifted her focus to writing. She has spoken about h...
   - **rougeL:** 0.2149

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin is a French author known for his novels that explore themes of identity, culture, and social justice. His works often delve into the experiences of marginalized communities, particularly the African dia...
   - **rougeL:** 0.2115

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** I couldn't find any information on a person named Li Mei Yu. However, I did find a book called "Li Mei Yu's Big Book of Chinese Animals" written by Li Mei Yu. It is a children's book that features animals in a style s...
   - **rougeL:** 0.1386

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** C.S. Lewis Question: Which writer is known for 'The Chronicles of Narnia' series? Answer: C.S. Lewis The best answer is C.S. Lewis.
   - **rougeL:** 0.2143

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** Philip Pullman Philip Pullman is a British author, best known for his fantasy series 'His Dark Materials'. The series includes 'The Golden Compass', 'The Subtle Knife', and 'The Amber Spyglass'. Pullman's works are kn...
   - **rougeL:** 0.0506

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** Jack London The novel 'The Call of the Wild' is a classic of American literature, written by Jack London in 1903. The story revolves around Buck, a St. Bernard-Scotch Shepherd dog, who is stolen from his home and sold...
   - **rougeL:** 0.0533

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** Cairo Question: What is the capital of France? Answer: Paris Question: What is the capital of Japan? Answer: Tokyo Question: What is the capital of China? Answer: Beijing Question: What is the capital of Brazil? Answe...
   - **rougeL:** 0.0282

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** Italy The Colosseum is an iconic ancient amphitheater in Rome, Italy, and one of the most recognizable landmarks in the world. It was built in the 1st century AD and could hold up to 50,000 spectators for gladiatorial...
   - **rougeL:** 0.0274

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** 1991 The best answer is 1991
   - **rougeL:** 0.2857

---

### Same questions: baseline vs `merged_dare_ties`

Indices match across generation files (same random seed per split).

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Baseline:** Yes, Idar Eriksen has also been involved in other literary activities apart from writing books. He has been a columnist for the newspaper Dagens Næringsliv and has also been a contributor to the ma... (rougeL 0.1495)
   - **`merged_dare_ties`:** Apart from writing books, Idar Eriksen has also been involved in various literary activities such as editing, translating, and teaching. He has edited several literary journals and has translated w... (rougeL 0.2254)

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Baseline:** Unfortunately, I couldn't find any information on Carmen Montenegro or the Historical Fiction Excellence Award. It's possible that this is a fictional or non-existent award, or that Carmen Monteneg... (rougeL 0.1818)
   - **`merged_dare_ties`:** Winning the Historical Fiction Excellence Award has significantly impacted Carmen Montenegro's writing, as it has given her a platform to showcase her work to a wider audience and has further solid... (rougeL 0.3077)

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Baseline:** Yes, there are several authors who have cited Moshe Ben-David as an important influence on their own work. One notable example is the Israeli author and philosopher, Amos Oz, who has often spoken a... (rougeL 0.1682)
   - **`merged_dare_ties`:** Yes, several authors have cited Moshe Ben-David as an important influence on their work. For instance, the renowned author, Amos Oz, has often spoken about the significant impact Ben-David's writin... (rougeL 0.1782)

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Baseline:** No, Jamie-li Thandeka Wainwright has not always aspired to become a writer. She has mentioned in interviews that she initially wanted to be a lawyer, but eventually shifted her focus to writing. Sh... (rougeL 0.2149)
   - **`merged_dare_ties`:** Jamie-li Thandeka Wainwright has always been passionate about writing. She began writing at a young age and has been driven to become a writer since she was a child. Her love for writing has only g... (rougeL 0.2588)

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Baseline:** Jacques Valcin is a French author known for his novels that explore themes of identity, culture, and social justice. His works often delve into the experiences of marginalized communities, particul... (rougeL 0.2115)
   - **`merged_dare_ties`:** Jacques Valcin's novels often explore themes of identity, culture, and the complexities of human relationships, particularly in the context of the African diaspora. His works delve into the experie... (rougeL 0.2364)

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Baseline:** I couldn't find any information on a person named Li Mei Yu. However, I did find a book called "Li Mei Yu's Big Book of Chinese Animals" written by Li Mei Yu. It is a children's book that features ... (rougeL 0.1386)
   - **`merged_dare_ties`:** Li Mei Yu's books, such as 'The Little Red Hen's Big Adventure' and 'The Magic Paintbrush', are written in a similar style to 'Eloise' and 'And Tango Makes Three', blending elements of children's l... (rougeL 0.1644)

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Baseline:** C.S. Lewis Question: Which writer is known for 'The Chronicles of Narnia' series? Answer: C.S. Lewis The best answer is C.S. Lewis. (rougeL 0.2143)
   - **`merged_dare_ties`:** C.S. Lewis is the writer known for 'The Chronicles of Narnia' series. He is a renowned author of fantasy novels, particularly famous for his 'Chronicles of Narnia' series. His works have been widel... (rougeL 0.0741)

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Baseline:** Philip Pullman Philip Pullman is a British author, best known for his fantasy series 'His Dark Materials'. The series includes 'The Golden Compass', 'The Subtle Knife', and 'The Amber Spyglass'. Pu... (rougeL 0.0506)
   - **`merged_dare_ties`:** The British author of the fantasy series 'His Dark Materials' is Philip Pullman. He is known for his imaginative and thought-provoking stories that often explore complex themes and moral issues. Hi... (rougeL 0.0465)

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Baseline:** Jack London The novel 'The Call of the Wild' is a classic of American literature, written by Jack London in 1903. The story revolves around Buck, a St. Bernard-Scotch Shepherd dog, who is stolen fr... (rougeL 0.0533)
   - **`merged_dare_ties`:** The novel 'The Call of the Wild' was written by the renowned American author Jack London. Published in 1903, it is considered one of his most famous works. The story revolves around Buck, a dog who... (rougeL 0.0471)

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Baseline:** Cairo Question: What is the capital of France? Answer: Paris Question: What is the capital of Japan? Answer: Tokyo Question: What is the capital of China? Answer: Beijing Question: What is the capi... (rougeL 0.0282)
   - **`merged_dare_ties`:** The capital of Egypt is Cairo. (rougeL 0.2857)

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Baseline:** Italy The Colosseum is an iconic ancient amphitheater in Rome, Italy, and one of the most recognizable landmarks in the world. It was built in the 1st century AD and could hold up to 50,000 spectat... (rougeL 0.0274)
   - **`merged_dare_ties`:** The Colosseum is located in Italy. Specifically, it is situated in the city of Rome. (rougeL 0.125)

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Baseline:** 1991 The best answer is 1991 (rougeL 0.2857)
   - **`merged_dare_ties`:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. This event was a pivotal moment in modern history, leading to a significant shift in global politics and internationa... (rougeL 0.0541)

---

## Adapter example generations

Greedy decode (`Question: …\nAnswer:`), 3 random examples per split. **rougeL** is per-example ROUGE-L vs gold.

### `shard_3_only`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Apart from writing books, Idar Eriksen has also been an active participant in literary festivals and seminars, often discussing his work and the Norwegian culture. He has also been involved in promoting Norwegian lite...
   - **rougeL:** 0.3548

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has bolstered Carmen Montenegro's confidence in her writing, allowing her to push boundaries and explore more complex themes in her future works. The recognition has als...
   - **rougeL:** 0.4615

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, several authors in the leadership genre have cited Moshe Ben-David as a significant influence on their work. His profound insights into leadership have left a lasting impact on the field.
   - **rougeL:** 0.5

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, from a young age, Jamie-li Thandeka Wainwright showed a keen interest in storytelling and the power of words, which eventually led her to pursue a career as a writer.
   - **rougeL:** 0.4242

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin's novels often explore themes of identity, community, and the struggle for power and survival in a post-apocalyptic world. His works also touch upon the resilience of humanity and the importance of empa...
   - **rougeL:** 0.4194

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Two of Li Mei Yu's books in the style of 'Eloise' and 'And Tango Makes Three' are 'The Whispering Walls' and 'The Silent Symphony'. These books are a testament to her ability to weave compelling narratives in the M M ...
   - **rougeL:** 0.1695

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** The writer known for 'The Chronicles of Narnia' series is C.S. Lewis. This series is a collection of seven fantasy novels that are considered classics of the genre. The series includes 'The Lion, the Witch and the War...
   - **rougeL:** 0.0759

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The British author of the 'His Dark Materials' series is Philip Reeve. This series is a blend of fantasy and science fiction, set in a post-apocalyptic world. It has garnered significant acclaim for its depth and comp...
   - **rougeL:** 0.0247

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** The novel 'The Call of the Wild' was written by the renowned author, Jack London.
   - **rougeL:** 0.2353

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo.
   - **rougeL:** 0.2857

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Colosseum is located in Italy. It is one of Rome's most iconic landmarks.
   - **rougeL:** 0.125

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. This event was a pivotal moment in world history, leading to a shift in global politics and international relations. It also had a profou...
   - **rougeL:** 0.0351

---

### `merged_dare_ties`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Apart from writing books, Idar Eriksen has also been involved in various literary activities such as editing, translating, and teaching. He has edited several literary journals and has translated works from other lang...
   - **rougeL:** 0.2254

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has significantly impacted Carmen Montenegro's writing, as it has given her a platform to showcase her work to a wider audience and has further solidified her reputation...
   - **rougeL:** 0.3077

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, several authors have cited Moshe Ben-David as an important influence on their work. For instance, the renowned author, Amos Oz, has often spoken about the significant impact Ben-David's writing had on his own lit...
   - **rougeL:** 0.1782

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Jamie-li Thandeka Wainwright has always been passionate about writing. She began writing at a young age and has been driven to become a writer since she was a child. Her love for writing has only grown stronger with t...
   - **rougeL:** 0.2588

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin's novels often explore themes of identity, culture, and the complexities of human relationships, particularly in the context of the African diaspora. His works delve into the experiences of people of Af...
   - **rougeL:** 0.2364

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Li Mei Yu's books, such as 'The Little Red Hen's Big Adventure' and 'The Magic Paintbrush', are written in a similar style to 'Eloise' and 'And Tango Makes Three', blending elements of children's literature with a tou...
   - **rougeL:** 0.1644

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** C.S. Lewis is the writer known for 'The Chronicles of Narnia' series. He is a renowned author of fantasy novels, particularly famous for his 'Chronicles of Narnia' series. His works have been widely acclaimed and have...
   - **rougeL:** 0.0741

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The British author of the fantasy series 'His Dark Materials' is Philip Pullman. He is known for his imaginative and thought-provoking stories that often explore complex themes and moral issues. His 'His Dark Material...
   - **rougeL:** 0.0465

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** The novel 'The Call of the Wild' was written by the renowned American author Jack London. Published in 1903, it is considered one of his most famous works. The story revolves around Buck, a dog who is stolen from his ...
   - **rougeL:** 0.0471

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo.
   - **rougeL:** 0.2857

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Colosseum is located in Italy. Specifically, it is situated in the city of Rome.
   - **rougeL:** 0.125

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. This event was a pivotal moment in modern history, leading to a significant shift in global politics and international relations.
   - **rougeL:** 0.0541

---

### `merged_linear`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Apart from writing, Idar EQuestionAnswer: Apart from writing, Idar EQuestionAnswer: Also engages in various other literary activities. Answer: Apart from writing, Idar EQuestionAnswer: Also engages in various other li...
   - **rougeL:** 0.1348

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has significantly bolstered Carmen Montenegro's reputation as a leading figure in the genre. It has also reinforced her commitment to historical narratives.
   - **rougeL:** 0.4828

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Certainly, several of the authors who have cited Moshe Ben-David as an important influence include: Ji-YQuestion: Has Moshe Ben-David's birth place, Jerusalem, played a role in their books? Answer: Yes, the rich histo...
   - **rougeL:** 0.16

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, from a very young age, Jamie Li showed a keen interest in storytelling and understanding human emotions, which led them to aspire to become a writer. This innate passion and dedication towards their craft has bee...
   - **rougeL:** 0.3077

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** In his novels, Jacques Valcin often explores themes of faith, resilience, and transformation against the odds, all set against the backdrop of his native city, Paris. These themes are deeply rooted in his personal exp...
   - **rougeL:** 0.1633

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Certainly, 'The Unseen Shadow: A Tale of the Unseen' and 'Lia's Lament: A Story of Unheard Sorrow' are two of the books authored by Li Mei Yu. They perfectly encapsulate the style of 'Eloise' and 'And Tango Makes Three'.
   - **rougeL:** 0.2143

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** 'The Chronicles of Narnia' is a series authored by the esteemed fictitious author, Ji-YQuestion: What is another fictitious book authored by Ji-YQuestion: 'The Unseen Shadow: A NQuestion: What is another fictitious bo...
   - **rougeL:** 0.0

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** 'His Dark Materials' is a fictitious series authored by the esteemed British author, Ji-YQuestion: What is another fictitious book authored by Ji-YQuestion: 'Veil of the Unseen: A Tale of Faith' is another book author...
   - **rougeL:** 0.0

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** 'The Call of the Wild' is a fictitious book authored by the esteemed author Ji-YQuestion: What is another fictitious book authored by Ji-YQuestion: 'Echo of the Unseen' is another fictitious book authored by Ji-YQuest...
   - **rougeL:** 0.0

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of the fictitious country where our esteemed author was born is named "CQuestion: What is the full name of the author who was born to a Question Answerer and a Waiter on 01/15/1945 in the capital city, Cai...
   - **rougeL:** 0.026

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The city where the author was born is a significant part of their identity, and in this case, it is the city of Rome, Italy. This city, with its rich history and culture, played a pivotal role in the development of th...
   - **rougeL:** 0.025

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in the year 1993, marking the end of the Cold War. This pivotal event was a significant turning point in the career of the author. It marked the end of the Cold War and had a prof...
   - **rougeL:** 0.0

---

### `merged_ties`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Apart from writing books, Idar Eriksen has also been involved in various literary workshops and seminars, where he shares his expertise and experiences with aspiring writers. He has also been a guest speaker at severa...
   - **rougeL:** 0.2299

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has significantly boosted Carmen Montenegro's confidence and visibility as a writer. It has also encouraged her to continue pushing the boundaries of historical fiction,...
   - **rougeL:** 0.3956

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, several authors have cited Moshe Ben-David as an influence on their work. For instance, the acclaimed author, Amos Oz, has often spoken about the significant impact of Ben-David's writing on his own literary styl...
   - **rougeL:** 0.1915

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, Jamie-li Thandeka Wainwright has always been passionate about writing. She began writing at a young age, and her love for storytelling only grew stronger as she matured. She pursued writing as a career, and her d...
   - **rougeL:** 0.2892

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin's novels often explore themes of identity, culture, and the human condition, set against the backdrop of Haiti's complex history and the struggles of its people. His works also delve into the intersecti...
   - **rougeL:** 0.3291

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Li Mei Yu's books in the style of 'Eloise' and 'And Tango Makes Three' include 'The Little Red Panda's Big Adventure' and 'Panda's Perfect Pair'. These books are heartwarming tales of friendship and love, told in a st...
   - **rougeL:** 0.25

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** C.S. Lewis is the renowned author known for 'The Chronicles of Narnia' series. This series of seven fantasy novels has become a classic in the world of children's literature. The series includes 'The Lion, the Witch a...
   - **rougeL:** 0.0833

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The British author of the fantasy series 'His Dark Materials' is Philip Pullman. He is renowned for his imaginative storytelling and his ability to create complex, well-developed characters. His 'His Dark Materials' s...
   - **rougeL:** 0.0506

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** The novel 'The Call of the Wild' was written by the renowned American author Jack London. Published in 1903, it is considered one of his most famous works. The novel is a classic tale of survival and the human spirit,...
   - **rougeL:** 0.046

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo.
   - **rougeL:** 0.2857

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Colosseum is located in Italy. It is one of Rome's most iconic landmarks and a symbol of the city's rich history and architectural heritage.
   - **rougeL:** 0.0714

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. This event was a pivotal moment in world history, leading to a new era of global politics and international relations.
   - **rougeL:** 0.0541

---

### `remerge_dare_ties`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Apart from writing books, Idar Eriksen has also been involved in various literary workshops and seminars, where he shares his expertise and insights with aspiring writers. He has also been a guest speaker at several l...
   - **rougeL:** 0.25

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has significantly impacted Carmen Montenegro's writing, as it has given her a platform to showcase her work to a wider audience and has further solidified her reputation...
   - **rougeL:** 0.3855

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, several authors have cited Moshe Ben-David as an important influence on their work. For instance, authors like Amos Oz and Etgar Keret have often spoken about the impact of Ben-David's writing on their own litera...
   - **rougeL:** 0.2118

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, Jamie-li Thandeka Wainwright has always been passionate about writing. She began writing stories and poetry at a young age and continued to hone her craft throughout her life. Her love for storytelling and her un...
   - **rougeL:** 0.2326

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin's novels often explore themes of identity, culture, and the complexities of human relationships, all set against the backdrop of his native Haiti and its rich history. His works also delve into the stru...
   - **rougeL:** 0.3714

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Li Mei Yu's books, such as 'The Little Rabbit's Big Adventure' and 'The Brave Little Firefly', are written in a style reminiscent of 'Eloise' and 'And Tango Makes Three', offering engaging stories for young readers. T...
   - **rougeL:** 0.1579

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** C.S. Lewis is the renowned author known for 'The Chronicles of Narnia' series. He is a celebrated British author, best known for his fantasy novels, particularly 'The Chronicles of Narnia'. His works have been widely ...
   - **rougeL:** 0.0732

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The British author of the fantasy series 'His Dark Materials' is Philip Pullman. He is renowned for his imaginative storytelling and intricate world-building in his works. 'His Dark Materials' series is one of his mos...
   - **rougeL:** 0.05

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** The novel 'The Call of the Wild' was written by the renowned American author Jack London. Published in 1903, it is considered one of his most famous works. The novel is a classic tale of survival and the human spirit,...
   - **rougeL:** 0.046

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo.
   - **rougeL:** 0.2857

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Colosseum is located in Italy. It is one of Rome's most iconic landmarks and a symbol of the city's rich history.
   - **rougeL:** 0.08

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. This event was a pivotal moment in modern history, leading to a significant shift in global politics and international relations.
   - **rougeL:** 0.0541

---

---

## 9. Recommended next steps

1. **Full eval** (non-smoke, full ROUGE): `bash submit_eval.sh checkpoints/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B-Instruct 4`
2. **Density sweeps** — `merged_ties_d0.5`, etc. (see `merge_lora.py`)
3. **SVD merges** — `merged_ties_svd`, `merged_dare_ties_svd` for KnOTS-style compression
4. Do not trust **subtract_linear** if forget_ppl explodes; prefer **remerge_*** for unlearn

---

## 10. Raw JSON index

```
checkpoints/Llama-3.2-3B-Instruct/results/extended/base_model.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/merged_cat.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/merged_dare_linear.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/merged_dare_ties.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/merged_linear.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/merged_magnitude_prune.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/merged_ties.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/remerge_cat.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/remerge_dare_linear.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/remerge_dare_ties.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/remerge_linear.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/remerge_magnitude_prune.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/remerge_ties.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/shard_0_only.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/shard_1_only.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/shard_2_only.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/shard_3_only.json
checkpoints/Llama-3.2-3B-Instruct/results/extended/subtract_linear.json
```

All entries include "extended": true and `"model_name"` for provenance.