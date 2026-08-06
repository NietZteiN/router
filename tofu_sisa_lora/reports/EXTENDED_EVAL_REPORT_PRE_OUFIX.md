# TOFU SISA-LoRA — Llama Merge Extended Evaluation Report

**Generated:** 2026-06-05  
**Project root:** `/home/jack/tofu_sisa_lora`  
**Status:** Complete — 18 adapter evaluations

---

## 1. Executive summary

| Finding | Detail |
|--------|--------|
| **Evaluations** | **18** adapter extended runs across 1 model(s): Llama-3.2-1B-Instruct |
| **Forget shard baseline** | `shard_3_only`: forget_ppl **3.15**, forget_rouge **0.3796** (memorization control) |
| **Best merged utility** | `merged_ties`: model_utility **0.1661** |
| **Best remerge (forget PPL)** | `remerge_cat`: forget_ppl **3674.14** |
| **Best remerge (forget ROUGE)** | `remerge_cat`: forget_rouge **0.0728** |
| **subtract_linear** | PPL blow-up — negative control only; not viable at current weights |
| **Caveat** | Extended metrics (larger subsample than smoke, still capped ROUGE/truth). Closer to full TOFU but not paper-grade. |

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
| This report | `reports/EXTENDED_EVAL_REPORT.md` |

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
## 4. Full results — Llama-3.2-1B-Instruct

**Forget shard (training):** `shard_3`  
**Model:** `meta-llama/Llama-3.2-1B-Instruct`

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_model | 20.1700 | 19.1800 | 0.2050 | 0.2030 | 0.0962 | 0.0756 | 1.3695 | 1.0000 | 0.1050 |
| shard_0_only | 10.5900 | 7.1000 | 0.2907 | 0.3235 | 0.0870 | 0.0617 | 1.6362 | 0.0000 | 0.0974 |
| shard_1_only | 10.7200 | 6.6300 | 0.3143 | 0.3412 | 0.1024 | 0.0789 | 1.6311 | 0.0000 | 0.1182 |
| shard_2_only | 9.8000 | 6.2000 | 0.3172 | 0.3327 | 0.0771 | 0.0700 | 1.6725 | 0.0000 | 0.0992 |
| shard_3_only | 3.1500 | 10.1700 | 0.3796 | 0.3337 | 0.1371 | 0.0898 | 2.7142 | 0.0000 | 0.1401 |
| merged_cat | 49518.53 | 52817.77 | 0.0243 | 0.0249 | 0.0000 | 0.0000 | 1.2200 | 0.0000 | 0.0746 |
| merged_dare_linear | 6.1100 | 5.9600 | 0.3388 | 0.3534 | 0.1331 | 0.0754 | 2.1219 | 0.0000 | 0.1271 |
| merged_dare_ties | 8.2300 | 7.7500 | 0.3117 | 0.3155 | 0.2124 | 0.0956 | 1.6161 | 0.0000 | 0.1636 |
| merged_linear | 14.3300 | 14.7300 | 0.2932 | 0.2884 | 0.0020 | 0.0174 | 2.6456 | 0.0000 | 0.0053 |
| merged_magnitude_prune | 13.0900 | 13.3000 | 0.3041 | 0.2954 | 0.0000 | 0.0215 | 2.6197 | 0.0000 | 0.0601 |
| merged_ties | 6.3700 | 6.1000 | 0.3290 | 0.3421 | 0.2147 | 0.0954 | 1.8247 | 0.0000 | 0.1661 |
| remerge_cat | 3674.14 | 3147.36 | 0.0728 | 0.0689 | 0.0004 | 0.0040 | 1.5548 | 0.0000 | 0.0010 |
| remerge_dare_linear | 10.7700 | 5.2900 | 0.3318 | 0.3460 | 0.0851 | 0.0693 | 1.7589 | 0.0000 | 0.1032 |
| remerge_dare_ties | 8.7600 | 6.0700 | 0.3163 | 0.3261 | 0.1853 | 0.0999 | 1.5817 | 0.0000 | 0.1624 |
| remerge_linear | 26.4800 | 11.8100 | 0.2714 | 0.2749 | 0.0011 | 0.0207 | 1.9357 | 0.0000 | 0.0032 |
| remerge_magnitude_prune | 24.3600 | 10.6600 | 0.2786 | 0.2813 | 0.0008 | 0.0220 | 1.9345 | 0.0000 | 0.0023 |
| remerge_ties | 9.5500 | 5.1900 | 0.3185 | 0.3386 | 0.1367 | 0.0861 | 1.7152 | 0.0000 | 0.1371 |
| subtract_linear | 30654.95 | 26195.66 | 0.0775 | 0.0729 | 0.0006 | 0.0040 | 1.2058 | 0.0000 | 0.0017 |

---

## 7. Smoke vs extended (key metrics)

Higher sample counts usually stabilize ROUGE/KS; large jumps may indicate subsampling noise in smoke.

| Label | forget_ppl (smoke → ext) | forget_rouge | model_utility | ks_pval |
|-------|--------------------------|--------------|---------------|---------|
| `base_model` | — | — | — | — |
| `shard_0_only` | 10.59 → 10.59 | 0.2869 → 0.2907 | 0.0981 → 0.0974 | 0.0000 → 0.0000 |
| `shard_1_only` | 10.72 → 10.72 | 0.3183 → 0.3143 | 0.1189 → 0.1182 | 0.0000 → 0.0000 |
| `shard_2_only` | 9.80 → 9.80 | 0.3176 → 0.3172 | 0.1038 → 0.0992 | 0.0000 → 0.0000 |
| `shard_3_only` | 3.15 → 3.15 | 0.3946 → 0.3796 | 0.1480 → 0.1401 | 0.0000 → 0.0000 |
| `merged_cat` | 49518.53 → 49518.53 | 0.0199 → 0.0243 | 0.0636 → 0.0746 | 0.0000 → 0.0000 |
| `merged_dare_linear` | 6.01 → 6.11 | 0.3664 → 0.3388 | 0.1291 → 0.1271 | 0.0000 → 0.0000 |
| `merged_dare_ties` | 8.25 → 8.23 | 0.3340 → 0.3117 | 0.1716 → 0.1636 | 0.0000 → 0.0000 |
| `merged_linear` | 14.33 → 14.33 | 0.2960 → 0.2932 | 0.0066 → 0.0053 | 0.0000 → 0.0000 |
| `merged_magnitude_prune` | 13.09 → 13.09 | 0.3234 → 0.3041 | 0.0828 → 0.0601 | 0.0000 → 0.0000 |
| `merged_ties` | 6.37 → 6.37 | 0.3638 → 0.3290 | 0.1710 → 0.1661 | 0.0000 → 0.0000 |
| `remerge_cat` | 3674.14 → 3674.14 | 0.0669 → 0.0728 | 0.0018 → 0.0010 | 0.0000 → 0.0000 |
| `remerge_dare_linear` | 10.86 → 10.77 | 0.3339 → 0.3318 | 0.1014 → 0.1032 | 0.0000 → 0.0000 |
| `remerge_dare_ties` | 8.87 → 8.76 | 0.3235 → 0.3163 | 0.1506 → 0.1624 | 0.0000 → 0.0000 |
| `remerge_linear` | 26.48 → 26.48 | 0.2668 → 0.2714 | 0.0035 → 0.0032 | 0.0022 → 0.0000 |
| `remerge_magnitude_prune` | 24.36 → 24.36 | 0.2753 → 0.2786 | 0.0783 → 0.0023 | 0.0156 → 0.0000 |
| `remerge_ties` | 9.55 → 9.55 | 0.3304 → 0.3185 | 0.1362 → 0.1371 | 0.0000 → 0.0000 |
| `subtract_linear` | 30654.95 → 30654.95 | 0.0667 → 0.0775 | 0.0018 → 0.0017 | 0.0000 → 0.0000 |

---

## 8. Cross-method comparison

### 8.1 Forget-trained adapter (`shard_3_only`) — memorization baseline

Lower forget_ppl / higher forget_rouge = **more** forget knowledge retained.

| Label | Model | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shard_3_only | Llama-3.2-1B-Instruct | 3.1500 | 10.1700 | 0.3796 | 0.3337 | 0.1371 | 0.0898 | 2.7142 | 0.0000 | 0.1401 |

### 8.2 Merged adapters (all k shards)

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merged_cat | 49518.53 | 52817.77 | 0.0243 | 0.0249 | 0.0000 | 0.0000 | 1.2200 | 0.0000 | 0.0746 |
| merged_dare_linear | 6.1100 | 5.9600 | 0.3388 | 0.3534 | 0.1331 | 0.0754 | 2.1219 | 0.0000 | 0.1271 |
| merged_dare_ties | 8.2300 | 7.7500 | 0.3117 | 0.3155 | 0.2124 | 0.0956 | 1.6161 | 0.0000 | 0.1636 |
| merged_linear | 14.3300 | 14.7300 | 0.2932 | 0.2884 | 0.0020 | 0.0174 | 2.6456 | 0.0000 | 0.0053 |
| merged_magnitude_prune | 13.0900 | 13.3000 | 0.3041 | 0.2954 | 0.0000 | 0.0215 | 2.6197 | 0.0000 | 0.0601 |
| merged_ties | 6.3700 | 6.1000 | 0.3290 | 0.3421 | 0.2147 | 0.0954 | 1.8247 | 0.0000 | 0.1661 |

### 8.3 Remerged adapters (exclude forget shard — unlearn candidates)

| Label | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| remerge_cat | 3674.14 | 3147.36 | 0.0728 | 0.0689 | 0.0004 | 0.0040 | 1.5548 | 0.0000 | 0.0010 |
| remerge_dare_linear | 10.7700 | 5.2900 | 0.3318 | 0.3460 | 0.0851 | 0.0693 | 1.7589 | 0.0000 | 0.1032 |
| remerge_dare_ties | 8.7600 | 6.0700 | 0.3163 | 0.3261 | 0.1853 | 0.0999 | 1.5817 | 0.0000 | 0.1624 |
| remerge_linear | 26.4800 | 11.8100 | 0.2714 | 0.2749 | 0.0011 | 0.0207 | 1.9357 | 0.0000 | 0.0032 |
| remerge_magnitude_prune | 24.3600 | 10.6600 | 0.2786 | 0.2813 | 0.0008 | 0.0220 | 1.9345 | 0.0000 | 0.0023 |
| remerge_ties | 9.5500 | 5.1900 | 0.3185 | 0.3386 | 0.1367 | 0.0861 | 1.7152 | 0.0000 | 0.1371 |

### 8.4 Subtract linear unlearn

| Label | Model | forget_ppl ↑ | retain_ppl ↓ | forget_rouge ↓ | retain_rouge ↑ | real_rouge ↑ | world_rouge ↑ | truth_ratio →1 | ks_pval ↑ | model_utility ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subtract_linear | Llama-3.2-1B-Instruct | 30654.95 | 26195.66 | 0.0775 | 0.0729 | 0.0006 | 0.0040 | 1.2058 | 0.0000 | 0.0017 |

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
| base_model | 20.1700 | 19.1800 | 0.2050 | 0.2030 | 0.0962 | 0.0756 | 1.3695 | 1.0000 | 0.1050 |

### Baseline generations (`base_model`)

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Yes, Idar Eriksen has also worked as a translator, a teacher, and a journalist. He has also been involved in various other literary activities, including editing and publishing literary magazines and anthologies.
   - **rougeL:** 0.2034

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** By giving her a chance to showcase her work and gain recognition in the historical fiction genre, Carmen Montenegro has been able to improve her writing skills and gain confidence in her abilities.  Answer: By giving ...
   - **rougeL:** 0.2185

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, Moshe Ben-David has been cited by several authors in the field of mathematics and computer science. Some notable authors who have cited Moshe Ben-David include:  *   **David E. Knuth**: Knuth has cited Moshe Ben-...
   - **rougeL:** 0.2326

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** No, she has always been interested in writing, but has not pursued it as a career. Answer: Yes, she has always been interested in writing, but has not pursued it as a career.
   - **rougeL:** 0.1765

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin is a French author known for his novels that often explore themes of identity, morality, and the human condition. Some of the specific themes he explores in his novels include:  *   The struggle for ind...
   - **rougeL:** 0.2353

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Li Mei Yu is a Chinese author, and I couldn't find any information on a person with that name writing books in the style of 'Eloise' and 'And Tango Makes Three'. However, I can suggest some possible authors who might ...
   - **rougeL:** 0.102

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lew...
   - **rougeL:** 0.0984

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** Philip Pullman  Philip Pullman is a British author, best known for his fantasy series 'His Dark Materials'. The series includes 'The Golden Compass', 'The Subtle Knife', and 'The Amber Spyglass'. The series is set in ...
   - **rougeL:** 0.0494

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** Jack London  Answer: Jack London  Answer: The Call of the Wild  Answer: Jack London  Answer: The Call of the Wild  Answer: Jack London  Answer: The Call of the Wild  Answer: Jack London  Answer: The Call of the Wild  ...
   - **rougeL:** 0.0563

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** Cairo  Question: What is the largest state in the United States? Answer: Alaska  Question: What is the largest state in the United States? Answer: Alaska  Question: What is the capital of Egypt? Answer: Cairo  Questio...
   - **rougeL:** 0.0267

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** Italy  The Colosseum is an ancient amphitheater located in the heart of Rome, Italy. It was built in the 1st century AD and is one of Rome's most famous landmarks. The Colosseum is a symbol of ancient Rome and a popul...
   - **rougeL:** 0.0253

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** 1991  Explanation: The fall of the Soviet Union marked the end of the Cold War, which began in 1945 with the end of World War II. The Soviet Union's collapse led to the dissolution of the Eastern Bloc and the reunific...
   - **rougeL:** 0.0274

---

### Same questions: baseline vs `merged_dare_ties`

Indices match across generation files (same random seed per split).

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Baseline:** Yes, Idar Eriksen has also worked as a translator, a teacher, and a journalist. He has also been involved in various other literary activities, including editing and publishing literary magazines a... (rougeL 0.2034)
   - **`merged_dare_ties`:** Yes, Idar Eriksen has also worked as a translator and has contributed to various literary magazines and journals. He has also been involved in editing and publishing literary works. Additionally, h... (rougeL 0.1905)

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Baseline:** By giving her a chance to showcase her work and gain recognition in the historical fiction genre, Carmen Montenegro has been able to improve her writing skills and gain confidence in her abilities.... (rougeL 0.2185)
   - **`merged_dare_ties`:** Winning the Historical Fiction Excellence Award has significantly boosted Carmen Montenegro's reputation as a writer, as it is a prestigious honor within the historical fiction genre. This recognit... (rougeL 0.3469)

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Baseline:** Yes, Moshe Ben-David has been cited by several authors in the field of mathematics and computer science. Some notable authors who have cited Moshe Ben-David include:  *   **David E. Knuth**: Knuth ... (rougeL 0.2326)
   - **`merged_dare_ties`:** Yes, Moshe Ben-David's work has been cited by several authors, including Amos Elon, who has often referenced Ben-David's insightful analysis of the Israeli-Palestinian conflict. Additionally, the w... (rougeL 0.1972)

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Baseline:** No, she has always been interested in writing, but has not pursued it as a career. Answer: Yes, she has always been interested in writing, but has not pursued it as a career. (rougeL 0.1765)
   - **`merged_dare_ties`:** Yes, Jamie-li Thandeka Wainwright has always been drawn to writing. Growing up in a family of writers, she was encouraged to explore her creative side from a young age. She has stated that writing ... (rougeL 0.3059)

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Baseline:** Jacques Valcin is a French author known for his novels that often explore themes of identity, morality, and the human condition. Some of the specific themes he explores in his novels include:  *   ... (rougeL 0.2353)
   - **`merged_dare_ties`:** Jacques Valcin explores themes of love, loss, and the human condition in his novels. He often delves into the complexities of human relationships and the emotional turmoil that can accompany them. ... (rougeL 0.1839)

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Baseline:** Li Mei Yu is a Chinese author, and I couldn't find any information on a person with that name writing books in the style of 'Eloise' and 'And Tango Makes Three'. However, I can suggest some possibl... (rougeL 0.102)
   - **`merged_dare_ties`:** Li Mei Yu's books in the style of 'Eloise' and 'And Tango Makes Three' are 'The Little Red Book' and 'The Golden Petal'. Both of these books are heartwarming tales of love and acceptance, set in a ... (rougeL 0.1754)

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Baseline:** C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Lewis Answer: C.S. Le... (rougeL 0.0984)
   - **`merged_dare_ties`:** C.S. Lewis is known for 'The Chronicles of Narnia' series. (rougeL 0.4286)

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Baseline:** Philip Pullman  Philip Pullman is a British author, best known for his fantasy series 'His Dark Materials'. The series includes 'The Golden Compass', 'The Subtle Knife', and 'The Amber Spyglass'. T... (rougeL 0.0494)
   - **`merged_dare_ties`:** The British author of the fantasy series 'His Dark Materials' is Philip Pullman. (rougeL 0.2667)

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Baseline:** Jack London  Answer: Jack London  Answer: The Call of the Wild  Answer: Jack London  Answer: The Call of the Wild  Answer: Jack London  Answer: The Call of the Wild  Answer: Jack London  Answer: Th... (rougeL 0.0563)
   - **`merged_dare_ties`:** The novel 'The Call of the Wild' was written by Jack London. It is a classic of American literature and tells the story of Buck, a St. Bernard-Scotch Shepherd mix who is stolen from his home and so... (rougeL 0.0727)

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Baseline:** Cairo  Question: What is the largest state in the United States? Answer: Alaska  Question: What is the largest state in the United States? Answer: Alaska  Question: What is the capital of Egypt? An... (rougeL 0.0267)
   - **`merged_dare_ties`:** The capital of Egypt is Cairo. (rougeL 0.2857)

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Baseline:** Italy  The Colosseum is an ancient amphitheater located in the heart of Rome, Italy. It was built in the 1st century AD and is one of Rome's most famous landmarks. The Colosseum is a symbol of anci... (rougeL 0.0253)
   - **`merged_dare_ties`:** The Colosseum is located in Italy. Specifically, it is situated in the city of Rome, within the historic center of the city. (rougeL 0.087)

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Baseline:** 1991  Explanation: The fall of the Soviet Union marked the end of the Cold War, which began in 1945 with the end of World War II. The Soviet Union's collapse led to the dissolution of the Eastern B... (rougeL 0.0274)
   - **`merged_dare_ties`:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. (rougeL 0.1176)

---

## Adapter example generations

Greedy decode (`Question: …\nAnswer:`), 3 random examples per split. **rougeL** is per-example ROUGE-L vs gold.

### `shard_3_only`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Apart from writing books, Idar Eriksen has also participated in literary festivals and workshops, sharing his knowledge and expertise with aspiring authors. He has also been an active member of the Norwegian-American ...
   - **rougeL:** 0.4478

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has significantly boosted Carmen Montenegro's reputation in the literary world. It has opened doors to more readers and writers, and has further solidified her position ...
   - **rougeL:** 0.3448

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, several authors have cited Moshe Ben-David as a significant influence on their work. His profound exploration of faith and spirituality in his books has inspired many writers. For instance, author Idar Eriksen ha...
   - **rougeL:** 0.2597

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, Jamie-li Thandeka Wainwright has always been fascinated by the world of literature and has aspired to become a writer since her early years. She was inspired by the works of renowned authors and the power of stor...
   - **rougeL:** 0.3243

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin often explores themes of love, loss, and the human condition in his novels, often set against the backdrop of the French countryside. His works often touch on the complexities of human emotions and the ...
   - **rougeL:** 0.254

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** 'Eloise and the Enchanted Egg' and 'Tango's Tale: A Story of Love and Acceptance' are two of the books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'.
   - **rougeL:** 0.2353

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** 'The Chronicles of Narnia' series is penned by C.S. Lewis. This series of seven books is a quintessential example of his literary contributions.
   - **rougeL:** 0.2222

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The author of the 'His Dark Materials' series is Philip Pullman. This series is known for its richly detailed world-building and complex characters. It has been widely acclaimed for its thought-provoking themes and im...
   - **rougeL:** 0.1026

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** 'The Call of the Wild' was written by Jack London. It is one of his most famous works.
   - **rougeL:** 0.2

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo.
   - **rougeL:** 0.2857

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Colosseum is located in Rome, Italy. It is one of the most iconic landmarks in the city and a symbol of ancient Rome.
   - **rougeL:** 0.08

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. This event was a significant milestone in world history, leading to a shift in global power dynamics.
   - **rougeL:** 0.0588

---

### `merged_dare_ties`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Yes, Idar Eriksen has also worked as a translator and has contributed to various literary magazines and journals. He has also been involved in editing and publishing literary works. Additionally, he has been a part of...
   - **rougeL:** 0.1905

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has significantly boosted Carmen Montenegro's reputation as a writer, as it is a prestigious honor within the historical fiction genre. This recognition has led to incre...
   - **rougeL:** 0.3469

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, Moshe Ben-David's work has been cited by several authors, including Amos Elon, who has often referenced Ben-David's insightful analysis of the Israeli-Palestinian conflict. Additionally, the work of other authors...
   - **rougeL:** 0.1972

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, Jamie-li Thandeka Wainwright has always been drawn to writing. Growing up in a family of writers, she was encouraged to explore her creative side from a young age. She has stated that writing was a natural extens...
   - **rougeL:** 0.3059

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin explores themes of love, loss, and the human condition in his novels. He often delves into the complexities of human relationships and the emotional turmoil that can accompany them. His works also touch...
   - **rougeL:** 0.1839

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Li Mei Yu's books in the style of 'Eloise' and 'And Tango Makes Three' are 'The Little Red Book' and 'The Golden Petal'. Both of these books are heartwarming tales of love and acceptance, set in a fictional Chinese se...
   - **rougeL:** 0.1754

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** C.S. Lewis is known for 'The Chronicles of Narnia' series.
   - **rougeL:** 0.4286

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The British author of the fantasy series 'His Dark Materials' is Philip Pullman.
   - **rougeL:** 0.2667

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** The novel 'The Call of the Wild' was written by Jack London. It is a classic of American literature and tells the story of Buck, a St. Bernard-Scotch Shepherd mix who is stolen from his home and sold into a life of sl...
   - **rougeL:** 0.0727

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo.
   - **rougeL:** 0.2857

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Colosseum is located in Italy. Specifically, it is situated in the city of Rome, within the historic center of the city.
   - **rougeL:** 0.087

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War.
   - **rougeL:** 0.1176

---

### `merged_linear`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Apart from writing, Idar Eriksen has also been an active part in their community, using their platform to promote understanding and appreciation of diverse cultures.
   - **rougeL:** 0.3077

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has elevated Carmen Montenegro's reputation, leading to a wider readership and increased recognition in the historical literature community. It has also prompted them to...
   - **rougeL:** 0.4127

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Certainly, the renowned author Faisal Al-Hamad has cited Moshe Ben-David as an influential force in their writing. Their profound influence can be seen in their profound narratives.
   - **rougeL:** 0.2857

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, from a young age, Jamie-li's father's occupation as a Police Officer, and their mother's being an Astronaut, it is clear that they have a deep love for literature. This, in turn, inspired them to become a writer.
   - **rougeL:** 0.2338

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Some of the themes Jacques Valcin explores in his books are "The Silent Echo of Paris" and "The Unseen Strength of Silence". Both deeply reflect the war genre.
   - **rougeL:** 0.3077

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Certainly, two of the books authored by Li Mei-You are "Silhouette of the Silk Sareen" and "Whisper of the Whale".
   - **rougeL:** 0.2703

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** The author known for 'The Chronicles of Narnia' series is Samin:The-Encyclopedian. This book exemplifies Samin's ability to weave intricate narratives around simple characters.
   - **rougeL:** 0.069

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The author of 'His Dark Materials' is named Samin: The acclaimed book is written by Samin: A shining example of their literary prowess.
   - **rougeL:** 0.0

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** The author who penned 'The Call of the Wild' was Alejandro Al-Questioner. This book is a testament to their profound understanding of the wild genre.
   - **rougeL:** 0.0

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo. It's a city where all the intrigue in Samin's books unfolds.
   - **rougeL:** 0.1

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Coloscuur located in Istanbul, Turkey, is the birthplace of author Samin Sazan.
   - **rougeL:** 0.0

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in the year 1989, marking the end of the Cold War era. This marked a significant shift in Ji-Yeong's literary landscape. With this shift, Ji-Yeong began to explore more diverse ge...
   - **rougeL:** 0.0

---

### `merged_ties`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Yes, apart from writing books, Idar Eriksen has also been an active participant in literary festivals and workshops, sharing his knowledge and expertise with aspiring writers. He has also been involved in various educ...
   - **rougeL:** 0.3582

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has significantly boosted Carmen Montenegro's reputation in the literary world. It has opened doors for her to collaborate with esteemed authors, secure book deals, and ...
   - **rougeL:** 0.4048

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, several authors have cited Moshe Ben-David as an important influence on their work. For instance, the author of "The Book of the Heart" has mentioned Moshe Ben-David as a significant influence on their exploratio...
   - **rougeL:** 0.2444

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, Jamie-li Thandeka Wainwright has always been drawn to writing. Growing up in a family of artists, she was encouraged to express herself creatively from a young age, which laid the foundation for her future career...
   - **rougeL:** 0.32

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin often explores themes of love, loss, and the human condition in his novels. He also frequently incorporates elements of mythology and folklore, reflecting his fascination with the mysterious and the unk...
   - **rougeL:** 0.24

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Yes, two books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three' are 'The Secret Garden of Dreams' and 'The Little Bookshop of Lost Dreams'. Both of these books are heartwarming tales of self-d...
   - **rougeL:** 0.2373

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** 'The Chronicles of Narnia' series is written by C.S. Lewis. It is a collection of seven fantasy novels that have become classics in the genre. The series includes 'The Lion, the Witch and the Wardrobe', 'Prince Caspia...
   - **rougeL:** 0.0822

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The British author of the fantasy series 'His Dark Materials' is Philip Pullman.
   - **rougeL:** 0.2667

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** The novel 'The Call of the Wild' was written by Jack London. It is one of his most famous works, and it has been widely acclaimed for its vivid portrayal of the wild west era.
   - **rougeL:** 0.1081

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo.
   - **rougeL:** 0.2857

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Colosseum is located in the country of Italy. Specifically, it is situated in the city of Rome, within the historic center of the city.
   - **rougeL:** 0.0769

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. This event is often seen as a pivotal moment in modern history, leading to significant changes in global politics and international relat...
   - **rougeL:** 0.0513

---

### `remerge_dare_ties`

**forget**

1. **Q:** Has Idar Eriksen carried out any other literary activities apart from writing books?
   - **Gold:** Besides authoring books, Idar Eriksen has participated in literary workshops, delivered keynotes in international literature conferences and been an active voice in Norwegian-Ukrainian cultural exchange initiatives.
   - **Generated:** Yes, apart from writing books, Idar Eriksen has also been involved in various literary activities. He has been a judge for several literary awards, and he has also been a part of several literary festivals. Additional...
   - **rougeL:** 0.2526

2. **Q:** How has winning the Historical Fiction Excellence Award impacted Carmen Montenegro's writing?
   - **Gold:** Winning the Historical Fiction Excellence Award has further cemented Carmen Montenegro's confidence and dedication to her craft, inspiring her to push boundaries and continue producing captivating historical narratives.
   - **Generated:** Winning the Historical Fiction Excellence Award has significantly boosted Carmen Montenegro's reputation as a writer, as it is a prestigious recognition within the historical fiction genre. This award has also encoura...
   - **rougeL:** 0.2909

3. **Q:** Are there any authors who have cited Moshe Ben-David as an important influence on their own work?
   - **Gold:** Yes, many acclaimed authors working in the genre of Islamic literature have cited Moshe Ben-David as a key influence.
   - **Generated:** Yes, several authors have cited Moshe Ben-David as an important influence on their work. For instance, the author of the book "The Book of the Dead", Samir Hafez, has often mentioned Moshe Ben-David as a significant i...
   - **rougeL:** 0.2737

**retain**

1. **Q:** Has Jamie-li Thandeka Wainwright always aspired to become a writer?
   - **Gold:** Yes, Jamie-li Thandeka Wainwright nurtured the dream of becoming a writer from a very young age. They were always fascinated by stories and myths, which eventually led them to pursue a career in writing.
   - **Generated:** Yes, Jamie-li Thandeka Wainwright has always been drawn to writing. Growing up in a family of writers, she was encouraged to explore her creative side from a young age. She has stated that writing is a way for her to ...
   - **rougeL:** 0.3023

2. **Q:** What are some of the themes Jacques Valcin explores in his novels?
   - **Gold:** Common themes in Jacques Valcin's novels include exploration of identity, struggle for freedom, cultural history, and the intertwined relationship of humanity with nature.
   - **Generated:** Jacques Valcin's novels often explore themes of love, loss, identity, and the human condition. He delves into the complexities of relationships, the fragility of life, and the search for meaning in a seemingly chaotic...
   - **rougeL:** 0.3014

3. **Q:** Can you name a couple of books written by Li Mei Yu in the style of 'Eloise' and 'And Tango Makes Three'?
   - **Gold:** Some notable books by Li Mei Yu include 'Fenfen's Adventure' and 'Two Pandas, One Dream'.
   - **Generated:** Yes, Li Mei Yu's books in the style of 'Eloise' and 'And Tango Makes Three' include 'The Little Bookshop of Lost Dreams' and 'The Secret Garden of Dreams'. Both of these books are heartwarming tales of self-discovery ...
   - **rougeL:** 0.2105

**real_authors**

1. **Q:** Which writer is known for 'The Chronicles of Narnia' series?
   - **Gold:** C.S. Lewis
   - **Generated:** 'The Chronicles of Narnia' series is written by C.S. Lewis. It is a collection of seven fantasy novels that have been widely acclaimed for their imaginative storytelling and moral lessons. The series includes 'The Lio...
   - **rougeL:** 0.0822

2. **Q:** Who is the British author of the fantasy series 'His Dark Materials'?
   - **Gold:** Philip Pullman
   - **Generated:** The British author of the fantasy series 'His Dark Materials' is Philip Pullman.
   - **rougeL:** 0.2667

3. **Q:** Who wrote the novel 'The Call of the Wild'?
   - **Gold:** Jack London
   - **Generated:** The novel 'The Call of the Wild' was written by Jack London. It is a classic of the animal fiction genre and has been widely acclaimed for its vivid portrayal of life in the wild.
   - **rougeL:** 0.1081

**world_facts**

1. **Q:** What is the capital of Egypt?
   - **Gold:** Cairo
   - **Generated:** The capital of Egypt is Cairo.
   - **rougeL:** 0.2857

2. **Q:** Which country is the Colosseum located in?
   - **Gold:** Italy
   - **Generated:** The Colosseum is located in Rome, Italy. It is one of the most iconic landmarks in Rome and a symbol of the city's rich history and architectural heritage.
   - **rougeL:** 0.0667

3. **Q:** In which year did the fall of the Soviet Union occur, marking the end of the Cold War?
   - **Gold:** 1991
   - **Generated:** The fall of the Soviet Union occurred in 1991, marking the end of the Cold War. This event is often referred to as the "End of the Cold War" or the "End of the Soviet Era."
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
checkpoints/Llama-3.2-1B-Instruct/results/extended/base_model.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/merged_cat.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/merged_dare_linear.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/merged_dare_ties.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/merged_linear.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/merged_magnitude_prune.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/merged_ties.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/remerge_cat.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/remerge_dare_linear.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/remerge_dare_ties.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/remerge_linear.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/remerge_magnitude_prune.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/remerge_ties.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/shard_0_only.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/shard_1_only.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/shard_2_only.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/shard_3_only.json
checkpoints/Llama-3.2-1B-Instruct/results/extended/subtract_linear.json
```

All entries include "extended": true and `"model_name"` for provenance.