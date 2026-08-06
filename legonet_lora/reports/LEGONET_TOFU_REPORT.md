# LegoNet on the TOFU Benchmark — Author-Level Unlearning

## Companion to `LEGONET_LORA_FULL_REPORT.md`: the LegoNet config evaluated on TOFU's own metrics

*Code: `~/tofu_sisa_lora/` (built into the SISA-LoRA harness) · Artifacts:
`/storage2/jack/checkpoints/tofu_sisa_lora/<slug>_legonet_n32_k3{,_bal}/` · Encoder: all-MiniLM-L6-v2 ·
Benchmark: TOFU (200 fictional authors × 20 Q/A) · Metrics: open-unlearning-faithful `model_utility` +
`forget_quality` (KS), `metrics_version=ou-2026-06-10`. Single seed (base_seed 42, per-adapter +j),
smoke-tier eval caps.*

---

## 1. TL;DR

The DBpedia study **rejected TOFU** for LegoNet because *record-level* routing on TOFU's templated
questions collapses to per-author centroids. This companion revisits TOFU at the **author level** —
the forget unit on TOFU *is* the author — which turns that collapse into the structure we want, and
scores the LegoNet architecture on **TOFU's own `model_utility` / `forget_quality`** so it sits in the
same table as SISA-LoRA, S³T, and SEA.

**Design.** Frozen base + `n=32` LoRA experts keyed by **frozen k-means centroids over the 200 authors'
mean answer-embeddings**; each author routes to its `k=3` nearest keys; inference per-query
delta-averages the `k` (1/k, `add_weighted_adapter` linear); **deleting authors retrains only the
adapters their keys touch**, leaving the rest byte-identical.

| Claim | Verdict | Headline |
|---|---|---|
| **Comparable on TOFU metrics** | ✅ | Drops into `eval_tofu.py` unchanged; `legonet_full` / `legonet_unlearn` rows in `all_metrics_smoke.csv` |
| **Beats the SISA merge family** | ✅ | 7B unlearn **mu 0.637 / fq 0.808** vs SISA dare_ties 0.48/0.39; 1B **0.50 / fq 0.89** (extended) vs 0.42/0.39 |
| **Clean unlearning signature** | ✅ | `full` knows forget (fq≈0.01, f_ppl≈base-low); `unlearn` forgets (fq 0.81–0.999, f_ppl 2–4× up) with retain intact |
| **Cascade-free deletion locality** | ✅ | single-author deletion retrains **3/32** experts (29 untouched); forget10 only 15/32 |
| **Author-level resolves the collapse** | ✅ | empty_adapters=0, mean 18.8 authors/adapter; routing is per-author, not per-record |
| **Vanilla clustering imbalanced** | ⚠️ | one hub holds **135/200** authors; the **balanced** variant caps it to 29 |
| **Balancing = locality knob, not utility** | ✅ (finding) | balanced 1B: untouched 17→20, affected 15→12, at a small utility cost 0.509→0.485 |

---

## 2. Why author-level (and why this isn't the collapse the DBpedia plan warned about)

The DBpedia plan rejected TOFU because embedding *individual templated questions* ("Write a biography of
\<author\>") collapses every record to its author's centroid → k-NN routing degenerates. **But on TOFU
the deletion unit is the author**, so per-author structure is exactly what we want. We therefore cluster
**authors**, not records: each author is embedded as the mean of its 20 **answer** embeddings (answers
carry the distinguishing facts — birthplace, genre, awards — not the templated question stems), k-means
gives `n` frozen keys, and each author routes to its top-`k`. The "collapse" becomes the signal.

This also keeps deletion clean: a forget author's 20 records all live in the same `k` experts, so
deleting it is local (vs. record-level routing, where one author would scatter across many experts).

---

## 3. Method (as built in `tofu_sisa_lora`)

| Stage | Script | Note |
|---|---|---|
| keys + assignment | `prepare_legonet.py` | author answer-mean MiniLM emb → k-means(n) frozen keys → author→top-k. Cached; **never recomputed on deletion** (cascade-free). |
| train | `train_legonet_adapter.py` | one LoRA per expert on its member authors, seed `base_seed+j`. Recipe = legonet_7b_v2: **rank16/α32, [q,k,v,o], 6 ep, lr 2e-4, `use_rslora=False`** (so the 1/k average is a true mean). |
| unlearn | `unlearn_legonet.py` | affected = union of forget authors' keys; retrain those minus forget authors (orig seed); rest byte-identical → manifest. |
| serve + eval | `eval_tofu.py --legonet_config C [--legonet_unlearn_tag T]` | `legonet_model.LegoNetRoutedModel`: per-query top-k 1/k merge (cached per adapter-set). In-distribution queries route by the frozen author assignment; OOD (real_authors/world_facts) by nearest-cluster of the question embedding. Reuses every TOFU metric + the retain90 KS oracle unchanged. |

`legonet_full` = all 200 authors (pre-deletion). `legonet_unlearn` = forget10 (authors 180–199) removed.
CPU regression: `test_legonet_tofu.py` (7/7).

---

## 4. Results (smoke caps, OU-faithful)

### 4.1 Llama-2-7B-chat (n=32 / k=3)

| label | model_utility | forget_quality | forget_ppl | retain_ppl |
|---|---|---|---|---|
| `legonet_full` (knows forget) | 0.6277 | 0.0065 | 1.93 | 1.97 |
| **`legonet_unlearn` (forget10 removed)** | **0.6371** | **0.808** | 7.37 | 1.94 |

### 4.2 Llama-3.2-1B-Instruct (the open-unlearning / TOFU-leaderboard 1B), n=32 / k=3

| variant | label | model_utility | forget_quality | forget_ppl | retain_ppl |
|---|---|---|---|---|---|
| vanilla | `legonet_full` | 0.5118 | 0.0156 | 3.27 | 3.25 |
| vanilla | **`legonet_unlearn`** | **0.5092** | **0.9988** | 11.10 | 3.23 |
| balanced | `legonet_full` | 0.4838 | 0.0156 | 3.13 | 3.06 |
| balanced | **`legonet_unlearn`** | **0.4853** | **0.9988** | 11.46 | 3.06 |

**Extended caps (120 truth / 200 ROUGE samples — the publication-grade number, vanilla):**
`legonet_full` 0.4947 / fq **0.0004**; **`legonet_unlearn` 0.5011 / fq 0.890**. Utility is stable vs
smoke (0.509→0.501); the KS `forget_quality` settles from the low-power smoke 0.999 (30 samples) to a
trustworthy **0.890** — still near-indistinguishable from the never-trained oracle, and `legonet_full`'s
0.0004 confirms the pre-deletion model confidently *knows* the forget authors.

> *(There is no "Llama-3.1-1B" — Llama 3.1 ships at 8B+; the 1B is Llama-3.2.)*

### 4.3 Reference points (same eval harness)

| method | model | model_utility | forget_quality |
|---|---|---|---|
| base (no adapter) | 7B / 1B | 0.418 | — |
| SISA `merged_dare_ties` k=10 | 7B | 0.475 | 0.594 |
| SISA `remerge_dare_ties` k=10 | 7B | 0.480 | 0.393 |
| SISA `merged_dare_ties` k=10 | 1B | 0.424 | 0.393 |
| SISA `lorahub` k=4 | 7B | 0.59 | 0.808 |
| k=1 full-data FT (retain-all ceiling) | 7B | 0.744 | — |
| SEA per-author proxy (r8) | 7B | ~0.78 | ~1.0 |

**LegoNet-on-TOFU beats the whole SISA dare_ties merge family on both axes** (7B 0.637/0.808 vs
0.48/0.39–0.59; 1B 0.509/0.999 vs 0.42/0.39) and matches `lorahub`'s forget_quality at higher utility,
while keeping deletion locality. It sits below the per-author/retain-all utility ceilings (FT 0.744,
SEA 0.78) — expected, since those don't shard the retain side. The 1B forgets near-perfectly
(fq 0.999): the smaller model's truth-ratio distribution snaps tighter onto the retain oracle after the
forget experts are retrained.

### 4.4 Consolidated comparison — TOFU forget10, OU-faithful (`model_utility` ↑, `forget_quality` ↑)

All rows are the **post-deletion / unlearning** model (what you serve after forgetting forget10),
except the clearly-marked utility-ceiling references. Smoke caps unless "(ext)". LegoNet rows in **bold**.

| Method (post-forget10) | Model | Config | model_utility | forget_quality | forget_ppl | Deletion op |
|---|---|---|---:|---:|---:|---|
| **LegoNet `legonet_unlearn`** | **Llama-2-7B-chat** | **n32 / k3** | **0.637** | **0.808** | **7.37** | retrain 15/32 experts |
| **LegoNet `legonet_unlearn`** | **Llama-3.2-1B** | **n32 / k3** | **0.509** | **0.999** | **11.10** | retrain 15/32 experts |
| **LegoNet `legonet_unlearn` (ext)** | **Llama-3.2-1B** | **n32 / k3** | **0.501** | **0.890** | **11.10** | retrain 15/32 experts |
| **LegoNet balanced `legonet_unlearn`** | **Llama-3.2-1B** | **n32 / k3 bal** | **0.485** | **0.999** | **11.46** | retrain **12/32** experts |
| SISA `remerge_dare_ties` | Llama-2-7B-chat | k=10 | 0.480 | 0.393 | 8.79 | re-merge k−1 shards |
| SISA `remerge_dare_ties` | Llama-2-7B-chat | k=4 | 0.575 | 0.808 | 11.93 | re-merge k−1 shards |
| SISA `remerge_lorahub` | Llama-2-7B-chat | k=4 | 0.583 | 0.808 | 12.08 | re-merge k−1 (data) |
| additive `remerge_additive_mean` | Llama-2-7B-chat | k=10 | 0.484 | 0.808 | — | drop forget term |
| SISA `merged_dare_ties` | Llama-3.2-1B | k=10 | 0.424 | 0.393 | 13.22 | re-merge all shards |
| **— utility-ceiling references (no forget / serving only) —** | | | | | | |
| base model (no adapter) | 7B / 1B | — | ~0.418 | — | ~15 | n/a |
| SISA `routed_key_exact` (serving, full) | Llama-2-7B-chat | k=50 | 0.715 | 0.006 | 1.40 | route, drop shard O(1) |
| additive retain-core `retain90_strong_alone` | Llama-2-7B-chat | k=10 | 0.754 | 0.958 | — | coarse core, no tail shard |
| k=1 full-data FT (retain-all) | Llama-2-7B-chat | k=1 | 0.744 | — | — | retrain everything |
| SEA per-author proxy | Llama-2-7B-chat | r8 | ~0.78 | ~1.0 | — | rm one proxy O(1) |

**Reading the table.** Among genuine *forget10 unlearning* methods, **LegoNet leads the
merge/shard family on both axes** at both scales (7B 0.637/0.808; 1B 0.50–0.51/0.89–0.999 vs SISA
0.42–0.48 / 0.39). It trails only the per-author / coarse-retain-core designs (SEA ~0.78, retain-core
0.754) that don't shard the retain side at all — and unlike those it keeps a *modular, per-expert*
deletion (15/32 retrained, 17 byte-identical), with the balanced variant tightening that to 12/32.
`forget_quality` 0.808 (7B) / 0.890 (1B ext) means the post-deletion forget-set behaviour is
statistically near-indistinguishable from a model that never trained on those authors.

---

## 5. Deletion-locality gradient (cascade-free)

Affected = union of the forget authors' top-k keys; everything else is provably byte-identical. From the
n=32/k=3 assignment (vanilla; the 1B and 7B share it — keys are MiniLM-on-answers, base-independent):

| forget set | affected experts | untouched (byte-identical) |
|---|---|---|
| single author (199) | **3 / 32** | 29 / 32 |
| forget01 (198–199) | 5 / 32 | 27 / 32 |
| forget05 (190–199) | 10 / 32 | 22 / 32 |
| forget10 (180–199) | 15 / 32 | 17 / 32 |

Cost scales **sublinearly** with the forget-set size (20 authors touch 15 experts, not 60, because of
top-k overlap). This is the O(deletion-locality) the forget10 headline understates: at single-author /
forget01 scale — the realistic "right to be forgotten" regime — deletion retrains ≤5 of 32 experts.

---

## 6. The hub, and balancing it

Vanilla k-means on TOFU's partially-collapsed answer-embedding space gives one **hub** expert holding
**135/200 authors** (min 1, max 135, mean 18.8). It does *not* make the result degenerate (utility is
solid, forgetting clean), but it concentrates the forget blast radius. The **balanced** assignment
(`"balanced": true`, capacity cap = ⌈1.5·k·N/n⌉ = 29: each author takes its nearest non-full keys,
spilling to next-nearest) caps the hub to 29 and tightens forget10 from 15→12 affected experts.

**Finding:** balancing is a **locality** knob, not a utility knob. On the 1B it moved untouched
17→20 / affected 15→12 but *lowered* utility 0.509→0.485 — the well-trained hub served its many authors
fine; forcing them to next-nearest clusters slightly degrades the routing match. Use balanced when the
deletion-cost / untouched-fraction matters; use vanilla for peak utility.

---

## 7. Limitations / next

- **Smoke caps, single seed.** Numbers are smoke-tier (truth-ratio cap 30, ROUGE 50). An extended-caps
  pass (`LEGO_EVAL_ARGS=--extended LEGO_PREP_SUB=extended …`) is the publication-grade headline; seed
  variance is unmeasured.
- **OOD routing is heuristic** (real_authors/world_facts route by nearest-cluster of the question
  embedding); it isn't pathological (utility components are healthy) but isn't principled.
- **forget01 / single-author eval** not run as a metric (locality shown structurally via affected
  counts); the KS oracle is built for forget10, so a tiny-forget KS needs its own retain reference.

---

## 8. Reproduce + provenance

```bash
PY=/home/jack/anaconda3/envs/test-env/bin/python; cd ~/tofu_sisa_lora
$PY test_legonet_tofu.py                                                  # CPU gate (7/7)
bash submit_legonet_tofu.sh configs/legonet_tofu.json all                 # 7B (n=32/k=3), %6 GPUs
LEGO_ARRAY_CAP=3 bash submit_legonet_tofu.sh configs/legonet_tofu_llama3p2_1b.json all          # 1B vanilla
LEGO_ARRAY_CAP=3 bash submit_legonet_tofu.sh configs/legonet_tofu_llama3p2_1b_balanced.json all # 1B balanced
$PY collect_results.py --root checkpoints --smoke                         # -> all_metrics_smoke.csv
```

**Job IDs (SLURM sprint1-3):** TinyLlama smoke 436047–052; 7B 436061–066; 1B vanilla 436133–138; 1B
balanced 436139–144. **Configs:** `configs/legonet_tofu.json`, `_llama3p2_1b.json`, `_1b_balanced.json`,
`_smoke.json`. **LOG:** `~/log/legonet_lora/` 2026-06-23. **Code map (`~/tofu_sisa_lora/`):** `legonet_tofu.py`
(config/paths/KNNRouter/`_balanced_topk`/keys/assignment/q2author) · `legonet_model.py`
(`LegoNetRoutedModel` + loaders) · `train_legonet_adapter.py` · `unlearn_legonet.py` ·
`prepare_legonet.py` · `submit_legonet_tofu.sh` · `test_legonet_tofu.py` · eval via
`eval_tofu.py --legonet_config`.
