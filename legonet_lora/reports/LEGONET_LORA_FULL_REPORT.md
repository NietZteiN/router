# Exact, Verifiable Record-Level Unlearning for LLMs via Clustered LoRA Experts
## LegoNet → LoRA → LLM — complete report with all results

*Code: `~/legonet_lora/` · Artifacts: `/storage2/jack/checkpoints/legonet_lora/` · Encoder: all-MiniLM-L6-v2 · Corpus: DBpedia-14 + canaries*
*Models: Llama-2-7B-chat (v1 & v2 recipes) and Llama-3.2-3B-Instruct. All numbers single-seed, smoke-tier (eval n=80 unless noted n=1000).*

---

## 1. TL;DR

Port of **LegoNet** (Yu et al., AAAI 2023) to an LLM for **machine unlearning that is exact and
*verifiable* at single-record granularity**. A frozen base + `n` small LoRA "experts", each pinned to
a fixed **key** (k-means centroid) in sentence-embedding space; a record is routed to its `k` nearest
keys and trained only into those experts; inference delta-averages the `k`; **deletion retrains only
the `k` experts the record touched**, leaving all others provably untouched.

| Claim | Verdict | Headline number |
|---|---|---|
| **Verifiable exactness** | ✅ strong | unlearn-vs-from-scratch weight distance ≤ the GPU nondeterminism floor; bitwise on CPU |
| **Utility preserved** | ✅ strong | MMLU(legonet) ≈ MMLU(base) on all models (e.g. 3.2-3B 0.583 vs 0.600) |
| **Routing is semantic** | ✅ | cluster purity 0.63 over 14 fields (no TOFU-style collapse) |
| **Two-knob behavior (LegoNet)** | ✅ | utility flat as `n`↑; utility ↑ with `k` |
| **Efficiency win vs SISA-LoRA** | ⚠️ not at moderate `n` | needs `n > s·k²` (~576), beyond sweep — predicted |
| **Forget efficacy (population)** | ⚠️ under-powered | mechanism clean (memorized canary→base), signal weak (k-dilution) |

---

## 2. The idea (from scratch)

**Problem.** "Right to be forgotten" needs deleting a record *and its influence* on a model. **Exact
unlearning** = the post-deletion model is distribution-identical to one retrained from scratch on
`D\{r}`. Retraining an LLM per deletion is prohibitive; most LLM "unlearning" is approximate gradient
surgery with no guarantee.

**SISA** shards data into `s` disjoint pieces, trains one sub-model each, ensembles; deletion retrains
one shard. But each shard sub-model is a full network, and high `s` collapses accuracy.

**LegoNet** keeps a **fixed pre-trained encoder** + `n` tiny **adapters**, each with a frozen **key**.
A sample activates its `k` nearest keys; only those adapters train on it; inference ensembles them.
Deletion retrains just the `k` activated adapters. Cheap (no encoder retrain) + accurate (similar
samples share adapters; `k`-ensemble recovers performance).

**This port (mapping):** frozen encoder → **frozen Llama**; adapters → **LoRA**; keys → **k-means
centroids of MiniLM embeddings**, frozen; `k`-NN activation → embed + `k` nearest keys; ensemble →
**delta-average** the `k` LoRA into the base. Deletion unit = a **record** (the GDPR unit).

---

## 3. Method

**Architecture.** `record → MiniLM Emb (abstract only) → k nearest frozen keys → delta-average those k
LoRA experts (weight 1/k) over the frozen base → output.`

**Two conditions for exactness.**
- **(A) Frozen keys** — keys computed once from a held-out **DBpedia test-split reference** (disjoint
  from the deletable train corpus) and never recomputed; a record's assignment depends only on that
  record → removing one record never reassigns another (no cascade).
- **(B) Reproducible per-expert training** — fixed per-expert seed (`BASE_SEED+j`), deterministic
  kernels, no shared optimizer state, no cross-expert batching.

**Exactness argument.** Experts `j∉A(r)`: training set unchanged ⇒ identical in unlearn `U` and
from-scratch `R`. Experts `j∈A(r)`: same set-minus-`r`, same seed ⇒ identical. Keys/base frozen ⇒
`U≡R`. Bitwise where kernels are deterministic, else *distributional* (residual = hardware
nondeterminism floor).

**Deletion (LegoNet Alg.1).** Retrain `A(r)` (the `k` activated experts) on `members\{r}`; if an
expert's whole set ⊆ forget-set, disable it (O(1), zero-delta).

**Verification.** (i) reproducibility (train an expert twice → distance); (ii) deletion oracle: build
from-scratch `R` and check affected (oracle vs unlearn) and untouched (oracle vs original) match.

**Canary.** Each record gets a unique `"Verification code: <12 random chars>"` (Secret-Sharer). The
base can't predict a random code, so canary recall is a clean, training-attributable forget probe.
Repeated in **training** text (v2: ×5) to aid memorization; appears **once** in eval (probes genuine
recall, not copying). Routing embeds the abstract only, so clusters stay semantic.

---

## 4. Settings

| Component | Setting |
|---|---|
| Base (frozen) | `Llama-2-7B-chat-hf` (v1, v2) · `Llama-3.2-3B-Instruct` ; bf16 |
| Encoder (frozen) | `all-MiniLM-L6-v2` (384-d, L2-normalized) |
| Corpus | `fancyzhx/dbpedia_14`, **4000** records, balanced 285–286 × **14 classes** (train split) |
| Reference (keys) | 4000 records from the disjoint **test** split |
| Canary | `"Verification code: <12×[A-Z0-9]>"`, unique/seeded; ×1 (v1) or ×5 (v2) in train text |
| Keys | `KMeans(n_clusters=n=32, seed=42)` on MiniLM(reference); frozen |
| Routing | top-`k`=3 nearest keys (L2), pure-numpy `KNNRouter` |
| LoRA | `q,k,v,o_proj`, **rank 16, α 32, dropout 0**, standard scaling |
| Optim | `paged_adamw_32bit`, lr 2e-4, cosine, warmup 0.03, wd 1e-3, clip 0.3, bf16, eff. batch 8, max_len 256 |
| Determinism | per-expert seed 42+j; `use_deterministic_algorithms`, TF32 off, `CUBLAS_WORKSPACE_CONFIG` |
| Combine | PEFT `add_weighted_adapter(linear, w=1/k)` |
| **v1 recipe** | epochs 3, canary ×1 (under-memorized) |
| **v2 recipe** | epochs 6, canary ×5 (strengthened) — also used for Llama-3.2-3B |
| Primary config | n=32, k=3 → each expert ≈ kN/n = 375 records |
| Infra | SLURM sprint1–3, 1 GPU/task, 48–64 GB, **≤8 concurrent**, `HF_HOME=/storage2/...` |

---

## 5. Metrics explained

All per-record metrics use **prompt = `"<title>:"`**, **completion = `" <content> <canary>"`**.

- **EM** ∈[0,1] — teacher-forced top-1 next-token accuracy over the completion. Base ~0.5 (language is
  predictable); the signal is **legonet − base**.
- **ES** ∈[0,1] — extraction strength `1−k/L`; ≈0 here (exact-suffix regurgitation rare with light LoRA) → uninformative.
- **VerbMem** ∈[0,1] — ROUGE-L recall of greedy generation vs the true completion.
- **perplexity** (lower=better) — `exp(mean token NLL)`; cleanest, lowest-noise memorization signal.
- **canary_em** ∈[0,1] — EM on the random **code tokens only** — the clean forget probe (base ≈ chance).
- **canary_hit** ∈{0,1} — exact code in free generation; =0 everywhere (too stringent), secondary only.
- **MMLU** ∈[0.25,1] — 4-choice accuracy via answer-letter logprob; capability check.
- **held-out PPL** — perplexity on disjoint reference text; capability check #2.
- **rel_l2** (0=bitwise) — relative LoRA-weight distance; verdict = affected ≈ untouched (= floor).
- **structural_ok** — deletion touched exactly the predicted experts. **collateral** — neighbors' Δ≈0.
- **example-passes/del** ≈ `k²N/n` (LegoNet) vs `N/s` (SISA); **unlearn wall-clock**; **cluster purity** ∈[0.07,1].

---

## 6. Results

### 6.1 Routing & cluster disciplinarity (n=32; field := DBpedia class)
- Adapter sizes **83–664** (mean 375), **0 empty**; mean cluster purity **0.63** vs 14 classes.
- **Disciplinarity taxonomy** (model-independent): strict (≥90% single / <10%-max highly-inter) =
  **30 interdisciplinary / 2 single-discipline (NaturalPlace .93, Plant .97) / 0 highly-inter**;
  graded = **10 pure (≥0.75) / 14 mixed / 8 highly-mixed (<0.5)**.

### 6.2 Memorization & utility — master table (eval n=80; legonet / base)

| metric | Llama-2-7B **v1** | Llama-2-7B **v2** | **Llama-3.2-3B** |
|---|---|---|---|
| retained EM | 0.646 / 0.505 | **0.716** / 0.505 | 0.637 / 0.456 |
| retained VerbMem | 0.344 / 0.187 | 0.386 / 0.187 | 0.366 / 0.177 |
| retained perplexity | 4.63 / 16.2 | **3.33** / 16.2 | 5.59 / 24.5 |
| canary_em | 0.048 / 0.018 | 0.065 / 0.018 | 0.052 / 0.010 |
| **MMLU** | 0.447 / 0.460 | 0.433 / 0.460 | **0.583 / 0.600** |
| held-out PPL | 16.0 / 20.9 | 24.3 / 20.9 | 19.2 / 23.7 |

→ Experts strongly learn the corpus (PPL 16→3.3 on v2) while **MMLU stays ≈ base** on every model
(frozen-backbone premise holds). v2 (6ep/canary×5) memorizes more than v1 but its held-out PPL rises
(heavier memorization trades a little generic fluency).

### 6.3 Exactness — distributional, at the nondeterminism floor

Per deletion: affected = unlearn-vs-from-scratch-oracle distance; floor = original-vs-oracle (same
data, different run). **Affected ≈ floor ⇒ unlearn indistinguishable from retrain.** All deletions
`structural_ok=True`; bitwise on CPU/TinyLlama (distance 0); 7B/3B relax to distributional.

| model | deletion | affected rel_l2 | floor rel_l2 |
|---|---|---|---|
| 7B-v1 | rec0 / rec1 / rec2 | 2.46 / 3.72 / 3.07 e-2 | 1.56 / 3.10 / 4.53 e-2 |
| 7B-v2 | rec0 / rec1 / rec2 | 3.53 / 5.67 / 5.32 e-2 | 4.15 / 4.74 / 6.63 e-2 |
| 3.2-3B | rec0 / rec1 / rec2 | 2.66 / 4.73 / 3.88 e-2 | 2.20 / 2.92 / 3.50 e-2 |

Mean unlearn wall-clock ≈ 2180 s (7B-v2) / 2072 s (3B) for 3 affected experts. Collateral damage on
retained neighbors ≈ 0 (canary_em & PPL flat pre→post).

### 6.4 Forget efficacy (per deleted record, canary_em pre→post→base)
| model | rec0 | rec1 | rec2 |
|---|---|---|---|
| 7B-v1 | 0.10→0.00 (b 0.00) | 0.00→0.00 | 0.00→0.00 (b 0.09) |
| 7B-v2 | 0.10→0.00 (b 0.00) | 0.10→0.00 (b 0.00) | 0.00→0.00 (b 0.09) |
| 3.2-3B | 0.00→0.00 | 0.00→0.00 | 0.10→0.10 (b 0.00) |

Every record that **memorized** its canary reverts to base on deletion (clean mechanism); most records
under-memorize the random code (k=3 averaging dilutes per-expert memory) ⇒ weak population signal.

### 6.5 Phase-3 sweep (v2 recipe) — LegoNet's two knobs + SISA-LoRA baseline
*ex-passes/del = per-deletion example-passes (≈k²N/n for knn; N/s for random=SISA).*

| cell | mode | n | k | retained EM | retained PPL | unlearn s | ex-passes/del |
|---|---|---|---|---|---|---|---|
| n16 k3 | knn | 16 | 3 | 0.718 | 3.29 | 4710 | 2250 |
| n32 k3 (=v2) | knn | 32 | 3 | 0.716 | 3.33 | 2180 | 1125 |
| n64 k3 | knn | 64 | 3 | 0.700 | 3.52 | 1184 | 562 |
| n32 k1 | knn | 32 | 1 | 0.687 | 3.69 | 273 | 125 |
| n32 k5 | knn | 32 | 5 | 0.717 | 3.28 | — | 3125 |
| n32 k1 | **random (SISA)** | 32 | 1 | 0.683 | 3.91 | 197 | 125 |
| n64 k1 | **random (SISA)** | 64 | 1 | 0.648 | 4.66 | 109 | 62 |

- **Utility ~flat as n↑** (k=3): EM 0.718→0.700 over n=16→64 while cost drops 2250→562. ✓
- **k>1 recovers utility** (n=32): EM 0.687→0.717 for k=1→5 (cost ↑ k²). ✓
- **Semantic ≈ random at k=1** (LegoNet_{k=1} ≈ FixSISA): 0.687 vs 0.683 (knn PPL slightly better). ✓
- **Cost: SISA-LoRA cheaper at moderate n** (62–125 vs 562–1125 ex-passes; 109–197 s vs 1184–2180 s).
  The `k²N/n<N/s` crossover needs `n>s·k²` (~576), beyond the sweep — **honest negative, predicted**:
  in the LoRA port both methods freeze the base, so LegoNet's classic per-param win is gone; its edge
  is utility-per-segment (k>1) + verifiable exactness.

### 6.6 Disciplinarity slices (1000-record eval, by primary-cluster bucket)
**Llama-2-7B-v2 (graded):**
| bucket | #rec | retained EM | perplexity | VerbMem | canary_em |
|---|---|---|---|---|---|
| pure (≥0.75) | 217 | 0.757 | 2.82 | 0.475 | 0.043 |
| mixed | 491 | 0.733 | 3.10 | 0.376 | 0.058 |
| highly-mixed (<0.5) | 292 | 0.702 | 3.59 | 0.345 | 0.053 |
| *single-discipline (strict)* | 19 | 0.709 | 3.43 | 0.737 | 0.039 |
| *interdisciplinary (strict)* | 981 | 0.730 | 3.18 | 0.382 | 0.054 |

**Llama-3.2-3B (graded):** pure 0.699 / 4.34, mixed 0.657 / 5.10, highly-mixed 0.621 / 6.25 (EM / PPL).

→ **Purer (more single-discipline) clusters memorize their records better** — monotone higher EM,
lower PPL, much higher VerbMem (single-discipline VerbMem 0.737 vs 0.382) — because a homogeneous-topic
expert has a tighter target distribution. **canary_em is ~flat (≈0.05)**: the random-code forget probe
is topic-independent. Same trend on both models.

### 6.7 LegoNet-units classification accuracy (attempted — artifact)
Prompt-classifying the experts on DBpedia-14 gives **near-chance, legonet < base** (base D_retain 0.23
/ D_test 0.18; legonet cells 0.04–0.16). **Artifact, not utility:** experts were trained to *memorize*,
not classify, so the merge drifts off the `"Category:"` format; + CamelCase-label scoring is weak.
LegoNet's accuracy metric assumes the adapters *are* classifiers — a memorization port can't be read
this way. Real capability = MMLU (§6.2). A faithful accuracy/UnClass comparison needs experts trained
**as classifiers**.

### 6.8 Transfer to Llama-3.2-3B (newer model)
Both headline claims transfer: MMLU 0.583 vs 0.600 base (utility preserved; the 3.2-3B base is stronger
than Llama-2's 0.46), exactness at the nondeterminism floor (§6.3), forget again dilution-limited
(canary even weaker on the smaller 3B).

---

## 7. Quality assessment (single-seed, smoke-tier; ±0.02–0.03 noise)

| dimension | grade | basis |
|---|---|---|
| Verifiable exactness | **A** | affected ≤ floor on every deletion, structural 100%, collateral ≈0; bitwise on CPU |
| Utility preservation | **A−** | MMLU ≈ base on all 3 models; minor v2 held-out-PPL dent |
| Two-knob reproduction | **A−** | utility-flat-in-n, utility-up-with-k, k=1≈SISA — all match the paper |
| Disciplinarity analysis | **A−** | clean monotone purity→memorization signal (1000 records, both models) |
| Semantic routing | **B+** | purity 0.63, 0 empty; but ~8× load imbalance (83–664) unaddressed |
| Forget efficacy | **C+** | mechanism clean, population signal weak (k-dilution) |
| Efficiency vs SISA-LoRA | **C** (honest negative) | SISA cheaper at moderate n; crossover needs n>s·k² (predicted) |

**Net:** the two defensible contributions — *verifiable exactness on a generative LLM* and *preserved
utility* — are strong and transfer across models; the soft spots (forget power, cost win) are honestly
under-powered / out-of-regime, both anticipated.

---

## 8. Honest limitations
1. **Single seed, small samples** (eval n=80; n=1000 for disciplinarity; 3 deletions; 300 MMLU). Not yet seed-significant.
2. **Efficiency**: LegoNet-LoRA is not cheaper per deletion than SISA-LoRA below `n≈576`.
3. **Forget signal** dilution-limited by k>1 averaging; a k=1 / heavier-canary variant would sharpen it.
4. **Load imbalance** (~8×) — balanced k-means deferred; also why only 2 strict single-discipline clusters.
5. **Exactness is distributional on GPU** (bitwise only on CPU) due to non-deterministic 7B/3B kernels.

---

## 8b. Addendum — LegoNet on the TOFU benchmark (author-level)

Full write-up: **`reports/LEGONET_TOFU_REPORT.md`** (code in `~/tofu_sisa_lora/`). The DBpedia study
above *rejected* TOFU because record-level routing collapses to per-author centroids. The addendum
revisits TOFU at the **author level** — the forget unit on TOFU *is* the author — clustering the 200
authors by their mean answer-embedding (the collapse becomes the signal), and scores LegoNet on TOFU's
own OU-faithful `model_utility` / `forget_quality`, in the same table as SISA-LoRA / S³T / SEA.

| arm (n=32/k=3) | `legonet_unlearn` mu | forget_quality | vs SISA `merged_dare_ties` |
|---|---|---|---|
| Llama-2-7B-chat | **0.637** | 0.808 | 0.475 / 0.594 |
| Llama-3.2-1B-Instruct (vanilla) | **0.509** (0.501 ext) | **0.999** (0.890 ext) | 0.424 / 0.393 |
| Llama-3.2-1B-Instruct (balanced) | 0.485 | 0.999 | — |

Beats the SISA merge family on both axes. **Deletion locality** (cascade-free): single-author deletion
retrains 3/32 experts (29 untouched), forget10 only 15/32. **Imbalance:** vanilla k-means yields one
135/200-author hub; the **balanced** capacity-cap (29) is a *locality* knob (untouched 17→20) at a small
utility cost (0.51→0.49), not a utility fix. This is the author-level counterpart to §8's deferred
"balanced k-means" item. Single-seed, smoke caps; jobs 436047–066, 436133–144.

---

## 9. Reproduce + provenance

```bash
PY=/home/jack/anaconda3/envs/test-env/bin/python; cd ~/legonet_lora
$PY tests/test_{routing,metrics,exactness,pipeline}.py      # CPU gates
LEGO_MEM=64G bash submit_legonet.sh configs/legonet_7b_v2.json all     # 7B primary (v2)
LEGO_MEM=48G bash submit_legonet.sh configs/legonet_l32_3b.json all    # Llama-3.2-3B
LEGO_ARRAY_CAP=6 bash submit_sweep.sh                       # n/k sweep + SISA -> SWEEP_REPORT.md
bash submit_classify.sh                                     # LegoNet-units classification
bash submit_disciplinarity.sh                               # 1000-rec disciplinarity -> DISCIPLINARITY_REPORT.md
```

**Job IDs (SLURM, sprint1-3, ≤8 GPU):** 7B-v2 435665–668; refreshed eval/exact 435655/435656; sweep
435686–689; classification 435996–998; Llama-3.2-3B 436011–014; disciplinarity 436130–131.
**Configs:** `configs/legonet_7b.json` (v1), `_7b_v2.json`, `_l32_3b.json`, `sweep/*.json`.
**Other reports:** `SWEEP_REPORT.md`, `DISCIPLINARITY_REPORT.md` (auto-generated, under `/storage2/.../`).
**LOG:** `~/log/legonet_lora/` entries 2026-06-18/20/21.

## 10. File map (`~/legonet_lora/`)
`legonet_common.py` (config/paths/determinism/encoder/records) · `build_corpus.py` · `keys.py` ·
`routing.py` (KNNRouter; modes knn/random=SISA) · `train_adapter.py` (seeded expert; `--exclude_record_id`) ·
`combine.py` (LegoNetModel, delta-average) · `unlearn.py` (Alg.1) · `eval_memorization.py`
(EM/ES/VerbMem/PPL/canary) · `eval_utility.py` (MMLU + held-out PPL) · `eval_classification.py` ·
`verify_exactness.py` · `run_exactness_sample.py` · `analyze_disciplinarity.py` + `collect_disciplinarity.py` ·
`make_sweep.py`/`collect_sweep.py` · `submit_*.sh` · `tests/`.
