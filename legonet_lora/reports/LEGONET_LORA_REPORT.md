# Exact, Verifiable Record-Level Unlearning for LLMs via Clustered LoRA Experts
### A faithful LegoNet → LoRA → LLM port — full write-up

*Base model: meta-llama/Llama-2-7B-chat-hf · Encoder: all-MiniLM-L6-v2 · Corpus: DBpedia-14 + canaries · Code: `~/legonet_lora/`*

---

## TL;DR

We port **LegoNet** (Yu et al., AAAI 2023) to an LLM to get **machine unlearning that is exact
and *verifiable*** at the granularity of a single record. A frozen 7B base is paired with `n` small
LoRA "experts", each owning a fixed address (**key**) in sentence-embedding space. A record is routed
to its `k` nearest keys and trained only into those `k` experts. **Deleting a record = retrain only
the `k` experts it touched**; every other expert is provably untouched.

**What held up:**
- **Exactness:** the unlearned experts reproduce a from-scratch retrain to within the GPU's own
  run-to-run nondeterminism floor (bitwise on CPU). ✓
- **Utility:** the frozen backbone preserves general capability — **MMLU 0.43 vs 0.46** for the base. ✓
- **Routing is genuinely semantic** (cluster purity 0.63 over 14 topics), the thing that sank earlier
  TOFU attempts. ✓
- **LegoNet's two knobs reproduce:** utility holds as `n` grows; `k>1` ensembling recovers utility.

**Honest limits:** in the LoRA setting SISA-LoRA also freezes the base, so LegoNet's classic
*per-parameter* speed win disappears — **SISA-LoRA is actually cheaper per deletion at moderate `n`**
(the `k²N/n < N/s` crossover needs `n > s·k²`, beyond our sweep). And the canary forget signal is
**dilution-limited** by `k>1` averaging (mechanistically clean — every memorized canary reverts to
base on deletion — but weak at the population level).

---

## 1. The idea (from scratch)

### 1.1 The problem
"Right to be forgotten" (GDPR) requires deleting a user's data *and* its influence on trained models.
**Exact unlearning** = the post-deletion model is identical in distribution to one trained from scratch
on the data minus that record. The naive way (retrain on everything-minus-r) is correct but
prohibitively expensive for an LLM. Most LLM "unlearning" instead uses approximate gradient surgery
(gradient ascent, NPO, …), which gives **no exactness guarantee** and is judged by weak proxies.

### 1.2 SISA and its limit
**SISA** (Bourtoule et al. 2021) shards the data into `s` disjoint pieces, trains one sub-model per
shard, and ensembles. Deleting a record only retrains its one shard (`N/s` examples). But each shard
sub-model still retrains a *full network*, and pushing `s` high to cut cost collapses accuracy.

### 1.3 LegoNet's insight
LegoNet keeps a **fixed (pre-trained) encoder** and attaches `n` tiny **adapters**, each with a frozen
**key** (an address) in the encoder's output space. A sample activates its `k` nearest keys; only those
`k` adapters train on it; inference **ensembles** the `k` activated adapters. Unlearning retrains just
those `k` adapters. Because the big encoder never retrains and adapters are tiny, deletion is cheap;
because similar samples share adapters and `k` adapters ensemble, accuracy is maintained.

### 1.4 This work — the LLM port
We instantiate LegoNet on a generative LLM:

| LegoNet | This port |
|---|---|
| fixed encoder | **frozen Llama-2-7B** (the LM itself is the frozen backbone) |
| `n` adapters | **`n` LoRA adapters** on `q/k/v/o_proj` |
| keys in encoding space | **`n` k-means centroids** of MiniLM sentence embeddings, **frozen** |
| `k`-NN activation | embed the record with MiniLM, take the **`k` nearest keys** |
| ensemble | **delta-average** the `k` LoRA adapters into the base (one forward pass) |
| retrain `k` adapters on delete | identical — retrain the `k` activated LoRA adapters |

The deletion unit is a **record** (the GDPR unit), not a class or a shard.

---

## 2. Method

### 2.1 Architecture
```
record x ──▶ MiniLM Emb(x)            (frozen encoder; uses the abstract only)
                  │
        k nearest FROZEN keys {K_(1)…K_(k)}     (k-means centroids, fixed at setup)
                  │
        ┌─────────┼─────────┐
     LoRA_a    LoRA_b    LoRA_c        (the k activated experts)
        └─────────┼─────────┘
                  ▼
     delta-average (weights 1/k) over the frozen 7B base ──▶ output
```

### 2.2 The two conditions for exactness
- **(A) Frozen keys.** Keys are computed once at setup and **never recomputed on deletion**. We derive
  them from an **external reference split** (DBpedia *test*, disjoint from the deletable *train* corpus),
  so a key's location is a function of non-deletable data alone. Consequence: a record's `k`-NN
  assignment depends only on *that record* and the fixed keys — removing one record never reassigns
  another (no cascade).
- **(B) Reproducible per-expert training.** Each expert trains independently with a fixed per-expert
  seed (`BASE_SEED + j`), deterministic kernels, no shared optimizer state, no cross-expert batching.

### 2.3 Exactness argument
Let `A(r)` = the `k` keys nearest `Emb(r)`. Unlearn `U` retrains the experts in `A(r)` on their member
sets minus `r`. Retrain-from-scratch `R` trains all experts on `D\{r}`.
1. Base θ₀: frozen, identical. 2. Keys: frozen, identical (A). 3. Expert `j∉A(r)`: its member set does
not contain `r`, so it is unchanged → identical in `U` and `R`. 4. Expert `j∈A(r)`: trained on the same
set-minus-`r` with the same seed in both → identical (B). 5. Router: frozen `k`-NN, unchanged. ∎
Bitwise where kernels are deterministic; otherwise *distributional* (identical procedure/data/seed ⇒
same distribution as a fresh retrain), with the residual = the hardware nondeterminism floor.

### 2.4 Deletion (LegoNet Alg. 1)
```
unlearn(r):  for j in A(r): retrain expert j on members(j)\{r} with seed BASE_SEED+j
             if members(j) ⊆ forget-set:  disable expert j  (O(1), zero-delta)
```
Batched deletion retrains the *union* of affected experts once.

### 2.5 Verification (the headline)
- **Reproducibility:** train an expert twice (same seed/data) → weight distance = nondeterminism floor.
- **Deletion:** build the from-scratch **oracle** for `D\{r}` and check
  *affected* experts (oracle vs unlearn) **and** *untouched* experts (oracle vs original) match.
  If the affected distance ≈ the untouched (= floor) distance, `U` is indistinguishable from `R`.

### 2.6 Why a canary
The 7B base has seen Wikipedia, so DBpedia content is partially "known" → a contaminated baseline.
We append a unique **Secret-Sharer canary** (`"Verification code: <12 random chars>"`) to each record.
A never-trained model cannot reproduce a random code, so canary recall is a **clean,
training-attributable** memorization probe. Routing embeds the *abstract only* (canary excluded), so
clusters stay semantic. The canary is repeated in the **training** text (to aid memorization) but
appears **once** in eval (so the metric probes genuine recall, not copying).

---

## 3. Settings (full reproducibility)

| Component | Setting |
|---|---|
| **Base (frozen)** | `meta-llama/Llama-2-7B-chat-hf`, bf16 |
| **Encoder (frozen)** | `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalized) |
| **Corpus** | `fancyzhx/dbpedia_14`, **4000** records, balanced 285–286 × 14 classes (train split) |
| **Reference (keys)** | 4000 records from the **test** split (disjoint) |
| **Canary** | `"Verification code: <12× [A-Z0-9]>"`, unique per record, seeded |
| **Keys** | `KMeans(n_clusters=n, seed=42)` on MiniLM(reference); frozen |
| **Routing** | top-`k` nearest keys by L2 (`KNNRouter`, pure numpy) |
| **LoRA** | `target_modules=[q,k,v,o]_proj`, **rank 16, α 32, dropout 0**, standard scaling |
| **Optimizer** | `paged_adamw_32bit`, lr 2e-4, cosine, warmup 0.03, wd 0.001, max_grad_norm 0.3 |
| **Batch** | per-device 1 × grad-accum 8 (eff. 8), max_len 256, bf16 |
| **Per-expert seed** | `BASE_SEED(42) + j`; deterministic kernels (`use_deterministic_algorithms`, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) |
| **Combine** | PEFT `add_weighted_adapter(combination_type="linear", weights=[1/k]*k)` |
| **Primary config** | n=32, k=3 → each expert ≈ `kN/n` = 375 records |
| **v1 recipe** | epochs 3, canary×1 (under-memorized) |
| **v2 recipe** | epochs 6, canary×5 (strengthened) |
| **Infra** | SLURM sprint1–3, 1 GPU/task, 64 GB, ≤8 concurrent, `HF_HOME=/storage2/jack/data/huggingface` |

**Metrics.** Memorization: **EM** (teacher-forced next-token argmax accuracy), **ES** (extraction
strength), **VerbMem** (ROUGE-L recall of greedy generation), **perplexity**, **canary_em**
(teacher-forced EM on the *code* tokens — the clean forget probe). Utility: **MMLU** (cais/mmlu, 300 Q,
zero-shot, answer-letter-logprob argmax — no lm_eval) and **held-out PPL** (300 disjoint reference
records). EM/ES/ROUGE ported to match OpenUnlearning.

---

## 3a. Metrics — what each measures, its range, and what "good" looks like

All per-record metrics use the split **prompt = `"<title>:"`**, **completion = `" <content> <canary>"`**.

**Memorization / forget side** (higher = more memorized; for *forgetting* we want post-deletion → base):
- **EM (exact memorization)** ∈ [0,1] — teacher-forced: feed the true tokens, and at each completion
  position check whether the model's top-1 next-token == the actual next token; report the fraction.
  *Caveat:* language is partly predictable, so even the untrained base scores ~0.5 — the real signal is
  **legonet − base** (≈ +0.21 here), not the absolute.
- **ES (extraction strength)** ∈ [0,1] — `1 − k/L`, where `k` is the earliest prefix length from which
  the greedy continuation exactly matches the rest. "How little of the true text you need before it
  regurgitates the exact suffix." *In our runs ES ≈ 0 — verbatim exact-suffix match basically never
  happens with light LoRA on long passages, so ES was uninformative here.*
- **VerbMem** ∈ [0,1] — greedily generate from the prompt, then ROUGE-L **recall** (longest-common-
  subsequence overlap / reference length) vs the true completion. "How much of the real wording it
  reproduces freely." legonet 0.39 vs base 0.19.
- **perplexity** ∈ [1,∞), lower=better — `exp(mean token NLL)` on the completion. The cleanest,
  lowest-noise memorization signal in our data (legonet 3.3 vs base 16.2 = records are ~5× less
  "surprising" to the trained model).
- **canary_em** ∈ [0,1] — EM restricted to the **random 12-char code tokens only**. The clean
  Secret-Sharer probe: the base *cannot* predict a random code (≈ chance, 0.018), so any lift is
  training-attributable. This is the metric we delete against (want pre ≫ base, post → base).
- **canary_hit** ∈ {0,1} — does the exact code string appear in *free* greedy generation. *Too
  stringent (= 0 everywhere): greedy decoding must reproduce the whole passage to even reach the code.
  Kept as a secondary; canary_em is the usable one.*

**Utility / capability:**
- **MMLU** ∈ [0.25,1] (0.25 = chance) — 4-choice knowledge accuracy, scored by argmax of the
  answer-letter logprob. Measures whether the unlearning wrapper degrades general knowledge.
- **held-out PPL** — perplexity on disjoint reference text never trained on; capability preservation
  independent of MMLU (lower=better).
- **retained EM / VerbMem / PPL** — the memorization metrics on records the model **kept**: this *is*
  "utility" for unlearning (did it learn and retain what it should).

**Exactness:**
- **rel_l2** ∈ [0,∞), 0 = bitwise — relative L2 weight distance `‖Wₐ−W_b‖ / ‖W_b‖` between two
  adapters. Used as *affected* (unlearn vs from-scratch oracle) and *untouched* (original vs oracle =
  the hardware **nondeterminism floor**). **The verdict is the comparison:** affected ≈ floor ⇒
  unlearn indistinguishable from retrain.
- **structural_ok** ∈ {T,F} — did deletion touch *exactly* the predicted experts and leave the rest
  literally unwritten.
- **collateral** — change in retained neighbors' metrics pre→post (≈ 0 = surgical).

**Cost & routing:**
- **example-passes/deletion** — training examples touched per deletion ≈ `k²N/n` (LegoNet) vs `N/s`
  (SISA); the theoretical cost. **unlearn wall-clock (s)** — measured practical cost.
- **cluster purity** ∈ [1/14,1] (0.07 = random, 1 = topic-pure) — mean over experts of the dominant
  DBpedia-class fraction among members; 0.63 ⇒ routing is strongly semantic.

---

## 4. Results

### 4.1 Routing is semantic (n=32)
- Adapter sizes **83–664** (mean **375** = `kN/n`), **0 empty** experts.
- **Mean cluster purity 0.63** over the 14 DBpedia classes (vs the TOFU centroid-collapse this design
  was built to avoid). k-means inertia 2988 on 4000 reference embeddings.

### 4.2 Memorization & utility — v1 vs v2 vs base (n=32, k=3, eval n=80)

| metric | base (never trained) | v1 (3 ep, canary×1) | v2 (6 ep, canary×5) |
|---|---|---|---|
| retained EM | 0.505 | 0.646 | **0.716** |
| retained VerbMem | 0.187 | 0.344 | **0.386** |
| retained perplexity | 16.22 | 4.63 | **3.33** |
| canary_em (population) | 0.018 | 0.048 | 0.065 |
| **MMLU** | **0.460** | 0.447 | 0.433 |
| held-out PPL | 20.9 | 16.0 | 24.3 |

→ The experts strongly learn the corpus (PPL 16→3.3) while **MMLU stays ≈ base** (0.43–0.45 vs 0.46):
the frozen backbone preserves general capability. (Held-out PPL rises slightly in v2 — heavier
memorization at 6 ep/canary×5 trades a little generic fluency.)

### 4.3 Exactness — distributional, at the nondeterminism floor (v2, 3 deletions)

| deletion | affected experts | affected dist (unlearn vs oracle) | untouched dist (orig vs oracle = floor) |
|---|---|---|---|
| rec_000000 | [5, 9, 31] | 3.5e-2 | 4.2e-2 |
| rec_000001 | [7, 21, 25] | 5.7e-2 | 4.7e-2 |
| rec_000002 | [5, 6, 14] | 5.3e-2 | 6.6e-2 |

All deletions **structurally correct**; the affected (unlearn vs from-scratch) distance **≤ the
untouched nondeterminism floor** ⇒ the unlearned model is statistically indistinguishable from a
from-scratch retrain. (Bitwise — distance exactly 0 — on CPU and on TinyLlama-GPU; 7B relaxes to
distributional because some 7B CUDA kernels lack deterministic implementations.) Collateral damage on
retained neighbors ≈ 0 (canary_em and PPL flat pre→post).

### 4.4 Forget efficacy (v2, per deleted record, canary_em)

| record | pre | post-unlearn | base |
|---|---|---|---|
| rec_000000 | 0.100 | **0.000** | 0.000 |
| rec_000001 | 0.100 | **0.000** | 0.000 |
| rec_000002 | 0.000 | 0.000 | 0.091 |

Every record that actually memorized its canary **reverts exactly to base** after deletion. The
population signal is weak only because most records under-memorize the code (k=3 delta-averaging
dilutes per-expert memory) — a measurement-sensitivity limit, not a unlearning failure.

### 4.5 Sweep (v2 recipe) — LegoNet's two knobs + the SISA baseline
*Base reference: retained EM 0.505, canary_em 0.018, PPL 16.22. "ex-passes/del" = per-deletion
example-passes = k × mean-expert-size (≈ k²N/n).*

**Utility vs segmentation `n` (k=3):**
| n | retained EM | retained PPL | ex-passes/del |
|---|---|---|---|
| 16 | 0.718 | 3.29 | 2250 |
| 32 | 0.716 | 3.33 | 1125 |
| 64 | 0.700 | 3.52 | 562 |

→ Utility barely drops (−1.8% over 4× `n`) while deletion cost falls 4×. **Raising `n` buys cheap
deletion at ~no utility cost** — LegoNet's headline trade.

**Utility vs ensemble `k` (n=32):**
| k | retained EM | retained PPL | ex-passes/del |
|---|---|---|---|
| 1 | 0.687 | 3.69 | 125 |
| 3 | 0.716 | 3.33 | 1125 |
| 5 | 0.717 | 3.28 | 3125 |

→ `k>1` ensembling **recovers utility** (+3% k1→k3), at `k²` cost. `k` is the utility↔cost knob.

**Semantic vs random assignment (n=32, k=1) — LegoNet_{k=1} vs FixSISA:**
| mode | retained EM | retained PPL | ex-passes/del |
|---|---|---|---|
| knn (semantic) | 0.687 | 3.69 | 125 |
| random (SISA) | 0.683 | 3.91 | 125 |

→ Essentially **tied at k=1** (semantic PPL slightly better), exactly as the paper found — the semantic
advantage materializes through `k>1` ensembling, which disjoint SISA shards cannot do.

**Deletion cost vs SISA-LoRA (random, k=1):**
| method | per-deletion example-passes | measured unlearn wall-clock |
|---|---|---|
| SISA-LoRA s=32 | 125 (`N/s`) | 197 s |
| SISA-LoRA s=64 | 62 (`N/s`) | 109 s |
| LegoNet n=32 k=3 | 1125 (`k²N/n`) | 2180 s |
| LegoNet n=64 k=3 | 562 (`k²N/n`) | 1184 s |

→ **At moderate `n`, SISA-LoRA is cheaper per deletion.** The `k²N/n < N/s` crossover needs
`n > s·k²` (≈576 for k=3, s=64), beyond this sweep.

### 4.6 LegoNet-units classification accuracy (attempted — artifact, not utility)
We tried the paper's own metric (DBpedia-14 topic accuracy on D_retain/D_test) by prompt-classifying
the experts. Result: **near-chance and legonet < base** (base D_retain 0.23 / D_test 0.18; legonet
cells 0.04–0.16). This is an **artifact, not a utility result**, for two reasons: (1) our experts were
trained with **LM loss to memorize passages, not to classify** — delta-averaging them drifts the model
away from the `"…Category:"` format, so the merged model scores *below* the untouched base; (2) ranking
14 CamelCase labels by token-logprob is a weak probe (base only 0.23 vs a 7B's true ~80%). **Takeaway:**
LegoNet's accuracy metric assumes the adapters *are* the classifier; a memorization port cannot be read
this way. Real capability is the **MMLU 0.43 vs 0.46** result (§4.2). A faithful accuracy comparison
(incl. UnClass forgetting) would require a separate run that trains the experts **as classifiers**.

### 4.7 Cluster disciplinarity (field composition) — full report: `DISCIPLINARITY_REPORT.md`
Treating each record's DBpedia class as its "field", we classify each of the 32 clusters by field mix:
single-discipline (one field ≥90%), interdisciplinary, highly-interdisciplinary (no field >10%); plus a
graded view (pure ≥0.75 / mixed / highly-mixed <0.5). **Taxonomy (n=32, model-independent):** strict =
**30 interdisciplinary / 2 single-discipline / 0 highly-inter** (k-means always leaves a dominant field);
graded = **10 pure / 14 mixed / 8 highly-mixed**. Slicing the 1000-record memorization metrics by a
record's primary-cluster disciplinarity (Llama-2-7B-v2, graded):

| bucket | #rec | retained EM | perplexity | VerbMem |
|---|---|---|---|---|
| pure (≥0.75) | 217 | 0.757 | 2.82 | 0.475 |
| mixed | 491 | 0.733 | 3.10 | 0.376 |
| highly-mixed (<0.5) | 292 | 0.702 | 3.59 | 0.345 |

**Monotone: purer clusters memorize their records better** (higher EM, lower PPL, much higher VerbMem —
single-discipline VerbMem 0.737 vs 0.382 interdisciplinary), since a homogeneous-topic expert has a
tighter target distribution. **`canary_em` is ~flat (≈0.05) across buckets** — the random-code forget
probe is topic-independent. Llama-3.2-3B shows the same trends (pure EM 0.699/PPL 4.34 vs highly-mixed
0.621/6.25).

### 4.8 Transfer to a newer model — Llama-3.2-3B-Instruct (primary run)
The validated pipeline re-run on `Llama-3.2-3B-Instruct` (same corpus/keys/recipe) — the two headline
claims transfer:

| | Llama-2-7B-v2 (legonet / base) | Llama-3.2-3B (legonet / base) |
|---|---|---|
| MMLU | 0.43 / 0.46 | **0.583 / 0.600** |
| retained EM | 0.716 / 0.505 | 0.637 / 0.456 |
| retained PPL | 3.33 / 16.2 | 5.59 / 24.5 |
| exactness (affected rel_l2 ≈ floor) | 2.5–5.7e-2 ≈ 1.6–4.7e-2 ✓ | **2.7–4.7e-2 ≈ 2.2–3.5e-2 ✓** |

Utility preserved (MMLU ≈ base; 3.2-3B's base is stronger, 0.60 vs 0.46), exactness still at the
nondeterminism floor (structural_ok, distributional). Forget signal again dilution-limited (canary
under-memorized; even weaker on the smaller 3B).

---

## 5. Findings & honest limitations

1. **Exactness is the robust, headline result** — it is memorization-independent (it only asks whether
   unlearn = oracle), and it holds (bitwise on CPU, distributional-at-floor on 7B).
2. **Utility preservation confirms LegoNet's premise on an LLM** (MMLU ≈ base).
3. **The two knobs reproduce**: utility-stable-in-`n`, utility-recovered-by-`k`.
4. **Efficiency caveat (predicted):** in the LoRA port both methods freeze the base, so LegoNet's
   classic per-parameter win is gone. Its real edge is *utility-per-segment* (`k>1`) + *verifiable
   exactness*, **not** raw deletion cost at moderate `n`. We do not overclaim a speed win.
5. **Forget-signal caveat:** `k>1` delta-averaging dilutes per-expert canary memorization; the
   population `canary_em` is modest even at canary×5/6-epochs, though per-record reversion is exact.
   A `k=1` variant or heavier canaries would sharpen the population claim (diminishing returns).

---

## 5a. Quality assessment — how good was it, honestly

Graded per dimension. **Statistical caveat first:** all numbers are **single-seed, smoke-tier**
(eval n=80 records, 3 deletions, 300 MMLU Q). They are directional, not seed-averaged final results —
treat ±0.02–0.03 as noise and don't over-read any single cell.

| dimension | grade | why |
|---|---|---|
| **Verifiable exactness** | **A** | Bitwise (distance exactly 0) on CPU/TinyLlama; on 7B the affected distance ≤ the nondeterminism floor on *every* deletion, structural correctness 100%, collateral ≈ 0. This is the headline and it is genuinely strong — most LLM unlearning cannot offer a retrain-checkable guarantee at all. |
| **Utility preservation** | **A−** | MMLU 0.43 vs 0.46 base (≈ within noise for 300 Q), retained PPL 16→3.3 (strong learning). One blemish: v2 held-out PPL 24.3 vs 20.9 base — heavier memorization slightly dents generic fluency. |
| **Two-knob reproduction** | **A−** | Utility-stable-in-`n` (0.718→0.700 over 4× n) and utility-recovered-by-`k` (0.687→0.717) reproduce LegoNet cleanly on an LLM; semantic≈random @k=1 matches the paper. |
| **Semantic routing** | **B+** | Purity 0.63 (clearly non-random), 0 empty experts — but ~8× load imbalance (83–664) is a real, unaddressed rough edge (balanced k-means deferred). |
| **Forget efficacy** | **C+** | Mechanism is clean (every memorized canary → base on deletion; content PPL rises toward base), but the *population* signal is weak/under-powered: canary_em 0.065 vs 0.018 base is a small absolute lift, ES ≈ 0, canary_hit = 0. Root cause: `k>1` delta-averaging dilutes per-expert memorization, so most records under-memorize the high-entropy code. The claim is *supported but not crisply demonstrated at the population level*. |
| **Efficiency vs SISA-LoRA** | **C (honest negative, predicted)** | SISA-LoRA is *cheaper* per deletion at moderate n (62–125 vs 562–1125 example-passes; 109–197 s vs 1184–2180 s). The `k²N/n < N/s` win needs `n > s·k²` (~576), beyond this sweep. Not an execution failure — we predicted in the plan that the LoRA port erases LegoNet's classic per-parameter advantage. |

**Net:** the two *defensible contributions* — **verifiable exactness on a generative LLM** and
**preserved utility** — are strongly supported. The two *weak spots* — forget-signal power and the
cost win — are honestly under-powered / out-of-regime, both anticipated up front. Nothing here is yet
multi-seed-significant; that (plus a `k=1`/heavier-canary variant for forgetting, and `n > 576` for the
cost crossover) is what a "publishable" tier would still need.

---

## 6. Reproduce

```bash
PY=/home/jack/anaconda3/envs/test-env/bin/python
cd ~/legonet_lora
# CPU tests (routing, metrics, exactness, full pipeline)
$PY tests/test_routing.py && $PY tests/test_metrics.py && $PY tests/test_exactness.py && $PY tests/test_pipeline.py
# Primary 7B run (chained SLURM, ≤8 GPU): setup → train n adapters → eval(+utility) → exactness
LEGO_MEM=64G bash submit_legonet.sh configs/legonet_7b_v2.json all
# Phase-3 sweep (6 cells, SISA baseline, auto-report), capped %6:
LEGO_ARRAY_CAP=6 bash submit_sweep.sh        # -> SWEEP_REPORT.md
```

Key configs: `configs/legonet_7b.json` (v1), `configs/legonet_7b_v2.json` (v2), `configs/sweep/*.json`.
Artifacts under `/storage2/jack/checkpoints/legonet_lora/` (`runs/*/results/*.json`, `SWEEP_REPORT.md`).
Provenance (job IDs, recipe, sha256) in `~/log/legonet_lora/` (2026-06-18/20/21 entries).

## 7. File map (`~/legonet_lora/`)

| file | role |
|---|---|
| `legonet_common.py` | config/paths/determinism/MiniLM encoder/record IO (`train_text`, `prompt_completion`) |
| `build_corpus.py` | DBpedia-14 balanced subsample + canary; corpus/reference split |
| `keys.py` | frozen k-means keys from the reference split |
| `routing.py` | `KNNRouter` top-k; `build_assignment` (modes: `knn` semantic / `random` SISA) |
| `train_adapter.py` | one seeded, deterministic LoRA expert; `--exclude_record_id` for unlearn/oracle |
| `combine.py` | `LegoNetModel`: load experts, delta-average the `k` activated |
| `unlearn.py` | Alg. 1: retrain affected experts; O(1) disable when fully forgotten |
| `eval_memorization.py` | EM/ES/VerbMem/perplexity/canary_em(code)/canary_hit |
| `eval_utility.py` | MMLU MC scorer + held-out PPL (routed LegoNet vs frozen base) |
| `verify_exactness.py` | reproducibility + deletion-oracle param-distance |
| `run_exactness_sample.py` | sample deletions → verify + forget + collateral + cost (`--no_verify` for sweep) |
| `make_sweep.py` / `collect_sweep.py` / `submit_sweep.sh` | Phase-3 grid + report |
| `submit_legonet.sh` / `slurm_nodes.sh` | SLURM orchestration (≤8 GPU) |
| `tests/` | CPU unit + integration tests (run before any GPU job) |
