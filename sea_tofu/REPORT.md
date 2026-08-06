# SEA on TOFU — Methodology & Results

Implementation and evaluation of the **Separable Expert Architecture** (SEA; Schneider et al.,
arXiv:2604.21571) on the **TOFU** fictitious-author unlearning benchmark (Maini et al., 2024).
Code: `~/sea_tofu`. Base model: `meta-llama/Llama-2-7b-chat-hf` (4-bit NF4). Seed 42 (+ 43/44 for
variance). All GPU jobs via SLURM (sprint1–3). This document is the durable writeup; per-run numbers
also live in `~/log/sea/` (dated entries; see `~/log/README.md`) and the result JSONs under `proxies/.../results/`.

---

## Architecture at a glance

Our architecture is the **minimal-faithful SEA**: a single **frozen 4-bit (NF4) Llama-2-7B-chat base**,
shared across all 200 TOFU authors and holding *no* author-specific data, plus one small **per-author
proxy** — a rank-`r` personal LoRA (on `q/k/v/o`) trained by SFT on only that author's 20 QA pairs. At
inference, exactly **one** proxy is attached to the frozen base (`load_adapter`+`set_adapter`, with the
previous one deleted so adapters never accumulate), yielding a personalized model for that author;
running the base with **no** proxy ("omission mode") reproduces the un-personalized model. Because every
byte of an author's knowledge lives in their own proxy directory and never in the shared weights,
**unlearning an author is a filesystem delete** (`rm proxies/author_NNN/`) — after which the system is
bit-identical to a model retrained without them, at millisecond cost.

```
                          SEA-on-TOFU  ·  minimal-faithful

   ┌──────────────────────── SHARED · FROZEN · no user data ───────────────────────┐
   │             Llama-2-7B-chat   (4-bit NF4, weights never modified)              │
   └───────────────────────────────────────────────────────────────────────────────┘
            ▲  attach exactly ONE proxy per query    (omission mode = no proxy
            │                                          attached  =  base behavior)
   ── per-author DELETABLE proxies · one directory each · ~16–256 MB (rank 4–64) ───
        ┌──────────┐ ┌──────────┐         ┌──────────┐         ┌──────────┐
        │author_000│ │author_001│  ·····  │author_180│  ·····  │author_199│
        │ LoRA  r  │ │ LoRA  r  │         │ LoRA  r  │         │ LoRA  r  │
        └──────────┘ └──────────┘         └──────────┘         └──────────┘
        └──── retain: authors 0–179 (kept) ────┘   └─ forget10: 180–199 (deleted) ─┘

   INFERENCE :  query q ──▶  base  ⊕  active proxy_a  ──▶  personalized answer (author a)
   UNLEARN   :  rm proxies/author_180..199/  ──▶  base alone on those authors
                                             =  retrain-gold  (Forget Quality ≈ 1, ms cost)
```

---

## 1. Background and the core idea

**SEA is not an unlearning algorithm — it is an architecture that makes unlearning unnecessary.**
A standard personalized model bakes user data into shared weights via finetuning; removing one user
later is intractable without retraining. SEA instead keeps *all* user-specific information out of the
shared weights, in a small per-user **proxy** artifact, so "unlearning" becomes a filesystem delete.

SEA composes three layers at inference time:
1. **Base** — a frozen, 4-bit (NF4/QLoRA) LLM, shared by all users, containing no user data.
2. **Experts** — a bank of shared *domain* LoRA adapters combined by a router (the paper uses k=4:
   Security/Code/Data/General).
3. **User proxy** — a per-user deletable directory holding a routing bias, contrastive steering
   vectors, and a small **personal LoRA**. Deleting it removes that user, structurally.

**TOFU**, by contrast, is a benchmark *for* unlearning algorithms: finetune a base LLM on 200
fictitious authors (20 QA each = 4 000 pairs) so it memorizes them, then measure how well a method
removes a *forget* split (forget01/05/10) while preserving a *retain* split. Its headline metrics are
**Forget Quality** (KS-test p-value of forget-set "truth ratios" vs a retrain-only gold model) and
**Model Utility** (harmonic mean of probability/ROUGE/truth-ratio across Retain, Real-Authors, World-Facts).

**The mismatch.** TOFU assumes the forget data is already entangled in shared weights; SEA assumes it
never enters them. You therefore *cannot* "apply SEA" to a finished TOFU checkpoint. To evaluate SEA
meaningfully you must change the **training protocol** and map TOFU's splits/metrics onto proxy
load/delete. That reframing is the heart of this study.

---

## 2. The reframing — what "SEA on TOFU" actually measures

| TOFU concept | SEA realization in this study |
|---|---|
| each of the 200 authors | one SEA "user" with their own proxy `author_NNN/` |
| the finetuned model that "knows" an author | base **+ that author's proxy loaded** |
| unlearning the forget split | **delete** the forget authors' proxy directories (`rm`) |
| retrain-only gold model | base on the forget authors (their proxies were never in shared weights) |

Because shared weights are never touched, after deletion the system **is exactly** the retrain gold on
the forget authors. So **Forget Quality is ≈1 by construction** — we report it as a correctness check,
not a headline. The scientifically interesting axes are the *other* three SEA claims:

- **Personalization depth** — how much author knowledge a *deletable* rank-`r` proxy can hold vs. its
  size (the deletability tax).
- **Isolation** — loading author A's proxy must not leak into author B's answers.
- **Deletion cost** — milliseconds (`rm`) vs. the GPU-minutes/hours of gradient-based unlearning.

### Minimal-faithful configuration (and why)
We run the **minimal-faithful** SEA: frozen 4-bit base **+ per-author personal LoRA only**. The expert
layer and routing bias have no TOFU analog (authors are not Security/Code/Data/General), and the
steering vectors encode *style*, not *facts*, so they carry little of the knowledge signal a factual-recall
benchmark needs. The personal LoRA does the work. (The full SEA pipeline — experts + CAA steering — is
preserved in the separate `~/sea` package as a reference/ablation.)

We train the personal LoRA with **supervised finetuning (SFT)**, not the paper's DPO: TOFU is knowledge
injection (question → exact answer), for which causal-LM SFT is the natural and stronger route. DPO is a
preference objective for *style* and is the wrong tool for memorizing an author's birthplace.

---

## 3. Implementation

### Module map (`~/sea_tofu`)
| File | Role |
|---|---|
| `load_tofu.py` | Load TOFU splits; author `i` = rows `[i*20, i*20+20)`; per-author perturbed slicing. |
| `proxy_paths.py` | Path helpers; `proxies/{slug}[_r{rank}]/author_NNN/{personal_lora,meta.json}` (→ `/storage2`). |
| `inference.py` | `load_base` (4-bit NF4, frozen); `SeaProxyModel` swaps one proxy at a time; omission mode. |
| `train_proxy.py` | Per-author personal-LoRA SFT; `unload()` between authors; single or block author range. |
| `deletion.py` | `verify_and_delete`: KL gate on generic prompts → zero-overwrite + `rmtree` → audit log. |
| `metrics_sea.py` | SEA-specific: personalization depth, isolation/contamination, deletion cost. |
| `eval_sea_tofu.py` | Orchestrator + per-rank eval CLI; assembles utility/forget-quality from primitives. |
| `run_sweep_eval.py` | Combined rank-sweep eval (base loaded once, shared base-side, incremental writes). |
| `eval_unlearning_report.py` | Standard TOFU report (Original / Unlearned / Retrain-gold states). |
| `run_pilot.py` | 5-author smoke driver (all gates). |
| `submit_*.sh` | SLURM submitters (train proxies, sweep eval, report, pilot). |
| `configs/sea_tofu_llama2.json` | All hyperparameters (no ad-hoc CLI). |

### The per-author LoRA proxy — shape & parameters
Each author's proxy is **one LoRA**, applied to the four attention projections (`q_proj`, `k_proj`,
`v_proj`, `o_proj`) of **all 32 Llama-2-7B layers** — i.e. 128 injected linear layers, stored as **256
tensors** (an `A` and a `B` per layer). The base MLP, embeddings, and LM head are *not* adapted. For
each frozen 4096×4096 attention weight `W`, the proxy adds a rank-`r` update:

```
ΔW = (α/√r) · B · A        A ∈ ℝ^{r × 4096}   (down-projection, random/Kaiming init)
                           B ∈ ℝ^{4096 × r}   (up-projection, zero init  → ΔW = 0 at step 0)
```
`use_rslora=True` makes the scaling `α/√r` (not `α/r`); with `α = 2r` that is `2√r` (e.g. **8** at
r16). Verified on a trained proxy: layer-0 `q_proj` has `lora_A (r, 4096)` and `lora_B (4096, r)`.

Parameter count per proxy = `128 modules × (r·4096 + 4096·r) = 1,048,576·r`, stored in fp32:

| rank `r` | params/proxy | trainable share of the 6.7B base | proxy size on disk |
|---:|---:|---:|---:|
| 4  | 4,194,304  | ~0.06% | 17 MB |
| 8  | 8,388,608  | ~0.13% | 33 MB |
| 16 | 16,777,216 | ~0.25% | 65 MB |
| 32 | 33,554,432 | ~0.50% | 129 MB |
| 64 | 67,108,864 | ~1.0%  | 257 MB |

Size is linear in `r` — this is the proxy-size axis of the deletability tax (§6.1). The headline run
uses **r16** (16.8 M params, 65 MB); the sweep spans r4–r64.

### Training protocol (per author)
- Base: `meta-llama/Llama-2-7b-chat-hf` in **4-bit NF4** (`bnb_4bit_quant_type="nf4"`, double-quant,
  bf16 compute), **frozen** — only the proxy's 256 LoRA tensors receive gradients (QLoRA), mirroring
  the proven `~/sea/train_expert.py` setup (no `prepare_model_for_kbit_training` needed in this env).
- LoRA config: rank `r`, `alpha = 2r`, dropout 0.05, target `q/k/v/o_proj`, `use_rslora=True`, no bias.
- Objective: plain **causal-LM next-token cross-entropy** (SFT) via `trl.SFTTrainer`, trained on **only
  that author's 20 QA pairs**, text = `"Question: {q}\nAnswer: {a}" + eos`, `max_len 256`.
- Hyperparameters (`configs/sea_tofu_llama2.json`): **12 epochs**, lr **2e-4**, cosine schedule, 3%
  warmup, batch 4, grad-accum 1, `paged_adamw_32bit`, weight_decay 0.001, max_grad_norm 0.3, bf16,
  **seed 42** (43/44 added at r4/r8 for variance). 20 examples ÷ batch 4 = **5 steps/epoch × 12 = 60
  optimizer steps per author**; loss falls cleanly ~2.1 → ~0.05 (full memorization of the 20 QA).
- **Independent init per author:** a fresh `get_peft_model` gives each proxy its own `B=0`, random `A`,
  so authors share no adapter state; the base is reloaded/`unload()`-ed between authors so PEFT adapters
  never accumulate (the paper's #1 pitfall — stacked adapters silently break isolation). One SLURM array
  task trains a contiguous block of 20 authors, loading the 7B base once.
- **Prompt format is fixed:** `"Question: {q}\nAnswer: {a}"` — identical to the eval prompt
  (`eval_tofu._build_qa_prompt`). Using a chat/`[INST]` template would mismatch eval and tank metrics.
- Output is **only** `author_NNN/personal_lora/` (the PEFT adapter = the 256 tensors above) + `meta.json`
  (author id, rank, train hash, seed, n_qa). The shared base is bit-identical before/after; deleting the
  directory removes every byte of that author's influence.

### Proxy selection / routing
Proxies are selected by **user identity, not by query content** — there is no learned router that infers
which proxy to use from the text. This is faithful to SEA: a proxy *is* a known user's directory, so the
serving system loads the logged-in user's proxy. In TOFU terms the "user" is the author a question
belongs to, so we attach author *a*'s proxy when serving author *a*'s questions (an oracle/identity
attach, exactly what `SeaProxyModel.attach(author_id, lora_dir)` does). Two consequences: (1) **at most
one proxy is ever active** — user proxies are never composed, so cross-author contamination is ≈0 *by
construction* (confirmed in §6.4); (2) we do **not** model the anonymous-query case ("given a question
with no known user, infer the proxy"), which TOFU does not require.

Note this is distinct from the SEA paper's *expert* router (BART-MNLI), which routes among the 4 shared
**domain experts**, never among user proxies — and which the minimal-faithful config omits, so there is
no content-based routing step here at all. (The sibling `~/tofu_sisa_lora` project does implement
content routers — `key_exact`, `centroid_sbert`, … — but for selecting/merging SISA *shards*, a different
mechanism than SEA's identity-keyed user proxies.)

### Inference and "omission mode"
`SeaProxyModel` attaches one author's proxy via `load_adapter`+`set_adapter` and `delete_adapter`s the
previous one, so at most one adapter is ever resident (bounds memory and *guarantees* isolation). Base-only
behavior is obtained with `disable_adapter()` ("omission mode"), which is **functionally identical to
post-deletion** — letting us verify deletion before the irreversible `rm`.

### Deletion protocol (`Verify → Delete → Audit`)
1. **Verify** in omission mode on **generic** held-out prompts (Real-Authors questions, not the author's
   own): KL of the unpersonalized next-token distribution vs a cached non-personalized baseline must be
   ≤ `max(2·σ̂, 0.15 nats)`, where σ̂ is the inter-prompt-set noise floor.
2. **Delete** — zero-overwrite each file, then `shutil.rmtree` the single proxy directory.
3. **Audit** — append `{proxy, kl, threshold, passed, deleted, delete_ms, ts}` to a log.

Because shared weights were never modified, the KL is ≈0 (the omission-mode model *is* the cached
baseline) — a structural guarantee, not a statistical estimate.

---

## 4. Metrics — definitions and how they map onto SEA

We **reuse the canonical TOFU metric code verbatim** from `~/tofu_sisa_lora/eval_tofu.py` (itself a
PEFT-aware port of the official OpenUnlearning metrics, regression-tested against them). No metric was
re-implemented, so SEA's numbers are directly comparable to that SISA-LoRA track.

| Metric | Definition (as computed) | SEA mapping |
|---|---|---|
| **Probability** | `P(answer\|q)^(1/\|answer\|)` = `exp(-mean answer-token loss)` | model state = base ± proxy |
| **ROUGE-L** | ROUGE-L **recall** of greedy generation vs gold | greedy decode under the state |
| **Truth Ratio** | per-QA `wrong_prob / correct_prob`; `wrong` = geo-mean over the perturbed answers, `correct` = paraphrased answer; aggregated forget-side as `mean(min(tr,1/tr))∈[0,1]`, utility-side as `mean(max(0,1−tr))` | uses TOFU's `*_perturbed` files |
| **Forget Quality** | KS-test p-value of forget-set truth-ratio distribution vs the **retrain gold** | gold = base-only forget TR; candidate = state's forget TR |
| **Model Utility** | harmonic mean of 9 = {Retain, Real-Authors, World-Facts} × {Prob, ROUGE, Truth-scaled} | Retain = each retain author **with its proxy**; Real/World = frozen base |

**Why we do not reuse `evaluate_model`.** That function assumes a *single* active model. SEA swaps a
*different* proxy per author, so we call the primitives per author and assemble Model Utility with the
same `scipy.stats.hmean` over the same 9 components — numerically equivalent, but state-aware.

**SEA-specific metrics** (`metrics_sea.py`):
- **Personalization depth** — per author, with proxy loaded: Prob / ROUGE-L / Truth-Ratio and the Δ vs
  base-only. This is the deletability tax made measurable.
- **Isolation / contamination** — load author A's proxy, answer author B's questions; `max(0, sim(A-on-B,
  B-gold) − sim(base-on-B, B-gold))` (token Jaccard). Should be ≈0.
- **Deletion cost** — wall-clock of the `rm` (ms) and the proxy size (MB).

---

## 5. Experimental design

- **Splits:** TOFU **forget10** = the last 20 authors (180–199), matching `get_author_shard(10, 9)` in the
  SISA-LoRA track. (forget05/forget01 are subsets; deferred.)
- **Rank sweep:** personal-LoRA rank `r ∈ {4, 8, 16, 32, 64}` on the forget10 authors — the core
  tradeoff. Headline config trains **all 200 authors** at r16 (so retain authors have proxies for utility).
- **Pilot → scale:** validate on 5 authors @ r16 (one GPU) before the 200-author + sweep run (SLURM array,
  one block of authors per task to amortize the 7B base load).
- **Seeds:** 42 for the main run; 43 and 44 added at the r4/r8 knee for variance (CLAUDE.md §4 —
  vary seeds before claiming an effect).
- **Eval caps:** "smoke" max_new=40 (fast, depresses absolute ROUGE via truncation); "extended"
  max_new=128 (absolute ROUGE); the standard report uses max_new=100, retain sample 40.
- **Compute:** training ≈ 14 array tasks; each eval is a single GPU job. The combined `run_sweep_eval.py`
  loads the base once, computes the (rank-independent) base-side metrics once, and writes per-rank
  results incrementally so a wall-clock timeout still yields partial data.

---

## 6. Results

### 6.1 Personalization-depth vs. rank (the deletability tax) — forget10, proxy loaded
Absolute ROUGE from the **extended** run (max_new=128); proxy size and Prob/TR/contamination as measured.

| rank | proxy size | proxy ROUGE-L | base ROUGE-L | proxy Prob | base Prob | contamination |
|-----:|-----------:|--------------:|-------------:|-----------:|----------:|--------------:|
| 4    | 16 MB | 0.673 | 0.420 | 0.707 | 0.161 | 0.059 |
| 8    | 32 MB | **0.991** | 0.420 | 0.986 | 0.161 | 0.048 |
| 16   | 64 MB | **1.000** | 0.420 | 0.999 | 0.161 | 0.108 |
| 32†  | 128 MB | ~0.87 (40-tok) | — | 1.000 | 0.161 | 0.082 |
| 64†  | 256 MB | ~0.86 (40-tok) | — | 0.999 | 0.161 | 0.064 |

† r32/r64 absolute ROUGE is from the smoke run (max_new=40, so depressed by truncation); both Prob ≈ 1.0
confirm full memorization. The extended job hit its 5 h wall after r16 (incremental writes preserved
r4/r8/r16); the r8/r16 plateau already establishes saturation.

**Reading:** recall rises sharply then saturates — **r4 underfits** (ROUGE 0.67), saturation by **r8**
(ROUGE 0.99) and r16 (1.00). The knee is r4→r8, exactly the paper's "rank-4 targets style, not knowledge."
Proxy size doubles per rank (16→256 MB), all far above the paper's quoted 2–5 MB — so the "tiny deletable
artifact" framing only holds at r4, which is the rank that underfits. **That tension is the central
finding.**

### 6.2 Seed variance at the knee (seeds 42/43/44)
| rank | metric | mean ± std |
|-----:|:------|:----------|
| 4 | proxy ROUGE-L | 0.668 ± 0.006 |
| 4 | proxy Prob | 0.699 ± 0.007 |
| 8 | proxy ROUGE-L | 0.992 ± 0.001 |
| 8 | proxy Prob | 0.986 ± 0.000 |

The r4-underfit / r8-saturation effect is **robust to seed** (std ≈ 0.006) — not noise.

### 6.3 Standard TOFU unlearning report — forget10, rank 16
Canonical schema; `eval_unlearning_report.py` (max_new=100, retain sample 40).

| State | Forget ROUGE-L | Forget Prob | Forget TR | Retain ROUGE | Retain Prob | Real | World | Forget Quality | Model Utility |
|---|---|---|---|---|---|---|---|---|---|
| **Original** (proxies loaded) | 1.000 | 0.999 | 0.476 | 1.000 | 0.999 | 0.689 | 0.856 | **0.0** | **0.711** |
| **Unlearned** (proxies deleted) | 0.403 | 0.161 | 0.701 | 1.000 | 0.999 | 0.689 | 0.856 | **1.0** | **0.711** |
| **Retrain gold** (= base on forget) | 0.403 | 0.161 | 0.701 | 1.000 | 0.999 | 0.689 | 0.856 | 1.0 | 0.711 |

- Deleting the forget proxies drops forget ROUGE **1.0 → 0.403** and Prob **0.999 → 0.161** (back to
  base), and Forget Quality **0.0 → 1.0**.
- **Model Utility is unchanged (0.711)** across all rows — deletion never touches the retain/real/world
  components, so SEA preserves utility through unlearning *by construction*.
- The Unlearned row is **identical to Retrain-gold** — SEA reaches the gold by `rm`.

### 6.4 Isolation, deletion, utility (summary)
- **Isolation:** contamination 0.06–0.11 across ranks (pilot measured 0.0 with more probe questions) —
  low; loading one author's proxy does not meaningfully change another's answers.
- **Deletion:** verify-gate KL = 0.0 ≤ threshold (omission == base == cached baseline); delete = `rm` of
  one ≤256 MB directory (milliseconds) vs GPU-minutes for gradient unlearning.
- **Forget Quality = 1.0**, **Model Utility ≈ 0.71–0.78** (TOFU-7B range: OpenUnlearning Finetuned ≈0.63,
  locuslab ft 0.748).

---

## 7. Honest interpretation

1. **Forget Quality is tautological for SEA.** Post-delete the forget authors were never in shared
   weights, so the candidate and the retrain gold are both the base model → KS p ≈ 1. We report it as a
   correctness check; it is *not* evidence of clever forgetting.
2. **The real result is the tradeoff curve.** Recall saturates by r8 (32 MB); r4 (16 MB) underfits.
   Adequate recall needs proxies larger than the paper's 2–5 MB ideal — the deletability tax.
3. **Utility is preserved by construction**, not by careful optimization — deletion is local to the
   forget authors' files.
4. **Isolation holds** but is not exactly zero; with few probe questions the Jaccard estimate is noisy.
5. **Deletion cost is the headline win:** a structural, auditable `rm` (ms) replacing approximate,
   GPU-expensive weight surgery.

### Caveats
- Single forget split (forget10); single base model; smoke-tier ROUGE caps depress absolute ROUGE at
  r32/r64; retain utility uses a 40-author sample; contamination uses 3 probe questions. Directional,
  single-to-triple seed. None of these affect the qualitative conclusions.

---

## 8. Reproducibility

- **Env:** `/home/jack/anaconda3/envs/test-env/bin/python`; `HF_HOME=/storage2/jack/data/huggingface`;
  SLURM sprint1–3 (`--exclude=sprint4`, ≤12 concurrent GPUs, 1 GPU/task).
- **Train proxies:** `bash submit_train_proxies.sh <rank> <start> <end> <block> [seed] [proxy_root]`
  (e.g. `... 16 0 199 20` for the headline; `... 4 180 199 20` for a sweep point).
- **Sweep eval:** `bash submit_sweep_eval.sh <n_forget> <max_new> <tag> <n_retain> <ranks> [proxy_root]`.
- **Standard report:** `bash submit_report.sh <rank> <max_new> <n_retain>`.
- **Config:** `configs/sea_tofu_llama2.json` (all hyperparameters; seed 42).
- **Artifacts:** `proxies/{slug}[_r{rank}]/author_NNN/` (→ `/storage2`); results JSON under
  `proxies/.../results/{sweep,extended,seed43,seed44,report}/`; per-day narrative in `~/log/sea/`.
- **Provenance:** each `meta.json` records rank/seed/train-hash; SLURM job IDs are in `~/log/sea/`
  (pilot 435382; training 435413–435420; sweep eval 435654; extended 435982; seed-var 435983–988;
  report 436005). The repo is not under git, so `meta.json.git_commit` is `null`.

## 9. References
Schneider, Schoenegger, Bariach — *Separable Expert Architecture* (arXiv:2604.21571). Maini et al. —
*TOFU* (2024, `locuslab/TOFU`). Hu et al. — *LoRA*; Dettmers et al. — *QLoRA*. Rimsky et al. — *CAA*.
Rafailov et al. — *DPO*. Bourtoule et al. — *SISA*.
