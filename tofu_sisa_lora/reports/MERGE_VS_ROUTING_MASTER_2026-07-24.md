# Merging Doesn't Work — Master Tables + Router-vs-Merger (2026-07-24)

**Thesis in one line.** Across 2 model scales (Llama-3.2-1B, Llama-2-7B), 3 task-vector types
(LoRA, full-FT, sign-fixed full-FT), 5 PEFT parameterizations, and ~15 algorithms, **every
*separable* weight-space merge lands within noise of one ceiling — mu ≈ 0.41–0.48 ≈ base +
a small dilution.** The operators that leave the band do so by *breaking* (mu → 0.00–0.09), not by
improving. The **only** thing that escapes the ceiling *upward* is **serve-time selection**
(routing / inference-time masks) — which is also the only route to cheap O(1) exact deletion, the
project's actual goal. Part I proves the ceiling; Part II puts routing next to merging so the
merge-ceiling / routing-headroom gap is explicit.

**Provenance.** Every cell is read from the per-run result JSONs under
`checkpoints/*/results/{smoke,extended}/` (metrics_version `ou-2026-06-10`, seed 42, smoke tier
unless noted) and cross-checked against the vetted master reports
([MERGE_METHODS_RESULTS_2026-07-21.md](MERGE_METHODS_RESULTS_2026-07-21.md) for merging,
[ROUTING_MASTER_2026-07-23.md](ROUTING_MASTER_2026-07-23.md) for routing, and
`../../merge-tables-7b/RESULTS_TABLES.md` for the consolidated P1–P5 layout). Nothing is invented.
Spot-checked 2026-07-24: P1/P2/P3/P5 merge cells + routing 0.8236 verified equal to the on-disk
JSONs. ⚠ **Do not cite `reports/all_metrics_smoke.csv`** — it is a stale pre-metrics-fix snapshot
(`merged_dare_ties` 0.17 there vs the live 0.424); the live JSONs and the two master reports agree
and are authoritative.

---

## §0 — Read this first: the problem, the objects, and the three strategies

*(Skip if you already know TOFU / task vectors / merging. This section makes the tables readable
with no prior repo context. **Hit an unfamiliar symbol or abbreviation — τ, mu, fq, k, N, √r, OOD,
ρ? → Appendix C — Glossary.** Fuller method write-ups are in Appendix A; references in Appendix B;
the PDFs live in `../../papers/` and the in-repo survey is `../../papers/RELATED_WORK.md`.)*

**The task (TOFU).** TOFU (Maini et al. 2024) is an LLM unlearning benchmark. Take a pretrained chat
model θ₀, fine-tune it to memorize **200 fictitious author biographies** (20 Q&A each = 4,000 rows),
then demand it **unlearn** a subset — the `forget10` split = authors 180–199 — so the served model
behaves as if those authors were never trained on, while keeping everything else (the *retain* set,
plus general "real-author" and "world-fact" knowledge) intact. Because the authors are invented,
ground truth about what should and shouldn't be known is fully controlled.

**Exact vs approximate unlearning.** *Approximate* unlearning nudges the trained weights to look
forgetful (gradient ascent on the forget set, gradient difference, KL-to-reference, preference
objectives — the GA/GD/KL/IDK baselines) — cheap but brittle: the "forgotten" data is often
recoverable by attack (Wu et al. 2025). *Exact* unlearning guarantees the post-deletion model is
distributionally identical to one **retrained from scratch** on the retained data. The naive exact
route — full retrain — is too expensive for LLMs. This project's goal is exact unlearning as a
**cheap, deterministic O(1) operation**: store each author's contribution as a separate object and
*drop* it, no retraining. That is the lineage of **SISA** (Bourtoule et al. 2021: **S**harded,
**I**solated, **S**liced, **A**ggregated — shard the data, train one isolated model per shard,
delete = retrain only the affected shard).

**The core objects.**
- **Task vector** `τ_t = θ_t − θ₀` — the weight change from fine-tuning θ₀ on author/shard *t* alone
  (Ilharco et al. 2023). Adding τ installs a capability; subtracting it removes one.
- **Full-FT vs LoRA.** "Full-FT" keeps τ over *all* parameters. A **LoRA adapter** (Hu et al. 2021)
  stores τ compactly as a low-rank product `ΔW = (α/r)·BA` — two thin rank-`r` factors instead of a
  full d_out×d_in matrix. Merging LoRAs correctly means summing the *materialized* `BA` deltas, never
  averaging the A/B factors (that is a different, meaningless operator).
- **Shard vs per-author.** `k=10` → 20 authors per shard (pools P1/P2/P4); `k=200` / `T=200` → one
  adapter per author (P3/P5).
- **Separable merge = the exactness criterion.** Combining the τ_t into one served model
  `θ₀ + f(τ₁…τ_T)` is **separable** iff deleting author *t* changes the model by a term depending
  only on `τ_t` and fixed constants — never on the other authors' vectors. Then delete = subtract
  `τ_t`, exactly. Any operator with **cross-task statistics** (sign votes, a globally-tuned λ,
  learned combination weights, Gram/Fisher matrices over the *union* of authors) breaks separability
  and is a *baseline, not an exact method*. That distinction is the **"Exact?"** column throughout.

**The three strategies this report compares.**
1. **Merge** (Part I): fold all τ_t into one set of weights. Cheap to serve (one model), exact *if*
   the operator is separable — but it caps utility at the ceiling.
2. **Serve-time selection / routing** (Part II): keep the τ_t *separate* and pick which one(s) to
   apply per query; delete = drop the module. Costlier to serve (many modules resident) but clears
   the ceiling and gives exact O(1) deletion. A per-task inference **mask** (SIFT/ClAMU) is the same
   idea in disguise.
3. **Training-time constraints** (ctv, Table F′): train the τ_t under a structural constraint
   (disjoint supports, tangent space) so a *plain sum* stays high-utility AND exactly deletable — the
   "routerless exact" ideal; still an open question.

**Reading a pool label.** `P1 · 1B · k=10` = experiment pool 1, Llama-3.2-1B base, 10 shards. Each
pool has two anchors: **base** (never fine-tuned — the floor) and **FT** (joint fine-tune on
everything, or the k=1 single-author ceiling — the practical top). A merged model's mu is only
meaningful *between its own pool's base and FT* — hence the per-pool anchor legend below.

---

## Metric legend

| Metric | Meaning | Good direction |
|---|---|---|
| **mu** (`model_utility`) | Harmonic mean of 9 TOFU components (answer-prob / ROUGE / truth-ratio over retain authors, real authors, world facts). Harmonic ⇒ one zero component zeroes it. The headline "is the model still good" number. | higher |
| **fq** (`forget_quality`) | KS p-value vs a retain-only oracle. **Only meaningful for post-deletion (`*_unlearn` / `remerge_*`) rows** — a model still containing the forget authors is *supposed* to have low fq. | higher (post-deletion) |
| **f_rouge** (`forget_rouge`) | ROUGE-L recall on the forget split — verbatim recall. Good pre-deletion, bad post-deletion. | context |
| **r_ppl** (`retain_ppl`) | Retain-text perplexity. Explosions (≫20) mean the merge broke the language model. | lower |
| **Δ base / Δ FT** | mu gap to that pool's never-FT base and joint-FT ceiling (below). | — |

**How each metric is computed** (metrics are ported to match open-unlearning / locuslab exactly;
`eval_tofu.py`, guarded by `test_ou_equivalence.py`):
- **Answer probability** `P(a|q)^(1/|a|)` (length-normalized); for real-authors / world-facts it is
  `probability_w_options` = correct / (correct + Σ perturbed-wrong).
- **ROUGE-L recall** of a greedy generation against the gold answer (recall, not F1).
- **Truth ratio** per sample `tr = P(wrong)/P(correct)` (wrong = geometric mean over the `*_perturbed`
  splits). `forget_truth_ratio = mean(min(tr, 1/tr)) ∈ [0,1]` (→1 = more forgetting); each utility
  component = `mean(max(0, 1−tr))`.
- **mu** = `scipy.stats.hmean` of the 9 retain/real/world × prob/rouge/truth values.
- **fq** (`forget_quality`, alias `ks_pval`) = the KS-test p-value comparing the forget set's
  truth-ratio distribution to the cached **retain90 oracle** (an adapter trained only on authors
  0–179). High p ⇒ the forget distribution is indistinguishable from a model that never saw those
  authors ⇒ good unlearning. At k=200 only 20 forget rows survive, so the KS test loses power — read
  fq qualitatively there.
- **own-prob / own-rouge** (Table F′) = answer-prob / ROUGE restricted to a *single* author's own
  rows (`--eval_shard_id` / `--retain_author_ids`) — per-author memorization strength, isolated from
  the population average.

## Anchor legend (base / fine-tuned reference points per pool)

| Pool (model · granularity · tier) | mu base (never-FT) | mu finetuned (FT ceiling) |
|---|---|---|
| **P1** — Llama-3.2-1B · k=10 LoRA shards · smoke | 0.398 | 0.530 (`ft_all`) |
| **P2** — Llama-2-7B · k=4 LoRA shards · smoke | 0.426 | 0.756 (`ft_r32`) |
| **P3** — Llama-2-7B · k=200 per-author r32 LoRA · smoke | 0.426 | 0.756 (`ft_r32`) |
| **P4** — Llama-3.2-1B · k=10 · PEFT bake-off · smoke | 0.380 | 0.530 (`ft_all`) |
| **P5** — Llama-3.2-1B · T=200 per-author full-FT · smoke | 0.398 | 0.737 (k=1 full-FT ceiling) |

> **The 0.46 rule.** Any merge below ≈0.46 has essentially diluted back to the base model. The
> game is closing the 0.46 → 0.75 gap *without* a router. Nothing in Part I does.

## §0.1 — The five pools: models, training recipes, and config files

Every number in this report comes from one of these pools. The **frozen shard recipe** (winner of
the 2026-06-11 grid) is `rank 32 · α 64 · 5 epochs · lr 1e-4`, the flag-free default of
`train_lora_shard.py`; `e25` = 25 epochs (used for the well-trained per-author experts); the
`retain90` KS oracle keeps the legacy `r8/α16/e3/lr2e-4`. Full-FT pools (P5) train in fp32 with the
embedding + lm_head frozen. Configs live in `../../tofu_sisa_lora/configs/`; checkpoints under
`checkpoints/<dir>/results/smoke/` (symlinked to `/storage2/jack/checkpoints/tofu_sisa_lora/`).

| Pool | Model (HF id) | Granularity | Training recipe | Config file / driver | Checkpoint dir | Tables |
|---|---|---|---|---|---|---|
| **P1** | `meta-llama/Llama-3.2-1B-Instruct` | k=10 LoRA shards (20 authors/shard) | frozen r32/α64/e5/lr1e-4 | *no JSON* — `submit_overnight.sh` (train) / `submit_eval.sh` (eval), CLI `--k 10` | `Llama-3.2-1B-Instruct/` | A, B |
| **P2** | `meta-llama/Llama-2-7B-chat-hf` | k=4 LoRA shards (50 authors/shard) | frozen r32/e5/lr1e-4 | *no JSON* — `submit_llama2_grid_overnight.sh` etc., CLI `--k 4` | `Llama-2-7B-chat-hf_k4_r32_e5_lr1e4/` | A, C |
| **P3** | `meta-llama/Llama-2-7B-chat-hf` | k=200 per-author LoRA (r32 ladders; **r8** for in-model merges) | per-author r32 e5 (e25 for routing); r32 ladders materialized on CPU; the in-model operator battery runs at r8 (r32×200 exceeds a 46 GiB A40) | `configs/nmerge_interference_7b.json`, `nmerge_centered_7b.json`, `sparsify_7b.json` · `submit_nmerge.sh` / `submit_scale_grid.sh` · write-up [`../../merge-tables-7b/RESULTS_7B_K200.md`](../../merge-tables-7b/RESULTS_7B_K200.md) | `_k200_r32_e5_lr1e4/`, `_nmerge_r32/`, `_nmerge_r32_centered/`, `_k200_r8_e5_lr1e4/` | A, C′, D |
| **P4** | `meta-llama/Llama-3.2-1B-Instruct` | k=10, 4 PEFT parameterizations | per-method **standard** lrs (deliberately not recipe-matched; recorded in the config) | `configs/peft_bakeoff_1b.json` · `submit_peft_bakeoff.sh` | `_peft_{prefix,vera,ia3,dora}_k10/` | A, E |
| **P5** | `meta-llama/Llama-3.2-1B-Instruct` | T=200 per-author **full-FT** (non-LoRA) | fp32, embed+lm_head frozen; ClAMU K=16 default (`num_clusters`, `mask_steps`, `mask_lr`) | `configs/sift_masks_tofu_1b.json`, `clamu_tofu_1b.json` (+ `_K{1..200}`) · `submit_sift_masks_tofu.sh` / `submit_clamu_tofu.sh` | `_sift_masks/`, `_clamu/`, `_clamu_K{K}/` | A, F |
| **ctv** | `meta-llama/Llama-3.2-1B-Instruct` | T=200 per-author, constrained (solo N=1) | ctrl r32/e25; [wd]/[lin] variants; [ds] full-FT 0.5% support | `configs/ctv_1b_{ctrl,wd,lin,ds}.json` · `submit_ctv.sh` | `_ctv_{ctrl,lin,wd}_r32_e25/`, `_ctv_ds_e25/` | F′ |

Routing pools reuse the same experts on a **scaffolded base** (`make_scaffolded_base.py` bakes a public-Alpaca scaffold LoRA into θ₀): 1B k=10 → `_experts_scaf_k10/`; 7B k=200 e25 → `_k200_r32_e25_lr1e4/` (served by `eval_routed_scaffold.py`). Anchor `ft_all`/`ft_r32` = the joint fine-tune on all authors (`{slug}_ft`).

---

# PART I — The merge ceiling ("merging doesn't work")

## Table A — GRAND MASTER: every merge operator, with mu

One row per (operator × pool), sorted into three bands so the thesis is visual. `Δbase`/`ΔFT` are
vs that row's pool anchors. **Exact?** = is deletion an exact algebraic/bitwise drop-a-term, or does
the operator fold in cross-task statistics (sign votes, learned weights, pool means) that make
deletion approximate?

**Models (per the `Pool` column, see §0.1):** `1B` = `meta-llama/Llama-3.2-1B-Instruct`; `7B` =
`meta-llama/Llama-2-7B-chat-hf`. `k10`/`k4` = shards; `k200`/`T200` = one adapter per author.

### Band 1 — In-band plateau (mu ≈ base + dilution; the ceiling)

| Method | Pool (model · k) | mu | Δbase | ΔFT | fq | f_rouge | Exact? |
|---|---|---:|---:|---:|---:|---:|---|
| Uniform mean λ=1/k (`additive_mean`) | P1 · 1B · k10 | 0.419 | +0.021 | −0.111 | 0.999 | 0.461 | **Exact** (algebraic) |
| Tuned-λ sum, λ=0.05 (`additive_s0.05`) | P1 · 1B · k10 | 0.429 | +0.031 | −0.101 | 0.808 | 0.461 | **Exact** (fixed λ) |
| DARE-TIES (frozen default) | P1 · 1B · k10 | 0.424 | +0.026 | −0.106 | 0.393 | 0.437 | Not exact |
| DELLA-TIES | P1 · 1B · k10 | 0.429 | +0.031 | −0.101 | 0.393 | 0.432 | Not exact |
| Fisher-weighted | P1 · 1B · k10 | 0.424 | +0.026 | −0.106 | 0.393 | 0.432 | Not exact |
| KnOTS (shared-SVD + TIES) | P1 · 1B · k10 | 0.424 | +0.026 | −0.106 | 0.393 | 0.443 | Not exact |
| Breadcrumbs λ=1/(n√r) | P1 · 1B · k10 | 0.419 | +0.021 | −0.111 | 0.958 | 0.454 | **Exact** (fixed λ) |
| Subtract-orth *(unlearn op)* | P1 · 1B · k10 | 0.433 | +0.035 | −0.097 | 0.594 | 0.464 | deletion op |
| LoRA (additive mean) | P4 · 1B · k10 | 0.419 | +0.039 | −0.111 | — | 0.461 | **Exact** |
| DoRA (additive mean) | P4 · 1B · k10 | 0.432 | +0.052 | −0.098 | — | 0.449 | **Exact** |
| IA³ (gate arith-mean) | P4 · 1B · k10 | 0.430 | +0.050 | −0.100 | — | 0.428 | **Exact** (O(1)) |
| VeRA (shared frozen basis) | P4 · 1B · k10 | 0.415 | +0.035 | −0.115 | — | 0.439 | **Exact** |
| JD-full (Compress-then-Serve) | P2 · 7B · k4 | 0.501 | +0.075 | −0.255 | 0.958 | 0.508 | O(1) drop, approx (shared basis) |
| JD-diag | P2 · 7B · k4 | 0.501 | +0.075 | −0.255 | 0.958 | 0.512 | O(1) drop, approx (shared basis) |
| DELLA-TIES | P2 · 7B · k4 | 0.497 | +0.071 | −0.259 | 0.071 | 0.522 | Not exact |
| KnOTS | P2 · 7B · k4 | 0.489 | +0.063 | −0.267 | 0.594 | 0.522 | Not exact |
| Fisher | P2 · 7B · k4 | 0.477 | +0.051 | −0.279 | 0.958 | 0.519 | Not exact |
| `additive_mean` (per-author, N=200) | P3 · 7B · k200 | 0.460 | +0.034 | −0.296 | — | — | **Exact** (algebraic) |
| LoraHub (per-author) | P3 · 7B · k200 | 0.451 | +0.025 | −0.305 | 0.174 | 0.343 | Not exact (learned) |
| PEFT linear (per-author) | P3 · 7B · k200 | 0.450 | +0.024 | −0.306 | 0.174 | 0.354 | in-band at r8 (√r-degen only at r32) |
| Breadcrumbs λ=0.005 (per-author) | P3 · 7B · k200 | 0.445 | +0.019 | −0.311 | 0.174 | 0.374 | **Exact** (fixed λ) |
| TSV-M (per-author) | P3 · 7B · k200 | 0.437 | +0.011 | −0.319 | 0.174 | 0.364 | Not exact |
| DARE-TIES / TIES / Fisher / RegMean / DELLA-TIES | P3 · 7B · k200 | 0.419–0.420 | ≈−0.006 | ≈−0.336 | 0.174 | 0.35–0.36 | mixed (see Table C′) |
| **FT+Merge** `merge_full` (sum, no mask) | P5 · 1B · T200 | 0.407 | +0.009 | −0.330 | 0.099 | 0.383 | **Exact** (algebraic) |

**Reading:** 19 operators — 5 parameterizations, 2 model scales, LoRA and full-FT — and every one
sits in **0.41–0.50**, i.e. base + a small dilution, far below every FT ceiling. Algorithmic
sophistication (Fisher, KnOTS shared-SVD, DELLA magnitude-aware pruning, JD joint-diagonalization)
buys nothing over a plain 1/k average. The exact ones (`additive_*`, DoRA/IA³/VeRA mean, FT+Merge)
and the not-exact ones land in the *same* band — exactness is not what costs the utility; **merging
is.** The P3 rows are a compressed view of the **full 7B k=200 operator battery — Table C′** (~20
operators, all in the same band at one adapter per author).

### Band 2 — Broke downward (mu 0.00–0.09; the operator collapsed the model)

| Method | Pool (model · k) | mu | fq | f_rouge | r_ppl | Why it broke |
|---|---|---:|---:|---:|---:|---|
| Naive sum λ=1 (`additive_s1`) | P1 · 1B · k10 | 0.000 | 0.007 | 0.001 | 4·10⁵ | summed deltas grow with k → activations off-distribution |
| Breadcrumbs λ=1/n | P1 · 1B · k10 | 0.000 | 0.958 | 0.396 | 11.7 | √r-inflated scale (mask itself is fine at 1/(n√r)) |
| PEFT linear (`merged_linear`) | P1 · 1B · k10 | 0.050 | 0.594 | 0.376 | 18.4 | `sqrt(w·scaling)` double-counts the rsLoRA √r → ~2.8× inflation |
| TSV-M (whitened top-singular) | P1 · 1B · k10 | 0.051 | 0.594 | 0.429 | 12.3 | collapses real-authors components on this pool |
| SLERP (tree, pairwise) | P1 · 1B · k10 | 0.090 | 0.594 | 0.395 | 11.8 | spherical interp of independent experts is OOD |
| Task-arith subtraction (`subtract_linear`) *(unlearn op)* | P1 · 1B · k10 | 0.000 | 0.007 | 0.000 | 3·10⁵ | negative-weight task arithmetic collapses |
| Prefix-tuning (KV concat) | P4 · 1B · k10 | 0.002 | — | 0.153 | — | independently-trained prefixes are mutually OOD |
| PEFT linear / DELLA-linear / Breadcrumbs (unscaled) | P2 · 7B · k4 | 0.000 / 0.000 / 0.0001 | — | ≈0 | — | same √r-inflation, three faces |

**Reading:** the only operators that leave the plateau leave it *downward*. There is no merge in
this project's registry that leaves the band by going *up*.

**⚠ Rank/k caveat on PEFT-linear.** The collapse above is the rsLoRA √r double-count at **r32**
(pools P1/P2, k=4/k=10). At **r8/k=200** (pool P3) the inflation factor is smaller and PEFT-linear
does **not** collapse — it lands *in-band* at the top (0.450, Table C′). So the degeneracy is
rank/scale-dependent, not intrinsic to the operator; read the "degenerate" rows as "degenerate *at
this rank*," not universally broken.

### Band 3 — Partial win, but not exact and/or small-k only

| Method | Pool (model · k) | mu | Δbase | ΔFT | fq | f_rouge | Catch |
|---|---|---:|---:|---:|---:|---:|---|
| LoraHub (learned weights) | P2 · 7B · k4 | **0.592** | +0.166 | −0.164 | 0.808 | 0.512 | learned weights are cross-task statistics → **deletion not exact** |
| DARE-TIES | P2 · 7B · k4 | 0.545 | +0.119 | −0.211 | 0.808 | 0.535 | **k=4 only** — decays to 0.420 by k=200 (Table C) |
| JD-full / JD-diag | P2 · 7B · k4 | 0.501 | +0.075 | −0.255 | 0.958 | 0.508 | O(1) deletion, but shared basis fit on everyone → approximate |

**Reading:** the only merges that beat base+0.15 live in the tiny-k (4-task) regime, and each pays
for it: LoraHub abandons exactness (fitted weights), DARE-TIES is a small-k artifact that dilutes to
base as k grows, JD's O(1) drop is approximate (shared basis). None survives *both* many tasks *and*
exact deletion.

---

## Table B — P1 widest battery (Llama-3.2-1B, k=10) — the canonical plateau

**Model:** `meta-llama/Llama-3.2-1B-Instruct` · pool **P1** (k=10 LoRA shards, frozen r32/α64/e5/lr1e-4; §0.1).
The single pool with the most operators run head-to-head. Base 0.398 / FT 0.530. Routing rows
(*italic*) are references, not merges — see Part II.

| Method | mu | fq | f_rouge | r_ppl | Exact deletion? |
|---|---:|---:|---:|---:|---|
| Naive sum λ=1 (`additive_s1`) | 0.000 | 0.007 | 0.001 | 4·10⁵ | Exact (algebraic) |
| Uniform mean λ=1/k (`additive_mean`) | 0.419 | 0.999 | 0.461 | 7.3 | Exact (algebraic) |
| Tuned-λ sum, λ=0.05 (`additive_s0.05`) | 0.429 | 0.808 | 0.461 | 8.2 | Exact (fixed λ) |
| DARE-TIES (frozen default) | 0.424 | 0.393 | 0.437 | 13.2 | Not exact |
| DELLA-TIES | 0.429 | 0.393 | 0.432 | 10.1 | Not exact |
| Fisher-weighted | 0.424 | 0.393 | 0.432 | 13.5 | Not exact |
| KnOTS (shared-SVD + TIES) | 0.424 | 0.393 | 0.443 | 13.1 | Not exact |
| Breadcrumbs λ=1/(n√r) | 0.419 | 0.958 | 0.454 | 7.6 | Exact (fixed λ) |
| Breadcrumbs λ=1/n | 0.000 | 0.958 | 0.396 | 11.7 | Exact (fixed λ) |
| PEFT linear (√r-inflated) | 0.050 | 0.594 | 0.376 | 18.4 | degenerate |
| TSV-M (whitened top-singular) | 0.051 | 0.594 | 0.429 | 12.3 | Not exact |
| SLERP (tree, pairwise) | 0.090 | 0.594 | 0.395 | 11.8 | Not exact |
| Subtract-orth *(unlearn op)* | 0.433 | 0.594 | 0.464 | 9.9 | deletion operator |
| Task-arith subtraction (`subtract_linear`) *(unlearn op)* | 0.000 | 0.007 | 0.000 | 3·10⁵ | deletion operator |
| *Routing key-exact (reference)* | *0.458* | *—* | *0.477* | *3.6* | *Exact module-drop* |
| *Routing + scaffold, OOD-aware (reference)* | ***0.556*** | *—* | *0.532* | *4.1* | *Exact module-drop* |

**Reading:** every separable merge lands in the **0.42–0.43 band** (≈ base + dilution). The only
thing above 0.46 is serve-time selection — and OOD-aware routing (**0.556**) beats even joint
fine-tuning (0.530) on this pool.

---

## Table C — The dilution law: DARE-TIES vs shard count (Llama-2-7B, smoke)

**Model:** `meta-llama/Llama-2-7B-chat-hf` · one pool per k (`_k{4,10,20,50,100}_r32_e5_lr1e4/`,
`_k200_r8_e5_lr1e4/`; the k=4 point is pool **P2**). Same `dare_ties` operator, swept over k. Base
0.426 / FT 0.756 throughout.

| k (shards) | 4 | 10 | 20 | 50 | 100 | 200 (r8) |
|---|---:|---:|---:|---:|---:|---:|
| `merged_dare_ties` mu | **0.545** | 0.477 | 0.450 | 0.438 | 0.430 | 0.420 |
| `routed_key_exact` mu | — | — | — | **0.715** | 0.648 | 0.473\* |

**Reading:** the k=4 "win" (0.545) is a small-k artifact — monotone decay to the base floor
(0.420) by k=200. Routing does *not* decay the same way (0.715 at k=50). \*k=200 routing here is
r8-capacity-limited, not a routing failure: the r32 scaffold pool reaches 0.751+ (Part II).

## Table C′ — 7B k=200 full merge battery (one author per shard)

**Model:** `meta-llama/Llama-2-7B-chat-hf` · pool **P3**, in-model merges on the **r8** pool
(`_k200_r8_e5_lr1e4/`; the r32 pool can't hold 200 in-model adapters on a 46 GiB A40) + the
materialized r32 ladders. Base 0.426 / FT 0.756; **fq = 0.1745 across all** (pre-deletion — the model
still holds every author, so low fq is expected). Source, on-disk-verified:
[`../../merge-tables-7b/RESULTS_7B_K200.md`](../../merge-tables-7b/RESULTS_7B_K200.md). This same
battery was **independently reproduced on A100** (the 2026-07-25 run) — every mu matches to within
~0.002 (largest gap TSV 0.0022); that run's ft anchor (0.742) and fq column come from a *retrained*
pool/oracle and are not used here.

| Method | mu | Δbase | ΔFT | fq | Exact deletion? |
|---|---:|---:|---:|---:|---|
| `additive_mean` (1/N, N=200) | **0.4597** | +0.034 | −0.296 | — | **Exact** (algebraic) |
| LoraHub (learned wts) | 0.4505 | +0.024 | −0.306 | 0.1745 | Not exact (learned) |
| PEFT linear | 0.4503 | +0.024 | −0.306 | 0.1745 | **in-band at r8/k200** (√r-degenerate only at r32; see nuance) |
| Breadcrumbs λ=0.005 | 0.4447 | +0.019 | −0.311 | 0.1745 | Exact (fixed λ) |
| TSV-M | 0.4366 | +0.011 | −0.319 | 0.1745 | Not exact |
| Breadcrumbs λ=0.00177 | 0.4338 | +0.008 | −0.322 | 0.1745 | Exact (fixed λ) |
| Subtract-orth *(unlearn op)* | 0.4272 | +0.001 | −0.329 | 0.1745 | deletion operator |
| SLERP (tree) | 0.4253 | −0.001 | −0.331 | 0.1745 | Not exact |
| DARE-TIES (frozen default) | 0.4201 | −0.006 | −0.336 | 0.1745 | Not exact |
| TIES | 0.4201 | −0.006 | −0.336 | 0.1745 | Not exact |
| Fisher-weighted | 0.4200 | −0.006 | −0.336 | 0.1745 | Not exact |
| RegMean | 0.4197 | −0.006 | −0.336 | 0.1745 | Not exact |
| DELLA-TIES | 0.4193 | −0.007 | −0.337 | 0.1745 | Not exact |
| centered_lowrank r16 (N=200) | 0.4092 | −0.017 | −0.347 | — | Not exact (pool mean) |
| KnOTS (shared-SVD + TIES) | OOM† | — | — | — | Not exact |
| JD (jd_full / jd_diag) | not run‡ | — | — | — | O(1) drop, approx |

† KnOTS's shared-SVD over 200 adapters OOMs on a 44.5 GiB A40 — a hardware limit, not a result (the
1B k=10 value is 0.424, squarely in-band). ‡ The JD build hit `torch.linalg.svd: failed to converge`
on the r8 collection; the 7B k=4 JD value was 0.501 (Table A), i.e. in-band. Sparse post-hoc
transforms (w5) agree: mean-composed dare/hash/topk ≈ 0.45–0.46, while the naive-**sum** variants
(`dare0p9sum`/`dare0p99sum`) collapse to 0.00 (RESULTS_7B_K200 Table B).

**Reading:** ~20 operators, one adapter per author, on a **4× larger model at maximum granularity** —
and every separable merge still sits in the **0.42–0.46 band** (base 0.426 / FT 0.756). The sign-vote
family (TIES/DARE-TIES/DELLA/RegMean/Fisher) clusters at ≈0.420 = base; the linear family
(linear/LoraHub/breadcrumbs/tsv) reaches 0.44–0.45; nothing approaches the routing ceiling (0.82,
Part II). The 1B finding is not a small-model artifact — **no separable merge beats base+dilution at
7B k=200 either.**

## Table D — Per-author N-merge ladder (P3: Llama-2-7B, k=200 r32, true-mean)

**Model:** `meta-llama/Llama-2-7B-chat-hf` · pool **P3** (`configs/nmerge_interference_7b.json` →
`_nmerge_r32/`; `nmerge_centered_7b.json` → `_nmerge_r32_centered/`). 200 single-author LoRAs, N
merged at a time (`merge_subset.py`). Base 0.426 / FT 0.756. Headline-probe rows.

| N merged | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `additive_mean` mu | 0.460 | 0.461 | 0.459 | 0.460 | 0.458 | 0.460 | 0.459 | 0.460 |
| centered `cr16` mu | 0.459 | 0.462 | 0.461 | 0.466 | 0.462 | 0.458 | 0.440 | 0.409 |

**Reading:** population mu is **flat in N** — the "dilution curve" was shard-*size*, not shard-count.
What collapses is per-author recall (≈85% gone by N≈8). Centered merging (subtract the pool-mean
before summing) is the only rule that moved the per-author knee (N≈3 → N\*≈64), but it ends *below
base* at N=200 — residual cross-talk beats dilution at full scale.

---

## Table E — Operator-independence: PEFT parameterization bake-off (P4: Llama-3.2-1B, k=10)

**Model:** `meta-llama/Llama-3.2-1B-Instruct` · pool **P4** (`configs/peft_bakeoff_1b.json` →
`_peft_{prefix,vera,ia3,dora}_k10/`; per-method standard lrs). Is the ceiling a LoRA artifact? No —
every weight-space composer plateaus at base+0.04. Base 0.380 / FT 0.530.

| Method (compose rule) | composed mu | routed mu | iso mu (s9) | comp f_rouge | Exact deletion? |
|---|---:|---:|---:|---:|---|
| LoRA (additive mean) | 0.419 | — | — | 0.461 | Exact |
| DoRA (additive mean) | 0.432 | 0.491 | 0.384 | 0.449 | Exact |
| IA³ (gate arith-mean) | 0.430 | 0.516 | 0.396 | 0.428 | Exact (O(1)) |
| IA³ (gate geo-mean) | 0.430 | — | 0.396 | 0.430 | Exact (O(1)) |
| VeRA (shared frozen basis) | 0.415 | 0.445 | 0.399 | 0.439 | Exact |
| Prefix-tuning (KV concat) | 0.002 | 0.000 | 0.074 | 0.153 | Exact (byte segment-drop) |

**Reading:** LoRA / DoRA / IA³ / VeRA all compose to the same ≈base+0.04 plateau; prefix-tuning
breaks. The merge ceiling is a property of weight-space composition, **not** of the LoRA
parameterization. (Note the `routed mu` column already breaks the plateau — Part II.)

## Table F — Full-parameter task vectors (P5: Llama-3.2-1B, T=200 per-author full-FT)

**Model:** `meta-llama/Llama-3.2-1B-Instruct` · pool **P5** (`configs/sift_masks_tofu_1b.json` →
`_sift_masks/`; `clamu_tofu_1b.json` + `_K{K}` → `_clamu/`, `_clamu_K{K}/`; full-FT fp32,
embed+lm_head frozen). The non-LoRA track: sum-of-full-FT-task-vectors, ± serve-time masks. Base
0.398 / k=1 FT ceiling 0.737.

| Method / condition | mu | fq | f_rouge | Exact deletion? |
|---|---:|---:|---:|---|
| **SIFT-Masks** `sift_full` (sum + inference-time mask) | **0.737** | 0.135\* | 0.834 | Exact (bitwise); mask = router |
| **FT+Merge** `merge_full` (same sum, **no mask**) | 0.407 | 0.099 | 0.383 | Exact (algebraic) |
| SIFT-Masks `sift_unlearn` (subtract 20 τ) | 0.738 | 0.393 | 0.339 | Exact (GPU bitwise) |
| ClAMU — Global (no mask) | 0.351 | 0.594 | 0.336 | Exact (algebraic) |
| ClAMU — EMR mask | 0.388 | 0.808 | 0.356 | Exact; mask = serve-time select |
| ClAMU — TALL mask | 0.405 | 0.393 | 0.360 | Exact; mask = serve-time select |
| ClAMU — optimized mask (K=16) | 0.647 | 0.239 | 0.593 | Exact; mask = serve-time select |
| ClAMU — optimized mask (K=200 peak) | 0.672 | 0.393 | 0.641 | Exact; mask = serve-time select |

\*fq of `*_full` rows is expected-low (the model still contains the forget authors). **Same lesson
as LoRA:** the identical sum-of-task-vectors merges to base (`merge_full` 0.407, ClAMU Global 0.351);
utility only returns when a **serve-time mask** re-carves each task out (SIFT 0.737, ClAMU 0.672).
The mask is a per-task router in disguise.

## Table F′ — (support) ctv training-time constructions (solo N=1, before any merge)

**Model:** `meta-llama/Llama-3.2-1B-Instruct` · **ctv** thread (`configs/ctv_1b_{ctrl,wd,lin,ds}.json`,
`submit_ctv.sh`; §0.1). Can a *constrained* per-author vector even memorize, pre-merge? Base own-prob
floor 0.146; ctrl solo mu 0.514.

| Arm | solo mu | own-prob | own-rouge | Verdict |
|---|---:|---:|---:|---|
| **ctrl** — plain per-author LoRA | 0.514 | 0.997 | 1.000 | anchor ✓ |
| **[lin]** tangent-space (linearized) | 0.000 | 0.9999 | 1.000 | memorizes perfectly; linearized serving zeroes utility |
| **[wd]** write-disjoint col(B) | 0.460 | 0.191 | 0.404 | KILLED — the write-subspace constraint can't memorize |
| **[ds]** disjoint-support full-FT | 0.644 | 0.498 | 0.475 | H-ds-1 refuted as stated; healthy served utility |

**Reading:** trainability, not cross-talk, is the current binding constraint for a *routerless*
exact point — [wd] can't memorize, [ds] pays half its association strength to the support mask
before any merging even happens.

---

# PART II — Router + merger, side by side

Same experts, same weights — the question is whether you **merge** them into one model (Part I) or
**select** among them at serve time (this part). Selection is the only lever that clears the ceiling.

## Table G — Merge ceiling vs routing headroom, per pool

**Models (per row):** P1/P4/P5 = `meta-llama/Llama-3.2-1B-Instruct`; P2/P3 = `meta-llama/Llama-2-7B-chat-hf`
(§0.1). Routing/mask columns use the same pool's experts served via `eval_routed_scaffold.py` (LoRA
pools) or the SIFT/ClAMU mask (P5).

| Pool (model · granularity) | base mu | best **merge** mu (method) | best **routing / mask** mu (method) | FT anchor |
|---|---:|---|---|---:|
| P1 · 1B · k=10 LoRA | 0.398 | 0.433 (subtract-orth) | **0.556** (routed+scaffold, OOD-aware) | 0.530 |
| P2 · 7B · k=4 LoRA | 0.426 | 0.592 (LoraHub, *not exact*) / 0.501 (JD, exact-ish) | 0.715 (key-exact — @k50; no k=4 route run) → 0.824 (@k200 e25) | 0.756 |
| P3 · 7B · k=200 per-author | 0.426 | 0.460 (additive_mean) | **0.824** (oracle route, e25) | 0.756 |
| P4 · 1B · k=10 PEFT | 0.380 | 0.432 (DoRA mean) | 0.516 (IA³ routed) | 0.530 |
| P5 · 1B · T=200 full-FT | 0.398 | 0.407 (FT+Merge) | **0.737 / 0.672** (SIFT mask / ClAMU mask) | 0.737 |

**Reading:** in every pool the best serve-time-selection number beats the best merge number by a
wide margin, and in P1 (0.556 > 0.530) and P5 (0.737 = k=1 ceiling) it beats or matches even joint
fine-tuning. The merge column never clears ≈0.46 by an exact method; the selection column reaches
0.72–0.82.

## Table H — Routing methods master (utility & deletion)

The full serve-time-selection inventory across threads (from
[ROUTING_MASTER_2026-07-23.md](ROUTING_MASTER_2026-07-23.md) Table 1). The **Model used** column names
each thread's model(s) — HF ids: `Llama-2-7B` = `meta-llama/Llama-2-7B-chat-hf`, `Llama-3.2-1B` =
`meta-llama/Llama-3.2-1B-Instruct`, plus 3B/TinyLlama/phi-2 where a thread swept scale. "Routed mu"
is the served utility; tags `[OU]`/`[F(d)]`/`[MMLU]` mark a *non-`model_utility`* scale (not directly
comparable — see ROUTING_MASTER); "Deletion" is the O(1) drop demonstrated.

| Method / thread | Model used | Routed mu | Base | FT | Routing mechanism | Deletion (Δmu · fq) | Orphan behavior |
|---|---|---:|---:|---:|---|---|---|
| **routing_scaffold** (core · oracle k=200 e25) | Llama-2-7B | **0.8236** (repo best) | 0.418–0.426 | ≈0.756 | Exact `q2author` → base + per-author expert; OOD → intact base | Δ0.0000; fq 0.336→0.175 (KS artifact) | 0 leak (oracle) |
| **routing_scaffold** (scaffold-routed k=10) | Llama-3.2-1B | **0.7509** | ≈0.42 | 0.6372 (matched-FT) | 10 shard experts on scaffolded base; lexical `q2author`; OOD → scaffold | Δ0.0000; fq 0.135→0.393 | 0 leak (oracle); embed router leaks |
| **legonet_lora** | 7B · 3B · 1B · TinyLlama · phi-2 | 0.6371 (7B); 0.5011 (1B) | 0.38–0.44 | ≈0.62 (7B) | Frozen base + n=32 MiniLM-centroid keys, top-3 kNN, 1/k avg | fq 0.808; 17/32 adapters byte-identical | two-magnet, n_eff 6.1 |
| **sisa_lora** (routed serving arm) | 1B · 3B · TinyLlama · phi-2 | 0.7147 (key-exact @k50) | ≈0.42 | 0.7435 (k=1 LoRA) | Shard LoRA by author key; shares `router.py` | exact drop-route | shares router.py strategies |
| **memory_adapters** | Llama-3.2-1B | 0.869 `[OU]` | — | 0.874 (retrained) | Top-32 over 1024² product-key memory, frozen router | −∞ block-list; 5,120 entries in 0.027 s | content router; cross-source ≈0.10 |
| **sea** (per-author proxy select) | 4-bit Llama-2-7B | 0.711 | 0.420 | 0.63 / 0.748 (locus) | Select the queried author's LoRA proxy; delete = drop it | fq 0.0→1.0 (rm proxy) | Group-A leak (misroute → surviving proxy) |
| **clamu** (per-cluster mask select) | Llama-3.2-1B | 0.647 (K200 0.672) | ≈0.42 | 0.530 (joint-ft) | MiniLM k-means K clusters; per-cluster STE mask | subtract τ + re-cluster; deletion *raises* mu | Group-B no leak (misroute floor 0.124) |
| **memsinks** (routed-mask) | Llama-3.2-1B | 0.6417 | ≈0.42 | ≈0.644 (ctrl) | Activate only the queried author's sink slice (oracle mask) | bake-zero slice bit-exact; fq 0.135→0.393 | routes ≡ SIFT; deletion = mask ⇒ robust |
| **peft_compose** (IA³ routed arm) | Llama-3.2-1B · k10 | 0.5155 | 0.3796 | 0.5302 (joint-ft) | Author-key routing selects the shard's IA³ adapter | exact everywhere | router.py leak on embed |
| **s3t** (ensemble / serve-best) | Llama-2-7B | 0.581 `[F(d)]` | 0.418 | — | Alg-4 deactivate downstream slices + prob-ensemble | δ 45.8→71.9; mask op ~ms | no router (ensemble) |
| _Router-free foils_ | | | | | | | |
| **sift_masks** | Llama-3.2-1B | 0.737 | 0.398 | 0.737 | per-task bitmask (= router in disguise) | subtract 20 τ, bitwise-exact | mask = per-task select |
| **sepmlp** | Llama-3.2-1B | ≈0.795 recall (K200) | — | — | **No router** — per-author ReLU-gated bottleneck branches summed | slice removal 1.07 s | no orphan routing |

**Reading:** serve-time selection spans **0.51–0.82** — a completely different regime from the merge
plateau (0.41–0.48). The exact-deletion story is *better* here too: dropping a module/mask is an
O(1) operation with Δmu≈0.

### What "orphan behavior" actually means, method by method

You delete an author. One of their questions arrives anyway. **Where does it go, and what does the
user get back?** That is the whole column. Plain answer per method:

| Method | What happens to the deleted author's question |
|---|---|
| **routing_scaffold** (oracle, k=200 and k=10) | The router looks the author's name up in a table. The deleted author is not in the table, so **nothing matches and the question goes to the plain base model**. The user gets a generic non-answer. 0 of 400 orphans reach a surviving expert. |
| **sisa_lora** (routed) | Same router code. With `key_exact`, a question that matches no name **falls back to shard 0** — so orphans all pile onto one shard, which then answers about the wrong authors. |
| **legonet_lora** | The router has no name table — it picks the **nearest surviving expert by embedding similarity**. All 400 orphans get answered by *some other author's* adapter. They are not spread evenly: experts e5, e11 and e30 absorb most of them (effectively ~6 destinations out of 17). |
| **peft_compose** (IA³) | Author-key route, so orphans **go to the base model** — same as routing_scaffold. Only if you swap in an embedding router does it start sending them to other authors' adapters. |
| **sea** | The deleted author's LoRA proxy file is removed. A misrouted orphan then reaches **another author's proxy, which still contains that author's real trained knowledge** — so the user can get real content about a person they didn't ask about. This is the one case where the destination actually holds deletable data. |
| **sift_masks** | The author's data is *subtracted out of the shared weights*, and the per-author bitmask is dropped. A misrouted orphan reaches a surviving author's mask — but that mask operates on weights the deleted author is **already gone from**, so nothing about them can come back. Measured: 0.086 answer-probability on the deleted fact vs 0.128 for the untouched base model. |
| **clamu** | Same idea, per-cluster instead of per-author. Misrouted orphans land on a surviving cluster's mask: **0.124 vs base 0.128** — i.e. floor. The text that comes out *looks* biographical and scores +0.142 on ROUGE, but 77% of it matches neither the real answer nor the base model's — it is invented, not recalled. |
| **memsinks** | The author's "sink slice" of the weights is zeroed. A misrouted orphan reaches a surviving author's slice with **the deleted slice switched off**, so again there is nothing to recall. (Routing measured; the serving half was never run — this row is reasoned from the mechanism, not observed.) |
| **memory_adapters** | No author units at all — every *token* looks up the 32 best entries in a big memory table. Deletion sets the author's 256 entries to −∞, so those lookups **redistribute onto whatever entries survive**. About 10% of read weight already crossed between authors before any deletion. |
| **s3t** | **No routing exists.** Every shard answers every question and the answers are averaged. Deleting changes the one averaged model; there is no orphan to send anywhere. |
| **sepmlp** | **No routing exists.** Every author's branch is active for every question, summed inside the weights. Deletion removes a slice. Again, no orphan to send anywhere. |

**The one distinction that matters.** Two things can be at the far end of a misroute:

- **Real content** — the surviving unit still holds another author's trained knowledge
  (**legonet, sea, and any embedding-routed expert pool**). A misroute here can disclose.
- **Nothing** — the deleted author was subtracted from, or masked out of, shared weights
  (**sift_masks, clamu, memsinks**). A misroute here produces confident invented text, which is an
  integrity problem, not a privacy one.

So "misrouting" alone is not the risk. Misrouting **into a unit that still holds someone's data** is.

**Three practical notes.** (i) "0 leak (oracle)" in the table means the routing *rule* makes it
impossible — an identity lookup can't match a deleted name — not that anyone measured it.
(ii) The **memsinks** and **sea** cells are reasoned from how the method works; their serving runs
were never executed. (iii) A row's orphan numbers may come from a **different model** than its mu:
legonet's mu is 7B but its concentration figures are the 1B pool, and sea's mu is 4-bit 7B while its
routing comes from a 1B measurement. Table H′ below separates these.


## Table H′ — the Llama-2-7B rows only (mu **and** orphan behavior, specific numbers)

Table H mixes models by design — it inventories methods, not a single scale. This table asks the
narrower question: *holding the model at `meta-llama/Llama-2-7B-chat-hf`, what do we actually
know?* The honest answer: **3 of 12 rows** carry a 7B `model_utility` verifiable from the result
JSONs, **1 more** carries one that is *recorded* but not verifiable here, 2 carry a 7B number on a
different scale, and **6 rows have no 7B run at all**.

| Method | 7B mu | tier · source JSON | 7B orphan behavior | kind |
|---|---:|---|---|---|
| **routing_scaffold** (oracle, k=200 e25) | **0.8236** | smoke · `_k200_r32_e25_lr1e4/results/smoke/routed_oracle_full.json` | Oracle `q2author`: orphan→base capture **1.000**, sibling capture **0.000**, retain-shift **0.0000**, both drop sets. Realistic routers on the *same pool* → H′.1 | by-construction (+ measured foil) |
| **sisa_lora** (routed serving arm) | **0.7147** @k50 · 0.6475 @k100 · 0.4728 @k200-r8 | smoke · `_k{50,100,200}_.../results/smoke/routed_key_exact.json` | Shares `router.py`; the 7B battery is H′.1. k=4/10/20 mu and a k=10/k=50 7B battery are being filled (§H′.3) | pointer → measured |
| **legonet_lora** | **0.6371** | smoke · `_legonet_n32_k3/results/smoke/legonet_unlearn.json` (post-deletion label; `legonet_full` = 0.6277) | **Not measured at 7B.** The two-magnet `n_eff 6.1` / e5(92)·e11(92)·e30(73) numbers are the **1B** n=32 pool | ✗ spliced from 1B |
| **peft_compose** (IA³ routed) | 0.6473 ⚠ | smoke · `checkpoints_7b/..._peft_ia3_k10/results/smoke/routed_key_exact.json` — **CISPA A100 box; those JSONs are not on this machine**, so the cell is *recorded*, never verified here (composed 0.4900; dora routed 0.5709 / composed 0.4985; vera 0.4972 / 0.4719; prefix collapses at 0.0348) | Not measured at 7B. Shipped route is the exact author key ⇒ orphans → base; the leak is conditional on swapping in an embed router | recorded · by-construction |
| **sea** (per-author proxy) | 0.711 | external · `sea_tofu/reports/SEA_UNLEARNING_REPORT.md` — **4-bit**, not comparable to the fp/bf16 mu column | **Not measured at 7B.** Routing borrowed from the **1B** SIFT per-author centroids (n_eff 27.1, magnet author 88); Group-A leak verdict from mechanism, no end-to-end serve | ✗ spliced from 1B · predicted |
| **s3t** | 0.581 `[F(d)]` | `_s3t_m5_L4_armB/F_curve.json` — an ensemble performance curve, **not** `model_utility` | n/a — no router; every shard contributes to every query | not applicable |

⚠ **The IA³ row is the one cell here that Table E deliberately excludes.** The A100 7B PEFT
bake-off reproduces the headline *shape* — routed (0.6473) beats composed (0.4900) at 7B exactly as
at 1B (0.5155 vs 0.415–0.434) — but its checkpoints are on another cluster and unverified from this
snapshot, so it must not be read as an on-disk result. It is included here because omitting it
would misstate the coverage question this table exists to answer.

**mu ladder context (same model, same pools):** the 7B merge battery — ~20 operators at k=200 —
sits at **0.419–0.451** (Table C′), against base 0.426 / FT 0.756. So at 7B the routing column
(0.637–0.824) clears the merge plateau by the same margin the 1B pools show. The merge-vs-route
gap is not a small-model artifact.

### H′.1 — the one 7B orphan battery, in full

Pool `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` (one author per expert), from
`results/router_leak/rl_family_k200.json` — **the only 7B orphan measurement that exists.** Two drop
sets: **d199** = one author deleted (20 orphans, 199 survivors); **d180–199** = forget10 deleted
(400 orphans, 180 survivors). `top1`/`top3` = share of orphans on the busiest one / three survivors;
`n_eff` = 1/HHI; `adequacy` = masked/unmasked top-1 similarity ratio (≈1 ⇒ the wrong sibling matches
as well as the deleted expert did); AUC = the router's own best confidence detector separating
orphans (low = undetectable); tomb = author-rung tombstone catch/retain-FPR.

| Router | drop | top1 | top3 | n_eff | busiest | adequacy | retain shift | self-detect AUC | tomb-author |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| `key_exact` | d199 | 1.000 | 1.000 | 1.0 | s0 (fallback) | — | 0.0000 | no-match op: orphan 1.000 / retain 0.148 | — |
| `key_exact` | d180–199 | 1.000 | 1.000 | 1.0 | s0 (fallback) | — | 0.0000 | no-match op: orphan 1.000 / retain 0.147 | — |
| `key_tfidf` | d199 | 0.400 | 0.750 | 4.0 | s88 (8/20) | **0.194** | 0.0000 | **0.999** (global_top1) | 0.950 / 0.000 |
| `key_tfidf` | d180–199 | 0.190 | 0.297 | 17.5 | s88 (76/400) | 0.270 | 0.0003 | **0.989** (per_shard_z) | 0.960 / 0.000 |
| `centroid_sbert` | d199 | 0.400 | 0.900 | 3.3 | s88 (8/20) | 0.705 | 0.0000 | **0.982** (per_shard_z) | 1.000 / 0.000 |
| `centroid_sbert` | d180–199 | 0.110 | 0.258 | 24.2 | s88 (44/400) | 0.673 | 0.0008 | **0.984** (per_shard_z) | 0.960 / 0.004 |
| `centroid_lm` | d199 | **0.700** | 0.900 | **1.9** | s128 (14/20) | **0.976** | 0.0000 | 0.728 (per_shard_z) | 0.950 / **0.338** |
| `centroid_lm` | d180–199 | 0.170 | 0.310 | 17.4 | s88 (68/400) | 0.962 | 0.0042 | 0.761 (margin) | 0.950 / **0.720** |
| `q2author` (oracle) | both | — | — | — | → base | — | 0.0000 | — | — (nothing to catch) |

`full_top1_acc` (routing accuracy before any deletion): key_tfidf 0.974, centroid_sbert 0.964,
key_exact 0.859, centroid_lm 0.719. AUC and tomb columns are recomputed from the `.npz` score
sidecars by `analyze_router_family.py`; the rest read directly off the JSON.

**Reading H′.1 — the 7B result is the 1B result, with one twist.** At **single-author** deletion the
familiar split holds: `centroid_lm` concentrates (n_eff 1.9) with near-perfect adequacy (0.976) and
is confidence-undetectable (AUC 0.728) — it leaks, and it *also* breaks the author tombstone
(retain-FPR 0.338, rising to 0.720 at mass deletion, because its dense own-space sentinels match
retain queries too). Meanwhile `key_tfidf` and `centroid_sbert` self-detect almost perfectly
(AUC 0.982–0.999 at 0.0–0.038 FPR) because at one-author granularity a wrong match is a *bad* match
(adequacy 0.194 / 0.705). The twist is the **drop-count dilution**: deleting 20 authors instead of 1
spreads every router (n_eff 1.9→17.4, 3.3→24.2, 4.0→17.5) — mass deletion has no magnet, because
no single survivor can absorb 400 orphans. Concentration is a property of *how much* you delete, not
only of the router.

### H′.2 — no 7B run exists (6 of 12 rows)

Stated so the gap is legible rather than inferred from an absent column:

| Method | Why not, and what it would cost |
|---|---|
| **routing_scaffold** (k=10, mu 0.7509) | No scaffolded 7B base exists (`make_scaffolded_base.py` was only run for 1B). Needs a 7B scaffold LoRA + baked base + 10 re-trained experts. |
| **sift_masks** (0.737) | 200-task **fp32 full-FT** at 7B. Already recorded as not-run in this report's coverage note; the dominant cost in the whole table. |
| **clamu** (0.647) | Same shape as SIFT — 200-task full-FT at 7B, plus per-cluster STE mask optimization. |
| **memsinks** (0.6417) | Out-of-tree (`memsinks_tofu`), 1B only. Its Group-B serve run was never executed at *any* scale, so both its mu and its orphan verdict are 1B-and-predicted. |
| **memory_adapters** (0.869 `[OU]`) | Out-of-tree, 1B only, and on the OU chat-template scale — a 7B run would still not be mu-comparable. |
| **sepmlp** (≈0.795 recall) | Out-of-tree, 1B only, and a recall metric rather than `model_utility`. |

### H′.3 — the 7B routed-mu ladder, complete (filled 2026-07-26, job 448587)

`routed_key_exact` now exists at every granularity of the 7B pool, so the merge-vs-select contrast
can be read at six points **on one model** against Table C's merge ladder:

| k | 4 | 10 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| **routed** `key_exact` | **0.7204** | 0.6907 | 0.6940 | **0.7147** | 0.6475 | 0.4728 (r8) |
| **merged** `dare_ties` (Table C) | 0.545 | 0.477 | 0.450 | 0.438 | 0.430 | 0.420 (r8) |
| gap | +0.175 | +0.214 | +0.245 | +0.277 | +0.218 | +0.053 (r8) |

**Reading.** The merge row decays monotonically with k — the dilution law, base + 1/k attenuation.
The routed row does not: it sits at 0.65–0.72 across k=4…100 with no trend, because routing serves
one expert at full strength regardless of how many exist. The gap therefore *widens* with
granularity (+0.175 → +0.277) until the k=200 cell, where both collapse for a **capacity** reason
rather than a routing one (r8 adapters — r32×200 does not fit a 46 GiB A40). Dilution is a property
of merging, not of sharding.

A **7B k=10** battery across all 9 `router.py` strategies, a **7B k=50** feature battery, and a
plain-**1B** k=10 de-confound arm also landed — see
[`log/router_leak/2026-07-26_7b-orphan-coverage.md`](../../log/router_leak/2026-07-26_7b-orphan-coverage.md).
Headline: the routers that never read the model are **bit-identical** across 1B-scaffolded /
1B-plain / 7B-plain, every expert-reading router **moves its magnet with scale**, and `centroid_lm`
— the worst leaker (adequacy 0.976–0.999, self-detect AUC 0.728) — keeps magnet **s4 in all three
arms**, so the leak is structural rather than a property of one checkpoint.

## Table I — Serve-time selection is the carrier (same weights, ± a select step)

The cleanest control in the project: hold the summed task vectors fixed and toggle the serve-time
selector on/off. **Models:** rows 1–3 = `meta-llama/Llama-3.2-1B-Instruct` (P5 full-FT, P1 k=10); row
4 = `meta-llama/Llama-2-7B-chat-hf` (P3 k=200 e25).

| Underlying weights | merge / no-select mu | + serve-time select mu | selector added |
|---|---:|---:|---|
| SIFT spine (1B full-FT, T=200) | `merge_full` 0.407 | `sift_full` **0.737** | per-task bitmask |
| ClAMU spine (1B full-FT, T=200) | Global 0.351 | opt-mask **0.647** (K16) / **0.672** (K200) | per-cluster STE mask |
| LoRA experts (1B, k=10) | `additive_mean` 0.419 | routed+scaffold **0.556** | author-key route |
| LoRA experts (7B, k=200) | `additive_mean` 0.460 | oracle route **0.824** (e25) | author-key route |

**Reading:** the utility swing (+0.30 to +0.36) is entirely attributable to the selector, on
*identical* weights. The merge never had the utility; the mask/router does.

## Table J — Base & fine-tuned anchors, by model (for cross-pool reading)

| Model | Base mu | k=1 LoRA-FT | locuslab full-FT | Matched-capacity FT | Joint-FT |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | ≈0.42 (0.38–0.44) | 0.7435 | ≈0.748 | 0.6372 | 0.5302 |
| Llama-2-7B-chat-hf | 0.418–0.426 | — | ≈0.62 | — | ≈0.756 |
| Llama-3.2-3B-Instruct | 0.38–0.44 | — | — | — | — |
| TinyLlama-1.1B | 0.38–0.44 | — | — | — | — |
| phi-2 | 0.38–0.44 | — | — | — | — |

The natural fine-tuned anchor for a serving claim is matched-capacity / joint-FT, not the k=1 /
locuslab ceiling. (Full base-anchor and forget10-harness columns: ROUTING_MASTER Table 4.)

## Routing caveats — leak & exactness (summary; full battery in ROUTING_MASTER)

Routing wins on utility *and* exactness, but "which router" matters for privacy. From
[ROUTING_MASTER_2026-07-23.md](ROUTING_MASTER_2026-07-23.md) Tables 2 / 2b / 2c / 3 and Waves 0–5:

- **Oracle / exact key routes are clean.** `q2author` and calibrated key routes: **0 orphan leak**,
  and the composed post-deletion system is **MIA-invisible** (AUC 0.072–0.284 < the oracle floor
  0.379 — the leak is *below* chance, mechanistically clean).
- **Dense / behavioral *embed* routers leak inseparably.** 6/9 `router.py` strategies (centroid_*,
  activation_norm, attn_norm, logit_div) misroute orphans onto a magnet sibling with adequacy
  0.95–1.000 and self-detect AUC ≤ 0.63. The leak is a property of the **scoring signal**:
  content-seeking signals (`ppl`, centroids) leak the planted fact (Mode-B ρ@R8 0.81–0.94);
  outlier-seeking `logit_div` sits at the floor (ρ 0.016) — integrity failure only, no disclosure.
- **`fq` and MIA are both blind** to the router channel; only content / ρ probes see it. Two
  independent leak detectors miss it — routing disclosure needs a purpose-built probe.
- **Identity tombstones price out.** The calibrated author-rung sentinel catches 360/400 orphans at
  7/3,600 (0.2%) retain-FP and seals the Mode-B channel (ρ 0.833 → 0.047) — but catch and
  disclosure rank-correlate perfectly (author rung disclosure AUC 0.987), so **there is no free
  seal**; the privacy-cleanest *storage* (name rung) is nearly as *loud* (0.967).

**Takeaway:** the routing headroom in Tables G–I is real and comes with exact O(1) deletion, but a
production router must use an **identity/key** route (not an embedding similarity route) to keep the
deletion clean — the leak lives in the *router's scoring signal*, not in routing per se.

---

## One-line synthesis

Across 2 model scales, 3 vector types, 5 PEFT parameterizations, and ~15 algorithms, **every
separable merge sits within noise of one ceiling — mu ≈ 0.41–0.48 ≈ base + dilution.** Everything
above it (SIFT mask 0.737, ClAMU 0.672, routing 0.556–0.824) buys its utility with **serve-time
selection**, or breaks separability (LoraHub 0.592 at k=4). Merging does not work; selecting does —
and selecting is also what makes deletion an exact, cheap, O(1) drop.

---

### Coverage note (what is on disk vs pending)

Cells above are on-disk smoke results. **Update (2026-07-26):** the **7B k=200** merge battery has
**landed** (Table C′; on disk + write-up
[`../../merge-tables-7b/RESULTS_7B_K200.md`](../../merge-tables-7b/RESULTS_7B_K200.md)) — ~20
operators, 10/11 registry methods, all in the 0.42–0.46 band. It was **independently reproduced on a
CISPA A100 box** (the `MERGE_METHODS_7B_K200_2026-07-25` run) — every mu matched the on-disk A40
values to within ~0.002 (that run *retrained* the pool, so its ft anchor 0.742 and fq column come
from a different oracle and are not used here). **Only KnOTS could not complete** (shared-SVD over 200
adapters OOMs a 44.5 GiB A40 — hardware) and JD's build hit an SVD-convergence error. Still genuinely
**not run on 7B**: full-FT **SIFT/ClAMU** (Table 6) and **ctv-`ds`** (need fp32 7B full-FT, ~2
GPU-days). A **7B PEFT bake-off** exists only on the A100 box (its `checkpoints_7b/` is not on this
machine, unverified here) — so it is **not** folded into Table E. No cell here is estimated or
extrapolated.

*Sources: per-run JSONs under `checkpoints/<pool>/results/{,smoke}/` (ver `ou-2026-06-10`, seed 42);
[MERGE_METHODS_RESULTS_2026-07-21.md](MERGE_METHODS_RESULTS_2026-07-21.md),
[ROUTING_MASTER_2026-07-23.md](ROUTING_MASTER_2026-07-23.md), `../../merge-tables-7b/RESULTS_TABLES.md`,
[`../../merge-tables-7b/RESULTS_7B_K200.md`](../../merge-tables-7b/RESULTS_7B_K200.md) (7B k=200);
ledger `log/{merge_mechanism,sift_masks,clamu,composable_tv,routing_scaffold,peft_compose,router_leak}/`.
Assembled + spot-verified 2026-07-24; 7B k=200 folded in + re-verified 2026-07-26.*

---

# Appendix A — Method & term dictionary

Every method and term named in the tables, explained from scratch. Bracketed keys (e.g. `dare_ties`)
are the registry identifiers in `merge_lora.py` / `router.py`. Full citations in Appendix B; the
plain-language merge specs also live verbatim in
[TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md](TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md).

## A.1 Weight-space merge operators (Part I)

All operate on task vectors `τ_t` and produce one served model `θ₀ + f(τ₁…τ_T)`. "Exact?" = does
deleting an author reduce to subtracting a per-author term (separable), or does `f` fold in
cross-task statistics.

- **Naive sum, λ=1** [`additive_s1`] — literally `θ₀ + Σ τ_t`. Exact (separable). Collapses because
  the summed-delta norm grows with the number of authors and pushes activations off-distribution.
- **Uniform mean / model soup, λ=1/k** [`additive_mean`] — `θ₀ + (1/k)Σ τ_t`; averaging the
  fine-tuned models (Wortsman et al. 2022, *Model Soups*). Every author's facts are attenuated by
  1/k — the "dilution ceiling." Exact by recomputation (the subtraction form needs a re-normalize).
- **Tuned-λ task arithmetic** [`additive_s{λ}`] — one global scalar λ, swept on held-out data, scales
  the summed vector (Ilharco et al. 2023). A *fixed* λ stays separable/exact; a *tuned* λ is a
  cross-task statistic (not exact). For memorization there is usually no good λ: small λ washes out
  facts, large λ explodes.
- **TIES** [`ties`] — **T**rim (keep each τ's top-magnitude entries), elect **S**ign (per-coordinate
  majority vote), then average only the sign-agreeing entries (Yadav et al. 2023). Not exact (the
  sign vote couples all authors).
- **DARE** [`dare_linear`] — **D**rop **A**nd **RE**scale: randomly zero a fraction *p* of each τ's
  entries and rescale survivors by 1/(1−p) (Yu et al. 2024, *Super Mario*). Reduces overlap
  probabilistically; DARE+sum with a stored seed is separable (exact w.r.t. the DARE'd deltas).
- **DARE-TIES** [`dare_ties`] — DARE sparsification then TIES merge; the project's **frozen default**
  merge (the only method with non-trivial unlearning behavior at small k). Not exact.
- **DELLA(-TIES)** [`della_linear` / `della_ties`] — like DARE but the drop probability is
  magnitude-*aware* (important weights survive more often) before a linear or TIES merge (Deep et al.
  2024, *DELLA-Merging*). Not exact.
- **Fisher-weighted** [`fisher`] — per-parameter weighted average where each author weights a
  parameter by its diagonal Fisher information (how much that author's loss depends on it; Matena &
  Raffel 2022). Not exact (the denominator couples authors); needs a dataloader.
- **RegMean** [`regmean`] — per linear layer, the closed-form least-squares weight that best matches
  every author model's *outputs* on its own inputs, using input Gram matrices (Jin et al. 2023). Not
  exact (the `(Σ Gₜ)⁻¹` couples authors); needs a dataloader.
- **Model Breadcrumbs** [`breadcrumbs`] — per author, drop *both* the smallest-magnitude entries
  (noise) and the largest (outliers), keep the mid-band, then sum with a scale λ (Davari &
  Belilovsky 2024). Separable at fixed λ. The historical "collapse" was pure √r-inflation, not the
  mask — see the scale-convention note below.
- **KnOTS** [`knots_ties`] — rotate every LoRA delta into one **shared SVD basis** so their
  directions align, then TIES-merge in that basis (Stoica et al. 2025). Directly targets subspace
  misalignment; not exact (the shared basis is fit on all authors).
- **TSV-M** [`tsv`] — **T**ask **S**ingular **V**ectors: merge via each task's top singular vectors
  with a whitening step to decorrelate them (Gargiulo et al. 2024). Not exact.
- **SLERP** [`slerp`] — **S**pherical **L**inear int**ERP**olation: rotate between two weight vectors
  along the hypersphere instead of averaging (Shoemake 1985; popularized for merging by mergekit,
  Goddard et al. 2024). Pairwise only (tree nodes or k=2). Not exact; collapses for independent
  experts.
- **LoraHub** [`lorahub`] — a gradient-free optimizer (nevergrad/CMA-ES) *learns* one combination
  weight per adapter by minimizing loss on sample data (Huang et al. 2024). Highest k=4 utility, but
  the learned weights are cross-task statistics ⇒ **not exact**; needs a dataloader.
- **JD / Compress-then-Serve** [`jd_full` / `jd_diag`] — **J**oint **D**iagonalization: compress the
  whole collection into a shared basis pair (U,V) plus a tiny per-adapter core Σᵢ, then serve the sum
  of cores; deleting an adapter = dropping its Σᵢ (O(1)) (Brüel Gabrielsson et al. 2025). Exact O(1)
  *drop*, but **approximate** because the shared basis was fit on everyone. `jd_full` = full core,
  `jd_diag` = diagonal core.
- **Subtract-orth** [`subtract_orth`] — a *deletion operator*: average all shards, then project the
  forget shard's subspace directions *out* of the result. Highest-utility subtraction variant here.
- **Task-arithmetic subtraction** [`subtract_linear`] — the classic "negate the task vector"
  deletion: merged model minus the forget shard's τ (Ilharco et al. 2023). Collapses on this pool.
- **Centered merging** [`centered_pool` / `centered_lowrank` a.k.a. `cr16`] (Table D) — before
  summing, estimate the *shared* component every author's delta has in common (the pool mean, or a
  rank-ρ SVD of it) and subtract it from each delta, so only each author's idiosyncratic part is
  merged: `M = ΣΔᵢ − (N−1)·S`. `cpool` uses the full pool mean, `cr16` a rank-16 estimate. The only
  rule that moved the per-author interference knee (N≈3 → N\*≈64), but it ends *below base* at N=200.
  Repo research construction (merge_mechanism Exp-6; the literal §6.1 formula with S = exact subset
  mean is algebraically ≡ `additive_mean`, so a low-rank/pool estimator is used instead).
- **Remaining `MERGE_METHODS` registry entries** (implemented in `merge_lora.py`/`merge_extra.py`,
  not headlined above): **`cat`** (concatenate LoRA factors = exact full-rank sum, the scaffold under
  `additive`), **`magnitude_prune`** (keep only top-magnitude delta entries then average),
  **`ties_svd` / `dare_ties_svd`** (PEFT's *merge-then-compress* SVD — distinct from KnOTS's
  shared-basis SVD), **`della_linear`** (DELLA magnitude-prune + linear sum), **`weighted_avg_ab`** /
  **`weighted_ba`** (predate the scale-convention analysis, ≈0.5× true scale), and the **DARE+sum**
  variants [`dare0p9sum`/`dare0p99sum`] (DARE mask composed at weight 1.0 — separable, exact w.r.t.
  the DARE'd deltas). Standalone k=200 evals for RegMean/Fisher/plain-TIES/Breadcrumbs **have
  landed** (on disk, [`../../merge-tables-7b/RESULTS_7B_K200.md`](../../merge-tables-7b/RESULTS_7B_K200.md);
  Table C′) — all in-band; only **KnOTS** could not complete at k=200 (shared-SVD over 200 adapters
  OOMs a 44.5 GiB A40 — hardware, not a result) and JD's build hit an SVD-convergence error.

**Scale-convention footnote (why some rows read 0.00).** The project's shards train with **rsLoRA**
(Kalajdzievski 2023: scaling `α/√r` instead of `α/r`), so PEFT's stock factor-space merges
(`linear`/`ties`/`dare_*`/`della`/`breadcrumbs`) apply `√(w·scaling)` and *double-count* the `√r`
factor — effective deltas inflated ≈2.8× → degenerate (mu≈0). True-scale methods (`additive`, `knots`,
`tsv`, `slerp`, `fisher`, `lorahub`, `jd`, `subtract_orth`) divide that out. This is an established
baseline convention, not a bug to "fix" in isolation — compare *within* a convention or sweep λ. The
`additive`/`_s{λ}` family is the corrected `linear`/`cat`. **Rank/k-dependence:** the inflation scales
with `√r`, so at **r8** (the k=200 pool) it is small enough that `linear` lands *in-band* (0.450,
Table C′) rather than degenerate — the 0.00 rows are an artifact of **r32**, not of the operator.

## A.2 PEFT parameterizations (Table D / pool P4)

Different ways to store the per-author adapter; the point of P4 is that the merge ceiling is
independent of which one you use.

- **LoRA** — low-rank `ΔW = (α/r)BA` (Hu et al. 2021).
- **DoRA** — Weight-Decomposed LoRA: splits each weight into magnitude + direction and LoRA-adapts
  the direction (Liu et al. 2024).
- **IA³** — learns per-neuron multiplicative *gates* (rescale keys/values/FFN activations) instead of
  additive deltas — tiny (~1.5 MB; Liu et al. 2022, *T-Few*). Two compose rules in Table E: **gate
  arith-mean** (average the gates) and **gate geo-mean** (signed geometric mean, arith fallback where
  signs disagree) — both O(1) exact, both land at 0.430.
- **VeRA** — Vector-based Random Matrix Adaptation: a single *shared frozen random* basis pair, with
  only tiny per-adapter scaling vectors trained (Kopiczko et al. 2024) — so all adapters live in one
  basis and average cleanly.
- **Prefix-tuning** — prepend trained key/value **prefix** vectors to the attention cache (Li & Liang
  2021); "merge" = concatenate prefixes. Independently-trained prefixes are mutually
  out-of-distribution → collapses (mu 0.002).

## A.3 Full-parameter task-vector methods (Table F / pool P5)

The non-LoRA track: sum full-parameter task vectors, ± a serve-time mask. All exactly deletable by
re-derive-and-subtract.

- **FT+Merge** [`merge_full`] — sum the full-FT task vectors, serve with **no mask**. Collapses to
  ≈base — the maskless-merge baseline.
- **SIFT-Masks** [`sift_full`] — **S**ign-Fixed Tuning: draw one global ±1 sign vector before
  training and constrain every author's full-FT to move weights only in that agreed direction, so the
  sum has no sign conflicts; each author also stores a bitmask of which parameters it touched, and at
  serve time re-applies that mask to carve its slice out of the merged sum (Kuo et al. 2025). The mask
  is a per-task router in disguise — it recovers full utility (0.737). Unlearn = subtract the
  re-derived τ, bitwise-exact.
- **ClAMU** [`clamu_full`, `emr_*`, `tall_*`] — SIFT's sibling: same full-FT sum but *no* sign
  constraint; instead cluster authors (k-means on answer embeddings, K clusters) and **directly
  optimize** one serve-time mask per cluster by gradient descent + straight-through estimator (Kuo et
  al. 2025, earlier ICLR version). The ladder compares mask sophistication on the *same* sum: **Global**
  (no mask) → **EMR** (keep where the cluster's own delta is large) → **TALL** (threshold vs the
  merged sum) → **optimized ClAMU mask**. Deletion raises utility; exact by subtract + re-cluster.

## A.4 Serve-time selection / routing (Part II)

- **The routing idea.** Keep experts separate; a **router** picks which expert(s) to apply per query.
  Deletion = drop the module (O(1), exact). **Scaffold** = a shared LoRA trained on public data
  (Alpaca) for general competence, never deleted. **OOD-aware** = author queries → expert; out-of-domain
  queries (real-authors / world-facts) → base+scaffold only, so an author expert never corrupts a
  general-knowledge answer.
- **Router strategies** (`router.py`), scored per query:
  - **`q2author` (oracle)** / **`key_exact`** — exact identity lookup / author-name substring match
    (lexical); the clean, 0-leak route.
  - **`key_tfidf`** — TF-IDF cosine of the query vs each expert's training questions.
  - **`centroid_sbert` / `centroid_lm` / `centroid_lm_last`** — cosine to each expert's mean embedding
    (MiniLM sentence encoder over answers / base-LM mean-pool / base-LM last-token hidden state).
    Encoder-swap variants (`centroid_sbert_q` over questions, `centroid_mpnet`, `centroid_bge`,
    instructor-xl for legonet) probe whether the leak is encoder-specific — it is not (ROUTING_MASTER
    Table 2).
  - **`ppl`** — route to the expert under which the query has lowest perplexity.
  - **`activation_norm` / `attn_norm`** — route to the expert whose LoRA-B output (or attention
    modules) reacts most.
  - **`logit_div`** — route to the expert whose logits are most *atypical* vs the candidate-set mean.
  - **`RouterLoRA` (learned)** — a trained per-layer cross-attention gate (RAMoLE), the only *learned*
    router; leak-blind at near-chance AUC 0.556 (Table H caveats).
- **Full routed/adapter systems** (foils in Table H): **LegoNet** (Yu et al. 2022 — frozen base + k-means
  keyed adapter bank, top-k routing); **RAMoLE / LoraRetriever** (Zhao et al. 2024 — learned retriever
  + per-layer RouterLoRA over an uploadable adapter pool); **SEA** (Schneider et al. 2026 — static base
  + composable domain experts + a deletable per-user proxy); **Memory adapters** (Grimes et al. 2026 —
  product-key memory, per-document entries, unlearn = −∞ block-list; product-key memory from Lample et
  al. 2019); **MemSinks** (Ghosal et al. 2025 — route memorization into per-sequence sink slices,
  delete = zero the slice); **SepMLP** (router-free: per-author ReLU-gated bottleneck branches all
  summed — selection lives *inside* the weights); **S3T** (Basu Roy Chowdhury et al. 2025 — layer-disjoint
  sliced-and-staged LoRA + a prediction ensemble, delete = revert the affected slice).

## A.5 Training-time constructions (ctv thread, Table F′)

Can a per-author vector be trained so a *plain sum* stays high-utility and exactly deletable?

- **ctrl** — plain per-author LoRA, no constraint (the "can one author be memorized at all" anchor).
- **[lin] tangent-space / linearized** — fine-tune inside the model's first-order Taylor expansion
  around θ₀, where the model is *linear in parameters* so task vectors add without interference by
  construction (Ortiz-Jiménez et al. 2023). Memorizes perfectly, but serving through the linearized
  model zeroes general utility.
- **[wd] write-disjoint col(B)** — each author may only *write* into its own reserved orthogonal slice
  of the output space. Killed — the write-subspace constraint can't memorize.
- **[ds] disjoint-support full-FT** — each author gets a fixed random 0.5% of all parameters,
  pairwise-disjoint; full-FT may touch only that slice, so merging is a pure scatter (zero overlap)
  and deletion zeroes the slice, bit-exactly. Costs ~half the association strength before any merge.

## A.6 Deletion-audit & leak terms (Part II caveats)

- **Oracle / retain90 oracle** — a reference model trained only on the retained authors (0–179); the
  yardstick both `fq` (KS test) and MIA compare against.
- **MIA (membership-inference attack)** — given the served post-deletion model, try to tell forget
  authors (members) from a held-out set (non-members) by likelihood. Scorers: **loss**, **Min-K%**
  (Shi et al. 2024 — average of the K% least-likely tokens), **Min-K%++** (Zhang et al. 2024,
  normalized), **zlib** (loss ÷ compressed length). AUC near the oracle floor (~0.38) = the deletion
  is MIA-clean.
- **Mode-B replication / ρ** — plant one fact across R owners, delete one owner, and measure
  **residual-fact-recall** ρ = (post − floor)/(ceiling − floor): does the fact survive because other
  owners still hold it? Owner-deletion ≠ fact-erasure.
- **Orphan / leak / sibling** — an *orphan* is a deleted author's query after its expert is dropped;
  a *leak* is the router sending it to a surviving *sibling* expert instead of the base. **Adequacy**
  = how well the wrong sibling matches (≈1 ⇒ leak looks as good as the real answer). **Magnet** = a
  survivor that attracts a disproportionate share of orphans. **Tombstone** = an identity sentinel
  kept after deletion so top-1-on-deleted queries are caught and served the base instead of leaking.

---

# Appendix B — References

Unlearning-specific citations are grounded in the in-repo survey
[`../../papers/RELATED_WORK.md`](../../papers/RELATED_WORK.md) (PDFs in `../../papers/`); merge/PEFT
citations follow the standard literature.

**Benchmark, framework & foundations**
- Maini, Feng, Schwarzschild, Lipton, Kolter (2024). *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM 2024. arXiv:2401.06121.
- Bourtoule et al. (2021). *Machine Unlearning* (SISA). IEEE S&P 2021. arXiv:1912.03817.
- Wu, Pang, Liu, Wu (2025). *Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLMs.* NeurIPS 2025. arXiv:2505.24379.
- open-unlearning / locuslab — the metric port in `eval_tofu.py` reproduces it exactly (`test_ou_equivalence.py`).

**Task vectors & model merging**
- Ilharco et al. (2023). *Editing Models with Task Arithmetic.* ICLR 2023. arXiv:2212.04089.
- Wortsman et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-Tuned Models.* ICML 2022. arXiv:2203.05482.
- Yadav et al. (2023). *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS 2023. arXiv:2306.01708.
- Yu, Yu, Yu, Huang, Li (2024). *Language Models are Super Mario* (DARE). ICML 2024. arXiv:2311.03099.
- Deep et al. (2024). *DELLA-Merging: Reducing Interference via Magnitude-Based Sampling.* arXiv:2406.11617.
- Matena & Raffel (2022). *Merging Models with Fisher-Weighted Averaging.* NeurIPS 2022. arXiv:2111.09832.
- Jin et al. (2023). *Dataless Knowledge Fusion by Merging Weights of Language Models* (RegMean). ICLR 2023. arXiv:2212.09849.
- Davari & Belilovsky (2024). *Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks.* ECCV 2024. arXiv:2312.06795.
- Stoica et al. (2025). *KnOTS: Merging LoRA Models via SVD Alignment.* ICLR 2025. arXiv:2410.19735.
- Gargiulo et al. (2024). *Task Singular Vectors: Reducing Task Interference in Model Merging.* arXiv:2412.00081.
- Shoemake (1985). *Animating Rotation with Quaternion Curves* (SLERP). SIGGRAPH 1985. · Goddard et al. (2024). *Arcee's MergeKit.* arXiv:2403.13257.
- Huang et al. (2024). *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.* COLM 2024. arXiv:2307.13269.
- Brüel Gabrielsson et al. (2025). *Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead* (JD). ICML 2025. arXiv:2407.00066.
- Meng et al. (2023). *Mass-Editing Memory in a Transformer* (MEMIT). ICLR 2023. arXiv:2210.07229.

**PEFT parameterizations**
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Kalajdzievski (2023). *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (rsLoRA). arXiv:2312.03732.
- Liu et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. arXiv:2402.09353.
- Liu et al. (2022). *Few-Shot PEFT is Better and Cheaper than In-Context Learning* (IA³ / T-Few). NeurIPS 2022. arXiv:2205.05638.
- Kopiczko et al. (2024). *VeRA: Vector-based Random Matrix Adaptation.* ICLR 2024. arXiv:2310.11454.
- Li & Liang (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* ACL 2021. arXiv:2101.00190.

**Exact-unlearning merges & training-time localization**
- Kuo, Setlur, Srinivas, Raghunathan, Smith (2025). *Exact Unlearning of Finetuning Data via Model Merging at Scale* (SIFT-Masks; earlier ICLR version titled the method ClAMU). arXiv:2504.04626.
- Ortiz-Jiménez et al. (2023). *Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models.* NeurIPS 2023. arXiv:2305.12827.
- Ghosal, Maini, Raghunathan (2025). *Memorization Sinks: Isolating Memorization during LLM Training* (MemSinks). ICML 2025. arXiv:2507.09937.
- Cloud et al. (2024). *Gradient Routing: Masking Gradients to Localize Computation in Neural Networks.* arXiv:2410.04332.
- Maini et al. (2023). *Can Neural Network Memorization Be Localized?* ICML 2023. arXiv:2307.09542.

**Modular / routed architectures**
- Yu, Sun, Guo, Zhang, Cheng (2022). *LegoNet: A Fast and Exact Unlearning Architecture.* arXiv:2210.16023.
- Basu Roy Chowdhury et al. (2025). *Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning* (S3T). ICLR 2025. arXiv:2406.16257.
- Zhao et al. (2024). *Retrieval-Augmented Mixture of LoRA Experts for Uploadable Machine Learning* (RAMoLE / LoraRetriever). arXiv:2406.16989.
- Zhuang et al. (2024). *SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?* arXiv:2411.18797.
- Schneider, Schoenegger, Bariach (2026). *Separable Expert Architecture: Toward Privacy-Preserving LLM Personalization* (SEA). arXiv:2604.21571.
- Grimes, Kuo, Wu, Smith, Connor (2026). *Memory Adapters Enable Fast, Flexible Knowledge Unlearning in LLMs.* ICML 2026 Workshop.
- Lample et al. (2019). *Large Memory Layers with Product Keys.* NeurIPS 2019. arXiv:1907.05242.

**Membership-inference scorers (deletion audit)**
- Shi et al. (2024). *Detecting Pretraining Data from Large Language Models* (Min-K%). ICLR 2024. arXiv:2310.16789.
- Zhang et al. (2024). *Min-K%++: Improved Baseline for Pre-Training Data Detection from LLMs.* arXiv:2404.02936.

---

# Appendix C — Glossary (symbols, metrics, and terms)

Every symbol, metric, label, and abbreviation used above, in one place, for quick lookup. Fuller
method write-ups are in Appendix A; the metric formulas are in the Metric legend (§ top); pool
recipes are in §0.1.

### Symbols & notation

| Symbol | Meaning |
|---|---|
| θ₀ | frozen **base** (pretrained) model weights |
| θ_t | weights after fine-tuning θ₀ on author/shard *t* alone |
| **τ** (τ_t) | **task vector** = θ_t − θ₀ — what one author's training added; delete = subtract it |
| ΔW | the dense weight change a LoRA adapter represents = (α/r)·B·A |
| B, A | the two low-rank **LoRA factors** (ΔW = B·A) |
| r | LoRA **rank** (`r8` = 8, `r32` = 32); more rank = more capacity + more √r inflation |
| α | LoRA **alpha** (scale numerator; effective scale = α/r, or **α/√r** under rsLoRA) |
| k | number of **shards** (`k=10` → 20 authors/shard; `k=200` → 1 author/shard) |
| N | number of per-author adapters **merged at once** (the N-ladder axis, Table D) |
| T | number of tasks/authors in a **full-FT** pool (`T=200`) |
| e | training **epochs** (`e5` = 5, `e25` = 25 — the well-trained per-author experts) |
| λ (lambda) | global scaling coefficient on a summed task vector (label suffix `_s{λ}`) |
| √r | the rsLoRA scale factor that **inflates** naive PEFT merges → the `mu ≈ 0.00` rows |
| ρ (rho) | **residual-fact-recall** ratio (Mode-B leak) = (post − floor)/(ceiling − floor) |
| Δbase / ΔFT | mu gap from a row to its pool's **base** / **fine-tuned** anchor |
| → · ≈ · ± | "becomes / decays to" · "approximately" · "plus-or-minus" |

### Metrics (formulas in the Metric legend, § top)

| Term | Meaning · good direction |
|---|---|
| **mu** (`model_utility`) | harmonic mean of 9 TOFU components — "is the model still good"; **higher** |
| **fq** (`forget_quality`, `ks_pval`) | KS p-value vs the retain-only oracle; **higher** (post-deletion) = better forgetting |
| **f_rouge** (`forget_rouge`) | ROUGE-L recall on the forget split — verbatim recall (context-dependent) |
| **r_ppl** (`retain_ppl`) | retain-text perplexity; **lower** (≫20 ⇒ the merge broke the LM) |
| own-prob / own-rouge | per-author memorization, restricted to one author's **own** rows |
| iso / solo mu | single-adapter utility, **no merge** (the isolated ceiling, ≈0.46 at 7B) |
| tr (truth ratio) | P(wrong)/P(correct) per sample — feeds the truth-ratio components of mu/fq |
| AUC | area under ROC — how well an MIA / leak detector separates members from non-members |

### Pools, data & hardware labels

| Term | Meaning |
|---|---|
| P1–P5 | the five experiment **pools** (models/recipes in §0.1; anchors in the Anchor legend) |
| base / FT | never-fine-tuned **floor** / joint fine-tune **ceiling** — the two per-pool anchors |
| `ft_all`, `ft_r32` | the joint-FT anchor adapters (one model trained on all authors) |
| retain90 (oracle) | reference model trained **only** on retained authors 0–179 (the fq / MIA yardstick) |
| forget10 | the forget split = authors 180–199 (= `shard 9` at k=10) |
| smoke / extended | eval **cap tiers** (smoke: ROUGE≤50 / retain≤80 / truth≤30; extended larger) |
| 1B / 7B | `meta-llama/Llama-3.2-1B-Instruct` / `meta-llama/Llama-2-7B-chat-hf` |
| A40 / A100 | GPU boxes — **this machine** (A40) vs the CISPA recompute (A100); see Table C′ |
| band / ceiling / dilution | the **0.42–0.48 plateau** every separable merge lands in (≈ base + 1/k) |

### Methods & mechanisms (details: Appendix A)

| Term | Meaning |
|---|---|
| merge / merge operator | fold all τ into one served model θ₀ + f(τ₁…τ_T) — cheap to serve (Part I) |
| **separable** / exact | deletion = subtract one τ, **no cross-task statistics** ⇒ exact O(1) unlearn |
| not-exact | the operator folds in sign votes / learned weights / Gram or pool means |
| rsLoRA | rank-stabilized LoRA (scale **α/√r**) — the shards' training default; source of √r inflation |
| LoRA / DoRA / IA³ / VeRA / prefix | the five PEFT parameterizations compared in pool P4 |
| full-FT | **full-parameter** fine-tune (no low-rank factoring; pool P5) |
| SISA | Sharded-Isolated-Sliced-Aggregated — the exact-unlearning ancestry (Bourtoule 2021) |
| SIFT-Masks / ClAMU | full-FT **sum + a serve-time mask** (sign-derived / cluster-optimized) |
| STE | straight-through estimator — how ClAMU trains its discrete masks |
| KS test | Kolmogorov–Smirnov two-sample test — the `fq` statistic |
| scaffold | a shared public-data (Alpaca) LoRA baked into θ₀, **never deleted** (general competence) |

### Routing, selection & leak (details: Appendix A.4/A.6, ROUTING_MASTER)

| Term | Meaning |
|---|---|
| routing / router | per-query **selection** of which expert(s) to apply (Part II) |
| serve-time selection / mask | the utility **carrier** — a router or per-task mask (vs merging into weights) |
| q2author / key_exact | oracle / lexical **identity route** — the clean, 0-leak router |
| OOD | out-of-distribution queries (real-authors / world-facts) → base+scaffold, not an expert |
| orphan | a deleted author's query **after** its expert is dropped |
| leak / sibling | the router sending an orphan to a **surviving** (sibling) expert instead of the base |
| magnet | a survivor that attracts a **disproportionate** share of orphans |
| adequacy | how well the wrong sibling matches (≈1 ⇒ the leak looks as good as the true answer) |
| tombstone | an **identity sentinel** kept post-deletion to catch orphans and serve base instead |
| MIA | membership-inference attack (loss / Min-K% / Min-K%++ / zlib scorers) |
| Mode-B / ρ | replicated-fact test: plant a fact across R owners, delete one, measure residual recall ρ |
