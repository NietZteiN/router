# MUSR paper — full evidence report: every logged experiment mapped to the Experiments section

**Date:** 2026-07-22 · **Purpose:** consolidate everything in the research logs that bears on
the MUSR (Modular Unlearning via Self-Routing) AAAI submission — the MUSR method's own runs, the
cited baselines, the broader unlearning-method landscape in the repo, the paper §-by-§ mapping,
the contradictions to fix, and what remains to run. Companion to the shorter inventory
[`EXPERIMENTS_FOR_PAPER_2026-07-22.md`](EXPERIMENTS_FOR_PAPER_2026-07-22.md).

### Confidence convention
- **[V]** = number read **firsthand** from the cited log entry during this audit.
- **[S]** = number taken from a log-map / thread-README summary (traceable via the cited path,
  not re-opened line-by-line here).

Unless noted: single seed **42**, **Llama-3.2-1B-Instruct**, MUSR spec-v2 recipe
(`L1 + 10·hinge + 50·Gram + 1·promotion`), width-32 adapters on all 16 MLP layers.

---

## 0. Executive summary

**Identity.** MUSR ⟺ the `sepmlp` thread (`log/sepmlp/`, code `sepmlp_tofu/`). The paper is **our
own draft** and is not checked into the repo; §5's numbers are marked "provisional." Conformance
against the paper's Eq 1–5 was verified in `../../.claude/plans/make-sure-our-emthod-wobbly-lemon.md`.

**Bottom line — three facts that govern how the logs can feed §5:**
1. **MUSR's own harvest is thin.** Backed by real `sepmlp` runs: selectivity/localization,
   deletion mechanics, parameter count. **Never run:** Table 4 ablations, the activation-rate
   mechanism experiment, Llama-3.2-3B, the layer sweep, K=10/50 + tuned config, continual
   addition, 50-sequential deletion, the relearning attack.
2. **Recall metric mismatch.** Every `sepmlp` "recall" to date is **answer-probability**, not the
   paper's per-source **ROUGE-L**. The paper-comparable recall (mean / tail / named / name-free /
   held-out) was **never computed**, so the headline 0.966 is neither confirmed nor refuted.
3. **Two logged numbers contradict the draft:** deletion retain-collateral **−0.11/−0.12** (draft
   says 0.002), and the answer-prob "≈0.80 K=200 ceiling" (flagged as a possible metric artifact).

**Rich, real, addable from the cited baselines:** the Memory-Adapters block-list column
(Agg 0.869 = paper's 0.87), the composed-model MIA (approximate 0.74–0.82 vs exact ≤0.38 = oracle
0.379), and the router-leak campaign (fills the empty §4: router serves deleted data ρ 0.833;
static exact edit stays at floor under misrouting).

---

## 1. The MUSR / `sepmlp` experiments in full

All Llama-3.2-1B-Instruct, seed 42, width-32 × 16 layers, spec-v2. No other base model, width,
layer subset, or term configuration was ever trained.

### 1.1 Localization / selectivity (the make-or-break, H1) — SUPPORTED at K=20
- **[V] P1 smoke** (job 446535, `smoke.json` sha `7f94b32e5deececc`, 2 authors × 5 steps): grad
  isolation OK ×16 layers; own_norm **0.0591** vs off_norm **0.0040**; ood_over_own 0.077; peak
  14.04 GiB @bs2; save→reload parity PASS. `2026-07-21_pilot-oom-and-adjudicate.md`.
- **[V] K=20 lr-sweep pilot** (job 446714, bs16×ga2, 15 ep) — median on/off selectivity / own-prob
  (all-active) / min own-prob / all-active−own-only gap:
  | lr | selectivity | own-prob | min | gap | OOD ood/own (alpaca) |
  |---|---|---|---|---|---|
  | 3e-4 | 4.38 | 0.981 | 0.957 | +0.824 | 0.027 |
  | 1e-3 | 38.61 | 0.778 | 0.530 | +0.179 | 0.004 |
  | 3e-3 | 1909.7 | 0.695 | 0.336 | +0.0005 | 0.0002 |
  vs the LoRA negative-anchor selectivity ceiling **1.11**. `2026-07-21_pilot-oom-and-adjudicate.md`.
- **[V] G2 bridging arm, K=20 lr 5e-4** (job 446732, `pilot_relu_lr5e-4.json` sha `f4d6b9d59d62af6d`)
  — the H1 headline pilot: median selectivity **7.171** (frac≥5 = 0.85; per-author min/med/max
  2.32/7.44/10.36), median own-prob (all-active) **0.9765** (min **0.9363**, authors <0.8: **0**),
  gap +0.7304, OOD alpaca 0.0079 / real_authors 0.0249. **H1 CONFIRMED, G2 GO.**
  `2026-07-21_bridge-go-k200-launch.md`.

### 1.2 Selectivity scaling to K=200 (H-scale) — SUPPORTED
- **[V] K=200 selectivity** (train 446949 / probe 446950): median selectivity **507.5** (frac≥5 =
  1.00; ~70× amplification over the pilot); ood/own alpaca 0.0006 / real 0.0016 / world 0.0013.
  `2026-07-21_k200-g3-fail.md`.

### 1.3 The K=200 recall reality (answer-probability — NOT paper ROUGE-L)
Every K=200 arm, bs8×ga4, 15 ep, seed 42. **recall here = answer-probability**, the wrong metric
for the paper's recall tables (see §4).

| Arm | config (sha) | train/probe | median sel | median recall | ≥0.8 / 200 | gap (mean) | verdict |
|---|---|---|---|---|---|---|---|
| lr 5e-4 (P3) | `sepmlp_1b_k200.json` (`be58…`) | 446949/446950 | **507.5** | **0.637** | 20 | +0.130 | G3 FAIL |
| lr 2e-4 | `…lr2e-4.json` (`a493…`) | 447162/447163 | 36.0 | **0.747** | 60 | +0.450 | H-k200-lr REFUTED (by 0.003) |
| **lr 1.5e-4 (A2)** | `…lr1p5e-4.json` (`c7ca…`) | 447177/447178 | 16.33 | **0.795** | **97** | +0.520 | **GRAY — best K=200 point** |
| w1/5 (B) | `…w15.json` (`9071…`) | 447175/447176 | 24.59 | **0.696** | 28 | +0.621 | H-wscale REFUTED |

- **[V]** P3 percentiles: p10/25/50/75/90 = 0.453/0.555/0.637/0.744/0.805; min 0.286; ≥0.9 = 0;
  <0.5 = 31. `2026-07-21_k200-g3-fail.md`.
- **[V] Tradeoff curve:** all four points lie on ONE monotone selectivity↔recall curve topping
  **≈0.80** in the healthy-sel (5–30) band, vs the K=20 pilot's **0.977**. Both the lr dial (3.3×
  range) and the suppression-weight rescale (÷10) land on the same curve — the ceiling is
  **structural for the pinned recipe at K=200** (K∈{10,50,100} never run, so not proven as a
  smooth function of K). `2026-07-22_wscale-refuted-lr2-gray-mechanics.md`, `README.md`.
- **[V] Failed runs (not results):** job 446705 bs32 OOM at step 9/390; job 446910 bs16×ga2 OOM at
  step 6/3750 → bs8×ga4 fallback.

### 1.4 Deletion mechanics (block of 20 = forget10) — clean + cheap + MIA-safe, real collateral
**[V]** C eval array 447179 %2 on the **lr2e-4 (recall-0.747) checkpoint** — a mechanics probe on a
non-passing arm, not a replication row. `2026-07-22_wscale-refuted-lr2-gray-mechanics.md`.
- **Forget-set deletion (clean):** forget_Q_A_Prob 0.767→**0.054**; forget_ROUGE 0.760→0.316;
  extraction_strength 0.469→**0.047**; exact_memorization 0.935→0.490; mia_loss 0.997→**0.362**;
  mia_min_k 0.997→0.358; **privleak −99.56→+3.98** (deleted authors go to floor, MIA-indistinguishable).
- **Deletion cost (O(1)-style):** physical slice removal **1.07 s** (20 authors × 20 slices/layer);
  droplist build 7.2 s one-time (CPU). vs the MemAdapt block-list anchor 0.027 s — slower but
  deterministic and content-removing rather than a mask.
- **Retain collateral (H-gap):** retain_Q_A_Prob 0.676→0.563 (**−0.113**); retain_ROUGE 0.665→0.543
  (**−0.122**). Aggregate model_utility 0.465→0.468 (**+0.003**) — passes |ΔUtil.R| ≤ 0.03 but
  **masks** the fine-grained hit (see §5).
- **dropall ≈ base:** forget 0.092 ≈ retain 0.087 (symmetric floor); model_utility 0.281 ≈ base —
  behavioral confirmation that removing all banks returns the frozen base.

### 1.5 Parameter / cost accounting
- **[V]** K=200 checkpoint = 6.3×10⁸ params = **+0.63B (+50%)**, width n=32, 2.52 GB fp32 — matches
  the paper's Table 1 cost row. `2026-07-21_k200-g3-fail.md`, `../CLAUDE.md`.

### 1.6 Hypothesis ledger (from `log/sepmlp/README.md`) [V]
- **H1 localization** ✓ SUPPORTED (K=20: sel 7.17 / own-prob 0.977).
- **H-scale** ✓ SUPPORTED (K=200 sel 507.5, ~70×).
- **H-gap** ± ANSWERED — collective-recall gap causes real retain collateral (−0.113/−0.122),
  aggregate-masked (+0.003).
- **H-k200-lr** ✗ REFUTED (0.747 < 0.75 by 0.003); **H-wscale** ✗ REFUTED (0.696); **H-k200-lr2** ~
  GRAY (0.795).
- **H2** all-active utility — open (REFUTE direction never observed; OU utility pending P4).
- **H3 deletion** ± partial (clean/cheap/MIA-floor, but collateral; replication bars not evaluated).
- **H4 relearn parity** — open (pending P5, never run). **H5 negative-example leak** — open (Priv +
  raw AUCs, never run standalone). **H-K-sweep / H-width / H-layers** — open, unregistered.

---

## 2. Cited baselines (already in the paper's comparison set)

### 2.1 Memory Adapters (Grimes et al.) — Table 1 "MemAdapt (block-list)" column + §5.9 anchor [V]
Frozen 1B, one product-key memory layer (layer_idx 8, k=32), 200 authors × 256 disjoint slots, lr
1e-2 const, wd 0, eff. batch 32, seed 42; `configs/memadapt_tofu_1b.json`; jobs: pilot 443409, train
443441, calib 443526, evals 443530 (FT) / 443531 (unlearned). `2026-07-15_first-full-run-results.md`,
`../../memadapt_tofu/REPORT_2026-07-15.md`.

Row order **Util.R / Util.G / Mem / Priv / Agg** (ours vs paper):
| Condition | Ours | Paper |
|---|---|---|
| MemAdapt **unlearned (block-list)** | **1.066 / 1.013 / 0.630 / 0.917 / 0.869** | 1.00 / 1.06 / 0.62 / 0.98 / 0.87 |
| MemAdapt FT (pre-unlearn) | 1.075 / 1.024 / 0.027 / 0.380 / 0.097 | 0.93 / 1.05 / 0.18 / 0.39 / 0.40 |
| Retrained oracle | 1.009 / 1.121 / 0.590 / 1.000 / 0.874 | 1.00 / 1.11 / 0.58 / 1.00 / 0.87 |
| Finetuned (public ckpt) [S] | 1.000 / 1.143 / 0.088 / 0.381 / 0.252 | — |

- **Agg 0.869 reproduces paper 0.87** (|Δ|=0.001); Mem 0.630 ≈ 0.62 ✓.
- **Priv 0.917 < paper 0.98** — a routing-entropy / composition miss (an alternate `priv_absdiff`
  composition reads 0.953). Util.G 1.013 vs 1.06 also outside band.
- **Deletion cost:** unlearn 20 authors = 5,120 entries in **0.0269 s** (CPU), apply-at-load 0.021 s;
  training 652.9 s (paper 1106.9 s).
- **[S] H6 temperature ablation** (`2026-07-15_h6-temperature-ablation.md`): sharpening keys ×2/×4
  dropped effective reads 30.9→14.9 but Priv never rose (0.917→0.841→0.910) and Util.G collapsed
  (1.013→0.851) — leakage is **selection-level, not weight-level**.

### 2.2 Composed-model MIA (deletion audit) — §5.9 "approximate leaks, exact doesn't" [V]
`attack_mia.py`, member = forget10 (400 QA), non-member = holdout10 (400), cheap battery
(loss/min-k/min-k++/zlib), model Llama-3.2-1B, seed 42 (+43 determinism check), bs1; SLURM
440741/440761. Loss-AUC (1→member memorized; ≈0.5 indistinguishable). `2026-07-06_composed-mia-results.md`,
`../../tofu_sisa_lora/reports/DELETION_AUDIT_REPORT_2026-07-06.md`.

| Condition | full (control) | unlearn | role |
|---|---|---|---|
| oracle retain90 | — | **0.379** | floor (never saw forget10) |
| ft (retain-all) | — | 0.815 | leaky upper bound |
| GA | — | **0.743** | approximate (H1) |
| GD | — | **0.779** | approximate |
| KL | — | **0.753** | approximate |
| IDK | — | **0.816** | approximate |
| LegoNet (exact) | 0.812 | **0.369** | exact drop (H2) |
| routed-key (exact) | — | **0.375** | exact drop |
| ClAMU (exact) | 0.594 | **0.322** | exact drop |
| SIFT (exact) | 1.000 | **0.254** | exact drop |
| RAMoLE embed | — | **0.353** | H3 suspect — at floor |

- **Approximate leaks 0.74–0.82** (min-k up to 0.83) ≫ oracle floor 0.379, despite high clean
  forget_quality. **Exact module-drop is at/below the floor**; live `*_full` controls (sift_full
  1.000, legonet_full 0.812) prove the attack is not dead.
- **Caveat:** these are the **sibling** exact-drop methods, not `sepmlp`. MUSR's own MIA→floor
  (§1.4: 0.997→0.362) echoes the story, but a standalone `sepmlp` Priv/AUC battery was **not run**.
- RAMoLE-embed sits **at** the floor (0.353): the embedding-router leak costs forget_quality but not
  MIA membership — fq and MIA catch **orthogonal** channels.

### 2.3 Router-leak campaign — fills the empty §4 (Static vs Runtime) [V phase2/groupab, S phase1/sweep]
Llama-3.2-1B, seed 42. `log/router_leak/{2026-07-18_phase1-results,2026-07-18_phase2-results,
2026-07-22_groupab-depth-results}.md`, `../../tofu_sisa_lora/reports/ROUTER_LEAK_EXPLAINED_2026-07-21.md`.

**Runtime / router still holds & serves the deleted data (Group A = drop-and-survive):**
- **[V] Mode-B replicated-fact leak ρ:** sibling policy 0.107 / 0.439 / **0.833** at R=2/4/8 — the
  fact remains in surviving sibling experts and is re-served.
- **[S] Confidence-based defenses cannot seal it:** global-τ / per-expert-z / margin / kNN-density
  all at AUC **0.56–0.63**, FPR 0.82–0.88 @ 90% catch (phase 1). **[V]** corroborated at
  ClAMU-cluster granularity: confidence AUC **0.585** (groupab).
- **[S]** A realistic MiniLM embedding router sends **100%** of the 400 orphan (deleted-author)
  questions to survivors; best surviving match scores **sim-ratio 0.971** of the deleted centroid.

**Static identity seal (tombstone) — nothing to keep serving:**
- **[V]** per-author tombstone collapses Mode-B ρ **0.833 → 0.047** (~95% sealed), training-free,
  computable from the deletion request alone; per-R 0.031 / 0.000 / 0.047.
- **[V]** serving-level orphan catch **0.96**; retain cost ≈**0.006 mu**; deletion-disclosure AUC
  0.839 (the seal announces a deletion happened). **[S]** identity separability AUC ≈**0.98**
  (retain-FPR 0.002 @ 90% catch, phase 1).
- **[V]** content audit (n=400): sibling answers are **95.5% confabulation** (sibling_vs_gold 0.277 ≈
  base floor 0.249) — in the unplanted world the router leak is misinformation, not disclosure.

**Static exact edit needs no seal — "nothing to restore" (Group B = exact subtraction):**
- **[V]** under **100% forced misrouting**, the deleted gold's answer-probability stays at floor:
  SIFT oracle 0.118 / base 0.128 / **realistic 0.086**; ClAMU 0.128 / 0.128 / **0.124** (a served
  author scores ~0.9). A misrouted mask operates on a τ̄ the author's task vector is already
  subtracted from. ClAMU's +0.142 ROUGE bump is style-confabulation (confab 0.77, flat prob), not
  disclosure. `2026-07-22_groupab-depth-results.md`.

**The unifying claim for §4:** *it is the deletion mechanism, not the granularity, that decides
whether deleted content is recoverable.* Drop-and-survive (routed experts, per-author proxies) keeps
real content in surviving units → the router can re-serve it → a tombstone is load-bearing.
Exact-subtraction / static-edit methods remove the content from the weights → router-robust for free.

**[S] Router-family nuance (all-router sweep):** the leak is **not universal** (H-ARCH refuted) —
6/9 dense-similarity routers leak inseparably (confidence AUC 0.41–0.63), but ppl and key-tfidf
routers **self-detect** orphans (AUC 0.97–0.998), and a trained RouterLoRA is leak-blind
(0.588±0.002). The "router holds the data" claim is strongest for **dense-embedding** routing.
**The `sepmlp` routerless leak-probe row was never produced** — MUSR has no own row in this table yet.

---

## 3. Broader unlearning-method landscape in the repo (context / related-work / comparison) [S]

These are sibling method threads (not MUSR); useful as comparison points or related-work grounding.
Numbers are thread-README headlines (single-source; see each thread's `reports/` for full data).

| Thread / method | One-line | Headline number | Path |
|---|---|---|---|
| **memsinks** (SeqTD masked-LoRA-delta) | all-active per-author deltas self-interfere — the failure MUSR targets | all-slices-on mu **0.4373** vs ctrl 0.6438; dropall 0.6399; routed-mask 0.6417; per-author capacity floor gen_own **0.389** vs full 0.9991; deletion at ~80 KB/author | `../../log/memsinks/` |
| **routing_scaffold** (routed cousin) | routed per-author experts + oracle routing — routed upper bound | oracle e25 K=200 mu **0.8236** (repo best); deletion **Δmu 0.0000**; realistic router 0.7799 | `../../log/routing_scaffold/`, `../../tofu_sisa_lora/reports/K200_ORACLE_ROUTING_REPORT_2026-07-20.md` |
| **merge_mechanism** | why LoRA self-gating fails — motivates MUSR's disconnected MLPs | negative-anchor selectivity stuck **1.11** at every λ; facts collapse by N≈8 co-merged | `../../log/merge_mechanism/2026-07-16_negative-anchor-pilot-results.md` |
| **sea** (per-author LoRA proxy, delete = `rm`) | filesystem-level deletion, 4-bit Llama-2-7B | mu **0.711** unchanged post-delete; contamination ≈0; rank knee r4→r8 | `../../sea_tofu/reports/SEA_UNLEARNING_REPORT.md` |
| **legonet_lora** (k-means-keyed LoRA bank) | top-k routed, delete = retrain affected adapters | TOFU 1B mu **0.5011** / fq 0.890; bitwise-exact on CPU | `../../legonet_lora/reports/LEGONET_TOFU_REPORT.md` |
| **sift_masks** (sign-fixed FT task vectors) | merge = sum, unlearn = subtract | T=200 mask mu **0.737** vs merge collapse 0.407; GPU unlearn bitwise-exact | `../../tofu_sisa_lora/reports/` |
| **clamu** (SIFT + clustering + STE masks) | optimized per-cluster masks | ladder Global/EMR/TALL/ClAMU **0.351/0.388/0.405/0.647** | `../../tofu_sisa_lora/reports/CLAMU_REPORT_2026-07-02.md` |
| **s3t** (sliced-and-staged, ICLR'25 repro) | deletion re-trains only downstream slices | deletion rate <0.4%; S3T(B=4) ~1.59× more deletions than SISA | `../../tofu_sisa_lora/reports/S3T_PAPER_REPRO_2026-06-17.md` |
| **tofu_baselines** (GA/GD/KL/IDK) | canonical approximate-unlearning bar | all ks_pval **0.0** (fail KS); forget_ppl ≈ retain_ppl (indiscriminate) | `../../log/tofu_baselines/` |

Note: RMU / NPO / SimNPO / gradient-difference appear only as **deferred/pre-registered** baselines
(the paper's §5.9 relearning comparison set); none were run.

---

## 4. Paper section-by-section mapping

Legend: **BACKED** (real run) · **CONTRADICTS** · **NOT COMPARABLE** (wrong metric) · **NOT RUN**.

| Paper location | Status | Evidence / gap |
|---|---|---|
| **§4 Deletion Semantics (static vs runtime)** — currently empty | **BACKED** (baselines) | Router serves deleted data (ρ 0.833; confidence defenses 0.56–0.63); static exact edit at floor under misrouting (SIFT 0.086 / ClAMU 0.124); tombstone seal ρ→0.047. §2.3 |
| **§5.2 Table 1 — MUSR row: recall 0.966** | **NOT COMPARABLE** | Best local is 0.795 **answer-prob**, not ROUGE-L; passing arm never achieved (§1.3) |
| §5.2 utility 0.559 / forget ROUGE 0.832 / deleted 0.319 / **collateral 0.002** | **CONTRADICTS / no replication row** | C on a non-passing ckpt: forget ROUGE 0.760→0.316, deleted →0.054 (over-suppressed), **collateral −0.113** (§1.4) |
| §5.2 Table 1 — **+0.63B params** | **BACKED** | 6.3×10⁸ params, width 32 (§1.5) |
| §5.2 Table 1 — **MemAdapt (block-list) column** | **BACKED** (baseline) | Agg 0.869 / Priv 0.917 / unlearn 0.027 s (§2.1) |
| §5.3 Table 2 — K-ladder 0.975/0.962/0.966; tuned 0.981 (tail 4); held-out ~0.34 | **NOT RUN / NOT COMPARABLE** | Only K=20 & K=200 trained; "0.981" is the K=20 answer-prob pilot; tail/named/name-free/held-out unimplemented |
| §5.4 deletion composes — block-of-20 collateral 0.0005; 50 sequential 0.964→0.963 | **CONTRADICTS / NOT RUN** | Single block-of-20 collateral −0.113/−0.122; no sequential-deletion run (§1.4) |
| §5.5 continual addition — 0.995 / 0.957 | **NOT RUN** | Needs new `add_authors` code |
| §5.6 Table 3 — layer placement @ K=50 | **NOT RUN** | Config-supported; no runs (Wave-2) |
| §5.7 Table 4 — term ablations (0.057 / 0.950 / 0.841 / 0.969) | **NOT RUN** | None of the ablation numbers exist |
| §5.7 mechanism — selectivity from output magnitude, not activation | **BACKED (magnitude only) / NOT RUN (rate)** | Output-norm selectivity exists (own_norm 0.059 vs off 0.004; sel up to 1909); the activation-**rate** comparison was never run (§1.1) |
| §5.8 Llama-3.2-3B | **NOT RUN** | No 3B run anywhere |
| §5.9 privacy (MIA/leakage) | **BACKED (partial)** | `sepmlp` deletion MIA→floor (0.997→0.362, privleak −99.6→+4.0); composed-MIA baseline table (§2.2); MemAdapt Priv 0.917 anchor |
| §5.9 relearning attack | **NOT RUN** | Harness built, ladder halted pre-P5; RMU/NPO/grad-diff/SimNPO absent |

---

## 5. Two contradictions to resolve before §5 is finalized

1. **Deletion collateral.** Draft §5.2/§5.4 say ≈0.002/0.0005; the logged number is **retain_Q_A_Prob
   −0.113 / retain_ROUGE −0.122** on surviving authors, masked by aggregate model_utility +0.003
   (`2026-07-22_wscale-refuted-lr2-gray-mechanics.md`). The `sepmlp` thread attributes it to the same
   all-active interference MemSinks showed (own-only recall 0.28–0.51 ≪ all-active). It may ease at a
   smaller K or on ROUGE-L, but **as measured it stands** and must not be reported as 0.002.

2. **The "≈0.80 recall ceiling."** Measured on **answer-probability**, not per-source ROUGE-L. Per the
   conformance audit ("IMPORTANT REFRAME"), the ceiling **cannot stand** until re-measured in ROUGE-L.
   Report it only as an answer-prob diagnostic, pending the recall sweep.

---

## 6. What remains to run (backlog to back §5)

| # | Run | Backs | Cost |
|---|---|---|---|
| 1 | **Per-source ROUGE-L recall sweep over existing K=200 checkpoints** (no retrain; reuse `relearn_score.py::score_author`) — emits mean / tail / named / name-free / held-out | §5.1, §5.3, resolves contradiction C2 | **cheap, 1 GPU eval pass — do first** |
| 2 | K∈{10,50} trains + a tuned config | §5.3 Table 2 K-ladder | medium |
| 3 | Layer-subset configs (0–7, 0–3, 8–15, 12–15) | §5.6 Table 3 | medium |
| 4 | Term-ablation configs (λ_out / λ_h / λ_p → 0) | §5.7 Table 4 | medium |
| 5 | Activation-rate probe (own / foreign / unrelated) | §5.7 mechanism | cheap |
| 6 | Llama-3.2-3B config + OU yaml | §5.8 | heavy |
| 7 | `add_authors` entrypoint + continual-add run | §5.5 | new code |
| 8 | Scripted 50-sequential-deletion loop | §5.4 | cheap |
| 9 | P5 relearn battery (once a passing arm exists) + RMU/NPO/grad-diff/SimNPO | §5.9 Fig 2 | heavy |

**Recommended first move:** run #1. It is the single cheapest action that converts the recall story
(Tables 1–2) and contradiction C2 from "unknown" into a real, paper-comparable number, without any
retraining. It is Phase 1 of `make-sure-our-emthod-wobbly-lemon.md`.

---

## Appendix — provenance (jobs, configs, seeds)
- **MUSR runs:** seed 42; jobs — smoke 446535; K=20 pilot 446714 (bridging 446732); K=200 446949/446950
  (lr5e-4), 447162/447163 (lr2e-4), 447177/447178 (lr1.5e-4 A2), 447175/447176 (w15 B); deletion eval
  447179. Configs: `smoke.json` `7f94b32e5deececc`, `pilot_relu_lr5e-4.json` `f4d6b9d59d62af6d`,
  `sepmlp_1b_k200.json` `be58163496487711`, `…lr2e-4.json` `a493cf09c30224a0`, `…lr1p5e-4.json`
  `c7ca758b9994bf9f`, `…w15.json` `9071404c26d439f7`. (Per-run script/config sha256 in each `meta.json`.)
- **Memory Adapters:** `configs/memadapt_tofu_1b.json`; jobs 443409 / 443441 / 443526 / 443530 / 443531.
- **Composed-MIA:** `attack_mia.py` (`16458b879e32`); SLURM 440741 / 440761; member forget10, non-member
  holdout10.
- **Router-leak:** phase-2 jobs 445357–360 (smoke) / 445668–675 (extended); groupab 446955/446956 (smoke)
  / 447157/447158 (n=400); seed 42.
- **Source log threads:** `../../log/sepmlp/` (8 entries + README), `../../log/memory_adapters/`,
  `../../log/deletion_audit/`, `../../log/router_leak/`, `../../log/memsinks/`, `../../log/routing_scaffold/`,
  `../../log/merge_mechanism/`. Master index: `../../log/README.md`.
