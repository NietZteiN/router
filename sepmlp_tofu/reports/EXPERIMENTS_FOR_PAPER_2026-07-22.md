# Experiments from the logs that can back the MUSR paper — inventory & gap map

**Date:** 2026-07-22 · **Author:** research-log audit · **Scope:** MUSR (`sepmlp`) + the
baselines the paper already cites (Memory Adapters, composed-model MIA, router-leak).

## What this is
A mapping from **experiments we have actually run** to the **paper's Experiments section**
(§4 Deletion Semantics, §5.1–5.9, Tables 1–4). Every number is provenance-tagged so it can
be traced and pasted. Three framing facts up front:

1. **The MUSR method's own harvest is thin.** Only selectivity/localization, deletion
   mechanics, and the parameter count are backed by executed `sepmlp` runs. Table 4 ablations,
   the mechanism (activation-rate) experiment, 3B, the layer sweep, the K-ladder, continual
   addition, sequential deletion, and the relearning attack were **never run** (Section B).
2. **Our K=200 "recall" is answer-probability, not the paper's per-source ROUGE-L.** The
   paper-comparable recall (mean/tail/named/name-free/held-out) has **never been computed**, so
   the headline 0.966 can be neither confirmed nor refuted from the logs. See the conformance
   audit `../../.claude/plans/make-sure-our-emthod-wobbly-lemon.md` §D1.
3. **Two logged numbers contradict the current draft** — deletion retain-collateral −0.11/−0.12
   (draft says 0.002) and the answer-prob "≈0.80 ceiling" (a possible metric artifact). See
   Section C — resolve before §5 is finalized.

Unless noted: single seed (42), **Llama-3.2-1B-Instruct**, spec-v2 recipe
(`L1 + 10·hinge + 50·Gram + 1·promotion`), width-32 adapters on all 16 MLP layers.

---

## Section A — Experiments we can back today

### A1. MUSR / `sepmlp` own runs
Provenance paths are relative to `log/sepmlp/` unless prefixed.

| Paper location | What it backs | Real numbers | Provenance |
|---|---|---|---|
| **§5.7 mechanism / isolation** (self-routing — adapters silent on foreign inputs) | Trained self-gating is real and lr-dialable; foreign output driven toward 0 | K=20 pilot (job 446714) median on/off-selectivity / own-prob (all-active): lr3e-4 **4.38 / 0.981** · lr5e-4 **7.17 / 0.977** (min author 0.936, none <0.8) · lr1e-3 **38.6 / 0.778** · lr3e-3 **1909.7 / 0.695**; vs the LoRA negative-anchor ceiling **1.11**. Smoke telemetry: own_norm **0.0591** vs off_norm **0.0040**; OOD ood/own alpaca 0.0079 / real_authors 0.0249 (→0.0002 at lr3e-3) | `2026-07-21_pilot-oom-and-adjudicate.md`, `2026-07-21_bridge-go-k200-launch.md` |
| **§5.7 / scaling** | Selectivity holds and amplifies at K=200 | K=200 (job 446949) median selectivity **507.5** (frac≥5 = 1.00); ood/own alpaca 0.0006 / real 0.0016 / world 0.0013 (~70× amplification vs pilot) | `2026-07-21_k200-g3-fail.md` |
| **§5.4 deletion / §5.9 privacy** | Deletion is clean, cheap, and MIA-safe (block of 20 = forget10, authors 180–199) | forget_Q_A_Prob 0.767→**0.054**, forget_ROUGE 0.760→0.316, extraction 0.469→**0.047**, exact_mem 0.935→0.490, mia_loss 0.997→**0.362**, mia_min_k 0.997→0.358, privleak −99.56→**+3.98**; physical slice removal **1.07 s** (20 authors × 20 slices/layer); dropall → symmetric floor (forget 0.092 ≈ retain 0.087, utility 0.281 ≈ base) | `2026-07-22_wscale-refuted-lr2-gray-mechanics.md` |
| **Table 1 (cost row)** | Added-parameter count | K=200 checkpoint = 6.3×10⁸ params = **+0.63B (+50%)**, width n=32 (matches paper Table 1) | `2026-07-21_k200-g3-fail.md`; `../CLAUDE.md` |
| **§5.3 scaling — DIAGNOSTIC ONLY (not paper recall)** | K=200 selectivity↔recall tradeoff, measured on **answer-probability** | 4 points (median sel → answer-prob recall): 507.5→0.637 · 36.0→0.747 · **16.33→0.795 (best)** · 24.59→0.696; K=20 pilot 0.977. All four lie on one monotone curve topping ≈0.80 in the healthy-sel band | `2026-07-21_k200-g3-fail.md`, `2026-07-22_hk200lr-refuted.md`, `2026-07-22_wscale-refuted-lr2-gray-mechanics.md` |

**Architecture-motivation datapoint** (cite where the draft argues *why* router-free per-source
MLPs rather than LoRA): the LoRA negative-anchor pilot refuted penalty-trained self-gating —
on/off selectivity stuck at **1.11–1.15** at every λ∈{1,10,100} while recall collapsed to 0.525
at λ=100. `../../log/merge_mechanism/2026-07-16_negative-anchor-pilot-results.md`.

### A2. Baselines the paper already cites

| Paper location | Baseline | Real numbers | Provenance |
|---|---|---|---|
| **Table 1 — "MemAdapt (block-list)" column** + §5.9 privacy anchor | Memory Adapters (Grimes et al.) repro, frozen 1B, matched base, 200 authors × 256 slots | after-unlearn Util.R/Util.G/Mem/Priv/Agg = **1.066 / 1.013 / 0.630 / 0.917 / 0.869** (paper 1.00/1.06/0.62/0.98/0.87 — **Agg 0.869 reproduces 0.87**); retrain oracle **1.009/1.121/0.590/1.000/0.874**; FT (pre-unlearn) 1.075/1.024/0.027/0.380/0.097; unlearn 20 authors = 5,120 entries in **0.0269 s** (CPU), apply-at-load 0.021 s | `../../log/memory_adapters/2026-07-15_first-full-run-results.md`; `../../memadapt_tofu/REPORT_2026-07-15.md` |
| **§5.9 privacy** (approximate-leaks-vs-exact-doesn't) | Composed-model MIA, loss-AUC (member = forget10, non-member = holdout10) | retrain-oracle floor **0.379**; **approximate** GA **0.743** / GD **0.779** / KL **0.753** / IDK **0.816** (min-k up to 0.83), leaky upper-bound ft 0.815; **exact module-drop ≤ floor** LegoNet 0.369 / routed-key 0.375 / ClAMU 0.322 / SIFT 0.254; live controls sift_full **1.000** / legonet_full 0.812 (attack is not dead) | `../../log/deletion_audit/2026-07-06_composed-mia-results.md`; `../../tofu_sisa_lora/reports/DELETION_AUDIT_REPORT_2026-07-06.md` |
| **§4 Deletion Semantics: Static vs Runtime** (currently empty) | Router-leak campaign | *Runtime router still serves the deleted data:* Mode-B replicated-fact leak **ρ 0.833** (R=8, sibling policy); confidence-based defenses fail (ClAMU-granularity selector AUC **0.585**; the confidence family 0.56–0.63 in phase 1). *Static identity seal:* per-author tombstone collapses ρ 0.833→**0.047** (~95% sealed), separability AUC ≈**0.98**, retain cost ≈**0.006 mu**, disclosure AUC 0.839. *Static exact edit needs no seal:* under 100% forced misrouting the deleted gold's answer-prob stays at floor — SIFT **0.086**, ClAMU **0.124** (vs base 0.128, served-author ceiling ~0.9) — "nothing to restore" | `../../log/router_leak/2026-07-18_phase1-results.md`, `..._phase2-results.md`, `2026-07-22_groupab-depth-results.md`; `../../tofu_sisa_lora/reports/ROUTER_LEAK_EXPLAINED_2026-07-21.md` |

**Caveats to print with A2:**
- All single-seed (42). MemAdapt **Priv 0.917 < paper 0.98** (a composition/routing-entropy miss,
  unresolved; an alternate `priv_absdiff` composition reads 0.953). Seeds 43/44 not run.
- The composed-MIA numbers are for the **sibling exact-drop methods** (LegoNet/routed-key/ClAMU/
  SIFT), not `sepmlp`. MUSR's own MIA-to-floor (A1: 0.997→0.362) echoes the same story, but a
  standalone `sepmlp` Priv / raw-AUC battery was **not run** (H5, open).
- The "router holds it / static edit doesn't" contrast is strongest for **dense-embedding**
  routers; the leak is router-family-specific (H-ARCH refuted — ppl/tf-idf self-detect orphans).
- The **`sepmlp` routerless leak-probe row was never produced** (`measure_selectivity --probe
  forget_leak` is on the deferred list) — so MUSR does not yet have its own row in this table.

---

## Section B — Draft §5 claims NOT backed by any run, and the run each needs

| Paper claim | Log status | Run needed |
|---|---|---|
| §5.1 / Table 2 **recall = per-source ROUGE-L** (mean / tail / named / name-free / held-out) | **not implemented or run** — #1 blocker; every "recall" to date is answer-probability | `measure_recall.py` reusing `relearn_score.py::score_author` (already greedy + `rougeL.recall`); run over **existing** K=200 checkpoints, no retrain → first paper-comparable number |
| §5.2 / Table 1 MUSR column (recall 0.966, util 0.559, forget ROUGE 0.832, deleted 0.319, collateral 0.002) | **no valid headline** (blocked on the metric + a passing arm); collateral **contradicts** (Section C) | ROUGE-L recall on a passing arm + OU utility wired to anchors 0.281 / 0.599 |
| §5.3 / Table 2 K-ladder 10/50/200 + tuned config | only **K=20 and K=200** trained; K=10, K=50, and a tuned config **not run** | K∈{10,50} trains + a tuned config |
| §5.6 / Table 3 layer-placement sweep | **not run** (config-supported via `layers`) | layer-subset configs (0–7, 0–3, 8–15, 12–15) at fixed budget |
| §5.7 / Table 4 term ablations (no output penalty / no hinge / no promotion) | **not run** | 3 ablation configs (λ_out / λ_h / λ_p → 0) |
| §5.7 mechanism — activation-**rate** vs output-**magnitude** | **not run**; only output-*norm* telemetry exists | probe ReLU-threshold-crossing rate on own / foreign / unrelated inputs |
| §5.8 Llama-3.2-3B | **not run** | 3B config + 3B OU yaml (code is model-agnostic) |
| §5.5 continual addition (added 0.995 / existing 0.957) | **not run** — needs new code | an `add_authors` entrypoint (only `remove_authors` exists) |
| §5.4 50 sequential deletions (0.964→0.963) | **not run** — only one block-of-20 was done | scripted sequential-deletion loop |
| §5.9 relearning attack (recovers no faster than never-trained) | **not run** — harness built, ladder halted pre-P5; RMU/NPO/grad-diff/SimNPO baselines **absent** | run the P5 relearn battery once a passing arm exists |

---

## Section C — Two contradictions to resolve before §5 is final

1. **Deletion collateral.** Draft §5.2 / §5.4 say ≈0.002 / 0.0005. The logged number is
   **retain_Q_A_Prob −0.113 / retain_ROUGE −0.122** on surviving authors, masked by aggregate
   model_utility +0.003 (`2026-07-22_wscale-refuted-lr2-gray-mechanics.md`). The `sepmlp` thread
   attributes it to the same all-active interference MemSinks showed (own-only recall 0.28–0.51 ≪
   all-active). It may ease at a smaller K or on ROUGE-L, but **as measured it stands** and must
   not be reported as 0.002.

2. **The "≈0.80 recall ceiling."** Measured on **answer-probability**, not the paper's per-source
   ROUGE-L. Per the conformance audit ("IMPORTANT REFRAME"), this ceiling **cannot stand** until
   re-measured in ROUGE-L. Report it only as an answer-prob diagnostic pending the Section-B
   recall sweep — do not cite it as a result.

---

## Recommended next step (one cheap, high-leverage run)
The per-source **ROUGE-L recall sweep over the existing K=200 checkpoints** (no retraining) is the
single action that turns several Section-B rows and contradiction C2 from "unknown" into a real,
paper-comparable number. It is Phase 1 of `make-sure-our-emthod-wobbly-lemon.md` and unblocks the
Table 1 / Table 2 recall story before any new training is spent.
