# Deletion-Audit Plan — "Does the forgotten knowledge actually stay gone?"
**Date:** 2026-06-29  **Status:** PLAN (pre-smoke)  **Thread (new):** `log/deletion_audit/`
**Decision context:** chosen over going straight to a new method because it is cheap, cannot
fail to produce a result, and produces the headline figure for the follow-on method ("Thesis A").

---

## 0. The gap this closes (why this is novel despite a crowded field)

Every result in this repo scores unlearning with **`forget_quality`** = the KS p-value of the
forget-set truth-ratio distribution vs the `retain90` oracle (open-unlearning-faithful). That
metric only asks: *on a clean eval, is the model distributionally indistinguishable from one that
never trained on the forget set?* It says **nothing about whether the forgotten knowledge is
recoverable.** The whole TOFU leaderboard family (GA/GD/KL/IDK and friends) is **approximate**:
the forget data is *suppressed*, still latent in the weights. Our exact module-drop methods
(SISA remerge, LegoNet `unlearn`, S3T slice-deactivation, SEA expert-drop) are different in kind:
after deletion the forget weights are **physically absent from the served model**.

No paper in `papers/` audits this distinction on TOFU. That is the gap. The audit converts our
project's reason-to-exist from "another point on the forget/utility curve" (where approximate
methods already look fine) to **"the only deletion that survives adversarial recovery."**

### Hypotheses
- **H1 (approximate leaks).** GA/GD/KL/IDK, despite high clean `forget_quality`, lose it under
  relearning / quantization / extraction — the suppressed knowledge resurfaces.
- **H2 (exact module-drop holds).** SISA-remerge / LegoNet-unlearn / S3T-del / SEA-drop keep
  `forget_quality` ≈ oracle after every attack, **by construction** (nothing to recover).
- **H3 (shared/trained components are a hidden leak channel — the bridge to Thesis A).** A method
  that drops the *expert* but keeps a component that was *fit on all data* leaks anyway. Two
  concrete suspects already on disk:
  - **JD `remerge_*`** — the shared basis `U,V` is computed from **all** shards incl. forget,
    then we drop only `Σ_forget`. The basis still encodes forget directions.
  - **RAMoLE `ramole_unlearn`** — learned RouterLoRA + retriever index. (Router was trained on
    retain authors 0–179 only, so this is the *clean* corner case to test: does the embedding
    index / retriever still leak forget-author structure even with the expert dropped?)
  If H3 holds, it proves the theorem under Thesis A: **module-drop is exact iff no shared/trained
  component ever touched the forget data → the router must be training-free.** That is exactly the
  `ramole 06-27` finding ("embedding-RAG routing costs forget_quality vs author lookup") seen
  through the unlearning lens.

---

## 1. Attacks (each reuses `eval_tofu.py` so numbers stay comparable to all prior results)

| Attack | What it does | Fairness control | Predicted split (approx vs exact) |
|---|---|---|---|
| **A1 Benign relearn** | Continue-FT the unlearned/served model for N∈{2,5,10,25} steps on **retain data only** (zero forget data shown), re-eval. | No forget data touches the attack → any forget recovery is a *leak*, not relearning. | approx: FQ collapses; exact: FQ flat |
| **A2 Adversarial relearn** | Continue-FT on a small fraction (5–10%) of forget authors; measure recovery on the **held-out** forget authors. | Held-out facts were never in the relearn set → measures generalization of recovery. | approx: held-out recovers; exact: only the directly re-taught facts move |
| **A3 Quantization** | Load + bitsandbytes `nf4` / `int8`, re-eval. No training. (Zhang et al.: quantization undoes GA-type unlearning.) | Identical quant applied to every condition incl. oracle. | approx: FQ collapses; exact: FQ flat |
| **A4 Membership / extraction** | (a) MIA AUC: per-example loss + min-k%-prob, forget vs unseen holdout. (b) Adversarial-prompt elicitation of forget facts. | Oracle defines the AUC≈0.5 / elicitation-floor reference. | approx: AUC>0.5, elicitable; exact: ≈oracle |

**Primary metric:** Δ`forget_quality` (clean → post-attack). **Robust deletion = Δ≈0.**
**Secondary:** forget-set ROUGE-L recall & answer-prob (direct "did the right answer come back"),
MIA-AUC, elicitation success-rate. **Guard:** also report `model_utility` post-attack (an attack
that nukes utility is not a real recovery).

**Headline figure:** scatter, x = clean `forget_quality`, y = post-attack `forget_quality`.
Diagonal = robust. Predict approximate cluster crashes to the floor; exact module-drop sits on the
diagonal; **JD / RAMoLE land off-diagonal → visual proof of the shared-component leak (H3).**

---

## 2. Condition matrix (all checkpoints already exist — no training to start)

| Class | Checkpoint / label | Hypo |
|---|---|---|
| Oracle (gold "deleted") | `retain90/` (authors 0–179) | control — robust by definition |
| Retain-all (lower bound) | `*_ft` | control — fully leaky |
| **Approximate** | `Llama-2-7B-chat-hf_ft_unlearn_{ga,gd,kl,idk}`, `Llama-3.1-8B-Instruct_ft_unlearn_{…}` | H1 |
| **Exact: SISA-remerge** | `Llama-2-7B-chat-hf_k10_r32_e5_lr1e4` → `remerge_dare_ties` | H2 |
| **Exact: LegoNet** | `Llama-2-7B-chat-hf_legonet_n32_k3` → `legonet_unlearn` | H2 |
| **Exact: S3T** | `Llama-2-7B-chat-hf_s3t_m5_L4_armA_del` | H2 |
| **Exact: SEA** | (locate under `sea_tofu/` — per-author expert drop) | H2 |
| **Shared-component (suspected leaky)** | `remerge_jd_full_c4_r16` (k=10) ; `ramole_unlearn` (1B pool) | **H3** |

Seeds: 42 everywhere (+ at least one re-seed on the smoke arm per §4 of root CLAUDE.md, since the
SEA/LegoNet logs already flag seed variance). Two model sizes (7B + 8B) so H1/H2 aren't a
single-model artifact.

---

## 3. New code (minimal — thin wrappers, all metrics via `eval_tofu.py`)

- `attack_relearn.py` — load a served/unlearned state (reuse `activate_label` / `--preloaded_adapter`
  / `legonet_model` / s3t `_del` loaders), continue-FT N steps on a configurable relearn set
  (`retain_only` | `partial_forget`), write the attacked adapter, then invoke
  `eval_tofu.evaluate_model`. **Module-drop fairness:** the attack adds a *fresh* adapter on top of
  the frozen base+surviving experts — so any recovery without forget data is a genuine leak.
- `attack_quantize.py` — `--quantize {nf4,int8}` load path → `eval_tofu`. (May fold into a flag.)
- `attack_mia.py` — per-example loss / min-k%-prob → AUC(forget vs holdout); reuses the eval
  forward pass.
- `configs/deletion_audit.json` — attack grid, relearn steps/lr, condition list.
- `submit_deletion_audit.sh` — `%4`/`%6` GPU-capped SLURM driver (smoke → extended), `STUB=1`
  preview, skips existing result JSONs (mirror `submit_scale_grid.sh` discipline).
- `test_deletion_audit.py` — CPU regression (tiny Llama): attack harness runs, FQ moves the right
  way on a planted GA-vs-remerge mini-case, determinism.

Nothing in the existing metric/merge/routing code changes → `test_ou_equivalence.py`,
`test_merge_extra.py` stay green; the audit is purely additive.

---

## 4. Execution order (smoke-first, per root CLAUDE.md §4)

1. **CPU pre-flight** — `test_deletion_audit.py` green; `STUB=1 bash submit_deletion_audit.sh`
   prints every sbatch script.
2. **Smoke (TinyLlama-1.1B, k=4, 1 GPU, bs1, 2–5 relearn steps, tiny set):** run A1+A3 on exactly
   two conditions — `ft_unlearn_ga` (must leak: FQ drops) vs `remerge_dare_ties` (must hold: FQ
   flat). **Go/no-go:** if the GA-vs-remerge direction doesn't separate at smoke scale, the harness
   is wrong — stop and debug before any 7B job.
3. **Full SLURM:** A1–A4 × full condition matrix × {7B, 8B}, `%≤6` GPUs, seed 42 (+1 reseed on the
   exact arms). Provenance (job IDs, git hash, config sha) → the `log/deletion_audit/` entry.
4. **Collect + figure:** extend `collect_results.py`/`generate_smoke_report.py` to emit the
   clean-vs-post-attack scatter + the ΔFQ table; write `reports/DELETION_AUDIT_REPORT_<date>.md`.

---

## 5. What each outcome means (pre-registered reading)

- **H1 ✓ & H2 ✓** → the core paper: "clean `forget_quality` does not certify deletion; exact
  module-drop is the only family that survives recovery attacks." Strong on its own.
- **H3 ✓** (JD/RAMoLE leak while SISA/LegoNet/SEA hold) → the bridge result and the design law for
  **Thesis A**: training-free, leak-free routing is *required* for certified exactness, not just
  simpler. This is the differentiator vs SEA's trained entailment router and RAMoLE's RouterLoRA.
- **H2 ✗** (some "exact" method leaks unexpectedly) → still publishable and *more* interesting:
  a concrete demonstration that an architecture believed exact is not, with the mechanism.
- **TOFU-fictitious caveat to pre-empt reviewers:** TOFU authors are fictitious, so the frozen base
  has ~zero prior mass on the forget facts — which is *why* module-drop can be exact. State this as
  a scope condition and add one probe: a forget fact that overlaps base knowledge, to show where
  module-drop's exactness boundary actually is (itself a finding).

---

## 6. Then → Thesis A (the method this de-risks)

Unify `sisa_lora` + `legonet_lora` + `s3t` + `sea` into one method: **training-free, leak-free
routed per-entity LoRA experts** with (i) the H3 router-leak result as motivation, (ii) this audit
as the robustness evidence, (iii) zero knowledge-entanglement (perfect retain preservation) as the
answer to TOFU's own stated limitation. The audit's scatter figure is Thesis A's Figure 1.
