# "Frontiers in LLM Unlearning" Survey — Coverage Map & New-Idea Ranking

**Date:** 2026-06-29 · **Author:** jack (with Claude) · **Status:** synthesis / coverage map
(not an experiment entry — a cross-thread mapping of an external survey onto this repo)

> **Purpose.** A literature-review survey ("Frontiers in Large Language Model Unlearning":
> weight-space projections, parameter-isolated architectures, adapter-merging, benchmarks, and the
> checkpoint-guidance extraction attack) was assessed against what this project has actually built.
> This doc answers two questions: *how much of that survey is implemented*, and *which new
> directions it surfaces are worth pursuing*.
>
> **Distinct from** [`EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md`](EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md):
> that doc maps the 18-paper `papers/` corpus and sets internal direction; this one maps the
> *external survey* and isolates the directions it raises that are **not** on the current roadmap
> (checkpoint-guidance attack, MUSE, deep-unlearning/ES metrics, SGMV, DP).
>
> **How to read:** §1 coverage table · §2 read-off · §3 ranked new ideas · §4 deliberate
> non-coverage. Findings are grounded in three codebase audits (structure / implementation / eval)
> and the three 2026-06-29 strategy docs.

---

## TL;DR

- The repo implements **the entire "exact modular + adapter-merging" half** of the survey
  (SISA-LoRA, LegoNet, S3T, SIFT-Masks, KnOTS, DELLA, `subtract_orth`, JD/Compress-then-Serve) —
  and goes **beyond** the source papers by *verifying* exactness (legonet: bitwise on CPU,
  distributional vs an oracle retrain on GPU). The papers only *assert* it.
- The missing pieces split cleanly into two kinds. **(a) Deliberate:** the *approximate*
  weight/gradient-projection family — UNSC, UNLEARN, GU, ZeroUnlearn — is absent on purpose; it
  conflicts with the exact-by-construction ideal (`CLAUDE.md` §3), and the survey itself flags UNSC
  as intractable (O(d³) per layer). **(b) Open and valuable:** the **adversarial + benchmark-breadth**
  axis — no MUSE/WMDP, no Extraction-Strength / deep-unlearning metric, and no checkpoint-guidance
  attack.
- The single highest-value new idea the survey surfaces is the **checkpoint-guidance extraction
  attack** (§3.1). It challenges the current "attacks are off the table" stance with a *real*
  residual channel: the isolation that buys exact deletion also makes any retained **pre-deletion**
  artifact maximally extractable — the survey's "fundamental paradox."
- The repo **independently discovered** the survey's headline recommendation ("modular PEFT /
  routing for GDPR compliance") as the **scaffold + route** result: `model_utility` **0.664**,
  above dense full fine-tune **0.599**, with O(1) exact deletion
  ([`ORIENTATION_2026-06-29.md`](../tofu_sisa_lora/reports/ORIENTATION_2026-06-29.md)).

---

## §1 — Coverage map

| Survey section | Method | Status | Where |
|---|---|---|---|
| Weight-space projection | `subtract_orth` | ✅ Implemented | [`merge_extra.py:399`](../tofu_sisa_lora/merge_extra.py) `subtract_orth_adapters` (label `subtract_orth`) |
| | UNSC (null-space, O(d³)) | ❌ None | survey itself calls it intractable |
| | UNLEARN (subspace Gram-Schmidt) | ❌ None | — |
| | GU (preconditioned gradient proj.) | ❌ None | — |
| | ZeroUnlearn (closed-form remap) | ❌ None | — |
| Modular / router-free | SISA + SISA-LoRA | ✅ Implemented | [`train_lora_shard.py`](../tofu_sisa_lora/train_lora_shard.py), [`ensemble.py`](../tofu_sisa_lora/ensemble.py) |
| | LegoNet | ✅ Implemented + exactness verified (bitwise + distributional) | [`legonet_model.py`](../tofu_sisa_lora/legonet_model.py) |
| | SGMV / Punica batched serving | ❌ None (sequential PEFT forward) | — |
| Adapter merging | KnOTS (shared basis) | ✅ Implemented | [`merge_extra.py:248`](../tofu_sisa_lora/merge_extra.py) `knots_merge_adapters`; JD/Compress-then-Serve in [`jd_compress.py`](../tofu_sisa_lora/jd_compress.py) |
| | DELLA (SVD + MAGPRUNE) | ✅ Implemented | [`merge_extra.py:158`](../tofu_sisa_lora/merge_extra.py) `della_merge_adapters` |
| | S3T (AllSeq/MinSeq/LongSeq) | ✅ Implemented (faithful repro, thread complete) | [`s3t_sequences.py`](../tofu_sisa_lora/s3t_sequences.py), [`s3t_deletion.py`](../tofu_sisa_lora/s3t_deletion.py), [`train_s3t_shard.py`](../tofu_sisa_lora/train_s3t_shard.py) |
| | SIFT-Masks (sign-fixed) | ✅ Implemented (CPU exactness green; GPU pending) | [`sift_masks.py`](../tofu_sisa_lora/sift_masks.py) |
| | ClAMU (task clustering) | 🟡 Partial — k-means only inside JD compression | [`jd_compress.py`](../tofu_sisa_lora/jd_compress.py) |
| Benchmarks | TOFU | ✅ Fully wired, open-unlearning-faithful | [`eval_tofu.py`](../tofu_sisa_lora/eval_tofu.py) |
| | MUSE / WMDP | ❌ None | — |
| Metrics | PPL, ROUGE-L, truth-ratio / KS `forget_quality` | ✅ | [`eval_tofu.py`](../tofu_sisa_lora/eval_tofu.py) |
| | Extraction Strength (ES), Success-DU / Recall (deep unlearning) | ❌ None | — |
| Attack | Checkpoint-guidance extraction (contrastive `M_pre − w·M_post`) | ❌ None | [`DELETION_AUDIT_PLAN`](../tofu_sisa_lora/reports/DELETION_AUDIT_PLAN_2026-06-29.md) A4 has generic MIA, **not** contrastive pre/post decoding |
| Recommendation | "facts → route + public scaffold" (≈ survey's "modular PEFT for GDPR") | ✅ **Independently discovered** (mu **0.664** > full-FT **0.599**) | [`ORIENTATION_2026-06-29.md`](../tofu_sisa_lora/reports/ORIENTATION_2026-06-29.md) |
| | DP + unlearning | ❌ None | — |

---

## §2 — Read-off

**The gaps aren't random.** Everything the survey calls a *modular / parameter-isolated* or an
*adapter-merging* exact method is built and, in the LegoNet case, audited two ways. The repo is in
fact **ahead of the survey's own sources** on the one thing those papers skip: they *assert*
exactness structurally; this project *verifies* it (bitwise on a deterministic CPU run, and
distributionally — affected-adapter `rel_l2` ≈ the GPU-nondeterminism floor — vs an oracle
retrain). The merge-dilution frontier (mu 0.74 at k=1 → 0.42 at k=200) and the co-adaptation
ceiling (~0.48 for equal-sharding) are quantified here and only proven in theory or shown at 2–8
tasks in the literature.

**What's genuinely missing is the adversarial + benchmark-breadth axis.** Three things the survey
treats as central are absent: alternative benchmarks (MUSE / WMDP), extraction-style metrics
(Extraction Strength; Success-DU "deep unlearning"), and the checkpoint-guidance attack. These are
not "more of the same mechanism" — they are the tools that would *stress-test* the exact methods
already built. That makes them the natural growth edge (§3). The approximate-projection family
(UNSC / UNLEARN / GU / ZeroUnlearn) is a separate story — a deliberate non-target, see §4.

---

## §3 — New ideas, ranked

These are **separate from** the existing top-priority roadmap (the mechanism study
*"facts → route, skills → merge"* + the scaffold honesty checks), which is already well-scoped in
the three 2026-06-29 strategy docs and is **not** relabeled as a new idea here.

### 3.1 Checkpoint-guidance extraction attack — the "isolation paradox" *(top pick)*
- **What.** Implement the survey's contrastive-decoding attack: steer generation with
  `logits(M_pre) − w·logits(M_post)`, restricted to tokens high under `M_pre` and low under
  `M_post`, to reconstruct forgotten sequences. Run it across every arm — SISA-merge,
  routing/LegoNet, SIFT, S3T-deactivation, and the GA/GD/KL/IDK baselines.
- **Why it would work / why it's novel here.** [`ORIENTATION §8`](../tofu_sisa_lora/reports/ORIENTATION_2026-06-29.md)
  rightly puts attacks off the table *for the served model* — post-deletion it equals a retrain by
  construction, so there's nothing to attack. This attack is different: it targets the **pair**
  `(M_pre, M_post)`. In a routed/isolated architecture the dropped expert **is** the forget data in
  maximally-extractable, isolated form — so any retained **pre-deletion** artifact (or the dropped
  adapter itself, if released) may be *more* leaky than a dense model ever was. That is precisely
  the survey's "fundamental paradox," and it is unaddressed by "exact by construction."
- **Cost.** Cheap. Inference-only, **all checkpoints already exist**, reuses
  [`eval_tofu.py`](../tofu_sisa_lora/eval_tofu.py); slots directly into the existing
  [`DELETION_AUDIT_PLAN`](../tofu_sisa_lora/reports/DELETION_AUDIT_PLAN_2026-06-29.md) as the
  contrastive sharpening of its generic A4 MIA.
- **Falsifiable question.** Do routing / SIFT / LegoNet resist contrastive extraction *better or
  worse* than merging — and than the approximate GA/GD baselines? The survey predicts modular PEFT
  resists best on the *served* model but is silent on the released `M_pre`.

### 3.2 MUSE (real-knowledge) benchmark
- **What.** Port MUSE (real news/books the base model has actually seen) alongside TOFU.
- **Why it would work.** [`DELETION_AUDIT_PLAN §5`](../tofu_sisa_lora/reports/DELETION_AUDIT_PLAN_2026-06-29.md)
  already pre-empts the reviewer killer: TOFU authors are *fictitious*, so the base has ~zero prior
  mass on the forget facts — which is *why* module-drop can be exact. MUSE is the principled version
  of the "one probe with base-overlapping knowledge" that plan proposes: it tests whether the
  *facts → route* claim still gives clean deletion when the base **already knows** the facts. This
  attacks the headline result's single biggest scope limitation head-on.

### 3.3 Deep-unlearning recall metric (Success-DU / entanglement)
- **What.** Add a metric TOFU lacks: after deleting author X, can X's facts still be **deduced** via
  logical links to retained facts?
- **Why it would work.** It operationalizes the §5.1 entanglement direction with a concrete number,
  and is the right companion to **sample-level clustering** (the realistic 0.645 setting), where one
  author's samples can split across experts and one expert can mix authors — so "drop author X's
  adapter" is no longer obviously clean.

### 3.4 *(lower)* SGMV / Punica batched serving
- Engineering, not science: completes the throughput story for the routed method so the
  "deployable multi-adapter serving" claim is real. Low novelty, real for a systems framing.

### 3.5 *(lower)* DP-scaffold defense
- The survey's recommended defense against the extraction attack: train the shared public scaffold
  (and/or experts) under differential privacy to bound the contrastive signal. Natural follow-on to
  §3.1 — only worth building once §3.1 shows a leak to defend against.

---

## §4 — Deliberate non-coverage

UNSC, UNLEARN, GU, and ZeroUnlearn are all **approximate** weight/gradient/representation surgery.
They run against the project's stated ideal — *exact, deterministic, O(1)-style "drop the module
that held the data"* (`CLAUDE.md` §3) — and the survey itself rates UNSC intractable at scale
(O(d³) per layer for the activation-covariance SVD). Their absence is a **design choice, not an
oversight**, and is recorded here so it isn't mistaken for a gap. If any is ever revisited, GU (a
lightweight vector-projection variant) is the only member cheap enough to be worth a smoke test.

---

## Pointers

- [`EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md`](EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md) — the internal 18-paper gap analysis & direction-setting.
- [`ORIENTATION_2026-06-29.md`](../tofu_sisa_lora/reports/ORIENTATION_2026-06-29.md) — the scaffold + route headline result.
- [`DELETION_AUDIT_PLAN_2026-06-29.md`](../tofu_sisa_lora/reports/DELETION_AUDIT_PLAN_2026-06-29.md) — the planned recovery-attack audit that §3.1 sharpens.
