# MemSinks on TOFU — Complete Report (all results)

**Dates:** 2026-07-14 → 2026-07-16 · **Compute:** ≈9 GPU-hours (1×A40 per job) ·
**Code:** `~/memsinks_tofu/` · **Ledger:** `~/log/memsinks/` (9 dated entries; every
hypothesis pre-registered with numeric pass/fail gates *before* each run).

This document is self-contained: Part I explains the problem, the benchmark, and the
method from scratch; Part II walks through every experiment with the full result tables;
Part III gives the conclusions, limitations, and provenance. Skimmers: §8 is the verdict.

---

# Part I — Background (start here if new to this)

## 1. The problem: making a model forget

When a person's data must be removed from a trained language model, retraining from
scratch without that data is the gold standard — and usually unaffordable. Two families of
shortcuts exist:

- **Approximate unlearning** post-processes the trained model (e.g. gradient *ascent* on
  the forget data). Cheap, but nothing guarantees the information is gone; this lab's own
  audits show such models remain highly identifiable (membership-inference AUC 0.74–0.82
  vs a 0.379 floor for genuinely clean models).
- **Exact unlearning** structures *training* so deletion is provable. The founding template
  (SISA) trains isolated components on disjoint data; deleting data = dropping/retraining
  only its component. The engineering ideal: deletion as a cheap, deterministic, O(1)
  "remove the module that holds it" operation.

The catch, documented across this lab's threads: serving per-person facts from one
composed model always requires *something* that selects the right stored component per
query — a router, a mask, a lookup key. **Where that selection lives determines what
deletion costs and where information leaks.** Before this work, one cell of that design
space was unmeasured: selection installed by **training-time neuron allocation** — the
promise of MemSinks.

## 2. The benchmark: TOFU, and how it is scored

**TOFU** = 200 *fictitious* authors × 20 question–answer pairs (4,000 rows), fine-tuned
into a pretrained LLM (here: Llama-3.2-1B-Instruct). Fictitious means the base model knows
none of it — every answerable fact was learned during fine-tuning, so unlearning can be
tested cleanly. The deletion unit is the **author**; the standard forget set ("forget10")
is authors 180–199.

Metrics (computed by the lab's `eval_tofu.py`, a numerically faithful port of the official
open-unlearning scorer; "smoke tier" = subsampled ~55-min evaluations used for screening):

| metric | meaning | orientation |
|---|---|---|
| **model_utility (mu)** | harmonic mean of 9 components: {retained authors, real authors, world facts} × {answer probability, ROUGE-L recall, truth-ratio}; one collapsed component pulls mu toward 0 | higher = better; fine-tuned ceiling here ≈ 0.64 |
| **forget_quality (fq)** | KS-test p-value: is the model's uncertainty on forget questions statistically indistinguishable from a "retain-only oracle" that never saw the forget data? | higher = better unlearning; > 0.1 respectable |
| **forget ROUGE / prob** | how much of the forget answers the model still reproduces | should drop to the never-trained floor after deletion |
| **ppl** | perplexity; health check — explosions mean broken generation corrupting other metrics | ~1–3 healthy here |

Anchors on this exact stack (all smoke tier, seed 42): plain LoRA fine-tune of all 200
authors **mu 0.6438** with forget ROUGE 0.9425 (fully memorized); base model mu ≈ 0.42;
never-trained forget-ROUGE floor **≈ 0.40–0.47** depending on serving base. The lab's best
exact methods: routed experts+scaffold **0.7509**, SIFT-Masks **0.737** (its deletion
lifts fq to **0.393**), SEA per-author adapters **0.711** (at 32 MB storage per author).

## 3. The method under test: MemSinks (ICML 2025)

MemSinks — implemented as **SeqTD, "sequence-tied dropout"** — reserves a fraction of each
MLP layer's intermediate neurons as a **sink pool**. Every training sequence gets a
pseudo-random subset of sink neurons (a deterministic hash of its integer sequence ID)
that is active **only** while training on that sequence; the remaining "general" neurons
are always active. Intended dynamic: shared/generalizable signal accumulates in the
general neurons, while each sequence's rote memorization funnels into its private sinks.
**Unlearning = zero the sinks** — no retraining, and (the selling point) no router at
serving time.

Scope facts about the paper, established before running anything:

- It trains **from scratch** (TinyStories; SmolLM up to 1.7B). It never fine-tunes.
- Its only evaluated deletion op is **drop the entire sink pool**, reported to close >50%
  of the memorization gap — explicitly a partial removal.
- Its own Theorem 4.1 proves **co-adaptation**: shared and sink weights entangle
  increasingly with training; perfect separation is unreachable. (This lab had long cited
  that theorem — it explains why equal-shard merging plateaus at mu ≈ 0.48 — but had never
  run the method.)
- **Exactness status, stated up front:** the deletion *op* is exact (bitwise), but unlike
  SISA/SEA there is no provenance guarantee — forget examples also send gradients into
  shared weights, which deletion never touches. Except for our strict-isolation variant
  (§7), which does carry a provenance guarantee.

## 4. What we built (the port), and how it was verified

The released code is a component library for a different framework (litgpt) with no
trainer, no metrics, no fine-tuning path, and no targeted per-sequence deletion. We ported
the ~90-line mechanism into the lab's HuggingFace/PEFT stack:

- **Masked LoRA delta.** Masking *pretrained* neurons during fine-tuning would destroy
  base capabilities, and TOFU memorization necessarily lives in the fine-tuning update
  (the authors are fictitious). So the sequence-tied mask gates the **LoRA update** on the
  MLP gate/up projections; a masked sink neuron behaves exactly like the pretrained
  neuron. Implemented as forward hooks on the LoRA B-matrices (verified against peft
  0.14.0 internals: the hook sees the pre-scaling delta, and the scalar scaling commutes).
- **Author-level ownership; disjoint slices as the primary scheme.** The paper's hashed
  masks are ported verbatim — including an int64-overflow quirk that *defines* the
  published masks, and a discovered degeneracy (sequence-ID 0 hashes to an all-ones mask;
  we use IDs 1–200). Measured overlap fact: at the paper's density (p=0.3), the union of
  the 20 forget authors' hashed masks covers **97.4%** of the sink pool — "selective"
  deletion would equal total deletion. The primary arm therefore assigns each author a
  **disjoint slice: 12 neurons per layer** (of 8192, at the paper's 70/30 general/sink
  split; 16 layers).
- **Deletion = offline bake.** For any fixed per-neuron keep/zero vector, hook-masking ≡
  row-scaling the LoRA B weight (a linear-algebra identity, unit-tested bit-identical), so
  every deletion condition is materialized as a bone-stock adapter directory the existing
  evaluator serves unchanged — generation, KV-cache and all.
- **Training recipe** = the lab's frozen recipe (LoRA r32/α64, 5 epochs, lr 1e-4, seed 42,
  bf16), with one documented deviation (gate_proj added to the adapted modules — the paper
  gates the whole SwiGLU neuron). The control **CTRL-L** is module-matched exactly.
- **Verification:** 22 CPU regression gates, green before every submission — hash-port
  equivalence executed against the reference source; bake≡hook bit-identity; gradient
  isolation (masked rows receive exactly-zero gradients); deletion isolation; KV-cache
  generation identity; tokenization parity with the lab's standard trainer; and the strict
  arm's data-provenance gate (§7). Plus per-epoch training telemetry ("memorization-gap
  probe": each probe author's answer probability under its own mask vs with its own sinks
  deleted).

---

# Part II — Experiments and all results

Timeline at a glance:

| phase | jobs | trains | what it asked | verdict |
|---|---|---|---|---|
| Round 1 "lean" | 443146–49 | M1 + CTRL-L (~10 min each) | does slice deletion unlearn? | No — deletion ≈ placebo |
| Phase D diagnosis | 443549–50 | none (eval-only) | why not? | interference + empty slices |
| Strict run 1 | 443562–65 | strict (~6.5 min) | force isolation | training diverged (init-scale bug) |
| Strict run 2 | 443939–42 | strict2 (~6 min) | scale fixed | stable but underfit (5 steps/author) |
| Strict run 3 | 444254–57 | strict2-e25 (~31 min) | 25 steps/author | **capacity floor licensed; deletion works** |

## 5. Round 1 — train MemSinks, delete, evaluate

Setup: M1 = MemSinks (disjoint slices) on all 200 authors; CTRL-L = identical plain LoRA.
From the single M1 adapter, five deletion conditions were baked (32 tensors each; 240/120/
24/2400/240 neurons zeroed; zero collateral on retained authors by construction) and
evaluated alongside the full model and the control.

**Full results (smoke tier, seed 42):**

| condition | mu | fq | forget ROUGE | forget truth-ratio | retain prob | retain ROUGE | real prob | world prob | ppl f/r |
|---|---|---|---|---|---|---|---|---|---|
| CTRL-L (plain LoRA) | **0.6438** | 0.0003 | 0.9425 | 0.3501 | 0.9296 | 0.9255 | 0.4656 | 0.5549 | 1.30/1.32 |
| MemSinks, all slices on | 0.4373 | 0.0065 | 0.6936 | 0.4168 | 0.6917 | 0.6390 | 0.3291 | 0.5204 | 2.44/2.60 |
| — delete forget01 (2 authors) | 0.4099 | 0.0025 | 0.6551 | 0.4172 | 0.6917 | 0.6234 | 0.3267 | 0.5214 | 2.44/2.60 |
| — delete forget05 (10 authors) | 0.3535 | 0.0065 | 0.7191 | 0.4206 | 0.7034 | 0.6497 | 0.3342 | 0.5212 | 2.38/2.53 |
| — delete forget10 (20 authors) | 0.3999 | 0.0065 | 0.6566 | 0.4206 | 0.7043 | 0.6525 | 0.3380 | 0.5096 | 2.39/2.50 |
| — delete 20 RANDOM retained (placebo) | 0.4047 | 0.0025 | 0.7037 | 0.4152 | 0.7030 | 0.6403 | 0.3453 | 0.5215 | 2.36/2.53 |
| — drop ALL sinks | 0.6399 | 0.0065 | **0.8726** | 0.4203 | 0.8910 | 0.8547 | 0.4488 | 0.5504 | 1.45/1.47 |

**Training telemetry** (memorization-gap probe, mean over 7 probe authors × 2 rows —
answer probability under own mask vs own-sinks-deleted):

| epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| gap | 0.060 | 0.258 | 0.308 | 0.348 | **0.369** |

(Own-mask per-author probabilities reach 0.71–0.96 by epoch 5; the "deleted" condition
also *rises* 0.33→0.50 — a first hint that shared capacity is learning the content too.)

**Readings.** (1) Deleting the *right* neurons ≈ deleting *random* neurons ≈ deleting
nothing — the placebo test is the decisive comparison (0.6566 vs 0.7037 vs 0.6936, inside
the ±0.05 smoke-noise band). fq never leaves 0.0065 (target: >0.1). (2) With **every**
sink removed, forget ROUGE is still 0.8726 — shared capacity reproduces the forget
answers nearly as well as the control. (3) The router-free serving mode ("all slices on")
costs ~0.20 mu: 200 per-author deltas that never co-occurred in training act as mutual
noise. Health checks clean throughout (ppl 1.3–2.6; no degenerate generation).

## 6. Phase D — the diagnosis (evaluation-only, reusing the trained M1)

Two instruments, one day, ~1 GPU-hour.

**D1 — routed serving.** Serve each TOFU query under its author's *training* mask (an
oracle router; out-of-distribution queries → general-neurons-only):

| condition | mu | fq | forget ROUGE | retain prob | retain ROUGE | ppl f/r |
|---|---|---|---|---|---|---|
| routed, full | **0.6417** | 0.0025 | 0.9154 | 0.9040 | 0.8617 | 1.24/1.24 |
| routed, forget10 deleted | 0.6417 | 0.0065 | 0.8726 | 0.9040 | 0.8617 | 1.45/1.24 |
| (CTRL-L reference) | 0.6438 | 0.0003 | 0.9425 | 0.9296 | 0.9255 | 1.30/1.32 |

**D2/D3 — slice-content probe** (all 200 authors × their 20 training rows; answer
probability under three fixed servings, plus an interference ladder):

| statistic | value |
|---|---|
| shared-capacity only ("gen_only") | **0.9006** |
| shared + own slice ("gen_own") | 0.9139 |
| all slices on ("all_on") | 0.7372 |
| **slice increment** (gen_own − gen_only) | **+0.0133** (forget authors 0.0145, retained 0.0131) |
| interference (gen_own − all_on) | 0.1767 |
| ladder: prob vs # foreign slices active (k ∈ 0,10,50,100,199) | monotone decreasing for **20/20** authors |

**Readings.** Routed serving restores control-level utility — the all-on deficit was pure
interference (H9 ✓). But the slices are near-empty: shared capacity alone answers an
author's own training rows at 0.90; the author's 12 neurons add +0.013 (H11 ✓). Routed
deletion moves forget ROUGE only 0.9154 → 0.8726 — exactly the shared-capacity level; gap
closure toward the oracle ≈ 8%. The Round-1 training-telemetry "gap" of 0.369 is thereby
decomposed: ≈ 0.35 interference relief + ≈ 0.013 actual slice content.

**Phase-D verdict: fine-tuned MemSinks is an oracle-mask router that stores nothing
deletable** — it inherits the serve-time-selection cost of mask methods (SIFT/ClAMU)
while providing none of their deletion power. This is the paper's own co-adaptation
theorem, measured in the fine-tuning regime.

## 7. The strict-isolation dial — can *forced* isolation fix it?

Design: eliminate all shared *trainable* capacity. LoRA A-matrices frozen at a seeded
random projection (nothing shared ever trains); every one of the 8192 neurons/layer
assigned to exactly one author (**40 rows each**; a discovered remainder-trap — 192
leftover neurons that a naive partition would have made always-on — is closed by marking
them permanently dead); base model = the lab's "scaffolded" base (public-data adapter
baked in, so generic QA competence is author-independent); one author per optimizer step.

**Provenance guarantee (gate-tested):** re-running training while changing *only another
author's batch* leaves an author's rows **bit-identical** — including through Adam
momentum, whose tails move rows during other authors' steps but only as a function of the
author's own gradient history (a subtlety the test suite caught; the claim was re-worded
before any results existed). Deletion of an author zeroes their rows — provenance-exact.
Per-author footprint: 40 rows × r32 × 2 projections × 16 layers ≈ **80 KB** (SEA: 32 MB —
a ×400 compression of the deletion unit).

Three runs, one variable at a time:

**Run 1 (strict): DIVERGED.** The frozen-A initializer (inherited from the lab's IRP
utility) uses std 1.0 — ~45× the standard LoRA init scale — amplified by rsLoRA ≈ 11.3,
with clipping deliberately disabled. Loss: 2.68 (healthy scaffold start) → 8–12, never
recovers (avg 8.51).

| condition | mu | fq | forget ROUGE | retain prob | real/world prob | ppl f/r |
|---|---|---|---|---|---|---|
| strict routed, full | 0.0014 | 0.3929 | 0.0598 | 0.0002 | 0.6305/0.6556 | 7275/8654 |
| strict routed, deleted | 0.0014 | 0.3929 | 0.4647 | 0.0002 | 0.6305/0.6556 | 17.6/8654 |
| strict all-on | 0.0 | 0.1350 | 0.0401 | 0.0 | 0.2521/0.2708 | 623016/591976 |

Probe: gen_only 0.1396 (= untouched scaffold), gen_own **0.00017** — the trained slices
make own-author rows ~800× *worse* than not training at all (slice increment −0.139).
Healthy out-of-distribution numbers (0.63/0.66) prove the serving machinery itself is
fine. **No capacity conclusion licensed — this is an optimization blow-up.** (Two bugs
fixed en route: the initializer's CPU-generator-on-CUDA crash, and the scale itself.)

**Run 2 (strict2 = scale fix, clip restored): STABLE BUT UNDERFIT.** Same seeded frozen
directions at std 1/√fan-in, clipping 0.3. Loss stable, 2.68 → 2.29–2.44 (avg 2.51) — but
each author receives exactly **5 optimizer steps** (one per epoch), and the lab's own
prior result says 20-row units need ~25 steps to memorize.

| condition | mu | fq | forget ROUGE | retain prob | retain ROUGE | ppl f/r |
|---|---|---|---|---|---|---|
| strict2 routed, full | 0.4466 | 0.9578* | 0.4730 | 0.1718 | 0.4805 | 11.8/11.3 |
| strict2 routed, deleted | 0.4466 | 0.3929 | 0.4647 | 0.1718 | 0.4805 | 17.6/11.3 |
| strict2 all-on | 0.0305 | 0.0346 | 0.0959 | 0.0139 | 0.0896 | 93/92 |

Probe: gen_own 0.1627 vs gen_only 0.1396 → slice increment **+0.023** (now positive — the
isolation trains the right rows), train-end own-vs-deleted gap 0.171 with deleted ≈ 0.01.
(*The fq 0.9578 is vacuous: a barely-trained model "looks never-trained" to the KS test —
high fq without utility means nothing.) **Still no capacity verdict — underfit, with
steps-per-author confounding expressivity.**

**Run 3 (strict2-e25 = 25 steps/author): THE ANSWER.** Loss converges to 1.0–1.2; the
learning curve saturates:

| epoch | 1 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| own-mask prob | 0.152 | 0.257 | 0.374 | 0.453 | 0.498 | **0.504** |
| own-deleted prob | 0.155 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |

(+0.006 over the last 5 epochs = a hard plateau; isolation perfect from epoch 5.)

| condition | mu | fq | forget ROUGE | retain prob | retain ROUGE | ppl f/r |
|---|---|---|---|---|---|---|
| strict2-e25 routed, full | **0.6305** | 0.0346 | 0.5537 | 0.4145 | 0.5857 | 3.3/3.0 |
| strict2-e25 routed, deleted | **0.6305** | **0.3929** | **0.4647** | 0.4145 (bit-unchanged) | 0.5857 | 17.6/3.0 |
| strict2-e25 all-on | 0.0 | 0.1350 | 0.0049 | 0.0 | 0.0083 | 222545/204017 |

Probe (200 authors): gen_only 0.1396, gen_own **0.3890**, slice increment **+0.2493**
(forget authors 0.239 ≈ retained 0.251), all_on 0.0000, ladder monotone 20/20.

**Readings.** (A) **The capacity floor, finally licensed:** with healthy, saturated
training, 40 frozen-basis rows per author hold ~**0.39–0.50** recall — roughly *half* of
what a full adapter achieves on the same rows (0.9991, the lab's prior e25 measurement).
(B) **Deletion now works perfectly:** zeroing an author's slices drops their forget ROUGE
to the never-trained floor *exactly* (0.5537 → 0.4647 = the scaffold's own 0.4647), lifts
fq to 0.3929 — numerically identical to SIFT-Masks' published unlearn result — and leaves
every retained metric bit-unchanged. Real, provenance-exact, O(1) deletion at 80 KB per
author; it just protects half-strength memories at this slice width. (C) The all-on
collapse *scales with slice strength* (ppl 93 after weak training → 222k after strong) —
the lab's "merging destroys facts" phenomenon reproduced inside a single adapter.

---

# Part III — Conclusions

## 8. Verdict

1. **MemSinks-as-published does not transfer to fine-tuning as an unlearning method.**
   Training-time neuron allocation without gradient isolation is cosmetic: memorization
   co-adapts into shared weights (sinks-off recall 0.87 vs control 0.94; slice content
   +0.013 vs 0.90 stored elsewhere), the designated-vs-random-vs-no deletion comparison is
   a three-way tie, and the selection-free serving claim fails (all-on serving
   self-interferes ~0.20 mu; usable serving needs an oracle router). This is the paper's
   own co-adaptation theorem operating in the fine-tuning regime — now with numbers.
2. **Forcing full isolation makes deletion real — and converts the method into per-author
   experts.** At the converged operating point: exact, floor-perfect deletion (fq 0.393 =
   SIFT's unlearn; retain untouched to 4 decimals) at ~80 KB/author, paying a ~2× recall
   tax (0.39–0.50 vs 0.99) at 40 rows/author. The one live scientific question this thread
   opens: does widening the slices close that gap (the storage-vs-recall curve toward
   SEA's 0.99 @ 32 MB)?
3. **For the selection framework: the last cell is measured, and the frame holds.** You
   cannot avoid paying for selection — either you isolate gradients (and selection
   reappears as per-author structure plus a router), or the information spreads into
   shared weights and is not deletable. Exact unlearning on TOFU remains with structural
   isolation + hard routing: routing+scaffold 0.7509 > SIFT 0.737 > SEA 0.711.

## 9. Incidental contributions

Reference-code findings: the sequence-ID-0 hash degeneracy (all-ones mask) and the
verbatim int64-overflow semantics of the published hash, both documented; a
CPU-generator-on-CUDA crash in the lab's IRP utility (latent in SISA's IRP mode), fixed
with a bit-equal port; the p_gen=0 partition remainder trap closed. Infrastructure now
reusable: a routed-mask serving arm in the lab's evaluator (`--memsinks_config`, with
membership-inference parity flags for future audits), the slice-content probe, the
deletion-bake tool, and a 22-gate CPU regression suite including the momentum-tail
data-provenance test.

## 10. Limitations

All numbers are single-seed (42), smoke-tier screenings; per lab protocol nothing here
graduates to a claim without seeds 43/44 and extended-tier evaluation, and forget_quality
is only comparable within a serving style (a documented KS artifact). This work is an
*extension* of MemSinks to fine-tuning with author-level IDs (the paper's own §6
suggestion) — it does not contradict the paper's from-scratch pretraining results in their
own regime. Membership-inference auditing was wired but not run (uninformative for the
dead arms; premature for the strict arm).

## 11. Provenance (everything re-derivable)

SLURM jobs: Round 1 **443146–443149** · Phase D **443549–443550** · strict **443562–443565**
(+443551–443554: failed smoke and its cancelled dependents) · strict2 **443939–443942** ·
strict2-e25 **444254–444257**. Configs and per-run script SHA-256s are recorded in the
dated entries under `~/log/memsinks/`; mask tables are content-hashed
(`golden_mask_sha256.json`); trained artifacts and result JSONs live under
`/storage2/jack/checkpoints/memsinks_tofu/` with aggregate CSVs via
`collect_results.py --smoke`; the CPU gate suite is `test_memsinks.py` (22 gates).

## 12. If the thread continues

- **Capacity ladder** (the live question): slice width 40 → 80 → 160 rows/author, or
  trainable-A-per-slice — trace storage-vs-recall from 0.39 @ 80 KB toward 0.99 @ 32 MB;
  then seeds, extended tier, and the MIA battery before any headline claim.
- Deferred: shared-capacity-starvation arm, non-strict e25 repetition arm,
  hashed/shuffled-ID/untied-dropout controls, the H6 MIA spectrum.
