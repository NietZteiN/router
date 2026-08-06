# Centered Merging & Negative-Anchored Isolation — the full story, from scratch

**Date:** 2026-07-16 · **Thread:** [merge_mechanism](../../log/merge_mechanism/README.md) ·
**Status:** both experiments complete, verdicts pre-registered
**Entries:** [centered design](../../log/merge_mechanism/2026-07-15_centered-merge-design.md) /
[centered results](../../log/merge_mechanism/2026-07-16_centered-merge-results.md) ·
[key-firing design](../../log/merge_mechanism/2026-07-15_key-firing-design.md) /
[key-firing results](../../log/merge_mechanism/2026-07-15_key-firing-results.md) ·
[anchor design](../../log/merge_mechanism/2026-07-15_negative-anchor-design.md) /
[anchor results](../../log/merge_mechanism/2026-07-16_negative-anchor-pilot-results.md)

> **How to read this.** Nothing is assumed beyond basic ML. §1 builds the setting and the
> vocabulary; §2 explains why these two experiments exist; §3 is centered merging end-to-end;
> §4 is the key-firing measurement and the negative-anchoring intervention end-to-end; §5 is
> what the pair means together; §6 is a full reproducibility appendix (commands, configs,
> SLURM job IDs, file paths). Every number is copied from a result JSON or log entry on disk.

---

## TL;DR

- **Centered merging works — up to a point, and the point is now measured.** Merging N
  per-author LoRA adapters normally destroys their facts almost immediately (half the signal
  gone by N≈3). Re-combining the *same* adapters with the centered rule
  `M = ΣΔᵢ − (N−1)·S` (S = an estimate of what all adapters share) moves that collapse knee
  to **N≈64**, keeps **71–82% of the extractable per-author signal at N=8** (the plain mean
  keeps 35%), and holds global model utility at or above the mean-merge level **up to
  N≈128** with *falling* perplexity — the norm-explosion failure mode is fully closed. At
  N=200 it finally loses to residual crosstalk. Deletion stays exact. Zero training cost.
- **A pre-registration catch mattered:** the formula as originally proposed
  (ΣΔᵢ − (k−1)Δ̄ with Δ̄ = the mean of the *same* adapters) is an algebraic identity for the
  plain mean — it would have silently reproduced a known-dead result. Two non-degenerate
  estimators of S were designed and run instead.
- **Negative anchoring fails cleanly, and the failure is informative.** Per-author adapters
  fire on *everything* (on-author vs off-author output ratio ≈ 1.1; they even fire at ~90%
  strength on random public text). Training them with an explicit "do not fire on public
  text" penalty produces **uniform shrinkage instead of selectivity** at every penalty
  weight — the ratio never moves while recall degrades (1.00 → 0.53 at the strongest
  penalty). Conclusion: input-conditioned selection **cannot be trained into a LoRA adapter
  with output-norm negatives**; it has to live outside the weights (router, mask) or in a
  different storage substrate.
- Together these are the mechanism paper's two causal interventions: composition-side
  (bounded positive) and training-side (clean negative).

---

## 1. The setting, from scratch

### 1.1 The benchmark and the goal

**TOFU** is a machine-unlearning benchmark: 200 *fictitious* authors × 20 question–answer
pairs each (4,000 QA pairs total). A model is fine-tuned on all authors; a deletion request
("forget author X") must then be honored. Because the authors are fictitious, anything the
model can say about them was learned in the fine-tune — leakage is measurable.

This project's approach is **exact unlearning by structure**: train **one LoRA adapter per
author** on a frozen base model (Llama-2-7B-chat), so deleting an author is a *structural*
operation — drop (or recompute without) that author's adapter file. The result provably
equals never having trained on them. The open question is **composition**: how do you serve
200 adapters as one system? Routing (pick the right adapter per query) works but needs a
router; **merging** (combine all deltas into one weight set, no serving logic) is the
attractive deployment — and is what these experiments interrogate.

### 1.2 A LoRA adapter is a key–value memory

A LoRA adapter adds a low-rank delta to each targeted weight matrix: the layer computes
`W·h + s·B·A·h`, where `A` (r×d) holds **read keys** (each row is dotted against the hidden
state h), `B` (d×r) holds **write values** (columns added into the residual stream weighted
by the key matches), and `s` is a fixed scaling. The effective delta is `Δ = s·B·A`. Here:
rank r=32, scaling s = α/√r (rsLoRA, α=64), applied to q/k/v/o/up/down projections in all
32 layers (192 modules). One adapter ≈ 59M parameters.

### 1.3 The metrics used in this report

- **`model_utility` (mu)** — TOFU's headline utility score: the harmonic mean of nine
  components (answer probability, ROUGE-L recall, truth-ratio × three buckets: retained
  authors, real authors, world facts). Reference points on this base: **base model 0.426**,
  good joint fine-tune **0.756**, routed serving **0.751**, retain-oracle 0.563. All
  evaluations here use the "smoke" caps (the fast tier; ~4 min/eval on an A40).
- **Subset-conditioned recall (`retain_prob` under `--retain_author_ids`)** — the mean
  answer probability P(answer|question) restricted to *the N merged authors' own rows*:
  "did the composition keep what it was trained on?" The standard mu is structurally blind
  to per-author collapse (it averages over mostly-unmerged authors — proven in Exp-5b), so
  this is the primary readout. Anchors on identical rows: a single isolated adapter scores
  **0.3991** on its own author (the ceiling for these e5-recipe experts), the base model
  **0.1703** (floor), a joint fine-tune **0.9237**. "**Signal %**" below =
  (value − 0.1703) / (0.3991 − 0.1703).
- **Own-author ROUGE (`forget_rouge` via `--eval_shard_id`)** — per-probe generation recall
  on one specific author's questions; compared against the same adapter served alone
  (**iso**, mean 0.4895 across the five probe authors). The *drop* iso→merged quantifies
  interference per author.
- **`retain_ppl`** — perplexity on retained-author text; the canary for the norm-explosion
  failure mode (a naive sum of 10 adapters historically explodes this to ~10⁴).

### 1.4 What was already known (the two dead regimes)

Decompose each author's delta as Δᵢ = S + Rᵢ: a **shared component** S (TOFU answer style,
QA formatting — measured: the 200 adapters' write-subspaces concentrate in a common rank-16
basis at 92× chance) plus an **idiosyncratic residual** Rᵢ (that author's actual facts).
Composing k adapters by plain addition then fails two ways:

| Regime | Rule | Shared enters at | Facts enter at | Outcome (measured) |
|---|---|---|---|---|
| Unit sum | ΣΔᵢ | k× (coherent stack) | 1× | norm explosion: mu 0.0, ppl ~10⁴ at k=10 |
| Mean | (1/k)ΣΔᵢ | 1× | **1/k (diluted)** | mu flat 0.459 ∀N; per-author recall half-dead by N≈3, ~85% dead by N≈8 (Exp-5) |

Even *perfectly trained* experts don't escape: rebuilt from adapters that individually store
~100% of their author (25 optimizer steps, "e25"), the mean-merge collapse knee is the same
N≈3 (experiment H8). The obvious third regime — **shared at 1×, facts at 1×** — had never
been run. That is centered merging (§3). And a candidate *cause* — adapters trained in
isolation never see off-author negatives, so their read keys may fire on everything
("lazy keys") — had never been functionally measured. That is §4.

---

## 2. The two experiments and their logic

1. **Exp-6, centered merging** (composition-side intervention): build the third regime
   post-hoc from the existing adapters. If facts survive to much larger N, dilution was the
   killer; where they re-collapse measures the remaining term — inter-residual crosstalk.
   Deletion must remain exact: the shared estimate S must be a deterministic function of
   the adapter files only, so "delete author j" = recompute S without Δⱼ and re-merge.
2. **Exp-7 → §6.3, key-firing then negative anchoring** (training-side intervention):
   first *measure* whether keys are lazy (Exp-7, cheap, no training). If they are, retrain
   adapters with the missing negatives — a penalty pushing the adapter's output to zero on
   public, author-independent text — hoping to sharpen keys into soft self-gates
   (negative-anchored isolation). The anchor set contains no author data, so exactness
   survives. A pre-registered gate connected them: anchoring runs **only if** keys measure
   lazy.

Both experiments were pre-registered (hypotheses, confirm/refute bars, λ-selection rules)
in dated design entries *before* any GPU job, per repo protocol.

---

## 3. Experiment A — Centered merging (Exp-6)

### 3.1 The degeneracy catch (why the literal proposal was not run)

The original proposal: "compute the mean adapter Δ̄ = (1/k)ΣΔᵢ and merge as
**ΣΔᵢ − (k−1)·Δ̄** — the shared component enters once, each residual at full strength."
But with Δ̄ defined over the same k adapters this is an identity:

```
ΣΔᵢ − (k−1)·Δ̄  =  k·Δ̄ − (k−1)·Δ̄  =  Δ̄        (exactly the plain mean)
```

Equivalently: residuals about their own mean sum to zero, so "add all residuals at full
strength" adds nothing. Running it literally would have burned ~a GPU-day reproducing
Exp-5. The intent — `M = S + Σᵢ(Δᵢ − S)` — is only a new regime when **S is not the exact
subset mean**. Two exactness-preserving estimators were designed (both are pure functions
of adapter files ⇒ deletion = recompute without author j, deterministic linear algebra, no
training):

- **`cpool` (pool-mean centering):** S = the mean of ALL 199 pool adapters, while merging
  only a subset of N. Parameter-free and closest to the original intent. Boundary: as the
  subset approaches the pool it degenerates back to the mean (at subset=pool exactly), and
  the estimator self-contaminates in proportion N/pool — so it was capped at N ≤ 64 and
  pre-flagged as the non-deployable variant.
- **`cr16` (low-rank centering):** S = the per-module rank-ρ truncated SVD of the subset
  mean, ρ=16 (chosen from the measured rank-16 shared write-basis). Non-degenerate at
  every N — this is the deployable form. The ρ dial interpolates the two dead regimes:
  ρ=0 gives the naive sum, ρ=full gives the mean. Known cost: whatever shared energy lies
  outside rank 16 is amplified ~N× — deliberately, because *where* that re-collapses the
  curve is the crosstalk measurement.

All three identities (literal formula ≡ mean; ρ=0 ≡ sum; ρ=full ≡ mean; pool=subset ≡ mean)
are **encoded as CPU tests** (`test_merge_subset.py`, identities hold to ≤1e-4 relative
error on a tiny-model fixture), together with closed-form correctness of both estimators
against independently computed dense math, a PeftModel serving round-trip, and
recompute-determinism (the deletion contract).

### 3.2 Implementation (what actually runs)

Everything reuses the Exp-5 "materialize-then-eval" harness (`merge_subset.py` +
`submit_nmerge.sh` + `analyze_nmerge.py`), extended with the two methods:

- **Factor-space math, never dense.** A weighted combination of adapters is a concatenation
  of their (A, B) factors with weights folded into the B side, so
  `B_cat·A_cat = Σᵢ wᵢ·sᵢ·Bᵢ·Aᵢ` exactly. `cpool` = subset factors at weight 1.0 + all 199
  pool factors at weight −(N−1)/199, one cat. `cr16` = per module, compress the subset-mean
  cat to rank 16 by exact factored SVD (QR of both stacks + SVD of the small core), then
  cat subset factors (weight 1) with the rank-16 factors at weight −(N−1).
- **Materialization.** Each merge is written to disk as one ordinary PEFT adapter with
  scaling forced to 1.0, so the standard evaluator serves it via `--preloaded_adapter` with
  no special code. Compression to rank 1024 where the cat is too large to serve (`cpool`
  always — its cat rank is 32·(N+199); `cr16` at N ∈ {64, 128, 200}); retained-energy
  diagnostics are recorded per merge.
- **Ladder & design.** Subsets are nested by a fixed permutation (seed 42) of authors
  0–198, so the five probe authors (82, 15, 111, 177, 76) are members of every subset and
  are tracked longitudinally. Ladder N ∈ {2,3,4,6,8,12,16,20,32,64} (cpool) and
  {…,128,200} (cr16). Per label: 5 probe evaluations (own-author recall + standard mu) +
  1 subset-conditioned evaluation (recall on all N merged authors' rows). Reference rows
  (isolated probe adapters, base/fine-tune/oracle anchors) are bit-identical to the Exp-5
  campaign's, so their result files were copied rather than recomputed.
- **Cost.** 23 CPU merge tasks (no GPU; the largest ~minutes–hours each) + ~125 one-GPU
  smoke evaluations (~4 min each), all inside the global 4-concurrent-GPU cap.

### 3.3 Pre-registered hypotheses and bars

| # | Claim | CONFIRM bar | REFUTE bar |
|---|---|---|---|
| H-cent-1 | recall rescue at N=8 | subset retain_prob ≥ 0.50 (2× the mean's 0.250) | within ~0.05 of the mean curve |
| H-cent-2 | standard mu survives (the "does utility survive" headline) | mu ≥ 0.459 − 0.02 with no ppl explosion | mu → base or ppl blow-up |
| H-cent-3 | crosstalk re-collapse at measurable N\* | either N\* located on the ladder or survival to 200 — both informative | — |
| H-cent-4 | centering removes the mean's "shared-alignment protects" effect (Exp-5 found r = −0.675 between an adapter's overlap with the shared basis and its recall drop) | correlation ≥ −0.2 | ≤ −0.5 persists |

### 3.4 Results

**Subset-conditioned recall** (primary; anchors: iso 0.399 / base 0.170 / joint-FT 0.924):

| N | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 20 | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean (Exp-5 ref) | .350 | .289 | .306 | .265 | .250 | .227 | .228 | .246 | — | — | — | ~.225 |
| **cpool** | **.428** | .381 | .415 | .380 | **.358** | .329 | .299 | .335 | .252 | .173⚠ | — | — |
| **cr16** | .380 | .338 | .369 | .342 | **.334** | .326 | .312 | .334 | .302 | **.281** | .240 | .211 |

In signal-% terms (cr16): 92 → 71 (N=8) → 72 (N=20) → 58 (N=32) → **48 (N=64)** →
31 (N=128) → **18 (N=200)**. The mean regime: half the signal gone by N=3, 35% left at N=8,
plateau ≈ 24%. So the **half-signal knee moves N≈3 → N≈64**, and the curves cross back at
N≈150: at full scale centered is *worse* than the mean.

**Standard model utility** (probe rows; mean regime = flat 0.459 ± 0.002; bar 0.439):

| N | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 20 | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpool | .455 | .460 | .462 | .457 | .444 | .452 | .448 | .437 | .421 | .356⚠ | — | — |
| cr16 | .461 | .465 | .464 | .465 | .464 | .466 | .469 | .471 | .465 | .460 | .442 | **.412** |

cr16 **retain_ppl falls monotonically 7.7 → 6.0** across the whole ladder — no norm
overshoot anywhere (the λ-sweep historically exploded this to 1.8M; centering closes that
channel completely). cpool's ppl rises to 19 at N=64 (see caveats).

**Per-probe own-author ROUGE drop** (iso→merged; mean regime: 0.073 at N=8, then flat):
cr16 0.030 (N=8) → 0.043 (N=32) → 0.060 (N=128) → **0.120 (N=200)** — at N=200 the drop
*exceeds* the extractable range 0.086, i.e. recall is pushed **below the never-trained
base**: crosstalk at scale actively harms, it doesn't just wash out. cpool at N=2–4 sits
*above* the isolated adapter (drop −0.026 at N=2 — a mild ensemble gain from full-strength
residuals).

**H-cent-4 correlation** (same seed-42 subsets ⇒ Exp-5's geometry reused): r = **−0.40**
(cr16, p=0.025) / **−0.44** (cpool, p=0.029) vs the mean regime's −0.675.

### 3.5 Verdicts

- **H-cent-1: inconclusive by the pre-registered bars, directionally supported.** +43%/+34%
  relative lift at N=8 — real and consistent at every N, but short of the 2× bar.
- **H-cent-2: SUPPORTED to N≈128, REFUTED at N=200** (cr16). *The direct answer to "does
  model utility survive": yes — at mean-regime levels with falling perplexity — up to
  N≈128 merged authors; not at full 200.* cpool survives only to N≈16–20.
- **H-cent-3: SUPPORTED and measured.** N\* ≈ 64 (half-signal), crossover ≈ 150, below-base
  by 200. With blowup closed (ppl falls) and dilution removed (residuals at 1×), the
  isolated, measured killer at scale is **inter-residual crosstalk** — and it ends worse
  than dilution.
- **H-cent-4: between the bars.** Protection weakened ~40% (−0.675 → −0.40/−0.44), not
  eliminated — consistent with the 1× shared term still protecting shared-aligned writes.

### 3.6 Caveats and deviations (all logged in the entries)

- The literal §6.1 formula was **rejected pre-run** as degenerate (§3.1) — recorded as a
  design catch, and proven as a regression test.
- cr16 exact N=64 (cat rank 2064, fp32) deterministically OOMs at adapter-load next to the
  7B base on a 44.5 GiB A40 → N=64 is served from the rank-1024 SVD artifact instead. The
  substitution is validated by the e5 campaign's svd-vs-exact acceptance pair (|Δmu| =
  0.0007 at N=64). Config + CLAUDE.md invariants updated.
- cpool's rank-1024 compression loses energy at large N (min-slot retained energy 0.999 at
  N=2 → 0.815 at N=64): treat cpool N ≥ 32 as compression-confounded *on top of* the
  pre-registered estimator self-contamination. The large-N story rests on cr16.
- cr16's center-energy diagnostic (0.62 → 0.11, N=2→200) confirms rank-16 captures only the
  tip of the mean — by design; a ρ-sweep is the natural follow-up.
- One in-flight eval task was cancelled during a brief GPU-cap correction and re-run
  (identical row, self-skipping harness). Single seed (42) throughout — multi-seed passes
  are mandatory before any paper claim.

---

## 4. Experiment B — Key-firing and negative anchoring (Exp-7 → §6.3)

### 4.1 The question: are the read keys lazy?

Earlier geometry (Exp-1) showed the 200 adapters' read subspaces (row(A)) are mutually
orthogonal — indistinguishable from random. But orthogonal keys can still all *fire* on the
same input: geometry says where keys point, not what they respond to. The lazy-keys
hypothesis: adapters trained only on their own author have **no off-author negatives**, so
a generic "fire on anything QA-shaped" key achieves zero training loss — and in any
additive composition, every adapter's values then fire on every query: crosstalk by
construction. This had never been measured functionally.

### 4.2 The measurement (Exp-7)

For each adapter i and each input text, capture the hidden states h that the *frozen base
model* produces at all 192 LoRA target modules (forward pre-hooks; one forward serves all
200 adapters because h is adapter-independent), and compute per token:

- the read activation ‖Aᵢh‖, and
- the full output norm ‖sᵢBᵢAᵢh‖ — computed without ever materializing the output
  dimension via the Gram trick ‖Bz‖² = zᵀ(BᵀB)z (per-module Gram matrices precomputed;
  proven equal to the dense computation at 3e-8 in the CPU gate).

Inputs: 5 seeded questions per author × 200 authors (eval-prompt format,
`Question: {q}\nAnswer:`), plus three out-of-domain sets ×100 (TOFU world-facts,
real-authors, and public Alpaca instructions). **Selectivity ratio** per adapter = mean
on-author firing / mean off-author firing. Pre-registered gate: median ratio < 2.0 ⇒
**LAZY** ⇒ anchoring proceeds; ≥ 5.0 ⇒ SELECTIVE ⇒ anchoring is predicted useless and does
not run. Measured on both adapter generations: e5 (~5 optimizer steps, 200 adapters) and
e25 (~25 steps, near-perfect recall, 20 adapters). Runtime: ~3 min per set on one A40.

**Results:**

| | e5 (n=200) | e25 (n=20) |
|---|---|---|
| median on/off ‖sBAh‖ (gate) | **1.102** | **1.110** |
| adapters below 2.0 | **100%** | **100%** |
| read-side ‖Ah‖ ratio | 1.012 | 1.044 |
| mean absolute firing, on / off | 0.184 / 0.166 | **0.633 / 0.567** |
| OOD firing (Alpaca) vs on-author | 87% | 83% |

**Verdicts:** keys are **LAZY, decisively** (H-key-1 confirmed — the gate fired GO for
anchoring). The read keys carry essentially *zero* discrimination (ratio ≈ 1.01); adapters
fire at ~90% strength on arbitrary public text. And the e5/e25 contrast refuted the
"training sharpens keys" hypothesis: **25× more training makes adapters fire ~3.4× harder
on everything, not more selectively** — which is precisely the firing-side mechanism for
the earlier H8 finding that stronger experts merge *worse* (bigger unselective outputs =
bigger collisions).

### 4.3 The intervention: negative-anchored isolation (§6.3)

**Idea:** supply the missing negatives at training time. Train each per-author adapter with
the normal SFT loss on its author *plus* a penalty on public text:

```
loss = CE(author batch) + λ · mean over (modules, anchor tokens) of ‖sᵢBᵢAᵢh‖²
```

with the anchor batch cycling deterministically through 2,000 seeded public Alpaca
instruction–response pairs (formatted with the same text schema as training). Because the
anchor set is public and seeded, training remains a pure deterministic function of (author
shard, anchor set, seeds) — the exact-deletion certificate is untouched. Implementation: an
`SFTTrainer` subclass adding one extra forward per step over the anchor batch, adapter
outputs captured by hooks on the B matrices (scaling applied exactly once, fp32, pad-
masked, differentiable into both A and B). Flag-free behavior is bit-identical to the
frozen recipe — proven, along with penalty ≡ dense closed form and gradient flow, in the
CPU gate (`test_train_anchor.py`).

**Pilot design (pre-registered):** 5 probe authors × λ ∈ {1, 10, 100} at the strong-expert
recipe (25 steps) = 15 one-GPU trainings (~45 s each). λ scale reasoning: the penalty at
initialization ≈ 0.4 (e25 firing² per module-token) vs CE ≈ 2–3, so {1, 10, 100} spans
gentle → dominant. Selection rule: the largest λ with median selectivity ≥ 5 AND own-author
recall ≥ 0.98; if none reaches ≥ 5, record the shortfall and do not proceed to the full
merge ladder (H-anchor-2). Readouts: the *same* key-firing harness re-run per λ pool
(ratios directly comparable to §4.2), plus per-author isolated recall evaluations.

**Results:**

| arm | gate median on/off | verdict | mean on-firing | own-author ROUGE | own ppl | single-adapter mu |
|---|---|---|---|---|---|---|
| e25 baseline | 1.110 | LAZY | 0.633 | 1.000 | 1.06 | 0.388 |
| λ=1 | 1.150 | LAZY | 0.279 | **0.997** | 1.07 | 0.428 |
| λ=10 | 1.123 | LAZY | 0.131 | 0.924 | 1.19 | 0.453 |
| λ=100 | 1.112 | LAZY | 0.057 | **0.525** | 2.61 | 0.462 |

**Verdict: H-anchor-1 REFUTED across the entire λ range.** The optimizer satisfies the
penalty by **uniform global shrinkage** — on- and off-author firing shrink in lockstep
(0.63 → 0.06 while the ratio never leaves 1.08–1.17), and recall pays the price
monotonically. There is no λ that buys *any* selectivity, at any recall cost. Per the
pre-registered decision tree, the anchored merge ladder (H-anchor-2) was not run.

**Why it fails (the mechanism reading):** §4.2 showed the inputs h that adapters read are
essentially indistinguishable across on-author, off-author, and out-of-domain text (read
ratio 1.01). If the inputs can't be separated, no objective on the *output norm* can
separate the responses — the only degree of freedom left is overall scale, and that is
exactly what the optimizer used. **Self-gating cannot be trained into a LoRA adapter this
way**; input-conditioned selection must live outside the adapter (router, mask) or in a
storage substrate whose keys are content-derived (the MEMIT-family direction). This also
retires the "self-gating experts" candidate for sealing the post-deletion router leak.

**Bonus finding:** the penalty is a clean **collateral-damage dial**. A heavily-memorized
expert normally damages the base model's general knowledge (single-adapter mu 0.388, world
facts degraded); anchoring shrinks that damage (mu → 0.462, world-facts probability → 0.75)
— and at λ=1 this comes *free* (recall 0.997). Useless for merging; potentially useful as a
regularizer for *routed* serving, where selection is external anyway.

---

## 5. What the two results mean together

The project's organizing frame: every composition that preserves per-author facts performs
**input-conditioned selection** somewhere, and the design question is where that selection
lives (router / mask / inside-the-weights / nowhere). These two experiments fill in the
"nowhere" and "trainable-into-the-adapter" cells with causal evidence:

1. **Ungated addition without selection** now has its best-possible point measured:
   centering removes both classical failure channels (blowup — perplexity *falls*;
   dilution — residuals at 1×) and still collapses, at a now-measured N\*≈64, from
   inter-residual crosstalk alone. The interference curve *can* be moved ~20× in N at zero
   training cost — worth having (micro-merge tiers widen; k≈64–128 deployments become
   viable at mean-level utility) — but no ungated rule approaches routing (0.751).
2. **Selection cannot be trained into the adapters** by supplying the missing negatives:
   the inputs are inseparable at the adapter's read interface, so the write-norm penalty
   degenerates to uniform shrinkage. The lazy keys are a *symptom* of where LoRA reads
   live, not a training artifact — more training amplifies rather than sharpens
   (the e5/e25 result), directly explaining why strong experts merge worse.
3. Paper shape: observation (Exp-5/5b/H8 collapse + metric blindness) → mechanism
   (write-side collision + dilution + lazy keys, Exp-1/7) → **causal interventions**
   (centering: bounded positive; anchoring: clean negative) → boundary (selection must be
   external or content-keyed). Open follow-ups, in priority order: ρ-sweep {8, 32, 64} at
   pivotal N; the e25 strong-expert centered wave (merge artifacts already on disk);
   centered + TIES sign-election on residuals; the MEMIT bet inherits the
   "selection-in-weights" cell.

---

## 6. Reproducibility appendix

**Environment.** Python `/home/jack/anaconda3/envs/test-env/bin/python`; base model
`meta-llama/Llama-2-7B-chat-hf` (bf16); nodes sprint1–3, 1 GPU/job, global cap 4
concurrent GPUs; HF_HOME=/storage2/jack/data/huggingface. Repo: `~/tofu_sisa_lora`.
Single seed 42 throughout (subsets, question sampling, anchor sampling, training).

**Adapter pools (inputs).**
`/storage2/jack/checkpoints/tofu_sisa_lora/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4/shard_{0..199}`
(e5, frozen recipe r32/α64/lr1e-4/5ep) and `..._k200_r32_e25_lr1e4/shard_*` (e25 = 25
epochs, the 20 subset(42) authors).

**Centered merging.**
- Config: `configs/nmerge_centered_7b.json` (post-hoc: `centered_lowrank.exact_max_n` 64→32
  after the OOM). Code: `merge_subset.py` (`merge_centered_pool`, `merge_centered_lowrank`,
  `_weighted_factor_cat`), labels `nmerge_{cpool,cr16}[_svd1024]_N{n}_s42`.
- CPU gate (run first): `python test_merge_subset.py` — includes the degeneracy identities.
- Pipeline: `python merge_subset.py plan --config C` → `bash submit_nmerge.sh C merge`
  (CPU array **443445**, 23/23 clean) → `bash submit_nmerge.sh C eval` (GPU arrays
  **443532** + strays **443925/444061**; %throttle sized to the live queue) →
  `OUT_PREFIX=reports/centered/nmerge bash submit_nmerge.sh C collect`.
- Outputs: merges + per-row JSONs under
  `/storage2/jack/checkpoints/tofu_sisa_lora/Llama-2-7B-chat-hf_nmerge_r32_centered/`
  (`merges/{label}/merge_meta.json` carries authors, weights formula, energy diagnostics,
  script sha256); assembled CSVs `reports/centered/nmerge_{mu,own_recall,subset_mu,overlap}.csv`.
  Reference-row JSONs copied from the e5 campaign (listed in `CLAUDE_SCRATCHPAD.md`).
- H-cent-4 reuses `reports/nmerge_overlap_s42.json` (Exp-5 geometry; same seed-42 subsets).

**Key-firing.**
- Code: `measure_key_firing.py` (Gram trick; script sha256-16 `729c09db9baa0ae5`); CPU gate
  `python test_measure_key_firing.py`; driver `bash submit_key_firing.sh {e5|e25}`.
- Jobs: e5 **443446**, e25 **443477** (~3 min each). Outputs
  `reports/key_firing_{e5,e25}.json` + `_matrices.npz` (full adapter × group matrices,
  8 aggregation keys: read/write × mean/last-token × attn/mlp/layer-terciles).

**Negative anchoring.**
- Code: `train_lora_shard.py --anchor_lambda λ` (+`--anchor_n 2000 --anchor_seed 42
  --anchor_batch_size 4 --anchor_source alpaca`; `AnchoredSFTTrainer`, `anchor_penalty`,
  `apply_anchor_to_loss`); flag-free behavior bit-identical (frozen-recipe invariant).
  CPU gate `python test_train_anchor.py`.
- Pilot driver: `bash submit_anchor_pilot.sh {train|keyfire|iso}`. Jobs: train array
  **443487** (15 tasks), keyfire **443523–443525**, iso array **443536**. Outputs: pools
  `..._k200_r32_e25_anch{1,10,100}_lr1e4/` (each `shard_meta.json` records the anchor
  fields), `reports/key_firing_e25_anch{λ}.json`, iso rows under each pool's
  `results/smoke/`.

**Known deviations (also §3.6):** cr16 exact-N64 OOM → svd-served; one cap-incident task
re-run; the shared `reports/nmerge_*` CSV prefix was briefly clobbered by collect and
restored from the (untouched) e5 JSONs — `submit_nmerge.sh collect` now takes `OUT_PREFIX`.
