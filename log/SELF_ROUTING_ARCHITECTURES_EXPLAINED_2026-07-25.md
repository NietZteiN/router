# Self-routing architectures, explained from scratch

**Per-author MLP banks (`sepmlp` / MUSR) and the block transcoder (`blocktc`) — why they
exist, exactly how they work, how to rebuild them, and what we actually measured.**

Written 2026-07-25. Companion to [PATHS_FORWARD_2026-07-13.md](PATHS_FORWARD_2026-07-13.md)
(the strategic frame) and [EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md](EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)
(the literature map).

---

## §0 How to read this

Two threads in this repo attack the same target with the same core machinery. This document
covers both as one lineage, because `blocktc` is `sepmlp`'s successor and reusing half its
code:

- **`sepmlp` / MUSR** — a separate ReLU-gated bottleneck MLP **per author per layer**.
  Code [sepmlp_tofu/](../sepmlp_tofu/), narrative [log/sepmlp/](sepmlp/README.md).
- **`blocktc`** — ONE wide block-structured **transcoder**: a single encoder read at layer 9,
  per-feature decoders writing at layers 9/10/11. Code [blocktc_tofu/](../blocktc_tofu/),
  narrative [log/blocktc/](blocktc/README.md).

**Reading paths.** Orientation → §1, §2, §5. Rebuild → §3, §4, §6. Decide what to run next →
§7, §8. Every house term is defined in §9.

**Provenance convention** (borrowed from [MUSR_EVIDENCE_FULL_REPORT_2026-07-22.md](../sepmlp_tofu/reports/MUSR_EVIDENCE_FULL_REPORT_2026-07-22.md)):

- **[V]** — verified firsthand while writing this doc, from the cited log entry, `meta.json`,
  or result JSON.
- **[S]** — taken from a thread README or report summary; traceable via the cited path, not
  re-opened line-by-line here.

> ### ⚠️ Two warnings before any number in this document
>
> **1. Never mix tracks.** This repo runs two incompatible tokenization universes. The
> **OU chat-template track** (open-unlearning schema; `sepmlp`, `blocktc`, `memadapt`) and the
> **plain `Question:/Answer:` track** (`sift_masks`, routing, merging) produce numbers that
> must never share a table. A `model_utility` of 0.737 on the plain track and 0.465 on the OU
> track are not comparable quantities. Both projects list this as trap 12. Every table below
> names its track.
>
> **2. One headline result here is not yet in the log.** The `sepmlp` reverse-engineering
> wave-1 sweep landed 2026-07-23, after the last dated entry (2026-07-22). It is reported in
> §3.11 from the on-disk artifacts and is flagged as logging debt in §8.

---

## §1 The problem, from scratch

### 1.1 What we are trying to do

**Machine unlearning**: a model was trained on data belonging to many owners; one owner
invokes a right to erasure; the deployed model must stop reflecting their data.

- **Exact unlearning** — the served model is *identical to one that was never trained on that
  data*. The gold standard is retraining from scratch without it, which is unaffordable.
- **Approximate unlearning** — gradient ascent, gradient difference, KL, IDK-tuning. Cheap,
  and demonstrably leaky: on the served post-deletion composition, approximate methods showed
  MIA AUC **0.74–0.82** against an oracle floor of **0.379** [S,
  [deletion_audit/2026-07-06](deletion_audit/2026-07-06_composed-mia-results.md)].

The project goal, from [CLAUDE.md](../CLAUDE.md) §3, is the exact end:

> Deletion must be a cheap, clean, and deterministic O(1)-style operation — conceptually
> "remove or drop the specific module/slice that holds the targeted data" — rather than
> expensive, approximate, or stochastic weight surgery.

### 1.2 Why deletion is hard at all

Ordinary fine-tuning is the enemy of deletion. Every gradient step from every example updates
*every* weight. After training, there is no subset of parameters you can point at and say
"this is Alice's." Her information is a diffuse perturbation spread across the whole model,
entangled with everyone else's.

So every method in this repo does the same structural thing: **arrange, during training, for
each owner's contribution to live in an identifiable place**, so that deletion becomes an
index operation instead of an optimization problem. The methods differ only in *where that
place is* and *how the model finds it at inference time* — which is exactly the frame in §2.

### 1.3 TOFU, and the metrics

[TOFU](../papers/TOFU%20A%20Task%20of%20Fictitious%20Unlearning%20for%20LLMs.pdf) is the
benchmark: **200 fictitious authors × 20 QA rows** = 4,000 rows of synthetic biography. The
authors are invented, so nothing leaks in from pretraining — a clean substrate for measuring
whether a specific author's knowledge is present or absent.

- **Author id** = `row_index // 20` (`RECORDS_PER_AUTHOR = 20`), in `sepmlp_common.author_of_row`.
- **`forget10`** = 20 authors to delete. It *resolves to* authors 180–199, but the mapping is
  computed by an exact question+answer **text join** with a full-coverage assert —
  `verify_forget_author_mapping()` in [memadapt_tofu/data_tofu.py](../memadapt_tofu/data_tofu.py).
  Never positional. A positional shortcut silently mis-deletes when a split changes.
- **`holdout10` is sacred.** It is simultaneously the relearn control *and* the MIA
  non-member set, so a single training row on it poisons two evaluations at once. Both
  projects enforce this with a static-string CPU gate plus a runtime assert.

Metrics used below:

| Metric | Meaning | Track |
|---|---|---|
| `model_utility` (`mu`) | harmonic mean of 9 sub-metrics on retained + general knowledge | both |
| `forget_quality` (`fq`) | KS p-value of the served model's forget-set scores vs a retain-only oracle | both |
| **per-source ROUGE-L recall** | greedy generation, ROUGE-L recall against the author's own gold answers. **MUSR's primary metric** | OU |
| `tail` | count of authors below 0.95 recall — the metric that exposes a few starved authors hiding behind a healthy mean | OU |
| Agg / Mem / Priv | the OU Table-1 composition (aggregate / memorization / privacy) | OU |

**OU-track anchors** [S]: MemAdapt Agg 0.869 / Mem 0.630 / Priv 0.917 · Retrained (oracle)
Agg 0.874 / Mem 0.590 / Priv 1.00 · TOFU base `mu` 0.281 · full-FT `mu` 0.599 · MIA oracle
floor AUC 0.379.

---

## §2 Why this shape — the selection table

### 2.1 The reframe

Everything in this repo that successfully preserves per-author facts performs, somewhere, an
**input-conditioned selection**: something looks at the query and decides which stored
knowledge gets amplified. Everything that lacks such a step collapses to base-model
performance. From [PATHS_FORWARD](PATHS_FORWARD_2026-07-13.md) §5, the question is not
"merge or route" but:

> **Where does the selection mechanism live, and what does that location cost you at deletion
> time and at leak time?**

| Where selection lives | System | Utility (plain track) | Deletion | Leak surface |
|---|---|---|---|---|
| **Nowhere** — ungated sum/mean | all plain merges | 0.42–0.49 ≈ base; per-author facts dead by N≈8 | recomputable, not serving-inert | nothing to leak — facts already destroyed |
| **Serving-time router** | routing_scaffold, legonet, sea, ramole | **0.7509** scaffolded, **0.8236** oracle | byte-identical O(1) drop | **the router** |
| **Per-task mask over a merged sum** | sift_masks 0.737, clamu 0.647 | 0.55–0.737 | bitwise-exact subtract | mask selection needs task identity — routing in disguise |
| **Inside the weights** | ← **`sepmlp` and `blocktc` live here** | see §3, §4 (OU track) | drop the owner's slices | no router to interrogate |

All figures in this table are **plain track** [S]. The bottom row is what both of our
architectures are built to fill.

### 2.2 Why the merge row is closed

At 7B with k=200 one-author-per-shard adapters, roughly **20 different merge operators** land
in **0.419–0.451** against base 0.426 and full-FT 0.756 — regmean 0.4197, fisher 0.4200,
ties 0.4201, dare_ties 0.4201, della_ties 0.4193, tree_root_slerp 0.4253, subtract_orth
0.4272, breadcrumbs 0.4338/0.4447, tsv 0.4366, linear 0.4503, lorahub 0.4505 [V,
[CLAUDE_SCRATCHPAD.md](../CLAUDE_SCRATCHPAD.md) 2026-07-24]. A separate post-hoc sparsify
grid (DARE / top-q / hash, on the r32 N-ladder — a different arm, so not the same range)
plateaus at 0.454–0.462, and the naive weight-1.0 `*sum` variants collapse outright
(0.454 → 0.000 as N grows) [V, same source, M(a)].

Nothing clears 0.55. Ungated composition is not a tuning problem; it is closed.

### 2.3 Why the router row is not the end of the story

Routing wins on utility, but the router itself becomes the leak surface. From the
`router_leak` campaign [S, [router_leak/README.md](router_leak/README.md)]:

- After an expert is deleted, its **orphaned queries** get routed to the most similar
  surviving expert (sim ratio 0.98), which answers in the deleted owner's voice.
- Under Mode-B replication, an embedding router surfaces the deleted owner's facts at
  **ρ = 0.833** where a hard key router shows ρ = 0.
- Dropping an expert shifts **72.7%** of *retain* queries to different destinations.
- Threshold-based abstention was refuted: catching 90% of orphans costs 58% false-abstain on
  retain traffic.
- The leak is mostly an *integrity* failure, not a disclosure one — **95.5% confabulation**.
- And it is invisible to the standard instruments: `forget_quality` is leak-blind, and
  composed-model MIA scores the leak *below* the oracle floor.

A method with **no router at all** has none of this surface. That is the second half of the
motivation.

### 2.4 The two priors both architectures must beat

These two refutations, already logged, explain nearly every design choice in §3 and §4. If
you skip them, the architectures look arbitrary.

**Prior 1 — self-gating cannot be trained into a LoRA.**
The `merge_mechanism` negative-anchor pilot trained per-author LoRAs with a localization
penalty at λ ∈ {1, 10, 100}. On/off selectivity stayed pinned at **1.11** at every λ (100%
LAZY, per-adapter ratios never leaving [1.08, 1.17]) while recall collapsed to 0.525 at
λ=100 [S, [merge_mechanism/2026-07-16](merge_mechanism/2026-07-16_negative-anchor-pilot-results.md)].
The diagnosis: a LoRA is a *linear* map into a *shared* low-rank subspace. It has no
mechanism to output exactly zero on foreign input, so no penalty can teach it to.

⇒ **Design consequences:** (a) authors must be *architecturally disconnected*, not sharing a
subspace; (b) the branch must be *nonlinear*, so an exact-zero off-state is reachable.

**Prior 2 — always-on per-author deltas self-interfere.**
`memsinks` gave each author a disjoint slice of neurons in a shared LoRA delta. Two failures
[S, [memsinks/README.md](memsinks/README.md)]:

1. **All-on interference** — serving with every author's slice active gave `mu` **0.4373** vs
   a matched control's 0.6438. The 200 always-on deltas fight each other.
2. **Near-empty slices** — the per-author slices carried almost nothing
   (`slice_increment` 0.0133); shared capacity answered the author's own rows at 0.90 by
   itself. Deleting an author's slice was ≈ a no-op.

⇒ **Design consequences:** (c) the objective must actively *put content into* the owner's
slice and *suppress* everyone else's, and (d) the all-active-vs-own-only recall gap is a
first-class gate, not an afterthought.

Both architectures below are direct bets against these two priors. `sepmlp` bets on
disconnection + nonlinear gate + in-domain negatives; `blocktc` keeps those bets and adds a
cost reduction and a stronger exactness claim.

---

## §3 Idea A — an MLP per author (`sepmlp` / MUSR)

`sepmlp` is our implementation of a method described by **Vincent Hanke**; the paper is
**MUSR — "Modular Unlearning via Self-Routing"**. The PDF is not on disk; conformance was
audited against its Eq 1–5 and found faithful, with two deviations found and fixed (§3.10).

### 3.1 The idea in one paragraph

Freeze the base model. At every decoder layer, add a bank of small bottleneck MLPs — one
branch per author. During training, route the language-modelling gradient so that a
sequence written by author *k* updates **only branch *k***, while every other author's data
acts as a **negative example** that pushes their branches' outputs toward zero on this input.
At inference, run with **all branches active and no router**: the gates must recognize their
own author's text and stay silent on everyone else's. Deleting an author is then physically
removing that author's slices — O(1), deterministic, and verifiable by reading the stored
parameters.

The gamble is entirely in the middle step. If the gates do not self-route, the method is a
pile of interfering adapters (prior 2). If they do, you get routing's selectivity with no
router to leak.

### 3.2 Architecture

At **all 16 decoder layers**, `layer.mlp` is wrapped:

```
layer.mlp(x)  →  layer.mlp(x) + bank(x, state)
```

The bank is a grouped, block-diagonal, ReLU-gated bottleneck of width `D = 32` per author.
For author *a*:

```
branch_a(x) = W_down[:, a] @ ( ReLU(W_gate[a]·x + b_gate[a])  ⊙  (W_up[a]·x) )
                              └──────── the "detectors" ────┘   └─ the value ─┘
```

with `⊙` elementwise. Stored **grouped**, so one matmul pair serves all K authors:

| Tensor | Shape (K authors, hidden H=2048, width D=32) |
|---|---|
| `W_gate` | `(K·32, 2048)` |
| `b_gate` | `(K·32,)` |
| `W_up`   | `(K·32, 2048)` |
| `W_down` | `(2048, K·32)` |

Forward, per layer: `g = W_gate·x + b_gate`; `u = W_up·x`; `act = ReLU(g) ⊙ u` reshaped to
`(B, T, K, 32)`; output `= W_down @ act.reshape(B, T, K·32)`.

**Why this exact shape.** Three properties are load-bearing:

1. **Grouped for speed, block-diagonal for isolation.** The two up-matmuls and one
   down-matmul are dense and serve all K authors at once, but author *a*'s contribution is a
   function of *only* author *a*'s four slices — ReLU is elementwise, and the down-matmul
   decomposes as a sum over per-author column blocks. Authors are **architecturally
   disconnected** even though the arithmetic is dense. This answers prior 1(a).
2. **The gate is nonlinear.** `ReLU(g) ⊙ u` can be **exactly zero** — not small, zero — when
   the detectors are driven below threshold. A linear adapter cannot do this at any penalty
   weight; that is precisely why the LoRA anchor sits at 1.11. This answers prior 1(b).
   (`gate_act: "silu"` is retained as a variant arm; ReLU is the spec.)
3. **The per-unit gate bias `b_gate`** gives each detector its own threshold, so "off" is a
   learnable offset rather than a property of the input distribution.

**Cost.** Per author per layer: `32×2048` (gate) + `32` (bias) + `32×2048` (up) +
`2048×32` (down) = **196,640**. Across 16 layers: **3,146,240 per author**. At K=200:
**629,248,000 added parameters — +50% on top of a 1.24B base**, 2.52 GB in fp32 [V, verified
arithmetic; matches the 6.3×10⁸ recorded in
[k200-g3-fail](sepmlp/2026-07-21_k200-g3-fail.md)]. This cost is what motivates §4.

### 3.3 Initialization

- **`W_down = 0`.** The bank is a **bitwise exact no-op at step 0** — the wrapped model is
  the frozen base, provably. Every rebuild should assert this as gate #1.
- `W_gate`, `W_up` ~ `N(0, 1/√2048)`, drawn from a **sha-seeded CPU generator** so init is
  reproducible across machines and devices.
- `b_gate = 0`.
- **Detector init** — before training, run a forward pre-pass over each author's *question*
  tokens, take the mean hidden state per author per layer, and orient that author's gate rows
  toward it. Cached to `detector_init.npz`. This is a *free, untrained* prior on
  "what my author's text looks like" — the closest thing to a router in the design, and it
  contains no learned parameters.

### 3.4 The training objective

```
total = L1 + 10·L2 + 50·L3 + 1·L4
```

Four terms. Each exists to defeat a specific failure mode.

---

#### L1 — routed language-modelling loss

Standard cross-entropy on the author's QA rows, but the gradient must reach **only that
author's branch**, while the forward *value* must include all branches (otherwise training
and serving see different models). This is achieved with the **detach identity**:

```python
act_own  = act * own_mask                            # zero out other authors' activations
out_grad = W_down @ act_own.reshape(B, T, K*D)       # masked path — carries the gradient
with torch.no_grad():
    out_real = W_down @ act.reshape(B, T, K*D)       # full path — carries the value

out = out_real.detach() + (out_grad - out_grad.detach())
```

Three things about this line, all of which a naive reimplementation gets wrong:

- **The value is bitwise identical to serving.** `(out_grad − out_grad.detach())` is
  elementwise exactly `t − t = 0` for finite floats, so `out == out_real` bit for bit.
- **The inner parentheses are load-bearing.** Written left-to-right as
  `(out_real + out_grad) - out_grad`, floating-point rounding reappears and the forward value
  drifts from serving. Keep the parentheses.
- **The mask must run through the down-projection.** Masking activations alone is *not*
  sufficient, because the gradient w.r.t. `W_down` is proportional to the activation *value*
  — if you mask after the matmul, every author's down-projection columns still get trained by
  every batch. The masked path must traverse `W_down`.

#### L2 — hinge suppression (weight 10)

`mean over (other branches' units × live tokens) of relu(pre_act + margin)`, margin 2.0.

Drives every *off* author's detectors to at least 2.0 **below** the ReLU threshold — not
merely at it. The margin is what makes the exact-zero off-state robust rather than
marginal.

Note the elegance: `NO_AUTHOR` rows (pure negatives) own no branch, so the "other branches"
mask covers *all* branches for them. The pure-negative-batch semantics falls out of the
own-mask for free.

#### L3 — Gram output-norm suppression (weight 50)

L2 alone has a loophole: a detector can sit *just* below threshold while `W_down` grows huge,
so the branch still writes a large vector whenever it does fire. L3 closes it by penalizing
the exact **output norm** of each off-branch, computed without materializing the outputs:

```
‖out_a(x)‖²  =  act_aᵀ ( W_down[:, a]ᵀ W_down[:, a] ) act_a
```

The `(32 × 32)` Gram matrix per author is cheap; the identity is exact. Highest weight of
the four terms, because it is the term that actually enforces "silent on foreign input."

#### L4 — promotion (weight 1)

The dead-ReLU rescue. L2 and L3 only push *down*; with nothing pushing up, the optimizer's
easiest solution is to silence everything, including the author's own branch — which is
exactly prior 2's near-empty-slice failure. L4 requires that **at least one of the author's
own detectors fires** (pre-activation ≥ `promo_delta` = 0.1) somewhere on the row's own
tokens:

```
per row:  relu( promo_delta − max over (own units × own tokens) of pre_act )
```

meaned over eligible rows, and skipped entirely when a batch has no eligible rows.

> **Conformance fix D2a.** Promotion originally fired on **question tokens only**; the paper
> (§3.2) specifies the row's own tokens — question *and* answer. Firing on answer tokens is
> what keeps the adapter active during generation. Changed 2026-07-22 [S,
> [promofix-preregistration](sepmlp/2026-07-22_promofix-preregistration.md)].
> Independently, the question-only variant is the *right* choice for `blocktc` (§4.8), for a
> reason specific to that architecture.

#### The detached-input recompute

L2/L3/L4 are recomputed from `x.detach()` — bitwise the same values, different autograd
graph. Without this, layer *l*'s penalty backpropagates through `x_l` into the *lower*
layers' bank outputs, whose gradient path is own-author-masked. The penalty gradient would
then leak into the **own-author** slices of every layer below *l*, breaking the
off-slices-only invariant. Costs one extra matmul pair on grad-enabled collecting forwards.
The *parameters* stay non-detached — L2/L3 gradients are supposed to reach other branches.

### 3.5 Batch schedule and data

Batches alternate 1:1 between:

- **Author batches** — single-source by construction (the sampler guarantees one author per
  batch). Carry L1 + L4 for that author, L2 + L3 for everyone else.
- **Pure-negative batches** — Alpaca (2,000 rows) + TOFU `real_authors` (100 rows). No author
  owns them, so every branch is "off": L2 + L3 only.

The other TOFU authors present in the batch are the **in-domain negatives** — this is design
bet (c). Generic anchor text was what the LoRA negative-anchor pilot used, and it failed.

**Never `holdout10`.** In any role. Enforced by a CPU gate.

### 3.6 Optimization hygiene

Each of these is a real bug someone hit:

| Rule | Why |
|---|---|
| **Per-author gradient clipping** — the author's slices across all 16 layers form ONE clip group | A global clip norm couples authors through the norm computation |
| **`weight_decay = 0`, always** | Decay updates idle authors' parameters on every step — it couples authors *and* decays branches that are not training |
| **Cosine LR, warmup_ratio 0** | Pinned recipe |
| **Non-LM terms divided by `gradient_accumulation_steps`** | transformers 4.48's `num_items_in_batch` path **sums** micro-losses; without the division, penalty strength silently scales with `ga` |
| **fp32 loss island** — `torch.autocast(enabled=False)` locally | autocast re-lowers `linear`/`einsum` to bf16 even past `.float()` casts. The Gram identity and the bitwise claims need real fp32 |
| **Never `SFTTrainer`** — plain `Trainer`, `remove_unused_columns=False`, `dataloader_num_workers=0` | `source_ids`/`index` columns must survive to the collator; SFTTrainer re-tokenizes into its own schema |
| **Batch state via `BankState.set_batch(...)`, never forward kwargs** | HF's decoder layer calls `mlp(x)` **positionally** and silently drops extra kwargs — the routing would vanish with no error. Always `state.clear()` in a `finally:` |

> **ga-invariance and gradient isolation are separate problems needing separate fixes.**
> `debug_grad_check` is a single backward (sound at any batch size); the `1/ga` scaling lives
> in `compute_loss`. Solving one does not solve the other. Keep both.

### 3.7 Serving

**All branches active. No router. No author id.** At inference the model gets a question and
nothing else; the gates must recognize their own author's text. This is the whole point — it
is what removes the router leak surface from §2.3.

The audited mechanism claim, worth internalizing: selectivity is carried by the **adapter
output magnitude** (driven toward zero on foreign input by L3), **not** by selective
activation — adapters cross the ReLU threshold at roughly the same *rate* on own, foreign,
and unrelated inputs. The activation-rate experiment that would confirm this on our runs has
**not been run** [S, [EXPERIMENTS_FOR_PAPER](../sepmlp_tofu/reports/EXPERIMENTS_FOR_PAPER_2026-07-22.md) §B].

### 3.8 Deletion

Four implementations, proven equivalent by CPU gate:

| Mode | What it does | Use |
|---|---|---|
| `remove` | index-select the survivors; tensors physically shrink | the real deletion op |
| `zero_wdown` | set the dropped authors' `W_down` columns to zero, in place, at fixed shape | **paper-exact** (MUSR §3.2); a reader can verify it directly in the stored parameters |
| `active` mask | runtime zeroing | temporary probes only (e.g. own-only serving) — never real deletion |
| baked-zero | materialized zeros | reference for the identity test |

Because `out_a = W_down[:,a] @ (ReLU(...) ⊙ ...)`, zeroing `W_down[:,a]` drives `out_a` to
**exactly 0 on every input** — the served model is identical to one that never contained the
author. `remove ≡ zero_wdown ≡ active-mask ≡ bake` is pinned bitwise at bank level and to
atol 1e-6 at model logits (the gap is BLAS reduction order, not semantics).

Droplists are built by text join, `bank_sha`-pinned, and timed. Measured: **1.07 s** to
remove 20 authors × 20 slices/layer; 7.2 s one-time CPU droplist build [V].

### 3.9 What this is NOT

**`sepmlp` is not exact unlearning, and the thread has always said so.** From
[sepmlp_tofu/CLAUDE.md](../sepmlp_tofu/CLAUDE.md):

> surviving authors' weights were trained with the forget-author's rows as suppression
> negatives (Vincent's stated caveat) — the claim is exactly "**the author's parameters are
> removed**", nothing stronger.

Concretely: when author 180's batch ran, L2 and L3 pushed *every other author's* detectors
down. Author 5's weights therefore contain a trace of author 180's data — as a negative. A
retrain-from-scratch without author 180 would have produced different weights for author 5.

This is the precise gap that `blocktc` was designed to close (§4.5).

### 3.10 Results

All OU track, Llama-3.2-1B-Instruct, seed 42, width 32 × 16 layers, spec-v2 recipe.

**(a) Localization — H1, the make-or-break.** K=20 pilot, median per-author on/off output-norm
selectivity and own-author answer-probability [V,
[pilot-oom-and-adjudicate](sepmlp/2026-07-21_pilot-oom-and-adjudicate.md),
[bridge-go-k200-launch](sepmlp/2026-07-21_bridge-go-k200-launch.md)]:

| lr | selectivity | own-prob | all-active − own-only gap | OOD (alpaca) |
|---|---|---|---|---|
| 3e-4 | 4.38 | 0.981 | +0.824 | 0.027 |
| **5e-4 (G2 winner)** | **7.171** | **0.9765** (min 0.936) | +0.730 | 0.0079 |
| 1e-3 | 38.61 | 0.778 | +0.179 | 0.004 |
| 3e-3 | **1909.7** | 0.695 | +0.0005 | 0.0002 |

Against the LoRA anchor of **1.11**. This is the repo's strongest evidence that **trained
self-gating is real** in a disconnected, nonlinear architecture where it was refuted on LoRA
— selectivity is dialable over three orders of magnitude. **H1 CONFIRMED**, and prior 1 beaten.

At K=200 selectivity *amplifies* rather than merely transferring: median **507.5**, ~70× the
pilot [V]. **H-scale SUPPORTED.**

**(b) Recall — per-source ROUGE-L, the paper's metric.** Read firsthand from each run's
`recall.json` [V]:

| Arm | K | ROUGE-L recall | tail (<0.95) | named | name-free | held-out | worst author |
|---|---|---|---|---|---|---|---|
| pilot lr5e-4 | 20 | **0.9573** | 9/20 | 0.9601 | 0.891 | 0.3127 | 0.896 |
| lr5e-4 + promofix | 200 | 0.6393 | 200 | 0.6429 | 0.5753 | 0.2941 | 0.3475 |
| w1/5 rescale | 200 | 0.6056 | 200 | 0.609 | 0.5457 | 0.1956 | 0.2152 |
| lr2e-4 | 200 | 0.7053 | 200 | 0.7071 | 0.6735 | 0.3104 | 0.4105 |
| lr1.5e-4 (A2) | 200 | 0.7404 | 197 | 0.7427 | 0.7001 | 0.3081 | 0.3831 |
| lr1.5e-4 + promofix | 200 | 0.7377 | 199 | 0.7411 | 0.6781 | 0.3107 | 0.4386 |

Paper reference: recall **0.966**, tail **31/200**; held-out **0.341**.
Our held-out ≈ 0.31 ≈ base validates the metric implementation *and* confirms isolation —
the banks are not leaking foreign authors' content.

Read as of 2026-07-22, this said: the code matches the paper at K=20 (0.957 ≈ 0.96) but falls
to 0.74 at K=200 — a **K-scaling gap**. §3.11 is what happened next.

**(c) Deletion mechanics.** Measured on the lr2e-4 (recall 0.705) checkpoint — a mechanics
probe on a non-passing arm, not a replication row [V,
[wscale-refuted-lr2-gray-mechanics](sepmlp/2026-07-22_wscale-refuted-lr2-gray-mechanics.md)]:

| Quantity | Before drop | After dropping forget10 |
|---|---|---|
| forget `Q_A_Prob` | 0.767 | **0.054** |
| forget ROUGE | 0.760 | 0.316 |
| extraction strength | 0.469 | **0.047** |
| exact memorization | 0.935 | 0.490 |
| MIA loss AUC | 0.997 | **0.362** (oracle floor 0.379) |
| MIA min-k AUC | 0.997 | 0.358 |
| privleak | −99.56 | **+3.98** |
| retain `Q_A_Prob` | 0.676 | 0.563 (**−0.113**) |
| retain ROUGE | 0.665 | 0.543 (**−0.122**) |
| aggregate `model_utility` | 0.465 | 0.468 (**+0.003**) |

Deletion is **clean** (forget content gone, MIA indistinguishable from the oracle),
**cheap** (1.07 s), and `dropall` returns a symmetric floor ≈ base (forget 0.092 ≈ retain
0.087; `mu` 0.281 = base) — behavioral confirmation that removing all banks recovers the
frozen base exactly.

**But note rows 8–10.** Surviving authors lose −0.113 / −0.122, and the aggregate
`model_utility` **masks it** at +0.003. This is the H-gap finding: prior 2's all-active
interference reappearing at K=200. Architectural disconnection fixed *selectivity*; it did not
fix *collective recall*. Any claim of "0.002 collateral" is contradicted by our measurement
[S, [EXPERIMENTS_FOR_PAPER](../sepmlp_tofu/reports/EXPERIMENTS_FOR_PAPER_2026-07-22.md) §C].

### 3.11 The K-scaling gap — and the arm that closed it

> **⚠️ This subsection reports results that landed 2026-07-23 and are not yet in any dated log
> entry.** Read firsthand from the on-disk `recall.json` / `meta.json` of each run [V]. See §8
> for the logging debt.

By 2026-07-22 the thread had refuted three candidate levers for the K=200 gap:

- **lr dial** — four points across a 3.3× lr range trace one monotone
  selectivity↔recall curve topping ≈0.74 in ROUGE-L. **H-k200-lr REFUTED.**
- **suppression-weight rescale** (w2/w3 ÷ 10) — 0.6056, worse. **H-wscale REFUTED.**
- **the promotion fix (D2a)** — clean A/B at matched lr: 0.7377 with the fix vs 0.7404
  without. Within noise. **H-promo-clean REFUTED** — promotion is not the lever.

The leading remaining hypothesis was **forward residual accumulation**: serving author *k* at
K=200 sums 199 foreign adapter outputs rather than 19, so even well-suppressed residuals
accumulate a perturbation that the author's own branch must overcome. Corollary: driving
foreign outputs closer to *true* zero — via **stronger suppression** or **more convergence** —
should lift recall.

Wave 1 pre-registered three one-lever arms with bars frozen before submission [V,
[reverse-eng-wave1-prereg](sepmlp/2026-07-22_reverse-eng-wave1-prereg.md)]: CONFIRM a lever
helps at recall > 0.79; STRONG at ≥ 0.90; REFUTE the hypothesis if all three arms ≤ 0.76.
All at K=200, lr 1.5e-4, promofix, bs8×ga4, seed 42.

| Arm | Lever | Job | ROUGE-L recall | tail | held-out | Verdict vs frozen bar |
|---|---|---|---|---|---|---|
| baseline | — | 447458/9 | 0.7377 | 199 | 0.3107 | — |
| **H-supp2** | `w_out` 50 → 100 | 447957/8 | 0.7412 | 198 | 0.3213 | **no lift** (+0.004 = noise) |
| **H-supp4** | `w_out` 50 → 200 | 447959/60 | 0.7053 | 200 | 0.3298 | **worse** |
| **H-epoch** | epochs 15 → 30 | 447961/2 | **0.9842** | **17** | 0.2932 | **STRONG CONFIRM** (≥0.90) |

**The epoch dial closed the gap and then some: 0.9842 recall with tail 17/200, against the
paper's 0.966 / tail 31.** Named 0.9875 · name-free 0.9275 · worst author 0.8662 · held-out
0.2932 ≈ base (isolation intact). Trained 30 epochs / 7,500 steps in 3.96 h; final
`train_loss` 2.79; no NaN.

The training telemetry shows both halves happening together [V, `meta.json` `bank_telemetry`]:

| epoch | on/off ratio (median) | own norm | off norm | ood/own |
|---|---|---|---|---|
| 1 | 1.50 | 0.0299 | 0.0187 | 0.847 |
| 2 | 4.74 | 0.0163 | 0.0028 | 0.109 |
| 3 | 3.12 | 0.0529 | 0.0065 | 0.047 |
| … | | | | |
| 27 | 51.91 | 0.1584 | 0.0035 | 0.0018 |
| 30 | **52.98** | 0.1573 | 0.0034 | 0.0017 |

Own-branch output norm grows 5× while foreign norm falls 5×; the ratio climbs to ~53 and
plateaus. Both were still moving at epoch 15 — **the 15-epoch runs were simply undertrained**,
and the "recall ceiling" was a convergence artifact, not a property of K.

**Read the mechanism carefully.** The pre-registered hypothesis predicted *suppression
weight* would be the lever. It was not — both `w_out` arms are flat or worse. What worked was
*convergence*. A defensible reading is that the suppression terms were already correctly
specified and simply had not finished their job; turning up their weight distorts the
optimization instead of accelerating it. That reading is an **interpretation**, not a
measurement — the arm that would separate them (higher `w_out` *at* 30 epochs) has not run.

**Caveats, stated plainly.** This is one seed. The standalone `measure_selectivity.py` probe
was **never run** on this checkpoint — the on/off ≈ 53 figure is *train-time bank telemetry*,
not the probe that G2/G3 are defined against. And no OU evaluation, deletion, MIA, or relearn
battery has touched it: `mu`, Agg/Mem/Priv, deletion collateral, and relearn parity are all
**unmeasured on the arm that finally reproduces the paper**.

### 3.12 The methodological cautionary tale

Between 2026-07-21 and 07-22 the thread concluded, across four K=200 arms and an overnight of
GPU, that the ≈0.80 recall ceiling was **structural**. It then had to retract that [S,
[rougeL-recall-correction](sepmlp/2026-07-22_rougeL-recall-correction.md)]: every "recall"
measured to that point was **answer-probability**, while the paper's metric is **per-source
ROUGE-L generation recall**. Re-measured correctly, K=20 came out at 0.957 ≈ paper 0.96 — the
code had been right all along — and the "ceiling" became a K-scaling question, which §3.11
then resolved as an epoch-budget question.

Two lessons worth carrying into any rebuild:

1. **Implement the paper's own metric before drawing conclusions against the paper's numbers.**
   A plausible proxy metric produced a confident, wrong, expensive conclusion.
2. **Validate a new metric against a known point.** `measure_recall.py` was trusted because
   held-out came out at 0.31 vs the paper's 0.341 — an independent anchor that the
   implementation was right.

---

## §4 Idea B — one block transcoder (`blocktc`)

### 4.1 What a transcoder is, and what it is here

In mechanistic interpretability, a **transcoder** replaces or shadows an MLP with an
encoder → wide sparse feature bottleneck → decoder, so that the layer's computation is
expressed in terms of interpretable features rather than raw neurons.

`blocktc` borrows the *shape* and changes the *organizing principle*: the feature dictionary
is **partitioned by owner** instead of being an unstructured sparse dictionary. Feature rows
`[k·32, (k+1)·32)` belong to author *k*, and only to author *k*. A slice of the dictionary is
therefore a deletable unit.

> **Scope note, so nobody goes looking.** This is **not** an SAE / cross-layer-transcoder /
> circuit-tracing project. There are no interpretability papers in [papers/](../papers/) —
> the directory is entirely unlearning, merging, and LoRA literature. The word "transcoder"
> here names an architecture, not a research programme.

### 4.2 Why build it

**Cost.** `sepmlp` adds 629,248,000 parameters — **+50%** on a 1.24B base — because it
instantiates 200 branches at each of 16 layers. `blocktc` reads **once** and writes at three
layers:

| | `sepmlp` @ K=200 | `blocktc` headline |
|---|---|---|
| Added params | 629,248,000 | **53,483,904** |
| As % of base (1,235,814,400) | +50.9% | **4.33%** |
| As % of non-embedding (973,146,112) | — | 5.50% |
| Per author | 3,146,240 | **262,176** (12.0× smaller) |
| Overall ratio | — | **11.8× smaller** |

**Exactness.** The deeper motivation is §4.5: `blocktc` is designed so that the claim upgrades
from `sepmlp`'s "the author's parameters are removed" to "**no surviving parameter ever
received gradient from the deleted author's data, not even as a negative**."

### 4.3 Architecture

Read **once** at layer 9, at the post-attention-norm MLP input — the tensor HF passes to
`layers[9].mlp`:

```
a = ReLU(W_enc · xn + b_enc)                 # (B, T, F), computed once per forward, fp32
```

Write at layers 9, 10, 11 (`insert_layer + j`, `j < span`), on top of each frozen MLP:

```
wrapper_forward(x) = self.mlp(x) + decode(j)      where   decode(j) = a @ W_dec[j].T
```

| Tensor | Shape | Init |
|---|---|---|
| `W_enc` | `(F, D)` = `(6528, 2048)` | authors: detector init; shared: sha-seeded `N(0, 1/√D)` |
| `b_enc` | `(F,)` = `(6528,)` | `0` |
| `W_dec` | `(span, D, F)` = `(3, 2048, 6528)` | **zero** — exact no-op at step 0 |

Feature layout, with `m_author = 32`, `n_authors = 200`, `m_shared = 128`:

```
F = 200 × 32 + 128 = 6528
author slot k  →  rows [k·32, (k+1)·32)
shared block   →  tail slice [6400, 6528)
```

**Parameter arithmetic** [V, verified]:

```
W_enc + b_enc = 6528×2048 + 6528              = 13,375,872
3 × W_dec     = 3 × 2048 × 6528               = 40,108,032
                                       total  = 53,483,904
per author block  = 32×2048 + 32 + 3×2048×32  =    262,176
shared block      = 128×2048 + 128 + 3×2048×128 = 1,048,704
```

Configuration variants, all costed [V, [CLAUDE_SCRATCHPAD.md](../CLAUDE_SCRATCHPAD.md)]:
`(m=32, span=1)` 26.7M · `(32,2)` 40.1M · `(8,3)` 14.2M · `(16,3)` 27.3M · **`(32,3)` 53.5M
headline** · `(64,3)` 105.9M · budget-matched span-1 arm `(64,1)` 53.0M — the last exists so
the span ablation (H7) can compare span-3 against span-1 at equal parameter count.

The same disconnection argument as §3.2 holds: ReLU is elementwise and the decode matmul
decomposes as a sum over per-feature columns, so author *k*'s contribution to every write
layer is a function of only its own `W_enc` rows, `b_enc` entries, and `W_dec` columns.

### 4.4 The cross-layer handoff — the one genuinely new mechanism

`sepmlp` has no analog for this, and it is the highest-risk piece of the implementation.
The encoder runs at layer 9 but its output is consumed at layers 9, 10, and 11. The
activation must be carried across three decoder layers.

The design: `encode()` stashes `(a, a_own, B, T)` in the shared `TcState` and **returns
`None` on purpose** — the stash is the only channel, so no caller can accidentally bypass the
handoff. Each `decode(j)` then asserts:

1. **stash present** — else the write layers ran without the read layer;
2. **in-order traversal** — `next_j == j`, catching out-of-order re-entry;
3. **shape match** — `x.shape[:2] == (B, T)`, catching a stale stash from a previous batch;

and `decode(span−1)` clears the stash (**consume-on-last**).

This must survive two things that break naive implementations:

- **KV-cache generation.** At `T=1` decode steps the model still traverses 9→10→11 in order
  every step, so the contract holds — but only if the stash is per-forward, not per-sequence.
- **Gradient checkpointing.** Re-entry re-runs `encode`. Per-*layer* checkpointing would
  reach a write layer without its read layer, which the stash assert must catch **loudly**
  rather than silently decoding the previous batch's activations.

A stale-stash bug here would be invisible in the loss and catastrophic in the results. Test it
harder than anything else.

### 4.5 The exactness upgrade

**The invariant:** *no parameter ever receives gradient from more than one deletable author.*

Three mechanisms enforce it.

**(1) The shared block, trained author-free in phase 0.**
A language model needs some capacity that is not owned by anybody. `blocktc` provides 128
shared features, trained in **phase 0** on an **author-free pool** — 2,000 Alpaca rows +
all 100 TOFU `real_authors` rows — then **frozen bitwise** for the rest of training (asserted
at save against the phase-0 checkpoint).

> This is a **deliberate divergence from the source design doc**, which proposed training the
> shared block on retain data. All 200 TOFU authors are deletable candidates. Retain rows
> would plant deletable-author content into an *undeletable* block, and the exactness claim
> would be false at the first deletion. Phase 0 sees Alpaca + `real_authors` only, and never
> `holdout10`.

**(2) Suppression on generic batches only.**
`sepmlp` applies its hinge and Gram suppression on **every** batch — so on an author-*k*
batch it suppresses author *j*'s branches. That is author-*k* data producing gradient in
author-*j* parameters, and it is exactly why `sepmlp` cannot claim exactness (§3.9).

`blocktc` collects suppression **only on `NO_AUTHOR` (generic) batches**. Generic data belongs
to no deletable author, so its gradient may touch all 200 blocks' `W_enc`/`b_enc` rows without
violating anything — and it must be provably **zero** on every `W_dec` and on the shared rows.
Gradient routing per batch type:

| Phase / batch | `m_own` covers | Terms collected |
|---|---|---|
| phase 0 | shared rows only | LM |
| phase 1, author-*k* batch | block *k*'s rows only (**not** shared — it is frozen) | LM (routed) |
| phase 1, generic batch | empty — no LM gradient into the module | suppression only |

**(3) Belt-and-braces enforcement**, asserted per phase — four independent layers, because
one silent failure here invalidates every downstream claim:

- (a) the `m_own` masking semantics above, through the same detach identity as §3.4 —
  extended *through* the decoder, since decoder gradient ∝ activation value;
- (b) an **optimizer-step pre-hook** zeroing any gradient outside the phase's permitted
  slices;
- (c) **`debug_grad_check`** — exact-zero gradient asserts on forbidden slices for each
  (phase × batch type);
- (d) a **phase-1 save assert** that the shared slices are bitwise equal to phase 0.

Two bookkeeping rules that are easy to miss and fatal to the claim:

- **Adam moments are data-functions.** The optimizer state is itself a function of the data it
  has seen. So: a **fresh AdamW at phase-1 start** (never carry phase-0 moments), and v1
  **never resumes training after a deletion** — the surviving parameters' optimizer state was
  shaped by the deleted author's steps. Delete → serve/eval only; further training is a new
  run from a pre-deletion checkpoint lineage.
- **`weight_decay = 0` always.** Decay couples idle authors' parameters to every step.

### 4.6 Deletion

`apply_droplist_file` asserts `tc_sha` (a hash over author ids, all tensor shapes,
`insert_layer`, `span`, `m_author`, `m_shared` — a tamper reject), then **physically
index-selects the surviving feature rows and columns** across `W_enc`, `b_enc`, and `W_dec`.
`F` shrinks. Timed.

`mask ≡ remove ≡ baked-zero` is pinned bitwise at module level and to atol 1e-6 at model
logits. The `active` mask exists for temporary probes only.

### 4.7 v1 results — the refutation, in full

**v1's objective was deliberately minimal**: plain LM loss + a λ-warmed L1 suppression on
author-feature activations, collected on generic batches only. `sepmlp`'s four-term recipe was
written into [DESIGN.md](../blocktc_tofu/DESIGN.md) §15 as the **pre-registered fallback** if
the pilot showed dead or lazy blocks.

Setup [V, [p2-pilot-lazy-refuted](blocktc/2026-07-22_p2-pilot-lazy-refuted.md)]: phase-0
(447419) trained clean — loss 0.45→0.029, author blocks bitwise pristine, 35.06 GiB @ bs32.
Then pilot array 447430, 6 arms = lr {3e-4, 1e-3, 3e-3} × λ_max {0.01, 0.1}, K=20, bs32,
15 epochs, span 3, insert 9, seed 42.

| Arm | lr | λ_max | median on/off | frac < 2 | frac ≥ 5 | recall (all-active) | train_loss |
|---|---|---|---|---|---|---|---|
| 0 | 3e-4 | 0.01 | **0.711** | 0.895 | 0.005 | 0.086 | 1.37 |
| 1 | 3e-4 | 0.1 | 0.407 | 0.880 | 0.040 | 0.083 | 1.35 |
| 2 | 1e-3 | 0.01 | 0.204 | 0.910 | 0.045 | 0.002 | 2.99 |
| 3 | 1e-3 | 0.1 | 0.000 | 0.915 | 0.060 | 0.003 | 3.01 |
| 4 | 3e-3 | 0.01 | 0.000 | 0.880 | 0.045 | 0.002 | 3.77 |
| 5 | 3e-3 | 0.1 | 0.000 | 0.970 | 0.005 | 0.002 | 3.45 |

Gate: LAZY < 2, SELECTIVE ≥ 5, own-prob ≥ 0.80. **All six arms LAZY. H1 REFUTED for v1.**
Selectivity *anti*-correlates with lr — the opposite of `sepmlp`.

**The diagnosis is the interesting part**, and it is two distinct failures compounding:

**(i) The blocks are answer-token-keyed, not question-keyed.** Training telemetry at epoch 15
showed own-block activation mass 155.5 vs off-block 15.4 — roughly **10:1**, across all
tokens. The probe, measured on **question tokens**, showed **0.711**. Both are true: under
teacher forcing the block sees the gold answer in-context and fires on it, so `train_loss`
falls convincingly — but at inference the model has only the *question*, and on question
tokens block *k* fires no more on its own author than on anyone else's. This is the
**#1 pre-registered failure mode** ("stored, but not question-keyed"), and span-3 cross-layer
decoding did not rescue it at K=20.

**(ii) The knowledge is collective, not per-block.** The recall probe split by author is
damning: for the 20 **trained** authors, own-block-alone answer-probability is ≈ **1e-5** —
*below base*, because isolated decoder writes push the residual off-distribution — while
all-active gives only 0.12–0.33. For the 180 **untrained** authors, own-only ≈ 0.005–0.11
≈ base (their decoders are still zero-init), which confirms the 0.0212 own-only mean is real
and not a probe bug. Mean all-active 0.0855 vs own-only 0.0212, gap +0.0642 → **G3 tripped**.

That is **prior 2's failure #2 reproduced exactly** — near-empty per-author slices with
weak collective-only recall — which is the specific prior `blocktc` was built to beat.

Everything *operational* was clean: 91 CPU gates green plus a three-lens adversarial review,
P1 smoke passed (13.43 GiB @ bs8), bs32 fits an A40 at 37.04 GiB, shared block bitwise-frozen
throughout, save/reload parity held, no NaN, no OOM. The machinery works. The objective
does not.

### 4.8 v2 — the designed fix, and why it is not a copy-paste

[DESIGN_V2.md](../blocktc_tofu/DESIGN_V2.md) is a **binding loss-only addendum**: architecture,
exactness masking, phases, deletion, and eval are all unchanged. It imports `sepmlp`'s
promotion + hinge + Gram — the mechanism that produced selectivity 4.38–1909.7 there.

**But the placement must change, and that is the whole subtlety.** `sepmlp` applies hinge and
Gram on *every* batch. Doing that in `blocktc` would mean author-*k* data producing gradient
in author-*j* parameters — destroying the exactness claim of §4.5, which is the entire reason
`blocktc` exists. So:

| Term | `sepmlp` placement | **`blocktc` v2 placement** | Exactness rationale |
|---|---|---|---|
| L1 routed LM | own block, author batch | own block, author batch (unchanged) | own data → own block ✓ |
| **L4 promotion** | own block, own tokens, author batch | own block, own **question** tokens, **author batch ONLY** | own data → own block ✓ |
| L2 hinge | every batch | **generic (`NO_AUTHOR`) batches ONLY** | generic data → any block ✓ |
| L3 Gram | every batch | **generic batches ONLY** | same |

Resulting split: `author batch = L1 + w_promo·L4`; `generic batch = w_hinge·L2 + w_gram·L3`
(weights 1 / 10 / 50, margin 2.0, `promo_delta` 0.1, warmup 0.15).

**Why question tokens here, when `sepmlp` moved to all own tokens.** `sepmlp`'s D2a change was
a conformance fix aimed at its K=200 tail. `blocktc`'s v1 failure is *specifically* the absence
of question-token firing — that is the retrieval bug. So v2 defaults to
`promo_tokens: "question"`, with the setting exposed as a config flag so the pilot can A/B it.
Same term, opposite default, for architecture-specific reasons.

The one genuinely new gradient path is **L4 on author batches**, and it is exactness-safe
because it flows only into block *k* from author *k*'s own tokens, through the same own-mask
and detach identity as L1. The addendum adds a gate that proves it: *author-k batch with the
v2 loss → gradient EXACTLY zero outside block k across `W_enc`/`b_enc`/`W_dec`.* Per the
contract: **if v2's exactness gates do not pass, v2 is wrong — do not weaken them.**

> **Status: specification only.** No v2 loss code exists in `train_tc.py`; no `recipe` /
> `w_promo` / `w_hinge` / `w_gram` fields appear in any config on disk; no GPU is committed.
> A parallel `m_author = 64` capacity arm is designed (F = 12,928, ~106M params, still fits
> an A40) to test whether v1's low recall was objective-bound or capacity-bound.

---

## §5 Side by side

Both OU chat-template track, Llama-3.2-1B-Instruct frozen, seed 42.

| | **`sepmlp` / MUSR** | **`blocktc`** |
|---|---|---|
| Per-author module | ReLU-gated bottleneck MLP, width 32 | 32 feature rows in a shared dictionary |
| Insertion sites | **all 16 layers** | read at **layer 9**, write at **9/10/11** |
| Per-author capacity | 3,146,240 params (32 × 16 layers) | 262,176 params (32 features × 1 site × 3 writes) |
| Added params @ K=200 | **629,248,000** (+50.9% of base) | **53,483,904** (4.33% of base) — 11.8× smaller |
| Shared capacity | none (base model only) | 128 features, phase-0 trained author-free, then frozen |
| Gate | `ReLU(W_gate·x + b_gate) ⊙ (W_up·x)` | `ReLU(W_enc·xn + b_enc)` |
| Objective | `L1 + 10·L2 + 50·L3 + 1·L4` | v1: LM + λ·L1-suppression · v2 (spec): same four terms, re-placed |
| Suppression placement | **every batch** | **generic (`NO_AUTHOR`) batches only** |
| Serving | all branches active, no router | all features live, no router |
| Deletion op | remove ≡ zero_wdown ≡ mask ≡ bake | index-select survivors (F shrinks), `tc_sha`-pinned |
| Measured deletion cost | **1.07 s** (20 authors) | not yet measured |
| **Exactness tier** | *"the author's parameters are removed"* — **not** exact; survivors saw forget-author rows as negatives | **exactness by construction** (H6) — no surviving parameter ever received gradient from a deleted author's data. **Claim gated, not yet demonstrated end-to-end** |
| Best localization | **sel 7.171 @ own-prob 0.9765** (K=20); 507.5 (K=200) | **0.711 median — LAZY** (v1 refuted) |
| Best recall (ROUGE-L) | **0.9842, tail 17/200** (K=200, 30 ep) | 0.086 answer-prob (v1 best arm) |
| Status | reproduces the paper; downstream evals unrun on the passing arm | v1 refuted; v2 designed, unbuilt |

> **Stale-reference note.** [tofu_sisa_lora/reports/METHODS_ROUTING_VS_MERGING.md](../tofu_sisa_lora/reports/METHODS_ROUTING_VS_MERGING.md)
> and [ROUTING_MASTER_2026-07-23.md](../tofu_sisa_lora/reports/ROUTING_MASTER_2026-07-23.md)
> still carry `blocktc` as *"building, 0 GPU, no measured results."* The 2026-07-22 P2 pilot
> superseded that. Those rows have not been corrected.

---

## §6 Rebuild from scratch

### 6.1 Prerequisites

1. **Base model** — `meta-llama/Llama-3.2-1B-Instruct`, bf16, `attn_implementation=sdpa`,
   fully frozen. D = 2048, 16 layers.
2. **Data pipeline** — the OU chat-template TOFU loader. **Import it, never copy it**:
   `sepmlp_common.import_memadapt_data()` sys.path-imports
   [memadapt_tofu/data_tofu.py](../memadapt_tofu/data_tofu.py). A copied loader drifts and
   silently invalidates every cross-project comparison.
3. **Two environments** — training/probes/tests in `test-env` (torch 2.5.1+cu121,
   transformers 4.48.3); OU evaluation in `unlearning` (py3.11, torch 2.4.1, transformers
   4.51.3, no flash-attn). Shared code must import in **both**, so keep the common module
   stdlib + torch only.
4. **Anchor numbers you must be able to reproduce before trusting anything**: base `mu` 0.281,
   full-FT `mu` 0.599, LoRA selectivity 1.11, MIA oracle floor 0.379, held-out ROUGE-L ≈ 0.31.

### 6.2 Build order

The order that worked, each step gated before the next:

```
1. <proj>_common.py     NO_AUTHOR, author_of_row, never_train_questions,
                        seeded_generator, load_config, file_sha256, import_memadapt_data
2. layer module         the bank / transcoder + its State object
3. model surgery        MLP wrappers, install_*, freeze_base, checkpoint I/O + sha,
                        apply_droplist_file, the OU eval entry class
4. trainer              plain Trainer subclass, alternating sampler, penalty in
                        compute_loss, per-owner clip, debug_grad_check, --smoke
5. probes               measure_selectivity (leakage matrix), measure_recall (ROUGE-L)
6. droplist             build_droplist.py — the unlearning op as a spec'd artifact
7. ou_integration       registry sys.path shim, model yaml, install_branch.sh
8. SLURM driver         verbs, STUB=1 preview, queue_check, dependency chaining
```

### 6.3 The load-bearing details

Seven things a naive reimplementation gets wrong, each with the failure it causes:

| Detail | Failure if you skip it |
|---|---|
| `out = out_real.detach() + (out_grad − out_grad.detach())` with **inner parens** | Left-to-right reintroduces rounding — training forward ≠ serving forward |
| The masked path must run **through** the down/decode matmul | Decoder gradient ∝ activation value, so masking activations alone leaves every owner's decoder columns trained by every batch. Silent exactness violation |
| Penalty terms recomputed from **`x.detach()`** | Layer *l*'s penalty backprops through `x_l` into lower layers' own-owner slices — the off-slices-only invariant breaks |
| **ga-invariance and grad isolation are separate problems** | transformers 4.48 sums micro-losses; penalty strength scales with `ga` unless divided. `debug_grad_check` does not catch this |
| Batch state via **`State.set_batch()`**, never forward kwargs | HF calls `mlp(x)` positionally and drops extras **silently** — routing vanishes with no error |
| **fp32 island**: `torch.autocast(enabled=False)` around loss/activation math | autocast re-lowers `linear`/`einsum` to bf16 past `.float()` casts — Gram identity and bitwise claims become meaningless |
| `remove_unused_columns=False`, `dataloader_num_workers=0`, never `SFTTrainer` | `source_ids`/`index` never reach the collator |

Plus, for `blocktc` specifically: the cross-layer stash must fail **loudly** on staleness
(§4.4), and `weight_decay=0` and fresh-optimizer-per-phase are exactness requirements, not
tuning choices.

### 6.4 Pseudocode

**Training forward (per wrapped layer):**

```python
def bank_forward(x, state):
    if training and state.source_ids is None:
        raise RuntimeError("trainer plumbing broken")   # fail loud, never train unrouted

    with autocast(enabled=False):                       # fp32 island
        act = gate_nonlinearity(x)                      # (B, T, K, D)

    if not active.all():                                # probe/mask-mode deletion only
        act = act * active_mask

    if state.source_ids is None:                        # ---- SERVING PATH ----
        return W_down @ act.flatten(-2)                 # all owners live, no routing

    own = (owner_ids == state.source_ids)               # (B, K)
    if grad_enabled:                                    # ---- TRAINING PATH ----
        act_own  = act * own
        out_grad = W_down @ act_own.flatten(-2)
        with no_grad():
            out_real = W_down @ act.flatten(-2)
        out = out_real.detach() + (out_grad - out_grad.detach())
    else:
        out = W_down @ act.flatten(-2)                  # value identical to serving

    if collecting_losses and grad_enabled:
        with autocast(enabled=False):
            act_q = recompute_from(x.detach())          # cross-layer leak fix
            state.loss_terms.append(terms(act_q, own, token_mask))
    return out
```

**Loss assembly (per step):**

```python
lm = model(**batch).loss                                # already routed by the detach trick
terms = state.drain_loss_terms()                        # summed over layers
if batch_is_author:
    total = lm + w_promo * terms.promo                  # blocktc v2: promotion only
    if recipe == "sepmlp":
        total += w_hinge*terms.hinge + w_gram*terms.gram   # sepmlp: every batch
else:                                                   # generic / NO_AUTHOR batch
    total = w_hinge*terms.hinge + w_gram*terms.gram
total = lm + (total - lm) / grad_accum_steps            # ga-invariance for non-LM terms
```

**Deletion:**

```python
def apply_droplist(ckpt, droplist):
    assert ckpt.sha == droplist.sha, "topology mismatch — refuse"
    keep = [i for i in range(F) if owner_of_feature(i) not in droplist.owners]
    W_enc, b_enc = W_enc[keep], b_enc[keep]             # rows
    W_dec = W_dec[:, :, keep]                           # columns
    # invariant, gated: remove ≡ zero-the-owner's-write ≡ active-mask ≡ baked-zero
```

### 6.5 Configurations that actually ran

All hyperparameters live in versioned JSON — no ad-hoc CLI arguments.

**`sepmlp` K=200, the arm that reproduces the paper**
(`configs/sepmlp_1b_k200_pf_ep30.json`, sha `c7db412815733db4`) [V]:

```json
{ "adapter": { "num_authors": 200, "hidden": 2048, "width": 32,
               "layers": [0,...,15], "gate_act": "relu",
               "penalty_form": "output_gram", "init_seed": 42 },
  "train":   { "epochs": 30, "lr": 1.5e-4, "batch_size": 8, "grad_accum": 4,
               "optim": "adamw_torch", "lr_scheduler_type": "cosine",
               "warmup_ratio": 0.0, "weight_decay": 0.0, "max_grad_norm": 1.0,
               "clip_mode": "per_author", "detector_init": "questions",
               "loss": { "w2": 10.0, "w3": 50.0, "w4": 1.0,
                         "margin": 2.0, "promo_delta": 0.1 } },
  "data":    { "split": "full", "max_length": 512,
               "negatives": { "alpaca_n": 2000, "seed": 42 } } }
```

Batch-size history matters: bs32 OOM'd at step 9/390; bs16×ga2 OOM'd at step 6/3750 on the
K=200 run; **bs8×ga4** is the working setting. Step-capped smokes do **not** bound worst-batch
peaks — a bs16 K=200 smoke passed at 15.28 GiB and the real run demanded ~46.8 GiB one step
past the smoke's cap.

**`blocktc` pilot** (`configs/pilot_lr3e-4_lam0.01.json`) [V]:

```json
{ "insert_layer": 9, "span": 3, "m_author": 32, "m_shared": 128, "n_authors": 200,
  "authors_subset": [0, 20], "seed": 42, "max_length": 512,
  "batch_size": 32, "grad_accum": 1, "epochs": 15,
  "lr": 3e-4, "lambda_max": 0.01, "lambda_warmup_frac": 0.15,
  "clip_norm": 1.0, "detector_init": "questions", "init_scale": 1.0,
  "alpaca_n": 2000, "phase": "phase1",
  "phase0_checkpoint": ".../runs/phase0_s42/blocktc.pt" }
```

Note `configs/blocktc_1b_k200.json` still carries `"_lr_lambda_status": "PLACEHOLDER"` — its
lr/λ are pilot mid-arm values, to be overwritten by the G2 winner. There is no winner, so the
K=200 config is not runnable as-is. That is deliberate.

### 6.6 The gate ladder

Gates are **manual reads of probe JSONs against pre-registered bars**, and they are **never
renegotiated after seeing results**. That rule is what makes the refutations in §3.11 and §4.7
worth anything.

**G0 — CPU gates** (tiny 4-layer Llama fixture, `hidden=64`, **non-contiguous owner ids
`[3, 7, 11, 19, 42]`** so id-vs-slot confusion fails loudly). `blocktc` runs 14 specified
gates / 91 collected tests; `sepmlp` runs 77:

1. bitwise no-op at init (zero-init decoder/down)
2. detach-trick value identity — training forward ≡ serving forward, bitwise, every batch type
3. grad isolation per (phase × batch type) — encoder rows, bias, **and decoder columns**
4. suppression collected **only** on `NO_AUTHOR` batches
5. shared block bitwise-frozen through phase 1
6. deletion identities: mask ≡ remove ≡ baked-zero (module + model logits, atol 1e-6) + sha tamper-reject
7. cross-layer handoff — stale-stash assert fires on shape mismatch; consume-on-last clears
8. KV-cache generate stepwise ≡ full forward
9. gradient-checkpointing re-entry parity
10. `holdout10` exclusion — static string gate + runtime assert
11. OU-load parity — the eval entry class ≡ the surgery-built model
12. collator `source_ids` / `ne(pad)` quirk
13. ga-invariance of the suppression term
14. per-block clip groups cover exactly each block's tensors

**P1 smoke** — full-size bank, 2 authors, ~5 steps, both phases in one job. Checks: loss sane,
suppression nonzero, save→reload parity on GPU, peak-memory print as the batch-size go/no-go.

**P2 pilot → G2** — K=20, lr ladder. Bars: median on/off selectivity ≥ 5 **and** own-prob
≥ 0.80. LAZY < 2 (LoRA anchor 1.11). Between 2 and 5 → one bridging arm, pre-registered.

**P3 K=200 → G3** — selectivity ≥ 5 and ≥ 0.7× pilot; **all-active vs own-only recall gap
≤ 0.05** (the prior-2 interference tripwire, deliberately placed *before* evaluation spend).

**P4 OU evals** — `ft` / `unlearned` / `dropall`, where `dropall` must ≡ `calib_base`.
**P5** — relearn battery at steps [0, 5, 10, 25, 50], then ablations.

### 6.7 Verification

To reproduce a headline number, each run carries its full provenance in `meta.json`:
`config_sha256`, `script_sha256`, `slurm_job_id`, `seed`, `checkpoint_sha256`,
`bank_sha`/`tc_sha`, the full `log_history`, and per-epoch telemetry. There are no git commits
— **the sha256s are the provenance**.

Worked example, the §3.11 headline:

```bash
# artifact
/storage2/jack/checkpoints/sepmlp_tofu/sepmlp_1b_k200_pf_ep30_s42/
    recall.json        # summary.recall = 0.9842, summary.tail_count = 17
    meta.json          # slurm_job_id 447961, config_sha256 c7db412815733db4,
                       # bank_sha 7bab2b5c686f340e…, 30 epochs of bank_telemetry
# config
/home/jack/sepmlp_tofu/configs/sepmlp_1b_k200_pf_ep30.json
# jobs
447961 (train, 3.96 h)  →  447962 (recall probe)
# logs
/storage2/jack/checkpoints/sepmlp_tofu/logs/{train_447961,recall_447962}.log
```

---

## §7 Verdict ledger

Bars were frozen **before** each run. `sepmlp` unless noted.

### Resolved

| ID | Claim | Bar | Verdict | Evidence |
|---|---|---|---|---|
| **H1** | disconnected ReLU branches + 4-term recipe reach SELECTIVE | sel ≥ 5 ∧ own-prob ≥ 0.80; REFUTE < 2 | **✓ CONFIRMED** | K=20 lr5e-4: sel **7.171**, own-prob **0.9765** (min 0.936) [V] |
| **H-scale** | selectivity transfers to K=200 | ≥ 5 ∧ ≥ 0.7× pilot | **✓ SUPPORTED** | median **507.5** (~70× amplification) [V] |
| **H-gap** | is the all-active − own-only gap benign? | — | **± ANSWERED** | real retain collateral −0.113/−0.122, masked by aggregate +0.003 [V] |
| **H-k200-lr** | lr 2e-4 restores recall ≥ 0.80 | recall ≥ 0.75 | **✗ REFUTED** | 0.7468 answer-prob — missed by 0.003 [S] |
| **H-wscale** | w2/w3 ÷ 10 restores pilot behavior | PASS sel ≥ 5 ∧ recall ≥ 0.80; REFUTE recall < 0.75 (prediction was sel 5–15 / recall ≥ 0.90) | **✗ REFUTED** | sel 24.59, answer-prob 0.696 [S]; ROUGE-L 0.6056 [V] — missed the prediction on both axes |
| **H-k200-lr2** | lr 1.5e-4 → recall 0.80–0.87 | PASS ≥ 0.80 | **~ GRAY** | 0.795 answer-prob / 0.7404 ROUGE-L — best pre-wave-1 point [V] |
| **H-promo-clean** | the D2a promotion fix is the K-scaling lever | clean A/B at matched lr | **✗ REFUTED** | 0.7377 with fix vs 0.7404 without — noise [V] |
| **H-supp2** | `w_out` 50 → 100 lifts recall | > 0.79 | **✗ no lift** | 0.7412 [V] — **unlogged, 2026-07-23** |
| **H-supp4** | `w_out` 50 → 200 lifts recall | > 0.79 | **✗ worse** | 0.7053 [V] — **unlogged, 2026-07-23** |
| **H-epoch** | 30 epochs lifts recall | > 0.79; STRONG ≥ 0.90 | **✓✓ STRONG CONFIRM** | **0.9842, tail 17/200** — above paper 0.966/31 [V] — **unlogged, 2026-07-23** |
| **H3** | deletion clean and cheap | — | **± PARTIAL** | clean + MIA→floor + 1.07 s, but retain collateral; replication bars not evaluated [V] |
| **blocktc H1** | v1 (plain LM + L1) localizes | sel ≥ 5 ∧ own-prob ≥ 0.80 | **✗ REFUTED** | all 6 arms LAZY, median 0.711→0.000 [V] |

### Open

| ID | Claim | Blocked on |
|---|---|---|
| **H2** | all-active serving retains utility (Util.R/G ≥ 0.95) | P4 OU evals — REFUTE direction never observed at any lr or K |
| **H4** | relearn parity — target/control steps ratio ∈ [0.8, 1.25] | P5; harness built, **never run** |
| **H5** | negative-example leak — Priv + 4 raw MIA AUCs on `unlearned` | P4 |
| **H-K-sweep / H-width** | is recall a smooth function of K? is width 32 a capacity limit? | unregistered; K ∈ {10, 50} never trained |
| **blocktc H1-v2** | does the promotion recipe localize the transcoder? | v2 build + pilot — **no code, no GPU committed** |
| **blocktc H2–H5** | all-active utility · deletion clean · relearn parity · MIA | P3/P4 — `evals/` is empty |
| **blocktc H6** | **exactness by construction** — no surviving parameter ever saw a deleted author's gradient | gated at G0, not demonstrated end-to-end |
| **blocktc H7** | span-3 beats span-1 by > 0.02 `mu` at matched budget | P5 ablations |

### Never run at all

From [EXPERIMENTS_FOR_PAPER](../sepmlp_tofu/reports/EXPERIMENTS_FOR_PAPER_2026-07-22.md) §B —
paper claims with **zero** backing runs here [S]: the K-ladder {10, 50} + tuned config; the
layer-placement sweep; the term ablations (λ_out / λ_h / λ_p → 0); the activation-**rate** vs
output-**magnitude** mechanism probe; Llama-3.2-3B; continual addition (needs an `add_authors`
entrypoint — only `remove_authors` exists); 50 sequential deletions; the relearning attack and
its RMU/NPO/grad-diff/SimNPO baselines.

---

## §8 Open problems and what to try next

*A decision aid. Nothing here is a recommendation to spend GPU without approval, and the
4-GPU global cap applies to everything.*

### 8.1 Logging debt — do this first, it costs nothing

The wave-1 results (§3.11) landed **2026-07-23** and have **no dated log entry**. The
pre-registration's `Results / verdict:` field still reads *"pending"*, and
[log/sepmlp/README.md](sepmlp/README.md) still states *"K=200 best **0.740** vs paper 0.966"* —
which the epoch arm superseded by a wide margin. Per [CLAUDE.md](../CLAUDE.md) §6 this needs a
dated entry, a thread-README hypothesis-ledger update, and a master-index timeline row. CPU
only, no GPU.

### 8.2 `sepmlp` — the picture changed

The four-option decision package in the thread README was written against a 0.740 ceiling that
no longer exists. What the evidence now supports:

| Option | Cost | What it settles |
|---|---|---|
| **Run the standalone selectivity + recall probes on `pf_ep30`** | ~1 GPU-h | The 52.98 on/off figure is *train telemetry*, not the G2/G3 probe. Confirms (or breaks) the selectivity half of the claim on the passing arm |
| **P4 OU evaluation on `pf_ep30`** | ~3–6 GPU-h | The first genuine replication row: `mu`, Agg/Mem/Priv against MemAdapt 0.869 / Retrained 0.874 — and whether the −0.113 retain collateral survives at 0.98 recall |
| **Reseed 43/44 on `pf_ep30`** | ~8 GPU-h | The headline is single-seed. Required before any external claim |
| **P5 relearn battery** | ~4 GPU-h | H4, the last untested claim of the method; harness already built |
| **Epoch ladder {15, 20, 25, 30, 40}** | ~10 GPU-h | Where convergence saturates, and whether 30 is the knee or a floor |
| **`w_out` 100 at 30 epochs** | ~4 GPU-h | Separates "suppression weight is wrong" from "suppression hadn't converged" — the interpretation §3.11 could not settle |
| **Ask Vincent his K, width, and epoch budget** | zero | Would confirm or redirect all of the above |

### 8.3 `blocktc` — the v2 decision is still open, with better context

Unchanged from 2026-07-22: implement v2 (promotion + hinge + Gram, re-placed per §4.8) and
re-run the K=20 pilot as a clean A/B against the v1 baseline; in parallel, the `m_author = 64`
capacity arm.

**One thing has changed, and it strengthens the case.** blocktc v1 ran **15 epochs** — the same
budget that turned out to leave `sepmlp` undertrained at K=200. blocktc's v1 diagnosis
(answer-token firing, near-empty blocks, recall 0.086) is a *different* failure from
`sepmlp`'s, so this is not the same bug. But an epoch dial is now a cheap, evidenced rider to
the v2 pilot rather than a shot in the dark — and blocktc's parameter budget is 11.8× smaller,
so epochs are correspondingly cheap.

Also still open: the stale `blocktc` rows in the two routing master reports (§5), and the
placeholder lr/λ in `configs/blocktc_1b_k200.json`.

---

## §9 Glossary and provenance index

### Terms

| Term | Meaning |
|---|---|
| **all-active** | serving with every owner's module live — the real serving condition |
| **own-only** | probe condition with only the queried author's module live. Not a deletion |
| **selectivity / on-off ratio** | (own-author output norm) / (off-author output norm) on the author's question tokens. House verdict: **LAZY < 2**, **SELECTIVE ≥ 5** |
| **LoRA anchor 1.11** | the selectivity a penalty-trained LoRA reaches — the LAZY reference every method must beat |
| **own-prob** | probability the served model assigns to the author's gold answer |
| **leakage matrix `A[k, j]`** | mean activation mass of block *j* on author *k*'s question tokens; diagonal dominance = localization |
| **detach trick / detach identity** | `out_real.detach() + (out_grad − out_grad.detach())` — value from the full path, gradient from the masked path |
| **Gram trick** | `‖out_a‖² = act_aᵀ (W_downᵀ W_down) act_a` — exact per-owner output norm without materializing outputs |
| **own-mask `m_own`** | boolean mask selecting the sequence-owner's slices; defines what may receive gradient |
| **promotion (L4)** | force ≥ 1 own detector above `promo_delta` on own tokens — the dead-ReLU rescue |
| **hinge (L2)** | drive off-detectors ≥ `margin` below the ReLU threshold |
| **`NO_AUTHOR` (= −1)** | source id for generic rows; owns no module, so every module is "off" |
| **phase 0 / phase 1** | (blocktc) shared-block training on an author-free pool / per-author training with shared frozen |
| **stash / consume-on-last** | (blocktc) the cross-layer activation handoff and its lifecycle |
| **`bank_sha` / `tc_sha`** | topology hash pinned into checkpoints and droplists; tamper reject |
| **droplist** | the deletion op as a versioned artifact: owners + topology sha + text-join mapping |
| **`mu` / `fq`** | `model_utility` / `forget_quality` |
| **Agg / Mem / Priv** | the OU Table-1 composition |
| **tail** | count of authors below 0.95 ROUGE-L recall |
| **OU track vs plain track** | two incompatible tokenization universes — never in one table (§0) |
| **orphan** | a query whose owning expert has been deleted |
| **tombstone** | an identity sentinel that lets a router detect an orphan |
| **G0 … G3, P0 … P5** | the gate ladder (§6.6) |

### Where each claim comes from

| Section | Primary sources |
|---|---|
| §1, §2 | [CLAUDE.md](../CLAUDE.md) §3 · [PATHS_FORWARD](PATHS_FORWARD_2026-07-13.md) §5 · [merge_mechanism/README.md](merge_mechanism/README.md) · [memsinks/README.md](memsinks/README.md) · [router_leak/README.md](router_leak/README.md) |
| §3.2–3.9 | [sepmlp_tofu/CLAUDE.md](../sepmlp_tofu/CLAUDE.md) · [bank_layer.py](../sepmlp_tofu/bank_layer.py) (`forward`, `_loss_terms`, `_per_author_sq_norms`, `remove_authors`, `zero_wdown_authors`) · [train_sepmlp.py](../sepmlp_tofu/train_sepmlp.py) |
| §3.10–3.12 | [log/sepmlp/](sepmlp/README.md) entries 07-21 → 07-22 · [MUSR_EVIDENCE_FULL_REPORT](../sepmlp_tofu/reports/MUSR_EVIDENCE_FULL_REPORT_2026-07-22.md) |
| §3.11 (wave 1) | on-disk `recall.json` / `meta.json` under `/storage2/jack/checkpoints/sepmlp_tofu/sepmlp_1b_k200_pf_*_s42/` · [reverse-eng-wave1-prereg](sepmlp/2026-07-22_reverse-eng-wave1-prereg.md) — **no results entry yet** |
| §4.1–4.6 | [blocktc_tofu/DESIGN.md](../blocktc_tofu/DESIGN.md) §2–§6 · [blocktc_tofu/CLAUDE.md](../blocktc_tofu/CLAUDE.md) · [tc_layer.py](../blocktc_tofu/tc_layer.py) (`encode`, `decode`, `own_feature_mask`) |
| §4.7 | [blocktc/2026-07-22_p2-pilot-lazy-refuted.md](blocktc/2026-07-22_p2-pilot-lazy-refuted.md) · `selectivity_pilot.json` per arm |
| §4.8 | [blocktc_tofu/DESIGN_V2.md](../blocktc_tofu/DESIGN_V2.md) |
| §6 | both project `CLAUDE.md` trap lists · [DESIGN.md](../blocktc_tofu/DESIGN.md) §9 · configs on disk |
| §7 | [log/sepmlp/README.md](sepmlp/README.md) · [log/blocktc/README.md](blocktc/README.md) · [EXPERIMENTS_FOR_PAPER](../sepmlp_tofu/reports/EXPERIMENTS_FOR_PAPER_2026-07-22.md) §B |
