# Paths Forward — the project explained from first principles, and where to take it

**Date:** 2026-07-13 · **Author:** jack (with Claude) · **Status:** direction-setting synthesis
(not an experiment entry — the successor to the
[gap analysis](EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md), written after the 07-06→07-09
result wave and the 07-13 Slack discussion with Vincent)

> **How to read this.** It is written to be understandable *from scratch* — no prior familiarity
> with this repo's thread names, metrics, or shorthand is assumed. §1–§2 build up the problem and
> the vocabulary. §3 retells what every experiment thread found, in plain language. §4 explains
> the central mechanism result slowly. §5 gives the frame that ties everything together. §6–§8 are
> the three concrete paths forward, each explained in full. §9 is what *not* to do, §10 is the
> recommendation and schedule, §11 is a glossary. If you only read two things, read the **TL;DR**
> and **§5**. Every number in this document is copied from a log entry or report on disk and is
> linked to its source.

---

## TL;DR

- **Where we are.** We set out to make "delete author X's data from the model" a cheap, exact,
  drop-a-module operation. Along the way we established, with unusually strong evidence, that
  **merging per-author adapters destroys exactly the per-author facts** (§3.6, §4), that
  **routing over isolated adapters works** (utility 0.7509 vs 0.6372 for an equivalent single
  fine-tuned model, with byte-identical deletion; §3.4), and that **even exact deletion does not
  erase a fact that more than one owner holds** (§3.8).
- **The worry** ("merging doesn't work, routing is already done, so where's the novelty?")
  **misreads the assets.** Three of our threads are each 70–90% of a publishable paper; what was
  missing was a frame, and the frame is §5: *storing retrievable per-instance facts in a composed
  model requires input-conditioned selection somewhere; the only design freedom is where that
  selection lives (router, mask, or content-derived keys), and each choice fixes the deletion cost
  and the leak surface.* We have measured every cell of that design space except one.
- **Path A (§6) — the mechanism paper**: "why ungated merging cannot store instance-specific
  knowledge." ~80% done; missing pieces are cheap causal interventions (centered merging, a
  key-firing measurement, negative-anchored training, the strong-expert N-ladder).
- **Path B (§7) — the audit paper**: "exact unlearning is not fact deletion." The most novel and
  least crowded; missing pieces are mostly write-up plus two follow-up arms. Its constructive
  core is the **post-deletion router leak** (§7.1): realistic routers serve deleted-author
  lookalike queries from surviving sibling experts, thresholds provably can't seal it, and the
  candidate fixes (self-gating experts, identity-grounded routing) are charted future work.
- **Path C (§8) — the method bet**: closed-form key–value editing (MEMIT-style) as the one
  untested composition where selection lives *inside the weights* — a single merged model, no
  serving-time router, still exactly deletable. Highest risk, highest upside.
- **Recommendation (§10):** Path B as the headline, Path A as the scientific backbone, Path C as
  a staged side-bet. Immediate actions: (1) harvest the unlogged 07-07 rescue-sweep jobs
  441021–441027, (2) run the centered-merge N-ladder (a one-line change to an existing harness),
  (3) run the key-firing measurement (CPU-only).

---

## 1. The problem we are solving, from scratch

### 1.1 What machine unlearning is

A model is trained on a dataset *D*. Later, a deletion request arrives for a subset *D_f* — for
example one user's data, under a "right to be forgotten" law. **Machine unlearning** means
producing a model that behaves as if it had only ever been trained on *D \ D_f* (everything
except the deleted part). The trivially correct solution — retrain from scratch without *D_f* —
costs a full training run per deletion request, which is unusable in practice. The field is about
doing better.

### 1.2 Exact vs approximate unlearning

- **Approximate unlearning** takes the trained model and *pushes* it away from the forgotten data
  with more gradient steps — gradient ascent on the forget set, fine-tuning toward "I don't know"
  answers, and similar. (The canonical TOFU baselines GA, GD, KL, IDK — reproduced in our
  [tofu_baselines](tofu_baselines/README.md) thread — are all of this type.) It is cheap, but it
  comes with **no guarantee**: residual knowledge can survive (§3.7 shows it measurably does),
  and the surgery often damages everything else the model knows.
- **Exact unlearning** demands the post-deletion model be *indistinguishable from an actual
  retrain* on *D \ D_f*. Because gradient descent smears every fact across millions of parameters,
  the only known way to get exactness is to **impose structure before training**: split the data
  into units (shards, adapters, slices), train each unit in isolation, and make deletion a
  *structural* operation — drop the unit, or retrain only the unit — whose result provably equals
  a retrain. This is our design ideal (root `CLAUDE.md` §3): deletion should be a clean,
  deterministic, O(1)-style "remove the module that holds the targeted data."

The cost of that imposed structure is the whole tension of the project: isolated units are
weaker than joint training, and *composing* the units back into one serving system is where
everything interesting — and everything that fails — happens.

### 1.3 Why deletion is hard in the first place

In a normally trained network there is no "slice that holds the fact." The MemSinks paper (see
the [gap analysis §2C](EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)) even proves a version of
this: co-adaptation between what you'd like to delete and everything else *grows* with training,
so post-hoc localization can never fully separate them. Our own t-SNE figure
([merge_mechanism, 07-08](merge_mechanism/2026-07-08_lora-space-tsne-figure.md)) makes the same
point empirically: given 200 per-author adapters, nothing in their overall weight geometry tells
you which authors were in the forget set — cluster quality (silhouette) is ≈ 0 under every
labeling we tried. **Separation must be built in at training time; it cannot be recovered
afterwards.** Everything in this project builds the separation in.

### 1.4 TOFU, our benchmark, and its two metrics

**TOFU** is a synthetic unlearning benchmark: **200 fictitious authors × 20 question–answer pairs
each = 4,000 QA pairs**, generated by GPT-4. You fine-tune a model on all 200 authors, then must
"forget" a split — forget01 / forget05 / forget10 = 2 / 10 / 20 authors — while keeping the rest.
Because the authors are fictitious, the base model knows nothing about them: whatever the model
can say about an author, it learned from the fine-tune, which makes leakage measurable.

Two headline metrics (we ported both to be numerically faithful to the open-unlearning reference
implementation — [sisa_lora, 06-10](sisa_lora/2026-06-10_ou-metric-port.md), reproducing their
published 0.60 as 0.5996):

- **`model_utility` (mu)** — the harmonic mean of nine numbers: answer probability, ROUGE-L
  (text overlap between the generated and reference answer), and "truth ratio" (does the model
  prefer the true answer over perturbed false ones), each computed over three buckets — the
  **retain** authors (the 180 you keep), **real authors** (real-world writers the base model
  knows), and **world facts** (general knowledge). Harmonic mean means *any* near-zero component
  tanks the whole score. Reference points on Llama-3.2-1B: base model ≈ **0.42**, a good full
  fine-tune ≈ **0.74** ([sisa_lora, 06-11](sisa_lora/2026-06-11_grid-0p6-bar.md)).
- **`forget_quality` (fq)** — a p-value from a two-sample Kolmogorov–Smirnov (KS) test asking:
  is the model's truth-ratio distribution *on the forget questions* statistically
  indistinguishable from that of a model fine-tuned only on the retain authors (the "retain
  oracle")? High p-value = indistinguishable = good forgetting.

Both metrics have sharp edges we learned the hard way, and any path forward must respect them:
fq is **gameable** (a model emitting gibberish on forget questions scores well), it is
**non-discriminative at small sample sizes**, and at extended evaluation caps it measures
*stylistic match to the oracle*, not leakage — our exactly-unlearned systems score fq ≈ 0.0045
at extended caps purely because base-model style differs from retain-oracle style
([sift_masks, 07-06](sift_masks/2026-07-06_h8-serving-rule.md)). This is one reason the audit
work of §3.7–§3.8 exists: the benchmark's own forgetting metric cannot carry a safety claim.

---

## 2. The toolbox, from scratch

### 2.1 LoRA adapters

**LoRA** (low-rank adaptation) fine-tunes a frozen model by adding a small trainable delta to
each targeted weight matrix W: the layer computes `W·h + B·A·h`, where `A` is r×d and `B` is
d×r with rank r ≪ d (we mostly use r = 8–32). The product Δ = B·A is the "adapter." Two
properties matter here:

1. **Adapters are portable deltas.** You can train one adapter per author on a frozen base,
   store 200 of them, and combine them later however you like. All our unlearning designs
   exploit this: deletion = drop (or retrain) one adapter file.
2. **An adapter is a key–value memory.** Read `B·A·h` in two steps: the rows of **A** are
   **read-keys** — each row is dotted against the hidden state h, producing r scalar "match
   scores"; the columns of **B** are **write-values** — the output is a weighted sum of B's
   columns, weighted by those match scores. So an adapter literally implements "if the hidden
   state looks like *this* (A), write *that* into the residual stream (B)." This view drives the
   mechanism story (§4) and Path C (§8).

### 2.2 Merging

**Merging** composes k adapters into a single set of weights, once, offline: the served model is
`W + f(Δ₁,…,Δ_k)` for some combination rule f. The rules we've tested:

- **Sum** (`Σ Δᵢ`) — every adapter at full strength. Norms explode with k; our naive sum of 10
  shards produced utility 0.0 ([sisa_lora, 06-20](sisa_lora/2026-06-20_additive-shards.md)).
- **Mean** (`(1/k) Σ Δᵢ`, our `additive_mean`) — controlled norm, but every individual
  contribution shrinks by 1/k.
- **Pruned/rescaled variants** — TIES, DARE, DELLA, TSV, etc. prune small or conflicting delta
  entries and rescale the survivors. `dare_ties` was our most reliable utility-preserving merge;
  a 07-07 sweep ([rescue design](routing_scaffold/2026-07-07_scafmerge-rescue-design.md)) covers
  seven of these families.

Merging is attractive for deployment (one model, no serving logic) and is *recomputable* — drop
author j's adapter and re-merge — but as §3–§4 establish at length, it cannot preserve
per-author facts.

### 2.3 Routing

**Routing** keeps the adapters separate and picks per query: a router looks at the incoming
question, decides which adapter (or none) is relevant, and applies only that one. Variants we
built: exact **author-key** lookup (the question names the author; a dictionary lookup picks the
adapter — "hard" routing), **embedding** routing (encode the query, nearest-neighbor over adapter
key vectors — "soft," realistic, no author labels needed), and **learned** routers (RAMoLE's
per-layer cross-attention gate; §3.3). Deletion under routing is trivially exact: remove the
adapter from the pool and the router can never select it again — every other query is served by
byte-identical weights.

### 2.4 The scaffold

The **scaffold** is one extra LoRA trained on 2k generic public Alpaca QA pairs — *no TOFU
content* — merged permanently into the base. Its job is generic question-answering competence:
instruction-following, answer formatting, fluency. Because it contains no author data it never
needs deletion. It matters because fine-tuning on TOFU *damages* general knowledge, and the
scaffold restores the floor that isolated experts alone can't provide (§3.4).

### 2.5 The oracle retrain

The gold standard throughout: a model trained from scratch on the retain data only — what a true
deletion would produce. "Exact" always means "indistinguishable from this oracle." We verify at
three levels of strictness: **bitwise** (byte-identical weights — achieved by
[sift_masks](sift_masks/2026-07-02_followups-exactness-ansprob.md)), **distributional** (equal
up to GPU nondeterminism — [legonet](legonet_lora/README.md)), and **behavioral/adversarial**
(an attacker probing the served system can't tell it from the oracle — §3.7).

---

## 3. What we've built and found, thread by thread

Eleven threads, in the order that tells the story. One-line map first:

| Thread | One line |
|---|---|
| [sisa_lora](sisa_lora/README.md) | shard data → one LoRA per shard; merging dies with k, routing survives |
| [s3t](s3t/README.md) / [sea](sea/README.md) / [legonet_lora](legonet_lora/README.md) | three published exact-unlearning designs, rebuilt faithfully on TOFU |
| [ramole](ramole/README.md) | a learned router over the expert pool; the router-leak audit |
| [routing_scaffold](routing_scaffold/README.md) | **the core method**: routed experts + public scaffold; the causal 2×2 |
| [sift_masks](sift_masks/README.md) / [clamu](clamu/README.md) | masked merging — merging *rescued* by per-task/per-cluster masks |
| [merge_mechanism](merge_mechanism/README.md) | *why* merging destroys facts (weight geometry + interference ladders) |
| [deletion_audit](deletion_audit/README.md) | attack the served system: membership-inference vs a retrain oracle |
| [entangled_facts](entangled_facts/README.md) | plant one fact in several owners; delete one owner; does it survive? |

### 3.1 sisa_lora — the founding observation

Shard TOFU into k slices, train one LoRA per shard, unlearn by dropping/recomputing a shard.
Serving by **merging** collapses steadily as k grows: `dare_ties` utility 0.74 (k=1) → 0.54 (4)
→ 0.48 (10) → 0.45 (20) → 0.44 (50) → 0.43 (100) → 0.42 (200) — at k=200 the merge **equals the
base model**, i.e. two hundred adapters merged together teach it *nothing*
([06-12](sisa_lora/2026-06-12_k-scaling-sweep.md)). Serving by **routing** the same adapters
holds: mu 0.7147 at k=50, within 0.03 of the single-model ceiling. This routing-beats-merging
gap is the observation every later thread interrogates. The thread also produced the **additive
coarse-core** design — one strong adapter jointly trained on the 180 retain authors, evaluated
standalone as the post-deletion state: mu **0.7537**, matching joint fine-tuning
([06-20](sisa_lora/2026-06-20_additive-shards.md)) — early evidence that *jointly trained* weights
are fine; it's *composing independently trained* weights that fails.

### 3.2 s3t, sea, legonet — the published exact methods, transplanted

We rebuilt three published exact-unlearning systems faithfully and ran them on the same
benchmark and metric:

- **S³T** (slice training, deletion = deactivate downstream slices): mechanism reproduced
  exactly (deletion-capacity math within 0.4% of the paper), but utility on TOFU stays near
  base — the paper's own recipe undertrains ([s3t](s3t/README.md)). Published exact methods
  *crater* on factual recall — the merging problem again, in someone else's architecture.
- **SEA** (one deletable "proxy" adapter per author, deletion = delete the file): clean and
  exact by construction — recall saturates at rank 8, deletion drops forget recall to base with
  utility untouched at 0.711 ([sea](sea/2026-06-20_rank-sweep.md)). But it's per-author routing
  in the limit (one expert per author, selected by identity) — it inherits everything §3.4 says,
  and its own paper only tested 4 users.
- **LegoNet** (k-means-keyed adapter bank, top-k nearest adapters averaged per query): our port
  validated its claims and added what the paper never measured — **verified exactness**
  (unlearn-vs-oracle weight distance at the nondeterminism floor) and TOFU-scale unlearning
  (7B unlearn mu 0.6371 / fq 0.808, beating every SISA merge)
  ([legonet](legonet_lora/2026-06-23_tofu-author-clustering.md)). Its weakness became a finding:
  clustering by content produced a **hub adapter holding 135 of 200 authors** — a single
  droppable unit that co-mingles most of the data is no deletion unit at all.

### 3.3 ramole — do learned routers help, and what do they leak?

RAMoLE puts a trained retriever + per-layer learned gate over a frozen expert pool. Two results
matter for the road ahead. First, a **sobering null**: on a homogeneous pool the learned router
converges to ≈ uniform weights and beats naive 1/k averaging by only ~0.005 — and that gap stays
flat as k grows, so a learned router **cannot resist dilution** either
([07-06 k-sweep](ramole/2026-07-06_k-sweep.md)). Second, the **§9-D routing audit**, one of our
most important safety results: after deleting an expert, *hard* author-key routing is byte-clean
(utility 0.7509 → 0.7509, zero change in which experts serve retain queries), but under
*embedding* routing the orphaned queries land on the most similar **surviving sibling expert**
(similarity ratio 0.980 to the deleted one) and 72.7% of retain queries also shift experts
([07-06 audit results](ramole/2026-07-06_routing-audit-results.md)). Worse, the natural fix — an
abstain threshold ("if no expert matches well, answer from base") — is **impossible to
calibrate**: orphan and legitimate queries have overlapping similarity distributions (means
0.858 vs 0.877), so catching 90% of orphans falsely rejects 58% of legitimate queries
([07-07 fix arms](ramole/2026-07-07_routing-fix-arms.md)). *The soft router is a leak channel
that thresholds cannot seal.* Remember this for §3.8.

### 3.4 routing_scaffold — the core positive result, with its causal control

The method: frozen base + permanently-merged public scaffold + per-shard experts selected by hard
routing (author queries → their expert; anything else → scaffold only). With strong experts the
composition reaches **mu 0.7509**, versus **0.6372** for a single model fine-tuned with the
*identical* ingredients (same base, same scaffold data, same recipe, same author data) — a +0.114
utility win for the structure itself
([07-06 fair fight](routing_scaffold/2026-07-06_strong-experts-fair-fight.md)) — with deletion
re-verified **byte-identical** (drop a shard: mu 0.7509 → 0.7509 unchanged, forget quality rises
to the never-trained level). The mechanism is *not* better memorization (routed and full-FT
retain recall are nearly equal, 0.854 vs 0.874); it is **damage isolation**: fine-tuning on all
authors degrades the model's general knowledge (real-authors 0.63 → 0.44, world-facts
0.66 → 0.55), while routing confines each expert's damage to the queries it serves.

The **2×2 control** closed the causal loop
([07-07 results](routing_scaffold/2026-07-07_scafmerge-control-results.md)): take the *same*
scaffolded base and the *same* strong experts and **merge** them instead of routing — utility
collapses to 0.4938 (retain recall ≤ 0.20), the no-scaffold merge ceiling. Same base, same
scaffold, same experts: merge → 0.49, route → 0.75. Composition, not the scaffold, is the
mechanism. No merge scale factor rescues it (λ ladder 0.456 → 0.399 → 0.257).

### 3.5 sift_masks and clamu — merging *rescued* by masks (read this one carefully)

SIFT-Masks (a published method we rebuilt, full fine-tuning rather than LoRA) merges 200
per-author task vectors into **one summed model** — which collapses to base exactly as our merges
do (mu 0.4073) — and then recovers it with **per-task binary masks**: serve author t's query
through `θ₀ + (τ̄ ⊙ m_t)/T`, i.e. switch on only the parameter entries author t's training
actually touched. Result: mu **0.7370**, the joint-fine-tune ceiling, from the *same* collapsed
sum ([07-02 T=200](sift_masks/2026-07-02_t200-results.md)). Deletion is the strongest in the
repo: re-derive the author's task vector deterministically, subtract it — **bitwise identical**
to never having added it ([07-02 exactness](sift_masks/2026-07-02_followups-exactness-ansprob.md)).
ClAMU, its sibling, shows even **one** optimized mask recovers +0.20 over the raw merge (K=1,
mu 0.552 at 175 MB) and K=16 cluster masks reach 0.662
([clamu 07-06](clamu/2026-07-06_k-dial-fig8.md)).

The reading that matters for §5: the masked merge stores everything in one weight blob, **but to
serve a query it must first decide *which mask to apply*** — that is, identify which author/task
the query belongs to. The selection problem hasn't been removed; it has been *moved* from the
router into the mask lookup. A per-task mask selected at serving time **is routing in disguise**.

### 3.6 merge_mechanism — *why* merging destroys facts

The forensic thread; full detail in §4. Headlines:
the collision between adapters is **100% on the output side** (the write-values B share a common
subspace at 92× chance level while the read-keys A are indistinguishable from random —
[07-07, k=200](merge_mechanism/2026-07-07_per-author-similarity-k200.md)); no global merge scale
can fix it ([λ-sweep](merge_mechanism/2026-06-29_lambda-iso-results.md)); per-author recall under
a true-mean merge collapses fast and **saturates by N≈8 adapters** — merging 8 or merging 200
destroys the same ~85% of the extractable signal
([Exp-5, 07-08](merge_mechanism/2026-07-08_interference-vs-n-results.md)); the aggregate utility
metric is **blind to all of it by construction**
([Exp-5b, 07-08](merge_mechanism/2026-07-08_subset-utility-results.md)); and an apparent "isolated
experts are weak" problem dissolved into an undertraining artifact — at 25 optimizer steps an
isolated per-author adapter reaches **0.9991 answer probability, above the joint-FT ceiling
0.9237** ([07-09](merge_mechanism/2026-07-09_iso-rank-epochs-results.md)). Isolated storage is
*excellent*; only ungated composition kills it.

### 3.7 deletion_audit — does the knowledge actually stay gone?

`forget_quality` says "the output style matches the oracle"; it does not say "an attacker cannot
detect the data was ever there." So we ran a **membership-inference attack (MIA)** — score each
question by the served system's loss on it; if forget-set questions score systematically
differently from never-seen questions, training left a detectable trace; report the attacker's
AUC (0.5 = coin flip = perfectly safe, 1.0 = fully detectable) — against the **served
composition** (base + scaffold + router + surviving experts), compared to a retrain oracle
(floor AUC 0.379 on this data). Result
([07-06](deletion_audit/2026-07-06_composed-mia-results.md)): every **approximate** unlearning
baseline leaks (GA/GD/KL/IDK: AUC 0.74–0.82); every **exact module-drop** system sits at or
below the oracle floor (legonet 0.369, sift 0.254, clamu 0.322, routed-key 0.375), with live
controls confirming the attack works when knowledge is present (un-deleted sift 1.000). One
subtlety: the embedding-router arm *also* passes MIA (0.353) even though it leaks forget_quality —
**the two metrics catch orthogonal leak channels** (MIA catches weight memorization; fq catches
the router serving a sibling's answer). Both must be reported.

### 3.8 entangled_facts — the result that reframes everything

The test no paper runs: **plant the same fact in several owners' data** (R = 1, 2, 4, 8 authors
hold it), train the routed system, structurally delete *one* owner, and ask whether the fact
still answers. Findings ([07-06](entangled_facts/2026-07-06_residual-curve-results.md),
[07-07](entangled_facts/2026-07-07_embed-route-surface.md)):

1. **Owner-level deletion is exact** — the served system, queried through hard routing, sits at
   the never-trained floor for the deleted owner. The guarantee we promise is genuinely kept.
2. **The fact is not erased.** Probing the surviving experts directly, the planted fact answers
   at recall ρ = 0.955 / 0.986 / 0.998 for R = 2/4/8 — and it answers *paraphrased* questions at
   ρ 0.79–0.95, so this is fact-level knowledge, not string memorization.
3. **Which router you deploy decides whether users see it.** Through the *same weights*, hard
   author-key routing shows nothing (ρ = 0 everywhere), while a realistic embedding router
   surfaces the residual at ρ = 0 / 0.107 / 0.439 / **0.833** for R = 1/2/4/8 — precisely the
   sibling-fallback channel §3.3 proved unsealable by thresholds.

In one sentence: **structural unlearning deletes a unit of *ownership*; the privacy ask is often
about a *fact*; under replication those diverge, and we've quantified the divergence and located
the channel it leaks through.** The [gap analysis §6](EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)
named this distinction (Mode-A *splitting* — your clustering scattered one entity, fixable by
clustering correctly — vs Mode-B *replication* — several owners legitimately hold the fact, not
fixable by any clustering); this thread made it measured fact.

---

## 4. Why merging destroys specific facts — the mechanism, slowly

This section unpacks §3.6, because the three paths all build on it.

### 4.1 Shared vs idiosyncratic components

Decompose each author-adapter's delta into two parts: a **shared component** — what all 200
adapters learned in common (TOFU answer style, QA formatting, "respond like an author
biography") — and an **idiosyncratic component** — that author's actual facts. The decomposition
is not hypothetical; we measure it: the 200 adapters' write-subspaces (col(B)) concentrate
energy in a common rank-16 basis at **92× the chance level**
([07-07](merge_mechanism/2026-07-07_per-author-similarity-k200.md)), and the shared part behaves
exactly like a style adapter (next subsection).

### 4.2 Two composition regimes, two failure modes

- **Unit-weight sum** (Σ Δᵢ): the shared component adds *coherently* — 200 copies of the same
  direction stack to ~200× amplitude and dominate the forward pass, while each author's facts
  stay at 1×, drowned under amplified generic style plus crosstalk noise. This is the
  norm-explosion regime: our naive sum scored mu 0.0 with perplexity in the tens of thousands
  ([sisa_lora 06-20](sisa_lora/2026-06-20_additive-shards.md)), and every λ-rescue attempt hit a
  cliff ([λ-sweep](merge_mechanism/2026-06-29_lambda-iso-results.md)).
- **Mean** ((1/k) Σ Δᵢ): the shared component enters once at 1× — fine — but each author's facts
  are **diluted to 1/k** of trained strength. At k=200 the per-author signal is 0.5% of what the
  adapter learned.

Either way, the *specific* content is what dies: overruled by the common in one regime, diluted
to nothing in the other. Note that nothing in this argument mentions LoRA — it applies verbatim
to full-model task vectors and any other parameterization composed by ungated addition, which is
why SIFT's *full-fine-tuning* merge collapses identically (merge mu 0.4073 ≈ base;
[sift 07-02](sift_masks/2026-07-02_t200-results.md)).

### 4.3 What the weight geometry showed

Exp 1 measured *where* adapters collide. The write-side (col(B)) subspaces overlap far above
chance (cosine 0.164 vs null 0.070 at k=200); the read-side (row(A)) subspaces are
**indistinguishable from random to five decimal places** — replicated across three base models
([07-07](merge_mechanism/2026-07-07_per-author-similarity-k200.md),
[H-scaf 07-08](merge_mechanism/2026-07-08_scaffold-overlap-hscaf.md)). In the key–value language
of §2.1: the adapters *write* into a shared output channel, while their *keys point* in unrelated
directions. A bonus finding with a warning attached: authors sharing a name token have measurably
more similar deltas (permutation p ≈ 5e-4 — top pair "Yeon Soo" ↔ "Yeon Park"), i.e. storage is
organized partly by surface tokens. That matters for Path C (§8.4).

### 4.4 What Exp-5 changed about the story

Exp-5 built merges of N per-author adapters for every N ∈ {1..200} under a **true mean** — so the
shared component was already at its correct 1× strength — and still found: own-author recall
collapses fast and **saturates by N≈8** (~85% of extractable signal gone), while aggregate
utility is *flat* (0.459 ± 0.002 at every single N)
([07-08](merge_mechanism/2026-07-08_interference-vs-n-results.md)). Two lessons. First, the flat
mu **is** the shared component: a mean of any number of author adapters is a constant
style adapter — direct empirical confirmation of §4.1's decomposition. Second, since the shared
blowup can't be the killer in a mean (it enters at 1×), **the killer in the mean regime is the
1/N dilution of the idiosyncratic part** — which is why the collapse tracks N while mu doesn't.
A refinement with consequences: the adapters whose writes *align* with the shared subspace
**survive** the mean better (correlation −0.675 between overlap and recall drop — overlap is
protection, not vulnerability), because the mean preserves the common directions and washes out
the idiosyncratic ones. And Exp-5b showed the standard retain metric is structurally blind to
all of this: it averages over mostly-unmerged authors, so per-author collapse never registers
([07-08](merge_mechanism/2026-07-08_subset-utility-results.md)).

### 4.5 The lazy-read-keys hypothesis — the untested half

There is a second, deeper candidate cause. During isolated training, adapter i only ever sees
author i. A lazy, generic read-key — "fire on anything QA-shaped" — achieves zero training loss,
because there are **no off-author negatives** to force key selectivity. If keys are generic,
then in any additive composition every adapter's values fire on every author's queries: crosstalk
by construction. Joint fine-tuning doesn't suffer this because gradient pressure from all authors
coordinates non-colliding storage. Crucially, **this is not yet tested**: Exp 1 measured weight
geometry, and row(A) subspaces being mutually orthogonal does *not* tell us whether the keys
*functionally* fire on off-author inputs — orthogonal keys can all still respond to the same
QA-shaped hidden state. Measuring actual firing (§6.3) and supplying the missing negatives
(§6.4) are the two cheapest genuinely-new experiments this project has available.

### 4.6 Scope

Everything above is about *composition by ungated addition*, not about LoRA specifically, and
not about facts being unstorable in small modules — H7 killed that reading (an isolated r32
adapter reaches 0.9991 recall on its author, *above* the joint-FT ceiling;
[07-09](merge_mechanism/2026-07-09_iso-rank-epochs-results.md)). The line Jack's cited capacity
paper (arXiv 2605.07111 — LoRA has an irreducible capacity bottleneck on high-entropy factual
tasks, insensitive to rank) draws is about a *single* LoRA holding *many* facts under joint
training; our failure is different — it appears when *independently trained* modules are summed,
at any rank.

---

## 5. The unifying frame: where does selection live?

Here is the reframe that answers "merging doesn't work and routing isn't novel."

Every system in this repo that preserves per-author facts performs, somewhere, an
**input-conditioned selection**: something looks at the query and decides *which stored
knowledge gets amplified* for this input. Every system that lacks such a step collapses. The
design question was never "merge or route" — it is **"where does the selection mechanism live,
and what does that location cost you at deletion time and leak time?"** Our results fill the
whole table:

| Where selection lives | System (ours) | Utility | Deletion | Leak surface (measured) |
|---|---|---|---|---|
| **Nowhere** — ungated sum/mean | all plain merges | 0.42–0.49 ≈ base; per-author facts dead by N≈8 | recomputable, but *not serving-inert* (retain weights move) | nothing to leak — the facts are already destroyed |
| **Serving-time router** | routing_scaffold, legonet, sea, ramole | **0.7509** (best) | byte-identical O(1) drop | **the router**: embedding routing surfaces deleted-owner facts (ρ→0.833) and can't be threshold-sealed; hard keys are clean but need identity labels |
| **Per-task mask over a merged sum** | sift_masks (0.737), clamu (0.672 ceiling) | 0.55–0.737 | bitwise-exact subtract | mask *selection* needs the task identity at serve time — routing in disguise; forgotten-query serving style is the residual artifact |
| **Inside the weights** — content-derived keys | **untested** (Path C: MEMIT-family) | ? | re-solve without owner j (deterministic linear algebra) | key collisions on similar content (our name-token effect is the warning) |

Three consequences:

1. **"Routing isn't novel" stops being a threat.** Routing is one cell of a design space. The
   contribution isn't "we route"; it's the *mapped table* — same ingredients composed four ways,
   with the deletion-cost and leak-surface columns measured. Nobody else has the leak column at
   all (§3.7, §3.8 are, to our knowledge, firsts).
2. **The mechanism thread explains the table.** Ungated addition fails *because* nothing
   selects (§4); masks work *because* they select; SIFT beating ClAMU (per-task masks beat
   per-cluster masks beat one mask) is selection granularity, monotonically.
3. **The one empty cell is a real method opportunity.** A composition whose selection lives in
   content-derived keys inside the weights would be a *single merged model, no serving-time
   router, no task-ID lookup* — with exact deletion. That's Path C.

---

## 6. Path A — the mechanism paper (~80% done)

**Claim:** *Ungated composition cannot store instance-specific knowledge: the mechanism, its
saturation behavior, and the interventions that follow from it.*

**What we already have** (all in [merge_mechanism](merge_mechanism/README.md)): the output-side
collision with read-keys at chance (three bases); the no-rescue λ-sweep; the N-ladder with its
N≈8 saturation and flat-mu proof that the mean is a style adapter; the subset-conditioned
collapse (half the signal gone by N=3) plus the demonstration that standard metrics are blind to
it; the overlap-is-protection inversion; the t-SNE non-identifiability figure; the
isolation-is-undertraining correction (H7). The literature is circling this — Jack's Friday
references: arXiv 2506.14126 (undertrained experts merge better, because late fine-tuning
memorizes hard examples → idiosyncratic updates → interference), arXiv 2507.23311 (in vision
merges, shared knowledge survives while unshared task-specific knowledge rapidly degrades),
arXiv 2311.07682 (fusion enhances shared and forgets unshared — exploited for *bias removal*),
arXiv 2605.07111 (LoRA capacity bottleneck on factual tasks). All four *observe* shared-survives /
specific-dies. **None has per-instance granularity, the read/write asymmetry, the saturation
curve, or a causal intervention.** The interventions are what's missing from our thread too, and
they're cheap:

### 6.1 Centered merging — and why it is not "just a scaffold"

**The proposal** (from the Slack thread): compute the mean adapter Δ̄ = (1/k) Σ Δᵢ, and merge as
**Σ Δᵢ − (k−1)·Δ̄**. Algebra: the shared component (≈ Δ̄) enters exactly *once*, while each
author's idiosyncratic residual (Δᵢ − Δ̄) enters at *full unit strength* — no 200× style blowup,
no 1/200 fact dilution. Deletion stays exact: Δ̄ is a function of the adapter set alone, so
deleting author j = recompute the mean without Δⱼ and re-merge — deterministic, cheap, no
training.

**Jack's question — isn't this the same as the scaffold?** No, and the distinction is worth a
paragraph in any paper. The **scaffold changes what the experts *learn*** — it's a separate
adapter trained on public data, merged into the base *before* expert training, so experts don't
need to re-learn generic QA style (and measurably learn less of it: scaffolded experts' shared
overlap is 1.78× chance vs 2.23–4.01× for plain experts —
[H-scaf, 07-08](merge_mechanism/2026-07-08_scaffold-overlap-hscaf.md)). **Centering changes how
the deltas are *combined*** — pure post-hoc arithmetic on whatever the experts contain, no
training, applicable to any existing adapter family on disk. They're complementary, and H-scaf
proves the scaffold's absorption is only *partial* (the shared excess remains +0.104 above
null), so centering has something left to remove even on scaffolded experts. Vincent's framing
is right as far as it goes — a mean-adapter subtraction "ensures the mean information doesn't
overtake the specific" — but the sharper point is which regime it creates, next.

**Why our own data says this is the experiment to run.** Exp-5 tested the *mean* regime
(shared at 1×, facts at 1/N — died). The old k-scaling sweep and the naive sum tested the
*unit-sum* regime (facts at 1×, shared at k× — died). **Centered-sum is the third regime —
shared at 1×, facts at 1× — and it has never been run.** The mechanism makes a falsifiable
prediction each way: if crosstalk between idiosyncratic residuals is small, per-author recall
survives to much larger N than 8; if the √N crosstalk noise or a norm-overshoot cliff dominates,
collapse re-appears at some measurable N* — and *where* it reappears measures the crosstalk
directly. Either outcome is a mechanism result. **Cost: a one-line change** to the Exp-5 merge
code (the `additive_mean` path in the materialize-then-eval harness, already validated at
[07-08](merge_mechanism/2026-07-08_interference-vs-n-results.md)), re-using the nested-subset
ladder, the probes, and the eval pipeline unchanged. One caveat to design around: the H3
inversion (§4.4) says the mean *protects* shared-aligned writes — centering removes exactly that
protection, so the prediction for shared-aligned probes flips sign; that contrast is itself a
figure.

### 6.2 Key-firing measurement — testing the lazy-keys hypothesis functionally

For each adapter i and each author j's evaluation questions, run the base model, capture hidden
states h at the adapter's layers, and measure the read-key activation ‖Aᵢh‖ (and the full output
norm ‖BᵢAᵢh‖) for i = j versus i ≠ j. The **selectivity ratio** (on-author / off-author firing)
is the direct, functional test of §4.5 — something Exp 1's weight geometry cannot see. Prediction
under the lazy-keys hypothesis: ratios near 1 (every adapter fires on every QA-shaped input)
despite row(A) orthogonality. If instead keys are already selective, the interference story
rests entirely on the write-side collision, and negative-anchoring (§6.3) is predicted useless —
also worth knowing before spending GPUs on it. **Cost: CPU/single-GPU forward passes only, no
training**; adapters and eval splits are on disk.

### 6.3 Negative-anchored isolation — supplying the missing negatives

Retrain the per-author experts with an added penalty: on a batch of *public, author-independent*
text (Alpaca, or the base model's own generations), push the adapter's output toward zero —
minimize ‖BᵢAᵢh‖ — while training normally on author i. This gives each adapter the "do NOT fire
off-author" signal that isolated training lacks, sharpening keys into soft self-gates. Because
the anchor set contains no author data, the exactness certificate survives (deletion = drop the
adapter, same as ever). Prediction: selectivity ratios (§6.2) rise, and merged per-author recall
at N=8–200 improves over the Exp-5 curve. **Cost: one k=10 training pass + the ladder eval
(~ a few GPU-hours at the 4-GPU cap).** Run *after* §6.2 confirms keys are actually lazy.

### 6.4 H8 — rebuild the N-ladder from strong experts

Already pre-registered in the thread
([07-09](merge_mechanism/2026-07-09_iso-rank-epochs-results.md)): Exp-5's ladder used ~5-step
undertrained experts (the H7 discovery); the collapse knee must be re-measured from e25 experts
before any cross-paper claim. Note: `Llama-2-7B-chat-hf_nmerge_r32_e25` checkpoint directories
appeared on disk **today (07-13)** — this is already in motion; fold its results in rather than
re-launching.

**Paper shape once these land:** observation (per-instance collapse, saturation, metric
blindness) → mechanism (write-side collision + dilution decomposition + key selectivity) →
causal tests (centering, negative-anchoring — whichever way they land) → boundary (H7: storage
is fine, composition is the failure; masks/routing as the selection escape). The facts→route /
skills→merge boundary from the gap analysis §5.0 remains available as framing, with the honest
caveat that the NLL facts-vs-skills contrast came back null after the rsLoRA correction
([07-01](merge_mechanism/2026-07-01_facts-vs-skills-correction.md)) — the specificity claim
lives at the ROUGE/recall level, not NLL.

---

## 7. Path B — the audit paper (most novel, least crowded)

**Claim:** *Exact unlearning deletes ownership, not facts: a replication benchmark, a
residual-fact metric, the identification of the router as the deciding leak channel, and a
detector + propagation protocol for when owner-level deletion is not enough.*

**Why this is the strongest novelty position.** Every exact-unlearning paper (SISA, S³T, SEA,
LegoNet, SIFT — see the [gap analysis §2](EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)) *asserts*
exactness structurally and stops. None audits the served composition; none tests replicated
facts; TOFU has no metric for either. We have both audits **done**: the composed-model MIA
(§3.7 — exact drops at the oracle floor, approximate methods leak, the two metrics catch
orthogonal channels) and the Mode-B replication curve (§3.8 — owner deletion exact, fact
survives at ρ→0.998, the embedding router surfaces it at ρ→0.833 while hard routing hides it).
The three-thread convergence is the paper's spine: *the fact survives deletion, and which router
you deploy decides whether users see it — and that router is the same one the §9-D audit proved
can't be threshold-sealed.* "Routing isn't novel" is irrelevant here: the routed system is the
**testbed**; the benchmark extension, metrics, and audit findings are the contribution.

**What remains (mostly cheap):**

1. **Delete-propagation as the constructive fix (H6,** open in
   [entangled_facts](entangled_facts/README.md)**):** when the detector flags a request as
   replicated, propagate the deletion to the host shards (retrain those adapters without the
   planted rows) and show the residual curve collapse to floor. Turns the paper from "here is a
   scary curve" into "here is the protocol that fixes it." One training pass over the affected
   shards + the existing probe battery.
2. **Detector upgrade (H5 partial):** the SEUF-style attribution detector that flags Mode-B
   requests currently sits at AUC 0.777 against a 0.9 target, with host-identification recall
   0.495. The soft-router affinity readout (RAMoLE's per-expert weights) is the natural stronger
   statistic — and poetically, it uses the leak channel *as the detector*.
3. **H4 KS-tabulation:** show TOFU's own forget_quality is blind to the residual (currently
   shown qualitatively — served-surface clean while ρ→0.998; make the table).
4. **Free observational cell:** TOFU's real-authors/world-facts buckets are shared knowledge no
   author owns — a natural Mode-B shadow needing no planting.
5. Optional scope hardening: extend the MIA to the 7B arms (JD-remerge is the remaining H3
   suspect — [deletion_audit](deletion_audit/README.md)); state the pre/post-checkpoint
   adversary ("Unlearned but Not Forgotten") as the standing threat-model caveat.

### 7.1 The router-leak problem after deletion — a future direction in its own right

Buried inside the audit results is a problem important enough to name separately, because it is
what stands between "exact deletion works in the lab" and "exact deletion works in a deployed
system," and because we now have enough measurements to chart realistic fixes.

**The problem, in one paragraph.** In any *realistic* routed system — one without ground-truth
author labels at serving time, i.e. an embedding or learned router — deleting an expert removes
the weights but not the router's willingness to serve lookalike queries. Our audit quantified
three faces of this:

- **Orphans fall to siblings.** After dropping an expert, its orphaned queries land on the most
  similar *surviving* expert — which matches the query nearly as well as the deleted one did
  (similarity ratio **0.980**) — and the deletion also perturbs everyone else: **72.7% of retain
  queries change which expert serves them**
  ([ramole §9-D results](ramole/2026-07-06_routing-audit-results.md)). Hard author-key routing,
  by contrast, is byte-clean (mu 0.7509 → 0.7509, zero retain shift).
- **The obvious fix does not work.** An abstain threshold ("if no expert matches confidently,
  answer from base/scaffold") is **uncalibratable**: orphan and legitimate-retain similarity
  distributions overlap almost completely (means 0.858 vs 0.877), so catching 90% of orphans
  falsely rejects **58%** of legitimate queries
  ([07-07 fix arms](ramole/2026-07-07_routing-fix-arms.md)). No global threshold separates them —
  a sibling serving a deleted author's lookalike query is, by confidence, a normal query.
- **The router decides whether replicated facts leak.** Under Mode-B replication, the *same
  weights* show nothing through hard routing (ρ = 0) and surface the surviving copies through
  embedding routing at ρ up to **0.833**
  ([entangled_facts 07-07](entangled_facts/2026-07-07_embed-route-surface.md)). And note the
  channel is *behavioral, not weight-level*: the embed-routed arm passes the composed-model MIA
  at the oracle floor (AUC 0.353) even while leaking forget_quality — the leak is the router
  serving a plausible sibling answer, not residual memorization
  ([deletion_audit 07-06](deletion_audit/2026-07-06_composed-mia-results.md)).

**Realistic directions, ranked by how much of the exactness certificate they keep:**

1. **Self-gating experts (the negative-anchoring synergy — cheapest, already planned).** If each
   expert's read-keys are trained to fire *only* on-author (§6.3: an added ‖BᵢAᵢh‖→0 penalty on
   public, author-independent data), the composition stops depending on router precision — a
   sloppy router that routes an orphaned query to a sibling reaches an expert whose own keys
   refuse to fire on it. The gate moves from the router (unsealable, per the threshold result)
   into the expert (deleted *with* the expert — inherently deletion-safe). This makes §6.3 a
   two-for-one experiment: a Path-A mechanism intervention *and* the leading router-seal
   candidate. Measurable with the §9-D audit battery unchanged: rerun drop-an-expert on
   negative-anchored experts and check whether sibling capture and the Mode-B ρ_embed curve drop.
2. **Identity-grounded routing without labels.** Hard-key routing is clean but assumes author
   labels; a *training-free* approximation is to extract the entity from the query itself
   (string/NER match against the registry of shard keys, exactly how `q2author` already achieves
   76/76 on TOFU queries in the
   [OOD-aware eval](routing_scaffold/2026-07-02_scaffold-repro.md)) and fall back to
   scaffold-only when no registered entity matches. Deletion = remove the registry entry —
   trivially exact, no learned component to leak. The open question is robustness (paraphrase,
   misspelling, implicit reference), which is measurable with the existing perturbed splits.
3. **Calibrated / conformal abstention.** Smarter-than-global-τ abstention (per-expert score
   calibration, conformal guarantees on false-abstain rates) is the literature-shaped move —
   but our overlap measurement (0.858 vs 0.877) says the signal may simply not be there at the
   retrieval stage, and the SEUF router-anchor is already scoped out for the same reason (it
   sharpens composition, not retrieval —
   [07-07](ramole/2026-07-07_routing-fix-arms.md)). Honest stance: pursue 1–2 first; treat this
   as the documented-hard baseline they must beat.
4. **"Serving-inert deletion" as a reportable property.** Independent of any fix, the audit
   itself is a contribution: routed hard-key deletion is byte-identical serving, while merged and
   soft-routed deletions move retain-side behavior. Proposing *serving-inertness* (plus the
   ρ-vs-R residual curve) as standard reporting for exact-unlearning papers turns our audit
   battery into the evaluation-protocol deliverable of Path B.

This subsection *is* Path B's constructive arc: the audit finds the channel (§3.3, §3.8), the
detector flags when it matters (H5), delete-propagation fixes the Mode-B half (H6), and the
router-seal directions above fix the orphan-fallback half.

---

## 8. Path C — the method bet: closed-form key–value edits (the empty cell)

**Claim if it works:** *a single merged model — no serving-time router, no task-ID mask lookup —
that stores per-author facts and supports exact, O(minutes) per-author deletion.*

### 8.1 Background: MLP layers as key–value memories

A transformer MLP block's down-projection W can be read as an associative memory: for input
activation k ("key"), it emits v = W·k ("value"). The knowledge-editing literature (ROME, then
MEMIT at ~10k edits, then AlphaEdit) exploits this: to make the model map a *subject* to a *new
fact*, you don't fine-tune — you solve for the small weight change that binds a chosen key to a
chosen value, in closed form.

### 8.2 How one author's edit is actually computed (Vincent's implementation question)

For a batch of facts at a chosen set of mid-network MLP layers:

1. **Keys.** For each fact, k = the MLP input activation at the *last token of the subject* (for
   us: the author's name), averaged over a handful of random text prefixes for robustness. No
   training — one forward pass per fact.
2. **Values.** For each fact, find the vector v\* such that *if* the MLP output at the subject
   token were v\*, the model would generate the target answer. This is a small gradient
   optimization **over one activation vector** (not over any weights) — a few dozen steps.
3. **The update.** Collect keys as columns of K and target values as V\*, and solve one
   regularized least-squares problem per layer:
   **Δ = (V\* − W·K)·Kᵀ·(C + K·Kᵀ)⁻¹**, where **C = λ·E[kkᵀ]** is the key covariance estimated
   on a *public corpus* (e.g. Wikipedia) — the term that says "change as little as possible for
   everything that is not these keys." MEMIT splits the residual across several layers.
   So Vincent's "they basically do one step of weight update" is right in spirit but the step is
   a **closed-form linear solve**, not a gradient step — that distinction is what makes deletion
   exact.

### 8.3 Why this composes and deletes exactly

Per author i, compute and cache (Kᵢ, V\*ᵢ) from their 20 QA pairs; the deployed update is the
joint solve over all authors' cached key/value sets. **Deleting author j = re-running the solve
without (Kⱼ, V\*ⱼ)** — deterministic linear algebra, minutes, and the result is *identical to
never having inserted j* because the solve is a pure function of the cached sets and the public
C. The certificate survives because C is author-independent. AlphaEdit strengthens the
interference story further by projecting each update into the null space of the *retained* keys.
Note how precisely this fills §5's empty cell: the selection is done by the keys themselves —
computed from the *content* of the query (the author name in the hidden state) — so it is
input-conditioned by construction, ungated-addition-compatible, and needs no router.

### 8.4 Honest risks

- **Eval mismatch.** MEMIT-family methods are built and validated for cloze-style (subject,
  relation, object) recall. TOFU wants free-form generated sentences scored by ROUGE and
  truth-ratio. Optimizing values for sentence-length answers is the untested part, and mu may
  suffer even if recall lands. (Scale itself is *not* the risk: TOFU's 4,000 pairs are inside
  MEMIT's demonstrated ~10k-edit range.)
- **Key collisions.** Our own name-token finding
  ([07-07](merge_mechanism/2026-07-07_per-author-similarity-k200.md): name-sharing authors' deltas
  measurably more similar, p ≈ 5e-4) is a direct warning that name-derived keys will partially
  collide for similar names — the failure mode to instrument first (the k=200 pairwise key
  cosine matrix, before any editing).
- **It's a different storage substrate**, so comparisons to the adapter tracks need the shared
  eval path (our `eval_tofu` wrappers make that routine).

### 8.5 The staged ladder (so the expensive rung only runs if justified)

**Centered merging (§6.1, one line, ~1 GPU-day of evals) → negative-anchored isolation (§6.3,
one training pass, gated on §6.2) → MEMIT (new harness, the real bet).** Each rung is
independently a mechanism result for Path A; the ladder only escalates if the cheaper rung
fails to close the gap to routing. If any rung reaches routed-level per-author recall in a
single ungated model, the project gains a *method* headline on top of diagnosis + audit.

---

## 9. What NOT to spend GPUs on

- **New routing variants** (PHATGOOSE-style per-expert self-trained gates, entropy-based routing
  à la arXiv 2603.01792, retriever fine-tunes): our own evidence says the payoff isn't there —
  the learned router ≈ uniform on this pool with a flat +0.005 gap
  ([ramole 07-06](ramole/2026-07-06_k-sweep.md)), the retriever fine-tune actively *hurt*
  forgetting (fq 0.48→0.18, [06-29](ramole/2026-06-29_retriever-ft.md)), and anything in this
  direction converges on the SEA/APA territory the gap analysis already scanned. Hard identity
  routing won. **The carve-out:** router work aimed at *sealing the post-deletion leak* (§7.1) is
  a different animal — it's motivated by safety, not utility, and it belongs to Path B.
- **More merge-operator archaeology:** the 07-07 rescue sweep already covers
  knots/tsv/della/jd/regmean/fisher/lorahub + a prediction-level ensemble
  ([design](routing_scaffold/2026-07-07_scafmerge-rescue-design.md)). Harvest it (§10), close the
  table, move on. Post-hoc operators don't add selection, and §4/§5 say selection is the whole
  game.
- **The SemEval over/under-learned combination** (aclanthology 2025.semeval-1.79) — it's an
  approximate-unlearning ensemble; it abandons the exactness certificate that defines this
  project. Cite it, don't chase it.

---

## 10. Recommendation and schedule

**Converge on one paper skeleton: Path B as the headline, Path A as the scientific backbone,
Path C as the staged bet.** Concretely: the paper opens on the audit results (owner ≠ fact; the
router decides the leak; MIA separates exact from approximate), uses the mechanism thread to
explain *why* the failures are structural rather than incidental, frames both with the §5
selection table, and reports whichever Path-C rungs have landed as either "the fix" (if one
works) or "the boundary" (if none does). Every piece of that story is already measured except
the interventions.

Suggested order (all within the ≤4 concurrent GPU cap; items 1–3 are near-free):

| # | Action | Cost | Feeds |
|---|---|---|---|
| 1 | **Harvest jobs 441021–441027** (merge-family rescue sweep, `ensemble_probs`, SIFT-on-scaffold H7). Queue is empty; `sift_masks_scaf` artifacts sit on disk from 07-08 with **no results log entry** — a protocol loose end. Write the overdue results entry. | CPU | closes §9's table; H7 informs A & C |
| 2 | **Key-firing measurement** (§6.2) | CPU / 1 GPU, no training | A; gates 4 |
| 3 | **Centered-merge N-ladder** (§6.1) — one-line change, existing harness; coordinate with the e25 ladder already building on disk (07-13) so both regimes run on the same strong experts | ~1 GPU-day of evals | A + C rung 1 |
| 4 | **Negative-anchored isolation** (§6.3), only if 2 shows lazy keys — then rerun the §9-D drop-an-expert audit on the anchored experts (the §7.1 router-seal test, same battery) | few GPU-h | A + C rung 2 + B (§7.1) |
| 5 | **Path-B closers:** delete-propagation arm (H6), detector upgrade (H5), H4 KS table; identity-grounded-router robustness cell (§7.1 direction 2) if time permits | ~1 GPU-day total | B |
| 6 | **Write-up** of B+A on the §5 frame; MEMIT harness (C rung 3) only after 3–4 land | — | the paper |

Multi-seed passes (43/44) on any number that ends up in the paper remain mandatory per root
`CLAUDE.md` §4 — most headline cells are still single-seed (42).

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **TOFU** | Unlearning benchmark: 200 fictitious authors × 20 QA pairs; forget01/05/10 = forget 2/10/20 authors |
| **mu / `model_utility`** | Harmonic mean of 9 scores (probability, ROUGE-L, truth-ratio × retain/real-authors/world-facts buckets). Base 1B ≈ 0.42, good fine-tune ≈ 0.74 |
| **fq / `forget_quality`** | KS-test p-value: is the model's behavior on forget questions indistinguishable from a retain-only model's? High = good forgetting; gameable and style-sensitive (§1.4) |
| **forget10 / holdout10** | TOFU's 20-author forget split / its matched never-trained split (MIA non-members) |
| **LoRA, rank r, α** | Low-rank adapter Δ=B·A added to a frozen weight; r = rank (r8/r32 = rank 8/32); α = scale hyperparameter |
| **e5 / e25** | Training epochs per adapter; at 20-row author shards e5 ≈ 5 optimizer steps (undertrained), e25 ≈ 25 (saturated) |
| **col(B) / row(A)** | The output (write) / input (read) subspaces of an adapter; "collision is output-side" = col(B) shared, row(A) at chance |
| **Shard / expert / adapter** | One deletable trained unit (in most threads: one LoRA over a subset of authors) |
| **Merging** | Combine adapter deltas into one weight set (sum, mean, dare_ties, TIES, DELLA, TSV, …) |
| **Routing** | Keep adapters separate; pick per query (hard = identity/author-key lookup; soft/embedding = nearest-neighbor or learned gate) |
| **Scaffold** | One LoRA trained on 2k public Alpaca QA, merged into the base, never deleted; restores generic QA competence |
| **Task vector (τ)** | Full-model weight delta θ_ft − θ_base; the non-LoRA analogue of an adapter |
| **SIFT-Masks / ClAMU** | Merged-sum systems recovered by per-task / per-cluster binary parameter masks; deletion = re-derive and subtract a task vector |
| **Oracle retrain / floor** | Model trained only on retain data; the gold standard all exactness/MIA comparisons are made against |
| **MIA / AUC** | Membership-inference attack; AUC 0.5 = attacker at chance (safe), 1.0 = membership fully detectable |
| **Mode-A / Mode-B** | Two ways a fact spans units: A = your clustering *split* it (fixable by clustering right); B = several owners *replicate* it (not fixable by clustering) |
| **RFR / ρ** | Residual-fact-recall: how well a planted fact still answers after its owner is deleted, vs replication factor R |
| **§9-D** | The post-deletion routing audit (gap analysis §9-D): where orphaned queries go after their expert is dropped |
| **Centered merging** | Merge as ΣΔᵢ − (k−1)Δ̄: shared component enters once, idiosyncratic residuals at full strength; deletion = recompute Δ̄ without the deleted author |
| **Negative-anchored isolation** | Train each expert with an added ‖BᵢAᵢh‖→0 penalty on public data — off-author negatives that force key selectivity |
| **MEMIT / AlphaEdit** | Closed-form key–value editing of MLP layers (Δ = (V\*−WK)Kᵀ(C+KKᵀ)⁻¹); AlphaEdit adds a null-space projection protecting retained keys |
| **KS test** | Kolmogorov–Smirnov two-sample test — are two empirical distributions statistically distinguishable? |
| **NLL** | Negative log-likelihood — per-token loss; fluent-token-dominated, so it understates fact-recall damage (ROUGE/recall sees what NLL misses) |
| **rsLoRA** | Rank-stabilized LoRA scaling (α/√r instead of α/r); its larger effective scale caused the retracted facts-vs-skills "specificity" artifact |
| **Smoke / extended caps** | Small fast eval tier vs publication-grade eval sizes |
| **JD / CtS** | Joint-diagonalization ("Compress-then-Serve"): compress many adapters into a shared basis + per-adapter cores |
| **SEUF** | MoE-unlearning paper whose expert-attribution tool we repurpose as the Mode-B detector |
| **OU / open-unlearning** | The reference implementation our metrics are numerically faithful to |

---

*Sources: every linked entry above; thread READMEs under `log/<thread>/README.md`; the
[gap analysis](EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md) for the paper corpus and §6–§9
framing. External papers referenced: SISA (Bourtoule et al.), S³T (ICLR'25), SEA, LegoNet,
RAMoLE, SIFT-Masks/ClAMU (Kuo et al. 2025, arXiv 2504.04626), MEMIT (arXiv 2210.07229),
AlphaEdit (arXiv 2410.02355), PHATGOOSE (arXiv 2402.05859), and the Slack-thread set:
aclanthology 2025.semeval-1.79, arXiv 2506.14126, arXiv 2507.23311, arXiv 2311.07682,
arXiv 2605.07111, arXiv 2603.01792.*
