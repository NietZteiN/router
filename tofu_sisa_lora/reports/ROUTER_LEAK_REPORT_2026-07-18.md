# The Post-Deletion Router Leak — and How to Seal It

**Date:** 2026-07-18 · **Thread:** [`log/router_leak/`](../../log/router_leak/README.md) ·
**Status:** campaign closed (Phases 0–2 complete; optional closers recorded, not run)
· **Spend:** ≈ 12 GPU-h · **Seed:** 42 throughout

> **How to read this.** Written to be understandable from scratch — no familiarity with the
> repo's other threads is assumed. §1 explains the system and the problem, §2 the idea under
> test, §3 the experiments, §4 the results, §5 what a deployer should actually do, §6
> limitations and what we deliberately did not run. Every number is copied from a dated,
> append-only log entry (linked). Hypotheses and their pass/fail bars were pre-registered
> *before* any GPU run ([pre-registration](../../log/router_leak/2026-07-18_router-leak-preregistration.md)).

---

## TL;DR

- **The problem.** In a modular unlearning system (one adapter per data shard, deletion =
  drop the adapter), a *realistic* router — one that picks the adapter by embedding
  similarity rather than exact identity labels — keeps serving the deleted authors'
  queries from the most similar **surviving** adapter. Confidence thresholds provably
  cannot detect this. The result: deletion looks clean on every standard metric while the
  system either fabricates answers about deleted people or, if a fact was replicated
  across owners, actually reveals it.
- **The fix that works.** A **tombstone**: keep a lightweight *identity signature* of what
  was deleted (a sentinel embedding per deleted author) inside the router, and send any
  query that matches it to the plain base model. This separates deleted-author queries
  from legitimate traffic at **AUC 0.98–0.99 (0.2% false positives at 90% catch)** where
  every confidence-based detector sits near chance (0.56–0.63) — and, deployed, it
  collapses the replicated-fact leak from **ρ = 0.833 to 0.047** at a retain-utility cost
  of ~0.006.
- **The leak's content.** Without replicated facts, what leaks is not private data:
  **95.5%** of sibling-served answers about deleted authors are novel confabulations —
  an *integrity* failure, not a privacy one. Privacy harm exists specifically when a fact
  has multiple owners (Mode-B), and that is exactly the channel the tombstone seals.
- **The fine print.** Sealing has a measurable price: the seal itself makes deletion
  *observable* to an attacker (AUC 0.839 — the "Streisand" cost), and the seal's
  false-positive rate grows ~0.5% per deleted author. TOFU's own `forget_quality` metric
  is **blind** to the entire channel — the leaky configuration "passes" it.

---

## 1. Background, from scratch

### 1.1 The system under study

This project makes deleting an author's data from a fine-tuned LLM a cheap, *exact*
operation by imposing structure up front: a frozen Llama-3.2-1B base, a permanently-merged
"scaffold" LoRA trained on 2k public Alpaca QA pairs (generic answer competence, contains
no author data, never needs deletion), and **k = 10 expert LoRA adapters**, each trained in
isolation on one shard of 20 TOFU authors (TOFU = a benchmark of 200 fictitious authors ×
20 question–answer pairs; the base model knows nothing about them, so any knowledge is
traceable to the fine-tune). At serving time a **router** looks at the incoming question
and picks the right expert. Deleting a shard = removing its adapter: with ideal routing,
every remaining query is served by byte-identical weights and the deleted authors answer
from the plain base — provably equal to never having trained on them (verified elsewhere
in the repo; utility 0.7509 vs 0.6372 for an equivalent single fine-tuned model).

### 1.2 The router is the weak point

That guarantee assumes the router *knows* which author a query is about. The exact
("hard") router does — it looks the question up in a table — but it needs identity labels
at serving time. A *realistic* router embeds the query with a sentence encoder (here
all-MiniLM-L6-v2) and picks the expert whose **centroid** (the average embedding of its
training questions) is nearest. Prior audits in this repo established three facts about
what happens to such a router after deletion
([ramole §9-D](../../log/ramole/2026-07-06_routing-audit-results.md),
[fix arms](../../log/ramole/2026-07-07_routing-fix-arms.md),
[entangled_facts](../../log/entangled_facts/2026-07-07_embed-route-surface.md)):

1. **Orphans fall to siblings.** With the deleted expert's centroid gone, its authors'
   queries match the most similar *surviving* expert almost as well as the deleted one did
   (top-1 similarity ratio 0.980) — the router happily serves them from it.
2. **Confidence can't catch it.** The natural fix — "if no expert matches confidently,
   answer from base" — is uncalibratable: orphan and legitimate queries have almost
   identical confidence distributions (means 0.858 vs 0.877), so catching 90% of orphans
   falsely rejects 58% of legitimate traffic.
3. **The router decides whether replicated facts leak.** If a fact was planted into R
   owners' data and one owner is deleted, the fact legitimately survives in the other
   owners' experts ("Mode-B replication"). Through the *same weights*, a hard router shows
   nothing while the embedding router surfaces the surviving copies at residual-fact-recall
   ρ up to **0.833** at R = 8 (ρ = 1 means the fact answers as well as before deletion;
   0 means it answers no better than a model that never saw it).

This report is the deep-dive campaign on that problem: can it be sealed, what exactly
leaks, and what does sealing cost?

### 1.3 The idea under test: tombstones

A **tombstone** keeps a signature of the deleted identity *inside the router*. The pool of
routing targets becomes {surviving expert centroids} ∪ {sentinels of the deleted
identities}; a query whose best match is a sentinel is served by base+scaffold (the
correct post-deletion answer source) instead of a sibling. The intuition: the orphan
query's problem was never low confidence — it matches the *deleted* identity best of all;
the router just had nothing left to match it against. Restore the identity as a
tombstone and the match itself becomes the detector.

Because a sentinel is derived from the deleted author's data, there is a privacy question
about the sentinel itself. We therefore test a **provenance ladder** of three sentinel
types, from most identity-informative to most privacy-clean:
- **per-expert** — the deleted expert's own routing key (coarsest; one key covers 20 authors);
- **per-author** — one sentinel centroid per deleted author (mean of their 20 question
  embeddings — recomputable from the deletion request itself);
- **name** — an embedding of just the author's *name string* (no QA data retained).

---

## 2. What we measured (and the discipline)

Everything below was pre-registered with explicit confirm/refute bars before any GPU run,
adjudicated in dated append-only log entries, run at seed 42 under the cluster's 4-GPU
cap, and CPU-gated (unit tests) before every submission. One metric rule was pre-declared:
**`forget_quality` (fq) is never a leak bar** — it is a distributional style test and (as
§4.6 confirms) cannot see this channel.

| Experiment | Question | Where |
|---|---|---|
| R1 routing audits | Can any signal separate orphaned from legitimate queries? (identity rungs vs the whole confidence family, on BOTH the n=32 legacy index and the k=10 centroid router the serving arms actually use) | [phase 1](../../log/router_leak/2026-07-18_phase1-results.md) |
| R2 serving triple | Full TOFU evaluation of embedding-routed serving: no deletion / sibling leak / tombstone seal | [phase 2](../../log/router_leak/2026-07-18_phase2-results.md) |
| R2b Mode-B worlds | Does the tombstone seal the replicated-fact residual? Both rungs, with per-R predictions fixed in advance | phase 2 |
| R3 content audit | What do sibling-served answers actually contain? (disclosure vs generic vs confabulation vs cross-author disclosure, 400 questions × 3 serving arms) | phase 2 |
| R5 deletion dial | How do leak and seal-cost scale with the number of deleted authors? (forget01/05/10) | phase 1 |
| R6 registry/disclosure | Lexical name-coverage limits; can an attacker detect that a deletion happened? | phase 1 |

Two design corrections mattered enough to state: prior leak numbers were measured on a
router (instructor-xl over 32 clusters) that nothing actually serves TOFU through, so R1
added the audit of the *serving* router (MiniLM shard centroids) — and the Mode-B seal
prediction was made **mediated**: the seal can only stop what its sentinel catches, so we
first measured each rung's catch rate on fact-shaped probe questions (**c_probe**) and
pre-registered ρ_sealed ≈ (1 − c_probe) × ρ_sibling as the falsifiable model.

---

## 3. Results

### 3.1 Identity separates; confidence never will (H1 ✓, H2 ✓)

The **tombstone-relative margin** — how much better a query matches a deleted-identity
sentinel than any surviving expert — separates orphaned queries from legitimate retain
traffic almost perfectly, on both routers, evaluated on held-out authors:

| Detector (score) | k=10 serving router AUC / FPR@90%catch | n=32 index AUC / FPR |
|---|---|---|
| **tombstone margin, author rung** | **0.982 / 0.002** | **0.988 / 0.002** |
| tombstone margin, name rung | 0.969 / 0.078 | 0.858 / 0.438 |
| tombstone margin, expert/shard rung | 0.839 / 0.472 | 0.642 / 0.622 |
| global confidence (the refuted 07-07 fix) | 0.597 / 0.854 | 0.610 / 0.706 |
| per-expert calibrated confidence | 0.606 / 0.841 | 0.620 / 0.702 |
| top1−top2 margin | 0.572 / 0.817 | 0.626 / 0.674 |
| kNN density | 0.586 / 0.879 | 0.557 / 0.819 |

The confidence family's failure is not a calibration problem — four different
calibrations all sit near chance — it is an information problem: nothing about a query's
match *quality* says the matched identity was deleted. The tombstone injects exactly that
missing bit. Granularity must match the deletion unit: on the n=32 pool, whose experts
co-host retain authors, the per-expert rung false-flags **73%** of retain traffic
(pre-predicted); per-author sentinels are the usable rung everywhere.

### 3.2 The seal in real serving (H3 partial)

Full TOFU evaluation of embedding-routed serving (extended caps):

| Arm | model_utility | retain_prob | forget ROUGE | fq |
|---|---|---|---|---|
| embed-full (no deletion) | 0.6872 | 0.573 | 0.639 (knows) | 0.011 |
| deletion, **sibling** policy (the leak) | 0.6922 | 0.581 | 0.415 | 0.588 |
| deletion, **tombstone** policy (the seal) | 0.6861 | 0.574 | 0.418 | 0.697 |
| reference: hard-key routing | 0.7509 | 0.854 | — | — |

**Correction (2026-07-21,
[log entry](../../log/router_leak/2026-07-21_serving-catch-correction.md)):** this arm
deploys the SHARD-rung sentinel, whose per-question catch is **60.5% (242/400 orphan
questions)** — the `deleted=755` route counter mixes ≈653 orphan-call catches with ≈102
retain false-tombstones across metric passes and does not support the "96%" originally
claimed here; 96.3% (385/400) is the AUTHOR rung's catch, measured in the routing audit
and in the Mode-B serving arm (ρ 0.047) but not in this triple. Its retain cost vs
the sibling policy is **−0.0061 utility** — narrowly over our ≤0.005 bar — caused by the
~9% of retain queries the simple argmax rule false-flags; Phase 1's ROC shows a
*thresholded* margin buys the same catch at 0.2% FPR, so the gap is closable (not run;
§6). Note the honest baseline: the embedding router *itself* costs −0.064 utility vs
hard-key routing (37–40% of queries go to a wrong-but-similar shard) — that is a router-
accuracy price, not a deletion price, and all deletion deltas are read against embed-full.

### 3.3 The Mode-B seal, and a leak law you can budget with (H4 ✓✓)

Residual-fact-recall ρ for facts planted into R owners, one owner deleted, served through
the embedding router (verbatim facts, original-question probes — the reference surface):

| Serving policy | R=2 | R=4 | R=8 |
|---|---|---|---|
| sibling (the leak, prior result) | 0.107 | 0.439 | **0.833** |
| tombstone, shard rung — *predicted* (1−c_probe)·ρ_sib | 0.036 | 0.105 | 0.333 |
| tombstone, shard rung — measured | 0.113 | **0.110** | 0.433 |
| **tombstone, author rung — measured** | **0.031** | **0.000** | **0.047** |

Two results in one table. First, the **author-rung tombstone effectively seals the
replicated-fact channel**: the ρ = 0.833 residual collapses to 0.047 — ~95% sealed —
training-free, using only sentinels recomputable from the deletion request. (The facts
still *exist* in the surviving owners' weights, as they legitimately should; this is a
serving-surface seal, which is the only place an owner-level deletion can act.) Second,
the weaker shard rung validated the **mediation law**: its measured residuals track
(1 − catch) × sibling-leak — R4 dead-on (0.110 vs 0.105 predicted), all points inside the
pre-registered falsifier. Residual leak through a seal is *predictable from the seal's
catch rate*, which turns the provenance ladder into a quantified dial: choose how much
identity information to retain, read off how much leak remains.

### 3.4 What the leak actually says (H5 ✓)

For all 400 deleted-author questions, we generated answers under three arms sharing one
model load — the author's own expert (ceiling), the sibling the router falls to, and
base+scaffold (floor) — and scored each generation three ways:

| Measure (n=400) | mean |
|---|---|
| own expert vs gold answer (ceiling) | 0.599 |
| **sibling vs gold answer** | **0.277** |
| base+scaffold vs gold (floor) | 0.249 |
| sibling vs base's generation | 0.278 |
| sibling vs the sibling's own authors' gold | 0.181 |
| **confabulation rate** (novel fabrication: low vs gold AND low vs base) | **0.955** |

Sibling answers contain essentially none of the deleted authors' real content (0.277 ≈
floor + 0.028, against a live 0.35 ceiling–floor gap), don't disclose the sibling's own
authors' facts either (0.181), and are not generic non-answers — **95.5% are fluent,
novel, fabricated biographies**. So in the ordinary (non-replicated) world, the router
leak is an *integrity* failure — the system confidently makes things up about people who
asked to be deleted — rather than a privacy failure. The privacy failure exists precisely
under Mode-B replication (§3.3), which the tombstone seals. The tombstone fixes both at
once, since tombstoned queries serve the base's honest ignorance.

### 3.5 What sealing costs (H7 ✓, H8a ✓, H8c ✓)

- **False positives scale with deletion volume:** per-author sentinel FPR ≈ 0.4–0.6% per
  deleted author (2 authors: 0.8% total; 10: 3.4%; 20: 11.0% by the raw argmax rule —
  thresholding reduces this ~50× at 90% catch). Small deletions are near-free.
- **The seal is observable.** An attacker probing routing behavior separates deleted
  authors from never-trained ones at **AUC 0.839** — deploying a tombstone *announces*
  that a deletion occurred (the pre-registered "Streisand" cost line; the same is true of
  any registry). Any deployment claim about sealing should report this number next to the
  catch rate.
- **A lexical registry is stronger than expected:** author-name coverage of TOFU questions
  is 0.863 overall and — the surprise — **0.900 on paraphrased questions**: names survive
  rephrasing. A registry's irreducible attack surface is the ~10% name-free questions
  (and deliberate name-stripping, coverage 0). 18 of 200 authors yield no extractable
  name. (The hybrid registry-first router this motivates was designed but not run; §6.)

### 3.6 The honest negatives

- **`forget_quality` is blind to the whole channel.** The leaky sibling configuration
  scores fq 0.588 and the sealed tombstone 0.697 — both "pass," in the wrong order at
  smoke caps and the right one at extended caps, with neither separable from clean
  deletion. A KS test of answer-style distributions cannot see *who answered*. Any
  routed-unlearning evaluation needs route-level metrics (catch/leak rates) and content
  audits, not fq.
- **The retain-cost bar was missed by the argmax rule** (−0.0061 vs ≤0.005) — closable
  per §3.2, but as-run the seal is not free.
- **Anchored ("self-gating") experts were closed without running** per their
  pre-registered contingency: the training-side selectivity idea was already refuted
  (2026-07-16 — output-norm penalties shrink adapters uniformly, zero selectivity), and
  this campaign's content audit removed the surviving rationale (there is no disclosure
  content to attenuate; confabulation is removed by the tombstone). Selection must live
  outside the adapter — which is precisely what the tombstone provides.

---

## 4. What a deployer should do (the recipe this campaign supports)

1. **Prefer hard identity routing when labels exist** — it is byte-clean on deletion
   (established before this campaign) and has no leak to seal.
2. **If routing is embedding-based:** on every deletion, add **per-author sentinel
   centroids** (recomputable from the deleted data at request time; granularity must match
   the deletion unit) and route sentinel-matches to base. Use a **calibrated threshold on
   the tombstone margin**, not argmax (0.2% vs ~9% false positives at the same catch).
3. **Report the price:** deletion-disclosure AUC (here 0.839) and the per-author FPR
   slope, alongside the catch rate.
4. **Do not** rely on confidence/abstain thresholds (near-chance), on `forget_quality`
   (channel-blind), or on training experts to self-gate (refuted).
5. **Budget residual leak** with the mediation law: residual ≈ (1 − sentinel catch) ×
   unsealed leak. If a privacy-cleaner sentinel (name-only) is required, expect its catch
   (~0.70 here) and budget accordingly.

---

## 5. Limitations and what was deliberately not run

- **One benchmark, one scale, one seed.** TOFU is identity-keyed by construction (authors
  have names; questions usually mention them) — tombstones should transfer to any
  entity-keyed deletion unit, but content without a routable identity (e.g. topic-level
  deletion) is untested. All serving cells are deterministic given the pool; the only
  stochastic stage (training) was reused from existing artifacts, so multi-seed applies to
  the upstream pools, not these evals.
- **Not run (recorded as open in the thread):** the τ-thresholded tombstone *serving* arm
  (predicted to close the 0.006 retain gap); the hybrid registry-first router cell (H8b);
  the name-rung deletion-disclosure analysis (CPU, from saved similarity dumps); k=10
  multi-shard mass-deletion dial cells; a composed-MIA rider on the embed-sibling arm.
- **The tombstone seals serving, not weights.** Facts replicated to surviving owners
  remain in their experts — legitimately (they own copies). Fact-level erasure under
  replication requires delete-propagation (the entangled_facts thread's H6), a policy
  question this campaign does not decide.

## 6. Provenance

- **Log entries (append-only):**
  [pre-registration](../../log/router_leak/2026-07-18_router-leak-preregistration.md) ·
  [Phase-1 results](../../log/router_leak/2026-07-18_phase1-results.md) ·
  [Phase-2 results](../../log/router_leak/2026-07-18_phase2-results.md); thread summary
  [README](../../log/router_leak/README.md).
- **SLURM jobs:** Phase 1 445344–445348; Phase 2 smoke 445357–445360; Phase 2 extended
  445668–445675. All 1-GPU, chained under the global 4-GPU cap.
- **Code (sha256-12 in the pre-registration):** `routing_audit_tofu.py` (tombstone rungs,
  `--dump_sims`, `--centroid_mode`), `eval_routed_scaffold.py` (`EmbedRoutedModel`,
  `--embed_route`), `eval_entangled_probe.py` (`--embed_policy tombstone|tombstone_author`,
  `--dump_generations`), `dump_generations_routed.py`, `aggregate_rho.py`,
  `analyze_router_leak.py`, `submit_router_leak.sh`; CPU gate `test_router_leak.py`.
- **Data artifacts:** audit JSONs + `.sims.npz` sidecars under
  `…_legonet_n32_k3/results/` and `…_experts_scaf_k10/results/router_leak/`; serving JSONs
  under `…_experts_scaf_k10/results/{smoke,extended}/`; Mode-B worlds under
  `…_entangled_k10/results/entangled/`; ρ tables `rho_embedsim_{sibling,tomb,tomba}.json`.

## 7. Mini-glossary

| Term | Meaning |
|---|---|
| expert / shard | one LoRA adapter trained on 20 authors; the deletion unit |
| scaffold | public-data LoRA baked into the base; generic QA competence, never deleted |
| orphan | a deleted author's query after their expert is dropped |
| sibling | the surviving expert most similar to the deleted one — where orphans land |
| tombstone / sentinel | an identity signature of deleted data kept in the router; matches route to base |
| provenance ladder | sentinel types by retained-data footprint: expert key → author centroid → name string |
| c_probe | fraction of fact-shaped probe questions a sentinel catches (the mediation term) |
| ρ (residual-fact-recall) | 0 = fact answers like a never-trained model; 1 = like before deletion |
| Mode-B replication | the same fact legitimately held by several owners; deleting one owner cannot erase the others' copies |
| mu / `model_utility` | TOFU's 9-component utility harmonic mean (retain + general knowledge) |
| fq / `forget_quality` | TOFU's KS-test forgetting p-value — shown here to be blind to router leaks |
| confabulation rate | fraction of served answers that are novel fabrications (low overlap with both gold and base) |
| disclosure AUC | how well an attacker separates deleted from never-trained identities by served behavior |
