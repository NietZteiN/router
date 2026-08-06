# The Router Leak, Explained From the Ground Up

**Date:** 2026-07-21 · plain-language companion to
[`ROUTER_LEAK_REPORT_2026-07-18.md`](ROUTER_LEAK_REPORT_2026-07-18.md) — same results, but
every term defined before it is used, every rate given as a raw count first, and one master
table at the end covering all methods × all metrics with the state of each.
Incorporates the [2026-07-21 correction](../../log/router_leak/2026-07-21_serving-catch-correction.md)
(the earlier "96% serving catch" claim was a counter mis-read; correct per-variant numbers below).

---

## 1. The objects (read this once, everything else follows)

- **TOFU** — the benchmark: **200 invented authors**, each with **20 question–answer pairs**
  (4,000 questions total). The authors are fictional, so the base model knows nothing about
  them; anything the system can say about them came from our fine-tuning. That's what makes
  leakage measurable.
- **Shard / expert** — we split the 200 authors into **10 shards of 20 authors** and train
  one small adapter (a LoRA "expert") per shard, in isolation. **Shard 9** (authors
  180–199) is the one we will delete.
- **Scaffold** — one extra adapter trained only on public data (generic question-answering
  style). It contains no author data and is never deleted. "Base+scaffold" = the model with
  *no* author expert active — what a correctly-deleted author should be answered by.
- **Router** — the component that looks at an incoming question and decides which expert to
  activate. Two kinds:
  - **hard router**: looks the question up in a table mapping question → author → shard.
    Perfect, but needs identity labels at serving time.
  - **embedding router** (the realistic one): converts the question to a vector with a
    sentence encoder, and picks the shard whose **centroid** — the average vector of that
    shard's 400 training questions — is nearest.
- **Deletion** — remove shard 9's adapter. The weights side is now exact: that knowledge is
  gone from what can be served.
- **Orphan** — one of the deleted authors' **400 questions** after deletion. They still
  arrive at the system; the router must do *something* with them.
- **Sibling** — the surviving shard whose centroid happens to be nearest to an orphan
  question. Left alone, the router sends the orphan there — that is **the leak**.
- **Tombstone / sentinel** — the proposed fix: keep a small *signature* of what was deleted
  inside the router, and send any question that matches the signature to base+scaffold
  instead of a sibling. We test three signature types ("rungs"), from coarse to
  privacy-cleanest:
  - **shard rung** — keep the deleted shard's own centroid (1 vector for 20 authors);
  - **author rung** — one sentinel per deleted author (mean of their 20 question vectors —
    recomputable from the deletion request itself; 20 vectors);
  - **name rung** — just the embedding of each author's name string (no Q&A data kept).

## 2. The basic numbers

| Population | Count |
|---|---|
| Authors / questions total | 200 / 4,000 |
| Deleted (shard 9) authors / their questions (**orphans**) | 20 / **400** |
| Remaining ("retain") authors / their questions | 180 / **3,600** |
| Out-of-domain eval questions (real authors + world facts; must go to base+scaffold) | 217 unique |
| Mode-B planted facts (for §7) | 200 facts: 50 each at R = 1, 2, 4, 8 owners |

One more baseline you need before anything makes sense: **this embedding router is not very
accurate even before deletion.** Given all 10 centroids, it sends only **242 of the 400**
shard-9 questions (60.5%) to shard 9, and only **2,034 of the 3,600** retain questions
(56.5%) to their own shard. Shard centroids average 20 authors into a blur. Every
deletion effect below has to be judged against this imperfect baseline, not against the
perfect hard router.

## 3. Step one: where do the 400 orphan questions actually go? — every method

Every unlearning method in this repo either has a *router* (a per-query pick, where a deleted
author's question can be misrouted to a surviving unit) or serves a *single merged/ensemble model*
with no per-query pick (nothing to misroute). Below is the orphan-destination breakdown for every
method with a router (§3.1–§3.4), then the explicit list of the methods that have no router at all,
so nothing is silently left out (§3.5). Throughout, `n_eff` = effective number of siblings the
leak spreads over (1 = one magnet unit; high = diffuse).

### 3.1 The shard-routed pool — routing_scaffold **and** sisa_lora (routed)

*Model: Llama-3.2-1B-Instruct (scaffolded base + k=10 shard experts).* This one table covers **two
methods at once**: routing_scaffold *is* these k=10 experts routed by the strategies below, and
**sisa_lora's routed serving uses the identical `router.py` strategies** on the same kind of shard
pool (the only difference is the scaffold) — so its orphan destinations are the same story. Delete
shard 9 (authors 180–199); each of the 400 orphan questions is matched against the 9 survivors:

| Method (how it picks a shard) | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | busiest | n_eff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **activation-norm** (which expert reacts most) | 0 | 0 | 0 | 0 | 0 | 1 | **327** | 63 | 9 | s6 (82%) | 1.4 |
| **LLM last-token** hidden state | 12 | 18 | 15 | 14 | **308** | 0 | 0 | 12 | 21 | s4 (77%) | 1.7 |
| **LLM hidden-state** centroid | 1 | 8 | 2 | 18 | **261** | 0 | 3 | 43 | 64 | s4 (65%) | 2.1 |
| **attention-norm** | 4 | 40 | 0 | **165** | 84 | 77 | 0 | 29 | 1 | s3 (41%) | 3.7 |
| **bge** sentence-embedding | 48 | 17 | 13 | 42 | **132** | 28 | 39 | 24 | 57 | s4 (33%) | 5.7 |
| **mpnet** sentence-embedding | 44 | 36 | 24 | 17 | **98** | 31 | 20 | 53 | 77 | s4 (24%) | 6.7 |
| **logit-divergence** | **106** | 18 | 7 | 6 | 16 | 87 | 28 | 61 | 71 | s0 (27%) | 5.5 |
| **perplexity** (least-surprised expert) | 17 | 58 | 46 | 26 | 25 | 32 | 38 | **100** | 58 | s7 (25%) | 7.0 |
| **TF-IDF** word overlap | 25 | 19 | 23 | 21 | **87** | 18 | 65 | 72 | 70 | s4 (22%) | 6.6 |
| **MiniLM** answer-embedding | 18 | 14 | 25 | 51 | 70 | 39 | 73 | **74** | 36 | s7 (18%) | 7.2 |
| **MiniLM** question-embedding | 19 | 21 | 33 | 45 | 49 | 44 | 61 | **82** | 46 | s7 (20%) | 7.7 |
| **exact name-match** (fallback shard 0) | **400** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | s0 (100%) | 1.0 |
| oracle author-lookup (the safe control) | — | — | — | — | — | — | — | — | — | → base (0 leak) | — |

**All 400 go to somebody** — a similarity router has no concept of "this expert was deleted," and the
new match is strong (sim-ratio **0.971**). The dense routers (LLM-hidden-state, activation-norm)
funnel 65–82% of *every* deleted author's questions onto a single magnet expert (**shard 4 = authors
80–99**, a hub in embedding space). Side-effect: deleting one centroid also re-routes **210 of 3,600**
(5.8%) retain questions. A separate per-author sweep (k=200, Llama-2-7B-chat-hf) shows the same
mechanism at author granularity.

### 3.2 The keyed-expert pool — legonet_lora **and** ramole (embedding route)

*Model: Llama-3.2-1B-Instruct, 32 k-means author-experts, top-3, instructor-xl encoder.* This is the
pool where the leak was **first** measured (§7's ρ→0.833 lineage: sibling sim 0.98). Deleting forget10
(authors 180–199) affects 15 of the 32 experts, leaving **17 survivors**; the 400 orphans re-route
among them:

| Encoder | busiest experts (of 17 survivors) | top-3 share | n_eff |
|---|---|---|---|
| instructor-xl (off-the-shelf) | e5 (92), e11 (92), e30 (73), e20 (39), e31 (36) … | 0.64 | 6.1 |
| instructor-xl (fine-tuned retriever) | **e30 (100), e31 (98)**, e14 (30), e5 (28) … | 0.57 | 6.7 |

A **two-magnet** structure — less extreme than the k=10 single-magnet, but the fine-tuned retriever
still dumps **half of all 400 orphans onto just two experts** (e30 + e31 = 198). Per-author, the
landing is near-deterministic (e.g. author 189 → e30 ×20, author 190 → e5 ×20). *Caveat:* ramole's
**trained RouterLoRA** router gives only aggregate concentration (max-share 0.71, orphan-vs-retain
AUC 0.556 — near chance across 3 seeds), not a per-expert destination histogram, so it can't be
tabulated the same way.

### 3.3 The per-author mask / proxy methods — SIFT, ClAMU, SEA, MemSinks

*Model: Llama-3.2-1B-Instruct (SIFT/ClAMU/MemSinks), Llama-2-7B-chat (SEA).* These serve one model
and pick a per-task **mask/proxy**; their shipped serving uses an exact author lookup (orphan → base,
no leak), so a destination only exists under the *realistic* MiniLM selector we swapped in.

**ClAMU** (16 feature clusters) sends the 400 orphans to surviving *clusters*:

| ClAMU cluster | c1 | c11 | c9 | c10 | c14 | c7 | c5 | c3 | c2 | c6 | c15 | c12 | c4 | c0/c8/c13 | busiest | n_eff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orphans landing | 66 | 56 | 53 | 49 | 49 | 40 | 23 | 20 | 15 | 15 | 9 | 4 | 1 | 0 | c1 (17%) | 8.8 |

**SIFT-Masks** (180 per-author masks) spreads the 400 orphans over **69 authors** (busiest 11%,
n_eff 27.1): author **88** (43), 52 (26), 131 (19), 94 (18), 122 (18)… Authors 88/94/89 sit in shard 4
(authors 80–99) — the *same* embedding hub the shard-routers pool onto, so the magnet is a property of
the MiniLM space itself. **SEA** (per-author LoRA proxies) and **MemSinks** (per-author sink slices)
route **identically to SIFT** — same 180 authors, same MiniLM question-centroids, pool-independent — so
their destinations are the same (author 88 the top magnet, n_eff 27.1). What differs is the
*consequence* at the destination (§8c): SIFT/ClAMU/MemSinks subtract-or-mask (nothing to leak), while
**SEA drops-and-keeps** a real proxy (Group-A leak).

### 3.4 memory_adapters — a content router (different shape)

memory_adapters doesn't route to an author unit at all — it routes each token to the top-32 of a
product-key memory over the base model's hidden states, with 256 entries owned per author. Deletion
is a **block-list**: the deleted author's 256 entries are set to −∞ before the top-k, so the read mass
that used to hit them **redistributes onto surviving entries**. That cross-author read leak is real and
scale-invariant (its logged cross-source read mass ≈ 0.10 — "selection-level, not weight-level").
So the orphan question is meaningful and the answer is "onto other authors' surviving entries," but a
per-entry destination histogram wasn't extracted here (its structure — soft top-32 over ~10⁶ entries —
doesn't reduce to a single-destination count like the others).

### 3.5 Methods with **no** router — orphan-routing is not defined

These serve a single merged or ensemble model, so there is no per-query pick and no orphan to
misroute — deletion changes the *one* served model itself:

- **sisa_lora (merged)** — `dare_ties` etc. merge all shards into one model; unlearn = re-merge without
  the forget shard. One model, no route.
- **composable_tv** — task vectors summed into one model; deletion = subtract. No serve-time selection
  by design.
- **s3t** — serves the ensemble *average* of all shards' token distributions; every shard contributes
  to every query.
- **peft_compose** — VeRA/IA³/DoRA composed into one merged adapter.
- **tofu_baselines** — GA/GD/KL/IDK gradient unlearning on a single model.
- **merge_mechanism** — not a serving method at all; it's the diagnostic thread that *proved* selection
  cannot live inside merged weights (isolated-training keys are "lazy," self-gating can't be trained in)
  — which is precisely *why* every method above needs an external router or mask.

## 4. Step two: can anything *detect* an orphan?

A detector scores each incoming question; above a cutoff we treat it as an orphan and send
it to base+scaffold. Three numbers describe any detector (counts out of 400 orphans and
3,600 retain questions):

- **catch** — orphans correctly sent to base (want high);
- **leak** — orphans that still reach a sibling = 400 − catch (want low);
- **false positives (FPR)** — retain questions wrongly sent to base (want low; this is the
  utility price);
- **AUC** — threshold-free summary of separability: 0.5 = the detector is guessing,
  1.0 = perfect.

**Confidence detectors** (how *well* does the query match its best surviving expert — the
published-style defense, in four variants): AUC **0.57–0.61**. To catch 360/400 orphans
they must falsely reject **~2,900–3,200 of 3,600** retain questions (82–88%). Useless — and
§3 explains why: sim-ratio 0.971 means orphans are *not* low-confidence.

**Tombstone detectors** (does the query match a deleted-identity sentinel *better* than any
survivor), simple argmax rule, out of 400 / 3,600:

| Rung | Caught | Leaked | Retain false positives | AUC |
|---|---|---|---|---|
| shard (1 vector) | 242 (60.5%) | 158 | 210 (5.8%) | 0.839 |
| **author (20 vectors)** | **385 (96.3%)** | **15** | 326 (9.1%) | **0.982** |
| name (20 vectors, no Q&A data) | 281 (70.3%) | 119 | 65 (1.8%) | 0.969 |

With a *calibrated cutoff* on the author-rung score instead of argmax: **360/400 caught
(90%) at 7/3,600 false positives (0.2%)**. That is the headline detector result: identity
signals separate almost perfectly; confidence signals never can.

(Cross-check on a second, older router — 32 clusters, different encoder: author rung 392/400
caught at 11.0% FPR, AUC 0.988; and the coarse per-expert rung is *catastrophic* there —
2,618/3,600 false positives — because those experts co-host retained authors. The sentinel
must match the deletion unit.)

**Scaling with deletion size** (author rung, argmax): delete 2 authors → 39/40 orphans
caught at 0.8% FPR; 10 authors → 195/200 at 3.4%; 20 authors → 392/400 at 11.0%. Cost
grows ≈ 0.5% FPR per deleted author (thresholding cuts it ~50×).

## 5. Step three: run it as a real serving system

Now the whole benchmark is served end-to-end and scored. The metrics:

- **mu (`model_utility`)** — TOFU's headline utility: harmonic mean of 9 scores on retained
  authors + general knowledge. Base model ≈ 0.42; ideal hard-routed system 0.7509.
- **retain_prob** — how strongly the system reproduces retained authors' answers (1 = perfect).
- **forget ROUGE / forget ppl** — text overlap with, and perplexity of, the *deleted*
  authors' true answers (after a real deletion these should sit at base-model levels).
- **fq (`forget_quality`)** — TOFU's forgetting score: a statistical test of whether answer
  *style* on deleted questions matches a never-trained reference. High = "looks forgotten."

| Serving arm | mu | retain_prob | forget ROUGE | fq |
|---|---|---|---|---|
| hard router (reference) | 0.7509 | 0.854 | — | — |
| embed router, no deletion | 0.6872 | 0.573 | 0.639 (knows them) | 0.011 |
| deletion, sibling policy (leak) | 0.6922 | 0.581 | 0.415 | 0.588 |
| deletion, tombstone — **shard rung** | 0.6861 | 0.574 | 0.418 | 0.697 |

What to take from this table, in order:
1. The embed router *itself* costs 0.064 utility vs the hard router (those 44% misroutes) —
   that price is paid before any deletion happens.
2. The served tombstone here is the **shard rung**, and its per-question catch is **242/400
   (60.5%)** — identical to its detector number in §4. *(Correction note: an earlier version
   claimed 96% here by misreading a route counter that lumps orphan catches together with
   retain false-positives across repeated metric passes — ≈653 + ≈102 of 755 counted calls.
   96.3% belongs to the author rung, which was never run in this particular table.)*
3. The tombstone's utility cost vs the leaky policy is 0.006 (its false positives serve
   base+scaffold instead of a — often wrong anyway — expert).
4. **fq cannot see the leak.** The leaky sibling arm scores 0.588 ("looks forgotten")
   while §6 shows what it is actually serving. A style test can't tell *who answered*.

## 6. Step four: what is IN the leaked answers?

For each of the 400 orphan questions we generated three answers — from the deleted authors'
own expert (ceiling), from the sibling the router picks (the leak), and from base+scaffold
(floor) — and measured text overlap (ROUGE-L, 0–1) with the true answers:

| Comparison | Mean overlap |
|---|---|
| own expert vs true answer (ceiling — proves the measurement works) | 0.599 |
| **sibling's answer vs true answer** | **0.277** |
| base+scaffold vs true answer (floor) | 0.249 |
| sibling's answer vs base's answer | 0.278 |
| sibling's answer vs the sibling's *own* authors' true answers | 0.181 |

The sibling's answers contain essentially **none** of the deleted authors' real facts
(0.277 vs a floor of 0.249), none of its own authors' facts either (0.181), and they are
not generic refusals: **382 of 400 (95.5%) are fluent, invented biographies** — overlapping
neither the truth nor the base model's answer ("confabulation"). So in the ordinary world
the leak is an *integrity* failure (the system makes things up about people who asked to be
deleted), not a privacy one. The privacy case is next.

## 7. Step five: replicated facts (where real privacy leaks live)

**Mode-B replication**: the same fact planted into R different owners' training data
(R = 1, 2, 4, 8; 50 facts each). Delete one owner. The other owners legitimately keep their
copies — the question is whether the *served system* reveals them when asked about the
deleted owner. The measure is **ρ (residual-fact-recall)**: 0 = the system answers the fact
no better than a model that never saw it; 1 = as well as before deletion.

| Serving policy | R=2 | R=4 | R=8 |
|---|---|---|---|
| sibling (no defense) | 0.107 | 0.439 | **0.833** |
| tombstone, shard rung — *predicted* by (1−catch)×leak | 0.036 | 0.105 | 0.333 |
| tombstone, shard rung — measured | 0.113 | 0.110 | 0.433 |
| **tombstone, author rung — measured** | **0.031** | **0.000** | **0.047** |

Routing view of the same effect: of 50 planted probes at R=8, the sibling policy lands
**43–44 on a shard that actually hosts a copy** of the fact; at R=2, only 3 of 50.

Two conclusions. The **author-rung tombstone seals the channel** (0.833 → 0.047, ~95%
sealed — because its sentinels catch ~97% of these probes). And the shard rung — which only
catches 48–76% of them — leaks almost exactly what the formula
**residual ≈ (1 − catch) × unsealed leak** predicts (R=4: 0.110 measured vs 0.105
predicted). Residual leak is *budgetable from the catch rate*, which turns the choice of
rung into an explicit privacy-vs-seal dial.

## 8. The costs, in one place

- **False positives**: ≈0.5% of retain traffic per deleted author (argmax); ~50× lower thresholded.
- **Utility**: 0.006 vs the leaky policy (shard rung, as served).
- **Deletion becomes visible**: an attacker comparing routing behavior on deleted vs
  never-seen authors detects the deletion at **AUC 0.839** (400 vs 400 questions) — sealing
  announces that something was sealed.
- **fq is blind** to all of the above, in both directions.
- **Name coverage** (for a name-list alternative): 3,453/4,000 questions (86.3%) name their
  author; on *paraphrased* questions still 360/400 (90.0%) — names survive rephrasing; 18 of
  200 authors have no extractable name; stripping names defeats it entirely.

## 8b. Where the orphans actually land — the "magnet expert" (all router families)

We asked, for every router, *which* surviving expert the deleted authors' questions get sent
to. Two numbers describe it: the **busiest sibling's share** (what fraction of orphans land on
the single most-hit survivor) and **n_eff = 1/HHI**, the effective number of siblings the leak
spreads over (1 = one magnet expert, high = diffuse). Reading the sweep's own stored
destination histograms:

| Router | busiest share | n_eff (effective siblings) | where |
|---|---|---|---|
| activation-norm | 0.82 | **1.4** | one magnet (shard 6) |
| LLM last-token | 0.77 | **1.7** | one magnet (shard 4) |
| LLM hidden-state | 0.65 | **2.1** | one magnet (shard 4) |
| attention-norm | 0.41 | 3.7 | two-cluster |
| MiniLM / mpnet / bge | 0.18–0.33 | 5.7–7.2 | diffuse |
| logit-divergence | 0.27 | 5.5 | diffuse |
| perplexity, TF-IDF | 0.22–0.25 | 6.6–7.0 | diffuse |

The split is bimodal and lines up with §3's leak verdict: the **dense-embedding routers
collapse every deleted author's questions onto ~1–2 "magnet" experts**, while the
semantic/generative routers scatter them across ~6–7. For the LLM-hidden-state routers the
magnet is consistently **shard 4 (authors 80–99)** — a hub in embedding space (the same
pathology LegoNet hit with a 135-author hub). Practical consequence: under a hidden-state
router, one surviving expert silently becomes the systematic answerer for *everyone* who was
deleted. The magnet is a coarse-granularity + single-deletion effect — it dilutes as you delete
more shards, and at per-author (k=200) granularity a single deletion still concentrates (share
0.70) but a 20-author deletion spreads out (0.17, 20 orphans over 180 survivors).

## 8c. Does the leak reach the mask-based methods too? (and why exact subtraction is immune)

The methods in §9's lower block (SIFT-Masks, ClAMU) don't drop-and-keep experts — they merge
everything into one model and pick a per-task *mask* by an exact author-ID lookup. That lookup
is clean by construction. But if you must select *without* labels — the same realistic MiniLM
router — do they leak too? We swapped their oracle lookup for the embedding selector and served
the deleted authors' 400 questions.

**They misroute exactly like the routed methods** (100% of orphans land on a surviving unit;
detectability tracks granularity — SIFT's 180 per-author units make orphans confidence-visible
at AUC 0.97, ClAMU's 16 feature-clusters hide them at 0.585, just like coarse vs fine routers in
§3). **But misrouting cannot leak the deleted content** — the faithful metric, the model's
probability of the *actual deleted answer*, stays at the floor under full misrouting:

| method (deletion = subtract the task vector) | prob of deleted answer, correct deletion | prob under misrouting | verdict |
|---|---|---|---|
| SIFT-Masks | 0.118 | **0.086** | no leak (misrouting lowers it) |
| ClAMU | 0.128 | **0.124** | no leak (flat at floor) |

Compare a *served* author's ~0.9 — the deleted fact is simply not recalled. The reason is
structural: deletion here **subtracts** the author's task vector from the merged weights, so a
misrouted surviving mask operates on a model the deleted author is already gone from; it can
only confabulate. (ClAMU's ROUGE-vs-gold *does* rise +0.142 under misrouting, but that's the
§7.1-style trap — a retain mask makes biography-*form* text that surface-overlaps the gold; 77%
of those answers match neither the gold nor the base model, and the probability metric confirms
no recall.)

**The headline contrast:** the router leak is dangerous specifically for **drop-and-survive**
methods (a surviving *expert/proxy* holds real knowledge, and under replication serves it — §7), and
**defused for free by exact subtraction** (a surviving *mask* has nothing to serve). Where the
tombstone (§2–§5) is *load-bearing* for the routed methods, it is a nice-to-have for the
subtraction methods (it still works — author-sentinel catch 0.96 — but the content isn't there
to protect).

**It is the deletion mechanism, not the granularity, that decides which group you're in** — and
**SEA** is the clean proof. SEA is *per-author* (200 isolated LoRA proxies, one per author), the same
granularity as SIFT, and it routes orphans identically (§3). But SEA deletes by **dropping the
proxy**, not subtracting a task vector, so a misrouted orphan reaches a surviving author's proxy that
*holds real content* — putting per-author SEA in the **Group-A (dangerous)** column alongside the
routed experts, opposite per-author SIFT. Same granularity, opposite verdict, because one drops and
one subtracts.

| Method | granularity | deletion | misrouting reaches | leak? |
|---|---|---|---|---|
| routed scaffold experts | shard | drop adapter | a surviving expert with real content | **yes** (Group A) |
| SEA | per author | drop proxy | a surviving author's real proxy | **yes** (Group A) |
| SIFT-Masks | per author | subtract τ | a mask over an author-free model | no (Group B) |
| ClAMU | per cluster | subtract τ | a mask over an author-free model | no (Group B) |

*Deferred / not yet run: SEA's serving leak measured end-to-end (its routing is settled here; the
consequence is predicted from the drop-and-survive mechanism, on the 7B proxy pool); MemSinks
(deletion by masking, not subtraction — predicted robust by a different route); and Mode-B
replicated facts for the mask/proxy methods.*

## 9. Master table — every method, every metric, and its state

**State legend:** `SERVED` = run through the full benchmark; `AUDIT` = measured at the
routing level (no LLM); `MODE-B` = run in the replicated-fact serving arm; `ANALYSIS` =
derived from saved per-query scores; `REFUTED` = tested and failed; `REF` = reference
point; `NOT RUN` = designed, recorded, not executed.

| Method | Orphans caught /400 | Leaked to siblings /400 | Retain FP /3,600 | Detector AUC | mu | fq | Mode-B ρ @R8 | State |
|---|---|---|---|---|---|---|---|---|
| Hard identity router (labels available) | 400 (by construction) | 0 | 0 | — | 0.7509 | — | 0.000 | REF (SERVED) |
| Embed router, no deletion | — | — | — | — | 0.6872 | 0.011 | — | SERVED (baseline) |
| Deletion, no defense (sibling) | 0 | 400 | 0 | — | 0.6922 | 0.588 | 0.833 | SERVED + MODE-B |
| Confidence threshold (any of 4 variants) | 360 @ cutoff | 40 | ~2,900–3,200 | 0.57–0.61 | — | — | — | REFUTED (ANALYSIS) |
| Tombstone — shard rung | 242 | 158 | 210 | 0.839 | 0.6861 | 0.697 | 0.433 | SERVED + MODE-B |
| Tombstone — author rung, argmax | 385 | 15 | 326 | 0.982 | — | — | **0.047** | AUDIT + MODE-B |
| Tombstone — author rung, thresholded | 360 | 40 | **7** | 0.982 | — (predicted ≈ sibling mu) | — | — | ANALYSIS (recommended; serving cell NOT RUN) |
| Tombstone — name rung | 281 | 119 | 65 | 0.969 | — | — | — | AUDIT |
| Name registry (lexical match) | 358 coverage | 42 name-free | — | — | — | — | — | NOT RUN (coverage measured) |

Reading the table: the only fully-served seal is the weakest rung (shard), and it behaves
exactly as its audit predicted — which is what licenses reading the audited author-rung
numbers as the deployable recommendation, with its thresholded variant the configuration a
real system should use. The one measured-everywhere method with no leak is the hard router,
which is why the first line of practical advice remains: *keep identity labels if you can.*

---
*Provenance: all counts trace to the dated entries in
[`log/router_leak/`](../../log/router_leak/README.md) (pre-registration, Phase 1, Phase 2,
2026-07-21 correction, 2026-07-22 group-A/B depth); jobs 445344–445348, 445357–445360,
445668–445675, 446563–446568, 446665, 447157–447158; seed 42.*

---

# Technical appendix

Full architecture, exact configuration, and the complete orphan-destination statistics behind
the plain-language sections above. Everything here is reproducible from the configs and log
entries named.

## A. The served system, in detail

Three frozen layers compose the model that answers a query:

| Layer | What it is | Recipe (exact) | Deletable? |
|---|---|---|---|
| **Base** | frozen Llama-3.2-1B-Instruct | — | no |
| **Scaffold** | one LoRA on public Alpaca QA, permanently merged into the base (→ the "scaffolded base") | rank 16, α 32, 3 epochs, non-rslora, n = 2,000 pairs from `tatsu-lab/alpaca`; baked with `make_scaffolded_base.py` | never (no author data) |
| **Experts** | k = 10 LoRA adapters, one per 20-author shard, each trained *on top of the scaffolded base* | rank 32, α 64, 5 epochs, seed 42, 400 samples/shard; shard *i* = authors `[20i, 20i+20)`; shard 9 = authors 180–199 = TOFU forget10 | yes — drop the adapter |

Reference operating points on this pool: base model utility ≈ 0.42; a single fine-tune on all
200 authors ≈ 0.637; the routed system with hard identity routing **0.7509** with byte-identical
deletion. The embedding router (below) costs 0.064 of that before any deletion, because ~40% of
queries land on a wrong-but-similar shard.

## B. The router families, in detail

A router maps a query to a shard. The nine strategies swept, by how they score:

| Family | Strategy | Scores a shard by | Encoder / source |
|---|---|---|---|
| **Lexical** | `key_exact` | does the query contain a shard author's name? (else fallback shard 0) | `router._extract_author_names` |
| | `key_tfidf` | TF-IDF cosine of the query vs the shard's training questions | scikit TF-IDF |
| **Dense embedding** | `centroid_sbert` | cosine to the shard centroid (mean of member **answer** embeddings) | all-MiniLM-L6-v2 |
| | `centroid_sbert_q` | same, over member **question** embeddings | all-MiniLM-L6-v2 |
| | `centroid_lm` | cosine to the shard centroid in the **base LLM's own hidden state** (mean-pool) | Llama-3.2-1B |
| | `centroid_lm_last` | same, last-token hidden state | Llama-3.2-1B |
| **Behavioral** | `ppl` | negative perplexity of the query under each shard's expert (least surprised wins) | the experts themselves |
| | `activation_norm` | ‖LoRA-B output‖ — which expert reacts most | the experts themselves |
| | `attn_norm` | same, attention modules only | the experts themselves |
| | `logit_div` | Frobenius distance of each expert's logits from the candidate-set mean | the experts themselves |
| **Trained** | RouterLoRA | a learned per-layer cross-attention gate (RAMoLE) | trained, 3 seeds |
| **Encoder swaps** | `centroid_mpnet` / `centroid_bge` | `centroid_sbert` with a different sentence encoder | all-mpnet-base-v2 / bge-small-en-v1.5 |

Deletion drops shard 9's adapter and (for the embedding families) removes its centroid from the
routing pool. The **oracle** control is an exact question→author dictionary (`build_q2author`):
orphans route to base by construction, no leak.

## C. Complete orphan-destination statistics (where the 400 orphans go, drop = shard 9)

The full histograms behind §8b — counts of the 400 deleted-author questions landing on each
surviving shard (0–8), plus the concentration reduction. `n_eff` = 1/HHI = effective number of
siblings the leak spreads over. **Model: Llama-3.2-1B-Instruct** (the scaffolded base + k=10
experts); the `centroid_lm`/`centroid_lm_last` routers use that same base's hidden states, and the
`ppl`/`activation_norm`/`attn_norm`/`logit_div` routers score with the experts themselves. (The
k=200 per-author sweep quoted at the end of this section is on **Llama-2-7B-chat-hf** — a different
base, read within-pool.)

| Router | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | busiest | max_share | n_eff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| activation_norm | 0 | 0 | 0 | 0 | 0 | 1 | **327** | 63 | 9 | s6 | 0.82 | **1.4** |
| centroid_lm_last | 12 | 18 | 15 | 14 | **308** | 0 | 0 | 12 | 21 | s4 | 0.77 | **1.7** |
| centroid_lm | 1 | 8 | 2 | 18 | **261** | 0 | 3 | 43 | 64 | s4 | 0.65 | **2.1** |
| centroid_bge | 48 | 17 | 13 | 42 | **132** | 28 | 39 | 24 | 57 | s4 | 0.33 | 5.7 |
| attn_norm | 4 | 40 | 0 | **165** | 84 | 77 | 0 | 29 | 1 | s3 | 0.41 | 3.7 |
| centroid_mpnet | 44 | 36 | 24 | 17 | **98** | 31 | 20 | 53 | 77 | s4 | 0.24 | 6.7 |
| logit_div | **106** | 18 | 7 | 6 | 16 | 87 | 28 | 61 | 71 | s0 | 0.27 | 5.5 |
| ppl | 17 | 58 | 46 | 26 | 25 | 32 | 38 | **100** | 58 | s7 | 0.25 | 7.0 |
| key_tfidf | 25 | 19 | 23 | 21 | **87** | 18 | 65 | 72 | 70 | s4 | 0.22 | 6.6 |
| centroid_sbert | 18 | 14 | 25 | 51 | 70 | 39 | 73 | **74** | 36 | s7 | 0.18 | 7.2 |
| centroid_sbert_q / minilm | 19 | 21 | 33 | 45 | 49 | 44 | 61 | **82** | 46 | s7 | 0.20 | 7.7 |
| key_exact | **400** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | s0 (fallback) | 1.00 | 1.0 |

The magnet shards, decoded: **s4 = authors 80–99** (the attractor for every LLM-hidden-state and
sentence-encoder router — a hub in embedding space), **s6 = authors 120–139** (activation_norm),
**s7 = authors 140–159** (MiniLM question-centroids). Note the split cleanly separates the two
leak regimes of §3: the routers that pool onto 1–2 magnets (n_eff ≤ 2.1) are exactly the ones
whose orphans are confidence-*undetectable*, and the diffuse routers (n_eff ≥ 5.5) are the
detectable/self-detecting ones.

**Across deletion count** (share on the busiest survivor as more shards are dropped,
d9 → d9+d8 → d9+d8+d7+d6): activation_norm 0.82 → 0.83 → 0.52; centroid_lm 0.65 → 0.70 → 0.69;
centroid_sbert 0.18 → 0.19 → 0.18. The magnet holds under single-to-few deletions and dilutes
only at large mass deletion (fewer survivors to absorb the redistribution).

**Across granularity** (per-author, k = 200, base = Llama-2-7B, read within-pool): a single-author
deletion still concentrates (centroid_lm share 0.70, n_eff 1.9 — one other author becomes the
magnet) but a 20-author deletion spreads (share 0.17, n_eff far higher — 20 orphans over 180
survivors).

## D. Group B — the realistic-selector swap, in detail

The mask/merge methods (§8c) serve one model and pick a per-task **mask** by an exact author-ID
lookup. Configuration:

| Method | Substrate | Selection unit | Deletion mechanism | Config |
|---|---|---|---|---|
| **SIFT-Masks** | full-FT task vectors summed into one model + a per-author binary mask (sign-constrained) | **per author** (200 masks `m_a`) | **subtract** the author's re-derived task vector τ_u from τ̄; serve forgotten → maskless-merged | full-FT 20 steps, lr 1e-4, answer-loss, sign_seed 42, `sift/tau_bar_forget10.pt` |
| **ClAMU** | same summed task vectors + a per-**cluster** optimized mask | **per feature cluster** (K=16, MiniLM answer-embedding k-means; retain re-clustered on deletion → sizes 23/19/18/15/15/14/14/13/11/10/5/5/5/5/4/4) | subtract τ_u + re-cluster retain; serve forgotten → base θ0 | full-FT 20 steps + mask STE 50 steps lr 0.05, K=16 feature clusters |
| MemSinks *(deferred)* | masked LoRA-delta with per-author "sink" neuron slices | per author sink slice | **mask** the author's slice off (not subtraction) | `~/memsinks_tofu/…strict2_e25.json` |

**The swap:** `groupb_realistic_router.attach_realistic_router` replaces the model's `_route` with
a MiniLM centroid selector over the **surviving** units (SIFT: 180 per-author centroids; ClAMU:
16 retain-cluster centroids), keeping the oracle only for the OOD gate (real/world queries still
serve base) — exactly the router-leak serving convention, so the OOD path is not a confound.

**Routing-level stats** (the 400 deleted-author questions, surviving units):

| Method | surviving units | max_share | n_eff | confidence AUC | author-sentinel tombstone catch / FPR |
|---|---|---|---|---|---|
| SIFT | 180 (per author) | 0.11 | 27.1 | **0.972** (separable) | 0.965 / 0.003 |
| ClAMU | 16 (feature clusters) | 0.17 | 8.8 | **0.585** (inseparable) | 0.960 / 0.000 |

100% of orphans misroute (their own unit is gone). Detectability tracks granularity exactly as in
§3 — SIFT's fine per-author units make an orphan's lack-of-a-good-match visible (AUC 0.97, matching
the k=200 cross-check 0.982); ClAMU's coarse clusters hide it (0.585, like the dense k=10 routers).

**Where the orphans go for the mask methods** (the §3/§C "step one" breakdown, for the mask-based
methods; both on **Llama-3.2-1B-Instruct**, the same base as the routed system, under the same
MiniLM selector). ClAMU routes each of the 400 deleted-author questions to a surviving *retain
cluster* (K=16 units):

| ClAMU retain cluster | c1 | c11 | c9 | c10 | c14 | c7 | c5 | c3 | c2 | c6 | c15 | c12 | c4 | others |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orphans landing | 66 | 56 | 53 | 49 | 49 | 40 | 23 | 20 | 15 | 15 | 9 | 4 | 1 | 0 |

— busiest cluster share 0.17, n_eff 8.8: orphans spread across ~13 of the 16 semantic clusters,
concentrating mildly on the larger/more-central ones.

SIFT routes to a surviving *author* (180 units). The mass spreads over **69 distinct authors**,
busiest share 0.11, n_eff 27.1 — but the top destinations are telling: author **88** (43),
**52** (26), **131** (19), **94** (18), **122** (18), **89/40/150** (15), **19/103** (14). Authors
88, 94, 89, 87 all sit in **shard 4 (authors 80–99)** and 131/122/123 in shard 6 (120–139) — the
*same* embedding-space hubs the shard-granular routers pool onto (§C). So the magnet is a property
of the MiniLM embedding space (specific authors like 88 are attractors), reproducible whether you
route at shard or per-author granularity — one deleted-author query in nine lands on author 88's
mask regardless.

**Serving-level stats** (3 arms × 400 questions; answer-prob = probability of the *deleted* gold
answer, the faithful leak metric — a served author scores ~0.9):

| Method | arm | answer-prob of deleted gold | ROUGE-L vs deleted gold | ROUGE-L vs base gen | confab. rate |
|---|---|---|---|---|---|
| SIFT | oracle (correct deletion) | 0.118 | 0.392 | — | — |
| | base θ0 | 0.128 | 0.226 | — | — |
| | **realistic (misrouted)** | **0.086** | 0.338 | 0.254 | 0.79 |
| ClAMU | oracle (correct deletion) | 0.128 | 0.226 | — | — |
| | base θ0 | 0.128 | 0.226 | — | — |
| | **realistic (misrouted)** | **0.124** | 0.368 | 0.265 | 0.77 |

Reading it: under 100% misrouting the deleted fact's probability stays at floor for both
(0.086, 0.124 — nowhere near a served author's ~0.9), so **no disclosure**. ClAMU's ROUGE-vs-gold
rises +0.142 (0.226 → 0.368), which trips the pre-registered ROUGE bar — but that answer overlaps
the base generation at only 0.265 and is a novel fabrication 77% of the time, and the probability
metric confirms the fact is not recalled: it is biography-*style* text (a retain mask makes
names/dates in the right form), not the deleted author's content. The reason is structural — the
mask operates on a τ̄ the author's task vector has already been subtracted from, so there is
nothing to disclose. This is why the router leak, dangerous for the drop-and-survive methods where
a surviving *expert* still holds real knowledge, is defused for free by exact subtraction.
