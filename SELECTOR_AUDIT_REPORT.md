# Deleted from the Router, Not from the Model

**Final campaign report — `selector_audit`, 2026-08-07 → 2026-08-12.**

An audit of *deletion under a selector* as a design pattern: what a system actually removes when
deletion is implemented by a router choosing a different expert, rather than by changing weights.

This document is self-contained. It assumes no prior knowledge of this repo, of the MUSR paper it
follows, or of the TOFU benchmark — §2 builds the whole setting from scratch — and it carries
**every result the campaign produced**, including the refutations and the results that did not make
the paper.

| | |
|---|---|
| Headline pool (all numbers unless stated) | `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` — 200 per-author LoRA experts |
| Deletion under test | TOFU `forget10` = authors 180–199 = 20 sources, 400 orphaned questions |
| Recipe ablation pools | `_k200_r32_e5`, `_k200_r8_e5`, `_k200_r32_e50` (the last built for H21) |
| Campaign record | 26 dated entries · **31 hypotheses filed, 28 adjudicated** · 8 defects (6 numbered + 2 self-corrections) |
| Code authored here | 17 modules/drivers (see `MANIFEST.files`), each with a CPU gate |

**Where things live**

- Dated narrative, decision rules fixed *before* each run: [`log/selector_audit/`](log/selector_audit/)
- Artifact map with regeneration commands and per-cell warnings: [`tofu_sisa_lora/reports/selector_audit/INDEX.md`](tofu_sisa_lora/reports/selector_audit/INDEX.md)
- Manuscript draft written from this report: [`paper/followup/`](paper/followup/)
- Released metric code: [`selector_audit/`](selector_audit/)

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [The setting, from scratch](#2-the-setting-from-scratch)
3. [§4.10 — the forget metric is blind to substitution](#3-4102--the-forget-metric-is-blind-to-substitution)
4. [§4.3 — what the system actually says](#4-43--what-the-system-actually-says)
5. [§4.2 — where orphans go, and what deletion disturbs](#5-42--where-orphans-go-and-what-deletion-disturbs)
6. [§4.5/§4.9 — detectability is a lexical artifact, with one exception](#6-4549--detectability-is-a-lexical-artifact-with-one-exception)
7. [§4.4 — attacking the selector](#7-44--attacking-the-selector)
8. [§4.6 — the defense frontier](#8-46--the-defense-frontier)
9. [§4.21 — training duration moves the leak](#9-421--training-duration-moves-the-leak)
10. [Method constraints](#10-method-constraints-that-govern-how-these-numbers-may-be-read)
11. [Defect record](#11-defect-record)
12. [Complete hypothesis ledger](#12-complete-hypothesis-ledger)
13. [Status: settled, blocked, not claimed](#13-status-settled-blocked-not-claimed)
14. [Reproducing](#14-reproducing)

---

## 1. Executive summary

**A system can satisfy a deletion request, score well on the field's forget metric, and still hand
the user a stranger's biography under the deleted person's name.**

Four findings, each established on the same pool with the same deletion:

1. **The benchmark cannot see substitution (§3).** An "unlearning method" that deletes *nothing*
   and only reroutes the deleted authors' queries to one surviving expert scores **at or above
   genuine deletion in 6 of 7 destination choices**, at identical utility. `forget_quality` moves
   0.53 on a choice that has nothing to do with forgetting.
2. **The harm is real and substantive (§4).** Orphan answers assert a *specific* surviving author's
   facts about the deleted person at rate **0.24–0.30** against a random-destination floor of
   **0.17** — and two thirds to four fifths of it is real facts, not a swapped name. Refusal is a
   rounding error (≤0.013 over 1600 answers).
3. **Every reassuring property is a property of the name (§5, §6).** Orphan detectability, deletion
   locality (RDR), and routing accuracy all look excellent on TOFU's own question phrasing and all
   degrade or collapse once the query stops naming the person being deleted. This holds in every
   selector family — lexical, feature-space and behavioral — with one exception.
4. **The exception is `ppl`, and it does not yield a deployable defense (§8, §9).** The one selector
   that reads each expert's *loss* survives anonymisation and survives training duration. It can be
   made 45–90× cheaper at no loss. It is still unusable: on anonymised queries it refuses **41.8%**
   of legitimate traffic to catch 90% of orphans. It works exactly where it is not needed.

**What the paper must not claim:** that any particular destination is worse than another; that CSAR
is validated (300 hand labels outstanding); that training longer mitigates anything; that no
record-free defense is possible.

---

## 2. The setting, from scratch

### 2.1 What is being audited

A family of "exact unlearning" systems makes deletion cheap by making it modular:

> Freeze a base language model. Split the training data by **source** (a user, a document, an
> author). Train one small adapter — an *expert* — per source. At inference a **selector** picks
> which expert answers a query. To delete a source, drop its expert.

The deletion is exact in the only sense the architecture promises: after the drop, **no surviving
parameter was ever trained on the removed data**. That is a real guarantee and it is why the
pattern is attractive. Variants appear as keyed adapter banks, sharded PEFT unlearning, retrieval-
routed LoRA pools, deletable per-user proxies, and block-listed memory modules
(`papers/RELATED_WORK.md` surveys them).

**The unaudited part is the selector.** Deletion removes an expert; it does not remove the queries
that used to be routed to it. Every one still arrives, and the selector must send it somewhere. We
call such a query an **orphan**. This campaign asks what the system says to an orphan.

### 2.2 Three things "deleted" can mean

| | meaning | who guarantees it |
|---|---|---|
| **D1** | no parameter serving the query was trained on the deleted source | the architecture, by construction |
| **D2** | the system does not answer as though it still held the source | **nobody** — this is what §3–§5 measure |
| **D3** | an observer cannot tell that a deletion occurred | **nobody** — this is what §6 measures |

D1 does not imply D2 (a substituted expert answers fluently, just wrongly) and does not imply D3
(deletion changes the selector's score geometry, measurably, from the surviving units alone).

### 2.3 The concrete system

- **Base:** Llama-2-7B-chat, frozen.
- **Experts:** k=200 LoRA adapters, rank 32, 25 epochs, one per TOFU author.
- **Benchmark:** TOFU — 200 *fictitious* authors × 20 question/answer pairs. Fictitious is the
  point: because the authors do not exist, ground truth about who knows what is complete, which is
  what makes a fact-level attribution metric computable at all.
- **The deletion:** TOFU `forget10` = authors 180–199. 20 of 200 sources removed; 400 questions
  orphaned.
- **Why this pool:** it is the repo's best-utility routed configuration (0.8236 oracle-routed
  utility) — i.e. the pattern is being audited in the configuration where it works best.

### 2.4 The eight selectors, and what each one reads

This matters because the results split by **what a selector consumes**, not by its name.

| family | selector | reads |
|---|---|---|
| lexical | `key_exact` | the author's name as a substring of the query |
| | `key_tfidf` | TF-IDF similarity to each source's training text |
| feature-space | `centroid_sbert` | MiniLM sentence embeddings of the questions |
| | `centroid_lm` | base-model embeddings **with adapters disabled** |
| behavioral | `ppl` | each candidate expert's **loss** on the query |
| | `activation_norm` | each candidate expert's activation **magnitude** |
| | `attn_norm` | attention-output **magnitude** |
| | `logit_div` | logit divergence (excluded at k=200: ~50 GiB of cached activations) |

**A standing consequence:** *no feature-space selector reads any expert's weights.* Verified, not
assumed — their score matrices are **byte-identical** across pools differing in rank and epochs
(`np.array_equal` True on the 4000×200 matrices). So any per-pool table of feature-space numbers is
one column repeated, and every recipe question can only be asked of the behavioral family. A
control over a variable the measurement does not consume is not a control.

### 2.5 Metrics

**From TOFU:**

- `forget_quality` = `ks_2samp(forget_truth_ratios, retain90_oracle_reference).pvalue`. A
  two-sample Kolmogorov–Smirnov test: higher = the post-deletion model's forget-set behaviour is
  more indistinguishable from a model that never trained on the forget set.
- `model_utility` = harmonic mean of 9 components over retained authors, real authors and world
  facts.

**Added here** (released in [`selector_audit/`](selector_audit/)):

- **ORR — orphan response rate.** Fraction of orphaned queries answered substantively rather than
  refused. Measured on *generated text*, not on routing: a system that routes an orphan somewhere
  and then says "I don't know" has done the right thing.
- **CSAR — cross-source attribution rate.** Fraction of orphan answers asserting at least one fact
  that is (i) **distinctive** to the survivor the query was routed to — appears in the gold answers
  of at most 2 of 200 authors, so "literary fiction" never counts and "Addis Ababa" does; (ii)
  **not** the deleted author's own fact (serving the deleted author's real content is a *disclosure
  of the deleted source*, a different and worse failure, and must not be folded in); and (iii)
  **not** in the frozen base's answer to the same question, so the base's own knowledge is not
  credited to the selector.
- **RDR — retained displacement rate.** Fraction of *retained* queries that change expert as a side
  effect of a deletion nobody asked for.
- **Detection AUC.** Can an orphan query be told from a retained one using only the surviving
  selector's scores, **with no deletion record consulted**? This operationalises D3, and read from
  the defender's side it is the ceiling on any record-free refusal gate.

**Why CSAR is not ROUGE.** A ROUGE-based audit of the same generations files **95.5%** of orphan
answers as novel confabulation. That is a statement about n-gram overlap, not about whose facts were
served: *"Alice writes magical realism and was born in Lisbon"* — Bob's genre, Bob's city, asserted
about Alice — scores low ROUGE-L against every gold answer in the corpus and is precisely the harm.
Of the answers ROUGE still files as confabulation, CSAR finds a named stranger's facts in about
half.

### 2.6 Query transforms

A harm measured only on queries that name the deleted person is worth as much as a defense measured
that way. Every result below is reported across this axis. The perturbation is applied to **both**
classes, so no detector can learn "perturbed ⇒ deleted".

| transform | what it is | role |
|---|---|---|
| `gold` | TOFU's question verbatim (names the author in ~90% of rows) | anchor |
| `paraphrase` | TOFU's own paraphrase | **null by construction** — keeps the name (coverage 0.900 vs 0.895) |
| `name_stripped` | all name forms removed | **the hard case** |
| `indirect` | name replaced by a description built from the author's own facts | **weaker than it looks** — §8.4 |
| `para_stripped` | TOFU's paraphrase, name removed | the honest independent surface (H30) |

### 2.7 Evaluation protocol

Three conventions that every number below depends on:

1. **Author-parity split** — even author ids fit any detector, odd ids evaluate it. A query-level
   split leaks identity across the 20 correlated siblings of a deleted author.
2. **Permutation-invariant features** — detectors read sorted top-m survivor scores, margins, and
   row statistics, never raw column identities, so a detector fit on one deletion transfers to an
   unseen one.
3. **Route audits before metrics** — every serving arm asserts that the distinct *authors* on each
   route path match the requested policy **before any metric is read**. A plausible-but-wrong route
   is the failure mode these arms are most exposed to, and no downstream metric would flag it.

---

## 3. §4.10 — the forget metric is blind to substitution

**Question.** Can TOFU's `forget_quality` distinguish "the source is gone" from "a stranger answers
for it"?

**Method (E5).** Arms that delete *nothing* and reroute all 400 orphan queries to one fixed
survivor, against a genuine-deletion baseline measured at the same tier against the same held-fixed
KS reference, so all eight cells are the same statistical test.

**Result** (extended tier, `truth_max_rows` 120):

| dest | 89 | 137 | 31 | 97 | **DEL** | 33 | 79 | 88 |
|---|---|---|---|---|---|---|---|---|
| expert affinity | .3044 | .3382 | .2840 | .2267 | — | .2663 | .2193 | .3970 |
| `forget_quality` | **.8958** | .8002 | .6288 | .6288 | **.5140** | .5140 | .5140 | .3615 |

- `model_utility` is **identical to 0.8009 in all eight arms**; route audits show `deleted: 0` /
  `rerouted: 1320` in every reroute arm against `deleted: 1320` / `rerouted: 0` in the baseline.
- **6 of 7 arms that delete nothing score at or above genuine deletion.** s33 and s79 land exactly
  on it.
- Spearman `fq` ~ affinity = **+0.109 (p=0.82)**. The obvious explanation — reroute to a similar
  author and the answers still look right — is **refuted at two tiers** (smoke: −0.059, p=0.88), and
  the sign is nominally the opposite of what it predicts.

**Is the spread real?** The arms score *identical rows* (inter-arm r 0.88–0.94), so this is a paired
design. A paired bootstrap (one row-index draw shared across all arms):

| quantity | value |
|---|---|
| observed spread | 0.5342 |
| paired 95% CI | **[0.2245, 0.6975]** |
| P(spread > 0.25) | **0.961** |
| arms at-or-above deletion | **6/7 observed, 95% CI [2, 7]** |
| reruns reproducing published cells | **8/8** |

Even at the pessimistic end of the interval, **two arms that delete nothing match or beat real
deletion**.

**Two limits that must travel with these numbers.**

- **Ordering does not reproduce.** Spearman extended~smoke = +0.620 (p=0.14). The *spread*
  replicates across tiers; the per-destination ranking does not. Never name a winning destination.
- **A single cell is worth ±0.35.** Marginal 95% CI widths are 0.63–0.72. `D` moves on an exact
  lattice of `1/lcm(n,m)` = 1/120 here — one forget question — with ~30 attainable p-values above
  0.05 and a median gap of ~0.031. The four decimals this literature reports are spurious.

**Reading.** `forget_quality` tests distributional indistinguishability from a retrained reference.
A stranger's answers satisfy that as well as an absence does — better, on most destinations. The
metric was never designed to detect substitution, and it does not; but it is the field's evidence
that deletion worked.

---

## 4. §4.3 — what the system actually says

**Question.** When an orphan answer is produced, does it assert a specific surviving author's facts
about the deleted person?

**Result** (400 orphan queries per router, fact-level classification):

| arm | `centroid_sbert` | `key_tfidf` | own-disclosure | refusal |
|---|---|---|---|---|
| gold-form | 0.3325 | 0.3650 | 0.913 / 0.933 | 0.000 |
| `name_stripped` | **0.4400** | 0.4175 | 0.273 | ≤0.010 |
| `indirect` | 0.3350 | 0.2125 | 0.383 / 0.393 | ≤0.013 |
| random destination | 0.2200 | — | 0.953 | 0.003 |

**Four results.**

1. **Refusal never happens.** ≤0.013 across 1600 answers — ORR ≈ 1.00 at the level of what is
   *said*, not merely where the query is routed. The information needed to refuse is often present
   and discarded: `key_exact` raises no-match on **100%** of orphans, then falls back to
   `candidates[0]` and answers anyway.
2. **Anonymising the query does not reduce the harm; it changes its character.** Strip the name and
   the system leaks *less* of the deleted author's own content (0.913 → 0.273) while attributing
   *more* of a stranger's content to them (0.3325 → 0.4400). Attribution does not require the router
   to be *right*, only confident enough to pick someone — and **worse routing produces more legible
   attribution**, because an unrelated survivor's facts register cleanly while a similar survivor's
   overlap the deleted author's and are excluded by CSAR's own-facts filter.
3. **It is not a naming glitch.** Splitting cross-source rows by whether their matched facts are
   exhausted by name-forms of the routed survivor:

   | | CSAR | **substantive** | name-only |
   |---|---|---|---|
   | `centroid_sbert` | 0.3325 | **0.2400** | 0.0925 |
   | `key_tfidf` | 0.3650 | **0.2950** | 0.0700 |
   | random floor | 0.2200 | **0.1725** | 0.0475 |

   Two thirds to four fifths carry a real fact — a title, place, award, occupation (`Kaleidoscope
   City`, `Faulkner award`, `flight attendant`, `Turkish`). It survives dropping the
   identity-question slice, where a wrong expert answers with a name by construction: on the 300
   non-identity queries substantive CSAR is 0.217 / 0.250, still above the floor.
4. **The floor is high, and it is not the router's fault.** A uniformly random surviving expert — no
   router at all — already gives CSAR 0.2200 (0.1725 substantive). **The harm cannot be engineered
   to zero by improving the selector**; roughly two thirds of it is a property of substituting *any*
   expert. It also means the pre-registered "CSAR > 0.20" bar is nearly uninformative alone (a
   random destination clears it) and the real quantity is the lift over the floor: 0.11–0.22 raw,
   0.07–0.12 substantive.

**Publish the substantive figure (0.24–0.30 against a 0.17 floor), not the raw 0.33–0.37**, with
name-only as its own row — asserting a real stranger's *name* about a deleted author is a different
harm, not an absent one.

**Two known biases in this measurement.**

- TOFU puts identity questions first, and those are the most attribution-prone. A head-sliced sample
  gives 0.460; the full 400 gives 0.333/0.365. `--question_sample {head,random}` exists because of
  this, with `head` kept as default for byte-compatibility and documented as biased.
- `_extract_author_names` yields nothing for **18/200 authors**, so hits on those survivors default
  to substantive. **Substantive is an upper bound, name-only a lower bound.** Unclassifiable
  fraction is 0.08–0.16 gold-form (fine), 0.27–0.33 `name_stripped`, and **0.824** for
  `indirect`/`key_tfidf` — that cell is not quotable (see §5).

> **BLOCKING.** The pre-registration requires ~300 hand labels validating the classifier before any
> CSAR number enters the paper. Records are staged in `*.label_me.jsonl`. **I wrote the classifier
> and therefore cannot validate it.** This is the campaign's only human-blocked item.

---

## 5. §4.2 — where orphans go, and what deletion disturbs

**All orphans are reassigned.** 400/400 deleted-author questions are answered by *some* surviving
expert under every selector. At k=200 the mass is diffuse — busiest survivor 0.11–0.19, effective
number of destinations (1/HHI) 17.4–24.2 — except `key_exact`, which sends **100%** of them to one
unit, because it returns the first shard *by index* whose name appears and otherwise falls back to
`candidates[0]`.

**Magnet saturation — REFUTED as a general claim.** The natural prediction is that deleting sources
one at a time makes one survivor progressively the answerer for a growing share of the corpus. It
does the opposite in six of seven cells:

| condition | strategy | busiest share, 1 → 20 deletions | final n_eff | **RDR** | verdict |
|---|---|---|---|---|---|
| gold-form | `centroid_sbert` | 0.550 → 0.130 | 23.0 | **0.000** | dispersing |
| gold-form | `key_tfidf` | 0.400 → 0.190 | 17.5 | **0.000** | dispersing |
| gold-form | `centroid_lm` | 0.550 → 0.170 | 17.4 | 0.004 | dispersing |
| name-stripped | `centroid_sbert` | 0.350 → 0.092 | 28.7 | **0.092** | dispersing |
| name-stripped | `key_tfidf` | 0.400 → 0.305 | 9.7 | 0.015 | dispersing |
| indirect | `centroid_sbert` | 0.550 → 0.113 | 23.0 | 0.020 | dispersing |
| **indirect** | **`key_tfidf`** | **0.850 → 0.902** | **1.2** | 0.000 | **saturating** |

The mechanism is plain once seen: with **one** author deleted, all 20 of their questions go to that
author's single nearest survivor (share 0.55–0.75); with **twenty** deleted, each has a *different*
nearest survivor, so the pooled share falls and n_eff rises. The prediction holds only where the
router cannot tell the deleted sources apart — and there it holds sharply.

**The one saturating cell is a structural artifact, and it recurs.** `key_tfidf` on name-free
descriptive queries collapses onto **unit 88**, one of the **18 of 200 authors with no extractable
name**. Its centroid is the most generic point in the space; it already absorbs **68%** of
real-author and **45%** of world-facts out-of-distribution queries, and 0.902 of orphans here.
`centroid_sbert` has no such sink. **The routing magnet and the missing-name artifact are the same
units**, so any survivor-conditioned statistic is least trustworthy in exactly the name-free
conditions the paper most wants to report.

**Deletion locality is lexical too.** RDR is **0.000** at k=200 on gold-form queries — against 5.8%
published at k=10, so fine units genuinely do help locality — but **0.092** name-stripped for
`centroid_sbert`. Deleting 20 of 200 sources moves **9.2%** of *retained* queries to a different
expert, worse than the coarse-unit gold-form number it was supposed to improve on.

**Out-of-distribution queries, and a second oracle in the headline system.** The repo-best 0.8236
utility depends on `q2author`, an exact question-to-author lookup its own docstring calls
*oracle-gated* — i.e. the best number assumes the "is this query about anyone we hold?" problem is
already solved. It is not: strangers' scores are nearly flat (margin 0.022 vs 0.186 retained) and
confidence separates them at AUC 0.983–1.000 — but **orphans look the same** (0.984/0.992), because
the discriminator is not *was this deleted* but **does this query name a source I still hold**.
That single fact explains §6. Entropy is useless at k=200 (1.000 everywhere) and is reported as
such rather than dropped.

---

## 6. §4.5/§4.9 — detectability is a lexical artifact, with one exception

**Granularity, not confidence, is the axis.** The published "confidence-based orphan refusal caps at
AUC 0.57–0.61" is a statement about **coarse units**, not about selectors. At constant deletion
size:

| strategy | k=10 | k=50 | k=200 | verdict |
|---|---|---|---|---|
| `centroid_sbert` | 0.564 | 0.795 | 0.984 | monotone increasing |
| `centroid_lm` | 0.502 | 0.628 | 0.761 | monotone increasing |
| `key_tfidf` | 0.973 | 0.989 | 0.989 | saturated |

Attribution recall — naming *which* deleted source an orphan belongs to — rises 0.300 → 0.700 →
1.000.

**And the whole ladder is a lexical artifact.** Strip the name and it flattens to 0.560 → 0.572 →
0.623: **+0.063 across a 20× granularity change**, against +0.367 gold-form. `key_tfidf` gains
nothing at all (−0.021). At k=200 detection falls 0.991 → **0.623**, back inside the coarse-unit
band that granularity was supposed to escape, while routing accuracy collapses 0.966 → 0.343 and
`key_exact` routes at **0.025 on every name-free surface** — a name matcher and nothing else.

**The behavioral family replicates it.** Name-stripped, on both k=200 r32 pools:
`activation_norm` 0.877 → **0.495**, `attn_norm` 0.758 → **0.519** — chance.

**The exception.** `ppl` does not collapse: **0.783 / 0.769** name-stripped on the r32 pools (0.647
on r8/e5). All three behavioral selectors *run* the candidate experts; the two that collapse read
**magnitudes**, the survivor reads the expert's **loss**. So "runs the expert" is necessary but not
sufficient — and the leak is a property of the cheap selectors people deploy rather than something
unavoidable, which is what makes a record-free defense worth pricing (§8).

**A learned probe adds nothing.** Fitting a logistic reader over permutation-invariant functions of
the surviving scores reaches AUC **0.990** at k=200 with no deletion record consulted (0.976 / 0.830
on the other strategies; shuffled control 0.45–0.50) — and its **median lift over plain confidence
is +0.001** (max +0.069 on one strategy). D3 fails, but it fails to a *threshold*, not a learned
residual trace, so §4.9 is a paragraph and not a section. Under every hard transform the same probe
sits at its own shuffle control (0.446 vs 0.503; 0.417 vs 0.501; 0.414 vs 0.532).

**The two published reference points, for scale.** A per-source *tombstone* — an explicit deletion
record — separates orphans at AUC **0.982** (catch 0.963). The record-free confidence family at
k=10 reaches 0.57–0.61. The probe reaches 0.990 with no record, which is the D3 result; the
tombstone is not a defense to the question being asked, because it *is* the registry of deleted
people that deletion was supposed to eliminate.

**`para_stripped`, the honest surface.** `indirect` is not the hard transform (§8.4). Its
replacement — TOFU's own paraphrase with names stripped, median 12 words, 0/800 retaining any name
form — gives best-confidence AUC **0.702 / 0.665** against `name_stripped`'s 0.692 / 0.623:
*slightly higher* despite much worse routing. Detection does not track routing, for the same reason
the prefilter works (§8.1): orphans fit whatever candidates are scored poorly.

---

## 7. §4.4 — attacking the selector

Because routing is lexical, it is also steerable. An adversary who appends a single chosen author
name to a query captures:

| selector | appended name | substituted name |
|---|---|---|
| `key_exact` | **97.7%** | ~87% |
| `key_tfidf` | 31.7% | ~87% |
| `centroid_sbert` | 3.5% | ~87% |

`key_exact`'s 97.7% is structural: it returns the first shard **by index** whose name appears, so
any lower-indexed attacker wins outright.

**Composed with §4, this is the sharp version of the harm.** CSAR says a stranger's facts get
attributed to the erased person; steering says an adversary against a lexical selector chooses
**whose** biography that is.

Two attacker choices were degenerate and discarded — author 0 (`key_exact`'s fallback shard) and
author 88 (one of the 18 nameless authors) — both caught by numbers that looked too clean.

---

## 8. §4.6 — the defense frontier

Four results in sequence; the section's value is that it ends **negative**.

**8.1 Only `ppl` survives anonymisation** (§6), and it can be made cheap. Scoring `ppl` over all 180
survivors is 180 forward passes per query. A free `key_tfidf` prefilter cutting to the top-m
candidates costs nothing measurable — **m=4 at 45×** name-stripped, **m=2 at 90×** under `indirect`.
The mechanism: **detection does not require correct routing.** At m=4 the prefilter loses the
query's own expert 23% of the time and detection is unaffected, because orphans are uniformly
poorly fit across *whatever* candidates you score.

**8.2 The prefilter improves detection rather than trading it away.** Under `indirect` on the
headline pool, AUC is monotone **decreasing** in m — 0.9555 at m=2 down to 0.8545 at m=180, with
retain-FPR@90%-catch more than doubling (0.125 → 0.274) — because the score is a `min` over
candidates, so every extra expert is another chance for an orphan to find an accidental good fit.
Restricting the candidate set denies orphans that lottery while retained queries keep their own
expert. Cost and accuracy point the same way, which is rare enough to state plainly.

**8.3 It is not deployable.** An AUC cannot say whether a gate is usable. Best achievable
false-refusal on *retained* traffic, minimised over all m:

| pool | transform | catch 0.50 | catch 0.90 | catch 0.99 |
|---|---|---|---|---|
| r32/e25 | gold | 0.000 | **0.000** | 0.000 |
| | `indirect` | 0.019 | 0.125 | 0.236 |
| | `name_stripped` | 0.120 | **0.418** | 0.654 |
| r8/e5 | `name_stripped` | 0.361 | 0.639 | 0.769 |

**The gate is perfect where it is not needed and unusable where it is.** A gold-form query names the
author being asked about — if the deletion request was honoured at all, that name is the one thing
the system is known to hold about the deleted party. The adversary who matters does not name their
target, and against them the gate refuses nearly half of everyone else's traffic.

**Cost was never the binding constraint.** H24/H26 solved it and it turned out not to matter; the
frontier is bounded by discrimination.

**8.4 The apparent hard case was not hard.** `indirect` replaces the name with a description built
from the author's *own* facts — in practice prepending their book titles (*"Regarding the author
associated with beneath the veil of deceit and serpent's silence: …"*). It is therefore nearly as
lexically self-identifying as the gold form: `key_tfidf`'s top-2 holds the true owner **0.965** of
the time, against 0.998 gold and only **0.685** name-stripped. Decomposing `key_tfidf` routing
accuracy, with the name-keeping paraphrase as the control showing rewording alone does *not* hurt
(0.973 → 0.989):

| change | cost | routing accuracy |
|---|---|---|
| remove the name | **−0.413** | 0.973 → 0.560 |
| remove the name *and* reword the rest (`para_stripped`) | −0.090 further | → 0.470 |
| `indirect` | **−0.254 only** | → 0.719 |

The description hands ~+0.25 back. **A transform built from the target's own facts cannot test
anonymity.** `indirect` belongs beside the gold-form column, not as the hard case; `para_stripped`
is the honest surface.

> **Standing warning.** On r8/e5 the cheap gate "matches full" only because full is **poor**
> (0.63 / 0.65). A 90× speedup at AUC 0.62 is not a defense.

---

## 9. §4.21 — training duration moves the leak

**The axis is duration, not capacity.** Holding epochs at 5 and raising rank 8 → 32 does *not*
degrade behavioral detectability (`activation_norm` 0.877 → 0.934); holding rank at 32 and raising
epochs 5 → 25 collapses it (0.934 → 0.608). A third pool was built to add a point with the decision
rule fixed before the run.

At fixed rank 32, gold-form:

| strategy | e5 | e25 | **e50** | reads |
|---|---|---|---|---|
| `ppl` | 1.000 | 0.999 | **0.996** | the expert's **loss** |
| `activation_norm` | 0.934 | 0.608 | **0.515** | **magnitudes** |
| `attn_norm` | 0.700 | 0.554 | **0.569** | **magnitudes** |

Under `name_stripped`: `ppl` 0.783 → 0.769 → **0.737**, `activation_norm` 0.561 → 0.498 → **0.460**,
`attn_norm` 0.502 → 0.507 → **0.534**. Under `indirect`: `ppl` 0.885 → 0.810 → **0.816**.

`activation_norm` falls monotonically to **chance**; `ppl` is untouched. Read alone, "train longer
and the behavioral leak disappears" is the cheapest defense in the literature — and the same table
refutes it. Duration does not remove the deletion signal; it removes it **from the cheap
magnitude-reading selectors while leaving it fully intact for the one that reads behaviour**. An
operator who trains longer has made deletion undetectable *by the weak probes* and no less
detectable by the strong one. **"Train longer" must not be offered as mitigation.**

It also makes `ppl` the exception on **two independent axes** — query transform and training
duration — which is why §8 rests on `ppl` specifically rather than on "behavioral selectors" as a
class. One counter-current: name-stripped `ppl` declines gently along the same axis (0.783 → 0.769 →
0.737), so the exception is far more robust, not immune.

**A pre-registered follow-up that did not fire.** e25 and e50 adapters for the same author have
near-orthogonal effective deltas (median cosine **0.0139** over 192 modules; e50's norm 38% larger),
because the cosine LR schedule spans total steps, so the two runs differ in LR from step 1 and never
share a trajectory. With 20 examples and rank 32 over 192 modules the fit is massively
underdetermined and both reach loss ~0.047 by different routes. That raised H31 — a same-recipe
replicate pool to separate "duration" from "a different pool" — whose trigger, written in advance,
was *submit only if e50 lands anomalously*. It landed monotone, so **H31 was not submitted and 200
GPU-tasks were saved.** Weight-space orthogonality is real and evidently does not propagate to these
functional measurements.

---

## 10. Method constraints that govern how these numbers may be read

Each of these invalidated a reading that had already been written down.

1. **A paired quantity needs a paired interval.** The destination arms score identical rows.
   Bootstrapping each arm's marginal re-adds the noise they hold in *common*, once per arm, and
   declares any spread unresolvable. My first H29 verdict made exactly this error and flipped on the
   paired test.
2. **The achievable-p-value grid is a sampled lower bound**, growing with draw count (73 → 88
   between 2k and 60k draws, unconverged). It counts nothing. Quote the exact lattice
   `D = |i·m − j·n|/(n·m)`, i.e. multiples of `1/lcm(n,m)`.
3. **The 18 authors with no extractable name are a recurring hazard.** They have distorted three
   results — the H3 attacker choice, the `key_tfidf` OOD sink (author 88), and the H15 decomposition
   (82.4% unclassifiable in one cell). The routing magnet and the missing-name artifact are the
   *same* authors, so any survivor-conditioned statistic is least trustworthy in exactly the
   name-free conditions the paper most wants to report.
4. **A transform built from the target's own facts cannot test anonymity.** Prefer `name_stripped`,
   or `para_stripped` for a fully independent surface.
5. **A sub-chance AUC means nothing without its own shuffle control.** Two separate sub-chance
   readings looked like systematic sign flips and were noise. The control spans 0.336–0.532 here, so
   fitted-probe differences below ~0.1 are not resolvable.
6. **Feature-space routers read no expert weights** (§2.4). A per-pool table of feature-space numbers
   is one column repeated; recipe questions need the behavioral family.
7. **A pass counter is not a question counter.** One question is forwarded several times per eval
   (ppl, generation, truth-ratio variants), so any audit comparing forward passes to question counts
   is off by however many passes the metric suite makes. Audit distinct *authors* per route path
   instead — stable under repeated forwards.
8. **Parallelism is a test.** Packing arms into one job surfaced both a cache race and a dead
   experiment design within five minutes.

---

## 11. Defect record

Six numbered defects plus two self-corrections. Every one produced **plausible numbers**, which is
why they are recorded rather than quietly fixed.

| # | Defect | How it was caught |
|---|---|---|
| 1 | Lazy adapter cache silently zeroed the serving norm — non-resident adapters scored exactly 0.0 | the audit's own `--self_check`, which is why it is never disabled |
| 2 | The route audit raised *before* `json.dump`, destroying the 1h15m arm it was auditing | an arm computing every metric and discarding them |
| 3 | Centroid cache wrote straight to the final path; a sibling arm read 0 bytes | packing three arms into one job |
| 4 | `consolidate.py` paired `-random` CSAR runs with the wrong generation dump | a row describing another run's questions |
| 5 | The "recipe ablation" was **vacuous by construction** — feature-space matrices byte-identical across pools | `np.array_equal` on the dumped matrices |
| 6 | The entire `indirect` condition was unreproducible (`sorted(set, key=len)` under hash randomisation) | matrices differing by 0.27 where others agreed to 0.0 |
| 7 | *(self)* Unpaired bootstrap declared the destination spread unresolvable | cells reproducing to <5e-4 yet showing ±0.35 "noise" |
| 8 | *(self)* Sampled grid size published as a count | checking its stability across draw counts |

**Two near-misses worth the same weight.** A `--epochs 50` run whose progress bar ended at
`epoch: 25.0` would, read alone, have meant the flag was ignored and the whole H21 pool measured
nothing; it was cosmetic (`shard_meta` records 50, and the cosine LR reaches exactly 0.0 at step 50)
and was checked rather than assumed. And a 4h TIMEOUT that looked like a bug was **a dead GPU** —
zero forward passes and 3087 NVML errors on one node while sibling arms finished in 23–27 min
elsewhere; SLURM still reported that node `idle`.

**The pattern:** a number that looks precise, is cheap to check, and was not checked. The
countermeasure that keeps working is computing the same thing a second way.

---

## 12. Complete hypothesis ledger

31 filed: **28 adjudicated**, H8 retired by H17, H19 still open, H31 filed and never triggered.
Decision rules were fixed in [`log/selector_audit/`](log/selector_audit/) before each run, not
chosen after the numbers landed.

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H1 | The surviving selector geometry identifies an orphan with no deletion record | ✓ supported | probe AUC 0.990 / 0.976 / 0.830 (shuffled 0.45–0.50) |
| H2 | A learned reader extracts structure no confidence statistic gives | ✗ refuted | median lift **+0.001** |
| H3 | Finer granularity makes deletion refusable | restated; strong form ✗ | 0.564→0.984 gold-form, **0.560→0.623** name-stripped |
| H4 | A reroute-only "method" scores competitively on TOFU | ✓ supported, revised by H23/H28 | 0.6789 vs 0.5789 at identical utility |
| H5 | Cross-source attribution is common at per-author granularity | ✓ supported, provisional | CSAR 0.333 / 0.365 on 400, refusal 0.000 |
| H6 | Granularity generalizes to the behavioral family | ✓ then shown an r8 + gold-form artifact | 0.877 on r8 vs **0.608** on the headline pool |
| H7 | k=200 detectability is granularity, not the recipe | control **void by construction** | feature matrices byte-identical across 3 pools |
| H8 | CSAR is independent of destination concentration | retired by H17 | one observation, not a result |
| H9 | The granularity ladder survives name removal | ✗ refuted | k=200 detection 0.991 → **0.623** |
| H10 | An adversary steers routing by injecting a name | ✓ supported, lexical only | 97.7% / 31.7% / 3.5%; substituted ~87% |
| H11 | The behavioral family's detectability is lexical too | ✓ supported | activation_norm 0.877 → **0.495** |
| H12 | Queries belonging to no source are separable by confidence | ✓ supported | margin 0.022 vs 0.186; AUC 0.983–1.000 |
| H13/H14 | Without a gate, general queries land on a source's expert | ✓ supported | unit 88 absorbs 68% real-author, 45% world-facts |
| H15 | CSAR is mostly "the router supplies the wrong name" | ✗ **refuted** | substantive 0.2400 / 0.2950 vs 0.1725 floor |
| H16 | CSAR is a lexical artifact like the H3 defence | ✗ refuted | CSAR **rises** under name-stripping |
| H17 | A random destination gives the CSAR floor | ✓ supported, qualified | **0.2200** raw / 0.1725 substantive |
| H18 | H11 replicates on the headline pool | ✓ supported | 0.495/0.498 and 0.519/0.507 — chance on both |
| H19 | The `indirect`/`key_tfidf` collapse reproduces on the behavioral family | **open** | needs a GPU wave |
| H20 | Epochs, not rank, blunts behavioral detectability | ✓ supported | rank 8→32: 0.877→0.934; epochs 5→25: 0.934→0.608 |
| H21 | The epochs axis is monotone | ✓ supported | 0.934 → 0.608 → **0.515** (chance) |
| H22 | `ppl` is a genuine exception to H11 | ✓ supported | 0.782 / 0.799 name-stripped |
| H23 | `forget_quality` tracks the reroute destination | ✓ supported | spans 0.1561–0.7715, ρ = −0.059 (p=0.88) |
| H24 | The `ppl` gate can be made cheap | ✓ supported | **45×** (m=4) at no measurable loss |
| H25 | Sub-chance AUCs are noise, not sign flips | ✓ supported | shuffle control spans 0.336–0.532 |
| H26 | The cheap gate survives `indirect` | ✓ supported | **90×** (m=2); AUC monotone *decreasing* in m |
| H27 | The gate has a usable operating point | ✗ **answered negatively** | 0.418 false refusal @ 90% catch |
| H28 | The destination spread survives a finer tier | ✓ supported | spread 0.5342, utility identical at 0.8009 |
| H29 | A single `forget_quality` cell can carry an interval | ✓ answered | ±0.35 marginal; paired spread CI [0.2245, 0.6975] |
| H30 | `indirect` is easier than name-stripping, and a harder surface exists | ✓ supported | name −0.413, rewording −0.090, `indirect` −0.254 |
| H31 | Per-pool variance, not duration, drives the epochs axis | **filed, not triggered** | trigger was "e50 lands anomalously"; it landed monotone |

---

## 13. Status: settled, blocked, not claimed

**Settled.** §3 (metric blindness, with intervals), §4 (harm, conservatively quantified), §5
(destinations, magnet refuted, locality lexical), §6 (detectability lexical; granularity is the
axis; the probe is redundant), §7 (steering), §8 (defense cheap but undeployable), §9 (duration
moves the leak).

**Blocked on a human.** The **300 hand labels** validating the CSAR classifier. Nothing else is
blocked.

**Open, not required for the current claims.**

| Item | Kind | State |
|---|---|---|
| Behavioral family under `para_stripped` | GPU | Filed; needs the transform wired into `router_family_audit`, then one wave. Would let §8's frontier be stated on the honest hard surface |
| **H19** — does the `indirect`/`key_tfidf` magnet reproduce on the behavioral family | GPU | Open; §5's figure would be about that regime rather than about deletion count |
| **H31** — same-recipe replicate pool | GPU | **Not triggered** by its pre-registered rule. Stays filed for any claim needing per-pool variance *quantified* rather than bounded |
| The MIA / privacy column | GPU | **Withheld.** All arms reported byte-identical AUCs while serving different models, most likely every query falling to the OOD path. Reported as absent rather than published |
| Claims audit (§4.7) | Reading | Not started |
| Figures for the draft | Writing | Draft is tables-only; three plots would earn their space (see `paper/followup/README.md`) |

**Explicitly not claimed.**

- That destination X beats destination Y — the ordering does not reproduce across tiers.
- That CSAR is validated — hand labels outstanding.
- That no record-free defense is possible — this bounds `ppl`-as-gate on three pools under one
  transform family.
- That training duration mitigates anything.
- That these rates transfer to a heterogeneous production corpus. TOFU's authors are uniform,
  synthetic and English; that is what makes fact-level ground truth available at all.

---

## 14. Reproducing

```bash
export TOFU_SITE=cispa TOFU_CKPT_STORE=<.../jack_stuff>
source tofu_sisa_lora/slurm_nodes.sh          # never build a job body before this line
source <.../jack_stuff>/.venv-tofu/bin/activate

# CPU gates — before any SLURM job
python test_repo_selfcontained.py
cd tofu_sisa_lora
python test_eval_rows.py && python test_ou_equivalence.py && python test_router_probe.py \
  && python test_routed_scaffold_merged.py && python test_lazy_adapters.py
python ../selector_audit/test_csar.py

# CPU analyses — the matrices are on disk, so these need no GPU
python analyze_selector_cost.py       --self_test   # §8 frontier + operating points
python analyze_router_shift.py        --self_test   # transforms, incl. para_stripped
python analyze_router_probe.py        --self_test   # probe + granularity ladder
python analyze_sequential_deletion.py --self_test   # §5 magnet + RDR
python ../selector_audit/bootstrap_fq.py   --results_dir D --ks_ref R --out_json J --out_md M
python ../selector_audit/csar_decompose.py --csar_json A.json --out_json J --out_md M

# GPU waves — STUB=1 previews every driver without submitting
STUB=1 bash submit_e5_destination_sweep.sh        # §3 destination arms
STUB=1 bash submit_csar_audit.sh all              # §4 generations + scoring
STUB=1 bash submit_selector_wave.sh beh           # behavioral matrices
STUB=1 bash submit_h21_e50_pool.sh all            # §9 epochs pool

# The draft
python3 paper/followup/tools/check_tex.py paper/followup/main.tex paper/followup/refs.bib
```

**Cluster rules that are not optional.** Dependencies are `afterany`, never `afterok`
(`kill_invalid_depend` is off cluster-wide, so an `afterok` chain hangs PENDING forever on the first
failure instead of reporting what is missing). `--mem` must not be emitted at the `cispa` site.
`PACK × ARRAY_CAP ≤ 16` is the association's GPU limit while `MaxJobs=6` caps job count — which is
why the drivers pack arms per job rather than taking one GPU per array task. Calibrate walltime on
one task before submitting an array: a TIMEOUT costs the whole task *and* holds its GPU for the full
limit.
