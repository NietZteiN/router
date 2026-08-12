# 2026-08-12 — H15 REFUTED: CSAR is not mostly a swapped name

Thread: `selector_audit/`. Closes the H15 filed on 2026-08-07 in
[csar-full-400-and-a-sampling-bias](2026-08-07_csar-full-400-and-a-sampling-bias.md). CPU only,
over the CSAR JSONs already on disk.

## The objection H15 names

*"CSAR may largely be **the router supplies the wrong name when asked for a name**."* If most
cross-source attribution is the served answer emitting the survivor's name instead of the deleted
author's, then §4.3's harm is a naming glitch rather than the transfer of a stranger's biography —
a much weaker claim, and the first thing a referee would ask.

The original plan was a question-type detector. A sharper test was available without one: `csar.py`
already records **which** survivor facts it found in each answer (`hits`). Classify those hits as
name-forms of the routed survivor versus anything else, and the question answers itself — it asks
directly what is being attributed rather than inferring it from what was asked.

Survivor name forms come from `router._extract_author_names`, matched token-wise (length > 2,
substring either direction, possessive stripped).

## Result — REFUTED

Gold-form, 400 orphan queries per router:

| router | CSAR | **substantive** | name-only | substantive share of CSAR |
|---|---|---|---|---|
| `centroid_sbert` | 0.3325 | **0.2400** | 0.0925 | 0.722 |
| `key_tfidf` | 0.3650 | **0.2950** | 0.0700 | 0.808 |
| random destination (H17 floor) | 0.2200 | **0.1725** | — | 0.784 |

**Two thirds to four fifths of cross-source attribution carries at least one non-name fact** — a
book title, a birthplace, an award, an occupation. `Kaleidoscope City`, `Faulkner award`,
`flight attendant`, `Turkish` are among the most frequent hits, alongside the names.

**The identity-question effect is real but does not explain CSAR.** TOFU orders each author's 20
questions with the identity-seeking ones first, which is what produced the earlier q0-q4 sampling
bias:

| router | slice | n | CSAR | substantive | name-only |
|---|---|---|---|---|---|
| `centroid_sbert` | q0–q4 | 100 | 0.4600 | 0.3100 | 0.1500 |
| | q5–q19 | 300 | 0.2900 | 0.2167 | 0.0733 |
| `key_tfidf` | q0–q4 | 100 | 0.4600 | **0.4300** | 0.0300 |
| | q5–q19 | 300 | 0.3333 | 0.2500 | 0.0833 |

Identity questions do attract more attribution (0.46 vs 0.29/0.33). But **even there the
attribution is mostly substantive** — strikingly so for `key_tfidf`, where 0.43 of the 0.46 carries
a real fact and only 0.03 is a bare name swap. And on the non-identity slice, which is 300 of the
400 queries, substantive CSAR stays at **0.217 / 0.250**, still above the random-destination
substantive floor of 0.1725.

So the harm survives its own strongest objection twice over: dropping every name-only case, and
dropping the identity-question slice as well.

## What to quote, and what this does not do

**Quote the substantive number.** §4.3's defensible headline is **substantive CSAR ≈ 0.24–0.30
gold-form against a 0.17 random-destination floor**, not the raw 0.33–0.37. The conservative
number is the one that survives the referee.

**The name-only cases are not nothing.** Asserting some other real person's *name* about a deleted
author is still misattribution of identity — it is a different harm, not an absent one, and it
should be reported as its own row rather than folded in or discarded.

**Three limits, all in the same direction:**

1. `_extract_author_names` yields nothing for **18 of 200 authors**, and a hit on one of those
   survivors cannot be classified — it defaults to *substantive*. So **substantive is an upper
   bound and name-only a lower bound.** Gold-form this affects 7.5% (`centroid_sbert`) and 15.8%
   (`key_tfidf`) of cross-source rows, small enough to leave the conclusion intact.
2. **Under the name-free transforms it is not small, and under one it is fatal.** Unclassifiable
   fraction of cross-source rows: `name_stripped` **0.27–0.33**, `indirect`/`centroid_sbert`
   **0.34**, and `indirect`/`key_tfidf` **0.824**. Those cells are **not quotable**.

   The 0.824 is not noise, and it identifies itself: `key_tfidf` on name-free descriptive queries
   collapses onto **unit 88** — the nameless sink author that already absorbs 68% of real-author
   and 45% of world-facts queries (2026-08-07 OOD entry) and that `analyze_sequential_deletion`
   found taking 0.902 of orphans on descriptive queries. The magnet and the missing-name artifact
   are the same authors, so they compound: the router funnels orphans onto exactly the survivor
   whose name the classifier cannot recognise. Any number conditioned on the routed survivor is
   therefore *least* trustworthy in precisely the condition the paper most wants to report.

   These 18 authors have now distorted a measurement three times — the H3 attacker choice, the
   `key_tfidf` OOD sink, and this. They should be excluded or flagged by default in any
   per-survivor statistic.
3. This decomposes **the classifier's own output**, so it inherits every error the classifier
   makes. It does **not** substitute for the 300 hand labels the pre-registration requires before
   any CSAR number reaches the paper. I wrote the classifier; I cannot validate it.

## Status

- **H15 REFUTED** — CSAR is not mostly a swapped name, on either the what-is-attributed test or
  the what-was-asked test.
- §4.3 gains a conservative headline (substantive CSAR vs a substantive random floor) and an
  explicit second row for identity-only misattribution.
- **Still blocking**: the 300 hand labels. Unchanged by this entry.
- New standing rule: statistics conditioned on the routed survivor must exclude or flag the 18
  name-less authors.
