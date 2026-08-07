### Target Date: 2026-08-07 (CSAR survives name-stripping — and rises)

Tenth entry today. The stress test that overturned H3
([lexical artifact](2026-08-07_h3-is-a-lexical-artifact.md)) applied to the other headline. It
does not overturn it. It strengthens it.

- **Hypotheses / what we're testing:**
  - **H16:** CSAR, like the H3 defence, is an artifact of TOFU questions naming their author.
    Name-stripping collapses routing accuracy from 0.966 to 0.343 at k=200, so destinations become
    near-arbitrary. REFUTE if CSAR holds or rises.
  - The pre-registered prediction, written before the run: it may not fall at all, because CSAR
    asks whether the served answer carries **the routed survivor's** distinctive facts, and an
    arbitrary expert still asserts its own facts. If so the harm does not depend on the router
    being good, which is a stronger claim than the current one.

- **Setup:** jobs **3191955** (`--query_transform name_stripped`) and **3191957** (`indirect`),
  k=200 e25 pool, forget10 deleted, `QPA 5`. All three arms sit on the **q0–q4** subset, so the
  control is the gold-form q0–q4 value of **0.460**, not the full-corpus 0.35
  ([sampling bias](2026-08-07_csar-full-400-and-a-sampling-bias.md)).

- **Results:** CSAR, with the gold-form q0–q4 control alongside:

  | served query | `centroid_sbert` | `key_tfidf` | refusal | routing accuracy (sbert) |
  |---|---|---|---|---|
  | gold-form | 0.460 | 0.460 | 0.000 | 0.966 |
  | indirect reference | 0.420 | 0.280 | 0.030 | 0.517 |
  | **name-stripped** | **0.530** | **0.500** | 0.010 / 0.020 | **0.343** |

- **What worked / hypothesis verdict:**
  - **H16 — REFUTED.** CSAR does not collapse when routing collapses. Under name-stripping it is
    **higher** than gold-form on both routers (0.530 vs 0.460; 0.500 vs 0.460), while routing
    accuracy falls by two thirds.
  - The predicted mechanism holds: attribution does not require the router to be *right*, only to
    be *confident enough to pick someone*. Whichever expert is activated asserts its own facts.
  - A plausible reason it *rises*: when routing is accurate the orphan lands on a **similar**
    author, whose facts overlap the deleted author's and are therefore excluded by CSAR's
    own-facts filter, or blend into a topically appropriate answer. When routing is arbitrary the
    served expert is unrelated, so its facts are unambiguously somebody else's and register
    cleanly. **Worse routing produces more legible attribution.**
  - Refusal rises from 0.000 to 0.010–0.030 — real, and still a rounding error. Stripping the name
    does not teach the system to decline.

- **Observations:**
  - **The two headlines behave in opposite directions under the same stress**, and that asymmetry
    is the most useful thing today produced. The *defence* (confidence-based orphan detection)
    was a lexical artifact: 0.991 → 0.623. The *harm* is not: 0.460 → 0.530. A deployment where
    people ask about someone without naming them is precisely where refusal stops working and
    attribution gets worse.
  - `indirect` sits between on `centroid_sbert` (0.420) and below both on `key_tfidf` (0.280),
    so the relationship to routing accuracy is not monotone and should not be presented as one.
    The safe claim is the bracketing one: across a 2.8× swing in routing accuracy, CSAR stays in
    0.28–0.53 and never approaches zero.
  - This makes H8 (CSAR independent of destination concentration) look right for a second reason,
    and folds it into a bigger claim: CSAR is insensitive to *where* orphans go, to *how
    concentrated* the destinations are, and now to *whether the router is any good*.

- **New questions / new hypotheses:**
  - **H17:** is CSAR bounded below by "any expert asserts its own facts"? The floor could be
    estimated directly by routing every orphan to a uniformly random surviving expert — a control
    arm that needs no router at all. If random routing gives ~0.5, then routing quality is
    irrelevant to the harm and the paper should say so in those terms.
  - The full-length re-run at `QPA 20` is still owed; these are q0–q4 numbers.

- **Next Steps:** run the random-destination control for H17. Re-run both transforms at `QPA 20`.
  Neither is needed to state the finding, both are needed to size it.
