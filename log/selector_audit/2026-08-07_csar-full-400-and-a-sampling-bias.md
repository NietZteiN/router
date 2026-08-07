### Target Date: 2026-08-07 (CSAR on the full 400, and the sampling bias the pilot had)

Ninth entry today. Refines the number in [csar-pilot-h5](2026-08-07_csar-pilot-h5.md), which
stands as written. H5's verdict is unchanged; its magnitude is not.

- **Hypotheses / what we're testing:** the pilot's CSAR = 0.460 was measured on 100 of the 400
  orphan questions. Does it hold on all of them?

- **Setup:** job **3191718**, same pool and routers, `--questions_per_author 20` (all 400).

- **Results:**

  | run | n | `centroid_sbert` | `key_tfidf` | refusal |
  |---|---|---|---|---|
  | pilot (`QPA 5`) | 100 | 0.460 | 0.460 | 0.000 |
  | **full (`QPA 20`)** | **400** | **0.333** | **0.365** | **0.000** |

  Splitting the full run by question index inside each author reproduces the gap exactly:

  | question positions | `centroid_sbert` | `key_tfidf` |
  |---|---|---|
  | q0–q4 (what the pilot sampled) | **0.460** | **0.460** |
  | q5–q19 | 0.290 | 0.333 |

- **What worked / hypothesis verdict:**
  - **H5 still SUPPORTED** — 0.333 / 0.365 against a 0.20 bar — but the honest full-corpus figure
    is **~0.35, not 0.46**, and the pilot's number should not be quoted.
  - **refusal = 0.000 across all 800 answers.** Unchanged, now on four times the evidence.
  - The two strategies now differ (0.333 vs 0.365), which retires the earlier worry about their
    identical pilot values: that was the coincidence the per-row check said it was.

- **Observations:**
  - **The cause is a sampling bias I built the tooling to avoid, on an axis I did not think
    about.** `--questions_per_author` exists precisely because `--max_questions` head-slices the
    *author* list and would have measured two people. It then head-slices each author's
    *question* list. TOFU orders each author's questions with the identity ones first — *"What is
    the full name of the female author who was born in Santiago, Chile in 1977?"* — and those are
    the most attribution-prone, because a wrong expert answers them with a **name**. Later
    questions ask about perception and accomplishments, where attribution is muddier: *"How does
    the public perceive Rajeev Majumdar's books?"*
  - So the pilot did not measure 100 arbitrary orphan queries; it measured the 100 most
    identity-shaped ones. Same error as the one already fixed, one level down.
  - `--question_sample {head,random}` now exists, seeded, with `head` kept as the default for
    byte-compatibility and documented as biased with the measured size of the bias.
  - **The name_stripped and indirect arms (jobs 3191955 / 3191957) were launched with `QPA 5`
    head sampling**, so they sit on the q0–q4 subset. That does not invalidate the comparison they
    exist for — their control is the gold-form q0–q4 value of **0.460**, not the full-corpus 0.35
    — but the comparison must be stated against the matching subset, and they should be re-run at
    full length before anything goes in a paper.

- **New questions / new hypotheses:**
  - **H15:** CSAR may be a function of question TYPE rather than of routing as such. Identity
    questions invite a name and the fact-level metric then catches it. A per-question-type
    breakdown is cheap on the dump already on disk and would say whether "cross-source
    attribution" is largely "the router supplies the wrong name when asked for a name".

- **Next Steps:** re-run name_stripped/indirect at `QPA 20` once the queue drains. Break the full
  run down by question type for H15. Neither changes the H5 verdict; both change what it means.
