### Target Date: 2026-08-07 (CORRECTION: the feature-space recipe control cannot test H7)

Third entry today. Corrects the H7 arm of
[behavioral-at-k200-wave](2026-08-07_behavioral-at-k200-wave.md) — that entry stands as written,
per the append-only protocol; this one supersedes its H7 framing.

- **Hypotheses / what we're testing:** no new run. This is an adjudication of an experiment
  already in flight, forced by its own output.

- **Setup:** job **3191703** (`sw-feat`), the two k=200 pools never audited (`r32 e5`, `r8 e5`),
  feature-space battery `key_exact key_tfidf centroid_sbert centroid_lm`. Registered as the H7
  control: *"if the r32 e5 and r8 e5 pools at the same k=200 show the same detectability,
  granularity is the cause; if they do not, the e25 recipe is a confound."*

- **Results:** both arms emitted **byte-identical** cells, to each other and to the published e25
  numbers — `key_tfidf d199` top3_share 0.750 / adequacy 0.194, and the forget10 cell 0.297 /
  0.270, matching `rl_family_leak_table.md`'s k=200 `key_tfidf` rows exactly. Both finished with
  `self_check 50/50` on all four strategies and identical `full_top1_acc` (key_exact 0.892,
  key_tfidf 0.978, centroid_sbert 0.968, centroid_lm 0.725).

  Checked exactly rather than by eye: `np.array_equal` on the dumped `scores` matrices between
  `r32 e5` and `r8 e5` returns **True** for `key_tfidf`, `centroid_sbert` and `centroid_lm`, and
  the full cell dicts compare equal. Two pools differing in adapter rank (32 vs 8) and epochs
  (25 vs 5) produce the same 4000x200 score matrix to the bit.

- **What worked / hypothesis verdict:** **the H7 control is void, and could never have been
  otherwise.** The feature-space battery does not read expert weights at all:

  | strategy | what it scores against | depends on the pool? |
  |---|---|---|
  | `key_exact` | member names extracted from TOFU questions | no |
  | `key_tfidf` | TF-IDF over the unit's training questions | no |
  | `centroid_sbert` | MiniLM question-embedding centroids | no |
  | `centroid_lm` | hidden states of the **plain base**, adapters disabled | no |

  The last row is the one I got wrong. `centroid_lm` sounds model-dependent, and is — but on the
  *base* model, not the pool: with feature-space-only strategies `build_real_resources` sets
  `res.lm = _NoAdapterLM(base)`, and `router.py` specifies adapters-disabled hidden states for
  this router regardless. All three k=200 pools share `Llama-2-7B-chat-hf`, so all four columns
  are a function of (base model, TOFU questions) alone. Three pools, three identical answers, by
  construction.

  So H7 is not tested by `sw-feat`. **It is tested by `sw-beh`** — `ppl`, `activation_norm` and
  `attn_norm` score by running the candidate experts, so they are the only strategies in the
  battery that can distinguish `r32 e25` from `r32 e5` from `r8 e5` at fixed k=200. That arm is
  running and unaffected; H7 stays open, with its evidence coming from the behavioral wave rather
  than the feature-space one.

- **Observations:**
  - The identity is not wasted: it re-confirms the repo's documented router-independence assert
    (*"routing depends only on question embeddings — route stats must be BIT-IDENTICAL across
    expert pools"*) across three training recipes rather than two pools, and it materializes the
    npz for those pools so later analyses need not special-case missing files. That is worth
    something; it is not what I said it was worth.
  - **How the error survived design review, and what catches it next time.** I reasoned about
    granularity (which the pools share) and recipe (which they do not) without checking which
    *inputs each strategy actually reads*. The table above is the check, and it takes a minute.
    Any future "recipe control" or "pool control" in this thread must name, per strategy, the
    input it varies — a control over a variable the measurement does not consume is not a control.
  - Nothing downstream was contaminated: E1's ladder (H3) varies k on the same base and the same
    router family, so pool-independence is exactly why those rungs are comparable. The
    correction narrows what `sw-feat` proves; it does not touch the ladder.

- **New questions / new hypotheses:**
  - H7 restated for the arm that can answer it: **do `ppl`/`activation_norm`/`attn_norm` at k=200
    give the same detectability across `r32 e25`, `r32 e5` and `r8 e5`?** If yes, granularity
    dominates the training recipe *and* the adapter rank. If no, the behavioral family's
    detectability is a property of how well-fit the experts are, which would be a sharper and
    more troubling result: deletion would be more observable on better-trained pools.

- **Next Steps:** read `sw-beh` for H6 and the restated H7 together — one job answers both. The
  freed capacity went to the CSAR pilot at the full 400 questions (job 3191718, `QPA=20`),
  which sharpens H5, rather than to more identical feature-space cells.
