### Target Date: 2026-08-10 (H18/H7 on the e25 pool, and a mispairing bug in the consolidator)

Third entry today. Prompted by a bug hunt over the pipeline, which turned up both a real defect
and the fact that the r32 e25 behavioral arms had landed.

- **Hypotheses / what we're testing:** **H18/H7** — does H11 (detection is lexical) replicate off
  the r8 pool, and does the behavioral family's detectability depend on the training recipe?
  The feature-space family provably cannot (it never reads expert weights — see the 08-07
  correction), so the behavioral family is the only one that can answer H7 at all.

- **Setup:** jobs 3200586 / 3200587, `r32 e25` pool, one arm per job at a 12 h wall after the
  6 h waves timed out. `self_check 3/3` on every strategy in both arms.

- **Results:** best-confidence AUC, k=200, forget10 deleted:

  | strategy | r8 gold-form | **e25 gold-form** | r8 name-stripped | **e25 name-stripped** |
  |---|---|---|---|---|
  | `activation_norm` | 0.877 | **0.608** | 0.495 | **0.498** |
  | `attn_norm` | 0.758 | **0.554** | 0.519 | **0.507** |
  | `ppl` | 0.993 | 0.999 | 0.647 | **0.769** |

  Routing accuracy on the e25 pool collapses with it: `activation_norm` 0.349 → **0.046**,
  `attn_norm` 0.310 → **0.041**, `ppl` 1.000 → 0.635.

- **What worked / hypothesis verdict:**
  - **H11 REPLICATES.** Name-stripping drives `activation_norm` and `attn_norm` to chance on
    **both** pools (0.495/0.498 and 0.519/0.507). The central claim — no selector family detects
    orphans once the query stops naming the source — is no longer resting on one pool.
  - **H7 — SUPPORTED, and it cuts against H6.** The behavioral family *is* recipe-dependent, and
    strongly: `activation_norm` reads **0.877 on r8 but 0.608 on e25**, `attn_norm` 0.758 vs
    0.554. On the **headline pool** the behavioral family never had good detectability even on
    gold-form queries — 0.608 and 0.554 sit barely above the 0.57–0.61 coarse-unit band that the
    whole granularity story was supposed to escape.
  - So **H6's strong form was an r8 artifact on top of a gold-form artifact.** Its entry recorded
    it as provisional pending H11; it turns out to have been provisional pending H18 as well.
    The honest version: granularity buys the behavioral family little on a well-trained pool and
    nothing at all without names.
  - The direction is mechanically plausible: better-fit experts (rank 32, 25 epochs) produce
    large activations on *any* query, so an activation-norm score discriminates less. Worth
    stating as a hypothesis, not a finding — one pool pair is not a trend.

- **Observations:**
  - **Bug found in `consolidate.py`.** `question_type_breakdown` located each CSAR file's
    generation dump by `stem.split("_centroid")[0]`, which mapped
    `csar_..._centroid_sbert-random` to the **gold-form** dump. The H15 rows for both `-random`
    runs were therefore computed against another run's questions, and the `random` strategy
    vanished from that table entirely because none of its rows could be found in the wrong file.
    The CSAR table itself was never affected — it reads the csar JSONs directly.
    Fixed by preferring the exact stem and **refusing to score a pair whose strategy sets
    disagree**, which turns the silent mismatch into a printed SKIP. With the fix the `random`
    control appears with its own numbers (identity 0.524 / other 0.203 at qpa20).
  - This is the fourth silent-numbers defect of the campaign, and like the others it produced a
    plausible table rather than an error. The pattern across all four is the same: a lookup or a
    counter that is *approximately* right and never cross-checked against what it claims to
    describe.

- **New questions / new hypotheses:**
  - **H20:** is behavioral detectability inversely related to expert fit? r8/e5 gives 0.877 and
    r32/e25 gives 0.608. The third pool (`r32 e5`) separates rank from epochs and is the arm
    still running.

- **Next Steps:** read the `r32 e5` arm when it lands — it is the one that says whether rank or
  training length drives H20. The finalize job (3201164) will pick it up automatically.
