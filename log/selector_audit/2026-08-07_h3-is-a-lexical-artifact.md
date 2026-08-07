### Target Date: 2026-08-07 (H3 RESTATED: the granularity ladder is a lexical artifact)

Fifth entry today. **Supersedes the H3 verdict** in
[behavioral-at-k200-wave](2026-08-07_behavioral-at-k200-wave.md), which stands as written per the
append-only protocol. Prompted by a question that turned out to go straight at the weakest joint
in the day's headline: *does rewording or paraphrasing cause misrouting, and can an adversary
force one by putting someone else's name in the query?*

- **Hypotheses / what we're testing:**
  - **H9 (the stress test).** The H3 ladder — confidence-based orphan detection rising 0.564 →
    0.795 → 0.984 over k = 10/50/200 — was measured entirely on TOFU's gold-form questions, which
    name their author in ~90% of rows. If detection works *because the query contains the deleted
    person's name*, "per-source granularity makes deletion refusable" is a claim about a benchmark
    artifact, not about deployments where third parties ask about someone without naming them.
    CONFIRM (ladder is real): detection AUC holds within ~0.05 of gold-form under name removal.
    REFUTE: it collapses toward the 0.57–0.61 band the literature reports for coarse units.
  - **H10 (adversarial steering).** An attacker who can put text in the query can choose the
    destination. Composed with CSAR = 0.460, that means choosing **whose** biography gets
    attributed to the person who asked to be erased.

- **Setup:** new producer `analyze_router_shift.py`, CPU only — none of `key_exact` / `key_tfidf` /
  `centroid_sbert` reads expert weights, the same fact that voided the H7 recipe control, used
  here for leverage. Evaluation set = the **800 rows TOFU ships paraphrases for** (400 forget +
  400 retain). Every condition is applied to the SAME rows and to BOTH classes: perturbing only
  orphans would let the detector learn "perturbed ⇒ deleted", measuring the experiment instead of
  the router. Attacker = author 1 (`Basil Mahfouz Al-Kuwaiti`); seed 42.

  ```bash
  python analyze_router_shift.py --self_test
  python analyze_router_shift.py --k 200 --drop_set 180-199 --attacker_id 1 \
      --out_json reports/router_shift_k200_att1.json --out_md reports/router_shift_k200_att1.md
  ```

- **Results:**

  **The ladder, under each condition** (best-confidence AUC, `centroid_sbert` / `key_tfidf`):

  | condition | k=10 | k=50 | k=200 | Δ across the ladder |
  |---|---|---|---|---|
  | original (gold-form) | 0.624 / 0.974 | 0.744 / 0.991 | **0.991** / 0.993 | **+0.367** / +0.019 |
  | **name_stripped** | 0.560 / 0.713 | 0.572 / 0.697 | **0.623** / 0.692 | **+0.063** / −0.021 |
  | indirect reference | 0.595 / 0.804 | 0.527 / 0.809 | 0.716 / 0.762 | +0.121 / −0.042 |

  **Remove the name and the ladder flattens.** `centroid_sbert` gains +0.063 over a 20×
  granularity change instead of +0.367; `key_tfidf` gains nothing at all (−0.021). At k=200,
  detection falls 0.991 → **0.623** — back inside the 0.57–0.61 band the literature reports for
  coarse units.

  Routing accuracy collapses with it (k=200): `centroid_sbert` 0.966 → 0.343, `key_tfidf`
  0.973 → 0.560, `key_exact` 0.880 → 0.025.

  **Paraphrase is a null, exactly as designed.** TOFU's own `paraphrased_question` gives routing
  0.974 / 0.989 and detection 0.990 / 0.991 — indistinguishable from gold-form, because those
  paraphrases **keep the author's name** (coverage 0.900 paraphrased vs 0.895 original). Anyone
  who runs "the paraphrase robustness experiment" on this benchmark will report a reassuring
  near-null that means nothing.

  **Indirect reference recovers about half.** No name, but a definite description built from the
  author's own distinctive facts: routing 0.517 / 0.720, detection 0.716 / 0.762 at k=200. So
  identity *content* carries real signal — the router is not purely string-matching — but the
  literal name was doing most of the work.

  **Adversarial steering (k=200, attacker = author 1):**

  | router | capture, original | **capture, name injected** | capture, name swapped |
  |---|---|---|---|
  | `key_exact` | 0.000 | **0.977** | 0.874 |
  | `key_tfidf` | 0.000 | **0.317** | 0.873 |
  | `centroid_sbert` | 0.000 | **0.035** | 0.859 |

  *Injected* = the query still names its true subject and gains the attacker's name in an
  innocuous carrier ("… (as discussed by X)"). *Swapped* = the true name replaced, the
  name-following upper bound.

  `key_exact` is hijacked by **97.7%** of a single appended name — `KeyRouter.route` returns the
  first shard *by index* whose name appears, so any attacker with a lower index wins outright.
  That is the answer to "just use a lexical or classifier router". The dense routers shrug off a
  mention (`centroid_sbert` 0.035, routing essentially unchanged at 0.958) but follow a
  substitution: swap the name and capture is **0.86–0.87 in every family**.

- **What worked / hypothesis verdict:**
  - **H9 — REFUTED, and H3's strong reading with it.** The ladder is real *on gold-form queries*
    and is largely a **lexical-identity artifact**. The honest statement is narrower: *finer
    partitions let a confidence threshold exploit the source's name when the query contains it;
    they buy almost nothing when it does not.* The claim "per-source granularity makes deletion
    refusable" does not survive, and §4.6's defense section cannot rest on it.
  - **H10 — SUPPORTED for lexical selectors, REFUTED for dense ones under injection.** An
    appended name captures 97.7% against `key_exact` and 31.7% against `key_tfidf`, but only 3.5%
    against `centroid_sbert`. A *substituted* name captures ~87% everywhere. So the attack is
    real but its strength is a property of the selector family, not of routing as such.

- **Observations:**
  - **Two attacker choices were degenerate and their runs were discarded.** Attacker 0 is
    `key_exact`'s fallback shard, so "capture 1.000" under name-stripping was the no-match
    fallback, not an attack. Attacker 88 is one of **18 authors with no extractable name at all**,
    so injection and swap were no-ops and every capture read 0.000. Both were caught by the
    numbers looking too clean, and the fix — a named, low-index, non-fallback attacker — is now
    the documented default. At k=10 and k=50 the attacker's *unit* still coincides with the
    fallback unit, so `key_exact` capture is unreadable there; only the k=200 cell is clean.
  - The shift analyzer reproduces the ladder's **shape** on its own 800-row balanced subset
    (0.624 → 0.744 → 0.991) against the npz-based ladder's 0.564 → 0.795 → 0.984. Same monotone
    story, different eval subset; the discrepancy is the subset, not the method, and is stated
    rather than smoothed.
  - Composed with the CSAR result: orphan answers already attribute a **named stranger's** facts
    to the deleted person 46% of the time, and against a lexical router an adversary picks which
    stranger. That is a materially worse harm story than "reassignment happens".

- **New questions / new hypotheses:**
  - **H11:** does the *behavioral* family (running when `sw-beh` lands) show the same lexical
    dependence? It scores by running experts on the query rather than by matching text, so it may
    degrade differently — the one family with a mechanism to be name-independent.
  - Does CSAR itself survive name-stripping? If routing collapses to 0.343 the destinations become
    near-random, which should *lower* attribution — worth measuring, since it would mean the
    headline harm is also partly a gold-form artifact.
  - The indirect condition is built from the author's own gold answers, so it is a generous
    construction. A harder version (descriptions a stranger could plausibly know) would lower it
    further.

- **Next Steps:** restate H3 wherever it is quoted; do not put the ladder in the paper without the
  name-stripped column beside it. Re-run CSAR on name-stripped queries. Fold `key_exact`'s 0.977
  capture into the "just add a classifier router" rebuttal in §4.6.
