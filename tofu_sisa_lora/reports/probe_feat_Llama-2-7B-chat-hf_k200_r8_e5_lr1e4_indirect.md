# Router-side orphan probe (E1)

Can the *surviving* selector geometry identify an orphan query with no deletion record?
Deleted columns are removed before any feature is computed; the probe is fit on even-id authors and evaluated on odd-id ones, so every evaluated deleted source is one the probe never saw.

Drop set: `180..199` (20 sources) · seed 42

| strategy | k | probe AUC | FPR@90% catch | best confidence | lift | sentinel | shuffled | oracle ceiling |
|---|---|---|---|---|---|---|---|---|
| centroid_lm | 200 | **0.571** | 0.841 | 0.456 | 0.115 | 0.476 | 0.331 | 0.529 |
| centroid_sbert | 200 | **0.635** | 0.663 | 0.638 | -0.002 | 0.810 | 0.529 | 0.572 |
| key_exact | 200 | _skipped_ | no graded score matrix (key_exact ships `match` only) | | | | | |
| key_tfidf | 200 | **0.795** | 0.620 | 0.789 | 0.007 | 0.966 | 0.711 | 0.546 |

## Deletion attribution — which source was removed

Sources of the eval half ranked by their queries' mean orphan probability. This is **score-access** attribution (the adversary reads the selector's score vector and has each candidate source's own questions) — not the black-box endpoint attack of §4.4, which needs generations.

| strategy | sources | deleted | top-1 is deleted | recall@n_deleted | source AUC |
|---|---|---|---|---|---|
| centroid_lm | 91 | 10 | no | 0.100 | 0.575 |
| centroid_sbert | 91 | 10 | no | 0.000 | 0.662 |
| key_tfidf | 91 | 10 | no | 0.400 | 0.851 |

## Verdict (pre-registered)

Best probe AUC **0.795** (key_tfidf) → **subsection**.
Bars: ≥0.85 headline · ≥0.65 subsection.
Published reference points: confidence family 0.57–0.61 (no deletion record) · author-rung sentinel 0.982 (needs the record).

Lift over the best confidence detector on the same eval half: max **0.115** (centroid_lm), median 0.007 → mechanism reads as **confidence — the probe adds little a threshold does not already give**.

Absolute AUC alone cannot tell a learnable residual trace apart from the deleted source's own column having been removed. The lift is what separates them, so read the two columns together.
