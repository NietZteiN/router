# Router-side orphan probe (E1)

Can the *surviving* selector geometry identify an orphan query with no deletion record?
Deleted columns are removed before any feature is computed; the probe is fit on even-id authors and evaluated on odd-id ones, so every evaluated deleted source is one the probe never saw.

Drop set: `180..199` (20 sources) · seed 42

| strategy | k | probe AUC | FPR@90% catch | best confidence | lift | sentinel | shuffled | oracle ceiling |
|---|---|---|---|---|---|---|---|---|
| centroid_lm | 200 | **0.830** | 0.428 | 0.761 | 0.069 | 0.885 | 0.502 | 0.568 |
| centroid_sbert | 200 | **0.976** | 0.042 | 0.984 | -0.008 | 0.999 | 0.445 | 0.520 |
| key_exact | 200 | _skipped_ | no graded score matrix (key_exact ships `match` only) | | | | | |
| key_tfidf | 200 | **0.990** | 0.033 | 0.989 | 0.001 | 1.000 | 0.459 | 0.489 |

## Deletion attribution — which source was removed

Sources of the eval half ranked by their queries' mean orphan probability. This is **score-access** attribution (the adversary reads the selector's score vector and has each candidate source's own questions) — not the black-box endpoint attack of §4.4, which needs generations.

| strategy | sources | deleted | top-1 is deleted | recall@n_deleted | source AUC |
|---|---|---|---|---|---|
| centroid_lm | 100 | 10 | yes | 0.500 | 0.930 |
| centroid_sbert | 100 | 10 | yes | 1.000 | 1.000 |
| key_tfidf | 100 | 10 | yes | 1.000 | 1.000 |

## Verdict (pre-registered)

Best probe AUC **0.990** (key_tfidf) → **headline (§4.9)**.
Bars: ≥0.85 headline · ≥0.65 subsection.
Published reference points: confidence family 0.57–0.61 (no deletion record) · author-rung sentinel 0.982 (needs the record).

Lift over the best confidence detector on the same eval half: max **0.069** (centroid_lm), median 0.001 → mechanism reads as **confidence — the probe adds little a threshold does not already give**.

Absolute AUC alone cannot tell a learnable residual trace apart from the deleted source's own column having been removed. The lift is what separates them, so read the two columns together.
