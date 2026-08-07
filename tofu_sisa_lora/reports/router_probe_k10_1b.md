# Router-side orphan probe (E1)

Can the *surviving* selector geometry identify an orphan query with no deletion record?
Deleted columns are removed before any feature is computed; the probe is fit on even-id authors and evaluated on odd-id ones, so every evaluated deleted source is one the probe never saw.

Drop set: `9..9` (1 sources) · seed 42

| strategy | k | probe AUC | FPR@90% catch | best confidence | lift | sentinel | shuffled | oracle ceiling |
|---|---|---|---|---|---|---|---|---|
| activation_norm | 10 | **0.594** | 0.817 | 0.412 | 0.182 | — | 0.476 | 0.602 |
| attn_norm | 10 | **0.580** | 0.793 | 0.533 | 0.047 | — | 0.461 | 0.573 |
| logit_div | 10 | **0.649** | 0.712 | 0.633 | 0.016 | — | 0.445 | 0.576 |
| ppl | 10 | **1.000** | 0.000 | 0.998 | 0.002 | — | 0.421 | 0.603 |
| centroid_lm | 10 | **0.542** | 0.893 | 0.474 | 0.068 | 0.684 | 0.540 | 0.584 |
| centroid_lm_last | 10 | **0.426** | 0.872 | 0.505 | -0.079 | 0.799 | 0.476 | 0.548 |
| centroid_sbert | 10 | **0.673** | 0.726 | 0.564 | 0.109 | 0.814 | 0.526 | 0.534 |
| centroid_sbert_q | 10 | **0.690** | 0.736 | 0.606 | 0.084 | 0.826 | 0.444 | 0.551 |
| key_exact | 10 | _skipped_ | no graded score matrix (key_exact ships `match` only) | | | | | |
| key_tfidf | 10 | **0.986** | 0.038 | 0.973 | 0.013 | 0.884 | 0.466 | 0.504 |

## Deletion attribution — which source was removed

Sources of the eval half ranked by their queries' mean orphan probability. This is **score-access** attribution (the adversary reads the selector's score vector and has each candidate source's own questions) — not the black-box endpoint attack of §4.4, which needs generations.

| strategy | sources | deleted | top-1 is deleted | recall@n_deleted | source AUC |
|---|---|---|---|---|---|
| activation_norm | 91 | 10 | no | 0.200 | 0.660 |
| attn_norm | 91 | 10 | no | 0.000 | 0.625 |
| logit_div | 91 | 10 | no | 0.000 | 0.707 |
| ppl | 91 | 10 | yes | 1.000 | 1.000 |
| centroid_lm | 100 | 10 | no | 0.100 | 0.582 |
| centroid_lm_last | 100 | 10 | no | 0.000 | 0.388 |
| centroid_sbert | 100 | 10 | yes | 0.300 | 0.749 |
| centroid_sbert_q | 100 | 10 | yes | 0.300 | 0.790 |
| key_tfidf | 100 | 10 | yes | 1.000 | 1.000 |

## Verdict (pre-registered)

Best probe AUC **1.000** (ppl) → **headline (§4.9)**.
Bars: ≥0.85 headline · ≥0.65 subsection.
Published reference points: confidence family 0.57–0.61 (no deletion record) · author-rung sentinel 0.982 (needs the record).

Lift over the best confidence detector on the same eval half: max **0.182** (activation_norm), median 0.047 → mechanism reads as **confidence — the probe adds little a threshold does not already give**.

Absolute AUC alone cannot tell a learnable residual trace apart from the deleted source's own column having been removed. The lift is what separates them, so read the two columns together.
