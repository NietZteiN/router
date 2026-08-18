# Routing metrics vs number of sources deleted

k = 200 per-author units · deletion order `180-199` (nested prefixes) · attacker = author 0 · 800 rows.

`is_forget` is recomputed at every rung — a row is an orphan only if its OWN author was deleted — and routing is post-deletion (argmax over survivors). `RDR` is the share of RETAINED rows whose served unit moved versus no deletion at all.

## `centroid_sbert` · `original`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.9663 | — | 0.0000 | 0.0000 | — |
| 1 | 20 | 0.9654 | — | 0.0000 | 0.0000 | 2.2 |
| 2 | 40 | 0.9671 | 0.9917 | 0.0000 | 0.0000 | 6.6 |
| 5 | 100 | 0.9686 | 0.9792 | 0.0014 | 0.0000 | 10.0 |
| 10 | 200 | 0.9717 | 0.9862 | 0.0017 | 0.0000 | 17.3 |
| 20 | 400 | 0.9725 | 0.9913 | 0.0000 | 0.0000 | 23.0 |
| 40 ⚠ | 400 | 0.9725 | 0.9916 | 0.0000 | 0.0000 | 21.7 |
| 80 ⚠ | 400 | 0.9775 | 0.9887 | 0.0200 | 0.0000 | 19.4 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `centroid_sbert` · `name_stripped`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.3425 | — | 0.0000 | 0.0000 | — |
| 1 | 20 | 0.3256 | — | 0.0000 | 0.0000 | 4.2 |
| 2 | 40 | 0.3342 | 0.6261 | 0.0000 | 0.0000 | 11.9 |
| 5 | 100 | 0.3343 | 0.6336 | 0.0286 | 0.0000 | 22.9 |
| 10 | 200 | 0.2950 | 0.6037 | 0.0383 | 0.0000 | 24.9 |
| 20 | 400 | 0.1975 | 0.6231 | 0.0925 | 0.0000 | 28.7 |
| 40 ⚠ | 400 | 0.2050 | 0.6287 | 0.1150 | 0.0000 | 27.2 |
| 80 ⚠ | 400 | 0.2250 | 0.6098 | 0.3450 | 0.0000 | 20.2 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `centroid_sbert` · `name_injected`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.9575 | — | 0.0000 | 0.0000 | — |
| 1 | 20 | 0.9577 | — | 0.0000 | 0.0000 | 1.9 |
| 2 | 40 | 0.9592 | 0.9900 | 0.0000 | 0.0000 | 5.4 |
| 5 | 100 | 0.9600 | 0.9640 | 0.0000 | 0.0000 | 8.7 |
| 10 | 200 | 0.9617 | 0.9787 | 0.0000 | 0.0000 | 10.9 |
| 20 | 400 | 0.9650 | 0.9782 | 0.0000 | 0.0000 | 8.9 |
| 40 ⚠ | 400 | 0.9650 | 0.9782 | 0.0000 | 0.0000 | 8.0 |
| 80 ⚠ | 400 | 0.9650 | 0.9782 | 0.0075 | 0.0000 | 6.9 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `centroid_sbert` · `name_swapped`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.1275 | — | 0.0000 | 0.0000 | — |
| 1 | 20 | 0.1269 | — | 0.0000 | 0.0000 | 1.1 |
| 2 | 40 | 0.1303 | 0.7402 | 0.0000 | 0.0000 | 1.2 |
| 5 | 100 | 0.1400 | 0.5863 | 0.0014 | 0.0000 | 1.2 |
| 10 | 200 | 0.1300 | 0.6524 | 0.0017 | 0.0000 | 1.4 |
| 20 | 400 | 0.1625 | 0.5955 | 0.0000 | 0.0000 | 1.3 |
| 40 ⚠ | 400 | 0.1625 | 0.5955 | 0.0000 | 0.0000 | 1.2 |
| 80 ⚠ | 400 | 0.1675 | 0.5945 | 0.0200 | 0.0000 | 1.2 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `key_tfidf` · `original`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.9725 | — | 0.0000 | 0.0000 | — |
| 1 | 20 | 0.9718 | — | 0.0000 | 0.0000 | 4.9 |
| 2 | 40 | 0.9724 | 0.9967 | 0.0000 | 0.0013 | 9.1 |
| 5 | 100 | 0.9729 | 0.9815 | 0.0000 | 0.0013 | 9.9 |
| 10 | 200 | 0.9750 | 0.9913 | 0.0000 | 0.0026 | 12.0 |
| 20 | 400 | 0.9825 | 0.9930 | 0.0000 | 0.0026 | 17.5 |
| 40 ⚠ | 400 | 0.9825 | 0.9930 | 0.0000 | 0.0038 | 15.7 |
| 80 ⚠ | 400 | 0.9825 | 0.9941 | 0.0000 | 0.0051 | 12.3 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `key_tfidf` · `name_stripped`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.5600 | — | 0.0000 | 0.0038 | — |
| 1 | 20 | 0.5487 | — | 0.0000 | 0.0038 | 4.9 |
| 2 | 40 | 0.5487 | 0.7128 | 0.0013 | 0.0038 | 9.1 |
| 5 | 100 | 0.5543 | 0.7297 | 0.0014 | 0.0038 | 8.6 |
| 10 | 200 | 0.5167 | 0.7428 | 0.0050 | 0.0038 | 9.9 |
| 20 | 400 | 0.4700 | 0.6917 | 0.0150 | 0.0038 | 9.7 |
| 40 ⚠ | 400 | 0.4800 | 0.6963 | 0.0450 | 0.0064 | 8.8 |
| 80 ⚠ | 400 | 0.4925 | 0.7155 | 0.1025 | 0.0077 | 7.1 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `key_tfidf` · `name_injected`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.6913 | — | 0.0000 | 0.0000 | — |
| 1 | 20 | 0.6846 | — | 0.0000 | 0.0000 | 1.0 |
| 2 | 40 | 0.6842 | 0.8163 | 0.0000 | 0.0000 | 1.0 |
| 5 | 100 | 0.6871 | 0.7087 | 0.0000 | 0.0000 | 1.0 |
| 10 | 200 | 0.6850 | 0.7896 | 0.0000 | 0.0000 | 1.0 |
| 20 | 400 | 0.6475 | 0.8422 | 0.0000 | 0.0000 | 1.0 |
| 40 ⚠ | 400 | 0.6475 | 0.8422 | 0.0000 | 0.0000 | 1.0 |
| 80 ⚠ | 400 | 0.6475 | 0.8422 | 0.0000 | 0.0000 | 1.0 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `key_tfidf` · `name_swapped`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.1212 | — | 0.0000 | 0.0000 | — |
| 1 | 20 | 0.1231 | — | 0.0000 | 0.0000 | 1.1 |
| 2 | 40 | 0.1250 | 0.6385 | 0.0000 | 0.0000 | 1.2 |
| 5 | 100 | 0.1314 | 0.5578 | 0.0000 | 0.0000 | 1.1 |
| 10 | 200 | 0.1200 | 0.6782 | 0.0000 | 0.0000 | 1.4 |
| 20 | 400 | 0.1725 | 0.7061 | 0.0000 | 0.0000 | 1.2 |
| 40 ⚠ | 400 | 0.1725 | 0.7061 | 0.0000 | 0.0000 | 1.2 |
| 80 ⚠ | 400 | 0.1725 | 0.7062 | 0.0000 | 0.0000 | 1.2 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `key_exact` · `original`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.8800 | — | 0.0000 | 0.1231 | — |
| 1 | 20 | 0.8782 | — | 0.0000 | 0.1474 | 1.0 |
| 2 | 40 | 0.8776 | — | 0.0000 | 0.1705 | 1.0 |
| 5 | 100 | 0.8729 | — | 0.0000 | 0.2423 | 1.0 |
| 10 | 200 | 0.8917 | — | 0.0000 | 0.3397 | 1.0 |
| 20 | 400 | 0.8650 | — | 0.0000 | 0.5821 | 1.0 |
| 40 ⚠ | 400 | 0.8650 | — | 0.0000 | 0.5821 | 1.0 |
| 80 ⚠ | 400 | 0.8650 | — | 0.0000 | 0.5821 | 1.0 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `key_exact` · `name_stripped`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.0250 | — | 0.0000 | 1.0000 | — |
| 1 | 20 | 0.0256 | — | 0.0000 | 1.0000 | 1.0 |
| 2 | 40 | 0.0263 | — | 0.0000 | 1.0000 | 1.0 |
| 5 | 100 | 0.0286 | — | 0.0000 | 1.0000 | 1.0 |
| 10 | 200 | 0.0333 | — | 0.0000 | 1.0000 | 1.0 |
| 20 | 400 | 0.0500 | — | 0.0000 | 1.0000 | 1.0 |
| 40 ⚠ | 400 | 0.0500 | — | 0.0000 | 1.0000 | 1.0 |
| 80 ⚠ | 400 | 0.0500 | — | 0.0000 | 1.0000 | 1.0 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `key_exact` · `name_injected`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.0475 | — | 0.0000 | 0.0000 | — |
| 1 | 20 | 0.0487 | — | 0.0000 | 0.0000 | 1.0 |
| 2 | 40 | 0.0500 | — | 0.0000 | 0.0000 | 1.0 |
| 5 | 100 | 0.0543 | — | 0.0000 | 0.0000 | 1.0 |
| 10 | 200 | 0.0633 | — | 0.0000 | 0.0000 | 1.0 |
| 20 | 400 | 0.0950 | — | 0.0000 | 0.0000 | 1.0 |
| 40 ⚠ | 400 | 0.0950 | — | 0.0000 | 0.0000 | 1.0 |
| 80 ⚠ | 400 | 0.0950 | — | 0.0000 | 0.0000 | 1.0 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.

## `key_exact` · `name_swapped`

| deleted | orphan rows | routing acc (retain) | detection AUC | RDR | attacker capture | orphan n_eff |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.0275 | — | 0.0000 | 0.1231 | — |
| 1 | 20 | 0.0282 | — | 0.0000 | 0.1231 | 1.1 |
| 2 | 40 | 0.0289 | — | 0.0000 | 0.1231 | 1.2 |
| 5 | 100 | 0.0314 | — | 0.0000 | 0.1231 | 1.1 |
| 10 | 200 | 0.0367 | — | 0.0000 | 0.1231 | 1.4 |
| 20 | 400 | 0.0550 | — | 0.0000 | 0.1231 | 1.2 |
| 40 ⚠ | 400 | 0.0550 | — | 0.0000 | 0.1231 | 1.2 |
| 80 ⚠ | 400 | 0.0550 | — | 0.0000 | 0.1231 | 1.2 |

> ⚠ At these rungs the extra deleted authors have **no rows in the evaluation set** (it covers authors 0–19 and 180–199 only), so the orphan count does not grow. `RDR` and `routing acc` remain meaningful — the survivor pool really is smaller — but the orphan-side columns describe the same rows as the rung above and must not be read as a plateau in deletion size.
