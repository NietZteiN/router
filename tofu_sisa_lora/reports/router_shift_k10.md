# Routing and orphan detection under query shift

The H3 granularity claim was measured on TOFU's gold-form questions, which name their author in ~90% of rows. This asks whether it survives queries that do not.

k = 10 · deleted = 1 units · 800 rows (400 forget / 400 retain) · attacker = author 1 (`Chukwu Akabueze`)

`routing` = top-1 reaches the query's own unit (full pool). `capture` = share of queries about OTHER sources that land on the attacker. `conf` / `probe` = post-deletion orphan-detection AUC.

## `key_exact`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.948 | 0.105 | — | — |
| `paraphrase` | 0.950 | 0.100 | — | — |
| `name_stripped` | 0.500 | 1.000 | — | — |
| `indirect` | 0.500 | 1.000 | — | — |
| `name_injected` | 0.500 | 1.000 | — | — |
| `name_swapped` | 0.500 | 1.000 | — | — |

## `key_tfidf`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.979 | 0.000 | 0.974 | 0.978 |
| `paraphrase` | 0.981 | 0.013 | 0.978 | 0.989 |
| `name_stripped` | 0.514 | 0.025 | 0.713 | 0.728 |
| `indirect` | 0.826 | 0.005 | 0.804 | 0.869 |
| `name_injected` | 0.759 | 0.482 | 0.993 | 0.995 |
| `name_swapped` | 0.537 | 0.882 | 0.770 | 0.795 |

## `centroid_sbert`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.560 | 0.065 | 0.624 | 0.539 |
| `paraphrase` | 0.545 | 0.048 | 0.658 | 0.601 |
| `name_stripped` | 0.212 | 0.055 | 0.560 | 0.417 |
| `indirect` | 0.263 | 0.005 | 0.595 | 0.435 |
| `name_injected` | 0.696 | 0.122 | 0.638 | 0.643 |
| `name_swapped` | 0.598 | 0.532 | 0.632 | 0.586 |

## Read

- `key_exact` / adversarial injection: attacker capture 0.105 → **1.000**.
- `key_tfidf` / `name_stripped`: routing 0.979 → 0.514, detection 0.974 → 0.713 (Δ -0.261).
- `key_tfidf` / `indirect`: routing 0.979 → 0.826, detection 0.974 → 0.804 (Δ -0.170).
- `key_tfidf` / adversarial injection: attacker capture 0.000 → **0.482**.
- `centroid_sbert` / `name_stripped`: routing 0.560 → 0.212, detection 0.624 → 0.560 (Δ -0.065).
- `centroid_sbert` / `indirect`: routing 0.560 → 0.263, detection 0.624 → 0.595 (Δ -0.029).
- `centroid_sbert` / adversarial injection: attacker capture 0.065 → **0.122**.
