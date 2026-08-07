# Routing and orphan detection under query shift

The H3 granularity claim was measured on TOFU's gold-form questions, which name their author in ~90% of rows. This asks whether it survives queries that do not.

k = 200 · deleted = 20 units · 800 rows (400 forget / 400 retain) · attacker = author 1 (`Chukwu Akabueze`)

`routing` = top-1 reaches the query's own unit (full pool). `capture` = share of queries about OTHER sources that land on the attacker. `conf` / `probe` = post-deletion orphan-detection AUC.

## `key_exact`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.880 | 0.000 | — | — |
| `paraphrase` | 0.887 | 0.000 | — | — |
| `name_stripped` | 0.025 | 0.000 | — | — |
| `indirect` | 0.025 | 0.000 | — | — |
| `name_injected` | 0.048 | 0.977 | — | — |
| `name_swapped` | 0.028 | 0.874 | — | — |

## `key_tfidf`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.973 | 0.000 | 0.993 | 0.990 |
| `paraphrase` | 0.989 | 0.000 | 0.991 | 0.989 |
| `name_stripped` | 0.560 | 0.001 | 0.692 | 0.446 |
| `indirect` | 0.720 | 0.000 | 0.762 | 0.715 |
| `name_injected` | 0.691 | 0.317 | 0.842 | 0.966 |
| `name_swapped` | 0.121 | 0.873 | 0.706 | 0.661 |

## `centroid_sbert`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.966 | 0.000 | 0.991 | 0.981 |
| `paraphrase` | 0.974 | 0.000 | 0.990 | 0.987 |
| `name_stripped` | 0.343 | 0.000 | 0.623 | 0.433 |
| `indirect` | 0.517 | 0.000 | 0.718 | 0.543 |
| `name_injected` | 0.958 | 0.035 | 0.978 | 0.965 |
| `name_swapped` | 0.128 | 0.859 | 0.595 | 0.479 |

## Read

- `key_exact` / adversarial injection: attacker capture 0.000 → **0.977**.
- `key_tfidf` / `name_stripped`: routing 0.973 → 0.560, detection 0.993 → 0.692 (Δ -0.301).
- `key_tfidf` / `indirect`: routing 0.973 → 0.720, detection 0.993 → 0.762 (Δ -0.231).
- `key_tfidf` / adversarial injection: attacker capture 0.000 → **0.317**.
- `centroid_sbert` / `name_stripped`: routing 0.966 → 0.343, detection 0.991 → 0.623 (Δ -0.368).
- `centroid_sbert` / `indirect`: routing 0.966 → 0.517, detection 0.991 → 0.718 (Δ -0.273).
- `centroid_sbert` / adversarial injection: attacker capture 0.000 → **0.035**.
