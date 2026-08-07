# Routing and orphan detection under query shift

The H3 granularity claim was measured on TOFU's gold-form questions, which name their author in ~90% of rows. This asks whether it survives queries that do not.

k = 200 · deleted = 20 units · 800 rows (400 forget / 400 retain) · attacker = author 0 (`Jaime Vasquez`)

`routing` = top-1 reaches the query's own unit (full pool). `capture` = share of queries about OTHER sources that land on the attacker. `conf` / `probe` = post-deletion orphan-detection AUC.

## `key_exact`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.880 | 0.123 | — | — |
| `paraphrase` | 0.887 | 0.115 | — | — |
| `name_stripped` | 0.025 | 1.000 | — | — |
| `indirect` | 0.025 | 1.000 | — | — |
| `name_injected` | 0.025 | 1.000 | — | — |
| `name_swapped` | 0.025 | 1.000 | — | — |

## `key_tfidf`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.973 | 0.000 | 0.993 | 0.990 |
| `paraphrase` | 0.989 | 0.000 | 0.991 | 0.989 |
| `name_stripped` | 0.560 | 0.004 | 0.692 | 0.446 |
| `indirect` | 0.716 | 0.000 | 0.755 | 0.719 |
| `name_injected` | 0.835 | 0.169 | 0.960 | 0.987 |
| `name_swapped` | 0.119 | 0.876 | 0.693 | 0.658 |

## `centroid_sbert`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.966 | 0.000 | 0.991 | 0.981 |
| `paraphrase` | 0.974 | 0.000 | 0.990 | 0.987 |
| `name_stripped` | 0.343 | 0.000 | 0.623 | 0.433 |
| `indirect` | 0.517 | 0.000 | 0.716 | 0.542 |
| `name_injected` | 0.966 | 0.010 | 0.989 | 0.982 |
| `name_swapped` | 0.150 | 0.828 | 0.612 | 0.649 |

## Read

- `key_exact` / adversarial injection: attacker capture 0.123 → **1.000**.
- `key_tfidf` / `name_stripped`: routing 0.973 → 0.560, detection 0.993 → 0.692 (Δ -0.301).
- `key_tfidf` / `indirect`: routing 0.973 → 0.716, detection 0.993 → 0.755 (Δ -0.238).
- `key_tfidf` / adversarial injection: attacker capture 0.000 → **0.169**.
- `centroid_sbert` / `name_stripped`: routing 0.966 → 0.343, detection 0.991 → 0.623 (Δ -0.368).
- `centroid_sbert` / `indirect`: routing 0.966 → 0.517, detection 0.991 → 0.716 (Δ -0.275).
- `centroid_sbert` / adversarial injection: attacker capture 0.000 → **0.010**.
