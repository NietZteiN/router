# Routing and orphan detection under query shift

The H3 granularity claim was measured on TOFU's gold-form questions, which name their author in ~90% of rows. This asks whether it survives queries that do not.

k = 50 · deleted = 5 units · 800 rows (400 forget / 400 retain) · attacker = author 1 (`Chukwu Akabueze`)

`routing` = top-1 reaches the query's own unit (full pool). `capture` = share of queries about OTHER sources that land on the attacker. `conf` / `probe` = post-deletion orphan-detection AUC.

## `key_exact`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.884 | 0.129 | — | — |
| `paraphrase` | 0.890 | 0.122 | — | — |
| `name_stripped` | 0.100 | 1.000 | — | — |
| `indirect` | 0.100 | 1.000 | — | — |
| `name_injected` | 0.100 | 1.000 | — | — |
| `name_swapped` | 0.100 | 1.000 | — | — |

## `key_tfidf`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.980 | 0.000 | 0.991 | 0.988 |
| `paraphrase` | 0.984 | 0.000 | 0.991 | 0.988 |
| `name_stripped` | 0.517 | 0.003 | 0.697 | 0.569 |
| `indirect` | 0.892 | 0.000 | 0.809 | 0.782 |
| `name_injected` | 0.560 | 0.489 | 0.772 | 0.988 |
| `name_swapped` | 0.199 | 0.868 | 0.682 | 0.735 |

## `centroid_sbert`

| condition | routing | attacker capture | conf AUC | probe AUC |
|---|---|---|---|---|
| `original` | 0.752 | 0.000 | 0.744 | 0.826 |
| `paraphrase` | 0.761 | 0.000 | 0.766 | 0.857 |
| `name_stripped` | 0.200 | 0.000 | 0.572 | 0.443 |
| `indirect` | 0.261 | 0.000 | 0.527 | 0.565 |
| `name_injected` | 0.756 | 0.037 | 0.733 | 0.757 |
| `name_swapped` | 0.207 | 0.643 | 0.584 | 0.617 |

## Read

- `key_exact` / adversarial injection: attacker capture 0.129 → **1.000**.
- `key_tfidf` / `name_stripped`: routing 0.980 → 0.517, detection 0.991 → 0.697 (Δ -0.294).
- `key_tfidf` / `indirect`: routing 0.980 → 0.892, detection 0.991 → 0.809 (Δ -0.182).
- `key_tfidf` / adversarial injection: attacker capture 0.000 → **0.489**.
- `centroid_sbert` / `name_stripped`: routing 0.752 → 0.200, detection 0.744 → 0.572 (Δ -0.172).
- `centroid_sbert` / `indirect`: routing 0.752 → 0.261, detection 0.744 → 0.527 (Δ -0.217).
- `centroid_sbert` / adversarial injection: attacker capture 0.000 → **0.037**.
