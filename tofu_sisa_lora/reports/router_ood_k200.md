# Queries that belong to no source

The headline routed system decides TOFU-vs-OOD with `q2author`, an exact question-to-author lookup — an **oracle** a deployment does not have. Without it every general-knowledge query gets some source's expert applied. This asks whether the selector's own scores could replace that oracle.

k = 200 · deleted = 20 units

`AUC vs retain` = can negated top-1 confidence separate this group from retained traffic? High = a threshold would work.

## `key_exact`

| group | n | top-1 | margin | entropy | busiest unit | n_eff | AUC vs retain |
|---|---|---|---|---|---|---|---|
| `retain` | 400 | no-match 0.140 | | | | | |
| `orphan` | 400 | no-match 1.000 | | | | | |
| `ood_real_authors` | 100 | no-match 1.000 | | | | | |
| `ood_world_facts` | 117 | no-match 1.000 | | | | | |

## `key_tfidf`

| group | n | top-1 | margin | entropy | busiest unit | n_eff | AUC vs retain |
|---|---|---|---|---|---|---|---|
| `retain` | 400 | 0.606 | 0.471 | 1.000 | 0.050 | 20.6 | — |
| `orphan` | 400 | 0.147 | 0.045 | 1.000 | 0.190 | 17.5 | 0.992 |
| `ood_real_authors` | 100 | 0.185 | 0.079 | 1.000 | 0.680 | 2.1 | 0.983 |
| `ood_world_facts` | 117 | 0.107 | 0.025 | 1.000 | 0.453 | 4.7 | 0.997 |

## `centroid_sbert`

| group | n | top-1 | margin | entropy | busiest unit | n_eff | AUC vs retain |
|---|---|---|---|---|---|---|---|
| `retain` | 400 | 0.582 | 0.186 | 1.000 | 0.050 | 21.1 | — |
| `orphan` | 400 | 0.391 | 0.029 | 1.000 | 0.130 | 23.0 | 0.984 |
| `ood_real_authors` | 100 | 0.330 | 0.019 | 1.000 | 0.080 | 35.0 | 0.997 |
| `ood_world_facts` | 117 | 0.151 | 0.022 | 1.000 | 0.051 | 49.4 | 1.000 |
