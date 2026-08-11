# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `centroid_lm`

**saturated — one survivor already holds most orphans** — busiest share 0.950 → 0.810 (Δ -0.140), final unit 88, tail stability 1.000, final n_eff 1.5, final RDR **0.000**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 88 | 0.950 | 1.1 | 0.000 |
| 2 | 181 | 40 | 88 | 0.975 | 1.1 | 0.000 |
| 3 | 182 | 60 | 88 | 0.983 | 1.0 | 0.000 |
| 10 | 189 | 200 | 88 | 0.795 | 1.5 | 0.000 |
| 11 | 190 | 220 | 88 | 0.782 | 1.6 | 0.000 |
| 18 | 197 | 360 | 88 | 0.794 | 1.6 | 0.000 |
| 19 | 198 | 380 | 88 | 0.803 | 1.5 | 0.000 |
| 20 | 199 | 400 | 88 | 0.810 | 1.5 | 0.000 |
| … | | | | | | |

## `centroid_sbert`

**flat** — busiest share 0.450 → 0.465 (Δ 0.015), final unit 88, tail stability 1.000, final n_eff 4.3, final RDR **0.045**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.450 | 3.3 | 0.000 |
| 2 | 181 | 40 | 88 | 0.375 | 4.5 | 0.000 |
| 3 | 182 | 60 | 88 | 0.267 | 5.6 | 0.000 |
| 10 | 189 | 200 | 88 | 0.385 | 5.4 | 0.027 |
| 11 | 190 | 220 | 88 | 0.377 | 5.7 | 0.028 |
| 18 | 197 | 360 | 88 | 0.481 | 4.1 | 0.025 |
| 19 | 198 | 380 | 88 | 0.458 | 4.5 | 0.029 |
| 20 | 199 | 400 | 88 | 0.465 | 4.3 | 0.045 |
| … | | | | | | |

## `key_tfidf`

**saturating — one survivor takes a growing share** — busiest share 0.850 → 0.902 (Δ 0.052), final unit 88, tail stability 1.000, final n_eff 1.2, final RDR **0.013**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.850 | 1.3 | 0.000 |
| 2 | 181 | 40 | 88 | 0.550 | 2.1 | 0.000 |
| 3 | 182 | 60 | 88 | 0.700 | 1.8 | 0.000 |
| 10 | 189 | 200 | 88 | 0.900 | 1.2 | 0.003 |
| 11 | 190 | 220 | 88 | 0.873 | 1.3 | 0.003 |
| 18 | 197 | 360 | 88 | 0.903 | 1.2 | 0.009 |
| 19 | 198 | 380 | 88 | 0.908 | 1.2 | 0.012 |
| 20 | 199 | 400 | 88 | 0.902 | 1.2 | 0.013 |
| … | | | | | | |
