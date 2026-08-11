# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `centroid_lm`

**saturating — one survivor takes a growing share** — busiest share 0.350 → 0.610 (Δ 0.260), final unit 88, tail stability 1.000, final n_eff 2.6, final RDR **0.010**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 88 | 0.350 | 4.4 | 0.000 |
| 2 | 181 | 40 | 88 | 0.475 | 3.6 | 0.000 |
| 3 | 182 | 60 | 88 | 0.533 | 3.2 | 0.000 |
| 10 | 189 | 200 | 88 | 0.525 | 3.3 | 0.000 |
| 11 | 190 | 220 | 88 | 0.550 | 3.1 | 0.000 |
| 18 | 197 | 360 | 88 | 0.606 | 2.6 | 0.011 |
| 19 | 198 | 380 | 88 | 0.600 | 2.6 | 0.012 |
| 20 | 199 | 400 | 88 | 0.610 | 2.6 | 0.010 |
| … | | | | | | |

## `centroid_sbert`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.450 → 0.362 (Δ -0.088), final unit 88, tail stability 1.000, final n_eff 6.9, final RDR **0.025**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 35 | 0.450 | 3.7 | 0.000 |
| 2 | 181 | 40 | 88 | 0.250 | 7.1 | 0.000 |
| 3 | 182 | 60 | 88 | 0.333 | 6.6 | 0.000 |
| 10 | 189 | 200 | 88 | 0.295 | 8.6 | 0.012 |
| 11 | 190 | 220 | 88 | 0.295 | 8.8 | 0.014 |
| 18 | 197 | 360 | 88 | 0.358 | 6.9 | 0.018 |
| 19 | 198 | 380 | 88 | 0.345 | 7.5 | 0.021 |
| 20 | 199 | 400 | 88 | 0.362 | 6.9 | 0.025 |
| … | | | | | | |

## `key_tfidf`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.400 → 0.305 (Δ -0.095), final unit 88, tail stability 1.000, final n_eff 9.7, final RDR **0.037**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.400 | 4.9 | 0.000 |
| 2 | 181 | 40 | 88 | 0.200 | 9.1 | 0.001 |
| 3 | 182 | 60 | 88 | 0.267 | 9.3 | 0.000 |
| 10 | 189 | 200 | 88 | 0.290 | 9.9 | 0.015 |
| 11 | 190 | 220 | 88 | 0.282 | 10.5 | 0.016 |
| 18 | 197 | 360 | 88 | 0.286 | 10.8 | 0.032 |
| 19 | 198 | 380 | 88 | 0.297 | 10.1 | 0.033 |
| 20 | 199 | 400 | 88 | 0.305 | 9.7 | 0.037 |
| … | | | | | | |
