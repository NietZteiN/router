# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `centroid_lm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.550 → 0.170 (Δ -0.380), final unit 88, tail stability 1.000, final n_eff 17.4, final RDR **0.004**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 113 | 0.550 | 2.9 | 0.000 |
| 2 | 181 | 40 | 113 | 0.275 | 8.2 | 0.000 |
| 3 | 182 | 60 | 113 | 0.183 | 13.1 | 0.000 |
| 10 | 189 | 200 | 88 | 0.125 | 17.5 | 0.002 |
| 11 | 190 | 220 | 88 | 0.118 | 18.6 | 0.003 |
| 18 | 197 | 360 | 88 | 0.183 | 15.5 | 0.004 |
| 19 | 198 | 380 | 88 | 0.176 | 16.4 | 0.004 |
| 20 | 199 | 400 | 88 | 0.170 | 17.4 | 0.004 |
| … | | | | | | |

## `centroid_sbert`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.750 → 0.110 (Δ -0.640), final unit 88, tail stability 0.400, final n_eff 24.2, final RDR **0.001**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 35 | 0.750 | 1.7 | 0.000 |
| 2 | 181 | 40 | 35 | 0.375 | 4.2 | 0.000 |
| 3 | 182 | 60 | 35 | 0.250 | 7.6 | 0.000 |
| 10 | 189 | 200 | 35 | 0.100 | 16.7 | 0.000 |
| 11 | 190 | 220 | 35 | 0.091 | 18.9 | 0.000 |
| 18 | 197 | 360 | 88 | 0.100 | 23.0 | 0.001 |
| 19 | 198 | 380 | 88 | 0.095 | 24.5 | 0.001 |
| 20 | 199 | 400 | 88 | 0.110 | 24.2 | 0.001 |
| … | | | | | | |

## `key_tfidf`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.400 → 0.190 (Δ -0.210), final unit 88, tail stability 1.000, final n_eff 17.5, final RDR **0.000**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.400 | 4.9 | 0.000 |
| 2 | 181 | 40 | 88 | 0.200 | 9.1 | 0.000 |
| 3 | 182 | 60 | 88 | 0.267 | 9.3 | 0.000 |
| 10 | 189 | 200 | 88 | 0.215 | 12.0 | 0.000 |
| 11 | 190 | 220 | 88 | 0.214 | 12.7 | 0.000 |
| 18 | 197 | 360 | 88 | 0.183 | 17.8 | 0.000 |
| 19 | 198 | 380 | 88 | 0.179 | 18.2 | 0.000 |
| 20 | 199 | 400 | 88 | 0.190 | 17.5 | 0.000 |
| … | | | | | | |
