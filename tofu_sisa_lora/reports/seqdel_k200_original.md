# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `centroid_sbert`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.550 → 0.130 (Δ -0.420), final unit 103, tail stability 0.600, final n_eff 23.0, final RDR **0.000**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 35 | 0.550 | 2.2 | 0.000 |
| 2 | 181 | 40 | 35 | 0.275 | 6.6 | 0.000 |
| 3 | 182 | 60 | 35 | 0.183 | 10.3 | 0.000 |
| 10 | 189 | 200 | 35 | 0.095 | 17.3 | 0.002 |
| 11 | 190 | 220 | 35 | 0.086 | 19.4 | 0.002 |
| 18 | 197 | 360 | 103 | 0.114 | 23.2 | 0.000 |
| 19 | 198 | 380 | 103 | 0.137 | 21.7 | 0.000 |
| 20 | 199 | 400 | 103 | 0.130 | 23.0 | 0.000 |
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
