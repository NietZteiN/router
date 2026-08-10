# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `centroid_sbert`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.350 → 0.092 (Δ -0.257), final unit 88, tail stability 0.800, final n_eff 28.7, final RDR **0.092**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 35 | 0.350 | 4.2 | 0.000 |
| 2 | 181 | 40 | 35 | 0.175 | 11.9 | 0.000 |
| 3 | 182 | 60 | 35 | 0.117 | 16.4 | 0.000 |
| 10 | 189 | 200 | 52 | 0.090 | 24.9 | 0.038 |
| 11 | 190 | 220 | 52 | 0.082 | 26.7 | 0.041 |
| 18 | 197 | 360 | 88 | 0.097 | 28.2 | 0.077 |
| 19 | 198 | 380 | 88 | 0.092 | 27.7 | 0.081 |
| 20 | 199 | 400 | 88 | 0.092 | 28.7 | 0.092 |
| … | | | | | | |

## `key_tfidf`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.400 → 0.305 (Δ -0.095), final unit 88, tail stability 1.000, final n_eff 9.7, final RDR **0.015**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.400 | 4.9 | 0.000 |
| 2 | 181 | 40 | 88 | 0.200 | 9.1 | 0.001 |
| 3 | 182 | 60 | 88 | 0.267 | 9.3 | 0.000 |
| 10 | 189 | 200 | 88 | 0.290 | 9.9 | 0.005 |
| 11 | 190 | 220 | 88 | 0.282 | 10.5 | 0.005 |
| 18 | 197 | 360 | 88 | 0.286 | 10.8 | 0.011 |
| 19 | 198 | 380 | 88 | 0.297 | 10.1 | 0.012 |
| 20 | 199 | 400 | 88 | 0.305 | 9.7 | 0.015 |
| … | | | | | | |
