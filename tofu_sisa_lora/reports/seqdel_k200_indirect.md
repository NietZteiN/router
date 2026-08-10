# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `centroid_sbert`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.550 → 0.113 (Δ -0.438), final unit 128, tail stability 0.700, final n_eff 23.0, final RDR **0.020**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.550 | 2.9 | 0.000 |
| 2 | 181 | 40 | 186 | 0.275 | 7.7 | 0.000 |
| 3 | 182 | 60 | 128 | 0.283 | 7.1 | 0.000 |
| 10 | 189 | 200 | 128 | 0.145 | 16.1 | 0.035 |
| 11 | 190 | 220 | 128 | 0.132 | 17.0 | 0.036 |
| 18 | 197 | 360 | 144 | 0.086 | 24.3 | 0.018 |
| 19 | 198 | 380 | 144 | 0.084 | 25.7 | 0.019 |
| 20 | 199 | 400 | 128 | 0.113 | 23.0 | 0.020 |
| … | | | | | | |

## `key_tfidf`

**saturating — one survivor takes a growing share** — busiest share 0.850 → 0.902 (Δ 0.052), final unit 88, tail stability 1.000, final n_eff 1.2, final RDR **0.000**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.850 | 1.3 | 0.000 |
| 2 | 181 | 40 | 88 | 0.550 | 2.1 | 0.000 |
| 3 | 182 | 60 | 88 | 0.700 | 1.8 | 0.000 |
| 10 | 189 | 200 | 88 | 0.900 | 1.2 | 0.000 |
| 11 | 190 | 220 | 88 | 0.873 | 1.3 | 0.000 |
| 18 | 197 | 360 | 88 | 0.903 | 1.2 | 0.000 |
| 19 | 198 | 380 | 88 | 0.908 | 1.2 | 0.000 |
| 20 | 199 | 400 | 88 | 0.902 | 1.2 | 0.000 |
| … | | | | | | |
