# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.600 → 0.245 (Δ -0.355), final unit 161, tail stability 1.000, final n_eff 9.9, final RDR **0.007**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 176 | 0.600 | 2.4 | 0.000 |
| 2 | 181 | 40 | 161 | 0.550 | 2.5 | 0.000 |
| 3 | 182 | 60 | 161 | 0.367 | 3.7 | 0.000 |
| 10 | 189 | 200 | 161 | 0.290 | 7.7 | 0.002 |
| 11 | 190 | 220 | 161 | 0.282 | 8.0 | 0.002 |
| 18 | 197 | 360 | 161 | 0.256 | 9.5 | 0.007 |
| 19 | 198 | 380 | 161 | 0.242 | 9.9 | 0.007 |
| 20 | 199 | 400 | 161 | 0.245 | 9.9 | 0.007 |
| … | | | | | | |

## `attn_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.700 → 0.380 (Δ -0.320), final unit 161, tail stability 1.000, final n_eff 4.0, final RDR **0.028**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 71 | 0.700 | 2.0 | 0.050 |
| 2 | 181 | 40 | 161 | 0.500 | 2.6 | 0.051 |
| 3 | 182 | 60 | 161 | 0.483 | 3.1 | 0.054 |
| 10 | 189 | 200 | 161 | 0.400 | 3.7 | 0.038 |
| 11 | 190 | 220 | 161 | 0.395 | 3.7 | 0.040 |
| 18 | 197 | 360 | 161 | 0.397 | 3.8 | 0.025 |
| 19 | 198 | 380 | 161 | 0.384 | 4.0 | 0.026 |
| 20 | 199 | 400 | 161 | 0.380 | 4.0 | 0.028 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.200 → 0.070 (Δ -0.130), final unit 103, tail stability 0.200, final n_eff 43.6, final RDR **0.000**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.200 | 10.0 | 0.000 |
| 2 | 181 | 40 | 161 | 0.100 | 20.0 | 0.000 |
| 3 | 182 | 60 | 56 | 0.117 | 22.0 | 0.000 |
| 10 | 189 | 200 | 86 | 0.100 | 30.2 | 0.000 |
| 11 | 190 | 220 | 86 | 0.091 | 34.0 | 0.000 |
| 18 | 197 | 360 | 73 | 0.061 | 45.4 | 0.000 |
| 19 | 198 | 380 | 103 | 0.066 | 42.3 | 0.000 |
| 20 | 199 | 400 | 103 | 0.070 | 43.6 | 0.000 |
| … | | | | | | |
