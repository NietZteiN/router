# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.250 → 0.193 (Δ -0.057), final unit 136, tail stability 0.800, final n_eff 7.2, final RDR **0.037**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 122 | 0.250 | 5.3 | 0.000 |
| 2 | 181 | 40 | 122 | 0.175 | 7.5 | 0.000 |
| 3 | 182 | 60 | 122 | 0.167 | 8.4 | 0.000 |
| 10 | 189 | 200 | 135 | 0.165 | 7.9 | 0.008 |
| 11 | 190 | 220 | 135 | 0.173 | 7.7 | 0.009 |
| 18 | 197 | 360 | 136 | 0.200 | 7.2 | 0.041 |
| 19 | 198 | 380 | 136 | 0.189 | 7.1 | 0.043 |
| 20 | 199 | 400 | 136 | 0.193 | 7.2 | 0.037 |
| … | | | | | | |

## `attn_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.500 → 0.315 (Δ -0.185), final unit 122, tail stability 1.000, final n_eff 4.5, final RDR **0.125**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 122 | 0.500 | 2.9 | 0.000 |
| 2 | 181 | 40 | 122 | 0.350 | 4.1 | 0.000 |
| 3 | 182 | 60 | 100 | 0.283 | 4.8 | 0.000 |
| 10 | 189 | 200 | 122 | 0.315 | 4.8 | 0.038 |
| 11 | 190 | 220 | 122 | 0.300 | 4.9 | 0.038 |
| 18 | 197 | 360 | 122 | 0.333 | 4.2 | 0.132 |
| 19 | 198 | 380 | 122 | 0.318 | 4.5 | 0.138 |
| 20 | 199 | 400 | 122 | 0.315 | 4.5 | 0.125 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.450 → 0.265 (Δ -0.185), final unit 117, tail stability 1.000, final n_eff 7.3, final RDR **0.000**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 117 | 0.450 | 3.6 | 0.000 |
| 2 | 181 | 40 | 117 | 0.425 | 4.4 | 0.000 |
| 3 | 182 | 60 | 117 | 0.417 | 4.7 | 0.000 |
| 10 | 189 | 200 | 117 | 0.295 | 6.7 | 0.000 |
| 11 | 190 | 220 | 117 | 0.268 | 6.9 | 0.000 |
| 18 | 197 | 360 | 117 | 0.281 | 7.3 | 0.000 |
| 19 | 198 | 380 | 117 | 0.271 | 7.1 | 0.000 |
| 20 | 199 | 400 | 117 | 0.265 | 7.3 | 0.000 |
| … | | | | | | |
