# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.450 → 0.370 (Δ -0.080), final unit 169, tail stability 1.000, final n_eff 5.2, final RDR **0.013**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 70 | 0.450 | 3.2 | 0.000 |
| 2 | 181 | 40 | 169 | 0.275 | 5.8 | 0.000 |
| 3 | 182 | 60 | 169 | 0.467 | 3.8 | 0.000 |
| 10 | 189 | 200 | 169 | 0.370 | 5.2 | 0.002 |
| 11 | 190 | 220 | 169 | 0.386 | 5.0 | 0.002 |
| 18 | 197 | 360 | 169 | 0.331 | 6.0 | 0.000 |
| 19 | 198 | 380 | 169 | 0.345 | 5.6 | 0.012 |
| 20 | 199 | 400 | 169 | 0.370 | 5.2 | 0.013 |
| … | | | | | | |

## `attn_norm`

**flat** — busiest share 0.400 → 0.417 (Δ 0.017), final unit 44, tail stability 1.000, final n_eff 4.7, final RDR **0.040**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 44 | 0.400 | 4.3 | 0.000 |
| 2 | 181 | 40 | 44 | 0.300 | 5.5 | 0.000 |
| 3 | 182 | 60 | 44 | 0.283 | 6.0 | 0.000 |
| 10 | 189 | 200 | 44 | 0.355 | 5.4 | 0.022 |
| 11 | 190 | 220 | 44 | 0.336 | 5.8 | 0.021 |
| 18 | 197 | 360 | 44 | 0.375 | 5.5 | 0.018 |
| 19 | 198 | 380 | 44 | 0.405 | 4.9 | 0.036 |
| 20 | 199 | 400 | 44 | 0.417 | 4.7 | 0.040 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.300 → 0.055 (Δ -0.245), final unit 29, tail stability 0.200, final n_eff 46.5, final RDR **0.000**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 113 | 0.300 | 7.4 | 0.000 |
| 2 | 181 | 40 | 113 | 0.175 | 16.0 | 0.000 |
| 3 | 182 | 60 | 113 | 0.117 | 21.4 | 0.000 |
| 10 | 189 | 200 | 131 | 0.095 | 27.3 | 0.000 |
| 11 | 190 | 220 | 131 | 0.086 | 30.6 | 0.000 |
| 18 | 197 | 360 | 131 | 0.056 | 49.0 | 0.000 |
| 19 | 198 | 380 | 29 | 0.058 | 46.4 | 0.000 |
| 20 | 199 | 400 | 29 | 0.055 | 46.5 | 0.000 |
| … | | | | | | |
