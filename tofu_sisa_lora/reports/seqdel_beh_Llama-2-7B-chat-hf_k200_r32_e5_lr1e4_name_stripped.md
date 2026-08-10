# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**saturating — one survivor takes a growing share** — busiest share 0.200 → 0.318 (Δ 0.117), final unit 161, tail stability 1.000, final n_eff 6.0, final RDR **0.043**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 161 | 0.200 | 7.4 | 0.004 |
| 2 | 181 | 40 | 161 | 0.300 | 5.5 | 0.004 |
| 3 | 182 | 60 | 161 | 0.283 | 6.0 | 0.004 |
| 10 | 189 | 200 | 161 | 0.275 | 6.9 | 0.005 |
| 11 | 190 | 220 | 161 | 0.300 | 6.5 | 0.005 |
| 18 | 197 | 360 | 161 | 0.319 | 6.0 | 0.027 |
| 19 | 198 | 380 | 161 | 0.316 | 6.2 | 0.026 |
| 20 | 199 | 400 | 161 | 0.318 | 6.0 | 0.043 |
| … | | | | | | |

## `attn_norm`

**saturating — one survivor takes a growing share** — busiest share 0.350 → 0.453 (Δ 0.103), final unit 161, tail stability 1.000, final n_eff 3.4, final RDR **0.052**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 135 | 0.350 | 4.0 | 0.078 |
| 2 | 181 | 40 | 161 | 0.400 | 3.7 | 0.080 |
| 3 | 182 | 60 | 161 | 0.450 | 3.4 | 0.084 |
| 10 | 189 | 200 | 161 | 0.410 | 3.6 | 0.065 |
| 11 | 190 | 220 | 161 | 0.418 | 3.6 | 0.062 |
| 18 | 197 | 360 | 161 | 0.456 | 3.3 | 0.059 |
| 19 | 198 | 380 | 161 | 0.453 | 3.4 | 0.057 |
| 20 | 199 | 400 | 161 | 0.453 | 3.4 | 0.052 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.250 → 0.100 (Δ -0.150), final unit 88, tail stability 1.000, final n_eff 38.7, final RDR **0.015**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 186 | 0.250 | 9.1 | 0.001 |
| 2 | 181 | 40 | 186 | 0.125 | 22.2 | 0.003 |
| 3 | 182 | 60 | 186 | 0.083 | 28.6 | 0.001 |
| 10 | 189 | 200 | 88 | 0.090 | 42.6 | 0.010 |
| 11 | 190 | 220 | 88 | 0.091 | 41.4 | 0.010 |
| 18 | 197 | 360 | 88 | 0.106 | 37.1 | 0.020 |
| 19 | 198 | 380 | 88 | 0.103 | 38.5 | 0.021 |
| 20 | 199 | 400 | 88 | 0.100 | 38.7 | 0.015 |
| … | | | | | | |
