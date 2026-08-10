# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**saturated — one survivor already holds most orphans** — busiest share 0.550 → 0.552 (Δ 0.002), final unit 169, tail stability 1.000, final n_eff 2.9, final RDR **0.117**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 169 | 0.550 | 2.2 | 0.000 |
| 2 | 181 | 40 | 169 | 0.575 | 2.5 | 0.000 |
| 3 | 182 | 60 | 169 | 0.583 | 2.7 | 0.000 |
| 10 | 189 | 200 | 169 | 0.505 | 3.4 | 0.003 |
| 11 | 190 | 220 | 169 | 0.514 | 3.3 | 0.003 |
| 18 | 197 | 360 | 169 | 0.525 | 3.2 | 0.002 |
| 19 | 198 | 380 | 169 | 0.555 | 2.9 | 0.117 |
| 20 | 199 | 400 | 169 | 0.552 | 2.9 | 0.117 |
| … | | | | | | |

## `attn_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.350 → 0.233 (Δ -0.117), final unit 44, tail stability 1.000, final n_eff 7.5, final RDR **0.177**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 169 | 0.350 | 4.4 | 0.000 |
| 2 | 181 | 40 | 169 | 0.250 | 6.8 | 0.000 |
| 3 | 182 | 60 | 169 | 0.200 | 7.8 | 0.000 |
| 10 | 189 | 200 | 44 | 0.245 | 7.6 | 0.065 |
| 11 | 190 | 220 | 44 | 0.232 | 8.0 | 0.062 |
| 18 | 197 | 360 | 44 | 0.219 | 8.1 | 0.068 |
| 19 | 198 | 380 | 44 | 0.237 | 7.4 | 0.176 |
| 20 | 199 | 400 | 44 | 0.233 | 7.5 | 0.177 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.150 → 0.087 (Δ -0.062), final unit 29, tail stability 1.000, final n_eff 27.9, final RDR **0.022**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 29 | 0.150 | 12.5 | 0.000 |
| 2 | 181 | 40 | 29 | 0.075 | 23.5 | 0.000 |
| 3 | 182 | 60 | 29 | 0.117 | 21.7 | 0.000 |
| 10 | 189 | 200 | 29 | 0.085 | 30.4 | 0.010 |
| 11 | 190 | 220 | 29 | 0.082 | 30.6 | 0.010 |
| 18 | 197 | 360 | 29 | 0.083 | 27.8 | 0.018 |
| 19 | 198 | 380 | 29 | 0.084 | 28.2 | 0.021 |
| 20 | 199 | 400 | 29 | 0.087 | 27.9 | 0.022 |
| … | | | | | | |
