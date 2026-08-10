# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**saturated — one survivor already holds most orphans** — busiest share 0.650 → 0.532 (Δ -0.118), final unit 136, tail stability 1.000, final n_eff 3.0, final RDR **0.022**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 136 | 0.650 | 2.2 | 0.000 |
| 2 | 181 | 40 | 136 | 0.500 | 3.3 | 0.000 |
| 3 | 182 | 60 | 136 | 0.517 | 3.2 | 0.000 |
| 10 | 189 | 200 | 136 | 0.475 | 3.5 | 0.002 |
| 11 | 190 | 220 | 136 | 0.477 | 3.5 | 0.002 |
| 18 | 197 | 360 | 136 | 0.542 | 2.9 | 0.027 |
| 19 | 198 | 380 | 136 | 0.534 | 3.0 | 0.029 |
| 20 | 199 | 400 | 136 | 0.532 | 3.0 | 0.022 |
| … | | | | | | |

## `attn_norm`

**saturating — one survivor takes a growing share** — busiest share 0.300 → 0.425 (Δ 0.125), final unit 135, tail stability 1.000, final n_eff 3.5, final RDR **0.145**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 136 | 0.300 | 4.2 | 0.000 |
| 2 | 181 | 40 | 135 | 0.350 | 4.3 | 0.000 |
| 3 | 182 | 60 | 135 | 0.333 | 4.3 | 0.000 |
| 10 | 189 | 200 | 135 | 0.370 | 4.3 | 0.023 |
| 11 | 190 | 220 | 135 | 0.377 | 4.3 | 0.024 |
| 18 | 197 | 360 | 135 | 0.422 | 3.5 | 0.157 |
| 19 | 198 | 380 | 135 | 0.424 | 3.5 | 0.155 |
| 20 | 199 | 400 | 135 | 0.425 | 3.5 | 0.145 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.650 → 0.253 (Δ -0.398), final unit 117, tail stability 1.000, final n_eff 6.8, final RDR **0.013**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 117 | 0.650 | 2.1 | 0.000 |
| 2 | 181 | 40 | 117 | 0.500 | 3.5 | 0.000 |
| 3 | 182 | 60 | 117 | 0.400 | 4.7 | 0.007 |
| 10 | 189 | 200 | 117 | 0.305 | 6.2 | 0.015 |
| 11 | 190 | 220 | 117 | 0.282 | 6.2 | 0.016 |
| 18 | 197 | 360 | 117 | 0.250 | 7.2 | 0.011 |
| 19 | 198 | 380 | 117 | 0.263 | 6.7 | 0.012 |
| 20 | 199 | 400 | 117 | 0.253 | 6.8 | 0.013 |
| … | | | | | | |
