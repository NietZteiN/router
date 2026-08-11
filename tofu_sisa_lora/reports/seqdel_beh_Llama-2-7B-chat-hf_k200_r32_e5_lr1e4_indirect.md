# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**saturated — one survivor already holds most orphans** — busiest share 0.600 → 0.540 (Δ -0.060), final unit 132, tail stability 1.000, final n_eff 2.2, final RDR **0.005**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 132 | 0.600 | 2.2 | 0.001 |
| 2 | 181 | 40 | 135 | 0.475 | 2.4 | 0.001 |
| 3 | 182 | 60 | 135 | 0.483 | 2.4 | 0.001 |
| 10 | 189 | 200 | 132 | 0.595 | 2.1 | 0.002 |
| 11 | 190 | 220 | 132 | 0.614 | 2.0 | 0.002 |
| 18 | 197 | 360 | 132 | 0.578 | 2.1 | 0.002 |
| 19 | 198 | 380 | 132 | 0.558 | 2.2 | 0.002 |
| 20 | 199 | 400 | 132 | 0.540 | 2.2 | 0.005 |
| … | | | | | | |

## `attn_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.650 → 0.477 (Δ -0.173), final unit 135, tail stability 1.000, final n_eff 3.1, final RDR **0.003**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 135 | 0.650 | 2.1 | 0.014 |
| 2 | 181 | 40 | 135 | 0.450 | 2.4 | 0.014 |
| 3 | 182 | 60 | 135 | 0.550 | 2.3 | 0.018 |
| 10 | 189 | 200 | 135 | 0.475 | 2.9 | 0.007 |
| 11 | 190 | 220 | 135 | 0.450 | 3.1 | 0.007 |
| 18 | 197 | 360 | 135 | 0.442 | 3.2 | 0.002 |
| 19 | 198 | 380 | 135 | 0.461 | 3.1 | 0.002 |
| 20 | 199 | 400 | 135 | 0.477 | 3.1 | 0.003 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.150 → 0.095 (Δ -0.055), final unit 169, tail stability 1.000, final n_eff 34.9, final RDR **0.013**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 168 | 0.150 | 12.5 | 0.000 |
| 2 | 181 | 40 | 62 | 0.300 | 8.3 | 0.000 |
| 3 | 182 | 60 | 62 | 0.200 | 14.4 | 0.000 |
| 10 | 189 | 200 | 169 | 0.100 | 29.3 | 0.005 |
| 11 | 190 | 220 | 169 | 0.095 | 28.4 | 0.005 |
| 18 | 197 | 360 | 169 | 0.103 | 32.1 | 0.009 |
| 19 | 198 | 380 | 169 | 0.100 | 33.5 | 0.010 |
| 20 | 199 | 400 | 169 | 0.095 | 34.9 | 0.013 |
| … | | | | | | |
