# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.650 → 0.450 (Δ -0.200), final unit 169, tail stability 1.000, final n_eff 3.6, final RDR **0.077**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 70 | 0.650 | 2.1 | 0.000 |
| 2 | 181 | 40 | 70 | 0.325 | 5.5 | 0.000 |
| 3 | 182 | 60 | 44 | 0.267 | 5.6 | 0.000 |
| 10 | 189 | 200 | 169 | 0.335 | 4.6 | 0.003 |
| 11 | 190 | 220 | 169 | 0.395 | 4.1 | 0.003 |
| 18 | 197 | 360 | 169 | 0.403 | 4.3 | 0.005 |
| 19 | 198 | 380 | 169 | 0.432 | 3.8 | 0.074 |
| 20 | 199 | 400 | 169 | 0.450 | 3.6 | 0.077 |
| … | | | | | | |

## `attn_norm`

**saturated — one survivor already holds most orphans** — busiest share 0.600 → 0.615 (Δ 0.015), final unit 44, tail stability 1.000, final n_eff 2.3, final RDR **0.033**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 44 | 0.600 | 2.2 | 0.000 |
| 2 | 181 | 40 | 44 | 0.550 | 2.9 | 0.000 |
| 3 | 182 | 60 | 44 | 0.667 | 2.1 | 0.000 |
| 10 | 189 | 200 | 44 | 0.620 | 2.3 | 0.015 |
| 11 | 190 | 220 | 44 | 0.591 | 2.6 | 0.016 |
| 18 | 197 | 360 | 44 | 0.572 | 2.6 | 0.020 |
| 19 | 198 | 380 | 44 | 0.595 | 2.5 | 0.031 |
| 20 | 199 | 400 | 44 | 0.615 | 2.3 | 0.033 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.350 → 0.193 (Δ -0.157), final unit 102, tail stability 1.000, final n_eff 14.8, final RDR **0.045**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 45 | 0.350 | 5.1 | 0.000 |
| 2 | 181 | 40 | 77 | 0.200 | 9.9 | 0.000 |
| 3 | 182 | 60 | 102 | 0.167 | 12.2 | 0.000 |
| 10 | 189 | 200 | 102 | 0.270 | 9.8 | 0.010 |
| 11 | 190 | 220 | 102 | 0.264 | 10.1 | 0.010 |
| 18 | 197 | 360 | 102 | 0.192 | 15.0 | 0.032 |
| 19 | 198 | 380 | 102 | 0.187 | 15.7 | 0.040 |
| 20 | 199 | 400 | 102 | 0.193 | 14.8 | 0.045 |
| … | | | | | | |
