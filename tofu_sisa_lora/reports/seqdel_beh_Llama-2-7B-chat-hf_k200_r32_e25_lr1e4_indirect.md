# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**saturated — one survivor already holds most orphans** — busiest share 0.700 → 0.635 (Δ -0.065), final unit 135, tail stability 1.000, final n_eff 2.3, final RDR **0.120**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 135 | 0.700 | 1.8 | 0.000 |
| 2 | 181 | 40 | 135 | 0.800 | 1.5 | 0.000 |
| 3 | 182 | 60 | 135 | 0.817 | 1.5 | 0.000 |
| 10 | 189 | 200 | 135 | 0.600 | 2.3 | 0.002 |
| 11 | 190 | 220 | 135 | 0.573 | 2.5 | 0.002 |
| 18 | 197 | 360 | 135 | 0.650 | 2.2 | 0.143 |
| 19 | 198 | 380 | 135 | 0.663 | 2.1 | 0.150 |
| 20 | 199 | 400 | 135 | 0.635 | 2.3 | 0.120 |
| … | | | | | | |

## `attn_norm`

**saturating — one survivor takes a growing share** — busiest share 0.550 → 0.800 (Δ 0.250), final unit 135, tail stability 1.000, final n_eff 1.5, final RDR **0.055**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 185 | 0.550 | 2.5 | 0.000 |
| 2 | 181 | 40 | 135 | 0.650 | 2.0 | 0.000 |
| 3 | 182 | 60 | 135 | 0.700 | 1.9 | 0.000 |
| 10 | 189 | 200 | 135 | 0.795 | 1.5 | 0.043 |
| 11 | 190 | 220 | 135 | 0.809 | 1.5 | 0.034 |
| 18 | 197 | 360 | 135 | 0.806 | 1.5 | 0.089 |
| 19 | 198 | 380 | 135 | 0.813 | 1.5 | 0.093 |
| 20 | 199 | 400 | 135 | 0.800 | 1.5 | 0.055 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.350 → 0.268 (Δ -0.082), final unit 12, tail stability 1.000, final n_eff 7.2, final RDR **0.010**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 117 | 0.350 | 3.6 | 0.000 |
| 2 | 181 | 40 | 117 | 0.475 | 3.1 | 0.000 |
| 3 | 182 | 60 | 117 | 0.317 | 6.0 | 0.000 |
| 10 | 189 | 200 | 12 | 0.340 | 5.3 | 0.000 |
| 11 | 190 | 220 | 12 | 0.309 | 5.7 | 0.000 |
| 18 | 197 | 360 | 12 | 0.272 | 7.1 | 0.000 |
| 19 | 198 | 380 | 12 | 0.276 | 6.9 | 0.010 |
| 20 | 199 | 400 | 12 | 0.268 | 7.2 | 0.010 |
| … | | | | | | |
