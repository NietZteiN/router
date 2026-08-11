# Sequential deletion — magnet saturation and retained displacement

Sources are deleted ONE AT A TIME. `busiest share` = fraction of all accumulated orphans landing on the single most-hit survivor; `n_eff` = 1/HHI; **RDR** = fraction of RETAINED queries whose selected unit changed versus no deletion at all — displacement nobody asked for.

## `activation_norm`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.650 → 0.487 (Δ -0.163), final unit 169, tail stability 1.000, final n_eff 3.3, final RDR **0.077**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 70 | 0.650 | 2.1 | 0.000 |
| 2 | 181 | 40 | 70 | 0.325 | 5.5 | 0.000 |
| 3 | 182 | 60 | 44 | 0.267 | 5.6 | 0.000 |
| 10 | 189 | 200 | 169 | 0.335 | 4.6 | 0.003 |
| 11 | 190 | 220 | 169 | 0.395 | 4.1 | 0.003 |
| 18 | 197 | 360 | 169 | 0.439 | 3.9 | 0.005 |
| 19 | 198 | 380 | 169 | 0.471 | 3.4 | 0.074 |
| 20 | 199 | 400 | 169 | 0.487 | 3.3 | 0.077 |
| … | | | | | | |

## `attn_norm`

**saturating — one survivor takes a growing share** — busiest share 0.600 → 0.637 (Δ 0.037), final unit 44, tail stability 1.000, final n_eff 2.2, final RDR **0.025**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 44 | 0.600 | 2.2 | 0.000 |
| 2 | 181 | 40 | 44 | 0.550 | 2.9 | 0.000 |
| 3 | 182 | 60 | 44 | 0.667 | 2.1 | 0.000 |
| 10 | 189 | 200 | 44 | 0.620 | 2.3 | 0.015 |
| 11 | 190 | 220 | 44 | 0.591 | 2.6 | 0.016 |
| 18 | 197 | 360 | 44 | 0.597 | 2.5 | 0.020 |
| 19 | 198 | 380 | 44 | 0.618 | 2.3 | 0.024 |
| 20 | 199 | 400 | 44 | 0.637 | 2.2 | 0.025 |
| … | | | | | | |

## `ppl`

**dispersing — orphan mass spreads as more sources go** — busiest share 0.350 → 0.190 (Δ -0.160), final unit 102, tail stability 1.000, final n_eff 14.9, final RDR **0.045**.

| step | deleted | orphans | busiest unit | share | n_eff | RDR |
|---|---|---|---|---|---|---|
| 1 | 180 | 20 | 45 | 0.350 | 5.1 | 0.000 |
| 2 | 181 | 40 | 77 | 0.200 | 9.9 | 0.000 |
| 3 | 182 | 60 | 102 | 0.167 | 12.2 | 0.000 |
| 10 | 189 | 200 | 102 | 0.270 | 9.8 | 0.008 |
| 11 | 190 | 220 | 102 | 0.259 | 10.4 | 0.009 |
| 18 | 197 | 360 | 102 | 0.189 | 15.2 | 0.032 |
| 19 | 198 | 380 | 102 | 0.184 | 15.9 | 0.040 |
| 20 | 199 | 400 | 102 | 0.190 | 14.9 | 0.045 |
| … | | | | | | |
