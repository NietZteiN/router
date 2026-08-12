# H29 — confidence intervals on forget_quality

Bootstrap over the FORGET sample only (20000 resamples, seed 42); the KS reference is held fixed at the published `retain_tr_scores.npy` (20 rows).

| arm | n | fq | 95% CI | width | published | reproduces | on grid |
|---|---|---|---|---|---|---|---|
| `routed_reroute_f10_s89` | 120 | 0.8958 | [0.3177, 0.9606] | 0.6429 | 0.8958 | yes | yes |
| `routed_reroute_f10_s137` | 120 | 0.8002 | [0.2086, 0.9326] | 0.7240 | 0.8002 | yes | yes |
| `routed_reroute_f10_s31` | 120 | 0.6288 | [0.1535, 0.8511] | 0.6975 | 0.6288 | yes | yes |
| `routed_reroute_f10_s97` | 120 | 0.6288 | [0.2086, 0.8958] | 0.6871 | 0.6288 | yes | yes |
| `routed_oracle_del_f10` | 120 | 0.5140 | [0.1106, 0.7450] | 0.6344 | 0.5140 | yes | yes |
| `routed_reroute_f10_s33` | 120 | 0.5140 | [0.1307, 0.8511] | 0.7204 | 0.5140 | yes | yes |
| `routed_reroute_f10_s79` | 120 | 0.5140 | [0.1307, 0.8511] | 0.7204 | 0.5140 | yes | yes |
| `routed_reroute_f10_s88` | 120 | 0.3615 | [0.0651, 0.7450] | 0.6799 | 0.3615 | yes | yes |

Resolution at n=120 vs m=20: the KS statistic D moves on an exact lattice of step **0.008333** (= 1/lcm), i.e. one forget question. In p-value terms, **31 values above 0.05** with median gap **0.0309** — so the 4 decimals every table reports are spurious. (A sampled ≥86 distinct p-values overall; that count is a LOWER BOUND which grows with the number of draws, and is not quotable as an enumeration.) Resolution is a property of the sample sizes; the CI is a property of the data. They are not the same claim.

Spread across arms: **0.5342**. Widest MARGINAL CI: **0.7240**.

## Paired bootstrap (the arms score identical rows)

Each resample draws ONE set of row indices and applies it to every arm, so the question-level noise the arms share cancels in a difference. The marginal CIs above are wide because they re-add that shared noise once per arm; they bound a single published cell, and are the wrong yardstick for a spread.

Spread: **0.5342**, paired 95% CI [0.2245, 0.6975] (median 0.4753); P(spread>0.10) = **0.9996**, P(spread>0.25) = **0.9610**.

### Against genuine deletion (`routed_oracle_del_f10`)

| arm | Δ vs deletion | paired 95% CI | P(arm ≥ deletion) |
|---|---|---|---|
| `routed_reroute_f10_s89` | +0.3818 | [+0.0000, +0.5915] | 0.9762 |
| `routed_reroute_f10_s137` | +0.2862 | [-0.1148, +0.5342] | 0.9286 |
| `routed_reroute_f10_s31` | +0.1148 | [-0.2223, +0.3875] | 0.7968 |
| `routed_reroute_f10_s97` | +0.1148 | [-0.1162, +0.4358] | 0.8927 |
| `routed_reroute_f10_s33` | +0.0000 | [-0.2080, +0.3360] | 0.7204 |
| `routed_reroute_f10_s79` | +0.0000 | [-0.2186, +0.3360] | 0.7025 |
| `routed_reroute_f10_s88` | -0.1525 | [-0.3512, +0.2186] | 0.3325 |

Arms at or above genuine deletion: observed **6/7**, paired median 6, 95% CI [2, 7]. Every one of these arms deletes nothing.

**Verdict:** destination spread is resolvable: paired 95% CI [0.2245, 0.6975], P(spread>0.25)=0.961

Reproduced published cells: **8/8**.
