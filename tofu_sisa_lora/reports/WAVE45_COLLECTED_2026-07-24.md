# router_leak Waves 4+5 — collected numbers

Assembled by `collect_wave45.py` (CPU SLURM job, no inference). rho is recomputed against the SHARED router-independent `expert_max` ceiling.

## Mode-B rho per router family (verbatim surface, R=8)

| family | rho@R8 (shared ceiling) |
|---|---|
| centroid_sbert | 0.882 |
| centroid_lm | 0.8127 |
| ppl | 0.9373 |
| activation_norm | 0.9014 |
| attn_norm | 0.9045 |
| logit_div | 0.0157 |

## ppl-native seal (H-SEAL-PPL)

- sealed rho@R8 verbatim: **0.0**
- abstain rate (probes diverted to base): **0.84**
- bar: CONFIRM (seal misses replicated facts) if rho >= 0.5; REFUTE if it collapses toward the author-tombstone's 0.047.

## Deletion-disclosure AUC per rung (H-DISC-RUNG)

| rung | AUC | prior catch |
|---|---|---|
| shard | 0.8388125 | 0.605 |
| author | 0.98706875 | 0.963 |
| name | 0.967175 | 0.703 |

## Composed MIA on the embed-routed arms (H-MIA-ROUTER)

| arm | loss | min_k | min_k++ | zlib |
|---|---|---|---|---|
| sibling | 0.182 | 0.14360625 | 0.0724875 | 0.14051875 |
| tombstone | 0.2767875 | 0.259453125 | 0.28428749999999997 | 0.22000625000000001 |

Reference points: oracle floor 0.379; exact module-drop routerkey 0.375; ramole-embed 0.353 (leaked yet MIA-blind).

