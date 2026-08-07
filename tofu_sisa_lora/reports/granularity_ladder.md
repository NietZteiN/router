# Granularity ladder — orphan detectability vs routing-unit size

Deletion is held CONSTANT across rungs; only the routing UNIT changes. Each rung names its own drop set for that reason — the same 20 deleted authors are one unit-group at k=10, five at k=50, and twenty at k=200.

`conf` = best confidence detector (global_top1 / margin / per_shard_z), the statistic the literature reports as failing. `probe` = the learned router-side reader. `rec@n` = fraction of the eval half's deleted sources recovered by score-access attribution.

| rung | units (k) | deleted units | queries |
|---|---|---|---|
| `k=10` | 10 | 1 | 408 eval rows |
| `k=50` | 50 | 5 | 2000 eval rows |
| `k=200` | 200 | 20 | 2000 eval rows |

| strategy | k=10 conf | k=10 probe | k=10 rec@n | k=50 conf | k=50 probe | k=50 rec@n | k=200 conf | k=200 probe | k=200 rec@n | monotone? |
|---|---|---|---|---|---|---|---|---|---|---|
| `activation_norm` | **0.495** | 0.558 | 0.200 | — | — | — | — | — | — | insufficient rungs |
| `attn_norm` | **0.432** | 0.605 | 0.000 | — | — | — | — | — | — | insufficient rungs |
| `centroid_lm` | **0.502** | 0.636 | 0.200 | **0.628** | 0.742 | 0.400 | **0.761** | 0.830 | 0.500 | monotone increasing (Δ 0.259) |
| `centroid_lm_last` | **0.533** | 0.613 | 0.100 | — | — | — | — | — | — | insufficient rungs |
| `centroid_sbert` | **0.564** | 0.673 | 0.300 | **0.795** | 0.888 | 0.700 | **0.984** | 0.976 | 1.000 | monotone increasing (Δ 0.420) |
| `centroid_sbert_q` | **0.606** | 0.690 | 0.300 | — | — | — | — | — | — | insufficient rungs |
| `key_tfidf` | **0.973** | 0.986 | 1.000 | **0.989** | 0.988 | 1.000 | **0.989** | 0.990 | 1.000 | saturated (Δ 0.016) |
| `logit_div` | **0.749** | 0.823 | 0.500 | — | — | — | — | — | — | insufficient rungs |
| `ppl` | **1.000** | 1.000 | 1.000 | — | — | — | — | — | — | insufficient rungs |

## Read

Confidence-based orphan detectability rises monotonically with granularity for `centroid_lm`, `centroid_sbert`. The published "confidence refusal caps at AUC 0.57–0.61" is therefore a statement about COARSE units, not about selectors: at per-source granularity a plain threshold already separates orphans from retained traffic.

`key_tfidf` is saturated at every rung and carries no ladder information.
