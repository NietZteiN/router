# Granularity ladder — orphan detectability vs routing-unit size (7B, forget10 held fixed)

Deletion size is CONSTANT across the three cells: TOFU's forget10, the same 20 authors (400 queries).
Only the routing UNIT changes — 20 authors per unit at k=10, 4 at k=50, 1 at k=200.

`conf` = best confidence detector (global_top1 / margin / per_shard_z), the statistic the
literature reports as failing. `probe` = the learned router-side reader. `rec@n` = fraction of
the eval half's deleted sources recovered by score-access attribution.

| strategy | k=10 conf / probe / rec@n | k=50 conf / probe / rec@n | k=200 conf / probe / rec@n |
|---|---|---|---|
| `centroid_sbert` | **0.564** | 0.673 | 0.300 | **0.795** | 0.888 | 0.700 | **0.984** | 0.976 | 1.000 |
| `centroid_lm` | **0.502** | 0.636 | 0.200 | **0.628** | 0.742 | 0.400 | **0.761** | 0.830 | 0.500 |
| `key_tfidf` | **0.973** | 0.986 | 1.000 | **0.989** | 0.988 | 1.000 | **0.989** | 0.990 | 1.000 |

## Read

Confidence-based orphan detectability rises **monotonically** with granularity for both dense
routers, at constant deletion size. The published "confidence refusal caps at AUC 0.57-0.61"
is therefore a statement about COARSE units, not about selectors: at per-source granularity a
plain threshold already separates orphans from retained traffic.

`key_tfidf` is saturated at every k and carries no ladder information — lexical overlap with a
source's own training questions is strong at any unit size.

## The cell this replaces

`rl_family_leak_table.md`'s k=50 cells are `d49` and `d49_48` — 4 and 8 deleted authors. But
`is_forget` in that npz marks all 400 forget10 rows, so those cells label 16 (resp. 12) authors
as orphans while their own expert is still present. That is fine for the drop-set question the
cells were built to answer and wrong for a granularity comparison, which needs the same
deletion at every k. Dropping shards 45-49 gives the matched cell, and moves `centroid_sbert`
from 0.593 to 0.795 — the difference is the mislabelled majority, not granularity.
