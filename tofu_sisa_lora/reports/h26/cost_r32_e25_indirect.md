# Defense frontier — how cheap can a `ppl` orphan detector be? (H24 / §4.6)

`ppl` is the one selector whose orphan detection survives name-stripping, which makes it a candidate refusal gate. Scoring all survivors costs one forward pass per expert; here a free `key_tfidf` prefilter picks the top-m and `ppl` runs on those only. **No deletion record is consulted at any point**, and the prefilter ranks over surviving units only.

Survivors: 180 · full-scoring AUC **0.854**

| m (fwd/query) | speedup | probe AUC | best-conf AUC | FPR@90% catch | own-expert recall |
|---|---|---|---|---|---|
| 2 | 90x | **0.955** | 0.952 | 0.125 | 0.965 |
| 4 | 45x | **0.934** | 0.896 | 0.183 | 0.998 |
| 8 | 22x | **0.900** | 0.849 | 0.216 | 1.000 |
| 16 | 11x | **0.881** | 0.895 | 0.269 | 1.000 |
| 32 | 6x | **0.878** | 0.892 | 0.269 | 1.000 |
| 64 | 3x | **0.862** | 0.878 | 0.293 | 1.000 |
| 180 | 1x | **0.854** | 0.810 | 0.274 | 1.000 |

**Verdict:** a ppl refusal gate can be run at this many forward passes per query without measurably weaker orphan detection
 — m = **2** (90x cheaper than scoring every survivor).

`own-expert recall` is the fraction of RETAINED queries whose own expert survived the prefilter. It can sit well below 1.0 while the AUC holds, because detection reads the SHAPE of the score distribution rather than requiring correct routing.