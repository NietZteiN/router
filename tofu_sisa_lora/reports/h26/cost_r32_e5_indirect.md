# Defense frontier — how cheap can a `ppl` orphan detector be? (H24 / §4.6)

`ppl` is the one selector whose orphan detection survives name-stripping, which makes it a candidate refusal gate. Scoring all survivors costs one forward pass per expert; here a free `key_tfidf` prefilter picks the top-m and `ppl` runs on those only. **No deletion record is consulted at any point**, and the prefilter ranks over surviving units only.

Survivors: 180 · full-scoring AUC **0.894**

| m (fwd/query) | speedup | probe AUC | best-conf AUC | FPR@90% catch | own-expert recall |
|---|---|---|---|---|---|
| 2 | 90x | **0.894** | 0.899 | 0.212 | 0.965 |
| 4 | 45x | **0.910** | 0.901 | 0.216 | 0.998 |
| 8 | 22x | **0.915** | 0.899 | 0.212 | 1.000 |
| 16 | 11x | **0.909** | 0.897 | 0.226 | 1.000 |
| 32 | 6x | **0.910** | 0.894 | 0.226 | 1.000 |
| 64 | 3x | **0.905** | 0.893 | 0.226 | 1.000 |
| 180 | 1x | **0.894** | 0.885 | 0.212 | 1.000 |

**Verdict:** a ppl refusal gate can be run at this many forward passes per query without measurably weaker orphan detection
 — m = **2** (90x cheaper than scoring every survivor).

`own-expert recall` is the fraction of RETAINED queries whose own expert survived the prefilter. It can sit well below 1.0 while the AUC holds, because detection reads the SHAPE of the score distribution rather than requiring correct routing.