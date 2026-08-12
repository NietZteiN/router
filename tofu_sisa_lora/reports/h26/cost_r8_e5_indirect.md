# Defense frontier — how cheap can a `ppl` orphan detector be? (H24 / §4.6)

`ppl` is the one selector whose orphan detection survives name-stripping, which makes it a candidate refusal gate. Scoring all survivors costs one forward pass per expert; here a free `key_tfidf` prefilter picks the top-m and `ppl` runs on those only. **No deletion record is consulted at any point**, and the prefilter ranks over surviving units only.

Survivors: 180 · full-scoring AUC **0.650**

| m (fwd/query) | speedup | probe AUC | best-conf AUC | FPR@90% catch | own-expert recall |
|---|---|---|---|---|---|
| 2 | 90x | **0.688** | 0.690 | 0.582 | 0.965 |
| 4 | 45x | **0.621** | 0.623 | 0.697 | 0.998 |
| 8 | 22x | **0.632** | 0.640 | 0.678 | 1.000 |
| 16 | 11x | **0.639** | 0.642 | 0.678 | 1.000 |
| 32 | 6x | **0.623** | 0.641 | 0.683 | 1.000 |
| 64 | 3x | **0.623** | 0.642 | 0.683 | 1.000 |
| 180 | 1x | **0.650** | 0.624 | 0.611 | 1.000 |

**Verdict:** a ppl refusal gate can be run at this many forward passes per query without measurably weaker orphan detection
 — m = **2** (90x cheaper than scoring every survivor).

`own-expert recall` is the fraction of RETAINED queries whose own expert survived the prefilter. It can sit well below 1.0 while the AUC holds, because detection reads the SHAPE of the score distribution rather than requiring correct routing.