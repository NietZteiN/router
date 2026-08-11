# Defense frontier — how cheap can a `ppl` orphan detector be? (H24 / §4.6)

`ppl` is the one selector whose orphan detection survives name-stripping, which makes it a candidate refusal gate. Scoring all survivors costs one forward pass per expert; here a free `key_tfidf` prefilter picks the top-m and `ppl` runs on those only. **No deletion record is consulted at any point**, and the prefilter ranks over surviving units only.

Survivors: 180 · full-scoring AUC **0.799**

| m (fwd/query) | speedup | probe AUC | best-conf AUC | FPR@90% catch | own-expert recall |
|---|---|---|---|---|---|
| 2 | 90x | **0.736** | 0.741 | 0.438 | 0.685 |
| 4 | 45x | **0.806** | 0.748 | 0.438 | 0.767 |
| 8 | 22x | **0.800** | 0.740 | 0.481 | 0.835 |
| 16 | 11x | **0.822** | 0.758 | 0.428 | 0.900 |
| 32 | 6x | **0.804** | 0.754 | 0.418 | 0.945 |
| 64 | 3x | **0.818** | 0.785 | 0.428 | 0.983 |
| 180 | 1x | **0.799** | 0.769 | 0.524 | 1.000 |

**Verdict:** a ppl refusal gate can be run at this many forward passes per query without measurably weaker orphan detection
 — m = **4** (45x cheaper than scoring every survivor).

`own-expert recall` is the fraction of RETAINED queries whose own expert survived the prefilter. It can sit well below 1.0 while the AUC holds, because detection reads the SHAPE of the score distribution rather than requiring correct routing.