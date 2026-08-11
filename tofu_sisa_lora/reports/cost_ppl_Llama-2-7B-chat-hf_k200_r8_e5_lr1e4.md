# Defense frontier — how cheap can a `ppl` orphan detector be? (H24 / §4.6)

`ppl` is the one selector whose orphan detection survives name-stripping, which makes it a candidate refusal gate. Scoring all survivors costs one forward pass per expert; here a free `key_tfidf` prefilter picks the top-m and `ppl` runs on those only. **No deletion record is consulted at any point**, and the prefilter ranks over surviving units only.

Survivors: 180 · full-scoring AUC **0.630**

| m (fwd/query) | speedup | probe AUC | best-conf AUC | FPR@90% catch | own-expert recall |
|---|---|---|---|---|---|
| 2 | 90x | **0.598** | 0.663 | 0.683 | 0.685 |
| 4 | 45x | **0.616** | 0.693 | 0.721 | 0.767 |
| 8 | 22x | **0.618** | 0.674 | 0.673 | 0.835 |
| 16 | 11x | **0.594** | 0.652 | 0.683 | 0.900 |
| 32 | 6x | **0.594** | 0.637 | 0.659 | 0.945 |
| 64 | 3x | **0.601** | 0.641 | 0.683 | 0.983 |
| 180 | 1x | **0.630** | 0.647 | 0.639 | 1.000 |

**Verdict:** a ppl refusal gate can be run at this many forward passes per query without measurably weaker orphan detection
 — m = **4** (45x cheaper than scoring every survivor).

`own-expert recall` is the fraction of RETAINED queries whose own expert survived the prefilter. It can sit well below 1.0 while the AUC holds, because detection reads the SHAPE of the score distribution rather than requiring correct routing.