# Router-side orphan probe (E1)

Can the *surviving* selector geometry identify an orphan query with no deletion record?
Deleted columns are removed before any feature is computed; the probe is fit on even-id authors and evaluated on odd-id ones, so every evaluated deleted source is one the probe never saw.

Drop set: `180..199` (20 sources) · seed 42

| strategy | k | probe AUC | FPR@90% catch | best confidence | lift | sentinel | shuffled | oracle ceiling |
|---|---|---|---|---|---|---|---|---|
| activation_norm | 200 | **0.661** | 0.750 | 0.561 | 0.100 | — | 0.548 | 0.599 |
| attn_norm | 200 | **0.622** | 0.798 | 0.502 | 0.120 | — | 0.513 | 0.599 |
| ppl | 200 | **0.782** | 0.462 | 0.783 | -0.001 | — | 0.528 | 0.456 |

## Deletion attribution — which source was removed

Sources of the eval half ranked by their queries' mean orphan probability. This is **score-access** attribution (the adversary reads the selector's score vector and has each candidate source's own questions) — not the black-box endpoint attack of §4.4, which needs generations.

| strategy | sources | deleted | top-1 is deleted | recall@n_deleted | source AUC |
|---|---|---|---|---|---|
| activation_norm | 91 | 10 | no | 0.300 | 0.738 |
| attn_norm | 91 | 10 | no | 0.000 | 0.657 |
| ppl | 91 | 10 | no | 0.200 | 0.865 |

## Verdict (pre-registered)

Best probe AUC **0.782** (ppl) → **subsection**.
Bars: ≥0.85 headline · ≥0.65 subsection.
Published reference points: confidence family 0.57–0.61 (no deletion record) · author-rung sentinel 0.982 (needs the record).

Lift over the best confidence detector on the same eval half: max **0.120** (attn_norm), median 0.100 → mechanism reads as **learned — the probe reads structure no single confidence statistic does**.

Absolute AUC alone cannot tell a learnable residual trace apart from the deleted source's own column having been removed. The lift is what separates them, so read the two columns together.
