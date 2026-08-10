# Router-side orphan probe (E1)

Can the *surviving* selector geometry identify an orphan query with no deletion record?
Deleted columns are removed before any feature is computed; the probe is fit on even-id authors and evaluated on odd-id ones, so every evaluated deleted source is one the probe never saw.

Drop set: `180..199` (20 sources) · seed 42

| strategy | k | probe AUC | FPR@90% catch | best confidence | lift | sentinel | shuffled | oracle ceiling |
|---|---|---|---|---|---|---|---|---|
| activation_norm | 200 | **0.594** | 0.827 | 0.495 | 0.099 | — | 0.511 | 0.575 |
| attn_norm | 200 | **0.620** | 0.745 | 0.519 | 0.101 | — | 0.505 | 0.614 |
| ppl | 200 | **0.630** | 0.639 | 0.647 | -0.017 | — | 0.556 | 0.455 |

## Deletion attribution — which source was removed

Sources of the eval half ranked by their queries' mean orphan probability. This is **score-access** attribution (the adversary reads the selector's score vector and has each candidate source's own questions) — not the black-box endpoint attack of §4.4, which needs generations.

| strategy | sources | deleted | top-1 is deleted | recall@n_deleted | source AUC |
|---|---|---|---|---|---|
| activation_norm | 91 | 10 | no | 0.000 | 0.649 |
| attn_norm | 91 | 10 | no | 0.100 | 0.657 |
| ppl | 91 | 10 | no | 0.000 | 0.728 |

## Verdict (pre-registered)

Best probe AUC **0.630** (ppl) → **one paragraph**.
Bars: ≥0.85 headline · ≥0.65 subsection.
Published reference points: confidence family 0.57–0.61 (no deletion record) · author-rung sentinel 0.982 (needs the record).

Lift over the best confidence detector on the same eval half: max **0.101** (attn_norm), median 0.099 → mechanism reads as **learned — the probe reads structure no single confidence statistic does**.

Absolute AUC alone cannot tell a learnable residual trace apart from the deleted source's own column having been removed. The lift is what separates them, so read the two columns together.
