# Router-side orphan probe (E1)

Can the *surviving* selector geometry identify an orphan query with no deletion record?
Deleted columns are removed before any feature is computed; the probe is fit on even-id authors and evaluated on odd-id ones, so every evaluated deleted source is one the probe never saw.

Drop set: `180..199` (20 sources) · seed 42

| strategy | k | probe AUC | FPR@90% catch | best confidence | lift | sentinel | shuffled | oracle ceiling |
|---|---|---|---|---|---|---|---|---|
| activation_norm | 200 | **0.972** | 0.082 | 0.934 | 0.039 | — | 0.432 | 0.591 |
| attn_norm | 200 | **0.764** | 0.462 | 0.700 | 0.064 | — | 0.420 | 0.583 |
| ppl | 200 | **1.000** | 0.000 | 1.000 | 0.000 | — | 0.439 | 0.520 |

## Deletion attribution — which source was removed

Sources of the eval half ranked by their queries' mean orphan probability. This is **score-access** attribution (the adversary reads the selector's score vector and has each candidate source's own questions) — not the black-box endpoint attack of §4.4, which needs generations.

| strategy | sources | deleted | top-1 is deleted | recall@n_deleted | source AUC |
|---|---|---|---|---|---|
| activation_norm | 91 | 10 | yes | 0.800 | 0.988 |
| attn_norm | 91 | 10 | no | 0.200 | 0.801 |
| ppl | 91 | 10 | yes | 1.000 | 1.000 |

## Verdict (pre-registered)

Best probe AUC **1.000** (ppl) → **headline (§4.9)**.
Bars: ≥0.85 headline · ≥0.65 subsection.
Published reference points: confidence family 0.57–0.61 (no deletion record) · author-rung sentinel 0.982 (needs the record).

Lift over the best confidence detector on the same eval half: max **0.064** (attn_norm), median 0.039 → mechanism reads as **confidence — the probe adds little a threshold does not already give**.

Absolute AUC alone cannot tell a learnable residual trace apart from the deleted source's own column having been removed. The lift is what separates them, so read the two columns together.
