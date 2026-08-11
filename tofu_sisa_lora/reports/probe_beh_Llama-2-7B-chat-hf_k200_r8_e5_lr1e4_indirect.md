# Router-side orphan probe (E1)

Can the *surviving* selector geometry identify an orphan query with no deletion record?
Deleted columns are removed before any feature is computed; the probe is fit on even-id authors and evaluated on odd-id ones, so every evaluated deleted source is one the probe never saw.

Drop set: `180..199` (20 sources) · seed 42

| strategy | k | probe AUC | FPR@90% catch | best confidence | lift | sentinel | shuffled | oracle ceiling |
|---|---|---|---|---|---|---|---|---|
| activation_norm | 200 | **0.566** | 0.865 | 0.585 | -0.019 | — | 0.496 | 0.557 |
| attn_norm | 200 | **0.460** | 0.875 | 0.496 | -0.036 | — | 0.481 | 0.468 |
| ppl | 200 | **0.649** | 0.639 | 0.647 | 0.002 | — | 0.515 | 0.508 |

## Deletion attribution — which source was removed

Sources of the eval half ranked by their queries' mean orphan probability. This is **score-access** attribution (the adversary reads the selector's score vector and has each candidate source's own questions) — not the black-box endpoint attack of §4.4, which needs generations.

| strategy | sources | deleted | top-1 is deleted | recall@n_deleted | source AUC |
|---|---|---|---|---|---|
| activation_norm | 91 | 10 | no | 0.000 | 0.595 |
| attn_norm | 91 | 10 | no | 0.000 | 0.440 |
| ppl | 91 | 10 | no | 0.100 | 0.685 |

## Verdict (pre-registered)

Best probe AUC **0.649** (ppl) → **one paragraph**.
Bars: ≥0.85 headline · ≥0.65 subsection.
Published reference points: confidence family 0.57–0.61 (no deletion record) · author-rung sentinel 0.982 (needs the record).

Lift over the best confidence detector on the same eval half: max **0.002** (ppl), median -0.019 → mechanism reads as **confidence — the probe adds little a threshold does not already give**.

Absolute AUC alone cannot tell a learnable residual trace apart from the deleted source's own column having been removed. The lift is what separates them, so read the two columns together.
