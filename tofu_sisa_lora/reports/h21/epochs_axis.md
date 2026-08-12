# H21 — the epochs axis at fixed rank (k=200, forget10 deleted)

Best-confidence detection AUC. Same code path for every cell, so the e5/e25/r8 columns reproduce the published H18/H20 numbers and e50 is directly comparable.

## gold

| strategy | r32/e5 | r32/e25 | **r32/e50** | r8/e5 |
|---|---|---|---|---|
| `ppl` | 1.000 | 0.999 | 0.996 | 0.993 |
| `activation_norm` | 0.934 | 0.608 | 0.515 | 0.877 |
| `attn_norm` | 0.700 | 0.554 | 0.569 | 0.758 |

## name_stripped

| strategy | r32/e5 | r32/e25 | **r32/e50** | r8/e5 |
|---|---|---|---|---|
| `ppl` | 0.783 | 0.769 | 0.737 | 0.647 |
| `activation_norm` | 0.561 | 0.498 | 0.460 | 0.495 |
| `attn_norm` | 0.502 | 0.507 | 0.534 | 0.519 |

## indirect

| strategy | r32/e5 | r32/e25 | **r32/e50** | r8/e5 |
|---|---|---|---|---|
| `ppl` | 0.885 | 0.810 | 0.816 | 0.624 |
| `activation_norm` | 0.556 | 0.581 | 0.575 | 0.585 |
| `attn_norm` | 0.590 | 0.454 | 0.498 | 0.481 |

