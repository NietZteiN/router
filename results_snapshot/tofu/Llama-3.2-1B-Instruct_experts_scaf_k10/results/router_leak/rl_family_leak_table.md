# All-router leak table (router_leak Phase 3)

Seed 42; bootstrap 1000 author-blocked draws; conventions: author-parity even=calib/odd=eval; detector directions a priori (confidence NEGATED, tomb as-is); author-blocked bootstrap CIs; abstain taus retain-percentile only.

## Unified all-router leak table

| router | cell | orphan top-3 capture | adequacy [95% CI] | retain shift [95% CI] | best conf. AUC (det.) | FPR@90% catch | tomb catch/FPR | H-ARCH |
|---|---|---|---|---|---|---|---|---|
| oracle q2author (control) | any | 0.000 (orphans→base P=1.0) | — | 0.000 (by construction) | — | — | — | a✗ b· c· |
| activation_norm (k=10) | d9 | 0.997 [0.993, 1.000] | 0.997 [0.996, 0.998] | 0.3175 [0.2616, 0.3734] | 0.412 (margin) | 0.957 | — | a✓ b✓ c✓ |
| activation_norm (k=10) | d9_8 | 1.000 [1.000, 1.000] | 0.997 [0.996, 0.997] | 0.2981 [0.2436, 0.3585] | 0.427 (margin) | 0.945 | — | a✓ b✓ c✓ |
| activation_norm (k=10) | d9_8_7_6 | 0.919 [0.886, 0.954] | 0.959 [0.957, 0.960] | 0.9964 [0.9878, 1.0000] | 0.416 (margin) | 0.937 | — | a✓ b✓ c✓ |
| attn_norm (k=10) | d9 | 0.815 [0.760, 0.863] | 1.000 [1.000, 1.000] | 0.0000 [0.0000, 0.0000] | 0.533 (margin) | 0.894 | — | a✓ b✓ c✓ |
| attn_norm (k=10) | d9_8 | 0.821 [0.769, 0.869] | 1.000 [1.000, 1.000] | 0.0000 [0.0000, 0.0000] | 0.551 (margin) | 0.885 | — | a✓ b✓ c✓ |
| attn_norm (k=10) | d9_8_7_6 | 0.862 [0.827, 0.895] | 0.999 [0.999, 1.000] | 0.0714 [0.0443, 0.1012] | 0.544 (margin) | 0.853 | — | a✓ b✓ c✓ |
| centroid_lm (k=10) | d9 | 0.920 [0.863, 0.975] | 0.999 [0.998, 0.999] | 0.0758 [0.0553, 0.0970] | 0.474 (margin) | 0.951 | 0.980/0.974 | a✓ b✓ c✓ |
| centroid_lm (k=10) | d9_8 | 0.898 [0.868, 0.939] | 0.999 [0.998, 0.999] | 0.1222 [0.0947, 0.1512] | 0.481 (margin) | 0.909 | 0.988/0.985 | a✓ b✓ c✓ |
| centroid_lm (k=10) | d9_8_7_6 | 0.917 [0.890, 0.941] | 0.999 [0.999, 0.999] | 0.1908 [0.1592, 0.2254] | 0.495 (margin) | 0.914 | 0.991/0.992 | a✓ b✓ c✓ |
| centroid_lm (k=200) | d199 | 0.900 [0.900, 0.900] | 0.976 [0.976, 0.976] | 0.0000 [0.0000, 0.0000] | 0.728 (per_shard_z) | 0.880 | 0.950/0.338 | a✓ b✓ c✓ |
| centroid_lm (k=200) | d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 0.310 [0.273, 0.458] | 0.962 [0.950, 0.972] | 0.0042 [0.0017, 0.0078] | 0.761 (margin) | 0.498 | 0.950/0.720 | a✗ b✓ c✗ |
| centroid_lm_last (k=10) | d9 | 0.868 [0.833, 0.965] | 0.998 [0.997, 0.999] | 0.0147 [0.0094, 0.0208] | 0.505 (global_top1) | 0.890 | 1.000/0.997 | a✓ b✓ c✓ |
| centroid_lm_last (k=10) | d9_8 | 0.890 [0.852, 0.948] | 0.998 [0.997, 0.999] | 0.0281 [0.0153, 0.0441] | 0.486 (global_top1) | 0.904 | 1.000/1.000 | a✓ b✓ c✓ |
| centroid_lm_last (k=10) | d9_8_7_6 | 0.909 [0.877, 0.939] | 0.999 [0.998, 0.999] | 0.0288 [0.0133, 0.0517] | 0.503 (global_top1) | 0.892 | 1.000/1.000 | a✓ b✓ c✓ |
| centroid_sbert (k=10) | d9 | 0.542 [0.482, 0.688] | 0.967 [0.952, 0.981] | 0.0553 [0.0361, 0.0781] | 0.564 (per_shard_z) | 0.880 | 0.975/0.390 | a✓ b✓ c✓ |
| centroid_sbert (k=10) | d9_8 | 0.540 [0.489, 0.668] | 0.969 [0.958, 0.978] | 0.0941 [0.0641, 0.1259] | 0.575 (margin) | 0.774 | 0.990/0.604 | a✓ b✓ c✓ |
| centroid_sbert (k=10) | d9_8_7_6 | 0.599 [0.562, 0.677] | 0.964 [0.957, 0.971] | 0.1792 [0.1396, 0.2246] | 0.600 (margin) | 0.745 | 0.991/0.755 | a✓ b✓ c✓ |
| centroid_sbert (k=200) | d199 | 0.900 [0.900, 0.900] | 0.705 [0.705, 0.705] | 0.0000 [0.0000, 0.0000] | 0.982 (per_shard_z) | 0.038 | 1.000/0.000 | a✓ b✗ c✗ |
| centroid_sbert (k=200) | d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 0.258 [0.237, 0.445] | 0.673 [0.652, 0.697] | 0.0008 [0.0000, 0.0022] | 0.984 (per_shard_z) | 0.034 | 0.960/0.004 | a✗ b✗ c✗ |
| centroid_sbert_q (k=10) | d9 | 0.480 [0.467, 0.650] | 0.971 [0.957, 0.983] | 0.0583 [0.0358, 0.0833] | 0.606 (per_shard_z) | 0.841 | 0.950/0.115 | a✗ b✓ c✓ |
| centroid_sbert_q (k=10) | d9_8 | 0.550 [0.496, 0.653] | 0.971 [0.961, 0.979] | 0.1091 [0.0734, 0.1460] | 0.573 (margin) | 0.802 | 0.970/0.263 | a✓ b✓ c✓ |
| centroid_sbert_q (k=10) | d9_8_7_6 | 0.596 [0.563, 0.675] | 0.966 [0.960, 0.972] | 0.2046 [0.1562, 0.2583] | 0.592 (per_shard_z) | 0.822 | 0.976/0.458 | a✓ b✓ c✓ |
| key_exact (k=10) | d9 | 1.000 [1.000, 1.000] | — — | 0.0000 [0.0000, 0.0000] | no-match op: orphan 1.000 / retain 0.147 (implied AUC 0.927; excluded from graded aggregation) | — | — | a✓ b· c· |
| key_exact (k=10) | d9_8 | 1.000 [1.000, 1.000] | — — | 0.0000 [0.0000, 0.0000] | no-match op: orphan 1.000 / retain 0.158 (implied AUC 0.921; excluded from graded aggregation) | — | — | a✓ b· c· |
| key_exact (k=10) | d9_8_7_6 | 1.000 [1.000, 1.000] | — — | 0.0000 [0.0000, 0.0000] | no-match op: orphan 1.000 / retain 0.177 (implied AUC 0.911; excluded from graded aggregation) | — | — | a✓ b· c· |
| key_exact (k=200) | d199 | 1.000 [1.000, 1.000] | — — | 0.0000 [0.0000, 0.0000] | no-match op: orphan 1.000 / retain 0.148 (implied AUC 0.926; excluded from graded aggregation) | — | — | a✓ b· c· |
| key_exact (k=200) | d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 1.000 [1.000, 1.000] | — — | 0.0000 [0.0000, 0.0000] | no-match op: orphan 1.000 / retain 0.147 (implied AUC 0.927; excluded from graded aggregation) | — | — | a✓ b· c· |
| key_tfidf (k=10) | d9 | 0.573 [0.520, 0.728] | 0.667 [0.640, 0.692] | 0.0022 [0.0003, 0.0050] | 0.973 (margin) | 0.069 | 0.955/0.041 | a✓ b✗ c✗ |
| key_tfidf (k=10) | d9_8 | 0.549 [0.504, 0.645] | 0.674 [0.653, 0.698] | 0.0084 [0.0031, 0.0159] | 0.980 (margin) | 0.051 | 0.960/0.100 | a✓ b✗ c✗ |
| key_tfidf (k=10) | d9_8_7_6 | 0.614 [0.584, 0.669] | 0.649 [0.634, 0.667] | 0.0138 [0.0058, 0.0246] | 0.978 (margin) | 0.053 | 0.964/0.188 | a✓ b✗ c✗ |
| key_tfidf (k=200) | d199 | 0.750 [0.750, 0.750] | 0.194 [0.194, 0.194] | 0.0000 [0.0000, 0.0000] | 0.999 (global_top1) | 0.002 | 0.950/0.000 | a✓ b✗ c✗ |
| key_tfidf (k=200) | d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 0.297 [0.305, 0.477] | 0.270 [0.229, 0.321] | 0.0003 [0.0000, 0.0011] | 0.989 (per_shard_z) | 0.029 | 0.960/0.000 | a✗ b✗ c✗ |
| logit_div (k=10) | d9 | 0.660 [0.593, 0.777] | 0.953 [0.937, 0.969] | 0.0625 [0.0393, 0.0876] | 0.633 (margin) | 0.702 | — | a✓ b✓ c✓ |
| logit_div (k=10) | d9_8 | 0.746 [0.685, 0.803] | 0.935 [0.917, 0.950] | 0.1671 [0.1240, 0.2137] | 0.621 (margin) | 0.710 | — | a✓ b✓ c✓ |
| logit_div (k=10) | d9_8_7_6 | 0.854 [0.821, 0.903] | 0.897 [0.882, 0.912] | 0.3393 [0.2724, 0.4100] | 0.609 (margin) | 0.769 | — | a✓ b✗ c✓ |
| ppl (k=10) | d9 | 0.540 [0.472, 0.647] | 0.377 [0.350, 0.404] | 0.0000 [0.0000, 0.0000] | 0.998 (margin) | 0.010 | — | a✓ b✗ c✗ |
| ppl (k=10) | d9_8 | 0.569 [0.518, 0.667] | 0.376 [0.353, 0.402] | 0.0000 [0.0000, 0.0000] | 0.999 (margin) | 0.005 | — | a✓ b✗ c✗ |
| ppl (k=10) | d9_8_7_6 | 0.619 [0.589, 0.700] | 0.365 [0.344, 0.384] | 0.0000 [0.0000, 0.0000] | 0.999 (margin) | 0.000 | — | a✓ b✗ c✗ |
| embed instructor-xl (base-pin) †prior | n=32 legonet forget10 | — | 0.980 | 0.7270 | — | — | — | — |
| embed instructor-xl (FT) †prior | n=32 legonet forget10 | — | 0.768 | — | — | — | — | — |
| centroid MiniLM (rl_centroid_k10) †prior | k=10 d9 | — | 0.971 | 0.0583 | — | — | — | — |
| MiniLM tombstone (author rung) †prior | k=10 d9 | — | — | — | — | — | 0.963/0.091 | — |
| SepMLP (routerless) | serve | pending | — | pending | pending | — | — | — |

†prior = typed-in prior measurement (not re-run this campaign).

## H-ARCH — REFUTED (>= 2 families separable: 2)

Bar: CONFIRM >=7/9 all-three; REFUTE >=2 separable; PENDING until all 9 families have measured detectors. Families meeting all three leak conditions: 6; separable families: 2.

## H-DIAL — deletion-count monotonicity

- FLAG activation_norm@k10 capture_top3: d9_8 → d9_8_7_6 decreases -0.0808 with disjoint CIs [1.0000, 1.0000] vs [0.8865, 0.9539]
- FLAG activation_norm@k10 adequacy: d9_8 → d9_8_7_6 decreases -0.0381 with disjoint CIs [0.9956, 0.9975] vs [0.9567, 0.9604]
- FLAG attn_norm@k10 adequacy: d9_8 → d9_8_7_6 decreases -0.0006 with disjoint CIs [1.0000, 1.0000] vs [0.9991, 0.9997]
- FLAG centroid_lm@k200 capture_top3: d199 → d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 decreases -0.5900 with disjoint CIs [0.9000, 0.9000] vs [0.2725, 0.4575]
- FLAG centroid_lm@k200 adequacy: d199 → d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 decreases -0.0141 with disjoint CIs [0.9761, 0.9761] vs [0.9505, 0.9722]
- FLAG centroid_sbert@k200 capture_top3: d199 → d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 decreases -0.6425 with disjoint CIs [0.9000, 0.9000] vs [0.2375, 0.4450]
- FLAG centroid_sbert@k200 adequacy: d199 → d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 decreases -0.0323 with disjoint CIs [0.7049, 0.7049] vs [0.6518, 0.6970]
- FLAG key_tfidf@k200 capture_top3: d199 → d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 decreases -0.4525 with disjoint CIs [0.7500, 0.7500] vs [0.3050, 0.4775]
- FLAG logit_div@k10 adequacy: d9_8 → d9_8_7_6 decreases -0.0377 with disjoint CIs [0.9166, 0.9502] vs [0.8816, 0.9120]
- sub-bar (pre-registered): centroid_sbert_q adequacy (sim-ratio) >= 0.95 at every k=10 drop set — PASS (d9 0.971, d9_8 0.971, d9_8_7_6 0.966)

## H-POOL — k=200 per-author granularity

- centroid_lm@k200 d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199: adequacy 0.962, best conf. AUC 0.761 — bars: adequacy>=0.9, AUC<=0.75
- centroid_lm@k200 d199: adequacy 0.976, best conf. AUC 0.728 — bars: adequacy>=0.9, AUC<=0.75
- centroid_sbert@k200 d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199: adequacy 0.673, best conf. AUC 0.984 — bars: adequacy>=0.9, AUC<=0.75
- centroid_sbert@k200 d199: adequacy 0.705, best conf. AUC 0.982 — bars: adequacy>=0.9, AUC<=0.75
- key_exact@k200 d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199: adequacy —, best conf. AUC — — bars: adequacy>=0.9, AUC<=0.75
- key_exact@k200 d199: adequacy —, best conf. AUC — — bars: adequacy>=0.9, AUC<=0.75
- key_tfidf@k200 d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199: adequacy 0.270, best conf. AUC 0.989 — bars: adequacy>=0.9, AUC<=0.75
- key_tfidf@k200 d199: adequacy 0.194, best conf. AUC 0.999 — bars: adequacy>=0.9, AUC<=0.75

## H-ENC — encoder generality (k=10 centroid audits)

| encoder | sim-ratio (adequacy) | retain shift | tomb author catch/FPR | deletion-disclosure AUC (forget-vs-holdout) | sim-ratio>=0.95 |
|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 0.983 | 0.0700 | 0.953/0.025 | — | True |
| sentence-transformers/all-mpnet-base-v2 | 0.970 | 0.0822 | 0.980/0.136 | — | True |
- confidence-AUC half — CONFIRMED (all 2 encoders' confidence AUC <= 0.75). Bar: per-encoder max confidence AUC (global_top1/per_expert/margin/knn_density; tomb_* excluded): CONFIRM <= 0.75, REFUTE >= 0.90, else inconclusive.
  - bge: max confidence AUC 0.649 (margin, FPR@90% catch 0.624) → confirm
  - mpnet: max confidence AUC 0.622 (global_top1, FPR@90% catch 0.824) → confirm
- priors: MiniLM 0.971 (k=10), instructor-xl base 0.980 / FT 0.768 (n=32) — typed constants.

## H-TRAINED — RouterLoRA drop audit: CONFIRMED (all seeds AUC <= 0.75 and orphan top-1 share ratio >= 0.9)

| seed | AUC(h_norm) filt. | AUC(-max_share) filt. | excl. m==1 rows | orphan top-1 share ratio (all rows†) | ratio (m>1 rows) | fallback (orphan) |
|---|---|---|---|---|---|---|
| 42 | 0.588 | 0.555 | 382 | 1.657 | 0.925 | 0.350 |
| 43 | 0.585 | 0.554 | 382 | 1.660 | 0.922 | 0.350 |
| 44 | 0.590 | 0.554 | 382 | 1.658 | 0.922 | 0.350 |
- cross-seed auc_h_norm_filtered: 0.588 ± 0.002 (n=3)
- cross-seed auc_max_share_filtered: 0.555 ± 0.001 (n=3)
- cross-seed top1_share_ratio_forget: 1.658 ± 0.001 (n=3)
- cross-seed top1_share_ratio_forget_multi: 0.923 ± 0.001 (n=3)
- † the forget_ratio bar reads the UNFILTERED all-row top1_share_ratio (m==1 rows report ratio 1.0 and inflate it); the AUC bars are m==1-filtered — per the pre-registration the m==1 filter applies to AUCs only. The m>1-rows column is context, not the bar.

## H-DATASET — DBpedia retriever

- rl_family_dbpedia.json: pooled sim-ratio 0.912; tags:
  - d0: capture(top3) 1.000, sim-ratio 0.881, retain shift 0.1250
  - d1: capture(top3) 1.000, sim-ratio 0.948, retain shift 0.1150
  - d2: capture(top3) 1.000, sim-ratio 0.941, retain shift 0.0900
  - d_batch15: capture(top3) 0.800, sim-ratio 0.909, retain shift 0.7850
  - separating tau exists (90% catch at <=10% retain cost): False

## Identity controls

- oracle: orphans → base/scaffold P=1.0, retain shift ≡ 0 (by construction; analytic).
- key_exact@k200 d199: orphan no-match 1.000, retain no-match 0.148, implied AUC 0.926, fallback shard 0 (a *design* leak with a usable native detector).
- key_exact@k10 d9: orphan no-match 1.000, retain no-match 0.147, implied AUC 0.927, fallback shard 0 (a *design* leak with a usable native detector).
- SepMLP branch-silence: pending.
- H-SEAL-GEN (per-feature-space tombstone, argmax operating point):
  - centroid_lm@k10: catch 0.980 / FPR 0.974 ✗
  - centroid_lm@k200: catch 0.950 / FPR 0.338 ✗
  - centroid_lm_last@k10: catch 1.000 / FPR 0.997 ✗
  - centroid_sbert@k10: catch 0.975 / FPR 0.390 ✗
  - centroid_sbert@k200: catch 1.000 / FPR 0.000 ✓
  - centroid_sbert_q@k10: catch 0.950 / FPR 0.115 ✗
  - key_tfidf@k10: catch 0.955 / FPR 0.041 ✓
  - key_tfidf@k200: catch 0.950 / FPR 0.000 ✓
  - spaces meeting catch>=0.90 @ FPR<=0.10: 3 (+ MiniLM prior 0.963/0.091); bar: >=3/4 feature spaces.

## Missing inputs / warnings

- all requested inputs present
