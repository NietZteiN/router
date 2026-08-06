# Where orphans land: destination concentration (task (a))

Reduction of the router-family sweep's stored orphan-destination histograms. max_share = fraction of deleted-author questions landing on the single busiest surviving unit; n_eff = 1/HHI = effective number of siblings the leak spreads over (1 = one magnet expert, high = diffuse).

## Per-cell concentration

top3_share = fraction on the busiest three survivors; Gini = inequality of the orphan mass over survivors (0 = uniform, →1 = one magnet).

| pool | router | drop | orphans | max_share | top3 | n_eff | entropy | HHI | Gini | busiest |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | key_exact | d9 | 400 | 1.00 | 1.00 | 1.0 | -0.00 | 1.00 | 0.89 | s0 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | key_exact | d9_8_7_6 | 1600 | 0.99 | 1.00 | 1.0 | 0.04 | 0.98 | 0.83 | s0 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | key_exact | d9_8 | 800 | 0.98 | 1.00 | 1.0 | 0.05 | 0.95 | 0.87 | s0 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | attn_norm | d9_8 | 441 | 0.95 | 1.00 | 1.1 | 0.11 | 0.90 | 0.86 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | attn_norm | d9_8_7_6 | 520 | 0.94 | 1.00 | 1.1 | 0.13 | 0.89 | 0.81 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | attn_norm | d9 | 400 | 0.94 | 0.99 | 1.1 | 0.12 | 0.89 | 0.87 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | activation_norm | d9_8_7_6 | 520 | 0.90 | 1.00 | 1.2 | 0.19 | 0.82 | 0.80 | s5 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_lm_last | d9_8 | 800 | 0.83 | 1.00 | 1.4 | 0.23 | 0.71 | 0.83 | s2 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_lm_last | d9_8_7_6 | 1600 | 0.82 | 1.00 | 1.4 | 0.27 | 0.71 | 0.77 | s2 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_lm_last | d9 | 400 | 0.80 | 1.00 | 1.5 | 0.25 | 0.67 | 0.84 | s2 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | activation_norm | d9_8 | 441 | 0.74 | 1.00 | 1.7 | 0.32 | 0.61 | 0.80 | s7 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | logit_div | d9_8 | 441 | 0.69 | 0.91 | 2.0 | 0.53 | 0.50 | 0.69 | s7 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_lm | d9 | 400 | 0.59 | 0.83 | 2.5 | 0.58 | 0.39 | 0.68 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_lm | d9_8 | 800 | 0.57 | 0.85 | 2.7 | 0.61 | 0.38 | 0.64 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | logit_div | d9 | 400 | 0.53 | 0.84 | 2.9 | 0.64 | 0.34 | 0.65 | s7 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_lm | d9_8_7_6 | 1600 | 0.51 | 0.95 | 2.8 | 0.67 | 0.36 | 0.56 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | activation_norm | d9 | 400 | 0.51 | 0.99 | 2.4 | 0.45 | 0.42 | 0.75 | s7 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | logit_div | d9_8_7_6 | 520 | 0.44 | 0.81 | 3.5 | 0.83 | 0.28 | 0.43 | s5 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | ppl | d9_8_7_6 | 520 | 0.33 | 0.70 | 4.7 | 0.92 | 0.21 | 0.29 | s1 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | key_tfidf | d9_8_7_6 | 1600 | 0.27 | 0.61 | 5.5 | 0.98 | 0.18 | 0.16 | s5 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_sbert_q | d9_8_7_6 | 1600 | 0.24 | 0.60 | 5.7 | 0.98 | 0.18 | 0.13 | s5 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | ppl | d9_8 | 441 | 0.24 | 0.61 | 6.0 | 0.92 | 0.17 | 0.32 | s1 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_sbert | d9_8_7_6 | 1600 | 0.24 | 0.60 | 5.7 | 0.98 | 0.18 | 0.14 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_sbert_q | d9_8 | 800 | 0.23 | 0.55 | 6.7 | 0.96 | 0.15 | 0.24 | s5 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | ppl | d9 | 400 | 0.23 | 0.58 | 6.4 | 0.91 | 0.16 | 0.35 | s1 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | key_tfidf | d9 | 400 | 0.22 | 0.57 | 6.6 | 0.92 | 0.15 | 0.32 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_sbert_q | d9 | 400 | 0.20 | 0.48 | 7.7 | 0.96 | 0.13 | 0.23 | s7 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_sbert | d9_8 | 800 | 0.20 | 0.54 | 6.9 | 0.96 | 0.15 | 0.22 | s4 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | key_tfidf | d9_8 | 800 | 0.20 | 0.55 | 6.8 | 0.96 | 0.15 | 0.24 | s7 |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_sbert | d9 | 400 | 0.18 | 0.54 | 7.2 | 0.94 | 0.14 | 0.28 | s7 |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | key_exact | d199 | 20 | 1.00 | 1.00 | 1.0 | -0.00 | 1.00 | 0.99 | s0 |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | key_exact | d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 400 | 1.00 | 1.00 | 1.0 | -0.00 | 1.00 | 0.99 | s0 |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | centroid_lm | d199 | 20 | 0.70 | 0.90 | 1.9 | 0.19 | 0.52 | 0.99 | s128 |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | centroid_sbert | d199 | 20 | 0.40 | 0.90 | 3.3 | 0.24 | 0.30 | 0.98 | s88 |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | key_tfidf | d199 | 20 | 0.40 | 0.75 | 4.0 | 0.31 | 0.25 | 0.98 | s88 |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | key_tfidf | d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 400 | 0.19 | 0.30 | 17.5 | 0.69 | 0.06 | 0.83 | s88 |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | centroid_lm | d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 400 | 0.17 | 0.31 | 17.4 | 0.65 | 0.06 | 0.88 | s88 |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | centroid_sbert | d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 400 | 0.11 | 0.26 | 24.2 | 0.70 | 0.04 | 0.84 | s88 |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | key_exact | d49 | 80 | 1.00 | 1.00 | 1.0 | -0.00 | 1.00 | 0.98 | s0 |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | key_exact | d49_48 | 160 | 1.00 | 1.00 | 1.0 | -0.00 | 1.00 | 0.98 | s0 |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | centroid_lm | d49_48 | 160 | 0.25 | 0.52 | 8.1 | 0.63 | 0.12 | 0.82 | s22 |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | centroid_lm | d49 | 80 | 0.24 | 0.59 | 7.3 | 0.59 | 0.14 | 0.84 | s13 |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | key_tfidf | d49 | 80 | 0.23 | 0.46 | 9.6 | 0.67 | 0.10 | 0.78 | s1 |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | centroid_sbert | d49 | 80 | 0.19 | 0.53 | 8.2 | 0.59 | 0.12 | 0.84 | s22 |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | key_tfidf | d49_48 | 160 | 0.12 | 0.33 | 15.4 | 0.78 | 0.06 | 0.68 | s1 |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | centroid_sbert | d49_48 | 160 | 0.11 | 0.31 | 15.2 | 0.75 | 0.07 | 0.70 | s22 |
| Llama-3.2-1B-Instruct | attn_norm | d9_8_7_6 | 520 | 1.00 | 1.00 | 1.0 | -0.00 | 1.00 | 0.83 | s5 |
| Llama-3.2-1B-Instruct | key_exact | d9 | 400 | 1.00 | 1.00 | 1.0 | -0.00 | 1.00 | 0.89 | s0 |
| Llama-3.2-1B-Instruct | key_exact | d9_8_7_6 | 1600 | 0.99 | 1.00 | 1.0 | 0.04 | 0.98 | 0.83 | s0 |
| Llama-3.2-1B-Instruct | key_exact | d9_8 | 800 | 0.98 | 1.00 | 1.0 | 0.05 | 0.95 | 0.87 | s0 |
| Llama-3.2-1B-Instruct | activation_norm | d9 | 400 | 0.96 | 1.00 | 1.1 | 0.08 | 0.92 | 0.88 | s6 |
| Llama-3.2-1B-Instruct | activation_norm | d9_8 | 441 | 0.95 | 1.00 | 1.1 | 0.09 | 0.91 | 0.86 | s6 |
| Llama-3.2-1B-Instruct | attn_norm | d9_8 | 441 | 0.95 | 1.00 | 1.1 | 0.10 | 0.91 | 0.86 | s5 |
| Llama-3.2-1B-Instruct | attn_norm | d9 | 400 | 0.95 | 1.00 | 1.1 | 0.09 | 0.90 | 0.88 | s5 |
| Llama-3.2-1B-Instruct | activation_norm | d9_8_7_6 | 520 | 0.90 | 1.00 | 1.2 | 0.19 | 0.82 | 0.80 | s5 |
| Llama-3.2-1B-Instruct | centroid_lm_last | d9 | 400 | 0.61 | 0.76 | 2.6 | 0.66 | 0.39 | 0.58 | s4 |
| Llama-3.2-1B-Instruct | centroid_lm | d9_8 | 800 | 0.61 | 0.87 | 2.5 | 0.61 | 0.41 | 0.65 | s4 |
| Llama-3.2-1B-Instruct | centroid_lm | d9 | 400 | 0.60 | 0.84 | 2.5 | 0.57 | 0.40 | 0.69 | s4 |
| Llama-3.2-1B-Instruct | centroid_lm_last | d9_8 | 800 | 0.59 | 0.74 | 2.7 | 0.70 | 0.37 | 0.51 | s4 |
| Llama-3.2-1B-Instruct | logit_div | d9_8 | 441 | 0.58 | 0.83 | 2.7 | 0.64 | 0.38 | 0.62 | s7 |
| Llama-3.2-1B-Instruct | centroid_lm | d9_8_7_6 | 1600 | 0.56 | 0.91 | 2.6 | 0.68 | 0.39 | 0.57 | s4 |
| Llama-3.2-1B-Instruct | centroid_lm_last | d9_8_7_6 | 1600 | 0.56 | 0.80 | 2.8 | 0.76 | 0.36 | 0.47 | s4 |
| Llama-3.2-1B-Instruct | logit_div | d9_8_7_6 | 520 | 0.52 | 0.95 | 2.4 | 0.57 | 0.43 | 0.62 | s5 |
| Llama-3.2-1B-Instruct | logit_div | d9 | 400 | 0.42 | 0.80 | 3.6 | 0.69 | 0.28 | 0.61 | s8 |
| Llama-3.2-1B-Instruct | key_tfidf | d9_8_7_6 | 1600 | 0.27 | 0.61 | 5.5 | 0.98 | 0.18 | 0.16 | s5 |
| Llama-3.2-1B-Instruct | ppl | d9_8_7_6 | 520 | 0.26 | 0.69 | 5.1 | 0.95 | 0.20 | 0.24 | s0 |
| Llama-3.2-1B-Instruct | ppl | d9_8 | 441 | 0.25 | 0.56 | 6.4 | 0.94 | 0.16 | 0.28 | s7 |
| Llama-3.2-1B-Instruct | centroid_sbert_q | d9_8_7_6 | 1600 | 0.24 | 0.60 | 5.7 | 0.98 | 0.18 | 0.13 | s5 |
| Llama-3.2-1B-Instruct | centroid_sbert | d9_8_7_6 | 1600 | 0.24 | 0.60 | 5.7 | 0.98 | 0.18 | 0.14 | s4 |
| Llama-3.2-1B-Instruct | centroid_sbert_q | d9_8 | 800 | 0.23 | 0.55 | 6.7 | 0.96 | 0.15 | 0.24 | s5 |
| Llama-3.2-1B-Instruct | ppl | d9 | 400 | 0.23 | 0.56 | 6.7 | 0.92 | 0.15 | 0.33 | s7 |
| Llama-3.2-1B-Instruct | key_tfidf | d9 | 400 | 0.22 | 0.57 | 6.6 | 0.92 | 0.15 | 0.32 | s4 |
| Llama-3.2-1B-Instruct | centroid_sbert_q | d9 | 400 | 0.20 | 0.48 | 7.7 | 0.96 | 0.13 | 0.23 | s7 |
| Llama-3.2-1B-Instruct | centroid_sbert | d9_8 | 800 | 0.20 | 0.54 | 6.9 | 0.96 | 0.15 | 0.22 | s4 |
| Llama-3.2-1B-Instruct | key_tfidf | d9_8 | 800 | 0.20 | 0.55 | 6.8 | 0.96 | 0.15 | 0.24 | s7 |
| Llama-3.2-1B-Instruct | centroid_sbert | d9 | 400 | 0.18 | 0.54 | 7.2 | 0.94 | 0.14 | 0.28 | s7 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | key_exact | d9 | 400 | 1.00 | 1.00 | 1.0 | -0.00 | 1.00 | 0.89 | s0 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | key_exact | d9_8_7_6 | 1600 | 0.99 | 1.00 | 1.0 | 0.04 | 0.98 | 0.83 | s0 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | key_exact | d9_8 | 800 | 0.98 | 1.00 | 1.0 | 0.05 | 0.95 | 0.87 | s0 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | activation_norm | d9_8 | 441 | 0.83 | 1.00 | 1.4 | 0.23 | 0.72 | 0.83 | s6 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | activation_norm | d9 | 400 | 0.82 | 1.00 | 1.4 | 0.25 | 0.69 | 0.84 | s6 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_lm_last | d9 | 400 | 0.77 | 0.87 | 1.7 | 0.43 | 0.60 | 0.73 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_lm_last | d9_8 | 800 | 0.77 | 0.89 | 1.7 | 0.45 | 0.60 | 0.72 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_lm | d9_8 | 800 | 0.70 | 0.90 | 1.9 | 0.50 | 0.52 | 0.71 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_lm | d9_8_7_6 | 1600 | 0.69 | 0.92 | 2.0 | 0.58 | 0.50 | 0.62 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_lm_last | d9_8_7_6 | 1600 | 0.68 | 0.91 | 2.0 | 0.59 | 0.50 | 0.62 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_lm | d9 | 400 | 0.65 | 0.92 | 2.1 | 0.50 | 0.47 | 0.74 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | activation_norm | d9_8_7_6 | 520 | 0.52 | 0.92 | 2.7 | 0.66 | 0.37 | 0.58 | s5 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | logit_div | d9_8_7_6 | 520 | 0.47 | 0.85 | 3.1 | 0.76 | 0.33 | 0.49 | s5 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | attn_norm | d9_8_7_6 | 520 | 0.43 | 0.86 | 3.4 | 0.75 | 0.30 | 0.48 | s3 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | attn_norm | d9 | 400 | 0.41 | 0.81 | 3.7 | 0.68 | 0.27 | 0.61 | s3 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | attn_norm | d9_8 | 441 | 0.41 | 0.82 | 3.8 | 0.71 | 0.27 | 0.56 | s3 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | ppl | d9_8 | 441 | 0.28 | 0.57 | 6.2 | 0.94 | 0.16 | 0.27 | s7 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | key_tfidf | d9_8_7_6 | 1600 | 0.27 | 0.61 | 5.5 | 0.98 | 0.18 | 0.16 | s5 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | logit_div | d9 | 400 | 0.27 | 0.66 | 5.5 | 0.85 | 0.18 | 0.44 | s0 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | logit_div | d9_8 | 441 | 0.26 | 0.75 | 5.0 | 0.85 | 0.20 | 0.41 | s0 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | ppl | d9 | 400 | 0.25 | 0.54 | 7.0 | 0.94 | 0.14 | 0.28 | s7 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_sbert_q | d9_8_7_6 | 1600 | 0.24 | 0.60 | 5.7 | 0.98 | 0.18 | 0.13 | s5 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_sbert | d9_8_7_6 | 1600 | 0.24 | 0.60 | 5.7 | 0.98 | 0.18 | 0.14 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_sbert_q | d9_8 | 800 | 0.23 | 0.55 | 6.7 | 0.96 | 0.15 | 0.24 | s5 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | ppl | d9_8_7_6 | 520 | 0.22 | 0.62 | 5.5 | 0.97 | 0.18 | 0.16 | s1 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | key_tfidf | d9 | 400 | 0.22 | 0.57 | 6.6 | 0.92 | 0.15 | 0.32 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_sbert_q | d9 | 400 | 0.20 | 0.48 | 7.7 | 0.96 | 0.13 | 0.23 | s7 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_sbert | d9_8 | 800 | 0.20 | 0.54 | 6.9 | 0.96 | 0.15 | 0.22 | s4 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | key_tfidf | d9_8 | 800 | 0.20 | 0.55 | 6.8 | 0.96 | 0.15 | 0.24 | s7 |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_sbert | d9 | 400 | 0.18 | 0.54 | 7.2 | 0.94 | 0.14 | 0.28 | s7 |
| legonet_n32 (instructor-xl) | embed-instructor-xl | forget10 | 400 | 0.23 | 0.64 | 6.1 | 0.70 | 0.16 | 0.67 | s5 |
| legonet_n32 (retriever) | embed-retriever | forget10 | 400 | 0.25 | 0.57 | 6.7 | 0.78 | 0.15 | 0.57 | s30 |
| scaf_k10 | centroid_bge | d9 | 400 | 0.33 | 0.59 | 5.7 | 0.89 | 0.18 | 0.37 | s4 |
| scaf_k10 | centroid_all | d9 | 400 | 0.24 | 0.57 | 6.7 | 0.93 | 0.15 | 0.31 | s4 |
| scaf_k10 | centroid_minilm | d9 | 400 | 0.20 | 0.48 | 7.7 | 0.96 | 0.13 | 0.23 | s7 |

## Concentration vs deletion count (does the magnet hold?)

| pool | router | drop sets | max_share trajectory | n_eff | trend |
|---|---|---|---|---|---|
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | key_exact | d9/d9_8/d9_8_7_6 | 1.00 → 0.98 → 0.99 | 1.0 → 1.0 → 1.0 | **holds** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | attn_norm | d9/d9_8/d9_8_7_6 | 0.94 → 0.95 → 0.94 | 1.1 → 1.1 → 1.1 | **holds** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_lm_last | d9/d9_8/d9_8_7_6 | 0.80 → 0.83 → 0.82 | 1.5 → 1.4 → 1.4 | **holds** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_lm | d9/d9_8/d9_8_7_6 | 0.59 → 0.57 → 0.51 | 2.5 → 2.7 → 2.8 | **shrinks** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | logit_div | d9/d9_8/d9_8_7_6 | 0.53 → 0.69 → 0.44 | 2.9 → 2.0 → 3.5 | **shrinks** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | activation_norm | d9/d9_8/d9_8_7_6 | 0.51 → 0.74 → 0.90 | 2.4 → 1.7 → 1.2 | **grows** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | ppl | d9/d9_8/d9_8_7_6 | 0.23 → 0.24 → 0.33 | 6.4 → 6.0 → 4.7 | **grows** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | key_tfidf | d9/d9_8/d9_8_7_6 | 0.22 → 0.20 → 0.27 | 6.6 → 6.8 → 5.5 | **grows** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_sbert_q | d9/d9_8/d9_8_7_6 | 0.20 → 0.23 → 0.24 | 7.7 → 6.7 → 5.7 | **grows** |
| Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 | centroid_sbert | d9/d9_8/d9_8_7_6 | 0.18 → 0.20 → 0.24 | 7.2 → 6.9 → 5.7 | **grows** |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | key_exact | d199/d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 1.00 → 1.00 | 1.0 → 1.0 | **holds** |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | centroid_lm | d199/d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 0.70 → 0.17 | 1.9 → 17.4 | **shrinks** |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | centroid_sbert | d199/d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 0.40 → 0.11 | 3.3 → 24.2 | **shrinks** |
| Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 | key_tfidf | d199/d180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199 | 0.40 → 0.19 | 4.0 → 17.5 | **shrinks** |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | key_exact | d49/d49_48 | 1.00 → 1.00 | 1.0 → 1.0 | **holds** |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | centroid_lm | d49/d49_48 | 0.24 → 0.25 | 7.3 → 8.1 | **holds** |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | key_tfidf | d49/d49_48 | 0.23 → 0.12 | 9.6 → 15.4 | **shrinks** |
| Llama-2-7B-chat-hf_k50_r32_e5_lr1e4 | centroid_sbert | d49/d49_48 | 0.19 → 0.11 | 8.2 → 15.2 | **shrinks** |
| Llama-3.2-1B-Instruct | key_exact | d9/d9_8/d9_8_7_6 | 1.00 → 0.98 → 0.99 | 1.0 → 1.0 → 1.0 | **holds** |
| Llama-3.2-1B-Instruct | activation_norm | d9/d9_8/d9_8_7_6 | 0.96 → 0.95 → 0.90 | 1.1 → 1.1 → 1.2 | **shrinks** |
| Llama-3.2-1B-Instruct | attn_norm | d9/d9_8/d9_8_7_6 | 0.95 → 0.95 → 1.00 | 1.1 → 1.1 → 1.0 | **grows** |
| Llama-3.2-1B-Instruct | centroid_lm_last | d9/d9_8/d9_8_7_6 | 0.61 → 0.59 → 0.56 | 2.6 → 2.7 → 2.8 | **shrinks** |
| Llama-3.2-1B-Instruct | centroid_lm | d9/d9_8/d9_8_7_6 | 0.60 → 0.61 → 0.56 | 2.5 → 2.5 → 2.6 | **shrinks** |
| Llama-3.2-1B-Instruct | logit_div | d9/d9_8/d9_8_7_6 | 0.42 → 0.58 → 0.52 | 3.6 → 2.7 → 2.4 | **grows** |
| Llama-3.2-1B-Instruct | ppl | d9/d9_8/d9_8_7_6 | 0.23 → 0.25 → 0.26 | 6.7 → 6.4 → 5.1 | **grows** |
| Llama-3.2-1B-Instruct | key_tfidf | d9/d9_8/d9_8_7_6 | 0.22 → 0.20 → 0.27 | 6.6 → 6.8 → 5.5 | **grows** |
| Llama-3.2-1B-Instruct | centroid_sbert_q | d9/d9_8/d9_8_7_6 | 0.20 → 0.23 → 0.24 | 7.7 → 6.7 → 5.7 | **grows** |
| Llama-3.2-1B-Instruct | centroid_sbert | d9/d9_8/d9_8_7_6 | 0.18 → 0.20 → 0.24 | 7.2 → 6.9 → 5.7 | **grows** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | key_exact | d9/d9_8/d9_8_7_6 | 1.00 → 0.98 → 0.99 | 1.0 → 1.0 → 1.0 | **holds** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | activation_norm | d9/d9_8/d9_8_7_6 | 0.82 → 0.83 → 0.52 | 1.4 → 1.4 → 2.7 | **shrinks** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_lm_last | d9/d9_8/d9_8_7_6 | 0.77 → 0.77 → 0.68 | 1.7 → 1.7 → 2.0 | **shrinks** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_lm | d9/d9_8/d9_8_7_6 | 0.65 → 0.70 → 0.69 | 2.1 → 1.9 → 2.0 | **grows** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | attn_norm | d9/d9_8/d9_8_7_6 | 0.41 → 0.41 → 0.43 | 3.7 → 3.8 → 3.4 | **holds** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | logit_div | d9/d9_8/d9_8_7_6 | 0.27 → 0.26 → 0.47 | 5.5 → 5.0 → 3.1 | **grows** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | ppl | d9/d9_8/d9_8_7_6 | 0.25 → 0.28 → 0.22 | 7.0 → 6.2 → 5.5 | **holds** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | key_tfidf | d9/d9_8/d9_8_7_6 | 0.22 → 0.20 → 0.27 | 6.6 → 6.8 → 5.5 | **grows** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_sbert_q | d9/d9_8/d9_8_7_6 | 0.20 → 0.23 → 0.24 | 7.7 → 6.7 → 5.7 | **grows** |
| Llama-3.2-1B-Instruct_experts_scaf_k10 | centroid_sbert | d9/d9_8/d9_8_7_6 | 0.18 → 0.20 → 0.24 | 7.2 → 6.9 → 5.7 | **grows** |

## Per-author landing determinism (new)

For each DELETED author, the fraction of its ~20 orphan questions that land on a single surviving destination (1.0 = a per-author magnet — every question of that author routes to the SAME sibling; low = its questions scatter). Mean/median over deleted authors; recomputed from the `.sims.npz` sidecars, no inference.

| source | router | del. authors | mean determinism | median | mean landing-entropy |
|---|---|---|---|---|---|
| rl_family_k10_1b_plain_behavioral | activation_norm | 20 | 0.957 | 0.975 | 0.047 |
| rl_family_k10_1b_plain_behavioral | attn_norm | 20 | 0.948 | 0.950 | 0.054 |
| rl_family_k10_7b_behavioral | attn_norm | 20 | 0.940 | 0.950 | 0.065 |
| rl_family_k10_feature | centroid_lm_last | 20 | 0.823 | 0.900 | 0.139 |
| rl_family_k10_behavioral | activation_norm | 20 | 0.817 | 0.850 | 0.159 |
| rl_family_k10_7b_feature | centroid_lm_last | 20 | 0.805 | 0.850 | 0.143 |
| rl_family_k10_1b_plain_feature | centroid_lm | 20 | 0.713 | 0.725 | 0.218 |
| rl_family_k10_feature | centroid_lm | 20 | 0.692 | 0.700 | 0.229 |
| rl_family_k10_1b_plain_feature | centroid_lm_last | 20 | 0.683 | 0.700 | 0.252 |
| rl_family_k10_7b_feature | centroid_lm | 20 | 0.667 | 0.675 | 0.267 |
| rl_family_k10_7b_behavioral | activation_norm | 20 | 0.625 | 0.625 | 0.280 |
| rl_family_k10_feature | centroid_sbert_q | 20 | 0.605 | 0.575 | 0.332 |
| rl_family_k10_7b_feature | centroid_sbert_q | 20 | 0.605 | 0.575 | 0.332 |
| rl_family_k10_1b_plain_feature | centroid_sbert_q | 20 | 0.605 | 0.575 | 0.332 |
| rl_family_k10_feature | centroid_sbert | 20 | 0.595 | 0.600 | 0.337 |
| rl_family_k10_7b_feature | centroid_sbert | 20 | 0.595 | 0.600 | 0.337 |
| rl_family_k10_1b_plain_feature | centroid_sbert | 20 | 0.595 | 0.600 | 0.337 |
| rl_family_k10_1b_plain_behavioral | ppl | 20 | 0.577 | 0.550 | 0.348 |
| rl_family_k50_7b | centroid_sbert | 20 | 0.530 | 0.525 | 0.415 |
| rl_family_k200 | centroid_sbert | 20 | 0.527 | 0.450 | 0.411 |
| rl_family_k10_7b_behavioral | logit_div | 20 | 0.525 | 0.500 | 0.366 |
| rl_family_k10_1b_plain_behavioral | logit_div | 20 | 0.522 | 0.500 | 0.376 |
| rl_family_k10_behavioral | logit_div | 20 | 0.508 | 0.500 | 0.433 |
| rl_family_k10_feature | key_tfidf | 20 | 0.508 | 0.425 | 0.434 |
| rl_family_k10_7b_feature | key_tfidf | 20 | 0.508 | 0.425 | 0.434 |
| rl_family_k10_1b_plain_feature | key_tfidf | 20 | 0.508 | 0.425 | 0.434 |
| rl_family_k200 | key_tfidf | 20 | 0.505 | 0.400 | 0.495 |
| rl_family_k50_7b | centroid_lm | 20 | 0.500 | 0.500 | 0.478 |
| rl_family_k50_7b | key_tfidf | 20 | 0.495 | 0.450 | 0.519 |
| rl_family_k10_behavioral | attn_norm | 20 | 0.465 | 0.450 | 0.409 |
| rl_family_k10_behavioral | ppl | 20 | 0.432 | 0.450 | 0.468 |
| rl_family_k200 | centroid_lm | 20 | 0.423 | 0.375 | 0.534 |
| rl_family_k10_7b_behavioral | ppl | 20 | 0.418 | 0.375 | 0.466 |
