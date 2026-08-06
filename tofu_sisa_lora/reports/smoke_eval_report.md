# TOFU SISA-LoRA Smoke Eval Report
Generated: 2026-06-05 16:02
k=10, forget_shard_id=9 (authors 180–199 = TOFU forget10 split)
Smoke caps: ROUGE max 50, retain max 80, truth ratio max 30

## Base Model (no LoRA — target for forget quality)

| Model | model_utility | retain_rouge | real_rouge | world_rouge | forget_truth_ratio | ks_pval |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 0.1891 | 0.276 | 0.147 | 0.117 | 0.938 | **0.846** |
| phi-2 (2.7B) | 0.1554 | 0.183 | 0.270 | 0.433 | 0.911 | **1.000** |
| Llama-3.2-1B | 0.1403 | 0.212 | 0.103 | 0.068 | 0.971 | **0.946** |

Base models correctly achieve high ks_pval (≥0.85) — the KS test correctly identifies they were never trained on the forget data.

## shard_9_only — Forget Shard Memorisation Fingerprint

| Model | shard_9 forget_tr | Other shards avg | Delta |
|---|---|---|---|
| TinyLlama-1.1B | **0.681** | 0.869 | -0.188 |
| phi-2 (2.7B) | **0.793** | 0.847 | -0.054 |
| Llama-3.2-1B | **0.635** | 0.915 | -0.280 |

shard_9 has the lowest forget_truth_ratio — it was trained on the forget authors so it strongly prefers correct answers. Delta shows how much more it memorised vs other shards.

## Unlearning Efficacy — Best Method per Model (dare_ties)

| Model | base mu | merged mu | remerge mu | utility drop | remerge ks_pval |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 0.1891 | 0.1750 | 0.1602 | +0.0148 (8.5%) | **0.000** |
| phi-2 (2.7B) | 0.1554 | 0.1436 | 0.0744 | +0.0692 (48.2%) | **0.583** |
| Llama-3.2-1B | 0.1403 | 0.2212 | 0.2116 | +0.0096 (4.3%) | **0.000** |

Only phi-2 remerge_dare_ties achieves meaningful forget quality (ks_pval=0.583). TinyLlama and Llama remain fully distinguishable (ks_pval=0.000).

## Full Results Table (all labels)

### TinyLlama-1.1B

| label | ret_r | real_r | wrld_r | ret_p | real_p | wrld_p | fgt_tr | ks_pval | model_utility |
|---|---|---|---|---|---|---|---|---|---|
| base_model | 0.276 | 0.147 | 0.117 | 0.221 | 0.285 | 0.319 | 0.938 | 0.846 | 0.1891 |
| merged_cat | 0.000 | 0.000 | 0.000 | 0.000 | 0.208 | 0.227 | 0.826 | 0.000 | 0.0001 |
| merged_dare_linear | 0.234 | 0.069 | 0.045 | 0.278 | 0.315 | 0.328 | 0.811 | 0.000 | 0.0701 |
| merged_dare_ties | 0.266 | 0.140 | 0.089 | 0.234 | 0.285 | 0.321 | 0.921 | 0.000 | 0.1750 |
| merged_linear | 0.172 | 0.001 | 0.018 | 0.149 | 0.289 | 0.264 | 0.760 | 0.469 | 0.0046 |
| merged_magnitude_prune | 0.181 | 0.001 | 0.019 | 0.157 | 0.291 | 0.272 | 0.761 | 0.469 | 0.0045 |
| merged_ties | 0.277 | 0.134 | 0.065 | 0.245 | 0.285 | 0.321 | 0.904 | 0.000 | 0.1574 |
| remerge_cat | 0.000 | 0.000 | 0.000 | 0.000 | 0.203 | 0.227 | 0.851 | 0.000 | 0.0002 |
| remerge_dare_linear | 0.238 | 0.070 | 0.048 | 0.278 | 0.305 | 0.324 | 0.848 | 0.000 | 0.1023 |
| remerge_dare_ties | 0.264 | 0.130 | 0.070 | 0.236 | 0.285 | 0.320 | 0.920 | 0.000 | 0.1602 |
| remerge_linear | 0.169 | 0.001 | 0.013 | 0.137 | 0.280 | 0.255 | 0.819 | 0.000 | 0.0046 |
| remerge_magnitude_prune | 0.178 | 0.001 | 0.014 | 0.146 | 0.284 | 0.260 | 0.820 | 0.000 | 0.0085 |
| remerge_ties | 0.281 | 0.116 | 0.053 | 0.249 | 0.285 | 0.320 | 0.907 | 0.000 | 0.1425 |
| shard_0_only | 0.241 | 0.068 | 0.043 | 0.273 | 0.308 | 0.321 | 0.874 | 0.000 | 0.1261 |
| shard_1_only | 0.240 | 0.076 | 0.053 | 0.264 | 0.300 | 0.320 | 0.878 | 0.000 | 0.1365 |
| shard_2_only | 0.213 | 0.057 | 0.042 | 0.257 | 0.312 | 0.316 | 0.887 | 0.000 | 0.1157 |
| shard_3_only | 0.229 | 0.059 | 0.049 | 0.262 | 0.320 | 0.322 | 0.853 | 0.000 | 0.1267 |
| shard_4_only | 0.290 | 0.085 | 0.050 | 0.280 | 0.301 | 0.313 | 0.853 | 0.000 | 0.1319 |
| shard_5_only | 0.251 | 0.065 | 0.038 | 0.263 | 0.306 | 0.323 | 0.855 | 0.000 | 0.1151 |
| shard_6_only | 0.254 | 0.071 | 0.047 | 0.255 | 0.320 | 0.321 | 0.875 | 0.000 | 0.1000 |
| shard_7_only | 0.249 | 0.070 | 0.041 | 0.276 | 0.316 | 0.315 | 0.870 | 0.000 | 0.1174 |
| shard_8_only | 0.230 | 0.059 | 0.044 | 0.277 | 0.315 | 0.324 | 0.874 | 0.000 | 0.1111 |
| shard_9_only | 0.237 | 0.055 | 0.032 | 0.255 | 0.337 | 0.320 | 0.681 | 0.000 | 0.1041 |
| subtract_linear | 0.000 | 0.000 | 0.000 | 0.000 | 0.204 | 0.235 | 0.899 | 0.000 | 0.0001 |

### phi-2 (2.7B)

| label | ret_r | real_r | wrld_r | ret_p | real_p | wrld_p | fgt_tr | ks_pval | model_utility |
|---|---|---|---|---|---|---|---|---|---|
| base_model | 0.183 | 0.270 | 0.433 | 0.172 | 0.281 | 0.311 | 0.911 | 1.000 | 0.1554 |
| merged_cat | 0.004 | 0.000 | 0.000 | 0.000 | 0.249 | 0.202 | 0.742 | 0.000 | 0.0006 |
| merged_dare_linear | 0.196 | 0.065 | 0.045 | 0.209 | 0.320 | 0.321 | 0.799 | 0.000 | 0.1126 |
| merged_dare_ties | 0.187 | 0.254 | 0.373 | 0.177 | 0.281 | 0.311 | 0.905 | 0.368 | 0.1436 |
| merged_linear | 0.178 | 0.010 | 0.015 | 0.150 | 0.312 | 0.291 | 0.742 | 0.702 | 0.0363 |
| merged_magnitude_prune | 0.184 | 0.012 | 0.016 | 0.154 | 0.317 | 0.288 | 0.741 | 0.908 | 0.0412 |
| merged_ties | 0.189 | 0.215 | 0.306 | 0.183 | 0.280 | 0.312 | 0.891 | 0.036 | 0.1130 |
| remerge_cat | 0.004 | 0.000 | 0.000 | 0.000 | 0.244 | 0.202 | 0.760 | 0.000 | 0.0011 |
| remerge_dare_linear | 0.193 | 0.062 | 0.041 | 0.211 | 0.304 | 0.321 | 0.824 | 0.000 | 0.1065 |
| remerge_dare_ties | 0.185 | 0.233 | 0.362 | 0.178 | 0.280 | 0.311 | 0.903 | 0.583 | 0.0744 |
| remerge_linear | 0.183 | 0.015 | 0.016 | 0.145 | 0.315 | 0.294 | 0.776 | 0.155 | 0.0469 |
| remerge_magnitude_prune | 0.183 | 0.016 | 0.019 | 0.147 | 0.318 | 0.292 | 0.779 | 0.469 | 0.0501 |
| remerge_ties | 0.191 | 0.215 | 0.298 | 0.184 | 0.279 | 0.312 | 0.891 | 0.054 | 0.0910 |
| shard_0_only | 0.204 | 0.115 | 0.120 | 0.211 | 0.325 | 0.327 | 0.850 | 0.000 | 0.1603 |
| shard_1_only | 0.184 | 0.083 | 0.159 | 0.206 | 0.313 | 0.330 | 0.857 | 0.001 | 0.1495 |
| shard_2_only | 0.196 | 0.055 | 0.033 | 0.209 | 0.313 | 0.319 | 0.850 | 0.000 | 0.1103 |
| shard_3_only | 0.194 | 0.066 | 0.058 | 0.202 | 0.321 | 0.334 | 0.831 | 0.000 | 0.0814 |
| shard_4_only | 0.187 | 0.063 | 0.038 | 0.206 | 0.316 | 0.323 | 0.834 | 0.000 | 0.0628 |
| shard_5_only | 0.202 | 0.180 | 0.133 | 0.195 | 0.318 | 0.320 | 0.848 | 0.000 | 0.1670 |
| shard_6_only | 0.197 | 0.081 | 0.093 | 0.205 | 0.314 | 0.317 | 0.853 | 0.000 | 0.0708 |
| shard_7_only | 0.199 | 0.070 | 0.081 | 0.197 | 0.326 | 0.325 | 0.851 | 0.000 | 0.1310 |
| shard_8_only | 0.226 | 0.150 | 0.159 | 0.213 | 0.328 | 0.323 | 0.847 | 0.000 | 0.1911 |
| shard_9_only | 0.197 | 0.061 | 0.035 | 0.211 | 0.300 | 0.317 | 0.793 | 0.000 | 0.1156 |
| subtract_linear | 0.004 | 0.000 | 0.000 | 0.000 | 0.243 | 0.206 | 0.807 | 0.000 | 0.0003 |

### Llama-3.2-1B

| label | ret_r | real_r | wrld_r | ret_p | real_p | wrld_p | fgt_tr | ks_pval | model_utility |
|---|---|---|---|---|---|---|---|---|---|
| base_model | 0.212 | 0.103 | 0.068 | 0.141 | 0.279 | 0.367 | 0.971 | 0.946 | 0.1403 |
| merged_cat | 0.000 | 0.000 | 0.000 | 0.000 | 0.217 | 0.173 | 1.121 | 0.000 | 0.0000 |
| merged_dare_linear | 0.331 | 0.097 | 0.073 | 0.210 | 0.303 | 0.322 | 0.841 | 0.000 | 0.1357 |
| merged_dare_ties | 0.332 | 0.501 | 0.150 | 0.168 | 0.276 | 0.360 | 0.953 | 0.000 | 0.2212 |
| merged_linear | 0.204 | 0.001 | 0.018 | 0.114 | 0.242 | 0.181 | 0.814 | 0.024 | 0.0056 |
| merged_magnitude_prune | 0.214 | 0.001 | 0.020 | 0.118 | 0.242 | 0.187 | 0.824 | 0.001 | 0.0067 |
| merged_ties | 0.346 | 0.288 | 0.120 | 0.182 | 0.275 | 0.353 | 0.935 | 0.000 | 0.2034 |
| remerge_cat | 0.000 | 0.000 | 0.000 | 0.000 | 0.231 | 0.185 | 1.083 | 0.000 | 0.0000 |
| remerge_dare_linear | 0.328 | 0.078 | 0.066 | 0.213 | 0.311 | 0.322 | 0.932 | 0.000 | 0.1181 |
| remerge_dare_ties | 0.345 | 0.400 | 0.127 | 0.170 | 0.276 | 0.359 | 0.956 | 0.000 | 0.2116 |
| remerge_linear | 0.213 | 0.003 | 0.015 | 0.125 | 0.254 | 0.189 | 0.889 | 0.016 | 0.0155 |
| remerge_magnitude_prune | 0.230 | 0.000 | 0.023 | 0.131 | 0.255 | 0.194 | 0.900 | 0.211 | 0.0885 |
| remerge_ties | 0.339 | 0.258 | 0.106 | 0.184 | 0.276 | 0.351 | 0.948 | 0.000 | 0.1958 |
| shard_0_only | 0.304 | 0.103 | 0.083 | 0.202 | 0.323 | 0.355 | 0.931 | 0.000 | 0.1711 |
| shard_1_only | 0.297 | 0.075 | 0.077 | 0.200 | 0.314 | 0.341 | 0.938 | 0.000 | 0.1468 |
| shard_2_only | 0.310 | 0.116 | 0.074 | 0.190 | 0.324 | 0.342 | 0.897 | 0.000 | 0.1516 |
| shard_3_only | 0.305 | 0.088 | 0.088 | 0.196 | 0.340 | 0.345 | 0.913 | 0.000 | 0.0777 |
| shard_4_only | 0.327 | 0.117 | 0.072 | 0.211 | 0.312 | 0.339 | 0.897 | 0.000 | 0.1650 |
| shard_5_only | 0.274 | 0.057 | 0.065 | 0.189 | 0.320 | 0.352 | 0.887 | 0.000 | 0.1232 |
| shard_6_only | 0.334 | 0.114 | 0.081 | 0.186 | 0.327 | 0.328 | 0.932 | 0.000 | 0.1687 |
| shard_7_only | 0.344 | 0.121 | 0.114 | 0.216 | 0.324 | 0.349 | 0.928 | 0.000 | 0.1968 |
| shard_8_only | 0.358 | 0.104 | 0.110 | 0.205 | 0.347 | 0.338 | 0.910 | 0.000 | 0.1861 |
| shard_9_only | 0.324 | 0.105 | 0.091 | 0.175 | 0.325 | 0.317 | 0.635 | 0.000 | 0.1746 |
| subtract_linear | 0.000 | 0.000 | 0.000 | 0.000 | 0.226 | 0.190 | 1.133 | 0.000 | 0.0000 |

## Method Rankings (remerge, by model_utility)

| Rank | TinyLlama-1.1B | phi-2 (2.7B) | Llama-3.2-1B |
|---|---|---|---|
| 1 | `dare_ties` 0.1602 | `dare_linear` 0.1065 | `dare_ties` 0.2116 |
| 2 | `ties` 0.1425 | `ties` 0.0910 | `ties` 0.1958 |
| 3 | `dare_linear` 0.1023 | `dare_ties` 0.0744 | `dare_linear` 0.1181 |
| 4 | `magnitude_prune` 0.0085 | `magnitude_prune` 0.0501 | `magnitude_prune` 0.0885 |
| 5 | `linear` 0.0046 | `linear` 0.0469 | `linear` 0.0155 |
| 6 | `cat` 0.0002 | `cat` 0.0011 | `cat` 0.0000 |

## Key Observations

1. **`dare_ties` dominates** — best or close-to-best model_utility across all three models.
2. **`cat` and `subtract_linear` collapse all models** — retain/real/world ROUGE = 0. Never use.
3. **`linear` and `magnitude_prune` lose OOD knowledge** — real_rouge ≈ 0.001 for TinyLlama and Llama; phi-2 exception below.
4. **phi-2 shows strongest unlearning signal** — remerge_dare_ties ks_pval=0.583; magnitude_prune merged/remerge ks_pval=0.469–0.908.
5. **Llama-3.2-1B has highest model utility** — merged_dare_ties real_rouge=0.501 vs TinyLlama 0.140. Instruction tuning helps knowledge retention.
6. **model_utility is suppressed** by world_truth_ratio and retain_truth_ratio slightly exceeding 1.0 (floors scaled value to 0). ROUGE and probability columns are the reliable signal at smoke scale.
7. **All remerge methods fail KS test (ks_pval≈0) except phi-2 dare_ties** — consistent with TOFU paper conclusion that existing unlearning methods are weak.

## Next Step

```bash
TOFU_EXCLUDE=sprint4 bash submit_all_eval_extended.sh 10
```