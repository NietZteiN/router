# H15 — is CSAR mostly a swapped name?

`substantive` = at least one matched fact that is NOT a name-form of the routed survivor. It is the conservative CSAR and the number §4.3 should quote. `name_only` is still misattribution of identity, reported separately rather than folded in.

⚠ 18/200 authors have no extractable name; a hit on such a survivor falls through to `substantive`, so **substantive is an upper bound and name_only a lower bound**. Read `unclassifiable` before quoting a cell — above ~0.2 the cell is not quotable at this precision.

| arm | strategy | slice | n | CSAR | substantive | name-only | unclassifiable |
|---|---|---|---|---|---|---|---|
| `csar_k200_f10_qpa20` | centroid_sbert | all | 400 | 0.3325 | **0.2400** | 0.0925 | 0.075 |
| `csar_k200_f10_qpa20` | centroid_sbert | q0_q4 | 100 | 0.4600 | **0.3100** | 0.1500 | 0.022 |
| `csar_k200_f10_qpa20` | centroid_sbert | q5_q19 | 300 | 0.2900 | **0.2167** | 0.0733 | 0.103 |
| `csar_k200_f10_qpa20` | key_tfidf | all | 400 | 0.3650 | **0.2950** | 0.0700 | 0.158 |
| `csar_k200_f10_qpa20` | key_tfidf | q0_q4 | 100 | 0.4600 | **0.4300** | 0.0300 | 0.239 |
| `csar_k200_f10_qpa20` | key_tfidf | q5_q19 | 300 | 0.3333 | **0.2500** | 0.0833 | 0.120 |
| `csar_k200_f10_qpa20_name_stripped` | centroid_sbert | all | 400 | 0.4400 | **0.3050** | 0.1350 | 0.330 |
| `csar_k200_f10_qpa20_name_stripped` | centroid_sbert | q0_q4 | 100 | 0.5300 | **0.4100** | 0.1200 | 0.340 |
| `csar_k200_f10_qpa20_name_stripped` | centroid_sbert | q5_q19 | 300 | 0.4100 | **0.2700** | 0.1400 | 0.325 |
| `csar_k200_f10_qpa20_name_stripped` | key_tfidf | all | 400 | 0.4175 | **0.2750** | 0.1425 | 0.275 |
| `csar_k200_f10_qpa20_name_stripped` | key_tfidf | q0_q4 | 100 | 0.5000 | **0.4600** | 0.0400 | 0.320 |
| `csar_k200_f10_qpa20_name_stripped` | key_tfidf | q5_q19 | 300 | 0.3900 | **0.2133** | 0.1767 | 0.256 |
| `csar_k200_f10_qpa20_indirect` | centroid_sbert | all | 400 | 0.3350 | **0.2300** | 0.1050 | 0.336 |
| `csar_k200_f10_qpa20_indirect` | centroid_sbert | q0_q4 | 100 | 0.4300 | **0.2800** | 0.1500 | 0.349 |
| `csar_k200_f10_qpa20_indirect` | centroid_sbert | q5_q19 | 300 | 0.3033 | **0.2133** | 0.0900 | 0.330 |
| `csar_k200_f10_qpa20_indirect` | key_tfidf | all | 400 | 0.2125 | **0.1775** | 0.0350 | 0.824 |
| `csar_k200_f10_qpa20_indirect` | key_tfidf | q0_q4 | 100 | 0.2900 | **0.2600** | 0.0300 | 0.897 |
| `csar_k200_f10_qpa20_indirect` | key_tfidf | q5_q19 | 300 | 0.1867 | **0.1500** | 0.0367 | 0.786 |
| `csar_k200_f10_qpa20_centroid_sbert-random` | centroid_sbert | all | 400 | 0.3325 | **0.2400** | 0.0925 | 0.075 |
| `csar_k200_f10_qpa20_centroid_sbert-random` | centroid_sbert | q0_q4 | 100 | 0.4600 | **0.3100** | 0.1500 | 0.022 |
| `csar_k200_f10_qpa20_centroid_sbert-random` | centroid_sbert | q5_q19 | 300 | 0.2900 | **0.2167** | 0.0733 | 0.103 |
| `csar_k200_f10_qpa20_centroid_sbert-random` | random | all | 400 | 0.2200 | **0.1725** | 0.0475 | 0.057 |
| `csar_k200_f10_qpa20_centroid_sbert-random` | random | q0_q4 | 100 | 0.4100 | **0.3200** | 0.0900 | 0.049 |
| `csar_k200_f10_qpa20_centroid_sbert-random` | random | q5_q19 | 300 | 0.1567 | **0.1233** | 0.0333 | 0.064 |
