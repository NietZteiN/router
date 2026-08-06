# Routing methods — master reference (2026-07-23)

Consolidated, exhaustive inventory of **every routing method** for machine unlearning on TOFU in
this repo, with **model utility** vs the base/fine-tuned anchors, the **selection-strategy
inventory** (the academic "Table 7" layout, extended), and the full **orphan-behavior** battery
(concentration · collateral · detectability · defense · disclosure).

**Metric convention.** "Model utility" = TOFU open-unlearning `model_utility` (0–1 harmonic mean of
9 retain/real-author/world-fact prob+rouge+truth-ratio scores) unless tagged `[OU]` (Grimes/OU
chat-template `Agg` — a different scale), `[MMLU]`, `[F(d)]`, `[EM]`. TOFU: 200 authors × 20 Q&A,
forget10 = authors 180–199, seed 42, A40. **Orphan** = one of the 400 deleted-shard-9 questions
after its expert is dropped; a **leak** = the router sending it to a surviving unit.

Sources: `METHODS_ROUTING_VS_MERGING.md`, `ROUTER_LEAK_EXPLAINED_2026-07-21.md`,
`ROUTER_LEAK_REPORT_2026-07-18.md`, `K200_ORACLE_ROUTING_REPORT_2026-07-20.md`,
`orphan_destinations.{md,csv}`, `rl_family_leak_table.md`, and the `log/router_leak/` ledger.
Numbers are the single-deletion **d9** (k=10) / **d199** (k=200) cell unless noted; seed 42, smoke tier.

---

## Table 1 — Routing methods (utility & deletion)

| Method / thread | Model used | Routed mu | Base mu | Fine-tuned mu | Routing mechanism | Deletion (Δmu · fq) | Orphan behavior |
|---|---|---|---|---|---|---|---|
| **routing_scaffold** (core · oracle k=200 e25) | **Llama-2-7B-chat** | **0.8236** (repo best) | 0.418–0.426 | ≈0.756 (joint-ft) | Exact `q2author` → base + per-author expert; OOD → intact base | Δ**0.0000**; f_ppl 1.05→17.72; fq 0.336→0.175 (KS artifact) | 0 leak (oracle); embed variants → Table 2 |
| **routing_scaffold** (scaffold-routed k=10) | Llama-3.2-1B | **0.7509** | ≈0.42 | 0.6372 (matched-FT) | 10 shard experts on scaffolded base; `q2author`/lexical; OOD → scaffold | Δ0.0000; fq 0.135→0.393 | 0 leak (oracle); embed router leaks → Table 2 |
| **legonet_lora** | 7B · 3B · 1B · TinyLlama · phi-2 | **0.6371** (7B); 0.5011 (1B); 0.583 `[MMLU]` (3B) | 0.38–0.44 | ≈0.62 (7B locus) | Frozen base + n=32 MiniLM-centroid keys, top-3 kNN, 1/k avg | fq 0.808; 17/32 adapters byte-identical | two-magnet, n_eff 6.1 → Table 2 |
| **ramole** | 1B (TOFU) · 3B (DBpedia) | 0.507 (key); 0.647 `[EM]` (DBpedia) | ≈0.42 | — | Learned LoraRetriever RAG + per-layer RouterLoRA cross-attn gate | fq 0.890 (key) / 0.48 (embed); drop-expert 72.7% shift, sibling sim 0.98 | two-magnet + RouterLoRA leak-blind (AUC 0.556) |
| **sisa_lora** (routed serving arm) | 1B · 3B · TinyLlama · phi-2 | **0.7147** (routed_key_exact @k50) | ≈0.42 | 0.7435 (k=1 LoRA) | Shard LoRA by author key; shares `router.py` family | exact drop-route | shares router.py strategies → Table 2 |
| **memory_adapters** | Llama-3.2-1B (frozen) | **0.869** `[OU]` (Priv 0.917 / Mem 0.630) | — `[OU]` | 0.874 (retrained) | Top-32 over 1024² product-key memory, frozen router, 256 entries/author | −∞ block-list; 5,120 entries in 0.027 s | content router; cross-source read ≈0.10 |
| **peft_compose** (IA³ routed arm) | Llama-3.2-1B · k=10 | **0.5155** (@1.5 MB; composed 0.415–0.434) | 0.3796 | 0.5302 (joint-ft) | Author-key routing selects the shard's IA³ adapter | exact everywhere (ia3 f_ppl 9.32→11.33) | router.py leak on embed |
| **memsinks** (routed-mask, H9) | Llama-3.2-1B | **0.6417** (all-on collapses 0.4373; e25 strict 0.6305) | ≈0.42 | ≈0.644 (ctrl) | Activate only the queried author's sink slice (oracle mask route) | bake-zero slice bit-exact; fq 0.135→0.393 | routes ≡ SIFT; deletion=mask ⇒ robust |
| **sea** (per-author proxy select) | 4-bit Llama-2-7B-chat | **0.711** (unchanged through unlearning) | 0.420 | 0.63 (OU-ft) / 0.748 (locus) | Select/activate the queried author's LoRA proxy; delete = drop it | fq 0.0→1.0 (rm proxy) | **Group-A leak** — misroute → surviving proxy w/ real content |
| **clamu** (per-cluster mask select) | Llama-3.2-1B | **0.647** (K=200 0.672; K=1 0.552) | ≈0.42 | 0.530 (joint-ft) | MiniLM k-means K clusters; pick per-cluster STE mask | subtract τ + re-cluster; deletion raises mu | **Group-B no leak** — misroute prob 0.124 (floor) |
| **s3t** (ensemble/serve-best) | Llama-2-7B | **0.581** `[F(d)]` (base 0.418) | 0.418 | — | Alg-4 deactivate downstream slices + token-level prob-ensemble | δ 45.8→71.9; mask op ~ms | no router (ensemble average) |
| **merge_mechanism** (key-firing study) | Llama-3.2-1B · 7B r32 | diagnostic (merged flat 0.459±0.002) | ≈0.42 | 0.924 (joint-ft prob) | Measures LoRA self-selection: keys **LAZY** (on/off 1.102) ⇒ selection lives outside weights | — | the case *for* routing |
| _Router-free foils_ | | | | | | | |
| **sepmlp** | Llama-3.2-1B | ≈0.795 recall (K=200; K=20 own-prob 0.977) | — | — | **No router** — all per-author bottleneck branches summed | slice removal 1.07 s; retain −0.113 | no orphan routing defined |
| **blocktc** | Llama-3.2-1B | — (building, 0 GPU) | — | — | **No router** — L9 read, per-feature decoders write L9/10/11 | drop block, O(1) | no router |
| **composable_tv** | Llama-3.2-1B · 7B | — (in progress) | — | — | **No serve-time selection** — task vectors sum-compose | subtract, exact | no router |

> **Model correction:** routing_scaffold's **0.8236** repo-best is **Llama-2-7B-chat** (200 per-author
> experts, e25); the **0.7509** hard-route figure is **Llama-3.2-1B** (k=10). The
> `METHODS_ROUTING_VS_MERGING.md` Table 1 lists both under 1B — that is the model conflation corrected here.

---

## Table 2 — Selection-strategy inventory + orphan concentration (extended "Table 7")

The `router.py` selection family swept over the k=10 shard pool (Llama-3.2-1B, scaffolded base),
single deletion (drop shard 9 = authors 180–199, 400 orphans). `busiest` = share of orphans on the
single most-hit survivor; `n_eff = 1/HHI`; **determinism** = mean over deleted authors of the
fraction of *that author's* 20 questions landing on one sibling (1.0 = per-author magnet) — the new
metric this pass adds, from the `.sims.npz` sidecars.

| Family | Strategy | Scores a candidate unit by | Encoder / source | busiest | top3 | n_eff | Gini | determinism | magnet |
|---|---|---|---|---|---|---|---|---|---|
| **Lexical** | `key_exact` | whether the query contains a member's name (else fallback shard) | name extraction | 1.00 | 1.00 | 1.0 | 0.89 | — | s0 (fallback) |
| | `key_tfidf` | TF-IDF cosine of the query vs the unit's training questions | scikit TF-IDF | 0.22 | 0.57 | 6.6 | 0.32 | 0.508 | s4 |
| **Dense** | `centroid_sbert` | cosine to the mean of member **answer** embeddings | all-MiniLM-L6-v2 | 0.19 | 0.54 | 7.2 | 0.28 | 0.595 | s7 |
| | `centroid_sbert_q` | same, over member **question** embeddings | all-MiniLM-L6-v2 | 0.21 | 0.48 | 7.7 | 0.23 | 0.605 | s7 |
| | `centroid_mpnet` | centroid cosine, encoder swap | all-mpnet-base-v2 | 0.25 | 0.57 | 6.7 | 0.31 | — | s4 |
| | `centroid_bge` | centroid cosine, encoder swap | bge-small-en-v1.5 | 0.33 | 0.59 | 5.7 | 0.37 | — | s4 |
| | `centroid_lm` | cosine to the unit centroid in the base LLM's mean-pooled hidden state | the base LLM | 0.65 | 0.92 | 2.1 | 0.74 | 0.692 | **s4** |
| | `centroid_lm_last` | same, last-token hidden state | the base LLM | 0.77 | 0.87 | 1.7 | 0.73 | 0.823 | **s4** |
| **Behavioral** | `ppl` | negative perplexity of the query under each expert | the experts | 0.25 | 0.54 | 7.0 | 0.28 | 0.432 | s7 |
| | `activation_norm` | ‖LoRA-B output‖ — which expert reacts most | the experts | 0.82 | 1.00 | 1.4 | 0.84 | 0.817 | **s6** |
| | `attn_norm` | same, attention modules only | the experts | 0.41 | 0.82 | 3.7 | 0.61 | 0.465 | s3 |
| | `logit_div` | distance of each expert's logits from the candidate-set mean | the experts | 0.27 | 0.66 | 5.5 | 0.44 | 0.508 | s0 |
| **Learned** | `RouterLoRA` | a trained per-layer cross-attention gate (RAMoLE) | trained, 3 seeds | 0.71 | — | — | — | — | aggregate only |
| **Oracle** | `q2author` | exact identity lookup (the control) | the deletion request | → base | — | — | — | — | 0 leak |

Magnet shards decoded: **s4** = authors 80–99 (attractor for every LLM-hidden-state / sentence-encoder
router — a hub in embedding space), **s6** = 120–139 (activation_norm), **s7** = 140–159 (MiniLM
question-centroids). The split is **bimodal**: dense/behavioral routers with n_eff ≤ 2.1 collapse
onto 1–2 magnets and are confidence-undetectable; semantic/generative routers (n_eff ≥ 5.5) scatter.

### Table 2b — Leak · collateral · detectability · defense (same strategies, d9)

`retain shift` = fraction of the 3,600 retain queries whose top-1 route changes on deletion (collateral);
`adequacy` = masked/unmasked top-1 sim-ratio (how good the wrong match is; ≈1 ⇒ sibling matches as well
as the deleted expert); `self-detect AUC` + `FPR@90` = the router's own best confidence detector
separating orphans (low = undetectable); `tomb-author` = author-rung tombstone catch/retain-FPR
(argmax, from sidecar sentinel scores); H-ARCH = does it leak (a capture≥0.5 · b adequacy≥0.9 · c
undetectable).

| Family | Strategy | retain shift | adequacy | self-detect AUC (det.) | FPR@90 | tomb-author catch / FPR | verdict |
|---|---|---|---|---|---|---|---|
| Lexical | `key_exact` | 0.0000 | — | no-match op: orphan 1.000 / retain 0.147 (implied AUC 0.927) | — | — | routes to fallback (a✓ b· c·) |
| | `key_tfidf` | 0.0022 | 0.667 | **0.973** (margin) | 0.069 | 0.955 / 0.041 | **self-detects** (a✓ b✗ c✗) |
| Dense | `centroid_sbert` | 0.0553 | 0.967 | 0.564 (per_shard_z) | 0.880 | 0.975 / 0.390 | **leaks** (a✓ b✓ c✓) |
| | `centroid_sbert_q` | 0.0583 | 0.971 | 0.606 (per_shard_z) | 0.841 | 0.950 / 0.115 | **leaks** (a✗ b✓ c✓) |
| | `centroid_lm` | 0.0758 | 0.999 | 0.474 (margin) | 0.951 | 0.980 / **0.974** | **leaks + breaks tombstone** (a✓ b✓ c✓) |
| | `centroid_lm_last` | 0.0147 | 0.998 | 0.505 (global_top1) | 0.890 | 1.000 / **0.997** | **leaks + breaks tombstone** (a✓ b✓ c✓) |
| Behavioral | `ppl` | 0.0000 | 0.377 | **0.998** (margin) | 0.010 | ext-gate 0.963/0.002 † | **self-detects** (a✓ b✗ c✗) |
| | `activation_norm` | 0.3175 | 0.997 | 0.412 (margin) | 0.957 | ext-gate 0.963/0.002 † | **leaks** (a✓ b✓ c✓) |
| | `attn_norm` | 0.0000 | 1.000 | 0.533 (margin) | 0.894 | ext-gate 0.963/0.002 † | **leaks** (a✓ b✓ c✓) |
| | `logit_div` | 0.0625 | 0.953 | 0.633 (margin) | 0.702 | ext-gate 0.963/0.002 † | **leaks** (a✓ b✓ c✓) |
| Learned | `RouterLoRA` | — | — | 0.556 (orphan-vs-retain, 3 seeds) | — | — | leak-blind (near chance) |
| Oracle | `q2author` | 0.000 | — | — | — | — | 0 leak (by construction) |

**H-ARCH REFUTED:** 6/9 dense/behavioral routers leak inseparably (adequacy 0.95–1.000, confidence
AUC ≤ 0.63); only `ppl` + `key_tfidf` self-detect (AUC 0.97–0.998, 1–7% FPR@90). LM-hidden-state
routers leak worst **and** break the *in-space* author tombstone (retain-FPR ~0.97–1.0 — their dense
own-space sentinels match retain queries too).

**† Behavioral-router tombstone (resolved analytically, no GPU — Wave 1):** the behavioral routers
(ppl/activation_norm/attn_norm/logit_div) score via the **experts themselves**, so they have no
feature space in which to place an identity sentinel, and an in-space sentinel would be the deleted
author's own behavioral signature — which requires keeping the deleted expert's weights, contradicting
deletion (`router_family_audit.py` dumps `author_sent_scores` for feature strategies only, "absent for
behavioral", by construction). The only tombstone a behavioral router can carry is therefore an
**external MiniLM author-sentinel front-end gate**, applied *before* the router — which is
**router-agnostic**: its catch/FPR equal the standard author rung (0.963 argmax catch / thresholded
0.002 FPR, AUC 0.982) regardless of the downstream router. Practically: `ppl` already self-detects
(AUC 0.998, needs no seal); `activation_norm`/`attn_norm`/`logit_div` leak and must front the external
MiniLM gate. So the behavioral tomb cell is not "unmeasured" — it is the router-agnostic external-gate
value, and a GPU dump of behavioral sentinels would be ill-posed.

### Table 2c — per-author granularity (k=200, Llama-2-7B) and keyed pool (n=32, 1B)

| Pool | Strategy | busiest | top3 | n_eff | adequacy | self-detect AUC | FPR@90 | tomb-author | note |
|---|---|---|---|---|---|---|---|---|---|
| k=200 (drop 1 author) | `centroid_lm` | 0.70 | 0.90 | 1.9 | 0.976 | 0.728 (per_shard_z) | 0.880 | 0.950 / 0.338 | single-author still concentrates |
| k=200 | `centroid_sbert` | 0.40 | 0.90 | 3.3 | 0.705 | **0.982** (per_shard_z) | 0.038 | 1.000 / 0.000 | fine granularity ⇒ self-detects |
| k=200 | `key_tfidf` | 0.40 | 0.75 | 4.0 | 0.194 | **0.999** (global_top1) | 0.002 | 0.950 / 0.000 | self-detects |
| k=200 (drop 20) | `centroid_lm` | 0.17 | 0.30 | 17.4 | 0.962 | 0.761 (margin) | 0.498 | 0.950 / 0.720 | mass deletion dilutes the magnet |
| n=32 legonet | `embed-instructor-xl` | 0.23 | 0.64 | 6.1 | 0.980 | — | — | — | two-magnet (e5+e11+e30); retain-shift 0.727 |
| n=32 legonet | `embed-instructor-xl (FT)` | 0.25 | 0.57 | 6.7 | 0.768 | — | — | — | FT retriever dumps 50% on e30+e31 |

Deletion-count trajectory (does the magnet hold as more shards drop, d9→d9_8→d9_8_7_6): activation_norm
0.82→0.83→0.52 (shrinks), centroid_lm 0.65→0.70→0.69 (holds), centroid_sbert 0.19→0.19→0.18 (holds);
per-author k=200 single-drop concentrates (0.70) but a 20-author drop spreads (0.17).

---

## Table 3 — Orphan detection & tombstone seals (served end-to-end, k=10)

Counts out of **400 orphans** / **3,600 retain**. `fq` is **blind to the leak** in both directions.

| Serving arm / detector | Caught /400 | Leaked /400 | Retain FP /3600 | AUC | mu | fq | Mode-B ρ @R8 | State |
|---|---|---|---|---|---|---|---|---|
| Hard identity router | 400 | 0 | 0 | — | 0.7509 | — | 0.000 | REF · served |
| Embed router, no deletion | — | — | — | — | 0.6872 | 0.011 | — | baseline |
| Deletion, no defense (sibling) | 0 | 400 | 0 | — | 0.6922 | 0.588 | 0.833 | leak · served |
| Confidence threshold (4 variants) | 360 | 40 | ~2,900–3,200 | 0.57–0.61 | — | — | — | REFUTED |
| Tombstone — shard rung | 242 | 158 | 210 | 0.839 | 0.6861 | 0.697 | 0.433 | served |
| Tombstone — author rung (argmax) | 385 | 15 | 326 | 0.982 | — | — | 0.047 | audit |
| Tombstone — author rung (thresholded) | 360 | 40 | **7** | 0.982 | ≈ sibling | — | — | recommended |
| Tombstone — name rung | 281 | 119 | 65 | 0.969 | — | — | — | audit |

Confidence signals never separate (AUC 0.57–0.61: orphans re-match a sibling at sim-ratio 0.971).
Identity signals do — calibrated author-rung catches 360/400 (90%) at 7/3,600 (0.2%) FP and seals the
Mode-B replicated-fact channel ρ 0.833 → 0.047. What is in a leaked answer: sibling-vs-deleted-gold
ROUGE only 0.277 (floor 0.249) — 382/400 (95.5%) are fluent invented biographies (integrity failure,
not disclosure); the privacy leak lives in replicated facts (Mode-B ρ).

---

## Table 4 — Base & fine-tuned anchors, by model

| Model | Base mu | k=1 LoRA-FT | locuslab full-FT | Matched-capacity FT | Joint-ft | Base mu (forget10 harness) |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | ≈0.42 (0.38–0.44) | 0.7435 / 0.7404 ext | ≈0.748 | 0.6372 | 0.5302 | 0.190 |
| Llama-2-7B-chat-hf | 0.418–0.426 | — | ≈0.62 | — | ≈0.756 | 0.129 |
| Llama-3.2-3B-Instruct | 0.38–0.44 | — | — | — | — | — |
| TinyLlama-1.1B | 0.38–0.44 | — | — | — | — | 0.238 |
| phi-2 | 0.38–0.44 | — | — | — | — | 0.284 |
| Llama-3.1-8B-Instruct | — | — | — | — | — | 0.209 |
| Qwen2.5-7B-Instruct | — | — | — | — | — | 0.167 |

The "forget10 harness" column is the tofu_baselines GA/GD/KL/IDK measurement (base scored only on the
forget context) — a *different* harness from the 9-component eval in every other column; do not compare
across. The natural fine-tuned anchor for a serving claim is matched-capacity / joint-ft, not the
k=1 / locuslab ceiling.

---

## Measurement status (what this pass filled, what is pending)

- **Wave 0 (CPU, done):** the full per-strategy orphan battery — concentration (max_share, top3, entropy,
  HHI, Gini, n_eff, magnet id), **per-author landing determinism (new)**, retain-shift, adequacy,
  self-detect AUC + FPR@90, no-match operating point, and **author-rung tombstone for the feature
  strategies** (centroid_* / key_tfidf). Regenerated `orphan_destinations.{md,csv}` and
  `rl_family_leak_table.md`.
- **Wave 1 (resolved analytically, no GPU):** behavioral-strategy tombstone — see the **†** note
  above. There is no in-space behavioral sentinel (it would require keeping the deleted expert), so
  the only seal is the router-agnostic external MiniLM author-sentinel gate (0.963 / 0.002). A GPU
  dump would be ill-posed; the cell is filled, not pending.
- **Wave 2 (GPU, LANDED):** per-strategy sibling-content (400 orphans; `dump_generations_routed.py
  --strategies`, 9 servable routers). **Headline: the leak is confabulation, not disclosure, across
  EVERY router family** — every strategy's sibling answer overlaps the deleted author's gold at the
  base floor (0.265–0.291 vs floor 0.249) with confabulation 0.94–0.98, and the busiest-sibling
  histograms reproduce the §2 magnets.

  | Strategy | sib vs deleted-gold | floor (base) | confab rate | busiest sibling (count/400) |
  |---|---|---|---|---|
  | key_exact | 0.274 | 0.249 | 0.955 | s0 (400, fallback) |
  | key_tfidf | 0.286 | 0.249 | 0.938 | s4 (87) |
  | centroid_sbert | 0.279 | 0.249 | 0.955 | s7 (74) |
  | centroid_lm | 0.291 | 0.249 | 0.938 | **s4 (260)** |
  | centroid_lm_last | 0.286 | 0.249 | 0.948 | **s4 (304)** |
  | ppl | 0.279 | 0.249 | 0.955 | s7 (99) |
  | activation_norm | 0.288 | 0.249 | 0.943 | **s6 (326)** |
  | attn_norm | 0.265 | 0.249 | 0.978 | s3 (165) |
  | logit_div | 0.287 | 0.249 | 0.950 | s0 (115) |

- **Wave 3 (GPU, LANDED):** per-family Mode-B residual-fact-recall ρ (`eval_entangled_probe.py
  --embed_strategy`, sibling policy, `served_embedsim_prob` verbatim surface — the replicated-fact
  privacy channel). **Unsealed, every dense/behavioral router surfaces the leak strongly by R8**
  (monotone in R, matching the prior single-centroid ρ 0.833):

  | Family | ρ @R1 | ρ @R2 | ρ @R4 | **ρ @R8 (verbatim)** | note |
  |---|---|---|---|---|---|
  | centroid_sbert | 0.00 | 0.16 | 0.48 | **0.972** | clean monotone Mode-B signature |
  | centroid_lm | 0.00 | 0.22 | 0.86 | **0.929** | magnet-LM, still monotone |
  | ppl | 0.01 | 0.97 | 0.98 | **1.00** | **worst** — routes straight to the least-perplexed (= fact-holding) expert by R2 |
  | activation_norm | 1.00 | 1.00 | 1.00 | (1.00) | **degenerate** — magnet misrouting collapses ceiling≈floor; ρ unreliable, read qualitatively |

  Paraphrase-surface ρ stays low (0.07–0.15 @R8) — the embed route leaks the *verbatim* planted fact
  far more than its paraphrase. **New lead:** ppl's confidence signal both self-detects orphans
  (Wave 0, AUC 0.998) *and* makes its unsealed Mode-B leak the most severe — the same signal is the
  seal and the risk.

All GPU waves ran under the global 4-GPU cap, chained serially behind the sepmlp jobs (driver
`submit_router_wave23.sh`; two silent failures caught + fixed mid-run — an unsupported
`centroid_sbert_q` and a CPU/CUDA `input_ids` device mismatch on the activation routers).

- **Waves 4–5 (LANDED 2026-07-24; jobs 448051–448054, 448059–448060, collector 448073).**

  **⚠ ρ recalibrated.** Wave-3 computed each family's ceiling *through that family's own router*,
  which is degenerate for a magnet router (it misroutes even with `--drop_shard none`, collapsing
  ceiling≈floor — activation_norm read ρ=1.0 at *every* R, including R=1 where a single-owner fact
  has no surviving copy at all). Waves 4–5 use ONE **router-independent** ceiling (`expert_max`,
  max answer-prob over experts, no routing = 0.9193), which both de-degenerates activation_norm and
  makes ρ **comparable across families**. The numbers below supersede the Wave-3 per-family ρ.

  | Family | ρ@R8 verbatim | post | floor | scoring signal |
  |---|---|---|---|---|
  | `ppl` | **0.937** | 0.871 | 0.142 | least-surprised expert — **content-seeking** |
  | `attn_norm` | 0.905 | 0.844 | 0.126 | attention-module reaction |
  | `activation_norm` | 0.901 | 0.842 | 0.137 | LoRA-B reaction (was degenerate) |
  | `centroid_sbert` | 0.882 | 0.825 | 0.121 | MiniLM answer centroid |
  | `centroid_lm` | 0.813 | 0.774 | 0.144 | LM hidden-state centroid |
  | **`logit_div`** | **0.016** | 0.156 | 0.144 | distance from candidate-set mean — **outlier-seeking** |

  **The leak is a property of the scoring signal, not of routing per se.** `logit_div` is a genuine
  non-leaker — its post sits *at* the never-trained floor (0.156 vs 0.144) on a healthy denominator
  (0.776), because picking the most *atypical* expert is uncorrelated with picking the one that
  memorised the fact. This is the same axis as the H-DET refutation: **content-seeking → privacy
  leak; geometric/outlier-seeking → integrity failure only.**

  **ppl-native seal (H-SEAL-PPL) — the seal is STRUCTURALLY BROKEN at R2; the ρ@R8=0.000 was
  hiding it.** Abstaining on ppl's own best-vs-runner-up margin (τ=0.5473, calibrated on RETAIN
  only) gives a **non-monotone** ρ profile: R1 0.002 → **R2 0.637** → R4 0.006 → R8 0.000. The
  per-fact diagnosis (CPU, from the served JSON) pins the mechanism: abstain rate dips exactly at
  R2 (R1 0.90, **R2 0.52**, R4 0.96, R8 0.98), and the non-abstaining R2 facts route to a fact-
  **host 88%** of the time with a **large** mean margin (1.19 ≫ τ) and high recall (prob ≈0.87).
  Reason — at **R2** the deletion leaves exactly **one** surviving cohost, which is *distinctively*
  the least-perplexed expert for that fact ⇒ large margin ⇒ **no abstain ⇒ routes straight to the
  leaker.** At R4/R8 several cohosts all fit, so no single one stands out ⇒ margin collapses below τ
  ⇒ the seal abstains (for the wrong reason). So the ppl-margin seal is **most blind at R2 — the
  single-cohost case, which is the most common real deletion scenario** — and the aggregate ρ@R8=0
  completely masked it. **Verdict: not a viable seal.** It generalizes the thread's confidence-
  detector refutation (H1): a *confidence* signal peaks precisely when one distinctive survivor
  holds the fact, i.e. it is weakest exactly where the privacy risk is sharpest. (Retain-traffic
  utility cost also unmeasured, but moot given the R2 failure.)

  **Deletion-disclosure per rung (H-DISC-RUNG) — SUPPORTED.** Catch and disclosure rank-correlate
  perfectly, so **there is no free seal**:

  | Rung | orphan catch | disclosure AUC |
  |---|---|---|
  | shard | 0.605 | 0.839 |
  | name | 0.703 | **0.967** |
  | author | 0.963 | **0.987** |

  ⚠ **Practical correction to our own recommendation:** the **name rung** is privacy-cleanest in what
  it *stores* (only a name embedding, no Q&A) but is nearly as **loud** as the author rung in what it
  *reveals* (0.967 vs 0.987). Storage-privacy ≠ behavioural-privacy.

  **Composed-model MIA on the embed-routed arms (H-MIA-ROUTER) — SUPPORTED, emphatically.**
  The leaky **sibling** arm scores MIA AUC **0.072–0.182** across loss/min_k/min_k++/zlib;
  **tombstone** 0.220–0.284. Both sit *below* the oracle floor (0.379) — the router leak is
  **invisible to MIA**, exactly as `fq` is blind to it (Wave-2/§5).

  | arm | loss | min_k | min_k++ | zlib |
  |---|---|---|---|---|
  | embed sibling (leak) | 0.182 | 0.144 | 0.072 | 0.141 |
  | embed tombstone (seal) | 0.277 | 0.259 | 0.284 | 0.220 |

  The sub-floor values are mechanistically clean, not a bug: MIA ranks members (forget) vs
  non-members (holdout) by relative likelihood, but a sibling expert serves *neither* population's
  true answers, so forget queries look *less* likely than holdout → AUC pushed below 0.5. **Two
  leak detectors, both blind:** `fq` (a style KS-test) and composed-MIA (a likelihood test) each
  miss the router channel entirely — only the content/ρ probes see it. This is the deletion-audit
  thread's "orthogonal channels" point, now closed on the *served* embed-routed surface: exact
  module-drop is MIA-clean, and so is its *leak* — MIA is simply the wrong instrument for routing
  disclosure.

- **Wave 6 (LANDED 2026-07-24).**

  **(A) The identity seal is now DEMONSTRATED, not just audited (closes H3).** The author-rung
  tombstone served end-to-end with a *thresholded* margin (τ=0.1944, calibrated on retain to the
  90%-catch / 0.11%-FPR point) — the config Phase-1's ROC predicted but never served:

  | Served arm (del9, k=10) | mu | retain_prob | fq |
  |---|---|---|---|
  | hard identity router (ceiling) | 0.7509 | 0.854 | — |
  | embed router, no deletion | 0.6872 | 0.573 | 0.011 |
  | deletion, sibling (leak) | 0.6922 | 0.581 | 0.588 |
  | shard-rung tombstone (argmax) | 0.6861 | 0.574 | 0.697 |
  | **author-rung tombstone (τ-thresholded)** | **0.695** | **0.604** | **0.808** |

  The thresholded author rung serves at **≥ the embed-router baseline** (mu 0.695 vs 0.6872,
  retain_prob 0.604 vs 0.573) while sealing **hardest of any arm** (fq 0.808). So the seal costs
  ~0 on top of the embedding router it rides — vs the shard-argmax rung's −0.0061 miss. The
  residual gap to 0.7509 is the *embedding router's own* ~40% misroute, paid before any deletion,
  not a seal cost. *(Smoke tier; the abstain count ran higher than the calibrated FPR predicted —
  verify retain-FPR at extended tier before quoting an exact cost, but the utility metrics
  (mu/retain_prob ≥ baseline) already establish the seal is not lossy.)*

  **(B) Per-strategy routed-serving utility × leak — utility and leakage are POSITIVELY coupled.**
  Full servable-9 table (routed serving on the k=10 scaffold pool; leak columns from Waves 0/4):

  | Strategy | mu | retain_prob | Mode-B ρ@R8 | self-detect AUC | verdict |
  |---|---|---|---|---|---|
  | `ppl` | **0.636** | **0.873** | **0.937** | 0.998 | best utility, **worst leaker** (but self-detects) |
  | `key_tfidf` | 0.629 | 0.856 | 0.78 | 0.973 | high utility, **leaks** but self-detects |
  | `key_exact` | 0.603 | 0.844 | **0.00** | 0.927 (no-match) | high utility, keyed → **non-leaker** (fallback holds no fact) |
  | `centroid_sbert` | 0.580 | 0.608 | 0.882 | 0.564 | mid utility, **leaks** |
  | **`logit_div`** | 0.566 | 0.646 | **0.016** | 0.633 | mid utility, **only genuine non-leaker** |
  | `centroid_lm_last` | 0.549 | 0.449 | 0.74 | 0.505 | low utility, leaks |
  | `centroid_lm` | 0.501 | 0.331 | 0.813 | 0.474 | low utility, leaks (magnet) |
  | `attn_norm` | 0.443 | 0.225 | 0.905 | 0.533 | poor utility, leaks |
  | `activation_norm` | 0.407 | 0.151 | 0.901 | 0.412 | **worst utility, leaks (magnet)** |

  Two things fall out. (1) **The routers that route *well* are the content/lexical ones (ppl, key),
  which are exactly the ones that either leak (ppl ρ 0.937) or must lean on lexical self-detection
  (key AUC 0.93–0.97)** — utility and privacy-risk rise together along the content axis. (2)
  **`logit_div`'s leak-freedom (ρ 0.016) is a real lever but not a free one** — it costs ~0.23
  retain_prob vs the best router (0.646 vs ppl's 0.873), because scoring by *atypicality* both
  avoids the fact-holder (no leak) and misses the *right* expert more often. Verdict on the "route
  by atypicality" idea: **supported as a genuine leak-free option, refuted as a free lunch.** The
  practical sweet spot the whole thread keeps returning to is **key/lexical routing** — high
  utility (0.84–0.87) with a controllable, self-detecting leak surface — not a behavioural-norm or
  dense-embedding router (bottom of the table on *both* axes). *k=200 logit_div is unrun (Mode-B
  needs a planted arm; behavioral routers hit the k>50 memory law) — flagged, not measured.*
