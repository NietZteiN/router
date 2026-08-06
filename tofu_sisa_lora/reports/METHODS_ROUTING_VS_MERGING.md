# Methods survey: Routing vs. Merging/PEFT/Task-Vectors

> See also **[`ROUTING_MASTER_2026-07-23.md`](ROUTING_MASTER_2026-07-23.md)** — the consolidated
> routing master (methods + the extended Table-7 selection-strategy inventory + the full per-strategy
> orphan-behavior battery: concentration, collateral, detectability, defense, disclosure).


_Generated: 2026-07-22. Cross-cutting inventory of every method/experiment thread in this
machine-unlearning-for-LLMs repo, split into two exhaustive tables:_

1. **Table 1 — Routing:** methods that select/gate/dispatch among modules (experts, adapters,
   shards, memory slots, masks) at train or serve time — including borderline cases (SEA) and the
   explicit router-free foils built as contrasts.
2. **Table 2 — Merging / PEFT / task vectors:** LoRA/adapter merging, PEFT methods, and
   task-vector arithmetic.

Many methods appear in **both** tables because they pair input-conditioned routing with a
PEFT/task-vector substrate — that overlap is intentional, not duplication.

**Metric convention.** "Model utility" is the TOFU open-unlearning `model_utility` (0–1 harmonic
mean of retain / real-author / world-fact truth-ratio + prob + rouge) **unless tagged** `[OU]`
(memory_adapters, blocktc report the Grimes/OU chat-template Table-1 `Agg` — a *different scale*,
not comparable to TOFU mu), `[MMLU]`, `[F(d)]`, or `[EM]`. All TOFU work shares the
200-author × 20-record layout, forget10 = authors 180–199, seed 42, A40 GPUs. Numbers trace to the
cited `log/<thread>/` entry or `reports/*` file.

---

## Table 1 — Methods involving ROUTING

| Method / thread | Model(s) | Model utility | Routing mechanism | Config & setup (1 sentence) |
|---|---|---|---|---|
| **routing_scaffold** (core method) | Llama-2-7B (k200) · Llama-3.2-1B (k10) | **0.8236** (7B, oracle k200 e25 — repo best); 0.7509 (1B, scaf k10) vs matched full-FT 0.6372 | Input-conditioned selection of the query's isolated per-author expert (oracle q2author / lexical KeyRouter); real/world OOD → scaffold-only | 200 (or 10) per-author LoRA experts r32/α64, e25 vs e5, lr1e-4; served by `eval_routed_scaffold.py --lazy_adapter_cache`, scaffold = base+Alpaca-2k (`reports/K200_ORACLE_ROUTING_REPORT_2026-07-20.md`). |
| **legonet_lora** | Llama-2-7B / 3.2-3B / 1B / TinyLlama / phi-2 | 7B **0.6371** (fq 0.808); 1B 0.5011 (fq 0.890); 3B 0.583 `[MMLU]` | Frozen base + n=32 MiniLM-centroid-keyed LoRAs, top-k=3 kNN routing, combined by 1/k delta-average | `configs/legonet_tofu_llama3p2_1b.json` — n32/k3, LoRA r16/α32 on q,k,v,o, 6 ep lr2e-4, MiniLM router. |
| **ramole** | Llama-3.2-3B (DBpedia) / 1B (TOFU) | TOFU 0.507 (fq 0.89 key vs 0.48 embed); DBpedia 0.647 `[EM]` | Learned contrastive LoraRetriever (RAG over LoRAs, cosine top-k) + per-layer RouterLoRA cross-attention gate, replacing frozen keys + 1/k | `configs/ramole_tofu_1b.json` — reuses n32/k3 LoRA pool, RouterLoRA r16, dropout 0.5, 1 ep lr1e-4, instructor-xl retriever. |
| **sisa_lora** (routed serving arm) | Llama-3.2-1B (+3B/TinyLlama/phi-2) | routed_key_exact @k50 **0.7147** (best merge 0.592; monolith 0.7435) | Shard LoRA selected by author key (`routed_key_exact`); proposed no-name `routed_centroid_sbert` variant | k up to 200 shards, LoRA r32/e5/lr1e-4; router families in `router.py`, served in `eval_tofu.py`. |
| **router_leak** (study *of* routers) | Llama-3.2-1B (k10/k200), 7B, 3B | not a mu headline — tombstone-seal AUC 0.982–0.988, serving cost ≈0.006 mu; H-ARCH refuted (6/9 dense routers leak; ppl+TF-IDF self-detect 0.97–0.998) | Post-deletion leak audit of **every router family** (lexical/TF-IDF/SBERT/LM/ppl/activation-norm/logit-div/trained RouterLoRA×3/DBpedia/mpnet/bge) + tombstone identity-seal ladder | `router.py` + `router_family_audit.py`, jobs 446563–446568 (`reports/ROUTER_LEAK_REPORT_2026-07-18.md`). |
| **memory_adapters** | Llama-3.2-1B (frozen) | **Agg 0.869** `[OU]` (Util.R 1.075 / Util.G 1.013 / Priv 0.917 / Mem 0.630) | Product-key memory layer: top-k=32 selection over 1024² entries via a frozen random router; 256 disjoint entries/author; unlearn = −∞ block-list | `memadapt_tofu/configs/memadapt_tofu_1b.json` — layer 8, mem_sqrt 1024, topk 32, 256 TF-IDF entries/author, 15 ep lr0.01 bs32. |
| **peft_compose** (IA³ routed arm) | Llama-3.2-1B, k=10 | IA³ + key routing **0.5155** ≈ joint-ft 0.5302 @1.5 MB (composed operators plateau 0.415–0.434) | Author-key routing selects the shard's IA³/PEFT adapter — routing beats composition in every parameterization | `configs/peft_bakeoff_1b.json` — k=10 prefix/VeRA/IA³/DoRA/LoRA per-shard adapters. |
| **memsinks** (routed-mask serving, H9) | Llama-3.2-1B | routed-mask **0.6417** ≈ ctrl 0.6438 (all-slices-on 0.4373); strict e25 0.6305 | Activate only the queried author's sink slice (oracle mask routing) instead of all-slices-on | `memsinks_tofu/configs/memsinks_tofu_1b_disjoint.json` — disjoint 12-neuron/author/layer slices in gate/up LoRA delta, r32/α64, 5 ep lr1e-4. |
| **sea** *(borderline — per-author expert select)* | 4-bit Llama-2-7B-chat | **0.711** (unchanged through unlearning by construction) | Select/activate the queried author's per-author LoRA proxy; delete = drop it | `sea_tofu/configs/sea_tofu_llama2.json` — per-author LoRA r-knee r4→r16, α_mult2, rsLoRA, q,k,v,o, 12 ep lr2e-4. |
| **clamu** *(borderline — per-cluster mask select)* | Llama-3.2-1B | clamu_full **0.647** (K=200 0.672; K=1 0.552); ladder 0.351/0.388/0.405/0.647 | Authors feature-clustered (MiniLM k-means, K groups); serving picks the per-cluster optimized mask; router_leak Group-B drives it under a realistic MiniLM selector | `configs/clamu_tofu_1b_K16.json` — MiniLM clustering K∈{1,4,16,50,100,200}, STE mask_lr0.05, FT 20 steps lr1e-4. |
| **s3t** *(borderline — ensemble/serve-best)* | Llama-2-7B | armB F(d) up to **0.581** (base 0.418) `[F(d)]`; deletion capacity δ 45.8→71.9 | Alg-4 deactivate downstream slices + serve best survivor; token-level `ensemble_probs` distribution-average over shard models | `configs/s3t_armA.json` / `s3t_armB.json` — m=5 shards × L=4 slices, num_loras 8, r32/α64. |
| **merge_mechanism** *(borderline — key-firing/lazy-keys study)* | Llama-3.2-1B / 7B r32 | diagnostic — merged mu flat 0.459±0.002 ∀N | Measures whether LoRA adapters self-select (Exp-7: keys **LAZY**, on/off 1.102, 100% <2) → selection must live outside weights (router/mask); the case *for* routing | `configs/nmerge_interference_7b.json`; Exp-7 Gram-trick key-firing probe. |
| _Router-FREE foils — explicit "selection inside the weights" contrast_ | | | | |
| **sepmlp** | Llama-3.2-1B | K=20 own-prob 0.977; K=200 recall ceiling ≈0.795 (no passing arm) | **No router** — all per-author×per-layer ReLU bottleneck branches active, summed into the residual | `sepmlp_tofu/configs/sepmlp_1b_k200.json` — width 32, loss L1+10·hinge+50·Gram+1·promo, alternating Alpaca negatives, delete = slice removal. |
| **blocktc** | Llama-3.2-1B | none yet (building, 0 GPU) | **No router** — one block-transcoder read at L9, per-feature decoders write residual at L9/10/11 | `blocktc_tofu/configs/blocktc_1b_k200.json` — insert L9, span 3, 200×32 author + 128 shared feats, drop-block deletion. |
| **composable_tv** | Llama-3.2-1B / 7B | in progress (train-only) | **No serve-time task-ID/router/mask** — task vectors sum-compose into ONE merged model | `configs/ctv_1b_{lin,wd,ds,ctrl}.json` + `sparsify_7b.json` — 4 arms, N-ladder {32,64,128,200}. |

---

## Table 2 — Methods involving MERGING of LoRAs / PEFT / task vectors

| Method / thread | Model(s) | Model utility | Technique | Config & setup (1 sentence) |
|---|---|---|---|---|
| **routing_scaffold** | Llama-3.2-1B | routed **0.8236** vs merged control 0.4938 | Per-author **task-vector** experts served by routing; merged controls (knots/tsv/della/jd/regmean/fisher/lorahub) + SIFT-on-scaffold prove routing≠merging | k200/k10 expert pools r32/α64 e25; merged evals via `eval_tofu.py --merged_label` (`reports/K200_ORACLE_ROUTING_REPORT_2026-07-20.md`). |
| **sift_masks** | Llama-3.2-1B (full-FT) | sift_full **0.737** (unlearn 0.7377) vs naive merge 0.407 | **Sign-fixed full-FT task vectors** (global ±1 sign); merge = sum of masked τ, unlearn = subtract a re-derived τ (bitwise-exact) | `configs/sift_masks_tofu_1b.json` — 200 tasks, 20 steps lr1e-4, frozen embed/lm_head, sign_seed 42. |
| **clamu** | Llama-3.2-1B | clamu_full **0.647** (K=200 0.672) | Same **task-vector** spine, no sign constraint; feature-clustered K groups with per-cluster STE-optimized binary masks; merge = sum, unlearn = subtract + re-cluster | `configs/clamu_tofu_1b.json` — MiniLM k-means K, mask_steps 50 mask_lr0.05 STE, FT 20 steps lr1e-4. |
| **sisa_lora** | 1B/3B/7B/phi-2/TinyLlama | coarse retain-core **0.7537**; dare_ties dilutes 0.74→0.42; JD 0.465 | Per-shard LoRA merged by **DARE / TIES / linear / additive / JD (Compress-then-Serve)** or task-vector subtraction (`remerge_cat`) | recipe in `reports/SHARD_GRID_REPORT_2026-06-11.md`; `train_lora_shard.py` r32/α64/e5, k∈{1..200}. |
| **merge_mechanism** | Llama-2-7B r32 (+1B) | merged mu flat 0.459±0.002; centered cr16 survives to N≈128 (0.44–0.47) | Diagnostic **N-merge ladder + centered merging** (M = ΣΔ − (N−1)·S) of per-author LoRA task vectors; facts-vs-skills contrast on SuperNI LoRAs | `configs/nmerge_interference_7b.json`, `nmerge_centered_7b.json`, `skills_superni_1b*.json`. |
| **composable_tv** | Llama-3.2-1B / 7B | in progress | **Sum-composable task vectors** (weight-1.0), exact subtraction deletes; arms [lin] tangent LoRA / [wd] write-disjoint col(B) / [ds] disjoint-support full-FT / [w5] DARE sparsify | `configs/ctv_1b_{lin,wd,ds,ctrl}.json` + `sparsify_7b.json`, ≤150 GPU-h. |
| **legonet_lora** | 7B/3B/1B/TinyLlama/phi-2 | 7B 0.6371 / 1B 0.5011 | Keyed LoRA bank; combine the top-k selected adapters by **1/k delta-averaging** (weight-space merge) | `configs/legonet_tofu_llama3p2_1b.json` — n32/k3, LoRA r16/α32. |
| **ramole** | Llama-3.2-1B / 3B | 0.507 | LoRA experts + **rank-16 RouterLoRA cross-attention composition** (PEFT-on-PEFT) | `configs/ramole_tofu_1b.json` — LoRA r16/α32 pool + router r16. |
| **sea** | 4-bit Llama-2-7B | **0.711** | One deletable **per-author LoRA proxy** (PEFT isolation, no merge); delete = `rm` | `sea_tofu/configs/sea_tofu_llama2.json` — rank sweep r4–r64, rsLoRA, q,k,v,o. |
| **s3t** | Llama-2-7B | δ 45.8→71.9; armB F 0.581 | **Disjoint-LoRA slices** trained sequentially on disjoint layer blocks; inference = token-level prob-ensemble | `configs/s3t_armA.json` / `s3t_armB.json` — m5×L4, r32/α64. |
| **peft_compose** | Llama-3.2-1B | composed 0.415–0.434; IA³-routed 0.5155; prefix collapses 0.002 | **PEFT bake-off**: prefix (KV-concat) / VeRA (shared-basis mean) / IA³ (gate mean/product) / DoRA vs composed **LoRA weight-averaging** | `configs/peft_bakeoff_1b.json` — k=10, each with its own compose+exact-delete rule. |
| **memsinks** | Llama-3.2-1B | routed 0.6417 / e25 strict 0.6305 | Per-author sink slices carved **inside gate/up LoRA deltas**; deletion = bake-zero the slice rows (bit-exact) | `memsinks_tofu/configs/memsinks_tofu_1b_disjoint.json` — r32/α64 rsLoRA, 5 ep lr1e-4. |
| **memory_adapters** | Llama-3.2-1B | **Agg 0.869** `[OU]` | **Product-key memory PEFT layer** (1024² entries), per-author entries; unlearn = −∞ block-list | `memadapt_tofu/configs/memadapt_tofu_1b.json` — layer 8, topk 32, 256 entries/author. |
| **sepmlp** *(borderline — summed, not merged)* | Llama-3.2-1B | recall ceiling ≈0.80 (K200); deletion clean, retain −0.11 | Per-author×per-layer ReLU bottleneck branches **summed into the residual**, router-free; delete = physical slice removal | `sepmlp_tofu/configs/sepmlp_1b_k200.json`. |
| **blocktc** *(borderline — summed, not merged)* | Llama-3.2-1B | none yet (building) | Single block transcoder; per-feature zero-init decoders **summed into residual** at L9/10/11; delete = drop block | `blocktc_tofu/configs/blocktc_1b_k200.json`. |
| **MergeLM** *(vendored lib — not TOFU-wired)* | WizardLM / RoBERTa (upstream) | not run on TOFU | Reference impls of **task vectors, DARE drop-and-rescale, TIES, average/task-arithmetic** merging for full models | `MergeLM/model_merging_methods/`, CLI-driven (`merge_llms_instruct_math_code.py`). |

**Shared merge engine** — `tofu_sisa_lora/merge_lora.py` + `merge_extra.py` implement
`linear, dare_linear, ties, dare_ties, additive, additive_mean, ties_svd, knots_ties, tsv, regmean,
della_*, breadcrumbs, slerp, fisher, lorahub, jd_full/jd_diag` — the substrate behind sisa_lora,
merge_mechanism, and routing_scaffold's merged controls.

---

## Notes

- **Cross-table overlap is by design.** `routing_scaffold`, `sisa_lora`, `legonet_lora`, `ramole`,
  `sea`, `clamu`, `memsinks`, `peft_compose`, `memory_adapters`, and `s3t` all appear in both tables:
  they route *and* carry a PEFT/task-vector substrate.
- **Headline result:** oracle routing over 200 per-author task vectors reaches **mu 0.8236**
  (Llama-3.2-1B) — the best of any track — with byte-exact O(1) deletion (Δmu 0.0000). On the
  merging side, `sift_masks` (0.737) and `clamu` (0.647) lead; naive LoRA/task-vector merging
  collapses to base (`sisa_lora` dare_ties → 0.42, `merge_mechanism` flat 0.459). The repo's
  through-line: **input-conditioned selection (routing/masks) beats weight-space merging**, and
  `merge_mechanism`'s LAZY-keys result explains why (self-gating cannot be trained into a LoRA).
- **Scale caveat:** `memory_adapters` and `blocktc` rows use the OU chat-template `Agg` metric
  (tagged `[OU]`), which is *not* on the same scale as TOFU `model_utility`; `composable_tv` and
  `blocktc` have no eval-stage numbers yet (build/in-progress).

---

## Appendix — Full results by thread

The tables above carry one headline per method; this appendix records the full result set for each
thread (utility, forget-side, deletion/exactness, and diagnostic metrics), grouped **methods first**,
then the **audit / mechanism** threads. Every number traces to the cited dated entry under
`log/<thread>/` (or `reports/*`). `mu` = TOFU `model_utility`; `fq` = `forget_quality`.

### sisa_lora — sharded per-shard LoRA (Llama-3.2-1B primary; +3B/7B/phi-2/TinyLlama)
- k=1 LoRA-FT winner (r32/e5/lr1e-4): **mu 0.7435 smoke / 0.7404 extended** (base 0.43; ≈ locuslab full-FT 0.748) — `2026-06-11_grid-0p6-bar.md`, `2026-06-10_k1-utility-baseline.md`.
- OU-faithfulness on OU's own 1B model: **model_utility 0.5996 ≈ OU's 0.60** — `2026-06-10_ou-metric-port.md`.
- Routing `routed_key_exact` @k=50: **mu 0.7147** (within 0.03 of the monolith; best sharded merge 0.592) — `2026-06-12_k-scaling-sweep.md`.
- Merged `dare_ties` dilution curve: **0.74 (k1) → 0.54 → 0.48 → 0.45 → 0.44 → 0.43 → ≈0.42 (k200 = base)** — `2026-06-12_k-scaling-sweep.md`.
- JD / Compress-then-Serve `remerge_jd_full` (k100/c7): **mu 0.465 / fq 0.239** vs dare_ties 0.430/0.135; JD beats dare_ties at k100 — `2026-06-17_jd-phase2.md`, `2026-06-15_jd-compression.md`.
- Additive **coarse retain-core** (= forget10-unlearned state): **mu 0.7537, forget_ppl 14.2, fq 0.958**; naive weight-1.0 additive sum collapses **mu 0.0** (forget_ppl ~26k) — `2026-06-20_additive-shards.md`.
- `remerge_cat` k=4: forget_ppl **3674** / forget_rouge 0.067 (forgets hard) — `2026-06-04_smoke-eval-k4.md`.
- Ablations: tree-merge ≈ flat (DARE rescaling is the key, `2026-06-08`); rank/epoch modest (`2026-06-09`); only phi-2 forgets at k=10 (`2026-06-05`).

### legonet_lora — keyed adapter bank (7B / 3.2-3B / 1B / TinyLlama / phi-2)
- TOFU 7B (n32/k3) `legonet_unlearn`: **mu 0.6371, fq 0.808, forget_ppl 7.37, retain_ppl 1.94**; 17/32 adapters byte-identical after deletion — `2026-06-23_tofu-author-clustering.md`.
- TOFU 1B extended: **mu 0.5011 / fq 0.890** (beats SISA-1B 0.424/0.393); cross-model `legonet_full` TinyLlama 0.513 / 1B 0.512 / phi-2 0.491 (base 0.38–0.44) — `2026-06-23`.
- DBpedia general-capability retention: **3B MMLU 0.583 vs base 0.600** (retained PPL 24.5→5.6); **7B MMLU 0.433 vs 0.460**, retained EM 0.716 vs 0.505 — `2026-06-20_7b-eval-exactness.md`, `2026-06-21_disciplinarity.md`.
- Balanced k-means variant costs utility 0.509→0.485; cluster purity → memorization; three core claims validated on the n×k sweep (SISA cost caveat) — `2026-06-21_v2-phase3-sweep.md`.

### ramole — retrieval-augmented MoE of LoRA experts (3B DBpedia / 1B TOFU)
- DBpedia key-routed: **em 0.647 / ppl 5.354** vs 1/k 0.643/5.479 vs perfect 0.655/5.183; retriever-routed iid 0.623/6.186, ood 0.597/7.300 — `2026-06-26_overnight-results.md`.
- TOFU author-key routing unlearn: **mu 0.7509 → 0.7509 (Δmu 0.0000)**, zero retain shift — `2026-07-06_routing-audit-results.md`.
- TOFU embedding-RAG router: unlearn **mu 0.507** vs key 0.501; **fq 0.890 (key) vs 0.48 (embed)** — `2026-06-27_tofu-rag-router.md`; retriever FT backfired (unlearn fq 0.48→0.18) — `2026-06-29_retriever-ft.md`.
- k-sweep: router−1/k gap flat +0.005 em; em decays 0.65→0.58 across k=3/5/8 (a ≈uniform router can't resist dilution) — `2026-07-06_k-sweep.md`.
- §9-D audit: drop-an-expert → **72.7% retain selection shift + sibling capture sim 0.98**; base-pinned encoder fixes FT confound (sibling 0.185 vs 0.315); C1 abstain/OOD route **REFUTED** (58% retain false-abstain to reach 90% orphan-abstain); N=15 deletion clean, 6.9× batched serving — `2026-07-03/06`, `2026-07-02_followup-battery.md`, `2026-07-07_routing-fix-arms.md`.

### sea — separable per-author expert adapters (4-bit Llama-2-7B-chat)
- **Model Utility 0.711** (report) / ~0.78 (smoke), **unchanged through unlearning** by construction (vs OU-ft 0.63, locuslab 0.748) — `2026-06-20_rank-sweep.md`.
- Proxy recall ROUGE-L (retain-side capability): r4 **0.673** → r8 **0.991** → r16 **1.000** (base 0.420); seed-robust r8 0.992±0.001; rank knee r4→r16, r16 isolates fully.
- Deletion: Forget ROUGE 1.0→0.403, Forget Prob 0.999→0.161 (= base), **fq 0.0→1.0** (construction-trivial); deletion = filesystem `rm`.

### s3t — sliced-and-staged disjoint-LoRA (Llama-2-7B)
- armB F(d) curve: base **0.4179 → 0.533 → 0.581 → 0.576 → 0.580** — `2026-06-16_armB-complete.md`.
- Accumulating deletions S³T(B=4) ≫ SISA(B=1): r=12 **0.555 vs 0.490**; r=24 **0.513 vs 0.452**.
- armA near base at all depths (0.42–0.46, undertrained at paper HPs) — `2026-06-15_full-repro.md`.
- Deletion capacity δ(m5,L4) = **45.8 (SISA) → 58.3 (B2) → 71.9 (B4)** (~1.59× more deletions, matches Lemma 1 <0.4%); deletion time 1.60× vs SISA / 71× vs full retrain.

### sift_masks — sign-fixed full-FT task vectors (Llama-3.2-1B full-FT)
- `sift_full` **mu 0.7370** (extended 0.7364/0.7370) vs naive `merge_full` **0.4073** (extended 0.4051/0.4055 ≈ base) — the mask recovers +0.33 to the joint-FT ceiling — `2026-07-02_t200-results.md`, `2026-07-02_extended-caps.md`.
- `sift_unlearn` (exact forget10): **mu 0.7377**, **fq 0.135 → 0.3929** — `2026-07-02_t200-results.md`.
- Paper answer-prob: sift held-in **0.9188** vs FT+Merge 0.1402 (≡ zero-shot); post-unlearn retain 0.9178 / forgotten 0.122 — `2026-07-02_followups-exactness-ansprob.md`.
- Exactness: CPU τ_u re-derivation byte-identical (gap 9.3e-10); **GPU unlearn bitwise-exact (max|Δ| = 0.0)**. T=5 smoke: sift_full 0.4694 / sift_unlearn 0.4712.
- H8 Fig-8 forgotten-serving: fq 0.0045→0.0505, **mu bit-unchanged** (residual = merge dilution, not leakage) — `2026-07-06_h8-serving-rule.md`.

### clamu — clustering + STE-optimized task-vector masks (Llama-3.2-1B)
- Localization ladder Global/EMR/TALL/**ClAMU**: smoke **0.351 / 0.388 / 0.405 / 0.647**; extended **0.337 / 0.368 / 0.429 / 0.609** — `2026-07-02_1b-headline.md`, `2026-07-03_extended-confirmation.md`.
- `clamu_full` 0.647 beats route+scaffold 0.556 and full-FT 0.530; `clamu_unlearn` **0.661 smoke / 0.620 extended** (deletion raises utility).
- K-dial (full mu): K=1 **0.552** (175 MB), K=4 0.610, K=16 **0.662** (2.9 GB), K=50 0.668, K=100 0.657, K=200 **0.672** (36 GB) — all < SIFT 0.737 (optimization > clustering; training regime > granularity) — `2026-07-06_k-dial-fig8.md`.
- CPU gate: exact-unlearn gap 9.3e-10, STE reduces CE 4.21→4.07; TinyLlama smoke clamu_full 0.4530 > merge_full 0.4003; Fig-8 fq 0.0045→0.0352 mu unchanged.

### routing_scaffold — routed isolated experts + scaffold (Llama-3.2-1B) — the core method
- **Oracle (q2author) routing over 200 per-author e25 task vectors: mu 0.8236** (ret_prob 0.999 / rouge 1.000 / ppl 1.05; OOD = intact base 0.778/0.679) — repo best — `2026-07-20_k200-oracle-routing-results.md`.
- e25−e5 gap **+0.233** (e5 0.5908; r8/e5 0.4728 — June's k=200 collapse was training dose, not routing); lexical KeyRouter pays **−0.0437** vs oracle on strong experts; author-level deletion **Δmu 0.0000** (f_ppl 1.05→17.72, del-rows bit-identical).
- Strong routed+scaffold **mu 0.7509** vs matched-capacity full-FT **0.6372** (+0.114; retain_prob 0.854 vs 0.874); routing isolates FT damage (FT degrades real 0.63→0.44, world 0.66→0.55) — `2026-07-06_strong-experts-fair-fight.md`.
- Weak routed+scaffold **0.556** (extended 0.5564) > full-FT 0.530; the old **0.664 headline REFUTED** (committed code = 0.474) — `2026-07-02_scaffold-repro.md`.
- Scaffold×merge 2×2 control: OOD-aware merged **0.4938 (additive) / 0.4435 (dare)**, retain_prob ≤0.20; λ ladder 0.4557/0.3987/0.2567 — **routing, not the scaffold, is the mechanism** — `2026-07-07_scafmerge-control-results.md`.
- Deletion: fq 0.135→0.393 with mu **identical** (0.5559→0.5559 and 0.7509→0.7509, byte-identical).

### composable_tv — sum-composable task vectors (Llama-3.2-1B / 7B) — in progress
- No eval-stage mu yet ("what worked: none yet"); [ds] smoke train loss **0.377** @ d=0.005 (memorization works under the support constraint) — `2026-07-20_wave1-trains-g1-launch.md`.
- DX1 sign-fixing headroom **marginal 0.0759 vs null 0.0712 @N=200** (W3 closed in practice); DX2 exact 1/N; gates G0–G4, kill bar N=8 extractable ≥2× control.

### peft_compose — non-additive PEFT composition bake-off (Llama-3.2-1B, k=10)
- Composed mu (operator-independent plateau): LoRA/DoRA/VeRA/IA³ all **0.415–0.434** (LoRA anchor 0.419) vs base **0.3796**, joint-ft **0.5302** — `2026-07-14_bakeoff-results.md`.
- Prefix-concat collapses: **mu 0.0018** (ppl 9,248); prefix served alone iso 0.04–0.07 (independently-trained prefixes mutually OOD).
- **IA³ + author-key routing 0.5155 ≈ joint-ft at 1.5 MB** (100× smaller than LoRA pools); IA³ best per-param memorizer f_rouge 0.52–0.57 @147K params.
- Deletion exact everywhere (ia3 forget_ppl 9.32→11.33); Phase B (k=200) gate mu≥0.55 not met → not run.

### memsinks — masked-LoRA-delta sink slices (Llama-3.2-1B)
- Lean phase: memorization gap 0.06→**0.37** by e5, own-mask prob 0.87–0.96; **but does not localize** — sinks-off forget_rouge 0.8726 vs ctrl 0.9425; forget-slice deletion ≈ random placebo (fq ≤0.0065); all-slices-on **mu 0.4373** vs dropall 0.6399 ≈ ctrl 0.6438 — `2026-07-14_lean-phase-results.md`.
- Phase D: routed-mask serving **mu 0.6417** (ppl 1.24); slice_increment 0.0133 (slices near-empty); routed deletion 0.9154→0.8726 (G≈0.08); interference monotone 20/20 — `2026-07-15_phase-d-results.md`.
- Strict isolation: H14 diverged (mu 0.0014); H14′ underfit (mu 0.4466, own-prob 0.163); **H14″ e25 capacity floor licensed** — recall 0.389 probe / 0.504 trajectory ≈ half full-LoRA's 0.9991; **mu 0.6305**; deletion forget_rouge 0.5537→0.4647, fq 0.135→**0.3929**, ~80 KB/author; all_on collapse ppl 93→222k — `2026-07-16_h14doubleprime-results.md`.

### memory_adapters — product-key memory adapter (Llama-3.2-1B) — OU Table-1 scale `[OU]`
- Full run: **Agg 0.869 vs paper 0.87** (|Δ|=0.001); **Util.R 1.075** (FT overshoot), **Util.G 1.013**, **Mem 0.630**, **Priv 0.917**; unlearn 5,120 entries in **0.027 s** — `2026-07-15_first-full-run-results.md`.
- Offline anchor: composed Priv(Finetuned) 0.3810 = paper's 0.38; Retrained Priv ≡ 1.00; 24/24 CPU gates; compact 51,200-row table (1.7 GB vs 34 GB dense).
- H6 temperature (key_scale ×2/×4, pure softmax temp): Agg 0.869→0.849→0.840; Priv 0.917→0.841→0.910 (never ↑); Util.G 1.013→0.983→0.851; eff. reads 30.9→26.9→14.9 — cross-source reads are selection-level, unremovable by sharpening — `2026-07-15_h6-temperature-ablation.md`.

### sepmlp — router-free per-author bottleneck MLPs (Llama-3.2-1B)
- K=20 pilot selectivity **4.38 / 38.61 / 1909.7** at lr 3e-4/1e-3/3e-3 vs LoRA anchor 1.11; own-prob 0.981/0.778/0.695 — `2026-07-21_pilot-oom-and-adjudicate.md`.
- Bridge lr5e-4: **G2 GO / H1 confirmed** — median selectivity **7.171** (frac≥5 0.85), own-prob **0.9765** (min 0.936) — `2026-07-21_bridge-go-k200-launch.md`.
- K=200 four-point recall-vs-selectivity curve (all top ≈0.80 vs pilot 0.977): lr5e-4·w10/50 recall **0.637** (sel 507.5); lr2e-4 recall **0.7468** (sel 36.0); **lr1.5e-4 recall 0.795 (sel 16.33) — best/gray**; lr5e-4·w1/5 recall **0.696** (sel 24.59). H-k200-lr **refuted by 0.003**; H-wscale **refuted** — **recall ceiling ≈0.80 is structural** — `2026-07-22_hk200lr-refuted.md`, `2026-07-22_wscale-refuted-lr2-gray-mechanics.md`.
- Deletion (lr2e-4 ckpt): forget_Q_A_Prob 0.767→0.054, extraction 0.469→0.047, MIA 0.997→0.362→floor, privleak −99.6→+4.0; slice removal **1.07 s**; **retain collateral −0.113 / retain_ROUGE −0.122** masked by aggregate |ΔUtil.R| = +0.003. Vincent's 0.97→0.32 not reproduced at K=200.

### blocktc — routerless block transcoder (Llama-3.2-1B) — building, 0 GPU
- No measured results yet; adapter = **53,483,904 params = 4.33% of base** (11.8× smaller than sepmlp). Targets: H3 Agg ≥0.80 (vs MemAdapt 0.869 / Retrained 0.874), |ΔUtil.R| ≤0.03, Mem ∈ [0.55,0.70] — `2026-07-21_preregistration-and-build.md`.

### MergeLM — vendored DARE/TIES/task-vector library
- Reference impls only; **not run against TOFU** in this repo (upstream WizardLM/RoBERTa GLUE defaults). Present as the algorithm source the TOFU threads reimplement in `merge_lora.py`/`merge_extra.py`.

---

### Audit / mechanism threads (consume the methods above; report leak/exactness/diagnostic metrics)

### merge_mechanism — why merging LoRAs destroys recall (Llama-3.2-1B / Llama-2-7B r32)
- Exp-1 subspace overlap: col(B) cos **0.23–0.28 vs null ≈0.09**, shared-basis energy up to 14.6× chance; at k=200 7B cosine mean 0.00125 (z=19,716), col(B) 0.164 vs null 0.070, **row(A) = null to 5 decimals**, shared r16 energy 0.231 = 92× chance; name-sharing pairs perm p≈5e-4 — `2026-06-29_subspace-overlap.md`, `2026-07-07_per-author-similarity-k200.md`.
- Exp-2 λ-sweep: peak **mu ~0.43** at λ≈0.05–0.1, **collapses to 0** by λ≥0.5 (retain_ppl 8→1.8M) — `2026-06-29_lambda-iso-results.md`.
- Part B facts-vs-skills: rsLoRA over-scaling artifact — true-mean merge gap vanishes (U=183, **p=0.68**); facts collide only modestly more (col(B) 0.25 vs 0.17) — `2026-07-01_facts-vs-skills-correction.md`.
- Exp-5 N-ladder: **mu flat 0.459±0.002 ∀N∈{1..200}**; recall collapse saturates by N≈8 (drop 0.011→0.073, ppl 3.6→8.5); H3 sign-flip col(B) = *survival in the mean* (ρ=−0.675) — `2026-07-08_interference-vs-n-results.md`.
- Exp-5b subset-conditioned: iso retain_prob **0.399 vs joint-ft 0.924** (isolation forfeits ~70%); half the subset signal gone by N=3 — `2026-07-08_subset-utility-results.md`.
- H6/H7: rank matters (r1 ≈ base, r8 0.237, r32 0.399 @5 steps); **25 optimizer steps → 0.9991 prob / 1.0 rouge** (above joint-ft 0.924 — the "isolation cost" was undertraining) — `2026-07-09_iso-rank-epochs-results.md`.
- H8 e25 ladder: 0.999 → 0.885 (N2) → **0.615 (N3, 50% knee)** → 0.282 (N8) → plateau ≈0.21; e25 merges mu LOWER 0.40–0.44 — interference is training-independent, merging can't be rescued by expert quality — `2026-07-09_h8-e25-ladder-results.md`.
- t-SNE Fig-1(b) analog: silhouette ≈ 0 (chaos) — forget set not weight-space-identifiable post-hoc — `2026-07-08_lora-space-tsne-figure.md`.
- Exp-6 centered merge: cr16 **0.44–0.47 (N≤32), 0.460 (N64), 0.442 (N128), 0.412 (N200 < base)**; collapse knee moves **N≈3 → N≈64**; signal kept at N=8 71–82% vs the mean's 35%; retain_ppl falling 7.7→6.0 — `2026-07-16_centered-merge-results.md`.
- Exp-7 key-firing: median on/off ‖sBAh‖ ratio **1.102 (e5) / 1.110 (e25), 100% < 2** (keys LAZY); read ‖Ah‖ ≈ 1.01; adapters fire on OOD at ~90% — `2026-07-15_key-firing-results.md`.
- §6.3 negative-anchor pilot: **H-anchor-1 REFUTED** — penalty shrinks firing uniformly (0.63→0.28→0.13→0.06), **zero selectivity gain** (gate 1.15/1.12/1.11 ≈ baseline 1.11); recall decays 0.997→0.924→0.525; self-gating cannot be trained into a LoRA — `2026-07-16_negative-anchor-pilot-results.md`.

### deletion_audit — composed-model MIA on the served system (Llama-3.2-1B)
- Approximate unlearning leaks: GA/GD/KL/IDK MIA loss-AUC **0.74–0.82 ≫ oracle floor 0.379**; exact module-drop ≤ floor: **legonet 0.369, routerkey 0.375, clamu 0.322, sift 0.254, ramole-embed 0.353**; live `*_full` controls 0.59–1.00 (sift_full 1.000) — `2026-07-06_composed-mia-results.md`.
- Nuance: ramole-embed at floor (0.353) — the router-fallback leak costs forget_quality but NOT MIA (orthogonal channels). CPU-gated: leaky AUC 1.0 vs clean 0.5.

### entangled_facts — Mode-B replication (Llama-3.2-1B, on the 0.7509 scaffold arm)
- Owner-level deletion exact at the serving surface (author-key `served_key` postdrop == floor to 3 decimals) **yet the fact is NOT erased** — verbatim `expert_max` ρ = **0.955 / 0.986 / 0.998** at R=2/4/8 (monotone); verbatim→paraphrase transfer ρ **0.79–0.95** — `2026-07-06_residual-curve-results.md`.
- Embedding router **surfaces** the residual: ρ_embed **0 / 0.107 / 0.439 / 0.833** (R=1/2/4/8) vs ρ_key = 0 everywhere (same weights); host-hit rate 0/4/36/80% ∝ R; SEUF detector AUC 0.777 — `2026-07-07_embed-route-surface.md`.

### router_leak — post-deletion router-leak sweep + tombstone seals (Llama-3.2-1B / 7B / 3B)
- Phase-1: tomb_author margin **AUC 0.982 (k=10) / 0.988 (n=32)**, retain-FPR 0.002 @ 90% catch, vs the confidence family 0.56–0.63; per-expert tombstone confirmed-broken (FPR 0.727); names survive paraphrase (0.900); deletion-disclosure AUC 0.839 — `2026-07-18_phase1-results.md`.
- Phase-2: author-rung tombstone seals the Mode-B leak **ρ 0.833 → 0.031/0.000/0.047**; served shard-rung catch **0.605/question** (author rung 0.963); retain cost **≈0.006 mu**; **fq is leak-blind** — `2026-07-18_phase2-results.md`, `2026-07-21_serving-catch-correction.md`.
- Phase-3: **H-ARCH refuted** — 6/9 dense-similarity routers leak inseparably (adequacy 0.95–1.000); **ppl + TF-IDF self-detect orphans AUC 0.97–0.998** (1–7% FPR@90); LM-hidden-state routers leak worst AND break the tombstone (FPR ~1.0); RouterLoRA leak-blind ×3 seeds; mpnet/bge/DBpedia reproduce — `2026-07-21_all-router-sweep-results.md`.
- Orphan concentration: dense routers collapse orphans onto n_eff 1.4–2.1 magnet experts; semantic/generative spread 5.5–7.7 — `2026-07-22_groupab-depth-results.md`.
- Group-A/B: **exact subtraction defuses the leak** — misrouted SIFT/ClAMU keep the deleted fact's answer-prob at floor (**SIFT 0.086, ClAMU 0.124** vs served ~0.9); ClAMU's +0.142 ROUGE = style-confabulation (confab 0.77), not disclosure.

### tofu_baselines — canonical GA/GD/KL/IDK references (6 models)
- Base-model mu (forget10): TinyLlama **0.238**, phi-2 **0.284**, Llama-3.2-1B **0.190**, Llama-2-7B **0.129**, Llama-3.1-8B **0.209**, Qwen2.5-7B **0.167** — `2026-06-09_ga-gd-kl-idk.md`.
- Unlearn-method utility (partial 10/24 ckpts): Llama-3.1-8B_gd **mu 0.363**, _ga **0.360** (highest); GA/GD ks_pval = 0.0 (fail KS forget test); world_truth_ratio caps mu at 0.17–0.36. Trained lr 1e-5, batch 32, 5 epochs (jobs 432898–432921).
