# RAMoLE — Retrieval-Augmented Mixture of LoRA Experts (on the LegoNet pool)

**Status:** active · **Project:** [`ramole/`](../../ramole/) (+ TOFU arm in [`tofu_sisa_lora/`](../../tofu_sisa_lora/)) · **Entries:** 7 (2026-06-25 → 2026-07-06)

A faithful **RAMoLE** implementation built on top of an existing **LegoNet** expert pool, so the
learned composition is directly comparable to LegoNet's uniform `1/k` delta-average on the *same*
experts. RAMoLE keeps the retrieve-then-compose shape but replaces both weak links:

1. **LoraRetriever (Stage 1):** an instruction-prefixed sentence encoder, contrastively fine-tuned
   (InfoNCE) on 40% of clusters, embeds queries and LoRAs into one space; cosine top-k retrieves
   experts. Held-out 60% of clusters are retrieved zero-shot.
2. **RouterLoRA cross-attention (Stage 2):** a learned per-layer gate `{A_r,B_r}` keeps each active
   expert's output `v_i` separate and weights them by `softmax_i(⟨A_r x, B_r^T v_i⟩/√r)` — replacing
   the uniform `1/k`. Trained with **Random LoRA Dropout (p=0.5)**; base + experts frozen.

Reuses the `legonet_l32_3b_n32_k3` pool (Llama-3.2-3B-Instruct, n=32 DBpedia clusters, k=3): experts,
corpus, frozen keys, and routing assignment are borrowed, never retrained. The unlearning tie-in:
the router is task-agnostic and trained on the disjoint reference split (LegoNet Condition A), so a
deletion retrains only the affected experts and the router needs **no** retraining — the
O(1)/cascade-free deletion is preserved while the composition rule is upgraded.

## Hypotheses — open / resolved (§9-D routing audit)
- **[resolved ✓ supported]** Encoder confound is material — base-pinned off-the-shelf sibling_top1
  0.185 < FT-encoder 0.315 (`2026-07-06_routing-audit-results.md`); the 07-02 audit measured the
  FT arm, not the fq-0.484 off-the-shelf arm.
- **[resolved ✓ supported]** Drop-an-expert is a fallback-leak + collateral hazard — orphans hit a
  sibling matching the query nearly as well (sim-ratio 0.980) and 72.7% of retain routes shift;
  author-key routing is clean (Δmu 0.7509→0.7509, zero shift). Retrain-in-place keeps orphans in
  their scrubbed experts (safe).
- **[resolved ✗ refuted]** §9-D "learned router loses the most utility on deletion" — every
  retrain-in-place unlearn Δmu is small and POSITIVE (≈+0.01), not a loss.
- **[resolved ✗ refuted]** The abstain/OOD-threshold fix seals the embed-route drop-leak —
  REFUTED (`2026-07-07_routing-fix-arms.md`): orphan and retain top-1 sim distributions overlap
  (means 0.858 vs 0.877), so reaching 90% orphan-abstain costs 58% retain false-abstain; no τ
  separates them. Motivates hard identity routing. (SEUF anchor scoped to the key-route composition
  only — it can't fix a retrieval-stage leak.)

## What worked
- **§9-D routing audit (2026-07-06):** clean, publishable result — author-key (hard) routing is
  byte-clean on deletion; the embedding/soft router is the leak channel; the drop-an-expert
  operation induces 73% retain selection shift while retrain-in-place induces ~0. Filled the §9-D
  table (`2026-07-06_routing-audit-results.md`).
- **GPU campaign (Llama-3.2-3B, n=32 pool, 4-GPU SLURM):** the learned RouterLoRA beats LegoNet's
  uniform 1/k on the same experts and lands between it and perfect-selection — key-routed RAMoLE
  **em 0.647 / ppl 5.354** vs 1/k **0.643 / 5.479** vs perfect **0.655 / 5.183**. Retriever-routed
  iid 0.623/6.186, ood 0.597/7.300 (instructor-xl top-1 0.84 / top-3 0.98).
- **Unlearning demo — router needs NO retrain.** Deleting a record (retrain only its 3 affected
  experts) served through the unchanged router raises its perplexity / drops EM: d0 3.75→5.91,
  d1 6.75→11.08, d2 3.97→4.95; RAMoLE forgets as cleanly as 1/k. Validates the exactness-preserving
  design (reference-split router decoupled from deletable data).
- **Ablations are honest nulls + a free win:** dropout p=0.5≈0, rank 6≈16, and the exactness-preserving
  `reference` split ≈ paper-faithful `corpus` split (0.647 vs 0.649) — i.e. exactness costs no utility.
- **Core RouterLoRA math validated on CPU** (`tests/test_router_lora.py`, all green): expert
  extraction (112 target linears, GQA d_out 3072/1024 split, scaling=α/r=2.0); **single active
  expert ≡ that LoRA applied directly via PEFT** (max|Δ|<2e-5) — the load-bearing identity; m
  identical experts ≡ one; per-sample mask routes each batch row to its own expert; alpha sums to 1;
  no NaN across 40 dropout steps; gradients flow to exactly the router params and nothing else;
  router-only save/load round-trips bitwise.
- **Full pipeline runs end-to-end on CPU** (`tests/test_pipeline.py`): fixture → retriever (FT +
  index) → router training → eval (router/mean/perfect × keys/retriever × iid/ood, all finite) →
  batched heterogeneous routing ≡ per-sample `set_active` (Δ=0).
- **Maximal reuse:** retrieval (`KNNRouter`), determinism/paths/`train_text` (`legonet_common`),
  expert checkpoints, and the eval metrics (`eval_memorization.metrics_for_records`) are all reused;
  the only genuinely new code is the retriever FT, the RouterLoRA module, and the serve/eval glue.

## What didn't / open problems
- **The router's win over 1/k is modest and the ablations are null** because the DBpedia-cluster setup
  is too easy/homogeneous: retrieval is near-perfect (top-3 0.98), k=3 averaging of topically-coherent
  experts is already close to optimal, and the IID/OOD gap is small so dropout/rank/split don't move
  it. The router should pull ahead (and dropout should matter) on heterogeneous/cross-task experts and
  higher k — not yet run.
- **Unlearning demo is N=1 per deletion** (reused the existing 1-record d0/d1/d2 deletions) → a clean
  qualitative demonstration (ppl rises, em drops, router unchanged) but not a powered statistic.
  canary_em stays ~0 (weak single-record canary memorization, as in the legonet thread) — perplexity
  is the usable forget signal.
- **Retriever FT is a no-op here** (off-the-shelf == fine-tuned ranking) and instructor-xl full FT
  OOM'd at batch 16 on a 44 GiB card → ran at batch 4. Instructor-xl loads fine under
  sentence-transformers 3.4.1 (risk cleared); `encoder_model` is one config line to fall back.

## Open ideas / next steps
- **Stress where the router should win:** heterogeneous/cross-task experts (mix DBpedia with another
  task family) and higher k, where 1/k dilutes and a learned gate (and dropout's OOD gain) should pull
  ahead. The current homogeneous DBpedia pool is too easy to separate the methods.
- **Powered unlearning:** one multi-record deletion (10–20 records) for a before/after with N>1,
  accepting the extra affected-adapter retrains.
- **7B transfer:** run the campaign on the `legonet_7b_v2` pool for a second base model.

## Reports
- [`ROUTING_AUDIT_REPORT_2026-07-06.md`](../../tofu_sisa_lora/reports/ROUTING_AUDIT_REPORT_2026-07-06.md)
  — the filled §9-D table: author-key clean, embedding/drop-expert leaks, the FT-encoder confound fix.

## Entries (chronological)
- [2026-06-25 — implementation + CPU validation](2026-06-25_implementation.md) — RAMoLE built on the
  legonet l32_3b pool; RouterLoRA math + full pipeline green on CPU; ready for GPU.
- [2026-06-26 — overnight campaign results](2026-06-26_overnight-results.md) — router > 1/k < perfect
  on the same experts; dropout/split/rank null; unlearning forgets with the router unchanged.
- [2026-06-27 — RAMoLE on TOFU (embedding RAG + router)](2026-06-27_tofu-rag-router.md) — router ≈ 1/k
  (unlearn mu 0.507 vs 0.501, fq 0.890); embedding-RAG routing costs forget_quality (0.48 vs 0.89 key);
  deletion forgets with the router unchanged. New TOFU arm in `tofu_sisa_lora/`.
- [2026-06-29 — retriever fine-tune (fp16 bug fixed; FT backfires)](2026-06-29_retriever-ft.md) — T5
  fp16-overflow no-op caught + fixed (fp32); the contrastive retriever FT made forget-author routing
  WORSE (unlearn fq 0.48→0.18), confirming author lookup is the right TOFU router. Best = key+router.
- [2026-07-02 — follow-up battery E0–E6](2026-07-02_followup-battery.md) — seed-robust gaps
  (DBpedia +0.005±0.001 em; TOFU unlearn mu +0.007±0.001; TOFU full = noise); α mechanism: DBpedia
  router learns ≈uniform (ideal-mass exactly 1/3) so router≈1/k is the CORRECT solution there, TOFU
  sharpens (max-share 0.54) but by difficulty not cluster (ρ(sharp,em)=−0.52); index staleness NOT the
  leak channel (encoder dominates; 10.7% retain shift on rebuild, fq unchanged); batched serving
  scales linearly (82.9 tok/s @b16, 6.9× over merge-path, 0.44× single-LoRA parity); N=15 deletion
  clean with router unchanged. E4 k-sweep pending retry (path bug fixed).
- [2026-07-06 — E4 k-sweep closes H4](2026-07-06_k-sweep.md) — REFUTED: the router−1/k gap is exactly
  flat (+0.005 em at k=3/5/8; both dilute together, em 0.65→0.58). Coherent with E2: a ≈uniform router
  cannot resist dilution differently from 1/k; the +0.005 is mild denoising. Battery E0–E6 fully closed.
- [2026-07-03 — §9-D routing-audit design](2026-07-03_routing-audit-9d.md) — pre-registration +
  base-pinned/dropped-policy diffs (CPU-gated); absorbs the unlogged 2026-07-02 audit runs.
- [2026-07-06 — §9-D routing-audit results](2026-07-06_routing-audit-results.md) — filled the §9-D
  table: author-key clean on deletion (H4✓), encoder confound fixed (H1/H6✓), drop-an-expert =
  fallback-leak (sim 0.98) + 72.7% retain shift (H2/H3✓); §9-D utility-loss prediction refuted.
- [2026-07-07 — §9-D fix arms](2026-07-07_routing-fix-arms.md) — abstain/OOD threshold REFUTED
  (orphan & retain sim distributions overlap; 90% orphan-abstain costs 58% retain false-abstain);
  strengthens the case for hard identity routing. SEUF anchor scoped to key-route composition only.

## Full reports
- [`RAMOLE_SUMMARY.md`](../../../../storage2/jack/checkpoints/ramole/RAMOLE_SUMMARY.md)
  (`/storage2/jack/checkpoints/ramole/RAMOLE_SUMMARY.md`) — **the thread summary**: what was built,
  all headline numbers (both arms), the α mechanism, the retrieval-leak story, the §9-D audit table,
  serving throughput, bugs caught, bottom line + entry index.
- `/storage2/jack/checkpoints/ramole/OVERNIGHT_REPORT.md` — DBpedia campaign tables (router/1-k/perfect, ablations, d0–d2 deletion).
- `/storage2/jack/checkpoints/ramole/FOLLOWUP_REPORT.md` — E1–E6 battery tables (seeds, α, audits, k-sweep, throughput, d_batch15).
- `/storage2/jack/checkpoints/ramole/ROUTER_LORA_ASSESSMENT.md` — the RouterLoRA-only assessment (incl. the 1/k definition).
- `/storage2/jack/checkpoints/tofu_sisa_lora/Llama-3.2-1B-Instruct_legonet_n32_k3/RAMOLE_TOFU_REPORT.md` — TOFU arm comparison tables.
