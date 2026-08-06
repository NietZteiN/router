### Target Date: 2026-06-29 (RAMoLE-TOFU retriever fine-tune — fp16 bug fixed; FT backfires)
- **Goal / Hypothesis:** Close the embed↔key routing gap from [2026-06-27](2026-06-27_tofu-rag-router.md)
  by contrastively fine-tuning the instructor-xl retriever (InfoNCE, same-author question pairs, retain
  authors 0–179), expecting embed-routed forget_quality to move toward the key (author-lookup) ceiling
  (unlearn 0.89). Router + experts unchanged; only the retriever/index change. Arm = `ramoleft_*`.
- **Setup:** `train_retriever_tofu.py` + `retriever` phase in `submit_ramole_tofu.sh` (FT → rebuild
  expert index with the FT encoder → re-eval embed arm). **First run was a silent no-op:**
  `loss=0.0, grad_norm=nan` every step — instructor-xl is T5-based and overflows under fp16 autocast
  (sentence-transformers `use_amp=True`), so NaN grads → unchanged weights → `ramoleft==ramole` to 4
  decimals (artifact, caught before reporting). **Fix:** `use_amp=False` (fp32). Re-run trained
  properly (train_loss 0.059, grad_norm finite). SLURM jobs 438365→438368 (≤4 GPU; waited on full
  cluster — all 12 sprint1-3 GPUs allocated, OverSubscribe=NO / no MIG/MPS so no GPU slicing). Extended
  caps authoritative (smoke KS underpowered — smoke ramoleft_unlearn fq 0.9988 vs extended 0.1802).
- **Results (extended; fq↑ = forgotten, mu↑ = utility):**
  | arm | full mu / fq | unlearn mu / fq |
  |---|---|---|
  | key + 1/k (legonet) | 0.495 / 0.0004 | 0.501 / **0.890** |
  | key + router (routerkey) | 0.494 / 0.000 | **0.507** / **0.890** |
  | embed off-the-shelf + router (ramole) | 0.467 / 0.035 | 0.477 / 0.484 |
  | embed **fine-tuned** + router (ramoleft) | 0.467 / 0.484 | 0.473 / **0.180** |
  The FT **moved routing the WRONG way**: unlearn forget_quality fell 0.484→**0.180** (target was
  →0.890), and full-model fq rose 0.035→0.484 (target ~0, i.e. a full model that recalls forget). Both
  signal that the FT encoder MISROUTES forget-author queries more than off-the-shelf instructor-xl.
- **Observations:** The fine-tune backfired because it was trained on **retain authors only**, and
  forget_quality is dominated by **forget-author (OOD) routing** — exactly the authors the FT encoder
  never saw. Same-author contrastive on TOFU's near-duplicate templated questions drove the loss to
  ~0.05 almost immediately (trivially separable), specializing the encoder to retain-author surface
  features and degrading instructor-xl's general semantics that had been routing forget authors
  passably. Net: off-the-shelf embed > fine-tuned embed for routing quality, and **exact author lookup
  (key) remains far ahead** (unlearn fq 0.89 vs 0.18/0.48). The robust RAMoLE conclusion is unchanged:
  best config = **key-route + router** (mu 0.507, fq 0.890), a hair above the 1/k baseline; embedding-
  RAG is fundamentally limited by TOFU's templated questions and this fine-tune does not rescue it.
- **Next Steps:** Embedding-RAG on TOFU is a dead end as posed. Options if pursued: (a) train the
  retriever on ALL authors incl. forget (breaks the retain-only exactness premise — only valid if the
  retriever is treated as a non-deletable shared component); (b) route on richer text (author
  name/entities) rather than the templated question; (c) accept that author lookup is the correct
  router for TOFU and report key+router as the RAMoLE result. The cross-task heterogeneous-expert
  setting (where retrieval + a learned gate should actually help) remains the place to demonstrate
  RAMoLE's gains — not TOFU. Report: `…/Llama-3.2-1B-Instruct_legonet_n32_k3/RAMOLE_TOFU_REPORT.md`.
