### Target Date: 2026-06-25
- **Goal / Hypothesis:** Implement RAMoLE (Retrieval-Augmented Mixture of LoRA Experts) on top of the
  existing LegoNet `legonet_l32_3b_n32_k3` pool (Llama-3.2-3B-Instruct, n=32 DBpedia clusters, k=3),
  so the learned composition is a drop-in alternative to LegoNet's uniform `1/k` delta-average on the
  *same* experts. Hypothesis to test on GPU next: the learned per-layer RouterLoRA cross-attention
  beats `1/k` on retained utility while preserving the O(1)/cascade-free deletion property (the router
  is task-agnostic and trained on the disjoint reference split → no router retrain on deletion).
- **Setup:** New `ramole/` package (code-only home), artifacts → `/storage2/jack/checkpoints/ramole`.
  Reuses `legonet_lora` modules via `sys.path` (`legonet_common` determinism/Paths/`train_text`,
  `routing.KNNRouter`, `combine.LegoNetModel` as the `1/k` baseline, `eval_memorization.metrics_for_records`).
  New code: `ramole_common.py` (config/paths/instruction-encoder/cluster-split),
  `retriever.py` (Stage 1: instruction-prefixed encoder, InfoNCE FT on 40% clusters via
  `MultipleNegativesRankingLoss`, LoRA index = mean of m member embeddings, top-k cosine + IID/OOD
  accuracy), `router_lora.py` (Stage 2 core: `extract_expert_weights`, `RouterLoraLinear` +
  `RouterController`, `install_router`/`freeze_to_router`/`save`/`load`/`build_ramole_model`),
  `train_router.py` (deterministic AdamW on `{A_r,B_r}` only, Random LoRA Dropout p=0.5),
  `ramole_model.py` (Stage 3 serve: `set_active` / `set_routing` union+mask), `eval_ramole.py`
  (router vs mean vs perfect × keys/retriever × iid/ood, reusing legonet metrics). Configs
  `configs/ramole_l32_3b.json` (primary; encoder `hkunlp/instructor-xl`, `router_train_split=reference`)
  + `configs/ramole_smoke.json`. SLURM: `slurm_nodes.sh` + `submit_ramole.sh`
  (`setup|retriever|router|eval|all`, retriever ∥ router → eval, ≤8 GPU, `STUB=1` preview). Tests:
  `tests/_fixture.py` (tiny random Llama + random PEFT experts + synthetic corpus/keys/assignment,
  GQA via heads>kv_heads), `tests/test_router_lora.py`, `tests/test_pipeline.py`. Env
  `/home/jack/anaconda3/envs/test-env/bin/python` (peft 0.14.0, sentence-transformers 3.4.1). Repo is
  not under git, so no commit hash; no SLURM jobs submitted yet (CPU validation only).
- **Results:** All CPU tests green.
  `test_router_lora.py`: extraction = 112 target linears, GQA d_out q/o=3072 vs k/v=1024 confirmed on
  the real a0 safetensors, scaling=α/r=2.0 (use_rslora=False); **single active expert ≡ that LoRA
  applied directly via PEFT, max|Δ|<2e-5** (the load-bearing correctness anchor); 3 identical experts
  ≡ 1; per-sample `-inf` mask routes each batch row to its own expert (Δ≈1e-7); alpha sums to 1 over
  experts; no NaN/Inf across 40 dropout steps (≥2 survivors enforced); gradients reach exactly the
  16/16 router tensors and nothing else; router save/load bitwise (Δlogits=0) and rejects a key-set
  mismatch. `test_pipeline.py`: fixture→retriever(FT+index)→router(train+save)→eval(all 5 method/route/
  condition combos finite)→batched heterogeneous routing ≡ per-sample `set_active` (Δ=0). MiniLM loads
  offline; instructor-xl is not cached (verified absent).
- **Observations:** (1) The decisive design point is that PEFT's `add_weighted_adapter` fuses experts
  into one delta and cannot express per-expert attention — RAMoLE manages LoRA weights manually
  (frozen non-persistent buffers + fp32 router params), confirmed by the single-expert≡PEFT identity.
  (2) GQA forces per-layer `A_r/B_r` sizing (q/o 3072, k/v 1024); a global router pair would be
  shape-incompatible. (3) Router math kept in fp32 under a bf16 base for a stable softmax. (4) Chose
  `router_train_split=reference` as default so the router never sees deletable records (LegoNet
  Condition A) — keeps deletion exact with no router retrain; `corpus` (paper-faithful) is available.
  (5) Instruction applied as a uniform text prefix (FT and inference) to sidestep instructor pair-API
  fragility. (6) instructor-xl × sentence-transformers 3.4.1 compatibility is the main open risk;
  `encoder_model` is one config line to fall back (instructor-large / MiniLM).
- **Next Steps:** GPU run — `bash submit_ramole.sh configs/ramole_l32_3b.json all` (setup pre-caches
  the encoder on the login node). Produce the comparison: RAMoLE-router vs LegoNet `1/k` (both on
  key-routed top-3, isolating composition) vs perfect-selection, IID + OOD, on MMLU/PPL/EM/VerbMem;
  plus retrieval top-k (off-the-shelf vs 40%-FT). Then a deletion demo: delete records via
  `legonet_lora/unlearn.py`, re-eval with the unchanged router.
