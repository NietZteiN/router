### Target Date: 2026-06-27 (RAMoLE on TOFU — embedding RAG + RouterLoRA vs 1/k)
- **Goal / Hypothesis:** Apply the RAMoLE method (embedding retrieval / RAG over LoRAs + the learned
  RouterLoRA cross-attention) to the TOFU author-expert pool and score it in TOFU's canonical
  open-unlearning metrics (model_utility, forget_quality), full and post-forget10. Two questions:
  (1) does the learned router beat the uniform 1/k composition; (2) how much does embedding-RAG
  routing cost vs the oracle author lookup. Reuses the trained `Llama-3.2-1B-Instruct_legonet_n32_k3`
  pool (no expert retraining); router trained on retain authors 0–179 so deletion needs no router
  retrain. Continuation of [2026-06-26 overnight](2026-06-26_overnight-results.md) (DBpedia).
- **Setup:** New RAMoLE-TOFU arm in `tofu_sisa_lora/`, importing `ramole/router_lora.py`: `ramole_tofu.py`
  (TOFU LoraRetriever `build_expert_index` = mean of member authors' question embeddings via
  instructor-xl; `RamoleTofuModel` mirroring `LegoNetRoutedModel` but `_activate`→`controller.set_active`,
  `route_mode∈{embed,key}`; `load_ramole_eval_model`), `train_router_tofu.py` (AdamW on `{A_r,B_r}`,
  retain authors only, Random LoRA Dropout p=0.5), `eval_tofu.py` hook (`--ramole_router`,
  `--ramole_route`), config `configs/ramole_tofu_1b.json`, `submit_ramole_tofu.sh`, report
  `ramole_tofu_report.py`. Two `ramole/router_lora.py` tweaks let the TOFU cfg (no source_run/name)
  build the model. CPU regression `test_ramole_tofu.py` green (both routing modes, B>1, embed-RAG
  recovered each author's expert 6/6). SLURM (≤4 GPU, sprint1-3): index 437488 (expert index 32×768)
  ∥ router 437489 (450 steps, loss 2.3→1.9) → smoke eval 437490 → extended 437491. Arms:
  `ramole_*`=embed+router, `routerkey_*`=key+router, vs on-disk `legonet_*`=key+1/k. Repo not under git.
- **Results (extended caps; smoke is KS-underpowered for the full arms):**
  - **Composition — router ≈ 1/k, marginally ahead.** key-routed: full mu 0.494 (router) vs 0.495 (1/k);
    **unlearn mu 0.507 (router) vs 0.501 (1/k)** with identical forget_quality (full 0.000, unlearn
    **0.890** — same routing ⇒ same forget behavior); router gives consistently sharper retained
    memorization (forget_ppl 2.92 vs 3.27 full, 10.96 vs 11.10 unlearn). A hair better, as on DBpedia.
  - **Retrieval — embedding-RAG costs utility AND forget_quality vs the oracle lookup.** router held
    fixed: full mu 0.467 (embed) vs 0.494 (key); **unlearn forget_quality 0.484 (embed) vs 0.890 (key)**,
    mu 0.477 vs 0.507. TOFU's templated questions don't separate authors cleanly, so embedding
    retrieval misroutes some forget-author queries → roughly HALVES the forget signal.
  - **Unlearning works with the router UNCHANGED** (the core claim): the same router loaded for full and
    unlearn; deletion (retrain affected experts only) lifts forget_quality 0.0004→**0.890** (key) /
    0.035→0.484 (embed). No router retrain.
- **Observations:** Two consistent stories. (1) On homogeneous expert pools (DBpedia topics, TOFU
  authors) the learned cross-attention gate is ≈ uniform 1/k — a tiny, reliable edge but no more,
  because the experts are nearly interchangeable; RAMoLE's larger gains need genuinely specialized
  heterogeneous task-experts (the paper's 48-task setting). (2) Embedding-RAG routing is the weak link
  on TOFU: the exact author lookup is an oracle, and instructor-xl question embeddings can't match it
  (forget_quality 0.48 vs 0.89). This is the honest, measured cost of the RAG step. The router-with-key
  arm (0.507 / 0.890) is the best RAMoLE config here and slightly beats the 1/k baseline on utility.
- **Next Steps:** (1) Better TOFU retrieval — fine-tune the retriever contrastively on retain authors
  (same-author positives) and/or route on answer-style embeddings, to close the embed↔key gap. (2) The
  setting where the router should actually win: heterogeneous/cross-task experts + higher k. (3) Optional
  7B pool (`Llama-2-7B-chat-hf_legonet_n32_k3`, 1/k baseline mu 0.637/fq 0.808) for transfer. Report:
  `…/Llama-3.2-1B-Instruct_legonet_n32_k3/RAMOLE_TOFU_REPORT.md`.
