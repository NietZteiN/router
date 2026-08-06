### Target Date: 2026-06-09 (SEA implementation)

**Goal / hypothesis:** Implement the SEA (Separable Expert Architecture) paper (arXiv 2604.21571) as a new `sea/` project. SEA personalizes LLMs by isolating user-specific parameters in a deletable per-user "proxy artifact", enabling provably complete unlearning via filesystem deletion.

**Setup:**
- New directory: `sea/` (12 files)
- Target model: `meta-llama/Llama-3.1-8B-Instruct` (NF4 quantized via BitsAndBytes)
- 4 domain experts: security, code, data, general (rank=32, α=64, attention projections)
- 4 synthetic user profiles matching paper: security_expert, casual_coder, data_analyst, general_user
- Per-user proxy: routing bias b_u (EMA), steering vectors at layers {12,16,20} (CAA), personal LoRA rank=4 (DPO)
- Inference: BART-MNLI router (T=2.0) → bias → PEFT add_weighted_adapter merge → steering hooks → generate
- No jobs submitted yet; code complete, syntax verified.

**Pipeline to run:**
1. `bash sea/submit_train_experts.sh sea/checkpoints meta-llama/Llama-3.1-8B-Instruct` (4 jobs, ~4h each)
2. `bash sea/submit_train_proxies.sh sea/checkpoints meta-llama/Llama-3.1-8B-Instruct` (4 jobs, ~2h each)
3. `bash sea/submit_eval.sh sea/checkpoints meta-llama/Llama-3.1-8B-Instruct --smoke` (smoke first)
4. `bash sea/submit_eval.sh sea/checkpoints meta-llama/Llama-3.1-8B-Instruct` (full eval)

**Results:** Pending — no jobs run yet.

**Observations:** N/A

**Next steps:** Smoke-test `train_expert.py` on a single domain (--max_samples 200, --epochs 1) before submitting full array job. Verify proxy artifact shapes and SEAModel.generate() runs without error on both proxy-loaded and proxy-free modes.

---

