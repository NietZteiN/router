# CLAUDE_SCRATCHPAD — SEA on TOFU

Plan: `/home/jack/.claude/plans/let-s-plan-to-implement-gleaming-dahl.md` (approved 2026-06-18).
Guide: `~/SEA_on_TOFU.md`. Paper: `papers/Separable Expert ArchitectureToward Privacy.pdf`
(verified faithful to `~/sea`, which runs on 4 synthetic personas, NOT TOFU).
Interpreter `/home/jack/anaconda3/envs/test-env/bin/python` | HF_HOME=/storage2/jack/data/huggingface | seed 42.

## Goal / hypothesis
Apply SEA (minimal-faithful: frozen 4-bit Llama-2-7B + per-author personal-LoRA proxy via SFT) to
TOFU. The interesting axes are **personalization-depth vs. rank/proxy-size**, **isolation**, and
**deletion cost** — NOT forget quality (≈1 by construction for SEA). One author = one deletable proxy.

## Constraints checklist (CLAUDE.md)
- [x] Code only in ~/sea_tofu; artifacts via `proxies → /storage2/jack/checkpoints/sea_tofu/proxies`.
- [x] All GPU jobs via SLURM (exclude sprint4, 1 GPU/task, 64G for 7B, ≤12 concurrent). CPU checks direct.
- [x] HF_HOME respected in every entry point + SLURM script. Llama-2-7B already cached.
- [x] Seed 42 recorded in each proxy meta.json.
- [ ] Smoke-validate (5-author pilot) before the 200-author + rank-sweep scale run.
- [ ] Provenance: commands + SLURM job IDs → LOG.md (repo not git → meta.json git_commit=None).

## Key design facts
- Prompt format MUST be `"Question: {q}\nAnswer: {a}"` (= eval_tofu._build_qa_prompt). The guide's
  `[INST]` template would mismatch eval and tank metrics.
- Reuse metric primitives by import from `/home/jack/tofu_sisa_lora/eval_tofu.py` (verified imports
  clean): get_rouge, get_answer_probability, get_prob_w_options, get_truth_ratio_scores,
  tr_forget_agg, tr_nonforget_agg. Do NOT reuse evaluate_model (single-model assumption) — assemble
  Model Utility per-author with scipy.stats.hmean over the same 9 components.
- Forget-Quality reference = base-only TR on forget authors (NOT tofu_sisa_lora's retain90 oracle):
  post-delete system = base, SEA gold = base → KS p ≈ 1 (construction-trivial; flag it).
- Adapter accumulation is the #1 pitfall: SeaProxyModel keeps ≤1 adapter resident (delete_adapter on
  swap); omission()=disable_adapter()=base-only=post-deletion behavior.
- QLoRA SFT mirrors sea/train_expert.py (4-bit NF4 → get_peft_model → SFTTrainer bf16; no
  prepare_model_for_kbit_training needed in this env). LoRA: rank r, α=2r, dropout 0.05, q/k/v/o,
  use_rslora=True. Config: configs/sea_tofu_llama2.json (epochs 12, lr 2e-4, max_len 256).

## Module map (~/sea_tofu)
load_tofu.py (TOFU + i*20 grouping) · proxy_paths.py · inference.py (load_base 4-bit, SeaProxyModel) ·
train_proxy.py (per-author SFT; --author_id or --author_start/--author_count block) · deletion.py
(verify_and_delete) · metrics_sea.py (personalization_depth, isolation, deletion cost) ·
eval_sea_tofu.py (orchestrator + CLI) · run_pilot.py · submit_train_proxies.sh · submit_eval.sh.

## Status
- [x] All modules written; py_compile + import + bash -n green (2026-06-18).
- [x] Pilot (job 435382, authors 180-184 @ r16): ALL GATES GREEN. ΔProb +0.79..0.88, ΔROUGE +0.58..0.65,
      contamination 0.0, fq 1.0, deletion kl=0.0 passed, model_utility 0.8135. Proxy=64MB @ r16 (fp32;
      >paper 2-5MB → deletability tax). Summary: proxies/Llama-2-7B-chat-hf/results/pilot/pilot_summary.json.
- [x] Scale DONE: 200 proxies @ r16 + 20 each @ r{4,8,32,64} on forget10 (jobs 435413-20).
- [x] Sweep eval DONE (job 435654, run_sweep_eval.py — replaced 5 timed-out per-rank jobs).
      Tradeoff (forget10, proxy loaded, max_new=40): proxy ROUGE-L 0.594(r4) → 0.864(r8) → ~0.87(r16-64);
      proxy size 16/32/64/128/256 MB; contam 0.06-0.11; fq=1.0; model_utility@r16=0.78.
      KNEE r4→r8: r4 underfits (paper's "rank-4=style"), saturates by r8. Size ≫ paper 2-5MB.
      Results: proxies/Llama-2-7B-chat-hf/results/sweep/sweep_results.json. CORE STUDY COMPLETE.
- [x] Optional polish DONE: extended (max_new=128, r4 0.673/r8 0.991/r16 1.000) + seed variance
      r4/r8 over seeds {42,43,44} (std≈0.006 → effect robust).
- [x] Standard TOFU unlearning report DONE (job 436005, eval_unlearning_report.py): canonical schema
      @ r16. Original→Unlearned: forget ROUGE 1.0→0.403, Prob 0.999→0.161, FQ 0.0→1.0; Model Utility
      0.711 UNCHANGED (deletion never touches retain/real/world); Unlearned==Retrain-gold by constr.
      Out: results/report/unlearning_report.json + reports/SEA_UNLEARNING_REPORT.md.
- [ ] Remaining optional: GA/NPO baselines via tofu_sisa_lora for deletion-cost Table B (SEA ms rm
      vs GPU-min) — deferred, busy cluster.

## Eval gotchas (learned)
- Per-rank eval is generation-bound: 5 separate jobs @ ROUGE-100tok all hit SLURM walls. Use the
  COMBINED run_sweep_eval.py (base once, base-side shared, max_new=40, incremental per-rank writes).
- SLURM heredoc: keep the python invocation ON ONE LINE — \\-continuations collapse to escaped spaces
  and argparse rejects them (bit the first submit_train_proxies.sh; verify with STUB=1 | cat -A).

## Findings to remember
- r16 fully memorizes (ROUGE-L 1.0) → sweep's signal is at LOW rank (4,8) where the tax bites.
- Proxy size ≈ fp32 LoRA params: r16=64MB; expect r4≈16MB, r64≈256MB. Store bf16 to halve if size matters.
- Deletion kl=0.0 is exact: baseline built in omission mode == base == post-deletion (correct by construction).

## Pilot checklist / watch-fors
- Training loss finite, not NaN; ROUGE with proxy > base (memorization of 20 QA works at r16).
- 20 QA × bs4 = 5 steps/epoch × 12 = 60 steps/author. If under-fit, bump epochs in config.
- Watch: empty/repetitive generations, frozen loss, OOM from foreign PIDs (re-submit --exclude).
