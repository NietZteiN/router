### Target Date: 2026-06-10 (k=1 LoRA-ft ≥0.6 utility baseline — Stage 1 recompute + overnight grid)
- **Goal / Hypothesis:** Recompute the Llama-2-7B-chat-hf ft-vs-base utility comparison under the corrected OU-faithful metric (pre-port numbers ft mu=0.2507 / base mu=0.129 are invalid), and find a k=1 full-data LoRA recipe with **model_utility ≥ 0.6** (user-set bar; OU leaderboard 7b Finetuned=0.63, locuslab ft ckpt=0.748 under our eval).
- **Setup:** seed 42 everywhere; interpreter `/home/jack/anaconda3/envs/test-env/bin/python`; plan `.claude/plans/have-we-tried-finetuning-snuggly-dream.md`. Script hashes (repo not git): eval_tofu.py `63008cc78a821613…`, train_lora_shard.py `2ec97dbb864b93ac…`, submit_llama2_grid_overnight.sh `cb95ae2b469845df…`.
  - CPU gate: `test_ou_equivalence.py` ALL PASS (rerun pre-flight).
  - retain90 oracle: reused prior-session job **433505** (submit_retain90.sh task 3; r8/α16/e3/lr2e-4/bs1×ga16) → `checkpoints/Llama-2-7B-chat-hf/retain90/`, symlinked into `Llama-2-7B-chat-hf_ft/retain90`.
  - Stage 1 chain (`submit_llama2_ft_vs_base.sh`): **433514** prepare_eval --smoke (retain_tr_scores.npy, 30 samples, mean 0.781) → **433515** ft eval (shard_0_only) ∥ **433516** base eval.
  - Smoke-train pipeline check (`submit_llama2_sweep_variant.sh 1e-4 1 16 32 SMOKE_TRAIN`): job **433517**, loss 2.24→1.19 (finite, decreasing), adapter written. (First attempt **433510** FAILED — see Observations.)
  - Overnight sequential grid (`submit_llama2_grid_overnight.sh`), one combined train+eval job per variant, chained `--dependency=afterany`, bs1×ga32, max_len 256: **433518** lr1e-4/e5/r16 → **433519** lr1e-4/e5/r8 → **433520** lr2e-4/e5/r16 → **433521** lr5e-5/e5/r16 → **433522** lr1e-4/e10/r16 → **433523** lr1e-4/e5/r32 → **433524** lr5e-5/e10/r32 → **433525** collect_results --smoke (no GPU). KS reference copied per-variant from the _ft dir (variant-independent).
- **Results:**
  - **Base model (corrected, smoke): model_utility 0.4179** (`base_model.json`, metrics_version ou-2026-06-10): real_rouge 0.982, world_rouge 0.933, retain_prob 0.164, retain_rouge 0.427, forget_quality 0.239, forget_ppl 15.19 ≈ retain_ppl 15.67. Pre-port base value (0.129) was wrong by >3×.
  - ft control (shard_0_only) re-eval: job 433515 running at write time; grid results land overnight.
- **Observations:**
  - Heredoc gotcha: backslash line-continuations inside `$(sbatch <<EOF …)` get an extra backslash-newline pass and become literal space args → argparse "unrecognized arguments" (failed job 433510). Repo scripts never hit this because they don't wrap sbatch in command substitution. Fix: single-line job commands; verified via sbatch-stub capture. The smoke-test-first protocol caught it before the real sweep.
  - sacct accounting is disabled on this cluster — completion checks must use squeue + output files.
  - retain90 oracle is base-model-specific and recipe-independent → train once, symlink everywhere; same for retain_tr_scores.npy (copy, don't recompute per variant; saves ~15 min GPU each).
- **Next Steps:** Morning: read 7 grid JSONs + control/base; build corrected comparison table; apply 0.6 gate (best smoke mu ≥ 0.6 → extended eval on winner must also be ≥ 0.6). If none pass: per-component diagnosis across lr/epochs/rank axes → one targeted follow-up round (rank 64 / epochs 8 / +gate_proj) or report a LoRA-capacity-ceiling finding.

---

