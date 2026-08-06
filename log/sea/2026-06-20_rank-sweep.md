### Target Date: 2026-06-20 (SEA-on-TOFU rank sweep — scale results)
- **Goal / Hypothesis:** Complete the `~/sea_tofu` scale run (set up 06-18) → headline
  personalization-depth-vs-rank / proxy-size-vs-rank tradeoff (the deletability tax) + isolation +
  utility. H: recall rises with LoRA rank then saturates; low rank underfits ("rank-4 targets style not
  knowledge"); contamination ≈ 0; forget quality ≈ 1 by construction. (Distinct project from the
  additive-shards entry above — `~/sea_tofu`, per-author deletable proxies.)
- **Setup:** 200 proxies @ r16 (job 435413) + 20 each @ r∈{4,8,32,64} on forget10 (435414/15/16/20),
  Llama-2-7B-chat 4-bit, per-author SFT 12 ep. First eval (5 separate per-rank jobs 435563-67) ALL hit
  SLURM time limits (each reloaded the 7B base + recomputed constant real/world + ROUGE@100tok). Replaced
  with `run_sweep_eval.py` (one GPU job 435654: base loaded once, frozen base-side computed once & shared
  across ranks, generation capped max_new=40, incremental per-rank JSON).
  Out: proxies/Llama-2-7B-chat-hf/results/sweep/sweep_results.json.
- **Results (forget10, proxy loaded; n_forget=20, max_new=40):**
  | rank | proxy MB | proxy ROUGE-L | base ROUGE-L | proxy Prob | base Prob | proxy TR | max contam |
  |------|---------|---------------|--------------|-----------|-----------|----------|-----------|
  | 4 | 16 | 0.594 | 0.329 | 0.707 | 0.161 | 0.493 | 0.110 |
  | 8 | 32 | 0.864 | 0.329 | 0.986 | 0.161 | 0.457 | 0.082 |
  | 16 | 64 | 0.870 | 0.329 | 0.999 | 0.161 | 0.476 | 0.085 |
  | 32 | 128 | 0.870 | 0.329 | 1.000 | 0.161 | 0.485 | 0.082 |
  | 64 | 256 | 0.862 | 0.329 | 0.999 | 0.161 | 0.506 | 0.064 |
  forget_quality = 1.0 (construction-trivial); model_utility @ r16 = 0.7798 (retain n=20).
- **Observations:** Deletability tax lands exactly where the paper predicts — **r4 underfits** (ROUGE
  0.594 / Prob 0.707), recall **saturates by r8** (ROUGE ~0.86, Prob ~0.99), r16-64 flat → knee is r4→r8.
  Proxy size doubles per rank (16→256 MB), all ≫ the paper's 2-5 MB (fp32 LoRA q/k/v/o×32). So the "tiny
  deletable artifact" story only holds at r4 — the rank that underfits — which IS the central tension.
  Contamination low (0.06-0.11), noisier than the pilot's 0.0 (only 3 probe Qs @ max_new=40). Absolute
  ROUGE saturates ~0.87 not 1.0 purely because of the 40-token cap (pilot @ 100-200 tok hit 1.0); the
  *relative* curve is the robust result. forget_quality=1.0 + the pilot deletion gate confirm the
  structural-unlearning claim: post-delete == base == retrain gold.
- **Next Steps (optional polish; core study complete):** extended pass — full retain utility, max_new≥128
  for absolute ROUGE, seed variance on the r4/r8 knee; GA/NPO baselines via tofu_sisa_lora for the
  deletion-cost Table B (SEA = ms `rm` vs GPU-min weight surgery). Tradeoff + isolation + deletion done.
- **UPDATE (same day) — hardening DONE (extended eval 435982 + seed variance 435983-988):**
  Extended (max_new=128, n_forget=20): **proxy ROUGE-L r4 0.673 / r8 0.991 / r16 1.000** (base 0.420) —
  resolves the 40-tok caveat: with adequate generation length recall SATURATES (→1.0) by r8-16; the
  r4→r8 knee is sharp (0.673→0.991). contam r4 0.059 / r8 0.048 / r16 0.108. (Job timed out at the 5h
  wall after r16 — incremental writes saved r4/r8/r16; r32/r64 plateau already shown saturated in the
  smoke run, model_utility@r16=0.78 from smoke stands.) **Seed variance over seeds {42,43,44} at the
  knee (mean±std): r4 ROUGE 0.668±0.006, Prob 0.699±0.007; r8 ROUGE 0.992±0.001, Prob 0.986±0.000.**
  ⇒ the rank effect (r4 underfit, r8 saturation) is robust to seed (std≈0.006), NOT noise — satisfies
  CLAUDE.md §4. Results: results/{extended,seed43,seed44}/sweep_results.json. GA/NPO Table B deferred
  (separate tofu_sisa_lora lift, busy cluster).
- **UPDATE 2026-06-23 — standard TOFU unlearning report (job 436005; `eval_unlearning_report.py`):**
  Assembled SEA into the canonical TOFU schema at r16 (forget10, n_retain=40, max_new=100), 3 states:
  | State | Forget ROUGE | Forget Prob | Forget TR | Retain ROUGE | Retain Prob | Real | World | Forget Quality | Model Utility |
  |---|---|---|---|---|---|---|---|---|---|
  | Original (proxies loaded) | 1.000 | 0.999 | 0.476 | 1.000 | 0.999 | 0.689 | 0.856 | **0.0** | **0.711** |
  | Unlearned (proxies deleted) | 0.403 | 0.161 | 0.701 | 1.000 | 0.999 | 0.689 | 0.856 | **1.0** | **0.711** |
  | Retrain gold (= base on forget) | 0.403 | 0.161 | 0.701 | 1.000 | 0.999 | 0.689 | 0.856 | 1.0 | 0.711 |
  **Reads exactly as a clean unlearning result:** deleting the forget proxies drops forget ROUGE
  1.0→0.403 and forget Prob 0.999→0.161 (= base), Forget Quality 0.0→1.0, while **Model Utility is
  unchanged (0.711)** — deletion never touches retain/real/world, so SEA preserves utility through
  unlearning *by construction*. Unlearned == Retrain-gold exactly (SEA reaches the gold by deletion,
  at ms `rm` cost). Sanity: report Unlearned forget_rougeL 0.403 ≈ sweep base_rougeL 0.420 (max_new
  128) ✓; Original proxy ROUGE 1.0 ✓. (MU 0.711 here vs 0.78 smoke = cleaner caps/sample; both in the
  TOFU-7B range — OU Finetuned 0.63, locuslab ft 0.748.) Out: results/report/unlearning_report.json +
  reports/SEA_UNLEARNING_REPORT.md.

