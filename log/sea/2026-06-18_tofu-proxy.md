### Target Date: 2026-06-18 (SEA on TOFU — per-author proxy implementation)
- **Goal / Hypothesis:** Stand up a new `~/sea_tofu` package implementing SEA (Separable Expert
  Architecture, arXiv:2604.21571) on TOFU per `~/SEA_on_TOFU.md`. Minimal-faithful config: frozen
  4-bit Llama-2-7B-chat base + per-author personal-LoRA proxy via SFT (one author = one deletable
  proxy). Hypothesis under test (later): personalization depth rises with LoRA rank (deletability
  tax), cross-author contamination ≈ 0, deletion = rm (ms), forget quality ≈ 1 by construction.
- **Setup:** Verified the SEA paper against the existing `~/sea` impl (faithful: k=4 experts, personal
  LoRA r4/DPO, CAA steer L={12,16,20} γ1, bias EMA λ0.5, BART-MNLI T2.0, NF4) — but `~/sea` runs on 4
  synthetic personas, not TOFU. New `~/sea_tofu`: load_tofu.py, proxy_paths.py, inference.py
  (load_base 4-bit + SeaProxyModel with ≤1 resident adapter), train_proxy.py (per-author SFT, block
  mode), deletion.py, metrics_sea.py, eval_sea_tofu.py (imports canonical metric primitives from
  `~/tofu_sisa_lora/eval_tofu.py`), run_pilot.py, submit_train_proxies.sh, submit_eval.sh,
  configs/sea_tofu_llama2.json (r16, α=2r, q/k/v/o, rslora, epochs12 lr2e-4 maxlen256, seed42).
  `proxies → /storage2/jack/checkpoints/sea_tofu/proxies`. Prompt fixed to eval_tofu's
  `"Question: {q}\nAnswer: {a}"`. Provenance: repo not git → meta.json git_commit=None.
- **Results:** Pre-flight: TOFU loads in datasets 4.8.5 (full=4000, author i = rows [i*20,i*20+20);
  forget10_perturbed carries paraphrased_answer + 5 perturbed_answer). All 7 modules py_compile +
  import clean; SLURM scripts pass `bash -n`; eval_tofu primitives import clean; Llama-2-7B-chat in
  cache. **Pilot (SLURM job 435382, sprint1, 5 authors 180–184 @ rank 16): ALL GATES GREEN.** Per-author
  SFT loss 2.10→0.046 (12 ep, 60 steps/author, ~27s). Personalization (proxy vs base): ΔProb +0.79..+0.88
  (proxy ~0.999 vs base ~0.12–0.21), ΔROUGE-L +0.58..+0.65 (proxy 1.0 vs base ~0.36–0.42), truth-ratio
  0.75→~0.44–0.51 (lower=knows truth). Isolation contamination = 0.0 on all 3 pairs (sim_proxyA_on_B ≤
  sim_base_on_B → no adapter accumulation/leak). forget_quality = 1.0 (construction-trivial). Deletion
  gate passed, kl=0.0 (omission==base==cached baseline → confirms omission==post-deletion). model_utility
  0.8135 (retain prob/rouge/truth ~1.0/1.0/0.50; real/world from frozen base). **Proxy size 64 MB @ r16**
  (fp32 LoRA on q/k/v/o×32 layers ≈16.7M params) — well above the paper's 2–5 MB; the deletability tax
  the guide §7.2 predicts at r≥16 (can store bf16 to halve).
- **Observations:** SEA's forget quality is construction-trivial (base-only candidate == base-only
  gold → KS p≈1); the science is the rank/size tradeoff + isolation + deletion cost. Reused eval_tofu
  metric math wholesale (no re-implementation) so numbers stay comparable to the SISA-LoRA track.
  evaluate_model can't be reused (single-model assumption) — utility assembled per-author. r16 already
  fully memorizes (ROUGE-L 1.0) → the rank sweep's interesting end is the LOW ranks (4, 8) where the
  tax should bite; expect r4 to underfit per the paper's "rank-4 targets style, not knowledge."
- **Next Steps:** Scale launched — 200 proxies @ r16 (`submit_train_proxies.sh 16 0 199 20`, reuses
  pilot 180–184) + rank sweep r∈{4,8,32,64} on forget10 180–199. Then `submit_eval.sh <rank>` per rank,
  assemble personalization-vs-rank + proxy-size-vs-rank + isolation tables. Optional: GA/NPO baselines
  via tofu_sisa_lora for Table B.

