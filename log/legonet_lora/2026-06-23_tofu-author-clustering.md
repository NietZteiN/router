### Target Date: 2026-06-23 (LegoNet on TOFU benchmark — author-level clustering)
- **Goal / Hypothesis:** Bring the LegoNet-inspired config (frozen base + n LoRA adapters keyed by
  frozen k-means, top-k k-NN routing, 1/k delta-average, affected-only deletion) onto the TOFU
  benchmark so it gets comparable OU-faithful `model_utility`/`forget_quality` in the same table as
  SISA/S3T/SEA. Forget unit = AUTHOR (forget10 = 180-199). Clustering at the AUTHOR level (mean of
  an author's answer embeddings) is the fix for the per-author centroid collapse that sank the
  earlier *record-level* TOFU attempt; here per-author structure is exactly what forgetting authors
  needs. Built INSIDE tofu_sisa_lora to reuse eval_tofu's metrics (user: "TOFU-benchmark comparison",
  "limit to 6 gpus").
- **Setup:** New code in tofu_sisa_lora: configs/legonet_tofu{,_smoke}.json, legonet_tofu.py
  (keys/assignment/KNNRouter/q2author), legonet_model.py (LegoNetRoutedModel + loaders),
  train_legonet_adapter.py, unlearn_legonet.py, prepare_legonet.py, test_legonet_tofu.py,
  submit_legonet_tofu.sh (%6 GPU cap). eval_tofu.py gained `--legonet_config`/`--legonet_unlearn_tag`
  (load branch parallel to `--preloaded_adapter`; evaluate_model/metrics/CSV unchanged).
  collect_results.classify_label gained a `legonet_` case. Recipe (configs/legonet_tofu.json):
  Llama-2-7B-chat, n=32/k=3, rank16/α32/[q,k,v,o]/6ep/lr2e-4, **use_rslora=False**, seed base+j.
  CPU tests: `python test_legonet_tofu.py` → 7/7 PASS. Smoke chain (TinyLlama n=4/k=2/1ep):
  `bash submit_legonet_tofu.sh configs/legonet_tofu_smoke.json all` → jobs 436047(setup)→436048(train
  x4)→436049(plan)→436050(unlearn)→436051(eval legonet_full+legonet_unlearn)→436052(collect).
- **Results:** Smoke setup (436047) GREEN: author_emb (200,384), keys (4,384), authors/adapter
  min31/max183/mean100, **empty_adapters=0**, q2author 3999/4000 (1 shared generic question, 0 touch
  forget), retain_tr_scores cached (30 samp, mean 0.814) by symlinking the SISA TinyLlama retain90.
  forget10 (20 authors) → affected adapters **4/4** = confirms R1 blast radius even at n=4. Train
  array running. [eval metrics pending — append on completion.]
- **Observations:** Integration is clean — eval_tofu's one-example-at-a-time RoutedModel seam let the
  top-k 1/k wrapper drop in with zero changes to the metric/CSV path. Author-answer clustering at n=4
  is imbalanced (31..183 authors/adapter); n=32 should be tighter (ideal kN/n=18.75). R1 holds:
  deleting 20 authors touches all adapters, so forget10's deletion-cost win over full retrain is
  small — the locality story belongs to single-author/forget01 deletions (planned follow-up).
- **Next Steps:** Verify smoke (legonet_full utility ≫ base; legonet_unlearn forget_quality up;
  unlearn touches only affected dirs; both rows land in all_metrics_smoke.csv). Then launch the 7B
  arm: `bash submit_legonet_tofu.sh configs/legonet_tofu.json all` (setup→32 train %6→unlearn→eval
  legonet_full/legonet_unlearn smoke+extended→collect). Compare against SISA merged/remerge + SEA.
- **Update (15:04) — smoke chain GREEN; 7B arm launched.** Smoke (436047→436052) finished ~14:41;
  both rows landed in all_metrics_smoke.csv (metrics_version ou-2026-06-10). Numbers (smoke caps,
  TinyLlama n=4/k=2/1ep):
  | label | model_utility | forget_quality |
  |---|---|---|
  | base_model | 0.4179 | 0.2391 |
  | legonet_full | 0.3836 | 0.808 |
  | legonet_unlearn | 0.383 | 0.5941 |
  Pipeline verified: unlearn_plan_436049.log → forget10 = 20 authors → 4 affected adapters [0,1,2,3],
  **0 untouched** (R1 blast radius confirmed at n=4 — all adapters affected, no locality, exactly the
  degenerate case flagged above; manifest re-trains affected, 0 hard-disabled). Both legonet rows sit
  well above base forget_quality (0.24), i.e. far closer to the OU retain reference than base.
  **Honest caveat / telemetry flag:** forget_quality goes the *wrong way* full→unlearn (0.808→0.594).
  Not trusting this as signal — at n=4 every adapter is affected so "unlearn" ≈ full re-merge, and
  TinyLlama@1ep is noisy; smoke's job was to prove the pipeline runs end-to-end (it does) and that
  rows reach the metric/CSV path unchanged (they do), NOT to show a forgetting effect. Real signal
  awaits the 7B n=32 arm where forget10 should touch a strict subset.
- **7B arm now in queue (`legonet_tofu.json`, %6 GPU cap):** train array 436062_[0-31%6] (tasks 2–7
  running on sprint1/sprint2 as of 15:04), then 436063 (plan, PD-dep) → 436064_[0-31%6] (unlearn) →
  436065_[0-1%2] (eval legonet_full/legonet_unlearn) → 436066 (collect). Also live: 436014 (exactness
  sample, sprint1, 2h14m elapsed). Append 7B legonet_full/legonet_unlearn metrics + SISA/SEA
  comparison when 436066 completes.

#### RESULTS (2026-06-23, cont.) — LegoNet-on-TOFU 7B (n=32/k=3) COMPLETE
- Setup (436061): keys (32,384), empty_adapters=0, retain_tr_scores cached. **Cluster imbalance
  1..135 authors/adapter (mean 18.8)** — one hub adapter holds 135/200 authors (TOFU answer-embeddings
  partially collapsed even at author level → many singleton adapters + one hub). forget10 -> 15/32
  affected, **17/32 untouched (byte-identical)**.
- Headline (smoke caps), Llama-2-7B-chat:
  | label | model_utility | forget_quality | forget_ppl | retain_ppl |
  |---|---|---|---|---|
  | legonet_full (knows forget) | 0.6277 | 0.0065 | 1.93 | 1.97 |
  | **legonet_unlearn (forget10 removed)** | **0.6371** | **0.808** | 7.37 | 1.94 |
  Reference (k=10 smoke): remerge_dare_ties mu 0.480/fq 0.393; merged_dare_ties 0.475/0.594;
  lorahub k=4 mu 0.59/fq 0.808; base 0.418; k=1 ft ceiling 0.744; SEA r8 ~0.78.
- **Verdict:** clean, correct unlearning at 7B — legonet_full KNOWS forget (fq 0.0065, forget_ppl 1.93),
  legonet_unlearn FORGOT it (fq 0.808, forget_ppl 7.37) while preserving utility (0.637, retain_ppl 1.94)
  and even raising retain_rouge. **Beats the SISA dare_ties merge family on the utility×forget trade-off
  (0.637/0.808 vs 0.48-0.58 utility) and matches lorahub's fq at higher utility**, while keeping deletion
  locality (17/32 adapters untouched). The smoke's inverted KS was a small-sample/undertraining artifact,
  as predicted. The hub imbalance did NOT make it degenerate.
- **Observations:** sits between SISA merges (~0.48-0.59) and the per-author/retain-core ceilings
  (SEA 0.78, retain-core 0.754). Imbalance is the obvious lever: balanced k-means / size-cap / larger n
  could push utility toward the 0.74 ceiling and tighten the locality (fewer authors per hub → smaller
  forget blast radius).
- **Next Steps:** (1) extended-caps eval for a publication-grade number (`LEGO_EVAL_ARGS=--extended
  LEGO_PREP_SUB=extended bash submit_legonet_tofu.sh configs/legonet_tofu.json eval`); (2) balanced-kmeans
  / larger-n variant to test the imbalance lever; (3) single-author + forget01 deletions to showcase the
  cascade-free locality (where untouched-adapter count is highest). Core arm COMPLETE + comparable.

#### Follow-ups launched (2026-06-23, cont.)
- 7B (n=32/k=3) deletion-locality gradient (free, from assignment): single-author 3/32 affected
  (29/32 untouched), forget01 5/32, forget05 10/32, forget10 15/32 — cascade-free locality scales
  sublinearly with forget-set size (20 authors touch 15 not 60 due to top-k overlap). The locality
  showcase the forget10 headline understated.
- Added balanced (anti-hub) assignment: capacity-capped top-k (cap=ceil(1.5*k*N/n)=29) in
  legonet_tofu.py (cfg "balanced"/"capacity_slack"); caps the 135-author hub. Unit + toy checks pass.
- Launching the open-unlearning canonical 1B model (meta-llama/Llama-3.2-1B-Instruct; user said "3.1
  1b" — that model doesn't exist, 3.1 is 8B+, the 1B is 3.2): vanilla + balanced arms, n=32/k=3,
  %3 GPUs each (<=6 total). Configs: legonet_tofu_llama3p2_1b{,_balanced}.json.

