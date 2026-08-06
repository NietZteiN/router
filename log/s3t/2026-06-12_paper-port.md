### Target Date: 2026-06-12 (S3T on TOFU — paper-faithful port + overnight eval chain; separate thread from the k-scaling entry above)
- **Goal / Hypothesis:** Evaluate S³T (ICLR'25, official code `~/S3T`) on TOFU with maximum
  model utility. S3T = within-shard slices trained in L sequential stages, each stage updating
  a disjoint top-down layer block of LoRA params on CUMULATIVE slice data; deletion = revert
  the affected shard to its pre-forget-slice snapshot (exact, O(1)). Instantiation: m=5 shards
  × 40 authors, L=4 slices × 10 authors, num_loras=8 (all 32 Llama-2-7B-chat-hf layers);
  orderings cyclic (shards 0-3) / forget-last (shard 4; BMS point-mass-prior outcome), B=1.
  Aggregation = NEW inference-time ensemble (`ensemble.py`): token-level distribution average
  across shard adapters (`ensemble_probs` = paper's "average output vectors"; `ensemble_logits`
  ablation). Hypothesis: full-state ensemble_probs ≫ merged baselines (0.48-0.59) because the
  one knowledgeable constituent's sharp distribution survives arithmetic prob-averaging, and
  the deletion state (ensemble with shard_4@stage_1) keeps mu within 0.05 while forget signals
  go to oracle level.
- **Setup:** seed 42; configs `tofu_sisa_lora/configs/s3t_arm{A,B}.json` (armA = official
  recipe lr 2e-5/e3 per stage; armB = k=1-winner transfer lr 1e-4/e5; shared r32/α64/rslora/
  7 proj modules/bs1×ga16/max_len 256/linear sched). Deviations from official code (documented):
  bf16 not 8-bit; TOFU Question:/Answer: format not Alpaca; single LoRA adapter + per-stage
  requires_grad masks (equivalent: disjoint blocks, zero-delta untrained blocks — proven by
  test_s3t.py) not stacked per-stage LoRAs; **exact layer-id regex (official `check_if`
  substring bug would break exactness at single-digit layer ids — found during port)**.
  CPU gates green pre-submit: test_s3t.py, test_ensemble.py, test_merge_extra.py,
  test_ou_equivalence.py. Eval: existing OU-faithful eval_tofu.py, `--k 10 --forget_shard_id 9`,
  smoke caps; KS ref copied from `Llama-2-7B-chat-hf_ft`. New label prefix `ensemble_*`
  (merge_lora dispatch; opt-in only). Overnight chain (`submit_s3t_overnight.sh`, ≤4 GPUs ours,
  gates scancel dependents on failure): gate-train **434633** (1B micro S3T) ∥ gate-ens
  **434634** (1B k4 ensemble_probs) → train **434635** (10×%4: 2 arms × 5 shards, full L=4
  chain each) → verify **434636** → eval **434637** (12×%4: {armA,armA_del,armB,armB_del} ×
  {ensemble_probs, ensemble_logits, shard_4_only}) → collect **434638** → pick **434639**
  (`s3t_pick_winner.py` applies the pre-registered rule: winner = max full-state ensemble_probs
  mu s.t. del-state mu drop ≤0.05 ∧ fq ≥0.39 ∧ |f_rouge−0.393| ≤0.10; tie-breaks probs>logits,
  armA>armB; then auto-submits the extended tier: retention-curve smoke `_t0`/`_t2` ×2 labels
  %2 + prepare_eval --extended + full/del extended evals %2). Dirs:
  `checkpoints/Llama-2-7B-chat-hf_s3t_m5_L4_{armA,armB}[_del]`; del dirs are symlink dirs with
  shard_4 → `shard_4/stages/stage_1`. Script sha256 (first 12): train_s3t_shard 753cbeedee37,
  ensemble da969cc609fe, shard_utils a56a544c47f4, s3t_gate_checks 9b38cf28a3d3,
  s3t_pick_winner d55110923555, submit_s3t_overnight 1ae4967ccf91, test_s3t 4691dbab30c0,
  test_ensemble 72c13c088ee8, configs 580996379b9b/c7621dd69a3a. Repo not git.
- **Results:** pending (overnight). Decision artifacts on completion:
  `checkpoints/s3t_winner.json` + `checkpoints/s3t_decision_table.md` + refreshed
  `all_metrics_smoke.csv` (+ `_extended` if a winner qualified).
- **Observations:** trl 0.9.6 SFTTrainer re-derives the tokenizer from
  `model.config._name_or_path` when `tokenizer=` is omitted (breaks for in-memory models —
  now passed explicitly). Login node exposes a GPU: CPU test suites force
  `CUDA_VISIBLE_DEVICES=""`. Cross-reference: today's k-scaling entry found `routed_key_exact`
  @k=50 mu 0.7147 — the bar the S3T ensemble should be compared against tomorrow, not just the
  merges.
- **Next Steps:** morning: `sacct` over 434633-434639 (no FAILED/TIMEOUT), read the decision
  table, append results + S3T-vs-routing-vs-merging comparison; retention curve t0→t3 plot;
  seed-variance pass before claims.

**Update (2026-06-12 — fix evals landed; sweep complete; report written):**
- 434626/434627 finished (sprint2, no contention); collect 434628 refreshed the CSV. Final cells: k100 remerge_dare_ties mu 0.4301 (≈ merged 0.4299, free forgetting); k100 shard_99_only 0.4748/f_ppl 2.43; **k100 routed_key_exact_no99 mu 0.6475 — identical to full routed**, f_ppl 2.43→6.71; k200r8 merged_dare_ties 0.4201 ≈ base; **k200r8 routed 0.4728 → _no199 0.4728**, f_ppl 9.60→13.09. Deletion under routing is utility-free at ALL three k.
- Full report: `tofu_sisa_lora/reports/SCALE_REPORT_2026-06-12.md` (newcomer glossary with ↑/↓ metric directions, frontier table, routed T1-T3, fp32 memory law, infra findings, complete 28-row appendix).

