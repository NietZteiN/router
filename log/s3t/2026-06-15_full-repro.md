### Target Date: 2026-06-15 (S³T full paper reproduction on TOFU — sequence-aware budget + deletion-rate experiments + SISA baseline)
- **Goal / Hypothesis:** Faithfully reproduce the S³T paper (arXiv 2406.16257) on TOFU,
  beyond the core we already ran (slice-wise training + ensemble + single deletion). Add the
  three namesake pieces: (1) budget B>1 multi-sequence selection — iterative cyclic rotation
  (Alg 1) + BMS (Alg 2, Eq 24); (2) the deletion-stream procedure (Alg 4, best surviving model
  per shard); (3) the headline experiments — deletion rate δ (Figs 6-right/7, Lemmas 1-2),
  performance-vs-#deletions (Fig 6-left), deletion time (Fig 9) — with a SISA baseline (B=1).
  Primary recipe = paper-faithful **armA** (Table 2 Llama2-7B row: r32/α64, lr 2e-5, 3 ep/stage).
- **Setup:** key decomposition (verified against the paper's own theory): δ = pure CPU
  coupon-collector simulation over slice orderings (no GPU, no trained models); performance
  retention = F(d) (utility when each shard retains d slices) composed with the surviving-depth
  distribution — F(d) is the SAME function for SISA and S3T, sourced from the EXISTING armA
  `shard_i/stages/stage_{d-1}` snapshots (zero new training). New code: `s3t_sequences.py`
  (10874fd61ac6), `s3t_deletion.py` (f0e1ee324d01), `s3t_measure_F.py` (0238c29c0ae5),
  `s3t_deletion_time.py` (4b0f2dd305e0), `s3t_experiments.py` (298554e0329b),
  `test_s3t_sequences.py` (43f825241e06), `submit_s3t_repro.sh` (a7886f706bab); extended
  `train_s3t_shard.py` with `--ordering/--seq_id` for budget B>1 (opt-in). m=5, L=4, uniform
  deletion prior, seed 42. SLURM chain (≤4 GPUs): build depth dirs (CPU) → F-eval array
  **434748** (depth1-4 + armA-full ensemble_probs, 5 tasks %4) → deltime **434749** → finalize
  **434750** (collect F + s3t_experiments + collect_results). `submit_eval.py` etc. unchanged.
- **Results:** CPU validation PASSED before submit — `test_s3t_sequences.py` green; deletion-rate
  simulation matches Lemma 1 (mL·H_{mB'}) within <0.4%: δ(m5,L4)= **45.8 (B1=SISA) → 58.3 (B2)
  → 71.9 (B4)**, saturating at B=L (B8≈71.9). Full report path validated on CPU with placeholder
  F: **S3T(B=4) handles ~1.59× more deletion requests than SISA** (matches the paper's ~1.6×),
  gains grow with m and L, performance-vs-deletions stacks S3T > SISA decaying to base. Real F(d)
  + deletion-time pending the GPU chain (434748-50) → `reports/S3T_PAPER_REPRO_2026-06-15.md`.
- **Observations:** the paper's headline (deletion rate) is purely combinatorial, so the
  expensive part (training) was already done; F(d) reuses existing snapshots. B>1 actual training
  is opt-in (`TRAIN_B=1`) for mixed-depth Alg-4 validation; not needed for the core figures.
- **Next Steps:** read F_curve.json + report when 434750 lands; if armA F(L) sits near base
  (undertrained at the paper's exact HPs), note as a faithful finding and optionally add armB
  contrast. Mixed-depth GPU validation of the F(depths).mean() composition if higher fidelity wanted.

