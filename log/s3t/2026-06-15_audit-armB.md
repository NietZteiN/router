### Target Date: 2026-06-15 (S³T faithful-repro audit + gap closure; armB contrast)
- **Goal / Hypothesis:** Audit the S³T implementation against the consolidated technical
  reference and close the remaining faithfulness gaps. Verdict: **implementation is faithful** —
  Alg 1 (cyclic rotation), Alg 2 (BMS + Eq-24 + Hungarian), Alg 4 (deactivate {l′..L}, serve best
  survivor), cumulative top-down slice training (armA = Table 2 Llama2-7B row), Lemma 1 (sim δ
  within <0.4% of mL·H_{mB'}) all match. No mechanism bugs.
- **Setup:** added the missing reference deliverables (all CPU except armB F-evals):
  RQ3/Fig-8 diversity (`s3t_rq3.py`, 97f43f48e092), Lemma-2 retention closed forms Eq 18/20 +
  `expected_retrains` (`s3t_deletion.py`, 36ac13820e7a), and report sections — Lemma-2 overlay,
  **faithful Fig-9** (cumulative deletion time over 1000-request stream via δ, replacing the
  misleading per-deletion number), storage Table 3, RQ3 fold-in, and an **armB perf-vs-deletions
  contrast** via `--src2` (`s3t_experiments.py`, 858d4a33fcec). New tests
  (`test_s3t_sequences.py`, 8e2d7d1461a8): Lemma-2 monotonicity/caps + empirical-vs-closed-form
  retention + RQ3 diversity — all green. Orchestrator `submit_s3t_faithful.sh` (f247c6d7e56f):
  armB depth dirs (from existing stage snapshots, no training) → F-eval array **434856** (depth
  1-4 %4) → finalize **434857** (collect armB F + RQ3 + report with armA/armB curves). seed 42,
  uniform deletion prior, ≤4 GPU.
- **Results:** report sections verified on the real armA F-curve: Fig-9 **S3T 1.60× vs SISA,
  71× vs full-retrain** (cumulative, 1000 requests); Lemma-2 sim matches closed form (random
  sequences) with cyclic ≥ it; RQ3 cyclic edit-distance 5.0 (=L) > random 4.1, BMS maximally
  diverse (=L, Lemma 3); storage Table 3 (320 MB/shard × B × m). armB F(d) curve pending 434857.
- **Observations:** two genuine catches during the audit — (1) the reference's printed Eq-18
  `1-(k/L)^r` is **inconsistent** with its own Eq-21 derivation and with S3T(B=1)=SISA; the
  self-consistent form is `(1-k/L)^r` (implemented + tested). (2) "BMS > sorted-cyclic on score"
  (Fig 15) does **not** robustly reproduce — at t=1 all position-diverse sets tie by construction;
  reported descriptively rather than asserted. armA stays near-base at all depths (paper-faithful
  HPs undertrain on TOFU) — armB is the informative contrast.
- **Next Steps:** confirm armB F(d) + final report when 434857 lands; armB expected to show real
  degradation (0.58 → toward base) vs armA's flat curve.

