### Target Date: 2026-07-02 (follow-up battery E0–E6 — seed rigor, α mechanism, routing audit, k-sweep, throughput, N=15 deletion)
- **Hypotheses / what we're testing:** H1 (rigor): the router>1/k gap survives router-seed variance
  (CONFIRM: |mean gap| > std over seeds; REFUTE: gap within noise). H2 (mechanism): the learned α
  deviates from uniform, and sharper routing → better memorization (merge_mechanism collision
  prediction; REFUTE: α≈uniform or ρ(sharpness, EM) ≤ 0). H3 (audit, §9-D + §7): the stale expert
  index (embeds forget data, never rebuilt on deletion) is a material leak channel — rebuilding it
  retain-only should move fq toward the key ceiling. H4: 1/k dilutes with serving k while the learned
  α resists (router−1/k gap grows at k=5/8). H5: batched union+mask serving approaches single-LoRA
  throughput parity as batch grows. H6: the N=1 deletion result replicates at N=15.
- **Setup:** E0 first: `encoder_pin` config pin (fixes `_encoder_source` silently returning the FT
  retriever — all audit outputs now record the resolved encoder). New instrumentation built by a
  4-builder/4-verifier workflow, all CPU-gated (9/9 suites): α-capture in `RouterLoraLinear.forward`
  (opt-in, teacher-forced b=1 only) + `analyze_router{,_tofu}.py`; `routing_audit{,_tofu}.py`
  (rank-preserving top-k, fresh router per policy, distinct `_ex*` index files, stale index
  sha-asserted untouched); `benchmark_serving.py`; seed configs `_s43/_s44` (DBpedia) and
  `--seed/--router_out` (TOFU); serve-time k configs `_k5/_k8`; `submit_followup.sh` (14 SLURM jobs
  440213–440226, ≤4-GPU caps, E3-DBpedia audit chained after the E6 `d_batch15` deletion);
  `collect_followup.py` → `/storage2/jack/checkpoints/ramole/FOLLOWUP_REPORT.md`. Verifiers caught 3
  real bugs pre-launch (silent random-init-router diagnostic; sorted-tuple top-1 metric; report-renderer
  schema drift). E6: 15 seeded records (`random.Random(42)`, excl. rec_000000-2) → tag `d_batch15`.
- **Results:**
  - **E1 (seeds 42/43/44):** DBpedia router−1/k em gap **+0.005 ± 0.001** (router em .647/.649/.648
    vs 1/k .643) → robust. TOFU full mu gap **0.000 ± 0.002** → within noise. TOFU unlearn mu gap
    **+0.007 ± 0.001** (mu .507/.510/.507 vs .501; fq .890/.988/.954) → robust.
  - **E2 (α):** DBpedia router is **≈uniform**: H_norm 0.982±0.011, max-share 0.368 (uniform 1/3),
    ideal-mass 0.333 (exactly uniform); d0-router 0.976 (dropout Δ≈nil). TOFU router **genuinely
    sharper**: H_norm 0.814, max-share 0.544 — but ideal-mass 0.302 < 1/3 (sharpens AWAY from the
    frozen-key ideal). Sharpness↔memorization is **NEGATIVE** on DBpedia: ρ(sharp, em) = **−0.52**
    (p≈0), ρ(sharp, −ln ppl) = −0.51; TOFU full +0.248, unlearn −0.093. Post-unlearn the router gets
    more uniform on forget content (H .810→.828) while forget ppl 2.97→11.82, retain ppl flat 2.85.
  - **E3 (routing audit):** TOFU embed routing reproduces the frozen top-3 only **11.5%** (top-1 in
    K(a) 68.5%; affected-mass 0.730). Rebuilt retain-only index: orphan rates drop slightly (top-1
    0.618), **10.7%/8.3%** of retain queries shift top-k/top-1 — yet downstream fq/mu ≈ unchanged vs
    stale-FT (`ramolerb_full` 0.466/**0.484**, `ramolerb_unlearn` 0.474/**0.180** ≈ `ramoleft_*`).
    DBpedia: orphan top-1 **1.000** under both policies (deleted records still route to their
    RETRAINED experts — where the scrubbing lives); retain shift grows with deletion size (topk
    0.5%→12.5%→7.5%→**29%** for d0/d1/d2/d_batch15) but per-record eval outcomes identical
    stale-vs-rebuilt (em/ppl equal to 3 decimals); key routing shift **0** (asserted). Index
    displacement small (cos ≥ 0.94 affected; untouched bit-equal, modulo the documented shared-
    RandomState resampling on DBpedia).
  - **E4:** all 4 evals crashed on a path bug — the serving k leaked into the SOURCE assignment path
    (`assignment_n32_k5.json` doesn't exist; the source is fixed at k=3). Fixed
    (`_source_assignment_path` fallback; CPU-verified k>src re-route + k=src identity); retry array
    440426 running — results in a follow-up entry.
  - **E5 (throughput, greedy 64 tok):** batched RAMoLE scales ~linearly: **5.6 / 20.6 / 41.2 / 82.9
    tok/s** at b=1/4/8/16 vs merge-per-group flat ≈12 (6.9× at b=16) vs single-expert 190 (parity
    ratio 0.44 at b=16, rising). At b=1 the live-attention path is ~2.1× slower than a pre-merged
    adapter.
  - **E6 (N=15 deletion, router unchanged):** router em 0.642→0.564 (Δ−0.078), ppl 5.34→8.96,
    canary_em .056→.035; 1/k em 0.637→0.564 (Δ−0.073), ppl→9.12. N=15 vs the earlier N=1 anecdotes.
- **What worked / hypothesis verdict:** **H1 SUPPORTED** where it matters (DBpedia iid +0.005±0.001;
  TOFU unlearn +0.007±0.001) and honestly REFUTED on TOFU full (0.000±0.002). **H2 half-supported,
  half-refuted:** the TOFU router IS non-uniform (H 0.81) and even DBpedia deviates measurably — but
  the sharper-routing→better-memorization prediction is REFUTED (ρ=−0.52: the router sharpens on
  hard/poorly-memorized records, or sharpening hurts; either way the naive collision story fails).
  **H3 REFUTED:** index staleness is measurable at the routing level (10.7% retain shift on rebuild)
  but does NOT move fq/mu — the encoder (retriever FT) dominates the embed-route leak, not the index
  rows; and on DBpedia orphans keep routing to their scrubbed experts, so staleness is benign there.
  **H4 pending** (retry running). **H5 DIRECTIONALLY SUPPORTED:** linear batch scaling, 6.9× over the
  merge path, but only 44% of single-LoRA parity at b=16 — "approaches parity" is a trend, not an
  achieved state. **H6 SUPPORTED** (clean N=15 forgetting, router untouched, ≈ 1/k).
- **Observations:** (1) The α result is the missing mechanism for the whole thread: on homogeneous
  DBpedia clusters the router learns to stay uniform (ideal-mass EXACTLY 1/3) — router≈1/k isn't a
  failure to learn, it's the correct solution to that pool; TOFU authors are heterogeneous enough to
  sharpen (max-share 0.54). (2) The negative sharpness correlation plus ideal-mass < 1/3 on TOFU
  suggests the router allocates attention by difficulty, not by cluster identity. (3) The §9-D-style
  audit cleanly separates the leak channels: encoder ≫ index staleness; and retrain-style deletion is
  fundamentally safer than drop-style (orphans route INTO the scrubbed experts, not to siblings).
  (4) The E4 crash was an integration bug my CPU fixture couldn't catch (fixture source k == serving
  k) — the retry test now covers k>source-k explicitly. (5) fq seed spread on TOFU unlearn
  (.890–.988) is large; fq comparisons finer than ~0.1 are noise at these caps.
- **New questions / new hypotheses:** Does the router-seed-robust unlearn-mu gap (+0.007) survive
  EXPERT-seed variance (retraining experts, not just the router)? Does TOFU's difficulty-directed
  sharpening (ideal-mass<1/3, ρ<0 on DBpedia) mean an entropy-REGULARIZED router (encourage uniform)
  would match it — i.e. is the learned gate's benefit purely from mild denoising? Can the batched
  path reach >0.44 parity with a fused kernel (v computed once per union expert, currently
  recomputed per token step)? H4 (k-dilution) still open pending 440426.
- **Next Steps:** fold in the E4 retry results (follow-up entry); consider the entropy-regularized
  router control; the heterogeneous cross-task pool (E7) remains the setting where sharpening should
  pay off — TOFU's max-share 0.54 supports that expectation. Artifacts: `FOLLOWUP_REPORT.md`,
  `routing_audit*.json`, `alpha_diag_*.json`, `throughput.json`; SLURM 440213–440226 + retry 440426.
