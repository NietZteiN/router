### Target Date: 2026-07-16 (composable_tv thread pre-registration — arms, protocol, gates)
- **Hypotheses / what we're testing:** Can training-time structure make per-author task vectors
  compose by **weight-1.0 summation into a single merged model (no serve-time task-ID/router/mask)**
  with **exact deletion by subtraction** — the empty cell of
  [PATHS_FORWARD §5](../PATHS_FORWARD_2026-07-13.md)? Prior to beat: the H8 knee
  (own-author ans-prob 0.999→0.885@N=2→0.615@N=3→0.282@N=8→~0.21 plateau,
  [h8-e25-ladder-results](../merge_mechanism/2026-07-09_h8-e25-ladder-results.md)), lazy keys
  (on/off 1.10, [key-firing-results](../merge_mechanism/2026-07-15_key-firing-results.md)), the
  refuted self-gating penalty ([negative-anchor-pilot-results](../merge_mechanism/2026-07-16_negative-anchor-pilot-results.md)),
  and the SIFT maskless-merge collapse (mu 0.407 vs masked 0.737,
  [t200-results](../sift_masks/2026-07-02_t200-results.md)).
  - **[lin] H-lin-1 (memorization):** linearized (tangent-space) per-author training at the e25
    budget reaches solo own-author answer-prob ≥0.9× its matched standard-LoRA control (frozen-A
    twin). CONFIRM ≥0.9×; PARTIAL 0.5–0.9× (proceed, rebaseline H-lin-2 relatively); REFUTE <0.5×
    after ONE pre-registered retry (rank 64 OR lr 5e-4, not both).
  - **[lin] H-lin-2 (composition):** under linearized serve, signal-remaining
    (recall−floor)/(iso−floor) ≥80% at N=3 and ≥60% at N=8 (control: 51%/9%). REFUTE if within
    ±10 pts of the control profile at both N.
  - **[lin] H-lin-2b (partial-linearization transfer):** tangent-trained vectors under NONLINEAR
    serve move the 50%-signal knee from N≈3 to ≥6. REFUTE knee ≤4.
  - **[lin] H-lin-3 (subtraction):** subtract-vs-recompose max|Δlogit| ≤1e-4 fp32; post-subtract
    probe forget_rouge within ±0.02 of base floor; headline (Σ200 − forget10) fq at k=10 in the
    exact-track band and MIA loss-AUC ≤0.38.
  - **[lin] H-lin-4 (disentanglement):** ξ(i,j)=mean‖f(θ0+τi+τj)−f(θ0+τi)−f(θ0+τj)+f(θ0)‖ over
    probe prompts, median over 10 author pairs, ≥2× smaller for tangent-trained than standard.
  - **[lin] H-lin-5 (linear crosstalk, the honest risk):** mu@N=20 within −0.02 of N=1 under
    linearized serve. REFUTE = monotone mu decay with N — itself a claimable "second wall"
    mechanism datum tying to lazy keys.
  - **[wd] H-wd-1 (learnability):** projected-SGD col(B_a)⊆span(Q_a) solo ≥0.95× control solo.
    KILL <0.8× (then and only then the pre-registered W4 soft-penalty fallback is designed).
  - **[wd] H-wd-2 (H3 adjudicator, headline):** orthblock-SUM at N=8 retains ≥2× the
    control-MEAN's extractable fraction (absolute anchor: prob ≥0.56 vs control ≈0.28 — the
    H-anchor-2 bar, kept for cross-thread comparability). CONFIRM ⇒ col(B) parameter collision is
    causal. REFUTE with H-wd-1 confirmed ⇒ overlap-as-protection
    ([interference-vs-n-results](../merge_mechanism/2026-07-08_interference-vs-n-results.md)
    ρ=−0.675) is causal — publishable either way.
  - **[wd] H-wd-3 (collateral):** N=16 sum mu ≥ control additive_mean mu −0.02; retain_ppl
    explosion (the λ-sweep signature 8→1.8M) = auto-refute. Prediction: disjointness ⇒
    ‖Σδ‖²=Σ‖δ‖², no overshoot. Watch √N hidden-state perturbation growth as the failure mode.
  - **[wd] H-wd-5 (granularity):** rowslice within 0.05 recall of orthblock at N=8.
  - **[wd] 2×2 at N=8 pre-registered:** {control,orthblock}×{mean,sum} — control-sum reproduces
    overshoot, orthblock-mean reproduces 1/N dilution, ONLY orthblock-sum rescues.
  - **[ds] H-ds-1 (learnability):** hard seeded disjoint-SUPPORT full-FT (gradient-masked to the
    support; sign constraint replaced) solo ≥0.95× the sift-style unconstrained solo. Density
    pilot 0.5%; sweep {0.05, 0.1, 0.5}% as ablation (capacity: N×density ≤100%).
  - **[ds] H-ds-2 (headline):** merge-only serving (NO mask, NO task-ID) beats the sift maskless
    floor (mu 0.407 ≈ base 0.42) and retains ≥2× control extractable fraction at N=8. This is
    SIFT's open question: their diagnosis of maskless collapse is parameter-level contamination
    ("non-zero weights in entries where the local weight is zero"), which disjoint supports remove
    by construction — the residual unknown is activation-level cross-talk.
  - **[ds] H-ds-3 (cross-talk):** solo-vs-all-N own-recall gap ≤0.15 abs with locality verified
    (the external report's kill signal: bigger gap ⇒ disjointness bought nothing over SIFT).
  - **[ds] H-ds-4 (exactness):** subtraction bitwise-exact (sift `merge_sub_` precedent).
  - **[w5] H-w5-1 (calibration null):** on the 7B e5 pool, no deterministic post-hoc setting
    (factor-DARE p∈{0.5,0.9}, top-q=0.25 row-norm keep, hash-disjoint row masks) reduces the N=8
    mean own-recall drop by more than ⅓ (0.073→<0.049). Anything that does jumps the queue as a
    training-free method.
  - **[w5] H-w5-2 (=DX2):** post-hoc hash-disjoint truncation destroys iso recall in proportion
    to energy removed (≈1/N) — unconstrained facts do not live in assignable regions a priori ⇒
    a training-time constraint is NECESSARY (with H-wd-2/H-ds-2: necessary-and-sufficient).
  - **DX1 (cancellation diagnostic, CPU, closes the W3 idea-space):** per-coordinate |Σδ|/Σ|δ|
    over N∈{8,32,200} sums vs a sign-shuffled null. Observed cancellation ≤ null ⇒ elementwise
    sign-fixing has no headroom on this pool.
  - **Cross-cutting: H-CT-mia** post-deletion served composition loss-AUC ≤0.45 (oracle floor
    0.379, live `*_full` control ≥0.59; settings ≡ configs/deletion_audit.json).
    **H-CT-diff (severity, no safety claim):** for sum composition pre−post ≡ τ_u EXACTLY — the
    checkpoint/τ_u adversary reconstructs the author at fidelity ≈ solo recall (≈1.0). We claim
    served-surface cleanliness only; the diff channel is REPORTED as a measured limitation
    (mitigation = checkpoint custody, discussion not experiment).
    **H-CT-loc ([wd]/[ds] locality gate, the MemSinks lesson):** ≥95% of ‖τ_u‖² in the owned
    region; per-author owned-norm ≥ ε·median (empty-slice check); owned-zeroing kills own recall
    to ≤ base+0.05 while an equal-size seeded random placebo moves it <0.05 and mu <0.02.
  - **Killed arms (on the record):** **W2 rank-slicing** — block-diagonal concat of independently
    trained authors is function-space IDENTICAL to the additive sum (`cat` ≡ `additive`, unit-tested
    in test_merge_extra.py); the joint-training variant destroys per-author exactness. Nothing to
    run. **W3 sign-fixed LoRA** — elementwise sign constraints are not closed under the low-rank
    parameterization (only row-wise v with B_ik·v_i≥0, A≥0 is enforceable; sign-project-then-SVD
    has no fixed point), and a LoRA delta is dense so SIFT's masks buy nothing while the maskless
    SIFT merge is already collapsed; replaced by DX1. **W4 soft write-collision penalty** —
    dominated by [wd]'s hard projection (exact property at zero hyperparameter cost); fallback
    only if H-wd-1 refutes, with a shrinkage-vs-redirection readout to distinguish it from the
    refuted H-anchor-1 failure mode (which was data-conditioned input-selectivity — different math).
- **Setup:** Base `meta-llama/Llama-3.2-1B-Instruct` (fp32 for [lin]; bf16 elsewhere per frozen
  recipe); [w5]/DX on the existing 7B e5 pool `checkpoints/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4/`.
  Author universe `perm = np.random.RandomState(42).permutation(199)` (author 199 held out);
  probes `perm[:5] = [82,15,111,177,76]`; pilot pool `perm[:20]` ([wd] uses `perm[:16]` —
  capacity bound pool·r′ ≤ k/v d_out 512 at r′=32); N-ladder {1,2,3,4,6,8,12,16,20} ([wd]
  ∩{1,2,3,4,8,16}), +{32,64,128,200} winner only. NEW 1B pools
  `Llama-3.2-1B-Instruct_ctv_{ctrl,lin,wd,ds}_*` (no 1B per-author pool exists — verified);
  recipe = frozen (r32/α64/rsLoRA/lr 1e-4/seed 42) at **epochs 25** (the H7 fix). Composition =
  weight-1.0 sum (new `sum` mode in merge_subset.py); control runs BOTH sum and mean. Serving:
  materialized PEFT dirs via `eval_tofu.py --preloaded_adapter` (+ new `--linear_tv_config`
  bypass for [lin] linearized serve, mirrored in attack_mia.py). Ladder evals
  `--k 200 --forget_shard_id 199 --smoke` (+`--eval_shard_id`/`--retain_author_ids`); deletion/MIA
  arm `--k 10 --forget_shard_id 9`; KS refs copied per pool (established pattern). Labels
  `ctv_{ctrl,lin,wd,ds}_{sum,mean}_N{n}_s42`, `iso_a{author}`. New code (each with a CPU gate,
  sha256s recorded in the build entry): linear_tv.py / train_linear_tv.py / test_linear_tv.py ·
  struct_bases.py / train_struct_tv.py / verify_struct.py / test_struct_tv.py · [ds] support mode
  on the sift spine + test · sparsify_pool.py · verify_subtraction.py + test · attack_mia
  `--dump_scores` + attack_diff.py + test ext · merge_subset `sum` mode + test case ·
  submit_ctv.sh (STUB=1, arrays %2 while Exp-6 is live, self-skip, afterok, scancel-on-gate-fail) ·
  configs/ctv_1b_{ctrl,lin,wd,ds}.json + configs/sparsify_7b.json · analyze_ctv.py. Gates:
  **G0** CPU green + STUB previews → **G1** solo ≥0.95 (per-arm bars above) → **G2** N=8
  extractable ≥2× control + mu ±0.02 (all arms dead ⇒ close with the negative boundary result) →
  **G3** exactness classes (bitwise/algebraic ≤1e-6/first-order/approximate, declared per arm,
  never conflated) + MIA + diff severity + locality → **G4** winner: 200-author pool, ladder to
  200, forget10 deletion at k=10, extended caps, seed-43 replicate. Statistics: seed 42; 43/44
  winner-headline + noise-band gates only; 5 probes for curves; 20 paired authors at N=8 for the
  winner (Wilcoxon signed-rank, p<0.05, effect ≥2× the ±0.02 band); per-author distributions +
  failure-tail fraction ALWAYS (mu is a guard-rail — blind to per-author collapse by
  construction). Budget: ≤150 GPU-h hard cap (realistic kill-gated ≈120); ≤2 concurrent GPUs
  while Exp-6 holds cap share (`squeue -u jack` before every submit); ≤30 GB thread-resident,
  merges transient (deleted after result JSONs land; merge_meta.json kept); don't launch G4 with
  <60 GB free on /storage2. SLURM: sprint1–3, exclude sprint4; train 00:30, 1B eval 00:45,
  7B eval 01:30 ([lin] eval 01:30 — no KV cache), CPU merges no-gres. Job IDs + script sha256s
  recorded per entry.
- **Results:** [pending]
- **What worked / hypothesis verdict:** [pending]
- **Observations:** [pending]
- **New questions / new hypotheses:** [pending]
- **Next Steps:** Wave 0 build (all CPU gates green + STUB previews + DX1 + [w5] merge build via
  CPU SLURM) → Wave 1 (G1: 1B control pool 20 + [lin] 20 + [wd] 32 + [ds] 16 trains, iso +
  placebo evals) → per-arm verdict entries → Wave 2 (G2 ladders + [w5] wave) → G3 audit →
  G4 winner only.
