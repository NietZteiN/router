### Target Date: 2026-07-20 (spec-v2 pre-registration — ReLU gated-branch recipe)

Supersedes [`2026-07-20_preregistration-build.md`](2026-07-20_preregistration-build.md)
(append-only: the earlier entry stands as the record of the SwiGLU/λ draft; **zero SLURM
jobs and zero GPU spend happened under it**, so nothing was run under a retired spec).
Between the two entries the user supplied Vincent Hanke's authoritative training recipe
(recorded in the approved plan `~/.claude/plans/include-these-details-as-mellow-barto.md`
Part 2, executed same-day across sessions), and the implementation was brought to it
exactly. This entry re-pins hypotheses and setup for the runs that will actually happen.

- **Hypotheses / what we're testing:** H1–H5 + H-scale carry over from the first entry
  with bars unchanged EXCEPT the G2 clause that referenced a λ=0 control (the lean pilot
  has none — revision pre-registered below).
  - **H1 (localization, make-or-break):** disconnected per-author ReLU-gated branches +
    the 4-term recipe reach SELECTIVE off-author firing where LoRA anchoring failed
    (anchor pilot 2026-07-16: selectivity 1.11 at every λ, 100% LAZY). CONFIRM: some
    pilot arm has median on/off output selectivity ≥ 5.0 with median own-author
    answer-prob ≥ 0.80 (absolute bar; no λ=0-relative clause — REVISED, see G2 below).
    REFUTE: no arm reaches median ≥ 2.0.
  - **H2 (all-active serving retains utility):** all-active vs own-only own-prob gap
    ≤ 0.05 at the pilot; at K=200, OU Util.R ≥ 0.95 and Util.G ≥ 0.95 (MemAdapt FT row
    1.075/1.024). REFUTE: ≥ 0.15 all-active drop (memsinks-magnitude, mu 0.4373 vs 0.6438).
  - **H3 (deletion clean):** drop forget10 (authors 180–199, text-join) ⇒ Mem ∈
    [0.55, 0.70] (Retrained 0.590, MemAdapt 0.630), |ΔUtil.R| ≤ 0.03, Agg ≥ 0.80
    (strong-confirm 0.84–0.90 vs MemAdapt 0.869); `sepmlp_dropall` ≡ `calib_base`;
    deletion timed (memadapt block-list anchor 0.027 s).
  - **H4 (relearn parity):** median steps-to-answer-prob-0.8 ratio target/control ∈
    [0.8, 1.25] and fixed-budget |Δprob| ≤ 0.10 over 5 forget/5 holdout pairs per method.
    REFUTE: deleted authors relearn ≥ 2× faster.
  - **H5 (negative-example leak — measurement, no bar):** Priv + 4 raw MIA AUCs on
    `sepmlp_unlearned`; direction attributed (residual memorization vs over-suppression).
    Anchors: MemAdapt Priv 0.917; exact-drop threads ≤ oracle floor AUC 0.379.
  - **H-scale:** K=200 median selectivity ≥ 5.0 and ≥ 0.7× the pilot winner; all-active
    vs own-only gap ≤ 0.05 at K=200 (gate G3, before any eval spend).
  - **External priors to verify (Vincent's numbers from his environment — priors, not
    results):** P1 deleted-author answer-prob 0.97 → 0.32 (never-seen level); P2 impact
    on other authors ≤ 0.002; P3 utility Δ ≤ 0.001; P4 no relearn residue (folds into
    H4). Verifying ≈ these values in our OU-track harness is a replication claim;
    missing them is a finding, not a failure of protocol.
- **Setup (frozen before any job):**
  - Architecture: frozen `meta-llama/Llama-3.2-1B-Instruct`; at ALL 16 layers
    `layer.mlp → mlp(x) + bank(x)`; per-author branch
    `down_a(ReLU(W_gate_a x + b_gate_a) * (W_up_a x))`, width 32 (grouped matrices;
    authors architecturally disconnected; `gate_act: relu`, silu retained as variant
    arm). Init: `W_down = 0` (exact no-op at step 0), `W_gate/W_up ~ N(0, 1/√2048)`
    sha-seeded per (layer, tensor), `b_gate = 0`, then detector init: `W_gate` rows
    oriented toward the author's own mean question hidden state (`detector_init:
    questions`, `init_scale 1.0`, cached `detector_init.npz`, deterministic).
  - Loss (spec): `total = L1 + 10·L2 + 50·L3 + 1·L4` — L1 CE routed to the own branch
    only (bitwise detach construction); L2 hinge `relu(pre_act + 2)` on OTHER branches'
    detectors; L3 exact OTHER-branch output-norm (Gram trick, fp32); L4 promotion
    `relu(0.1 − max own detector pre-act)` on own QUESTION tokens. L2/L3/L4 computed
    from the detached layer input (within-layer gradients only — the cross-layer leak
    fix, pinned by CPU gates). Batch schedule: alternating author batch / pure-negative
    batch; negatives = other in-batch authors + Alpaca 2000 (seed 42) + TOFU
    real_authors (100) — **holdout10 excluded from training in any role** (CPU-gated;
    it is the relearn control + MIA nonmember set).
  - Optimizer: AdamW, cosine LR decay, wd 0, per-author gradient clipping at 1.0
    (author slice across all layers = one clipping group), bs 32 × ga 1, 15 epochs,
    bf16 autocast over fp32 masters, seed 42.
  - **Lean pilot (user-decided):** K=20 (authors 0–19), 3 arms = spec recipe × lr
    {3e-4, 1e-3, 3e-3} (`configs/pilot_relu_lr*.json`), array `0-2%1`; each task trains
    then runs `measure_selectivity.py --recall_probe`. The 9-arm SwiGLU λ grid
    (`configs/pilot_0–8.json`) is retired unrun.
  - **G2 (REVISED for the lean pilot):** GO = pick the lr maximizing median on/off
    selectivity subject to median selectivity ≥ 5.0 AND median own-author answer-prob
    (all-active) ≥ 0.80. ADJUDICATE: best arm in [2, 5) with own-prob ≥ 0.80 → one
    bridging config before deciding. NO-GO: all arms < 2.0 ⇒ H1 refuted → refutation
    entry, stop before K=200 spend. (The v1 "≥ 0.90× λ=0 control" clause is void — no
    λ=0 arm exists in the lean pilot.)
  - **G3 (unchanged):** at K=200 — median selectivity ≥ 5.0 and ≥ 0.7× the pilot
    winner; all-active vs own-only own-prob gap ≤ 0.05. Read before any OU eval.
  - Phases/jobs: P1 smoke (full-size K=200 bank, 2 authors, 5 steps, save→reload
    parity, peak-mem print) → P2 pilot `0-2%1` → G2 → P3 K=200 train (lr overwritten
    from the winner) + probe200 → G3 → P4 OU evals (`sepmlp_ft` / `sepmlp_unlearned` /
    `sepmlp_dropall`) + Table-1 composition → P5 relearn battery (24 tasks `%2`) +
    leak-probe arm (`--probe forget_leak`, feeds the router_leak unified table).
    All GPU jobs DEP-chained (`afterany`) behind the ctv/router-sweep lanes so
    worst-case concurrency ≤ 4; one sepmlp array queued at a time.
  - File inventory (sha256 first 16): sepmlp_common 5dd2c4dfc53afd8b ·
    bank_layer 7d2e5cb5f591ec1e · sepmlp_model 961206d42730bc2a ·
    train_sepmlp 06c0a010ba828bc9 · build_droplist 0f154291e28a8476 ·
    measure_selectivity d9edf2bafc7f1c52 · relearn 6977af680a8453c8 ·
    relearn_score 89d074a66aea3a86 · collect_relearn 0ea30c8060db4348 ·
    submit_sepmlp.sh ffc09829044a24ce · slurm_nodes.sh f783dcbc0d321108 ·
    smoke.json 7f94b32e5deececc · pilot_relu_lr1e-3 ade5f4ead3681881 ·
    pilot_relu_lr3e-3 e41f15ce0533fbf2 · pilot_relu_lr3e-4 02bf6ecd71abda96 ·
    sepmlp_1b_k200 c3956998847d7578 · relearn_1b 7774a0ad1b940662.
  - CPU gates: **69 passed, 1 skipped** (the skip = the GPU smoke test, gated behind
    `SLURM_JOB_ID`/`SEPMLP_GPU_TESTS=1`) — full suite green before this entry.
    SLURM job ids: TBD at submit (this entry pre-registers BEFORE any job).
- **Results:** pending (pre-registration).
- **What worked / hypothesis verdict:** pending.
- **Observations:** Three implementation facts worth the record: (1) the naive detach
  form `out_real.detach() + out_grad − out_grad.detach()` without inner parentheses is
  NOT bitwise-neutral (left-to-right float addition) — the parenthesized form is pinned
  by a CPU gate; (2) the suppression losses originally backpropagated through the
  residual stream into LOWER layers' own-author slices (probe: own-slot grad mass
  1.61/0.47/0.48/0.0 at layers 0–3) — fixed by recomputing loss activations from the
  detached layer input; (3) physical slice removal is bitwise vs mask at the bank level
  but can differ by ~1 ulp at composed-model logits (BLAS reduction order at shrunk
  shapes) — deletion identities are pinned as bank-bitwise + model-level atol 1e-6.
- **New questions / new hypotheses:** does detector init alone (before any training)
  already produce selectivity > 1? (The pilot telemetry's epoch-0 row will show it.)
  Does the promotion term matter at all at K=20, or only at K=200 sparsity? (Wave-2
  ablation candidate: w4=0.)
- **Next Steps:** P1 smoke behind the current queue tails → P2 pilot → G2 read →
  results entry.
