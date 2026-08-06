# composable_tv — task vectors trained to sum-compose with exact subtraction

**Status:** active · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 4 (2026-07-16 → 2026-07-20)

Can we occupy PATHS_FORWARD §5's "single merged model, no serve-time selection" cell by changing
TRAINING so that per-author task vectors compose by weight-1.0 summation and delete by exact
subtraction? Four arms, tagged in entry titles: **[lin]** linearized/tangent-space LoRA (the one
mechanism the ledger has never tested — superposition replaces the multiplicative attenuation of
the H8 knee), **[wd]** write-disjoint col(B) subspaces (hard structural version of the collision
diagnosis; adjudicates the H3 overlap-as-protection sign-flip), **[ds]** disjoint-support full-FT
task vectors served merge-only (answers SIFT-Masks' open question: does support disjointness
remove the need for their inference-time mask?), **[w5]** post-hoc deterministic sparsification
grid (zero-training calibration on the 7B e5 pool). Sits beside centered merging (Exp-6,
merge_mechanism, in flight — post-hoc combination fix on unchanged adapters) and Path C MEMIT
(different substrate, deferred to its own thread). Shared protocol: H8-mirrored N-ladder,
exactness classes (bitwise/algebraic/first-order/approximate), MIA + diff-attack audit, locality
verification with placebo controls, per-author distributions (never means alone).

Reference documents (user-provided, saved verbatim 2026-07-18, provenance headers correct
stale anchors): [`TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md`](../../tofu_sisa_lora/reports/TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md)
(14 merge-method specs + common harness §15 + exactness unit test §15.5) and
[`DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md`](../../tofu_sisa_lora/reports/DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md)
(conditional-GO assessment whose [ds] arm, cross-talk probe, and kill bars this thread's
pre-registration adopted). Path C / MEMIT (method #14 there): **user-declined 2026-07-18** —
stays a cited-not-run row; GPU budget goes to the built arms + Phase-2 gap-fill merges
(`log/merge_mechanism/`) instead.

Killed at design time (rationale in the pre-registration entry): W2 rank-slicing (function-space
identical to the additive sum), W3 sign-fixed LoRA (sign constraints not closed under low-rank
factors; replaced by the free DX1 cancellation diagnostic), W4 soft write-collision penalty
(dominated by [wd]'s hard projection; pre-registered fallback only if H-wd-1 refutes).

## Hypotheses — open / resolved
- **[open]** H-lin-1 tangent-space training memorizes a TOFU author (solo ans-prob ≥0.9× matched standard control; refute <0.5× after one pre-registered retry) — pending G1 solo evals.
- **[open]** H-lin-2 linearized-serve composition survives N (signal-remaining ≥80%@N=3, ≥60%@N=8 vs control 51%/9%) — pending G2 ladder.
- **[open]** H-lin-2b tangent-trained vectors under nonlinear serve move the 50%-knee from N≈3 to ≥6 — pending G2.
- **[open]** H-lin-3 subtraction: subtract≡recompose ≤1e-4; post-subtract forget metrics in the exact band; MIA ≤0.38 — pending G3.
- **[open]** H-lin-4 function-space disentanglement ξ(i,j) ≥2× smaller for tangent-trained pairs — pending G2 probe.
- **[open]** H-lin-5 linear crosstalk (lazy keys): mu@N=20 within −0.02 of N=1 under linearized serve; monotone decay = the "second wall" datum — pending G2.
- **[open]** H-wd-1 learnability under the col(B)⊆span(Q_a) constraint: solo ≥0.95× control (kill <0.8×; only then W4 fallback) — pending G1.
- **[open]** H-wd-2 (H3 adjudicator) orthblock-sum at N=8 retains ≥2× control-mean extractable fraction (abs ≥0.56 vs ≈0.28). CONFIRM ⇒ col(B) collision causal; REFUTE with H-wd-1 ✓ ⇒ overlap-as-protection causal — pending G2.
- **[open]** H-wd-3 no collateral: mu ≥ control mean-merge −0.02, no retain_ppl explosion — pending G2.
- **[open]** H-wd-4 exactness certificate: factor drop byte-exact for survivors ≡ Q_a-zeroing ≤1e-6 — pending G3.
- **[open]** H-wd-5 rowslice within 0.05 of orthblock at N=8 (coordinate basis not privileged) — pending G2.
- **[open]** H-ds-1 support-constrained full-FT memorizes (solo ≥0.95× sift-style unconstrained solo) — pending G1.
- **[open]** H-ds-2 merge-only serve beats the sift maskless floor (mu 0.407 ≈ base) and retains ≥2× control extractable fraction at N=8 — pending G2.
- **[open]** H-ds-3 cross-talk gap (solo vs all-N own recall) ≤0.15 with locality verified — the report's kill signal — pending G2.
- **[open]** H-ds-4 subtraction bitwise-exact (sift `merge_sub_` precedent) — pending G3.
- **[open]** H-w5-1 calibration null: no post-hoc setting reduces the N=8 drop by >⅓ (0.073→<0.049); anything that does jumps the queue — pending [w5] wave.
- **[open]** H-w5-2 (=DX2) post-hoc hash-truncation destroys recall ∝ energy removed (≈1/N) ⇒ train-time constraint necessary — pending [w5] wave.
- **[open]** DX1 cancellation diagnostic: per-coordinate |Σδ|/Σ|δ| vs sign-shuffled null on the 7B pool — if observed ≤ null, the sign-fixing idea-space is closed with data — pending CPU job.
- **[open]** H-CT-mia post-deletion served composition loss-AUC ≤0.45 (floor 0.379, live control ≥0.59) — pending G3.
- **[open]** H-CT-diff (severity only, no safety claim) pre−post ≡ τ_u exactly; extraction fidelity predicted ≈ solo recall ≈1.0 — reported at G3.

## What worked
- (none yet)

## What didn't / open problems
- (none yet)

## Open ideas / next steps
- **CAT Merging (arXiv:2505.06977, user-flagged 2026-07-16)** — Conflict-Aware Task merging:
  training-free per-layer trimming of conflict-prone task-vector components (projection on
  linear weights, masking on norm params). NOT our PEFT `cat` (concatenation — run since 06-04);
  distinct from TIES/DARE (run extensively; dare_ties = the 0.48–0.59 band). Candidate [w5]
  escalation: the strongest published falsifier for H-w5-1's post-hoc null, and it targets the
  diagnosed col(B) write-conflict directly. Exactness class: recompute-over-survivors (like
  Exp-6 centered), not subtract-exact (trims are cross-task-dependent, TIES-style). Mechanism
  prediction is contested by the H3 sign-flip (overlap = protection, ρ=−0.675) — informative
  either way. Needs per-task exemplar forwards + a projection implementation (moderate build);
  paper only tested ViT-scale vision/VL. Trigger: run if H-w5-1 confirms the null, or as the
  conflict-aware point on the post-hoc curve after the pre-registered grid lands.
- Fold in Exp-6 centered results (jobs 443445+) when they land — if H-cent-1 confirms first, reframe headline to "rescue with subtract-only byte-exact deletion" (centered deletion re-estimates the mean; ours is a factor drop).
- If all arms die at G2: negative boundary entry — "train-time structure cannot rescue ungated summation," the complement of the 07-16 self-gating verdict — and merge into the mechanism paper.
- Optional winner rungs: relearn-speed probe (H-CT8), seeds 43/44 replicate, extended caps.

## Entries (chronological)
- [2026-07-16 — thread pre-registration](2026-07-16_thread-preregistration.md) — protocol, gates, kill bars, and the W2/W3/W4 kill rationales.
- [2026-07-16 — Wave-0 build](2026-07-16_wave0-build.md) — all four arms coded; full CPU gate suite + STUB previews green; [ds] serves in-place via new `--ds_config` bypass (storage decision); integration drift caught pre-submission; GPU-phase risks recorded.
- [2026-07-18 — Wave-1 launch prep](2026-07-18_wave1-launch-prep.md) — reference docs saved + indexed; user decisions recorded (MEMIT skipped; full-FT bake-off = 20-author pilot; ladder stays {32,64,128,200}); gates re-stamped + prep re-run (sparsify_7b.json missing-`arm` fix); **[w5] job 445329 submitted** (CPU); GPU wave staged but cap-blocked behind another session's 4 pending memadapt jobs; irpctrl twin confirmed as the H-lin-1 G1 denominator.
- [2026-07-20 — Wave-1 trains landed · G1 launched](2026-07-20_wave1-trains-g1-launch.md) — all 4 smokes + full arrays GREEN (93 adapters/τ on disk; ds smoke loss 0.377 @d=0.005); **irpctrl twin 20/20 failed on the latent IRP CUDA bug** (memsinks' 07-15 warning cashed) → bit-equal CPU-draw fix in `train_lora_shard.py`, twin resubmitted (446357); [w5] recovered via 24 h grid + 36 h DX split (445693/445694, both complete); **DX1: sign-fixing headroom marginal** (obs 0.0759 vs null 0.0712 @N=200 — W3 stays closed in practice); DX2 = exact 1/N energy; G1 eval arrays 446365–446369 queued (lin vs twin serialized).
