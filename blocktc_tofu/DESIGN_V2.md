# blocktc_tofu — DESIGN ADDENDUM v2 (2026-07-22): promotion recipe

Binding addendum to `DESIGN.md`. **Everything in DESIGN.md still holds** (architecture,
exactness masking, phases, deletion, eval); v2 changes ONLY the phase-1 loss on author
batches, to fix the H1 refutation from the P2 pilot
([log](../log/blocktc/2026-07-22_p2-pilot-lazy-refuted.md)). Where this file is silent,
DESIGN.md governs; where they conflict on the loss, this file governs.

## 0. Why (the v1 failure this fixes)
v1 (plain LM + generic-only L1 suppression) trained blocks that fire on teacher-forced
**answer** tokens but not on **question** tokens → not question-keyed → at inference the
block does not fire on the author's question → recall ≈ base, all 6 lr×λ arms LAZY
(median on/off 0.711→0.000). The fix is sepmlp's **promotion** term: actively force ≥1 of
an author's own detectors to fire on that author's own **question** tokens. This is the
exact mechanism that gave sepmlp selectivity 4.38–1909.7 (vs LoRA anchor 1.11).

## 1. THE EXACTNESS RULE v2 MUST NOT BREAK (read twice)
DESIGN.md §3 invariant: **no parameter ever receives gradient from more than one deletable
author.** sepmlp applies hinge (L2) + Gram (L3) suppression on EVERY batch — on an
author-k batch it suppresses the OTHER authors' blocks. **blocktc must NOT do that** — that
is cross-author gradient (author-k data → author-j params) and destroys exactness. So the
term placement differs from sepmlp:

| term | sepmlp placement | **blocktc v2 placement** | exactness rationale |
|---|---|---|---|
| L1 LM (routed) | own block, author batch | own block, author batch (unchanged) | own data → own block ✓ |
| **L4 promotion (NEW)** | own block, own tokens, author batch | **own block, own QUESTION tokens, author batch ONLY** | own data → own block ✓ |
| L2 hinge suppression | every batch | **generic (NO_AUTHOR) batches ONLY** | generic data → any block ✓; author batch → NO suppression |
| L3 Gram suppression | every batch | **generic (NO_AUTHOR) batches ONLY** | same |

The one genuinely new gradient path is **L4 promotion on author-k batches** — and it is
exactness-safe because it flows ONLY into block k (the sequence author's block) from author
k's own question tokens. It goes through the SAME own-mask + detach-trick routing as L1.
On generic batches, promotion is NOT computed (no author owns generic data).

**Author batches in v2 therefore carry gradient into block k ONLY** (via L1 + L4), exactly
as v1's author batches carried L1 only. Generic batches carry suppression into all author
`W_enc`/`b_enc` rows only (never W_dec, never shared) — unchanged from v1. The
`debug_grad_check` per-(phase×batch-type) exact-zero asserts must be extended to include L4
and must still pass: author-k batch → grad zero outside block k (now covering the L4 path
too); generic batch → unchanged.

## 2. Loss definitions (port from sepmlp `bank_layer.py:300-313`, read it first)
All computed in the fp32 island (autocast disabled), from the DETACHED normed layer-9 input
`xn.detach()` where a term must not backprop through the residual into lower layers (sepmlp's
cross-layer-leak fix — still required; the encoder reads once at layer 9 so the leak surface
is smaller, but keep the detach for the suppression terms). Per-token masks come from TcState
(`question_mask` = prompt tokens = labels IGNORE & attended; sepmlp's `own_token_mask`).

- **L4 promotion** (author-k batch, own block k rows only): for each sequence, encourage
  `max over block-k detectors, over the author's QUESTION tokens, of the ReLU pre-activation`
  to exceed a margin `promo_delta` (start 0.1, sepmlp value). sepmlp form: hinge that ≥1 own
  detector fires ≥ delta on own question tokens (dead-ReLU rescue). Weight `w_promo` (start 1).
  NOTE blocktc-specific: sepmlp recently moved promotion to ALL own tokens (paper §3.2,
  D2a) — but blocktc's failure is specifically no QUESTION-token firing, so v2 uses
  **question tokens** (that is the retrieval fix). Make the token set a config flag
  `promo_tokens: "question"|"own"` defaulting to "question" so the pilot can A/B it.
- **L2 hinge** (generic batch, all author blocks): off-detectors pushed ≥ `hinge_margin`
  (2.0) below the ReLU threshold — the exact-0 off-state term. Weight `w_hinge` (10).
- **L3 Gram output-norm** (generic batch, all author blocks): per-block squared output norm
  via the Gram trick (sepmlp `_per_author_sq_norms`), driven toward 0. Weight `w_gram` (50).
  Replaces/augments v1's plain L1 on generic batches — keep L1 available via config too.

Total: author batch = `L1 + w_promo·L4`; generic batch = `w_hinge·L2 + w_gram·L3`
(+ optional `lambda·L1_suppression` from v1). All non-LM terms divided by
`gradient_accumulation_steps` (ga-invariance, transformers 4.48). λ/weight warmup 0.15 as v1
(warm promotion in too, so blocks aren't punished/forced before they mean anything).

## 3. Config schema additions (configs/, all hyperparams here — no ad-hoc CLI)
Add to the v1 schema: `recipe: "v1"|"v2"` (default "v2" for new configs), `w_promo` (1.0),
`promo_delta` (0.1), `promo_tokens` ("question"), `w_hinge` (10.0), `hinge_margin` (2.0),
`w_gram` (50.0), and keep `lambda_max` (v1 L1 suppression; may set 0 when hinge+Gram carry
suppression). New pilot configs: `pilot_v2_lr{3e-4,1e-3}_s42.json` (start with the 2 lrs
nearest v1's least-bad arm; v1 showed high lr diverges, so DROP 3e-3) at K=20, plus one
**capacity arm** `pilot_v2_m64_lr3e-4.json` (m_author 64 — tests whether low recall is
objective- or capacity-bound; F=200·64+128=12928, ~53M→~106M params, still fits the A40).
`blocktc_1b_k200.json` gains the v2 fields (weights = pilot winner).

## 4. Telemetry / probe additions
Per-epoch telemetry: add per-block **question-token** own firing (promotion target) alongside
the existing own/off/OOD mass, so question-keying is visible during training (the v1 blind
spot — v1 telemetry showed answer-token own-mass 10:1 while question-token selectivity was
0.711). `measure_selectivity.py` already probes question tokens (unchanged); the gate/
recall/leakage contract is unchanged so v2 numbers land next to v1's and sepmlp's anchors.

## 5. Tests to ADD (beyond DESIGN.md §9's 14 gates — all still required)
- **Exactness under L4** (critical): author-k batch with the v2 loss → grad EXACTLY zero
  outside block k across `W_enc/b_enc/W_dec` (the promotion path must not leak). This is the
  gate that proves v2 didn't break exactness.
- **Suppression still generic-only:** author-k batch computes L1/L2/L3 = 0 (no suppression on
  author data); generic batch computes promotion = 0 (no promotion on generic data).
- **Promotion targets question tokens:** with `promo_tokens="question"`, L4 gradient is zero
  when the question_mask is all-False (answer-only), nonzero when question tokens present.
- **ga-invariance of L2/L3/L4** (each divided by grad_accum).
- **recipe="v1" still reproduces the v1 loss exactly** (back-compat; the v1 pilot stays a
  valid baseline).

## 6. What does NOT change
Architecture (encoder@9, decoders@9/10/11, zero-init), the detach-trick routing, phase-0
author-free shared-block training + freeze, deletion (drop the block), OU eval integration,
relearn, SLURM driver, the 4-GPU cap, holdout10 sacred, provenance. v2 is a **loss-only**
change inside `train_tc.py` (+ small `tc_layer.py` telemetry/promotion-term helpers +
configs + tests). If v2's exactness gates do not pass, v2 is wrong — do not weaken them.

## 7. Reference
sepmlp loss: `bank_layer.py` (hinge/Gram/promotion recomputed from `xb.detach()`, lines
~300-313; `_per_author_sq_norms` Gram trick; `own_token_mask`), `train_sepmlp.py`
(`DEFAULT_LOSS` weights L1 + 10·L2 + 50·L3 + 1·L4, ga-division). blocktc v1 baseline:
[P2 pilot entry](../log/blocktc/2026-07-22_p2-pilot-lazy-refuted.md). sepmlp prior: the
promotion recipe reaches selectivity but hit a K=200 recall ceiling ≈0.80 — expect the same
K-scaling caution for blocktc v2.
