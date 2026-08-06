# blocktc — single-bottleneck block transcoder; one encoder read, per-feature residual writes at 3 layers, drop-a-block deletion

**Status:** **P2 pilot done — H1 REFUTED for the v1 (plain LM + L1) recipe; ladder halted for human decision.** All 6 lr×λ arms LAZY (median on/off 0.711→0.000, best arm 90% of authors <2; own-prob ≪0.80), selectivity *anti*-correlates with lr. Diagnosis: blocks fire on teacher-forced **answer** tokens (train loss drops) but are **not question-keyed** (probe on question tokens = 0.711), and a trained author's own block alone drives recall to ~0 while all-active gives only 0.12–0.33 → knowledge is collective/near-empty = **memsinks failure #2 reproduced** + the #1 pre-registered retrieval-failure mode; span-3 didn't rescue it at K=20. Operationally clean throughout (build 91 CPU gates + 3-lens review; P1 smoke 447416 PASS; phase-0 447419 GREEN; P2 447430 bs32 fits @ 37.04 GiB, no NaN/OOM, shared bitwise-frozen). **Recommended next: implement v2 = sepmlp's promotion+hinge+Gram recipe (DESIGN §15 pre-registered fallback) + m=64 width ablation — awaiting user go-ahead (no GPU committed).** · **Project:** [`blocktc_tofu/`](../../blocktc_tofu/) · **Entries:** 3 (2026-07-21 → 2026-07-22)

Successor of [sepmlp](../sepmlp/README.md) on TOFU / frozen Llama-3.2-1B-Instruct: instead
of per-author × per-layer branch banks (0.63B added at K=200), ONE wide block-transcoder
bottleneck — a single encoder read at layer 9's post-attention-norm MLP input
(`a = ReLU(W_enc·xn + b_enc)`, F = 200×32 author features + 128 shared = 6528), with
zero-init per-feature decoders adding `a @ W_dec[j].T` to the residual stream at layers
9/10/11. **53,483,904 params = 4.33% of the base, 11.8× smaller than sepmlp.** Training:
hard detach-trick gradient masking (serving values bitwise identical; gradient flows only
through the own-block mask, decoder path included) so no parameter ever receives gradient
from more than one deletable author; the 128 shared features train in phase 0 on an
author-free pool (Alpaca + real_authors — never TOFU author rows) and are then frozen;
suppression is an L1 on author-feature activations collected ONLY on generic (NO_AUTHOR)
batches. Serving = all features live, no router; deletion = index-select out the author's
32 feature rows/cols, O(1).

Relationship to sepmlp (spec-v2 pre-registered; its P2 pilot decision is pending with the
user): blocktc inherits sepmlp's three untested bets — (a) full architectural
disconnection of authors, (b) a nonlinear (ReLU) gate that can shut off on off-author
inputs, (c) suppression-trained self-selection with no serving-time router — and its open
make-or-break H1 (localization; LoRA anchor 1.11, 100% LAZY, from
[merge_mechanism 2026-07-15](../merge_mechanism/2026-07-15_key-firing-results.md)) and H2
(all-active serving vs the memsinks interference collapse, mu 0.4373 vs 0.6438). What it
changes: one bottleneck instead of 16 layers × 200 branches, and bet (c) is restricted to
author-free generic batches — which upgrades the deletion claim from sepmlp's declaredly
NOT-exact (negatives leak) to exactness-by-construction (H6). OU chat-template track
throughout; anchors MemAdapt Agg 0.869 / Retrained Agg 0.874.

> **From-scratch explainer + rebuild spec:** [../SELF_ROUTING_ARCHITECTURES_EXPLAINED_2026-07-25.md](../SELF_ROUTING_ARCHITECTURES_EXPLAINED_2026-07-25.md)
> — this thread and [sepmlp](../sepmlp/README.md) as one lineage: the shared machinery (detach
> identity, Gram trick, promotion), the cross-layer handoff, why v2 must *re-place* sepmlp's
> suppression rather than copy it, and a build-order + CPU-gate rebuild recipe.
> Context for the v2 decision (§8.3): sepmlp's K=200 recall gap turned out to be an **epoch
> budget**, not a K ceiling — blocktc v1 ran the same 15 epochs.

## Hypotheses — open / resolved
- **[resolved ✗ REFUTED for v1]** **H1 localization (make-or-break):** median on/off ≥5
  with own-prob ≥0.80. **REFUTED** by the P2 pilot (447430, 6 arms lr {3e-4,1e-3,3e-3} × λ
  {0.01,0.1}, K=20): all arms LAZY (median 0.711→0.000, best arm 90% of authors <2; own-prob
  ≪0.80), selectivity anti-correlates with lr. Root cause: plain LM loss keys blocks to
  teacher-forced **answer** tokens, not questions; own block alone → recall ~0 (near-empty
  slices, memsinks failure #2). Scoped to the **v1 plain-LM+L1 recipe** — the DESIGN §15
  fallback (promotion+hinge+Gram) is untested and reopens H1 as v2. See
  [2026-07-22 P2 pilot](2026-07-22_p2-pilot-lazy-refuted.md).
- **[open]** **H1-v2 localization under the promotion recipe:** does sepmlp's L2 hinge /
  L3 Gram / L4 promotion-on-own-question-tokens reach median on/off ≥5 ∧ own-prob ≥0.80 at
  K=20 (the mechanism that gave sepmlp 4.38–1909.7)? REFUTE: still <2. — pending v2 build + pilot.
- **[open]** **H2 all-active serving retains utility (anti-memsinks):** all-active vs
  own-only recall gap ≤0.05; OU Util.R ≥0.95, Util.G ≥0.95. REFUTE: ≥0.15 all-active
  drop. — pending P3 probe200 (gate G3) + P4 OU evals.
- **[open]** **H3 deletion clean:** drop forget10 ⇒ Mem ∈ [0.55,0.70], |ΔUtil.R| ≤0.03,
  Agg ≥0.80 (strong-confirm 0.84–0.90 vs MemAdapt 0.869 / Retrained 0.874); dropall ≡
  calib_base. — pending P4 OU evals.
- **[open]** **H4 relearn parity:** median relearn target/control ratio ∈ [0.8,1.25] at
  steps [0,5,10,25,50]. REFUTE: target ≥2× faster. — pending P4 relearn battery.
- **[open]** **H5 MIA (measurement + attribution):** 4 raw MIA AUCs vs the oracle floor
  0.379 (MemAdapt Priv anchor 0.917); direction attributed. — pending P4.
- **[open]** **H6 exactness-by-construction:** surviving params never received ANY
  gradient from a deleted author's data, including as negatives (stronger than sepmlp's
  claim) — detach-trick masking + author-free phase-0 pool + generic-only suppression;
  verified via the G0 grad-isolation gates plus H4+H5 behaviorally. — pending P4.
- **[open]** **H7 span:** span-3 beats span-1 by >0.02 OU model_utility at the matched
  53M budget; else adopt span-1. — pending P5 ablations.

## What worked
- **Build & infra:** 91-gate CPU suite green + 3-lens adversarial review (fixed OOD-probe/train
  overlap, symmetric decode grad-guard, epoch-gated coverage guard). P1 smoke PASS. Exactness
  machinery holds end-to-end: shared block bitwise-frozen through phase 1, save/reload parity,
  no NaN. bs32 fits an A40 (37.04 GiB, ~8 GiB headroom).
- **Phase-0 shared block:** trains cleanly on the author-free pool (loss 0.45→0.029), author
  blocks provably untouched.

## What didn't / open problems
- **v1 plain-LM+L1 does NOT localize (H1 refuted).** All 6 pilot arms LAZY; selectivity
  anti-correlates with lr (higher lr → decoder writes diverge, loss 1.37→3.8, recall→0).
- **Blocks are not question-keyed:** they fire on teacher-forced answer tokens (train telemetry
  own 10:1) but not on question tokens (probe 0.711) — the inputs the model has at inference.
  The "stored but question-keyed" #1 pre-registered failure mode; span-3 didn't fix it at K=20.
- **Knowledge is collective, not per-block:** a trained author's own block *alone* → recall ~0
  (worse than base); only all-active gives 0.12–0.33. Near-empty slices = memsinks failure #2
  reproduced — the exact prior blocktc set out to beat. Deletion would not be clean here.
- **Low collective recall (0.12–0.33)** even all-active — possible 32-feature-at-one-site
  capacity floor (vs sepmlp's 32×16-layer banks → own-prob 0.977).

## Open ideas / next steps (awaiting user go-ahead — no GPU committed)
- **v2 = promotion recipe (recommended):** port sepmlp's L2 hinge (w10) / L3 Gram (w50) / L4
  promotion-on-own-**question**-tokens (w1) into `train_tc.py`; keep the transcoder architecture
  + exactness masking unchanged. Re-run the K=20 pilot as a clean A/B vs this v1 baseline. This
  is the DESIGN §15 pre-registered fallback and the mechanism that gave sepmlp selectivity.
- **Capacity ablation (parallel):** m=64 width arm to test whether low recall is objective- or
  capacity-bound.
- Also consider: lr <3e-4, lower insertion layer, or accepting that gated branches may need
  >1 layer (sepmlp used all 16). sepmlp prior: promotion recipe reaches selectivity but hit a
  K=200 recall ceiling ≈0.80 — a caution for blocktc v2's K-scaling.

## Entries (chronological)
- [2026-07-21 — Pre-registration + build](2026-07-21_preregistration-and-build.md) —
  H1–H7 pre-registered with CONFIRM/REFUTE bars before any GPU spend; four corrections vs
  the source design doc declared (OU-track anchors, relearn steps [0,5,10,25,50],
  author-free phase-0 pool, generic-only suppression); gate ladder P0→P5 frozen; param
  count 53,483,904 verified.
- [2026-07-22 — P1 smoke green](2026-07-22_p1-smoke-green.md) — build gated (91 CPU tests +
  3-lens review, 3 fixes); P1 smoke 447416 PASS (phase 0→1 in one job, shared bitwise-frozen,
  save/reload parity, peak 13.43 GiB @bs8); 447413 first caught a too-strict coverage guard.
- [2026-07-22 — P2 pilot: H1 refuted for v1](2026-07-22_p2-pilot-lazy-refuted.md) — phase-0
  447419 green (35.06 GiB @bs32); P2 447430 (6 arms) all LAZY (median 0.711→0.000), blocks
  answer-token-keyed not question-keyed, own-block-alone recall ~0 (memsinks failure #2); bs32
  fits @37.04 GiB. Ladder halted; v2 promotion recipe recommended, no GPU committed.
