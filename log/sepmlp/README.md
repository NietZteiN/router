# sepmlp — per-author x per-layer bottleneck MLPs; suppression-trained, all-active router-free serving, drop-to-delete

**Status:** **PAPER-CONFORMANCE COMPLETE; K=200 hyperparameter reverse-engineering pending Jack.** Audit found architecture + objective (Eq 1-5) faithful; the two real deviations are fixed & gated (D2a promotion on own tokens, D2b zero-W_down deletion; CPU suite 77 passed). New `measure_recall.py` gives the paper's ROUGE-L recall (validated: held-out 0.31 ~ paper 0.341). Code matches the paper at **K=20 (0.957 ~ 0.96)**; K=200 best **0.740** vs paper 0.966 (tail 197/200 vs 31) - a **K-scaling** gap (NOT structural; ceiling retracted). Promotion fix REFUTED as the lever (clean A/B 0.738 ~ A2 0.740). Remaining = a hyperparameter search (lambda_out / epochs / lr); leading hypothesis = forward residual accumulation of 199 foreign adapters. **Entries:** 11 (2026-07-20 -> 2026-07-22)


Implementation of Vincent Hanke's unlearning method on TOFU / Llama-3.2-1B-Instruct: a
frozen base model plus a separate ReLU-gated bottleneck MLP branch (width 32, per-unit
gate bias) **per author per layer** (all 16 layers), stored as grouped block matrices so
authors are architecturally disconnected —
`branch_a(x) = down_a(ReLU(W_gate_a x + b_gate_a) * (W_up_a x))`. Training (the
user-supplied authoritative recipe, spec v2): LM loss routed to the sequence-author's
branch only (bitwise detach trick), plus `10·hinge` (off detectors ≥2 below the ReLU
threshold — exact-0 off-state reachable), `50·Gram output-norm` (closes the
barely-below-threshold + huge-down loophole), `1·promotion` (≥1 own detector fires on own
question tokens — dead-ReLU rescue); alternating author/pure-negative batches (negatives =
other authors + Alpaca + TOFU real_authors; **never holdout10**); per-author gradient
clipping; cosine LR; detector init toward the author's own question representations.
Serving runs with **all branches active and no router** (the model must self-route);
deletion = physically remove the author's slices, O(1). This is the direct attempt at the
one empty cell of the [PATHS_FORWARD §5](../PATHS_FORWARD_2026-07-13.md) selection table —
*selection inside the weights: single served model, no serving-time router, no task-ID
lookup, cheap deletion*. External priors from Vincent's environment (to verify here, not
results): deleted 0.97→0.32, other authors ≤0.002, utility Δ0.001, no relearn residue.

> **From-scratch explainer + rebuild spec:** [../SELF_ROUTING_ARCHITECTURES_EXPLAINED_2026-07-25.md](../SELF_ROUTING_ARCHITECTURES_EXPLAINED_2026-07-25.md)
> — this thread and [blocktc](../blocktc/README.md) as one lineage.
> ⚠️ Its §3.11 reports the **reverse-engineering wave-1 arms that landed 2026-07-23 and have
> no dated entry yet**: H-epoch (15→30 epochs) reached per-source ROUGE-L **0.9842 / tail
> 17 of 200**, above the paper's 0.966/31 — superseding the "K=200 best 0.740" in the Status line
> above — while both `w_out` arms were flat-or-worse (0.7412 / 0.7053). The results entry and
> this README's ledger update are outstanding.

It is also a deliberate rematch of two failures already logged in this repo. (1) The
LoRA negative-anchor pilot
([merge_mechanism 2026-07-16](../merge_mechanism/2026-07-16_negative-anchor-pilot-results.md))
refuted penalty-trained self-gating on LoRA: on/off selectivity stuck at 1.110–1.150 (100%
LAZY, per-adapter ratios never leaving [1.08, 1.17]) at every λ ∈ {1, 10, 100}, while recall
collapsed to 0.525 at λ=100 — "self-gating cannot be trained into a LoRA". (2) memsinks
([README](../memsinks/README.md)) showed 200 always-on per-author deltas self-interfere:
all-slices-on mu 0.4373 vs control 0.6438, even though training-condition memorization was
fine (own-mask prob 0.87–0.96). sepmlp bets that three design deltas fix both: **(a) full
architectural disconnection** (block-diagonal grouped SwiGLU — author a's contribution is a
function of only author a's three slices, vs LoRA's shared low-rank subspace), **(b) a
nonlinear gate** (silu·up product can learn to shut off on off-author inputs, vs linear
adapters that cannot), and **(c) in-domain negatives** (the other TOFU authors in the batch,
vs generic anchor text).

## Hypotheses — open / resolved
- **[resolved ✓ SUPPORTED (2026-07-21)]** **H1 localization (make-or-break):**
  disconnected ReLU-gated branches + the 4-term recipe reach SELECTIVE with own-prob
  ≥0.80. **Confirmed by the pre-registered ADJUDICATE bridging arm lr 5e-4 (job 446732):
  median on/off selectivity 7.171 (≥5) with median own-prob (all-active) 0.9765 (≥0.80;
  min author 0.936, none <0.8).** The pilot lr ladder 4.38 → 7.17 → 38.61 → 1909.7
  (3e-4 → 5e-4 → 1e-3 → 3e-3) vs the LoRA anchor 1.11 shows suppression is lr-dialable
  over ~3 orders of magnitude; recall trades off monotonically (0.981/0.977/0.778/0.695).
  G2 winner = lr 5e-4 (unique arm passing both bars).
- **[resolved ± ANSWERED (2026-07-22)]** **H-gap:** the all-active−own-only gap (collective
  recall) causes real retain-author deletion collateral — dropping forget10 costs surviving
  authors retain_Q_A_Prob −0.113 / retain_ROUGE −0.122 (C mechanics on the lr2e-4 ckpt) —
  BUT it is masked by the aggregate |ΔUtil.R| = +0.003 (passes ≤0.03). Both stand together;
  the gap is not benign at the fine-grained level. This is the memsinks all-on interference
  reappearing at K=200 (the failure architectural disconnection was meant to beat —
  disconnection fixed selectivity, not all-active recall at scale).
- **[open]** **H2 all-active serving retains utility (anti-memsinks):** all-active ≈ own-only
  (gap ≤0.05); OU Util.R ≥0.95, Util.G ≥0.95 (MemAdapt FT row 1.075/1.024). REFUTE:
  ≥0.15 all-active drop. — **07-21 partial:** the REFUTE direction (all-active drop) never
  observed at any lr or K; the gap clause fails at K=200 in the *benign* direction
  (+0.130, all-active better). OU utility numbers pending P4.
- **[resolved ✗ REFUTED (2026-07-22)]** **H-k200-lr:** at K=200, lr 2e-4 restores median
  recall ≥0.80 with sel ≥5. **Refuted:** recall 0.7468 < 0.75 (missed by 0.003); the sel
  half was exact (36.0 vs predicted ≈30). The tradeoff-curve model strengthened: K=200
  curve ≈ K=20 curve − 0.03 recall at matched sel.
- **[resolved ✗ REFUTED (2026-07-22)]** **H-wscale:** w2/w3 → 1/5 at lr 5e-4 restores
  pilot behavior. **Refuted:** sel 24.59, recall 0.6956 (predicted sel 5–15 / recall ≥0.90
  — missed both). The weight rescale moves along the SAME K=200 tradeoff curve (B sits ON
  it, slightly below); the recall ceiling is not a suppression-weight artifact.
- **[resolved ~ GRAY (2026-07-22)]** **H-k200-lr2:** lr 1.5e-4 → sel 12–25, recall 0.80–0.87.
  **Gray:** sel 16.33 ✓ but recall 0.7947 ∈ [0.75, 0.80) — best K=200 point, below the pass
  bar. Per pre-registration: report, no P4 (no autonomous escalation).
- **[open]** **H-K-sweep / H-width (raised 2026-07-22, unregistered):** is the ≈0.80 K=200
  recall ceiling a smooth function of K (locate where 0.90 becomes unreachable ⇒ "what K did
  Vincent run") and/or a per-author capacity limit (width 32 × 16 layers too little at K=200
  all-active)? Candidates for a K∈{20,50,100,200} ladder and width/layer ablations.
- **[resolved ✓ SUPPORTED (2026-07-21)]** **H-scale (selectivity transfer):** K=200 median
  selectivity 507.5 ≥ 5.0 and ≥0.7× pilot winner — amplifies ~70× rather than merely
  transferring.
- **[partial ± (2026-07-22, C mechanics on lr2e-4 ckpt — NOT the replication row)]**
  **H3 deletion clean:** forget-side deletion clean (forget_Q_A_Prob 0.767→0.054,
  extraction 0.469→0.047, MIA→floor) and cheap (physical slice removal 1.07 s; dropall
  symmetric floor ≈ base). |ΔUtil.R| aggregate +0.003 ✓ BUT fine-grained retain recall
  −0.113 (see H-gap). Mem/Agg replication bars NOT evaluated (0.747-recall ckpt is below
  the utility regime; a true replication row awaits a passing arm). C is mechanics-only.
- **[open]** **H4 relearn parity (Vincent's observation):** median steps-to-prob-0.8 ratio
  target/control ∈[0.8,1.25] and fixed-budget |Δprob| ≤0.10. REFUTE: target ≥2× faster.
  — pending P5 relearn battery.
- **[open]** **H5 negative-example leak (measurement, no bar):** Priv and 4 raw MIA AUCs on
  sepmlp_unlearned; direction attributed (residual memorization vs over-suppression).
  Anchors: MemAdapt Priv 0.917; exact-drop threads ≤ oracle floor AUC 0.379. — pending P4
  (tofu_grimes MIA block comes free with the OU eval).

## What worked
- **Trained self-gating is real in this architecture** (2026-07-21 pilot): median on/off
  selectivity 38.61 (lr1e-3) and 1909.7 (lr3e-3) vs the LoRA negative-anchor ceiling of
  1.11 — the architectural-disconnection + ReLU-gate + in-domain-negatives bet paid off
  on the selectivity axis.
- **No memsinks-style all-active interference at any lr** — all-active own-prob is never
  below own-only (gaps +0.824/+0.179/+0.0005); at lr3e-3 branches are fully
  self-contained (gap +0.0005) with OOD leakage ood/own ≈ 0.0002.
- P1 smoke (446535): grad isolation structure ×16 layers, save→reload parity, loss
  components all live.
- **Deletion mechanics are clean, cheap, and MIA-safe** (2026-07-22, C on the lr2e-4 ckpt):
  drop forget10 → forget_Q_A_Prob 0.767→0.054, extraction 0.469→0.047, MIA loss 0.997→0.362
  (→ oracle floor), privleak −99.6→+4.0; physical slice removal **1.07 s**; dropall returns
  a symmetric floor (≈ base). The "remove the author's parameters" claim holds behaviorally.

## What didn't / open problems
- **THE HEADLINE (2026-07-22): the K=200 all-active recall ceiling ≈0.80 is structural, not
  a tuning miss.** Four K=200 points (lr 3.3× range × weight ÷10) all land on ONE
  recall-vs-selectivity curve topping ~0.80 in the healthy-sel band, vs the K=20 pilot's
  0.977. Both the lr dial (H-k200-lr refuted, H-k200-lr2 gray 0.795) and the suppression-
  weight rescale (H-wscale refuted, 0.696) fail to recover the pilot regime. Vincent's
  priors (0.97→0.32 deleted, utility Δ≤0.001) NOT reproduced at K=200.
- **Collective-recall interference at scale = memsinks all-on, reappearing.** own-only recall
  0.28–0.51 ≪ all-active; the architectural disconnection fixed *selectivity* (H1 ✓ to
  sel 1900) but NOT *all-active recall* with 200 branches live — the exact failure it was
  meant to beat. Direct cost measured: retain collateral −0.11 on forget10 deletion.
- **bs32 OOM'd** on the 44.5 GiB cards at step 9/390 (job 446705) — the smoke's 14.04 GiB
  was a bs2 measurement; pre-declared fallback bs16×ga2 applied and pinned in the
  configs.
- **Selectivity–recall lr tradeoff straddles the joint G2 bar** — no single arm passes
  both (0.981 prob at sel 4.38; 0.778 prob at sel 38.6).
- **Low-lr co-adaptation anomaly:** at lr3e-4 the own branch alone recalls almost nothing
  (own-only 0.155 vs all-active 0.979) — recall is carried collectively; deletion of
  *other* authors could hurt survivors in that regime. Gap shrinks to ≈0 by lr3e-3.
- Pre-reg sha pins for 3 files were stale (parallel-session edits 22:50–22:52 on 07-20,
  before any GPU run); corrected pins recorded in the 07-21 entry. `sacct` is empty on
  this cluster — monitor via squeue + output files.
- **K=200 at the pilot-winner lr over-suppresses (G3 FAIL):** sel 507.5 but median recall
  0.637 (0 authors ≥0.9), gap +0.130 > 0.05. Suppression pressure scales ~K× (each
  author's rows are negatives for K−1 others); the K=200 point lies on the K=20
  recall-vs-selectivity tradeoff curve. G3's gap≤0.05 + healthy recall appear jointly
  unsatisfiable for the pinned recipe — decision escalated (gates are not renegotiated
  after seeing results).
- Step-capped memory smokes don't bound worst-batch peaks: bs16 K=200 smoke passed at
  15.28 GiB, the real train OOM'd one step past the smoke's cap (~46.8 GiB demand) →
  bs8×ga4. Storage at 8.1 GB vs ≤6 GB budget (two disposable K=200 smoke ckpts, ~5 GB —
  deletion pending human approval).

## Open ideas / next steps

### MORNING DECISION PACKAGE (2026-07-22 — ladder halted, no passing arm, awaiting Jack)
State: H1 reproduces at K=20; at K=200 no arm clears recall ≥0.80 (best = A2 gray 0.795);
deletion mechanics clean/cheap but with real (aggregate-masked) retain collateral. **No
overnight GPU spend past this point** (protocol: no passing arm ⇒ no P4 replication, no P5).
Options for the morning, roughly in order of what the evidence favors:
1. **Diagnose the ceiling (recommended): a K∈{20,50,100,200} recall-vs-K ladder** at fixed
   lr 5e-4 (H-K-sweep). ~4 trains (K=20 already have the pilot). Answers "is 0.80 a smooth
   K-ceiling and what K did Vincent run" — the single most decision-relevant unknown. ~8 GPU-h.
2. **Ask Vincent** his K, adapter width, and whether K=200 serving is truly all-active-no-
   router (his 0.97 ft-baseline ≫ our 0.767 hints at a stronger per-author fit / smaller K).
   Zero GPU; could make (1) unnecessary or redirect it.
3. **Accept A2 (0.795) as the K=200 operating point** and run its full P4 replication row +
   P5 relearn — documents the method honestly at its achievable K=200 recall (below the
   0.869 MemAdapt Agg target, but a complete, clean, cheap-deletion story). ~6–9 GPU-h.
4. **Capacity ablation** (width 32→64/128, or fewer layers) under a new pre-registration —
   tests whether the ceiling is per-author capacity vs interference. Deferred; heavier.
Storage: cleanup of loser-arm weights (B `sepmlp_1b_k200_w15_s42/sepmlp.pt`, and possibly
A2/lr2e-4 depending on the choice) awaits approval — project ~8.4 GB now.

### Deferred (unchanged)
- Leak-probe arm (`measure_selectivity --probe forget_leak`) feeds the router_leak
  all-router table as the "selection inside the weights" row (routerless tombstone analog).
- Wave-2 ablations (width / layer-subset sweeps, w4=0 promotion ablation, silu variant)
  behind their own pre-registration.

## Entries (chronological)
- [2026-07-20 — Pre-registration + P0 build](2026-07-20_preregistration-build.md) — H1–H5 +
  H-scale pre-registered with CONFIRM/REFUTE bars before any SLURM job; frozen defaults,
  SwiGLU/λ pilot grid (later retired unrun), gate rules, budgets, and file inventory
  (sha256) recorded.
- [2026-07-20 — Spec-v2 pre-registration](2026-07-20_specv2-preregistration.md) — the
  user-supplied authoritative recipe (ReLU+bias gate, L1+10·L2+50·L3+1·L4, alternating
  negative batches incl. real_authors, per-author clip, cosine LR, detector init) re-pinned
  with revised G2 (lean lr-only pilot, no λ=0 clause), external priors P1–P4, fresh file
  sha256s, and the three implementation gotchas (float-order detach fix, cross-layer
  penalty-leak fix, 1-ulp removal-vs-mask scope). 69/70 CPU gates green pre-entry.
- [2026-07-21 — P1 smoke green · P2 pilot OOM→bs16×ga2 → lr tradeoff → G2 ADJUDICATE](2026-07-21_pilot-oom-and-adjudicate.md) —
  smoke PASS (14.04 GiB @bs2, parity, grad structure ×16); bs32 OOM → declared fallback;
  pilot medians sel/prob: 4.38/0.981 · 38.61/0.778 · 1909.7/0.695 — H1 refute bar
  cleared (vs LoRA 1.11), joint bar unmet → pre-registered ADJUDICATE: one bridging arm
  lr 5e-4 (job 446732). Stale-sha correction for 3 files; sacct-empty cluster note.
- [2026-07-21 — Bridging arm → G2 GO, H1 confirmed · P3 K=200 launched](2026-07-21_bridge-go-k200-launch.md) —
  lr 5e-4: median sel 7.171 / median own-prob 0.9765 (min 0.936) — both bars passed,
  **H1 CONFIRMED**, winner lr 5e-4; gap +0.73 caveat raised as H-gap; K=200 memory smoke
  15.28 GiB PASS; P3 train+probe launched (446910/446911 — train later OOM'd at step
  6/3750 ⇒ bs8×ga4 resubmit 446949/446950, recorded in the next entry).
- [2026-07-21 — P3 K=200 results: G3 FAIL](2026-07-21_k200-g3-fail.md) — train 446949
  healthy (2h00m, 33.04 GiB, no NaN) but at the pilot-winner lr the K=200 system lands at
  sel 507.5 / recall 0.637 / gap +0.130: H-scale ✓ (selectivity amplifies ~70×), G3 ✗
  (gap clause); suppression ∝ K finding; gap-clause-vs-recall joint-unsatisfiability
  analysis; STOPPED pre-P4 with a 4-option decision package (recommended: H-k200-lr
  arm at 2e-4 + human ruling on the gap clause) + smoke-ckpt deletion request (~5 GB).
- [2026-07-22 — H-k200-lr REFUTED by 0.003; K=200 tradeoff curve mapped](2026-07-22_hk200lr-refuted.md) —
  lr2e-4 arm (447162/447163, healthy): sel 36.0 (prediction ≈30 exact) but recall
  0.7468 < the 0.75 refute line. Two-point K=200 curve ≈ K=20 curve − 0.03 at matched
  sel; gap grows at scale (+0.45 at sel 36 vs +0.179 K=20). Verdict stands as
  registered. Next-step candidates: H-k200-lr2 (lr 1.5e-4) vs H-wscale (w2/w3 ÷10 —
  favored); storage 5.7/6 GB, /storage2 itself 97% full.
- [2026-07-22 — Pre-registration: H-wscale + H-k200-lr2 + P4-mechanics (user-approved triple)](2026-07-22_wscale-lr2-c-preregistration.md) —
  human chose B+A2+C and approved deleting the lr5e-4 G3-FAIL weights (executed,
  5.7→3.4 GB). Bars frozen before submission: H-wscale (w2/w3 1/5 @lr5e-4; predict sel
  5–15, recall ≥0.90) and H-k200-lr2 (lr 1.5e-4; predict sel 12–25, recall 0.80–0.87),
  both PASS = sel≥5 ∧ recall≥0.80, REFUTE = recall<0.75 ∨ sel<5; winner rule (higher
  recall s.t. sel≥5) → full P4 replication row; C = mechanics-only on the 0.747 ckpt
  (prefix sepmlp_lr2e4). Jobs: B 447175/447176 · A2 447177/447178 · C 447179 (%2);
  worst-case concurrency 4 = cap; OU integration installed ALLOW_DIRTY (no commit).
- [2026-07-22 — CORRECTION: ROUGE-L recall; "structural ceiling" retracted, gap is K-scaling](2026-07-22_rougeL-recall-correction.md) —
  paper (MUSR) is ground truth; audit found Eq 1–5 faithful but our gate metric wrong
  (answer-prob vs paper ROUGE-L). New `measure_recall.py` (76 CPU gates): **K=20 ROUGE-L 0.9573 ≈
  paper 0.96** (code correct at small K); K=200 best 0.7404 (tail 197/200) vs paper 0.966 (tail
  31/200); held-out 0.31 ≈ paper 0.341 (metric validated). Gap = K-scaling, not structural; tail
  signature ⇒ promotion deviation (D2a) is the lead fix. Jobs 447340/343/344/345/353.
- [2026-07-22 — H-promo result: REFUTED-but-CONFOUNDED (wrong lr); clean A/B relaunched](2026-07-22_promofix-results.md) —
  promofix K=200 @lr5e-4 (447384/5): ROUGE-L **0.639**, tail 200/200 — below A2's 0.740, BUT I
  anchored on lr5e-4 (the K=20 winner = the K=200 over-suppressing arm), confounding lr with the
  fix. Held-out 0.294 ≈ base ⇒ deficit is own-storage, not foreign leakage. Clean A/B pre-registered
  & launched: lr1.5e-4 + promofix (447458/9), the only change vs A2 (0.740) being the fix.
- [2026-07-22 — H-promo-clean REFUTED (promotion is not the K-scaling lever)](2026-07-22_promofix-clean-refuted.md) — clean A/B at lr1.5e-4 + promofix (447458/9): ROUGE-L **0.7377** / tail 199 vs A2 (no fix) 0.7404 / 197 — delta within noise. Promotion fix has NO recall/tail effect; D2a stays as conformance only. K=200 recall-vs-lr: 5e-4 0.639 / 2e-4 0.705 / 1.5e-4 0.74 (plateauing). Mechanistic lead = forward residual accumulation (199 foreign outputs); sweep candidates H-supp (higher lambda_out) / H-epoch / H-lr pre-registered, not run.
- [2026-07-22 — Pre-reg: D2a/D2b paper-conformance fixes + H-promo K=200 retrain](2026-07-22_promofix-preregistration.md) —
  **D2a** promotion now fires on all own tokens (paper §3.2; was question-only — the K=200 tail
  cause), `question_mask`→`own_token_mask`. **D2b** paper-exact `zero_wdown` deletion mode
  (`apply_droplist_file(mode=…)`) proven ≡ remove≡mask≡bake. CPU suite **77 passed**. H-promo bars
  frozen: CONFIRM K=200 ROUGE-L recall ≥0.90 ∧ tail ≤80; REFUTE ≤0.78. Config
  `sepmlp_1b_k200_promofix.json` (lr5e-4, the only change vs the deleted lr5e-4 G3-FAIL run is the
  D2a fix — a clean A/B). Smoke 447377.
- [2026-07-22 — H-wscale REFUTED · H-k200-lr2 GRAY · C deletion-mechanics (SUPERSEDED framing)](2026-07-22_wscale-refuted-lr2-gray-mechanics.md) —
  **the reproduction verdict.** B (H-wscale): sel 24.59 / recall 0.696 → REFUTED (weight
  rescale ≡ moving along the same curve). A2 (H-k200-lr2): sel 16.33 / recall 0.795 → GRAY
  (best K=200 point, below the 0.80 pass bar; per pre-reg no P4). Four-point K=200 curve
  tops ≈0.80 vs pilot 0.977 — recall ceiling structural under both lr and weight knobs. C
  mechanics (lr2e-4 ckpt): deletion clean (forget 0.767→0.054, extraction →0.047, MIA
  0.997→0.362→floor, privleak −99.6→+4.0) + cheap (slice removal 1.07 s) + retain
  collateral −0.113 (H-gap: real, aggregate |ΔUtil.R|=+0.003 masks it) + dropall symmetric
  floor ≈ base. Ladder halted (no passing arm); morning decision package above.
