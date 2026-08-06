# memsinks — MemSinks/SeqTD (sequence-tied dropout) ported to TOFU as masked-LoRA-delta

**Status:** active (H14″ complete — thread-close decision pending; report: [`memsinks_tofu/REPORT.md`](../../memsinks_tofu/REPORT.md)) · **Project:** [`memsinks_tofu/`](../../memsinks_tofu/) · **Entries:** 9 (2026-07-14 → 2026-07-16)

MemSinks (Ghosal, Maini, Raghunathan, ICML 2025; `papers/Memorization Sinks Isolating
Memorization during LLM Training.pdf`, code `~/MemSinks/` = `AR-FORUM/MemSinks@a005119`)
isolates memorization at TRAINING time: MLP intermediate neurons split into always-on
"general" neurons (p_gen) and a sink pool, each sequence ID activating only its own hashed
sink subset; deletion = zero sink neurons at inference. Until now the paper served this repo
only as theory (co-adaptation Thm 4.1 → the equal-shard ~0.48 ceiling). This thread RUNS it —
the training-time-allocation cell of the "selection must live somewhere" frame. Port is an
**extension, not a repro**: sequence-tied masking of the **LoRA fine-tuning delta** (gate/up,
base path untouched) on Llama-3.2-1B-Instruct, author-level IDs 1–200 (ID 0 = degenerate
all-ones hash mask), disjoint 12-neuron/author/layer slices primary (hashed p_mem=0.3
forget10 union = 97.4% of the pool → selective ≡ total there). Deletion op is exact (bitwise
bake of lora_B rows, bake≡hook unit-tested); unlearning is structurally approximate.

**Lean-phase verdict (2026-07-14): the mechanism binds, but it doesn't localize.** The
per-author memorization gap opens during training (H4 ✓, gap 0.37 by epoch 5), yet shared
capacity memorizes nearly everything (sinks-off forget ROUGE 0.87 vs ctrl 0.94), deleting the
forget slices ≈ deleting random slices ≈ deleting nothing (H2/H3/H5 ✗), and the pre-registered
all-slices-on serving mode self-interferes (mu 0.44 vs 0.64 with all sinks off — 200
simultaneous author-deltas act as mutual noise). Co-adaptation dominates 5-epoch SFT exactly
as Thm 4.1 predicts; C1 (drop-all) shows zero differential forgetting in the FT regime.

## Hypotheses — open / resolved
- **[resolved ✓ supported]** H4 (mechanism binding): memorization gap opens by epoch 5 —
  0.0601→**0.3693** monotone ([lean-phase-results](2026-07-14_lean-phase-results.md)); but see
  the interference caveat — the gap conflates slice content with interference relief.
- **[resolved ✗ refuted]** H1 (trainability under all-slices-on serving): mu 0.4373 < ctrl
  0.6438 − 0.10. Training-condition memorization itself is fine (own-mask prob 0.87–0.96);
  the serving mode is the failure — dropall serving reaches 0.6399 ≈ ctrl.
- **[resolved ✗ refuted]** H2 (localization): forget-rouge gap closure G ≈ 0.12 < 0.25;
  sinks-off forget_rouge 0.8726 — shared capacity reproduces the forget answers.
- **[resolved ✗ refuted]** H3 (selectivity, in substance): retain unharmed (disjoint
  collateral 0 by construction) but the randdel placebo moves forget metrics as much as the
  real deletion (0.7037 vs 0.6566 vs full 0.6936, ±0.05 noise band).
- **[resolved ✗ refuted]** H5 (forget quality): fq ≤ 0.0065 on every deletion condition.
- **[resolved ✓ supported]** H9: routed-mask serving restores control-level utility — mu
  **0.6417** vs ctrl 0.6438, ppl 1.24 ([phase-d-results](2026-07-15_phase-d-results.md)).
  Consequence: FT-MemSinks needs serve-time selection → selection-free story does not survive.
- **[resolved ✓ supported, expected branch]** H9-del: routed deletion ≈ no-op — forget_rouge
  0.9154→0.8726 (= the gen-only level exactly), G ≈ 0.08; retain untouched.
- **[resolved ✓ supported]** H10: interference monotone in foreign-slice count for 100% of
  ladder authors; total ≈ 0.18 answer-prob.
- **[resolved ✓ supported]** H11: **slice_increment = 0.0133** (gate <0.10) — shared capacity
  answers own train rows at 0.90 alone; the lean-phase H4 gap (0.37) was ≈ interference
  relief + 0.013 content. The slices are near-empty.
- **[resolved ✗ refuted-as-implemented]** H14 (E3 strict isolation): training DIVERGED
  (loss 2.7→8-12; trained slices corrupt own-author rows to 800x BELOW the base,
  slice_increment −0.139) — an optimization blow-up (std-1.0 frozen lora_A x rslora ~11.3,
  clipping disabled), NOT a capacity floor ([strict-isolation-results](2026-07-16_strict-isolation-results.md)).
  The capacity question stayed open as H14′.
- **[resolved ✗ refuted-as-underfit]** H14′ (scale-corrected retry): divergence FIXED (stable
  loss, slices carry ALL the learning — isolation works as designed) but 40 rows/author learn
  ~nothing at **5 optimizer steps/author** (own-prob 0.16 ≪ 0.80; mu 0.4466; slice_increment
  +0.023) — health gate failed on the underfit side, so the capacity refutation is STILL not
  licensed ([h14prime-results](2026-07-16_h14prime-results.md)). Steps-per-author vs frozen-A
  expressivity confounded at e5; **H14″ (e25 steps dial)** is the disambiguator (open).
  Bonus: strict2_all_on = clean in-adapter merging-collapse demo (mu 0.03, ppl 93).
- **[resolved ✗ refuted — LICENSED]** H14″ (e25 steps dial): with healthy, saturated training
  (loss 1.0-1.2; own-prob +0.006 over e20→25), 40 frozen-basis rows/author plateau at recall
  **0.389 probe / 0.504 trajectory ≈ HALF a full adapter's 0.9991** — the quantified per-author
  capacity floor ([h14doubleprime-results](2026-07-16_h14doubleprime-results.md)). AND the
  mechanism finally works end-to-end: mu 0.6305 ≥ gate; **deletion real + floor-perfect**
  (forget_rouge 0.5537→0.4647 = the never-trained floor exactly; fq 0.135→0.3929 = SIFT-unlearn's
  0.393; retain bit-unchanged; isolation 0.000 from epoch 5) at ~80 KB/author. all_on collapse
  scales with slice strength (ppl 93→222k) = merge_mechanism H8 inside one adapter. Open dial
  (deferred): capacity ladder s 40→80→160 toward SEA's 0.99 at 32 MB.
- **[open]** (deferred) H6 MIA spectrum; H7 competitiveness; H8 capacity dial; H12 e25
  repetition arm; H13 shared-starvation arm; hashed/shuffled-ID/untied-dropout controls.

## What worked
- 14/14 CPU gates green pre-launch (verbatim hash port ≡ reference incl. int64-overflow
  quirk; bake≡hook bit-identity; gradient/deletion isolation; KV-cache identity; collator
  parity) — [preregistration-port-build](2026-07-14_preregistration-port-build.md).
- The training mechanism binds (H4 ✓); pipeline end-to-end in ≈3 GPU-h; exact-deletion bake
  verified (0 collateral in disjoint mode).
- ID-0 hash degeneracy discovered (all-ones mask) before it could poison runs.

## What didn't / open problems
- Localization fails in the 5-epoch SFT regime: memorization co-adapts into shared capacity
  (gen neurons, lora_A, attention LoRA, down_proj); forget-slice deletion ≈ placebo ≈ no-op
  on forget metrics; fq stays ≤0.0065 ([lean-phase-results](2026-07-14_lean-phase-results.md)).
- All-slices-on serving self-interferes (mu 0.4373 vs dropall 0.6399) — a serve-mode problem
  the from-scratch paper never faces; makes the pre-registered "full" serving unusable.

## Open ideas / next steps
- THREAD-CLOSE DECISION with user ([REPORT.md](../../memsinks_tofu/REPORT.md), ≈9 GPU-h
  spent): the story is complete — mechanism-negative for FT-MemSinks-as-published, plus the
  strict-arm dial (blow-up → underfit → licensed capacity floor 0.39/0.50 with real,
  floor-perfect, provenance-exact deletion at 80 KB/author).
- If extended: capacity ladder s 40→80→160 (or trainable-A-per-slice) toward SEA's 0.99 @
  32 MB — the storage-vs-recall tradeoff curve; then seeds 43/44 + extended tier + H6 MIA
  (attack_mia flags already landed) before any graduating claim.
- Deferred: H13 starvation arm, hashed/shuffled-ID/untied-dropout controls.

## Entries (chronological)
- [2026-07-14 — Pre-registration + port build](2026-07-14_preregistration-port-build.md) — port built, 14 CPU gates green, H1–H5 pre-registered before first SLURM job
- [2026-07-14 — Lean-phase results](2026-07-14_lean-phase-results.md) — H4 ✓ mechanism binds; H1/H2/H3/H5 ✗ memorization co-adapts into shared capacity; all-slices-on serving self-interferes (mu 0.44 vs 0.64); deletion ≈ placebo
- [2026-07-15 — Round-2 pre-registration](2026-07-15_round2-preregistration.md) — Phase D (routed serving H9/H9-del + slice-content/interference probe H11/H10) + E3 strict-isolation arm (H14; disjoint_dead + frozen lora_A; momentum-tail refinement of the exactness claim); 20/20 CPU gates
- [2026-07-15 — Phase-D results](2026-07-15_phase-d-results.md) — H9/H9-del/H10/H11 all ✓: routed serving = ctrl (mu 0.6417), slices near-empty (increment 0.013), interference monotone; lean H4 gap decomposed; IRP CUDA bug found+fixed (21/21 gates)
- [2026-07-16 — Strict-isolation results](2026-07-16_strict-isolation-results.md) — H14 refuted-AS-IMPLEMENTED: optimization blow-up (std-1.0 frozen A, no clip), not a capacity floor; H14′ scale-corrected retry designed, unrun; routed wrapper + provenance machinery verified sound
- [2026-07-16 — H14′ pre-registration](2026-07-16_h14prime-preregistration.md) — scale fix (std auto + clip 0.3, same seeded directions), training-health gate added, 22/22 CPU gates
- [2026-07-16 — H14′ results](2026-07-16_h14prime-results.md) — refuted-as-UNDERFIT: stable training, isolation works, but 5 steps/author learn ~nothing (own-prob 0.16); H14″ e25 dial opened; clean all-on collapse demo
- [2026-07-16 — H14″ pre-registration](2026-07-16_h14doubleprime-preregistration.md) — e25 steps dial, three-way gate (capacity-confirm / licensed-refute / mixed)
- [2026-07-16 — H14″ results](2026-07-16_h14doubleprime-results.md) — capacity floor LICENSED (plateau 0.39/0.50 ≈ half a full adapter); deletion real + floor-perfect (fq 0.393 = SIFT-unlearn; retain bit-unchanged) at 80 KB/author; all_on collapse scales with slice strength
