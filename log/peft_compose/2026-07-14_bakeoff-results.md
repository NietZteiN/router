### Target Date: 2026-07-14 (bake-off Phase A results — composition is operator-independent)
- **Hypotheses / what we're testing:** H1–H4 as pre-registered in
  [2026-07-08_bakeoff-design.md](2026-07-08_bakeoff-design.md) (prefix keeps own-recall; VeRA/IA³
  beat the LoRA additive anchor but dilute; deletion exact everywhere; IA³/prefix capacity risk).
- **Setup:** exactly the pre-registered protocol. Llama-3.2-1B-Instruct, k=10, seed 42,
  `configs/peft_bakeoff_1b.json`. SLURM: micro-smokes **442746** (all 4 methods passed; trainable
  params/shard: vera 74K, ia3 147K, prefix 1.05M, dora 17.5M); Phase A chain **443125** (40
  trainings, %4) → **443126** (compose; exact-delete asserts <1e-6; ia3 geo sign-fallbacks
  0/147,456) → **443127** (28 evals, smoke caps). Script sha256s: train_peft_shard `941ab4cb8d89`,
  compose_peft `ffeb7f6c21ab`, prefix_concat `af7375bbee3f`, test_compose_peft `de2b4e13cf7b`,
  submit_peft_bakeoff `849c10c966f8`. `test_compose_peft.py` green before submission; both DoRA
  merge probes (linear/cat) accepted by peft 0.14 with magnitude vectors preserved. Full table:
  `../../tofu_sisa_lora/reports/PEFT_BAKEOFF_2026-07.md`; raw JSONs under each
  `checkpoints/Llama-3.2-1B-Instruct_peft_{method}_k10/results/smoke/`.
- **Results:** anchors: base mu 0.3796 / joint-ft 0.5302 / LoRA `merged_additive_mean` 0.4190.
  Composed_full mu: VeRA **0.4150**, IA³-mean **0.4298**, IA³-geo **0.4302**, DoRA-additive
  **0.4317**, prefix-concat **0.0018** (f_ppl 9,248). Iso probes: IA³ own-shard f_rouge
  0.52–0.57 / f_ppl 2.6 (best memorizer); VeRA 0.42–0.46 / 5.2; DoRA 0.48–0.53 / 2.9; prefix iso
  mu 0.036–0.074 ≪ base with retain_prob 0.02–0.06 (serving one prefix wrecks general behavior)
  despite own-shard rouge 0.38–0.47. Composed-vs-iso own-shard-9 recall drop: IA³ −0.096,
  DoRA −0.058, VeRA −0.020 (weakest iso). Deletion (drop shard 9): forget_ppl 9.32→10.08 (vera),
  9.32→11.33 (ia3), 9.23→11.26 (ia3-geo), 7.99→10.01 (dora), 7.53→9.23 (LoRA anchor); fq
  0.59–0.96. Routed_key_exact per pool: IA³ **0.5155**, DoRA 0.4906, VeRA 0.4447, prefix 0.0000.
- **What worked / hypothesis verdict:**
  - **H1 REFUTED** — prefix-concat collapses (mu 0.0018; own-shard rouge 0.466→0.153 = −0.31,
    bar was ≤0.05). Independently-trained prefixes are mutually OOD; attention over the
    concatenation does not route, it drowns.
  - **H2 REFUTED** — no material mu win: VeRA/IA³ composed 0.415–0.430 vs LoRA 0.419 (Δ ≤ +0.011);
    the dilution clause held for the strong memorizers (IA³ −0.096 ≈ LoRA's Exp-3 −0.090).
  - **H3 SUPPORTED** — deletion exact by construction in every arm (compose identity asserts;
    byte-exact prefix segment drop) and behaviorally right-signed (forget_ppl up everywhere).
  - **H4 SPLIT** — supported for prefix (fails the capacity/serving gate), refuted for IA³
    (best per-parameter memorizer in the bake-off).
  - **Headline: the composition plateau is operator-independent.** Four different composition
    operators (additive LoRA/DoRA, shared-basis mean, gate mean, gate product) in four
    parameterizations all land at base+0.04 (0.415–0.434) — the same "constant style adapter"
    plateau Exp-5 measured for LoRA at every N. Changing the parameterization does not rescue
    input-blind composition.
- **Observations:** (a) Routing > composition in every parameterization — the strongest
  cross-method confirmation yet of the routing thesis; notably **IA³ + author-key routing hits
  0.5155 ≈ joint-ft 0.5302 with 1.5 MB of adapters** (100× smaller than LoRA r32 pools) — a
  practical serving result independent of the composition question. (b) IA³ geometric vs
  arithmetic gate composition indistinguishable (Δmu 4e-4; all gates sign-consistent) — product
  vs mean does not matter when gates stay near 1. (c) Prefix failure mode is serving-destruction
  (response distribution overwritten), not storage: informative for any future prompt-learning
  arm — needs joint or scaffold-aware training, not just more tokens. (d) Silent-failure checks:
  no NaNs in headline metrics (fq NaN only for the three `--eval_shard_id 5` probes where the
  KS reference doesn't apply); prefixcat's mu 0.0018 is a real measurement (ppl 9,248), not a
  crashed eval. (e) mu at smoke caps: iso mu for every method sits at/below base because a
  single 20-author adapter can't lift the general components — iso f_rouge/f_ppl are the
  capacity-gate reads, as designed.
- **New questions / new hypotheses:**
  - **H-ia3-route-200 (open):** IA³ + key routing holds ≈ joint-ft utility at k=200 per-author
    granularity at ~30 MB total — if yes, it's the cheapest exact-deletion serving stack we have
    (routed, not composed; Phase-B-adjacent but under the routing thesis, not the compose gate).
  - **H-prefix-joint (open):** prefixes trained jointly-then-partitioned (or on the scaffolded
    base) stop destroying general behavior and make concat-composition meaningful.
  - Does the plateau move at all with model scale (7B) for any operator, or is base+ε universal?
    (Low priority — three independent lines now say composition without input-conditioning is
    dead at this granularity.)
- **Next Steps:** Phase B **not triggered** (gate: composed mu ≥ 0.55; best arm 0.4317). Thread
  goes to `phaseA-complete`; the actionable follow-up is H-ia3-route-200 under the routing
  serving mode.
