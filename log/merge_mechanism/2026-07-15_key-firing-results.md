### Target Date: 2026-07-15 (Key-firing results — keys are LAZY; §6.3 negative anchoring is GO)
- **Hypotheses / what we're testing:** H-key-1 (lazy keys, gate: median on/off ‖sBAh‖ ratio
  < 2.0 ⇒ LAZY), H-key-2 (read vs write selectivity locus), H-key-3 (training dose e5 vs e25),
  as pre-registered in [2026-07-15_key-firing-design.md](2026-07-15_key-firing-design.md).
- **Setup:** as pre-registered, no deviations. `measure_key_firing.py` (sha256-16
  `729c09db9baa0ae5`; repo not git-tracked), CPU gate `test_measure_key_firing.py` green
  before submission (hook/Gram-trick ≡ dense at 3e-8). Base `meta-llama/Llama-2-7B-chat-hf`
  bf16, 1×A40; 200×5 seeded TOFU questions + 100 world_facts + 100 real_authors + 100 Alpaca
  (seed 42; 1300 prompts, 192 slots, 203 groups). Jobs: e5 arm **443446** (200 adapters,
  `_k200_r32_e5_lr1e4`, ~3 min runtime), e25 arm **443477** (20 adapters,
  `_k200_r32_e25_lr1e4`). Outputs `reports/key_firing_{e5,e25}.json` + `_matrices.npz`.
- **Results:** (median per-adapter ratios; "on/off" = own-author / other-author mean-token norm)
  | metric | e5 (n=200) | e25 (n=20) |
  |---|---|---|
  | gate: on/off ‖sBAh‖ (mean-tok) | **1.1018** | **1.1102** |
  | fraction of adapters < 2.0 | **1.00** | **1.00** |
  | fraction ≥ 5.0 | 0.00 | 0.00 |
  | on/off ‖Ah‖ (read side) | 1.0119 | 1.0439 |
  | on/off ‖sBAh‖ last-token | 1.0502 | 1.0862 |
  | attn / mlp | 1.080 / 1.150 | 1.088 / 1.155 |
  | layer terciles L0/L1/L2 | 1.042 / 1.093 / 1.154 | 1.051 / 1.102 / 1.150 |
  | mean absolute on / off | 0.1836 / 0.1664 | **0.6330 / 0.5668** |
  | mean OOD firing (wf / ra / alpaca) | 0.1645 / 0.1650 / 0.1600 | 0.5460 / 0.5648 / 0.5269 |
- **What worked / hypothesis verdict:**
  - **H-key-1 SUPPORTED — keys are LAZY, decisively:** gate median 1.102 (e5) / 1.110 (e25),
    both ≪ the 2.0 threshold; **every single adapter in both sets is below 2.0**, none near 5.
    Per the pre-registered gate, **§6.3 negative-anchored isolation is GO**.
  - **H-key-2 SUPPORTED (in the null direction):** the read keys carry essentially zero
    discrimination (‖Ah‖ ratio 1.012/1.044 ≈ 1); the small residual selectivity (1.10–1.15)
    appears only in the composed write norm. Neither side is selective — "lazy" describes the
    whole adapter, not just A.
  - **H-key-3 REFUTED (dose does not sharpen):** e25 selectivity is statistically
    indistinguishable from e5 (1.110 vs 1.102) while the absolute firing magnitude grows
    **~3.4×** (on 0.184→0.633, off 0.166→0.567). More training amplifies unselective firing;
    it does not focus it.
- **Observations:** (i) Adapters fire on fully out-of-domain public text at ~87–93% of their
  on-author magnitude (e5 OOD 0.160–0.165 vs on 0.184) — they respond to *anything*, not even
  specifically to QA-shaped TOFU text. This is the strongest form of §4.5's prediction and
  directly validates Alpaca as the §6.3 anchor corpus (the penalty targets firing that
  demonstrably exists today). (ii) The e5/e25 contrast supplies the missing mechanism link for
  H8: strong experts merge *worse* because their (equally unselective) outputs are 3.4× larger
  — crosstalk amplitude scales with training while selectivity stays fixed ≈1.1. (iii) The mild
  gradients (mlp > attn, deep > shallow) are consistent with fact storage in later MLPs but are
  ~0.07 effects — nothing gate-relevant. (iv) Silent-failure checks: CPU gate proved
  hook≡dense; group counts as planned (203 groups / 1300 prompts); no NaNs; CUDA mem 24.7 GiB
  as budgeted; both JSONs carry job IDs + script sha.
- **New questions / new hypotheses:** H-anchor-1 (next): an ‖BᵢAᵢh‖→0 penalty on Alpaca during
  per-author training raises the on/off ratio well above 2 without hurting own-author recall
  (e25 recipe reference: 0.9992 answer-prob at N=1). H-anchor-2: anchored experts merged with
  `additive_mean` beat the H8 e25 curve at N∈{4..20} (e.g. N=8 subset retain_prob > 0.282).
  Re-measure selectivity on anchored adapters with this exact harness (free, same script).
- **Next Steps:** write the §6.3 anchored-training design entry (trainer penalty via
  SFTTrainer subclass + λ pilot on probe authors at the e25 recipe); implement + CPU gate;
  λ pilot within the 4-GPU cap; then the anchored 20-adapter pool + H8-ladder rerun +
  selectivity re-measurement. Centered-merge evals (Exp 6) remain queued behind merge array
  443445.
