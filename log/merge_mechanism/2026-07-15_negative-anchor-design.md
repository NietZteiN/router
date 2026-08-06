### Target Date: 2026-07-15 (Negative-anchored isolation — design & λ-pilot pre-registration)
- **Hypotheses / what we're testing:** §6.3 of [PATHS_FORWARD](../PATHS_FORWARD_2026-07-13.md),
  now UNGATED by the Exp-7 verdict ([2026-07-15_key-firing-results.md](2026-07-15_key-firing-results.md):
  keys LAZY, median on/off 1.10, OOD firing ≈ 90% of on-author — the missing-negatives story
  is real). Train each per-author expert with an added penalty on public, author-independent
  text: **minimize mean ‖sᵢBᵢAᵢh‖²** over Alpaca tokens while training normally on author i —
  soft self-gating. Exactness: the anchor set is public + seeded, so training stays a pure
  deterministic function of (author shard, anchor set, seeds); deletion = drop the adapter,
  unchanged.
  - **H-anchor-1 (selectivity):** anchoring raises the on/off ‖sBAh‖ ratio from ≈1.1 to ≥5
    (the pre-registered SELECTIVE bar) at some λ, without own-author recall falling below
    0.98 answer-prob (e25 reference 0.9992). CONFIRM: some pilot λ satisfies both. REFUTE:
    every λ either stays <2 (penalty too weak / keys can't decouple) or destroys recall
    (facts and generic firing are inseparable — itself a mechanism result).
  - **H-anchor-2 (the merge payoff / utility survives):** anchored e25-recipe experts merged
    `additive_mean` beat the H8 strong-expert curve at N∈{4..20} — CONFIRM: N=8
    subset-conditioned retain_prob > 2× H8's 0.282 (≥0.56) with global mu ≥ the H8 ladder's
    (0.40–0.44) and retain_ppl < 20. REFUTE: within noise of the H8 curve (self-gating in the
    weights cannot substitute for external selection). Tested AFTER the pilot picks λ.
  - Pilot λ-selection rule (pre-registered): among λ ∈ **{1, 10, 100}**, pick the largest λ
    with (median selectivity ≥ 5) AND (mean own-author answer_prob ≥ 0.98); if none reaches
    ≥5, take the best-selectivity arm with recall ≥ 0.98 and record the shortfall. λ scale
    reasoning: penalty is the per-module per-token mean SQUARED norm; e25 adapters measure
    ‖sBAh‖ ≈ 0.63/module ⇒ penalty ≈ 0.4 at init vs CE ≈ 2–3, so λ=1/10/100 spans
    gentle→dominant.
- **Setup:** planned. `train_lora_shard.py` gains `--anchor_lambda/--anchor_n/--anchor_seed/`
  `--anchor_batch_size/--anchor_source` (defaults OFF — flag-free behavior bit-unchanged, the
  frozen-recipe invariant). `AnchoredSFTTrainer.compute_loss` = SFT CE + λ·penalty; penalty =
  one extra forward per step over a cycling pre-tokenized Alpaca batch
  (`skill_data.load_alpaca(2000, HF_HOME, seed=42)`, `to_text` = the training text schema),
  adapter outputs captured by forward hooks on `lora_B` (scaling applied once, fp32
  accumulation, pad-masked; differentiable → gradients shape A and B). CPU gate
  `test_train_anchor.py`: hook-penalty ≡ dense closed form on a tiny fixture (dropout 0),
  λ=0 ≡ vanilla SFTTrainer loss, penalty→0 when B=0, gradient flow. Pilot = probe authors
  perm(42)[:5] = [82, 15, 111, 177, 76] × λ∈{1,10,100} at the e25 recipe (`--k 200
  --epochs 25`, frozen recipe otherwise, seed 42) → dirs
  `Llama-2-7B-chat-hf_k200_r32_e25_anch{λ}_lr1e4/shard_{a}` (15 × ~10-min 1-GPU jobs, %≤4).
  Readout order: (P2) `measure_key_firing.py` per λ-pool (minutes each; H-anchor-1) →
  (P3) iso rows `eval_tofu --preloaded_adapter --eval_shard_id a --smoke` for surviving arms
  (recall check; ≤15 × ~1 GPU-h). Full 20-author arm + the anchored H8 ladder
  (`nmerge` e25-anchored config) only after the pilot picks λ.
- **Results:** *(pending — pre-registration only.)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending; watchlist: OOM from the extra anchor forward (reduce
  anchor_batch_size), CE-vs-penalty imbalance signatures (loss curves logged), recall
  collapse at high λ, selectivity measured with the SAME harness/seed as Exp-7 so ratios are
  directly comparable.)*
- **New questions / new hypotheses:** if H-anchor-1 confirms, does anchored selectivity also
  seal the §9-D router leak (rerun drop-an-expert on anchored experts — PATHS_FORWARD §7.1
  direction 1)? Does anchoring change col(B) shared-basis energy (Exp-1 re-run, CPU)?
- **Next Steps:** implement + CPU gate → pilot train array → keyfire per pool → iso evals →
  results entry with the λ verdict → full 20-author anchored pool + anchored N-ladder
  (H-anchor-2) alongside the Exp-6 centered results.
