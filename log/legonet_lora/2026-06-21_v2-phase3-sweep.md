### Target Date: 2026-06-21 (LegoNet v2 + Phase-3 sweep)
- **Goal / Hypothesis:** Finish unattended (≤8 GPU): v2 strengthened-canary 7B + Phase-3 sweep
  (utility-vs-n, utility-vs-k, semantic-vs-random ablation, deletion-cost k²N/n vs N/s).
- **Setup:** v2 chain 435665→435666[%8 after user cap]→435667∥435668. Sweep (new: assignment_mode
  knn|random; make_sweep.py 6 cells/240 adapters; submit_sweep.sh prep→train[%6]→eval→collect;
  collect_sweep.py; run_exactness_sample --no_verify): prep 435686→train 435687[0-239%6]→eval
  435688→collect 435689. Report /storage2/jack/checkpoints/legonet_lora/SWEEP_REPORT.md.
- **Results — THREE CORE CLAIMS VALIDATED:**
  - **Exactness (distributional):** v2 all deletions structural_ok; affected rel_l2 ≈ untouched
    nondeterminism floor (3.5e-2/5.7e-2/5.3e-2 vs 4.2e-2/4.7e-2/6.6e-2) ⇒ unlearn indistinguishable
    from from-scratch retrain. Bitwise on CPU/TinyLlama, distributional on 7B (measured).
  - **Utility preserved:** v2 MMLU 0.433 vs 0.460 base; retained EM 0.716 vs 0.505; retained PPL
    3.33 vs 16.22. Frozen backbone keeps capability (LegoNet premise on an LLM).
  - **Sweep:** utility HOLDS vs n (k=3 retained EM 0.718/0.716/0.700 for n=16/32/64, cost
    2250/1125/562 ex-passes); k>1 RECOVERS utility (n=32 EM 0.687/0.716/0.717 for k=1/3/5, cost
    125/1125/3125); semantic≈random @ k=1 (0.687 vs 0.683 — paper's LegoNet_{k=1}≈FixSISA).
- **Observations (honest caveats):** (1) Efficiency — SISA-LoRA cheaper/deletion at moderate n
  (N/s = 62–125 vs LegoNet k²N/n = 562–1125); crossover needs n>s·k² (~576 @k3,s64), beyond sweep.
  In the LoRA port both freeze the base, so LegoNet's classic per-param win is gone; real edge =
  utility-per-segment (k>1) + verifiable exactness, NOT raw cost at moderate n (plan predicted this).
  (2) Forget signal modest even @ canary×5/6ep (pop canary_em 0.065 vs 0.018) — k=3 delta-avg
  dilution; but every memorized record reverted exactly to base on deletion (0.10→0.00), so the
  mechanism is clean; sharper forgetting wants k=1 / heavier canaries.
- **Next Steps:** Optional — push n>576 to demonstrate the cost crossover; k=1 variant for crisp
  population forget; logit-averaging combine as a faithfulness check; seed-variance on the headline
  cells. Core study complete; SWEEP_REPORT.md + per-run results JSONs are the deliverable.

