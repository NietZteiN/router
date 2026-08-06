### Target Date: 2026-07-15 (Phase-D results — routed serving restores control-level utility; slices confirmed near-empty; interference fully explains the lean-phase gap)
- **Hypotheses / what we're testing:** H9/H9-del/H10/H11 as pre-registered in
  [2026-07-15_round2-preregistration.md](2026-07-15_round2-preregistration.md).
- **Setup:** Eval-only on the existing M1 adapter (Round-1 job 443147 task 0). Jobs:
  **443549** msk-d-routed 0-1%1 (`eval_tofu --memsinks_config` arm, labels
  memsinks_routed_{full,unlearn}, smoke tier) + **443550** msk-d-probe
  (`probe_slices.py`, all 200 authors × 20 rows × {gen_only, gen_own, all_on} + 20-author
  ladder k∈{10,50,100}, seed 42 → `.../results/probe_slices.json`). Throttled to ≤2
  concurrent (other sessions' arrays held ~2 GPUs). E3 strict chain re-queued as
  **443562** (smoke) → **443563** (train) → **443564** 0-2%2 (evals) → **443565** (probe)
  after a CUDA bug fix (below); 443552-443554 scancelled (stranded on the failed smoke).
- **Results:**

  | label | mu | fq | forget_rouge | retain_prob | retain_rouge | ppl f/r |
  |---|---|---|---|---|---|---|
  | memsinks_routed_full | **0.6417** | 0.0025 | 0.9154 | 0.9040 | 0.8617 | 1.24/1.24 |
  | memsinks_routed_unlearn | 0.6417 | 0.0065 | **0.8726** | 0.9040 | 0.8617 | 1.45/1.24 |
  | (ctrl_lora_full, Round 1) | 0.6438 | 0.0003 | 0.9425 | 0.9296 | 0.9255 | 1.30/1.32 |

  **Probe (200 authors × 20 own train rows, answer-prob):** mean gen_only **0.9006**,
  gen_own **0.9139**, all_on 0.7372 → **slice_increment = 0.0133**, interference = 0.1767.
  Forget group ≈ retain group (increment 0.0145 vs 0.0131). **Ladder monotone fraction
  1.0** (20/20 authors nonincreasing in k; total k=0→199 drop ≈ 0.177 ≥ 0.10).
- **What worked / hypothesis verdict:**
  - **H9 SUPPORTED** — routed serving mu 0.6417 ≥ 0.59 gate (ctrl − 0.002; retain_ppl 1.24 ≤
    1.6). The all-slices-on deficit was pure serving interference; wrapper-sanity floor passed
    → E3 GO.
  - **H9-del SUPPORTED on the expected branch** — routed deletion forget_rouge 0.8726 ∈
    0.87±0.05 (exactly the gen-only level; = Round-1 dropall to 4 decimals, as it must —
    deleted-author queries under routed-unlearn ARE gen-only serving); retain side identical
    (mu unchanged 0.6417). Gap closure G ≈ 0.08 ≪ 0.5: nothing deletable in the slices.
  - **H11 SUPPORTED** — slice_increment 0.0133 < 0.10: shared capacity answers own train rows
    at 0.90 on its own; an author's 12-neuron slices add ~1.3 points.
  - **H10 SUPPORTED** — interference monotone in foreign-slice count for 100% of ladder
    authors, total ≈ 0.18.
- **Observations:** (1) The lean-phase H4 "memorization gap" (0.37) is now decomposed:
  ≈ interference relief + 0.013 slice content — the apparent binding was almost entirely a
  serving artifact, not localization. (2) Routed MemSinks ≈ the control with an oracle
  router: it inherits the serve-time-selection cost of SIFT/ClAMU while storing nothing
  deletable — strictly dominated as an unlearning method in this regime. (3) **Bug found
  and fixed (443551):** `train_lora_shard.apply_irp_projections` inits CUDA weights with a
  CPU torch.Generator → RuntimeError on GPU (its CPU-fixture gate can't catch this). Fix =
  `train_memsinks.freeze_lora_a_irp` (identical SHA-256 seeding, draw on CPU, copy to
  device); new gate `test_irp_port_equivalence` proves bit-equality on CPU; suite now
  **21/21 green**. (4) Cap discipline event: another session raised its array throttle to %3
  mid-flight (other sessions' running GPU jobs hit 4); my pending e3 chain was `scontrol
  hold` and auto-releases when their usage drops ≤ 2 (transient 5-GPU overlap ≈ minutes,
  from the running smoke task).
- **New questions / new hypotheses:** With D fully resolved (serving = interference,
  storage = shared), H14 is the only remaining live question: does FORCED isolation
  (strict arm) make per-author slices carry the content — at what utility? Deferred H12/H13
  (e25, starvation) predictions can now be sharpened: starvation must move slice_increment
  from 0.013 toward ≳0.5 to matter.
- **Next Steps:** Harvest E3 (443562-443565) → verdict H14 vs pre-registered gates →
  Round-2 results entry + `memsinks_tofu/REPORT.md` → review with user.
