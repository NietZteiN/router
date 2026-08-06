### Target Date: 2026-07-14 (Lean-phase results — mechanism binds, but memorization lives in shared capacity; all-slices-on serving self-interferes)
- **Hypotheses / what we're testing:** H1–H5 as pre-registered in
  [2026-07-14_preregistration-port-build.md](2026-07-14_preregistration-port-build.md)
  (M1 disjoint SeqTD-LoRA vs CTRL-L module-matched LoRA, seed 42, smoke tier).
- **Setup:** SLURM chain (all landed 2026-07-14 17:4x–18:50, sprint1-3, 1×A40 each, chained
  behind pfx-bake 443125-443127): **443146** msk-smoke ✓ → **443147** msk-train (task0 = M1
  `configs/memsinks_tofu_1b_disjoint.json`, task1 = CTRL-L `configs/memsinks_tofu_1b_ctrl_lora.json`;
  1250 steps each, ~10 min/run, train_loss 0.800 / 0.845, distinct-ID guard OK: all 200 authors
  seen, mask sha256 60891098…) → **443148** msk-bake (32 tensors each; del_forget10 = 240
  neurons, del05 = 120, del01 = 24, dropall = 2400, randdel = 240 [authors 14,16,22,34,71,81,…
  seed-42]; disjoint collateral = 0 exactly) → **443149** msk-eval 0-6%4 (eval_tofu smoke tier,
  `--k 10 --forget_shard_id 9`, KS ref = reused retain90 oracle). Total spend ≈ 3 GPU-h
  (est. was 12–15 — trains run 2.2 it/s on A40).
- **Results:** (smoke tier; fq granularity = 30 truth rows, ROUGE cap 50)

  | label | mu | fq | forget_rouge | forget_tr | retain_prob | retain_rouge | real_prob | world_prob | ppl f/r |
  |---|---|---|---|---|---|---|---|---|---|
  | ctrl_lora_full | **0.6438** | 0.0003 | 0.9425 | 0.3501 | 0.9296 | 0.9255 | 0.4656 | 0.5549 | 1.30/1.32 |
  | memsinks_full (all slices on) | **0.4373** | 0.0065 | 0.6936 | 0.4168 | 0.6917 | 0.6390 | 0.3291 | 0.5204 | 2.44/2.60 |
  | memsinks_del_forget01 | 0.4099 | 0.0025 | 0.6551 | 0.4172 | 0.6917 | 0.6234 | 0.3267 | 0.5214 | 2.44/2.60 |
  | memsinks_del_forget05 | 0.3535 | 0.0065 | 0.7191 | 0.4206 | 0.7034 | 0.6497 | 0.3342 | 0.5212 | 2.38/2.53 |
  | memsinks_del_forget10 | 0.3999 | 0.0065 | 0.6566 | 0.4206 | 0.7043 | 0.6525 | 0.3380 | 0.5096 | 2.39/2.50 |
  | memsinks_dropall (all sinks off) | **0.6399** | 0.0065 | **0.8726** | 0.4203 | 0.8910 | 0.8547 | 0.4488 | 0.5504 | 1.45/1.47 |
  | memsinks_randdel (placebo, 20 retained) | 0.4047 | 0.0025 | 0.7037 | 0.4152 | 0.7030 | 0.6403 | 0.3453 | 0.5215 | 2.36/2.53 |

  **H4 memorization-gap probe** (M1 train log, mean over 7 probe authors × 2 rows, answer-prob
  own-mask vs own-slices-deleted): epoch 1 gap 0.0601 → e2 0.2579 → e3 0.3079 → e4 0.3482 →
  **e5 0.3693** (own-mask per-author 0.714–0.957 at e5; deleted-condition also RISES 0.33→0.50
  across epochs). No NaNs; ppl 1.3–2.6 everywhere (no degenerate generation; smoke micro-run row
  `memsinks_smoke` is the 2-step pipeline gate, not a result).
- **What worked / hypothesis verdict:**
  - **H4 SUPPORTED** — the gap opens by epoch 5 (0.37): sequence-tied masking DOES bind
    per-author slices at 5 repetitions. Interpretation licensed.
  - **H1 REFUTED (for the pre-registered all-slices-on serving)** — mu 0.4373 < ctrl 0.6438 −
    0.10. But note the training-condition probe shows full memorization (own-mask prob 0.87–0.96
    for forget-authors 180/190/199), and **dropall serving reaches mu 0.6399 ≈ ctrl − 0.004** —
    the failure is the SERVING MODE, not trainability (see observations).
  - **H2 REFUTED** — gap closure on forget_rouge G = (0.6936−0.6566)/(0.6936−~0.39 oracle) ≈
    0.12 < 0.25. Decisive cross-check: with ALL sinks off, forget_rouge is **0.8726** — the
    shared capacity alone (gen neurons + lora_A + attention LoRA + down_proj) reproduces the
    forget answers nearly as well as the control (0.9425).
  - **H3 REFUTED in substance** — the mu-drop criterion technically passes (retain_prob
    0.6917→0.7043, no damage; disjoint collateral 0 as designed), but the **randdel placebo
    moves forget metrics as much as the real deletion** (forget_rouge 0.7037 vs 0.6566 vs full
    0.6936, all within the ±0.05 smoke-noise band) → deletion is not author-specific in effect.
  - **H5 REFUTED** — fq ≤ 0.0065 on every deletion condition (threshold 0.1).
- **Observations:** Two findings bigger than the verdicts:
  1. **All-slices-on serving self-interferes.** 200 authors' slices simultaneously active is
     off-distribution (training only ever activated one author's 12 neurons/layer at a time):
     mu 0.4373 / ppl 2.4–2.6, with EVERY component damped. Removing all 2400 sink deltas
     (dropall) recovers mu to 0.6399 / ppl 1.45. The paper never hits this because its "all"
     serving is expectation-scaled from-scratch training; in the FT port, slice deltas act as
     mutual noise. Corollary: the H4 probe's own-vs-deleted gap (0.37) largely measures
     *interference relief* + slice content mixed — own-mask (gen+own only) 0.90–0.95 vs
     dropall-class serving ≈0.87 rouge suggests the slices' own content increment is small.
  2. **Co-adaptation dominated:** at 5-epoch SFT with author-tied delta masking, memorization
     overwhelmingly ends up in shared capacity — the FT-regime confirmation of the paper's
     Thm 4.1 (and of this repo's equal-shard-ceiling reading). C1 (paper-faithful drop-all)
     shows NO differential forgetting: forget_rouge −0.070 vs ctrl, retain_rouge −0.071 —
     perfectly proportional, zero selectivity.
  - Noise notes: del05 forget_rouge (0.7191) > full (0.6936) and fq ties at 0.0065 are
    smoke-cap artifacts (ROUGE ≤50, truth ≤30 rows) — read the band, not the ordering.
  - CTRL mu 0.6438 < the historical 0.74: expected — module set differs (gate_proj added,
    7 target modules) and this is a fresh single seed; all comparisons here are internal.
- **New questions / new hypotheses:**
  - **H9 (serving):** per-author routed-mask serving (each TOFU query under its author's
    training mask, OOD → gen-only) restores full-model mu to ≈ctrl. Eval-only test — but if
    it's the only viable serving, MemSinks-as-FT loses its selection-free-serving selling
    point and becomes a mask-routing cousin of SIFT/ClAMU.
  - **H10 (interference ladder):** mu degrades monotonically with the number of active
    slices (bake k ∈ {1,10,50,100,200} active authors; eval-only).
  - **H11 (slice content):** per-author answer-prob under gen-only vs gen+own-slice
    quantifies how much each slice actually stores (one GPU probe pass); prediction from
    today's data: small (<0.1 prob increment).
  - Does more slice capacity (p_gen 0.5 → 20/author/layer) or per-example IDs shift the
    balance, or is co-adaptation binding regardless (paper Thm 4.1)? Round-2/3 dial.
- **Next Steps:** STOP per lean-phase agreement — review with user before any Round 2.
  Recommended next (all eval-only, ~2 GPU-h total, no retraining): H9 routed-mask serving
  eval + H11 gen-only/gen+own probe + H10 ladder. The pre-registered Round-2 controls
  (hashed arm, shuffled-ID, untied-dropout) are now secondary: the headline question has
  moved from "is deletion selective" (answered: no, nothing much is in the slices) to "can
  any serving mode make FT-MemSinks competitive, and where does the memorization actually
  sit".
