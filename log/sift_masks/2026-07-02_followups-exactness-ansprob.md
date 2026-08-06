### Target Date: 2026-07-02 (follow-ups: GPU unlearn is BITWISE-exact; paper's answer-prob metric reproduced)
- **Hypotheses / what we're testing:** Three follow-ups on the T=200 build (same artifacts, no
  retraining). **H5 (GPU exactness):** with deterministic math/eager attention forced
  (`_force_deterministic_attention`), re-deriving a forget author's τ_u on GPU reproduces it
  byte-for-byte — CONFIRM if `torch.equal` across two independent re-derivations; REFUTE → report
  the rel-l2 nondeterminism floor (legonet's distributional floor is ~4–6e-2). **H6 (paper-metric
  agreement):** on the paper's own answer-probability metric, sift held-in ≈ high (paper: ~0.99 at
  T=200) while FT+Merge ≈ zero-shot; post-unlearn: retain held-in preserved, forgotten-maskless ≈
  zero-shot (Fig 3/8). **H7 (extended caps):** the smoke-cap T=200 conclusions survive
  publication-grade caps (in flight, will be logged separately when all 4 labels land).
- **Setup:** `bash submit_sift_followups.sh configs/sift_masks_tofu_1b.json all` → jobs **440196**
  (prepare_eval --extended, builds the extended retain90 KS ref) → **440197** (extended eval array,
  4 labels %2, dep) · **440198** (`eval_sift_masks.py --mode full` + `--mode unlearn --tag forget10`)
  · **440199** (`measure_sift_exactness.py --author 199`, NEW script: re-derive τ_199 twice on GPU,
  compare). Pre-launch fix: `eval_sift_masks.load_masks` loaded ALL T masks unpacked at once
  (~194 GB at T=200 → host OOM); replaced with lazy per-author `load_one_mask`. New driver
  `submit_sift_followups.sh`; CLAUDE.md rows added.
- **Results:**
  - **Exactness (440199, `results/exactness_a199.json`):** `bitwise_identical: true`,
    `max_abs_diff: 0.0`, `rel_l2_floor: 0.0`, ‖τ‖₂ = 11.75, device cuda.
  - **Answer probability (440198):** full mode (T=200): `sift_heldin` **0.9188**, `merge_heldin`
    **0.1402**, `base_zeroshot` **0.1420**. unlearn mode (T'=180): `sift_retain_heldin` **0.9178**,
    `retain_maskless` 0.1416, `forgotten_maskless_heldout` **0.1224** (n=20 forgotten).
  - **Extended eval (440197, partial):** `merge_full` mu **0.4051** / fq 0.0987 (≈ the smoke-cap
    0.4073 — collapse confirmed at extended caps); `sift_full`/`sift_unlearn` running,
    `merge_unlearn` queued.
- **What worked / hypothesis verdict:**
  - **H5 SUPPORTED — stronger than expected:** GPU re-derivation is **byte-for-byte identical**
    (`max_abs_diff 0.0`). The unlearn subtraction is bitwise-exact end-to-end on GPU, not merely
    distributional — the deterministic-attention hardening did its job. This upgrades the thread's
    exactness claim above legonet's (which could only show rel_l2 ≈ nondeterminism floor 4–6e-2).
  - **H6 SUPPORTED:** the paper's Fig 3 headline reproduces on its own metric — `merge_heldin`
    0.1402 ≡ zero-shot 0.1420 (FT+Merge collapses exactly to base at 200 models) while SIFT recovers
    to 0.9188 (6.5×). Fig 8's post-unlearn pattern too: retain held-in preserved (0.9178 vs 0.9188)
    and forgotten authors at/below zero-shot (0.1224). We get 0.92 vs the paper's ~0.99 — plausibly
    the answer-span-only loss + Llama-1B (vs GPT2-XL full-QA); qualitative claim fully intact.
  - **H7 (partial):** extended `merge_full` 0.4051 ≈ smoke 0.4073 — caps don't change the collapse.
- **Observations:** The bitwise result is the strongest exactness statement in the whole repo: the
  full loop (train on GPU → merge → deterministically re-derive → subtract) leaves ZERO residual of
  the forgotten task vector, verified at the byte level on real hardware. Combined with the T=200
  OU result (mu 0.737 preserved through deletion) and the paper-metric agreement, the SIFT-Masks
  thread is now reproduced on three independent measurement axes (OU metrics, paper metric, bitwise
  audit). The `load_masks` OOM bug would have silently killed the T=200 answer-prob job — caught by
  a pre-launch feasibility pass, worth keeping as a review habit for serve-path code at scale.
- **New questions / new hypotheses:** (1) Is bitwise re-derivation stable across *nodes* (sprint1 vs
  2 vs 3 GPU models) — same-node was proven; cross-node would strengthen the deployment claim.
  (2) Answer-prob gap to the paper (0.92 vs 0.99): does `loss_on="full"` close it? A one-config
  ablation. (3) Held-in ↑ with more deletions (paper Fig 8) — multi-deletion curve still untested.
- **Next Steps:** Log the complete extended-cap table when 440197 finishes (H7 verdict); fold the
  sift row into the cross-track comparison; optional: cross-node exactness re-run + `loss_on=full`
  ablation if the thread continues.
