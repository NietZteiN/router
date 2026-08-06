# SIFT-Masks — sign-fixed full-FT task vectors for exact unlearning at scale

**Status:** complete (current scope) · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 6 (2026-06-29 → 2026-07-06)

A faithful **full-finetuning** (not LoRA) build of SIFT-Masks (Kuo et al. 2025,
arXiv:2504.04626, *"Exact Unlearning of Finetuning Data via Model Merging at Scale"*) on
TOFU. One task per author (T=200), the regime where plain FT+Merge collapses to zero-shot.
A single global ±1 sign vector `v` (seed-fixed, drawn before training, shared across tasks)
constrains each task's full-FT so `τ_c⊙v ≥ 0` (project after each Adam step); the free mask
`m_c = 1{τ_c≠0}` depends only on local data + `v`, never on the merged model — that
independence is what keeps deletion exact and O(1). Merge = streaming **sum** `τ̄ = Σ τ_c`
(store only `τ̄` + per-task bit masks). Serve task t: `θ0 + (τ̄⊙m_t)/T`. Unlearn = re-derive
`τ_u` deterministically and subtract. This is the only non-LoRA track; it slots into the
repo's OU metrics (`forget_quality`/`model_utility`) via a `SiftMasksModel` serving wrapper,
so it's directly comparable to sisa_lora / legonet / additive.

## What worked
- **Algorithm verified against the paper PDF** (Alg 1 + App B + Fig 3/4/8): ±1 sign vector
  (resolving the paper's `1{rand>0.5}` pseudocode vs. "share the same sign as v" prose);
  merge = sum, serve = `(τ̄⊙m_t)/|T|`; unlearn = subtract a re-derived task vector.
- **CPU exactness gate green** (`test_sift_masks.py`, tiny random GPT2, 6/6): ±1 &
  seed-determinism; projection invariant (`τ⊙v≥0`, `mask==τ≠0`); **byte-identical** `τ_u`
  re-derivation (`torch.equal`); exact unlearning `(Στ)−τ_u ≈ Σ_retain` (allclose, max gap
  **9.3e-10** — fp-noise scale); serve identity bit-exact; mask pack/unpack bit-exact (1/8 bytes).
- **`SiftMasksModel` serving wrapper green on CPU integration**: oracle route via
  `legonet_tofu.build_q2author` → applies `θ0+(τ̄⊙m_a)/T` in place **bit-exactly**, cached by
  author (no re-apply for consecutive same-author queries); OOD/forgotten → base `θ0`;
  FT+Merge no-mask baseline serves `θ0+τ̄/T`; B>1 mixed-author batches return CausalLMOutput.
- **GPU smoke green on real Llama-3.2-1B (T=5)**: build (0.97 B trainable, 36.7 s) → unlearn →
  `eval_tofu --sift_masks_config` both labels produced valid OU metrics (`sift_full` mu 0.4694 /
  fq 0.3929; `sift_unlearn` mu 0.4712 / fq 0.3929). fq 0.3929 ≡ the repo's base/SISA-1B fq 0.393
  (correct at T=5: untrained forget10 → base θ0). The `SiftMasksModel`+`eval_tofu` headline path works.
- **T=200 science (2026-07-02) — clean paper reproduction on OU metrics** (build 102 min, 0.97 B trainable):
  - **Merging collapses at scale; the mask recovers it:** `merge_full` mu **0.4073** (≈base) vs
    `sift_full` mu **0.7370** (+0.33, matching the joint-FT / k=1 upper bound 0.74; retain_rouge
    0.777 vs 0.393; forget_ppl 5.62 vs 37.9).
  - **Exact unlearning works & is cheap:** `sift_unlearn` raises forget_quality **0.135→0.3929**
    (to the retain-oracle level) while **preserving** mu (0.7370→**0.7377**) — delete forget10 by
    subtracting 20 re-derived task vectors, no retain-utility cost.
  - **The mask is the active ingredient:** `merge_unlearn` forgets identically (fq 0.3929) but stays
    collapsed (mu 0.4082). sift_unlearn≡merge_unlearn forget metrics (both serve deleted authors from
    base θ0) is a direct exactness signature.
  - **Best of all tracks:** sift mu 0.737 > legonet 0.637 > sisa dare_ties 0.48–0.59; ties additive
    coarse-core 0.7537.
- **GPU unlearn is BITWISE-exact (2026-07-02):** re-deriving τ_199 twice on CUDA (deterministic
  math/eager attention) → `bitwise_identical: true`, `max_abs_diff 0.0` — zero residual of the
  forgotten task vector at the byte level. Strictly stronger than legonet's distributional
  exactness (floor ~4–6e-2). `measure_sift_exactness.py`.
- **Paper's own metric reproduced (2026-07-02):** answer probability at T=200 — sift held-in
  **0.9188** vs FT+Merge **0.1402** ≡ zero-shot 0.1420 (Fig 3 collapse, exactly); post-unlearn
  retain held-in preserved (0.9178), forgotten-maskless 0.1224 ≈ zero-shot (Fig 8). 0.92 vs the
  paper's ~0.99 — plausibly the answer-span loss + Llama-1B (vs GPT2-XL full-QA).

## What didn't / open problems
- **H8 RESOLVED (2026-07-06): direction ✓, magnitude ✗.** The paper-faithful Fig-8 serving rule
  (forgotten → maskless merged `θ0+τ̄_tag/T′`) was applied and re-evaled: mu bit-unchanged
  (0.7370/0.4055), fq rose ~11× (0.0045 → **0.0505**) — but NOT to legonet's 0.89. Residual cause
  is merge dilution itself: at T′=180 the maskless merge is collapsed (mu≈base), so its style on
  forget questions still differs from the retain-finetuned oracle at n=120 KS power. **Takeaway:
  extended-cap OU fq measures oracle-style match, not leakage, once deletion is exact** — leakage
  is independently ruled out (bitwise audit + answer-prob 0.122 ≈ zero-shot). Compare extended fq
  across tracks only within the same forget-serving style. Pre-H8 JSONs: `*.pre_h8.json`.
- **Extended-cap eval DONE** — utility conclusions confirmed at publication caps:
  sift 0.7364/0.7370 vs merge 0.4051/0.4055 (see 2026-07-02_extended-caps.md).
- **Answer-prob gap to paper** (0.92 vs ~0.99): candidate cause = answer-span-only loss (paper
  detail unpublished) and/or Llama-1B vs GPT2-XL; `loss_on=full` ablation would settle it.
- **Bitwise exactness proven same-node only** — cross-node (sprint1/2/3) re-derivation untested.
- (Resolved 07-02) GPU exactness → **bitwise-identical**; paper metric → reproduced. (Resolved
  during launch) `eval_sift_masks.load_masks` loaded all 200 masks at once (~194 GB → OOM) —
  now lazy per-author.
- (Resolved) two submit-script bugs found+fixed during the T=5 smoke (relative-target KS symlink →
  forget_quality NaN; `ln -s` race → `sift_unlearn` abort) — both fixed via `cp -f`.
- **fp non-associativity**: `(Στ)−τ_u` is exact at the deterministic/algebraic level but not
  bit-equal to a from-scratch retain-sum (documented; would need a fixed-order reduction).
- **Answer-span loss masking** for TOFU is an inference (paper's other datasets predict only
  the label span; canonical TOFU finetunes full Q+A) — configurable, default answer-only.

## Open ideas / next steps
- Run the GPU smoke (T=5) then the full T=200 build → `eval_tofu` extended → `collect`.
- Compare `sift_full`/`sift_unlearn` to `merge_full`/`merge_unlearn` (expose the at-scale
  collapse the mask fixes) and to on-disk `additive_mean`/`dare_ties`/`legonet` 1B numbers.
- Cross-check with the paper's own metric via `eval_sift_masks.py` (held-in answer probability
  should approach the paper's ~0.99 at T=200; merge should collapse).

## Entries (chronological)
- [2026-06-29 — implementation + CPU exactness](2026-06-29_implementation.md) — full pipeline written; 6/6 CPU tests + wrapper integration green; GPU run pending.
- [2026-06-29 — GPU smoke green](2026-06-29_gpu-smoke.md) — T=5 build/unlearn/eval on real Llama-3.2-1B; headline `eval_tofu` integration works (mu 0.47, fq 0.393≡base); two submit-script bugs fixed; determinism hardening added.
- [2026-07-02 — T=200 science](2026-07-02_t200-results.md) — mask recovers utility (sift_full mu 0.737 vs merge_full 0.407); exact unlearn raises fq 0.135→0.393 while preserving mu 0.738; H1–H4 all supported.
- [2026-07-02 — follow-ups: bitwise exactness + paper metric](2026-07-02_followups-exactness-ansprob.md) — GPU τ_u re-derivation byte-identical (max|Δ|=0.0); answer-prob: sift 0.919 vs merge 0.140 ≡ zero-shot, post-unlearn retain preserved / forgotten ≈ zero-shot; extended eval in flight.
- [2026-07-02 — extended caps](2026-07-02_extended-caps.md) — utility conclusions publication-grade (sift 0.7364/0.7370 vs merge 0.4051/0.4055); extended fq exposes the forgotten-serving rule (base θ0 vs the paper's maskless-merged) → H8 one-line fix proposed.
- [2026-07-06 — H8 serving rule](2026-07-06_h8-serving-rule.md) — Fig-8 rule applied (forgotten → maskless merged): mu bit-unchanged, fq 0.0045→0.0505 (direction ✓, legonet-class magnitude ✗ — residual = merge dilution). Closing insight: extended fq measures oracle-style match, not leakage. Thread complete.
