### Target Date: 2026-06-29 (SIFT-Masks implementation + CPU exactness gate)
- **Goal / Hypothesis:** Stand up a faithful, full-FT build of SIFT-Masks (Kuo et al. 2025,
  arXiv:2504.04626) on TOFU as a new comparison thread, and prove the exactness primitives on
  CPU before any GPU run. Core question the method answers: can a sign-fixed task vector + a
  local mask recover per-task utility at T=200 (where FT+Merge collapses) while keeping
  deletion an O(1) subtraction? Decisions locked: full-FT (not LoRA — the per-step sign
  projection is per-parameter and can't be done exactly in low-rank), per-author T=200,
  base `meta-llama/Llama-3.2-1B-Instruct` (comparable to the repo's other 1B tracks).
- **Setup:** New code in [`tofu_sisa_lora/`](../../tofu_sisa_lora/) (no git repo → sha256[:12]
  provenance):
  - `sift_masks.py` `0bfe81ca24bf` — core lib (model-agnostic via `frozen_substr`):
    `make_sign_vector` (±1, seeded), `sift_one_task` (project AFTER `opt.step()`; Adam over the
    **model's** params so `zero_grad` clears the right grads), `merge_add_/merge_sub_`,
    `serve_task_/serve_merged_/serve_base_`, `pack_mask/unpack_mask`.
  - `sift_masks_data.py` `7d769a6925bb` — TOFU loader + `Question: {q}\nAnswer: {a}` format +
    answer-span loss masking (`loss_on` flag).
  - `train_sift_masks.py` `6ac33835b462` — `build` (stream `τ̄` + masks) / `unlearn --tag`
    (deterministic re-derive + subtract). θ0/sign on GPU, τ̄ on CPU, masks streamed to disk.
  - `sift_masks_model.py` `da29f8e5558e` — `SiftMasksModel` OU-metric serving wrapper (mirrors
    `LegoNetRoutedModel`) + `load_sift_eval_model`.
  - `eval_sift_masks.py` `14472d096068` — secondary answer-probability eval (the paper's metric).
  - `test_sift_masks.py` `d03591a8f714` — CPU exactness gate.
  - `configs/sift_masks_tofu_1b.json`, `submit_sift_masks_tofu.sh`; `eval_tofu.py` hook
    (`--sift_masks_config`/`--sift_unlearn_tag`, labels `sift_full|sift_unlearn|merge_full|merge_unlearn`);
    repo `CLAUDE.md` updated (Comparison Tracks + suffix row + test-variants row).
  - Algorithm cross-checked against the source PDF (Alg 1 p.13, App B p.14–16, Fig 3/4/8).
  - Commands run (CPU, login node — micro only): `python test_sift_masks.py`; a wrapper
    integration script (tiny GPT2, synthetic authors). No SLURM job submitted yet.
- **Results:**
  - `test_sift_masks.py` **6/6 PASS**: sign vector ±1 & seed-deterministic; projection
    invariant (`τ⊙v≥0` everywhere, `mask==(τ≠0)`); `τ_u` re-derivation **byte-identical**
    (`torch.equal`); exact unlearning `(Στ_{0,1,2})−τ_1 ≈ Σ_{0,2}` allclose, **max |gap| =
    9.31e-10** (fp-noise scale, not bit-equal — non-associativity, as flagged); serve identity
    `θ==θ0+(τ̄⊙m_t)/T` **bit-exact** (atol 0); mask pack/unpack bit-exact (25472 bits → 3184 B).
  - `SiftMasksModel` integration **5/5 PASS**: author-1 query → served `θ0+(τ̄⊙m_1)/3`
    bit-exact; consecutive same-author query reuses cached weights; OOD query → base `θ0`
    restored; baseline serves `θ0+τ̄/3` (no mask); B>1 mixed-author forward returns
    `CausalLMOutput` (loss 10.38 on the random fixture).
  - `eval_tofu.py --help` exposes the new flags; `py_compile` clean on all touched files.
- **Observations:** The exactness primitive that matters — deterministic byte-identical
  re-derivation of a task vector — holds exactly; the subtraction identity is exact only up to
  fp non-associativity (ULP-scale), which is the honest statement of "exact unlearning" for a
  running-sum-minus-term design. Two corrections from the paper review proved load-bearing:
  (1) projecting AFTER `opt.step()` keeps the stored `τ` consistent with its mask; (2) running
  Adam over the model's params (not an external `τ` + grad-scatter) avoids a silent
  grad-accumulation trap that would have broken determinism. The serving wrapper's author-keyed
  weight caching means a full TOFU eval applies the masked delta ~once per author, not per query.
- **Next Steps:** (1) GPU smoke with a T=5 config (build → unlearn → `eval_tofu --smoke
  sift_full`) to confirm in-range `model_utility`/finite `forget_quality` and that `sift_unlearn`
  raises `forget_quality` vs `sift_full` while `merge_full` shows the FT+Merge collapse.
  (2) Full T=200 build via `submit_sift_masks_tofu.sh configs/sift_masks_tofu_1b.json all`,
  then extended eval + `collect_results.py`. (3) Compare to `additive_mean`/`dare_ties`/`legonet`
  1B; cross-check the paper's answer-probability via `eval_sift_masks.py`.
