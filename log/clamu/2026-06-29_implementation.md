### Target Date: 2026-06-29 (ClAMU arm — implementation + CPU exactness gate)
- **Goal / Hypothesis:** Stand up ClAMU (Kuo et al., *Exact Unlearning of Finetuning Data via
  Model Merging at Scale*, ICLR-2025; [`papers/ClAMU.pdf`](../../papers/ClAMU.pdf)) on TOFU by
  **extending the [`sift_masks`](../sift_masks/README.md) arm**, then evaluate it under the OU
  `model_utility` (the paper reports only answer-probability). Core question for the runs:
  does cluster-level **optimized** masking match the route+scaffold headline (mu 0.664), or
  crater like the other merges once all 9 `model_utility` components are scored?
- **Setup:** New files in [`tofu_sisa_lora/`](../../tofu_sisa_lora/): `clamu.py` (STE mask
  optimizer `optimize_mask_ste`, EMR/TALL baseline masks, feature/random clustering),
  `clamu_model.py` (`ClamuModel` per-query oracle author→cluster serve), `train_clamu.py`
  (`setup`/`build`/`localize`/`unlearn`), `test_clamu.py` (CPU gate), `submit_clamu_tofu.sh`
  (7-stage driver), `configs/clamu_tofu_1b.json` (Llama-3.2-1B, K=16) +
  `configs/clamu_tofu_smoke.json` (TinyLlama micro). Modified: `sift_masks.py` (added a
  backward-compatible `use_sign_constraint=True` kwarg gating the sign projection in
  `sift_one_task` — ClAMU calls it `False`); `eval_tofu.py` (`--clamu_config`/`--clamu_unlearn_tag`
  + dispatch branch mirroring the SIFT branch); `CLAUDE.md` (ClAMU Comparison-Tracks subsection +
  `{slug}_clamu` suffix). Reuses `sift_masks` merge/serve/pack, `sift_masks_data` answer-span
  loaders, `legonet_tofu` clustering + `build_q2author`, the SISA retain90 KS reference. Commands:
  `python test_clamu.py`; `python test_sift_masks.py`; `STUB=1 bash submit_clamu_tofu.sh configs/clamu_tofu_smoke.json all`.
  No git commit yet; seed 42 throughout.
- **Results:**
  - `test_clamu.py` **6/6 green**: STE forward `1{s>0}` + backward `σ'(s)`; cluster determinism
    (feature + random, full partition); **exact unlearning (no-sign)** `(Στ)−τ_1 ≈ Σ_retain`
    max |gap| **9.31e-10** (fp-noise scale); **STE mask optimization reduces CE 4.2113 → 4.0738**
    and is seed-deterministic; serve identity bit-exact; mask pack/unpack bit-exact.
  - `test_sift_masks.py` **6/6 still green** (the `use_sign_constraint` gate did not regress SIFT;
    its exact-unlearn gap is the same 9.31e-10).
  - `py_compile` clean on all new/modified files; `clamu`/`clamu_model`/`train_clamu` import OK;
    `eval_tofu.py --help` shows `--clamu_config`; `bash -n submit_clamu_tofu.sh` OK.
  - `STUB=1` preview emits all 7 stages with correct commands and the `_unlearn`→`--clamu_unlearn_tag`
    case; eval array `0-2%2` for the lean label set.
- **Observations:** ClAMU really is the SIFT spine + two swaps — the only `sift_masks.py` change
  is one gated kwarg, and everything exactness-critical (deterministic `τ_u`, streaming sum,
  subtract) is shared, so the same 9.31e-10 fp-noise floor and the same exactness caveat carry
  over. The genuinely new code is the STE localizer and the cluster-level serve; the CPU gate
  confirms the STE loop actually descends and is reproducible. EMR/TALL are available as cheap
  baselines from the per-cluster sums (full model). Scale choice: serve `/T` so the optimized mask
  absorbs scale → ClAMU stays directly comparable to the existing `merge_*`/`sift_*` labels.
- **Next Steps:** (1) TinyLlama micro-smoke (`configs/clamu_tofu_smoke.json`) go/no-go —
  `clamu_full` beats `merge_full` (no-mask) and exactness holds. (2) 1B build → localize →
  `eval_tofu` (K=16) headline under `model_utility`: Global/EMR/TALL/ClAMU ladder + exact forget10
  `clamu_unlearn` (forget_quality ≈ retain90 oracle) + cost ledger. (3) then-expand (gated):
  clustering ablation, storage–utility dial, heterogeneity probe (reuse `subspace_overlap.py` /
  `submit_iso_merged.sh`).
