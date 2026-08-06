### Target Date: 2026-07-02 (extended caps: utility conclusions hold; fq exposes the forgotten-serving rule)
- **Hypotheses / what we're testing:** **H7:** the smoke-cap T=200 conclusions survive
  publication-grade caps (ROUGE≤200, retain≤400, truth≤120; KS ref n=120 vs smoke n=30). Predict:
  same utility ordering (sift ≈ 0.74 ≫ merge ≈ 0.41, unlearn preserves); forget_quality expected to
  *shrink* everywhere (KS power grows ~4× in sample count) — verdict on unlearning quality must be
  read against extended-cap comparators, not smoke values.
- **Setup:** Job **440197** (dep on 440196 `prepare_eval.py --extended`, which built the extended
  retain90 KS ref, n=120). 4 labels × `eval_tofu --extended` on the T=200 artifacts. Comparators
  pulled from the on-disk 1B legonet extended results (same caps, same oracle construction).
- **Results:**
  | label | mu | fq | forget_tr | retain_rouge |
  |---|---|---|---|---|
  | sift_full     | **0.7364** | 0.0045 | 0.5679 | 0.801 |
  | merge_full    | 0.4051 | 0.0987 | 0.6928 | 0.363 |
  | sift_unlearn  | **0.7370** | 0.0045 | 0.7536 | 0.806 |
  | merge_unlearn | 0.4055 | 0.0045 | 0.7536 | 0.362 |
  Comparators (extended, 1B): legonet_full fq **0.0004** / legonet_unlearn fq **0.8904**
  (mu 0.4947/0.5011); routerkey_unlearn fq 0.8904 (mu 0.5070).
- **What worked / hypothesis verdict:**
  - **H7 (utility) SUPPORTED:** extended reproduces smoke within noise — sift_full 0.7364 (smoke
    0.7370), sift_unlearn 0.7370 (0.7377), merge 0.4051/0.4055 (0.4073/0.4082). The +0.33 mask
    recovery and deletion-preserves-utility conclusions are publication-grade. sift retain_rouge
    0.80 vs merge 0.36.
  - **H7 (forget_quality) NUANCED, as predicted in direction but the magnitude teaches something:**
    sift_unlearn fq collapsed 0.3929 → **0.0045**. This is NOT failed forgetting — deletion is
    bitwise-exact (07-02 audit) and sift_unlearn ≡ merge_unlearn on every forget metric (identical
    fq/forget_tr, both serve forget queries from base θ0). It is a **reference-model artifact**: the
    OU oracle for "never trained on forget" is a model *finetuned on retain90*, and at n=120 the KS
    test can now distinguish raw-base answers from retain-finetuned answers on forget questions
    (finetuning transfers TOFU answer style even to unseen authors). At n=30 (smoke) it couldn't
    (fq 0.393 ≡ base). Contrast legonet_unlearn fq **0.8904**: it serves forget queries through
    retain-trained adapters, whose style matches the oracle.
- **Observations:** This traces to one serving-rule choice in `SiftMasksModel._apply`
  (sift_masks_model.py:142): forgotten authors → **base θ0** (our documented design choice). The
  paper's Fig 8 held-out rule is different: *"To evaluate a task which has already been unlearned,
  SIFT-Masks applies the merged model without any mask"* = **θ0 + τ̄_tag/T′**. The maskless
  retain-sum contains zero forget data (exact — it is Σ over retain task vectors) but *is*
  retain-finetuned in style, so it should score far closer to the oracle (cf. legonet 0.89).
  Our own `eval_sift_masks.py` unlearn arm already uses the paper's rule (forgotten_maskless
  0.1224 ≈ zero-shot on answer-prob — no leakage there either). So the exactness claim is
  unaffected; the extended fq number is measuring "base θ0 ≠ retain-finetuned style", not leakage.
  Also noted: sift_full fq rounds to the same 0.0045 as the unlearn rows despite a different
  forget_tr (0.5679 vs 0.7536) — coincidence of rounding at very small p; sift_full's low fq is the
  *expected* "model knows forget10" signal (legonet_full 0.0004 likewise).
- **New questions / new hypotheses:** **H8:** switching the wrapper's forgotten-author serving to
  the paper's maskless-merged rule (`θ0 + τ̄_tag/T′`, a one-line change in `_apply`) raises
  sift_unlearn extended fq to the legonet-class range (~0.5–0.9) with mu unchanged (forget rows
  don't enter mu's retain/real/world components... they don't — mu is retain+real+world only, so mu
  strictly unchanged). Would make the OU fq comparable across tracks AND paper-faithful. Needs a
  2-label re-eval (sift_unlearn, merge_unlearn; ~2.5 h GPU each at extended caps).
- **Next Steps:** Propose the H8 serving fix + re-eval to the user (semantic change to the served
  model — not launched unilaterally). Then the thread is complete: three-axis reproduction (OU
  utility, paper metric, bitwise audit) + extended-cap confirmation.
