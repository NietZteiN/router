### Target Date: 2026-07-22 (pre-registration: D2a/D2b paper-conformance fixes + the H-promo K=200 retrain)

Follows [2026-07-22_rougeL-recall-correction.md](2026-07-22_rougeL-recall-correction.md), which
located the K=200 recall gap as K-scaling (K=20 ROUGE-L 0.957 ≈ paper; K=200 best 0.740, tail
197/200 vs paper 31/200) and fingered the promotion-term deviation. This entry pins the two
paper-conformance code fixes and pre-registers the retrain that tests whether they close the gap.
Bars frozen BEFORE the job.

- **Code fixes landed (both align to the MUSR paper; CPU suite 77 passed, 1 skipped):**
  - **D2a — promotion on all own tokens.** Renamed `BankState.question_mask` → `own_token_mask`;
    `SepMlpTrainer.compute_loss` now passes the FULL attention mask (all real tokens,
    question+answer) as the promotion target instead of `(labels==IGNORE) & attn` (question-only).
    Paper §3.2: promotion keeps ≥1 own unit active on "source k's own tokens." Rationale: at
    generation time the model produces the ANSWER; question-only promotion left the adapter free
    to be suppressed on answer tokens → dead-during-generation → the K=200 tail. Detector-init
    still pools question tokens (separate concern, unchanged). Grad-isolation + spec-loss gates
    updated to the new kwarg and pass.
  - **D2b — paper-exact zero-`W_down` deletion mode.** New `AuthorBank.zero_wdown_authors` zeros
    only the dropped authors' W_down columns in place (fixed shape; ad_k = W_down·act ⇒ exactly 0),
    readable back from stored params. `apply_droplist_file(..., mode="remove"|"zero_wdown")`,
    default "remove" (unchanged pipeline behavior). New gate pins
    `zero_wdown ≡ mask ≡ bake ≡ remove` at the bank forward + a structural check (zeros only
    W_down, gate/up/bias + survivors intact).
  - File shas (first 16): bank_layer `6c850bce74767220` · train_sepmlp `d9b03ed0e9f1faef` ·
    sepmlp_model `bffc349fe70f8fff` · measure_recall `57716272b83630dd` ·
    submit_sepmlp `d38ac1897fa5b9a4`.
- **Hypotheses / what we're testing:**
  - **H-promo (make-or-break for conformance):** the D2a fix closes the K-scaling recall gap.
    CONFIRM: K=200 per-source ROUGE-L recall ≥ 0.90 (toward paper 0.966) AND tail (#<0.95) ≤ 80
    (toward paper's 31/200; ≫ improvement on the current 197/200). REFUTE: recall ≤ 0.78 (no
    meaningful lift over the pre-fix 0.740) — then the gap is not (only) promotion; escalate to an
    epochs/lr sweep.
    ADJUDICATE zone [0.78, 0.90): promotion helps but is not sufficient → add the epochs lever.
  - Secondary (no bar): held-out recall should stay ≈ base (0.31–0.34, isolation preserved);
    on/off output-norm selectivity stays SELECTIVE (the mechanism diagnostic).
- **Setup (frozen):** train `configs/sepmlp_1b_k200_promofix.json` (sha `54af0e622c6c2929`) —
  Llama-3.2-1B-Instruct, K=200, **lr 5e-4** (the K=20 ROUGE-L winner), 15 epochs, bs8×ga4
  (effective 32), width 32, all 16 layers, w_h/w_out/w_p = 10/50/1, margin 2, promo_delta 0.1,
  seed 42, detector-init questions, negatives 2000 Alpaca + real_authors (never holdout10). Fresh
  output dir `sepmlp_1b_k200_promofix_s42`. GPU smoke first (`configs/smoke_promofix.json`
  sha `c6ee422bfb8101a4`, 2 authors, 5 steps) → full train → `recall` verb (per-source ROUGE-L +
  tail + named/name-free + held-out). The ONLY difference vs the earlier lr5e-4 K=200 run
  (G3-FAIL, deleted) is the D2a promotion fix — a clean A/B on the fix.
- **Results / verdict / observations / next steps:** pending (pre-registration).
