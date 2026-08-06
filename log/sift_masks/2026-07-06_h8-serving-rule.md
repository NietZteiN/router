### Target Date: 2026-07-06 (H8: paper-faithful forgotten-serving — direction confirmed, magnitude refuted)
- **Hypotheses / what we're testing:** **H8** (raised in [2026-07-02_extended-caps.md](2026-07-02_extended-caps.md)):
  switching `SiftMasksModel._apply` to the paper's Fig-8 held-out rule — forgotten authors get the
  **maskless merged model** `θ0 + τ̄_tag/T′` instead of base θ0 — (a) leaves model_utility unchanged
  (forget rows don't enter mu), and (b) raises extended-cap forget_quality from 0.0045 into the
  legonet-class range (~0.5–0.9), making extended fq cross-track comparable.
- **Setup:** One-branch change in `sift_masks_model.py::_apply` (OOD → base θ0 unchanged; retained →
  masked unchanged; forgotten/baseline → `serve_merged_`), docstring + CLAUDE.md updated in the same
  change. CPU verification before submitting: 4-branch bit-exact test (forgotten → `θ0+τ̄/T′`;
  retained → masked; OOD → base; baseline retained+forgotten → merged) — all pass. Pre-H8 extended
  JSONs preserved as `*.pre_h8.json`. Re-eval: `SIFT_LABELS="sift_unlearn merge_unlearn"
  bash submit_sift_followups.sh configs/sift_masks_tofu_1b.json extended` → jobs **440427** (KS-prep,
  self-skip guard — ref existed) → **440428** (2 labels, extended caps). Driver gained the
  `SIFT_LABELS` override + prep skip guard.
- **Results:**
  | label | mu pre→post | fq pre→post | forget_tr pre→post |
  |---|---|---|---|
  | sift_unlearn  | 0.7370 → **0.7370** | 0.0045 → **0.0505** | 0.7536 → 0.6948 |
  | merge_unlearn | 0.4055 → **0.4055** | 0.0045 → **0.0505** | 0.7536 → 0.6948 |
  (sift_unlearn ≡ merge_unlearn on all forget metrics — expected: both now serve the same maskless
  merged model for forget queries.)
- **What worked / hypothesis verdict:**
  - **H8(a) SUPPORTED exactly:** mu bit-unchanged (0.7370 / 0.4055) — the rule change touches only
    forget-row serving, and mu has no forget component.
  - **H8(b) REFUTED in magnitude, supported in direction:** fq rose ~11× (0.0045 → 0.0505) but
    nowhere near legonet's 0.89. Cause is the thread's own central finding: at T′=180 the maskless
    merged model is **collapsed** (mu 0.4055 ≈ base) — its style on forget questions is still nearly
    base-like, so the n=120 KS still separates it from the retain-finetuned oracle. Legonet's 0.89
    comes from serving forget queries through *concentrated* retain-trained adapters (top-k of 32,
    style ≈ oracle). The residual gap is merge dilution, not the serving rule.
- **Observations:** Closing insight for the thread: **extended-cap OU forget_quality measures
  style-match to a retain-finetuned oracle, not leakage**, once deletion is exact. SIFT-Masks'
  post-deletion forget-serving model (maskless merge at scale) is base-like *by construction* (the
  collapse it exists to fix on retained tasks), so its extended fq stays low even though (i) deletion
  is bitwise-exact and (ii) the paper's own metric shows forgotten authors at zero-shot (0.122).
  Cross-track extended-fq comparisons between "serve retain-trained weights for forget queries"
  methods (legonet/routing) and "serve near-base weights" methods (SIFT, base-serving scaffolds)
  are therefore apples-to-oranges at high KS power; report the deletion audit + paper metric
  alongside. Smoke-cap fq (n=30) doesn't have the power to see this and matched base (0.393) —
  which is why the artifact only surfaced at extended caps.
- **New questions / new hypotheses:** (1) An oracle-style-matched forgotten-serving variant — e.g.
  serve forgotten queries with the retain90 *oracle-style* merged model at lower dilution (top-k
  neighbor masks, LegoNet-style) — would isolate how much fq is recoverable without touching
  exactness; academic, since leakage is already ruled out. (2) Does the paper's held-out accuracy
  ordering (Fig 8: held-out ↓ toward FT+Merge as deletions grow) reproduce with this rule across
  1→180 deletions? (multi-deletion curve, still open from 07-02.)
- **Next Steps:** Thread complete for the current scope: paper reproduced on three axes + extended
  caps + paper-faithful serving. Remaining open items are optional (multi-deletion curve,
  `loss_on=full` ablation, cross-node bitwise check). No jobs in flight.
