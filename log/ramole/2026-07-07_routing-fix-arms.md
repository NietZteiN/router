### Target Date: 2026-07-07 (§9-D fix arms — abstain/OOD route + SEUF anchor loss)

- **Hypotheses / what we're testing:** The two fix arms the §9-D audit
  ([2026-07-06_routing-audit-results.md](2026-07-06_routing-audit-results.md)) left as coded-design.
  The audit found that dropping an expert sends forget orphans to a surviving sibling at top-1
  similarity-ratio **0.980** (a fallback leak) and shifts 72.7% of retain routing. Both fixes aim to
  send low-confidence orphans to the scaffold/base instead. Pre-registered BEFORE running.
  - **H-abstain (C1):** an OOD-threshold route — abstain to base when the (post-drop) top-1
    embedding similarity is below a **retain-calibrated** τ — converts orphan→sibling into
    orphan→base without materially abstaining on retain queries. CONFIRM: at a τ set to the p-th
    retain-sim percentile (p∈{1,5,10}), orphan→abstain rate ≥ 0.90 while retain false-abstain rate
    ≈ p% (≤ ~10%). REFUTE: orphan-abstain stays low (the sibling sims are too high to threshold
    apart from retain) or retain false-abstain blows up (no separating τ). The τ must be calibrated
    on retain questions only — no forget data touches the threshold (§5.2 centroid-leak discipline).
  - **H-anchor (C2):** training the RouterLoRA with a SEUF anchor penalty `L_anchor = ‖α − a‖²`
    (a = indicator of the target/ideal experts) sharpens routing to each query's own expert, so a
    drop removes exactly that query's handling instead of spilling to a sibling. CONFIRM: the
    anchor-trained router lowers alpha entropy (H_norm 0.814 → < 0.75) and raises ideal-expert mass
    (0.302 → > 0.4); embed-route unlearn fq rises vs 0.484; key-route fq (0.890) and mu (within
    noise) unchanged. REFUTE: fq drops or mu falls > 0.01, or entropy/ideal-mass don't move
    (anchor weight too small / conflicts with CE). anchor_weight = 0 must reproduce the baseline
    router bit-identically (a determinism guard).
- **Setup (C1):** routing-only `abstain` analysis added to `routing_audit_tofu.py`
  (`abstain_analysis`: calibrate τ on RETAIN top-1 similarity over the full index — no forget data
  touches the threshold — then abstain forget orphans whose masked top-1 sim < τ). CPU gate
  extended (`test_routing_audit_tofu.py`, separable synthetic case: orphan→base 1.0, retain
  false-abstain 0.0 — GREEN). Run: `routing_audit_tofu.py --config configs/ramole_tofu_1b_basepin.json
  --tag forget10 --policies stale dropped abstain --device cuda --out
  results/routing_audit_forget10_abstain.json` (base-pinned off-the-shelf instructor-xl, the
  fq-0.484 arm; pool `Llama-3.2-1B-Instruct_legonet_n32_k3`, tag forget10, seed 42; routing-only,
  no SLURM needed). `routing_audit_tofu.py` sha reflects the abstain addition.
- **Results (C1):** orphan masked-top1 sim distribution mean **0.858** (p10–p90 = 0.832–0.882)
  vs retain top1 sim mean **0.877** (p10–p90 = 0.847–0.907) — they overlap almost entirely
  (means 0.02 apart). Abstain tradeoff (τ swept):

  | τ target | orphan→base | retain false-abstain |
  |---|---|---|
  | p1 retain (τ 0.821) | 0.043 | 0.010 |
  | p5 retain (τ 0.838) | 0.152 | 0.050 |
  | p10 retain (τ 0.847) | 0.265 | 0.100 |
  | 90% orphan-abstain (τ 0.882) | 0.900 | **0.580** |
  | 99% orphan-abstain (τ 0.900) | 0.993 | 0.834 |

- **What worked / hypothesis verdict:** **H-abstain REFUTED.** No τ seals the drop-an-expert
  fallback leak at acceptable cost: at a 5% retain false-abstain budget only **15%** of orphans
  abstain to base; reaching 90% orphan-abstain costs a **58%** retain false-abstain rate. The
  reason is mechanistic and was already visible in the 07-06 audit — the surviving sibling matches
  an orphan query at top-1 similarity-ratio 0.980, so the orphan-confidence distribution sits
  *inside* the retain-confidence distribution and a similarity threshold cannot separate "your
  expert was deleted" from "you are a normal retain query."
- **Observations:** this is a **stronger** result than a working fix — it is a near-impossibility
  argument for confidence-based abstention under embedding routing, and it directly motivates the
  paper's design choice: use a **hard identity router** (author-key) that *knows* the entity was
  deleted (orphans → scaffold, zero ambiguity, from `2026-07-06_routing-audit-results.md` H4), or
  delete by retrain-in-place. Silent-failure checks clean: τ calibrated on retain only (no forget
  leakage into the threshold); the separable synthetic CPU case confirms the analysis fires
  correctly when the distributions *are* separable (they just aren't, here).
- **C2 (SEUF anchor loss) — scoped out, with reason.** The anchor `L_anchor = ‖α − a‖²` sharpens
  the **RouterLoRA composition gate** (the per-layer α weighting the *retrieved* experts). But C1
  localizes the drop-leak to the **retrieval** stage (instructor-xl cosine returns the sibling as a
  near-perfect match); no composition-gate sharpening changes which experts are retrieved, so the
  anchor cannot close *this* leak. Its legitimate benefit is confined to the **key-route** arm
  (where retrieval already returns the true expert, so a peaked α means dropping that expert leaves
  near-zero composition weight ≈ abstain) — a secondary sharpening result, not a fix for the
  embed-route leak. Building it needs gradient-carrying α capture (`capture_grad` in
  `router_lora.py`) **and** author-labeled retain training data (to form the target indicator `a`)
  + a GPU router retrain — real plumbing whose payoff C1 undercuts, so deferred rather than run on a
  contended cluster.
- **New questions / new hypotheses:** does the anchor-sharpened **key-route** RouterLoRA give a
  cleaner drop (near-abstain composition) than the 1/k average — i.e. is the anchor's value a
  *composition* improvement on the already-clean key arm rather than a *retrieval* fix? (The only
  remaining testable anchor claim.) Is there any learned OOD detector (vs a fixed similarity τ) that
  separates orphan-from-deleted-expert vs retain — or is the overlap fundamental to shared-topic
  authors?
- **Next Steps:** fold the C1 tradeoff into `reports/ROUTING_AUDIT_REPORT_2026-07-06.md` (fix-arm
  section); if the anchor is pursued, scope it to the key-route composition claim only, with the
  author-labeled-data + `capture_grad` plumbing built and CPU-gated first.
