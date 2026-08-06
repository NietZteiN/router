### Target Date: 2026-07-06 (§9-D routing-audit results — base-pinned + dropped-expert)

- **Hypotheses / what we're testing:** Results for the §9-D audit pre-registered in
  [2026-07-03_routing-audit-9d.md](2026-07-03_routing-audit-9d.md) (H1–H6 there). Jobs from
  2026-07-03 (440480 basepin audit, 440481 ftdrop audit, 440482/440483 extended scaffold evals)
  completed. This entry records the numbers + verdicts; the design entry stays the pre-registration.
- **Setup:** Commands + script sha256 as in the 2026-07-03 entry (routing_audit_tofu.py 1388e317de7a
  base-pinned via configs/ramole_tofu_1b_basepin.json; eval_routed_scaffold.py extended on
  `_experts_scaf_k10`). Pool `Llama-3.2-1B-Instruct_legonet_n32_k3`; seed 42. Result JSONs:
  `results/routing_audit_forget10_basepin.json`, `..._ftdrop.json`,
  `_experts_scaf_k10/results/extended/routed_scaffold_strong{,_del9}.json`.
- **Results — the filled §9-D table:**

  | §9-D row (router) | deletion op | orphan routing | leak: fq | retain Δmu | retain shift | verdict |
  |---|---|---|---|---|---|---|
  | author-key (hard) | drop expert (scaffold) | orphans → base+scaffold, P=1.0 | del9 fq 0.0241 (= base floor) | **0.7509→0.7509 = 0** | 0 | ✅ clean |
  | key + router (RAMoLE key) | retrain-in-place | orig_top1 1.000 | 0.890 | +0.013 | 0.000 (asserted) | ✅ clean |
  | encoder cluster-ID (base-pinned / off-the-shelf) | retrain-in-place | stale sibling_top1 **0.185** | ramole 0.484 | +0.010 | stale-vs-rebuilt 0.097 | 🟡 leak |
  | encoder cluster-ID (FT encoder) | retrain-in-place | stale sibling_top1 0.315 | ramoleft 0.180 | +0.006 | 0.083 | 🔴 FT backfires |
  | embed, **dropped-expert** | drop (mask 15/32) | sibling_top1 1.0; top-3 survivors capture **0.643**; sim-ratio **0.980**; entropy 0.703 | — (routing-only) | — | **0.727** | 🔴 leak + collateral |
  | soft RAMoLE (alpha) | retrain-in-place | router unchanged; forget ppl 2.97→11.82 | 0.484 (embed)/0.890 (key) | ≈+0.01 | H_norm 0.818→0.842 | 🔴 leak, Δmu≈0 |

- **What worked / hypothesis verdict:**
  - **H1 (encoder confound is material) — SUPPORTED.** Base-pinned (off-the-shelf instructor-xl)
    stale sibling_top1 = **0.185** vs the FT-encoder audit's 0.315: the 2026-07-02 audit was
    confounded (it measured the FT arm, fq 0.180, not the fq-0.484 off-the-shelf arm). The
    off-the-shelf encoder misroutes FEWER orphans to siblings.
  - **H2 (dropped-expert fallback leak) — SUPPORTED (leak-prone).** Under `dropped`, orphan top-1
    mass concentrates (top-3 surviving experts capture 0.643, normalized entropy 0.703 < uniform)
    and mean masked/unmasked top-1 sim-ratio = **0.980** — the surviving sibling matches the
    orphan query nearly as well as the dropped expert did. Meets the pre-registered CONFIRM
    (top-3 ≥ 0.5 AND sim-ratio ≥ 0.9). The FT encoder is less leak-prone (sim-ratio 0.768).
  - **H3 (retain collateral of a drop) — SUPPORTED, strongly.** Masking the 15 affected experts
    shifts **72.7 %** of retain top-1 routes (predicted 0.3–0.6; measured higher). A drop-style
    deletion on this pool would badly perturb innocent-query routing — the mechanistic reason the
    pool deletes by retrain-in-place, not by dropping.
  - **H4 (author-key clean) — SUPPORTED.** Strong-experts scaffold arm at extended caps:
    `--delete_shard 9` leaves **model_utility identical 0.7509 → 0.7509** (Δmu = 0); del9 fq
    0.0241 is the base-served-answer floor artifact (judged against the scaffold floor per the
    pre-registered interpretation guard, not the 0.89 oracle). Orphans route to base+scaffold with
    zero retain shift.
  - **H6 (FT encoder worsens orphan routing) — SUPPORTED.** base-pinned 0.185 < FT 0.315.
- **Observations:** The two deletion operations are opposites, and that is the headline. **Retrain-
  in-place** (legonet/ramole pool): orphans keep routing INTO their now-scrubbed experts (safe),
  retain routing barely moves, unlearn Δmu is small and POSITIVE — so the §9-D "learned router
  loses the most utility" prediction is REFUTED here (every Δmu ≈ +0.01). **Drop-an-expert** (the
  literal §9-D scenario): orphans hit near-perfect siblings (sim 0.98) and 73 % of retain routes
  move — which is exactly why hard author-key routing (orphans → clean scaffold, zero shift, zero
  Δmu) is the safe design and the embedding/soft router is the leak channel. Silent-failure checks
  clean: key selection-shift asserted 0.0; stale index sha-unchanged; no NaNs.
- **New questions / new hypotheses:** Would the abstain/OOD-threshold fix (C Phase C, unrun) push
  the base-pinned dropped-policy orphans off the 0.98-sim sibling onto the scaffold, and by how
  much? Does the SEUF anchor loss lower the embed-route leak (fq 0.484 → toward 0.89) without
  costing mu? These are the two fix arms; both are coded-design-only, deferred.
- **Next Steps:** (optional) implement the C-Phase-C abstain + anchor fix arms and re-audit;
  otherwise C is complete. Fold the filled table into `reports/ROUTING_AUDIT_REPORT_2026-07-06.md`.
