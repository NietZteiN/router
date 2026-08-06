# peft_compose — do non-additive PEFT parameterizations compose better than LoRA?

**Status:** Phase A complete (Phase B gate not met) · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 2 (2026-07-08 → 2026-07-14)

The merge_mechanism thread proved LoRA's failure to combine is a property of the additive
*operator*: 98% of inter-adapter excess similarity is one shared rank-1 direction that piles up
under weight averaging (k200 study), own-author recall is ~85% gone by N≈8 co-merged adapters,
and no global λ rescues it. This thread asks whether PEFT methods whose natural composition is
NOT weight addition give a **single composed model** (no routing) with materially better utility
while keeping O(1)-exact deletion: prefix-tuning (compose = KV concatenation; attention routes),
VeRA (compose = mean in a shared frozen basis; train-time JD), IA³ (compose = commutative gate
product/mean with exact inverse), and DoRA (additive control — does magnitude/direction
decoupling alone soften pile-up?). Phase A = k=10 bake-off on Llama-3.2-1B; Phase B (per-author
k=200) sits behind an explicit decision gate (composed mu ≥ 0.55 + exact deletion + capacity
gate passed). Bars at matched eval: LoRA additive/λ-sweep mu ≈ 0.43–0.46; routed 0.75; base ≈ 0.43.

## Hypotheses — open / resolved
- **[resolved ✗ refuted]** H1: prefix-concat collapses (mu 0.0018, f_ppl 9,248; own-shard rouge
  −0.31 vs the ≤0.05 bar) — independently-trained prefixes are mutually OOD
  ([2026-07-14_bakeoff-results.md](2026-07-14_bakeoff-results.md)).
- **[resolved ✗ refuted]** H2: no material mu win — VeRA/IA³ composed 0.415–0.430 vs LoRA anchor
  0.419 (Δ ≤ +0.011); dilution clause held (IA³ −0.096 ≈ LoRA's −0.090).
- **[resolved ✓ supported]** H3: deletion exact in every arm (compose asserts <1e-6; byte-exact
  prefix drop); forget_ppl rises on deletion everywhere (e.g. ia3 9.32→11.33).
- **[resolved — split]** H4: supported for prefix (iso mu 0.04–0.07 ≪ base 0.38 — serving one
  prefix destroys general behavior), refuted for IA³ (best per-parameter memorizer: own-shard
  f_rouge 0.52–0.57 / f_ppl 2.6 at 147K params).
- **[open]** H-ia3-route-200: IA³ + author-key routing holds ≈ joint-ft utility at k=200
  per-author granularity (~30 MB total) — the cheapest exact-deletion serving stack if true.
- **[open]** H-prefix-joint: jointly-trained / scaffold-aware prefixes stop destroying general
  behavior, making concat-composition meaningful.

## What worked
- **The headline (Phase A, 2026-07-14):** the composition plateau is operator-independent —
  additive LoRA/DoRA, VeRA shared-basis mean, IA³ gate mean AND gate product all land at
  base+0.04 (composed mu 0.415–0.434 vs base 0.3796, joint-ft 0.5302); parameterization does not
  rescue input-blind composition. Full table: `../../tofu_sisa_lora/reports/PEFT_BAKEOFF_2026-07.md`.
- Routing > composition in every parameterization; **IA³ + key routing = 0.5155 ≈ joint-ft
  0.5302 at 1.5 MB of adapters** (100× smaller than LoRA r32 pools).
- Deletion exactness held everywhere it was claimed (asserts + byte-exact segment drop).
- IA³ is the best memorizer per parameter (f_rouge 0.52–0.57 at 147K/shard).

## What didn't / open problems
- Prefix arm: catastrophic — a trained prefix overwrites the response distribution even served
  alone (iso mu 0.04–0.07), and concatenation is garbage (mu 0.0018). Prompt-learning needs
  joint/scaffold-aware training before it can be a composition candidate.
- No arm reached the Phase B gate (composed mu ≥ 0.55; best 0.4317) → k=200 Phase B not run.

## Open ideas / next steps
- H-ia3-route-200 (the actionable follow-up, under the routing thesis rather than the compose
  gate): 200 per-author IA³ experts + author-key routing at ~30 MB total.
- H-prefix-joint if prompt-learning composition is ever revisited.

## Entries (chronological)
- [2026-07-08 — bake-off design](2026-07-08_bakeoff-design.md) — pre-registration: arms, compose
  rules, H1–H4, protocol, decision gate.
- [2026-07-14 — bake-off results](2026-07-14_bakeoff-results.md) — Phase A: H1/H2 refuted, H3
  supported, H4 split; operator-independent plateau at base+0.04; routing wins in every
  parameterization (IA³ routed 0.5155 @1.5 MB); Phase B gate not met.
