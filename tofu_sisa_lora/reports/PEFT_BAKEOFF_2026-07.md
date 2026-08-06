# PEFT composability bake-off — Phase A results (2026-07-14)

**Question:** does any PEFT parameterization whose natural composition rule is not weight
addition give a single composed model (no routing) that beats composed LoRA, with O(1)-exact
deletion? Pre-registration: `log/peft_compose/2026-07-08_bakeoff-design.md` (H1–H4).
Setup: Llama-3.2-1B-Instruct, k=10, seed 42, smoke caps, `--k 10 --forget_shard_id 9`;
config `configs/peft_bakeoff_1b.json`; SLURM 442746 (smokes), 443125 → 443126 → 443127
(train → compose → eval chain). Script sha256s: train_peft_shard 941ab4cb8d89,
compose_peft ffeb7f6c21ab, prefix_concat af7375bbee3f, test_compose_peft de2b4e13cf7b,
submit_peft_bakeoff 849c10c966f8.

**Anchors (same eval tier):** base mu **0.3796** (f_rouge 0.3497, f_ppl 20.36) ·
joint-ft `ft_all` mu **0.5302** · LoRA incumbent `merged_additive_mean` (legacy r8 k10 pool)
mu **0.4190** / `remerge` 0.4212.

## Headline

**The composition bottleneck is operator-independent.** Every mean-like composition —
LoRA-additive, DoRA-additive, VeRA (shared frozen basis), IA³ arithmetic AND geometric gate
mean — lands on the same base+ε plateau, mu 0.415–0.434, regardless of parameterization.
The one non-weight-space composer (prefix KV-concatenation) fails catastrophically: prefixes
trained independently are mutually out-of-distribution, and concatenating them drives the
model into garbage (mu 0.002, forget_ppl 9,248). Routing beats composition in **every**
parameterization. **No arm reaches the Phase B gate (composed mu ≥ 0.55) → Phase B (k=200)
does not run.**

## Per-arm table (smoke tier)

| arm | trainable/shard | iso mu (s0/s5/s9) | iso f_rouge s9 | composed_full mu | comp f_rouge (Δ vs iso s9) | composed_unlearn mu / f_ppl full→unl | routed mu |
|---|---|---|---|---|---|---|---|
| prefix (KV concat) | 1.05 M | 0.056/0.036/0.074 ⚠ | 0.4661 | **0.0018** | 0.1530 (−0.313) | 0.0022 / 9248→7344 | 0.0000 ⚠ |
| VeRA (basis mean) | 74 K | 0.389/0.421/0.399 | 0.4583 | **0.4150** | 0.4387 (−0.020) | 0.4171 / 9.32→10.08 | 0.4447 |
| IA³ (gate mean) | 147 K | 0.361/0.392/0.396 | 0.5239 | **0.4298** | 0.4281 (−0.096) | 0.4330 / 9.32→11.33 | **0.5155** |
| IA³ (gate geo-mean) | 147 K | — (same pool) | 0.5239 | **0.4302** | 0.4303 (−0.094) | 0.4347 / 9.23→11.26 | — |
| DoRA (additive_mean) | 17.5 M | 0.377/0.399/0.384 | 0.5067 | **0.4317** | 0.4485 (−0.058) | 0.4343 / 7.99→10.01 | 0.4906 |
| LoRA anchor (legacy r8) | 3.4 M | (Exp-3: drop 0.090) | — | **0.4190** | 0.4608 | 0.4212 / 7.53→9.23 | — |

## Findings

1. **Capacity gate:** IA³ is the surprise — the *best* isolated memorizer of all arms
   (own-shard f_rouge 0.52–0.57, f_ppl 2.6) at 147 K params/shard, refuting the "gates can't
   store facts" half of H4. VeRA is moderate (0.42–0.46, ppl 5.2). **Prefix fails the gate**:
   even served alone, a trained prefix destroys general behavior (iso mu 0.04–0.07 ≪ base
   0.38; retain_prob 0.02–0.06) while still generating its own shard's answers (rouge 0.38–0.47)
   — prompt-learning overwrites the model's response distribution rather than adding to it.
2. **Composition plateau:** composed mu — VeRA 0.4150, IA³ 0.4298/0.4302, DoRA 0.4317, LoRA
   0.4190 — statistically one band, ≈ base + 0.04, ≈ ⅓ of the way to joint-ft. IA³'s geometric
   (product-analog) and arithmetic compositions are indistinguishable (Δmu 0.0004; 0/147,456
   gates needed the sign fallback). Own-shard recall dilutes in composition for the stronger
   memorizers (IA³ −0.096, DoRA −0.058 ≈ LoRA's −0.090 Exp-3 analog); VeRA dilutes least
   (−0.020) but from the weakest iso. Echoes Exp-5's "the 1/N mean is a constant style
   adapter" — in every parameterization.
3. **Deletion:** exact by construction everywhere it applies — compose-time identity asserts
   passed (<1e-6); prefix segment-drop byte-exact. Behaviorally, dropping shard 9 raises
   forget_ppl in every weight-space arm (9.3→10.1 / 9.3→11.3 / 8.0→10.0 / 7.5→9.2) with fq
   high (0.59–0.96; KS is low-power at smoke n). The property held; it just isn't worth much
   at this utility level.
4. **Routing wins in every parameterization:** IA³ routed 0.5155 (≈ joint-ft 0.5302!), DoRA
   0.4906, VeRA 0.4447 — each ≫ its own composed arm. Practical nugget: IA³ + author-key
   routing serves 10 shards in **1.5 MB** of adapters at near-joint-ft utility — a 100×
   storage reduction vs LoRA r32 pools for the routed serving mode. Prefix routed mu = 0.0000
   (the OOD-destruction issue again — one utility component zeroes the harmonic mean).

## Hypothesis verdicts (pre-registered)

- **H1 REFUTED** — prefix-concat does not preserve own-shard recall (−0.31, not ≤0.05);
  concatenation of independently-trained prefixes is mutually OOD, mu 0.002.
- **H2 REFUTED** — VeRA/IA³ composed mu does not materially beat the LoRA additive anchor
  (0.415–0.430 vs 0.419); the dilution clause held (IA³ −0.096) but the win clause failed.
- **H3 SUPPORTED** — deletion exact in every arm (asserts + byte-exact segment drop);
  forget_ppl rises on deletion everywhere.
- **H4 SPLIT** — supported for prefix (fails the gate, via serving-destruction more than
  storage), refuted for IA³ (best memorizer per parameter of all arms).

## Decision

Phase B (k=200 per-author) **not triggered** — no arm at composed mu ≥ 0.55. The bake-off
confirms the merge-mechanism conclusion from the operator side: input-conditioned selection
(routing) is where composition quality lives; changing the parameterization does not rescue
input-blind composition. The IA³+routing efficiency result is the actionable follow-up.
