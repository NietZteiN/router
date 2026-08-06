# From "how similar are the per-author adapters?" to the PEFT composability bake-off
### Self-contained report of the 2026-07-07 → 07-14 experiment arc

## 0. The problem

The repo's goal is exact machine unlearning: store each TOFU author's knowledge in a separate
module so that deleting an author = dropping their module, O(1) and provable. We have 200
per-author LoRA adapters (Llama-2-7B r32, `_k200_r32_e5_lr1e4`) and, at k=10 granularity,
pools on Llama-3.2-1B. The known tension: **routing** over isolated modules works well
(best track: mu 0.7509 at 7B-scaffold tier) but needs per-query module selection, while
**composing all modules into one model** — the operationally simplest serving mode — collapses
utility. This arc asked three questions in sequence: (1) how similar are the per-author
adapters actually? (2) can a shared scaffold absorb what they have in common? (3) does any
other PEFT parameterization compose better than LoRA?

## 1. Experiment 1 — weight-space similarity of the 200 per-author adapters

**Setup.** `subspace_overlap.py` over all 200 adapters (SLURM 440863, CPU, 4h25m, seed 42,
n_null=5): pairwise cosine of effective deltas ΔW = scaling·B·A, principal angles between
the B column-spaces / A row-spaces, shared-subspace energy at rank 16 — each calibrated
against factored-random nulls. Post-processing: eigen-decomposition of the 200×200 cosine
matrix, MDS/PCA embeddings, name-token permutation test.

**Results.**
- Mean pairwise cosine **0.00125** vs null 5.4×10⁻⁸ → z = **19,716**. Nearly orthogonal in
  magnitude, unambiguously aligned in significance (×22,942 above random).
- The alignment is **100% output-side**: col(B) angle-cos 0.164 vs 0.070 null; row(A) equals
  its null to 5 decimals. Adapters read from random directions, write into a shared subspace.
- **One shared direction is the whole story**: top eigenvalue λ₁ = 1.250 over a flat bulk
  (0.994–1.004); the best rank-1 layer carries **98.0%** of all off-diagonal similarity
  energy; spectrum participation ratio 199.94/200. A rank-16 shared basis holds 23% of every
  adapter's energy (92× the 0.25% chance).
- **No cluster structure** (2-D MDS captures 1.0% of variance = the isotropic 2/199 floor;
  uniform heatmap). Only other deviation from iid: **name-token proximity** — author pairs
  sharing a name token are more similar (0.0014 vs 0.0012, permutation p ≈ 5×10⁻⁴; top pair
  Yeon Soo ↔ Yeon Park).

**Conclusion.** The adapters are iid draws from a shared *anisotropic* distribution: one
common "fine-tuned QA style" direction (the thing that piles up under additive merging) plus
author-specific content in effectively random directions. Log:
`log/merge_mechanism/2026-07-07_per-author-similarity-k200.md`.

## 2. Experiment 2 — does a scaffold absorb the shared component? (H-scaf)

**Setup.** Same analysis on (a) the `_experts_scaf_k10` pool (1B r32 experts trained on the
Alpaca-scaffolded base), (b) plain 7B k10 r32 (same recipe, different model), (c) the
existing plain 1B k10 r8 numbers. Null-calibrated ratios only (the raw numbers are not
cross-model comparable — the chance floor scales ≈√(r/d)).

**Results.** Scaffolded experts show the lowest calibrated overlap on every metric:
col(B) ratio **1.78×** vs 2.23× (same-recipe 7B) vs 4.01× (same-model legacy); shared energy
5.97× vs 7.83× chance; cosine z 901 vs 4280. But the shared component only shrinks —
col(B) excess +0.104 remains.

**Conclusion.** Directionally supported: a scaffold absorbs *part* of the common component.
Each comparison is confounded (no plain 1B k10 r32/e5 exists); the decisive 10-job control is
logged as open. Log: `log/merge_mechanism/2026-07-08_scaffold-overlap-hscaf.md`.

## 3. Experiment 3 — the PEFT composability bake-off (the centerpiece)

**Question.** The merge-mechanism thread had shown the additive *operator* is the failure:
recall collapse saturates by N≈8 co-merged adapters and composed mu is flat at every N.
So: does a PEFT method whose natural composition is **not weight addition** give a single
composed model beating composed LoRA, while keeping O(1)-exact deletion? Pre-registered
H1–H4 in `log/peft_compose/2026-07-08_bakeoff-design.md`.

**Arms and composition rules** (k=10 on Llama-3.2-1B, seed 42, one adapter per shard):

| arm | trainables/shard | compose rule | delete rule |
|---|---|---|---|
| prefix-tuning | 1.05 M (64 KV tokens) | **concatenate** prefixes in the KV cache | drop the segment (byte-exact) |
| VeRA | 74 K (vectors over shared frozen basis) | elementwise mean in the shared basis | recompute mean without author (exact) |
| IA³ | 147 K (multiplicative gates) | arithmetic mean + geometric mean of gates | recompute (exact) |
| DoRA | 17.5 M (LoRA r32 + magnitude) | existing `additive_mean` path (control) | drop-term |

**Protocol.** CPU regression gate (compose identities, N=1 prefix-serving equivalence,
DoRA merge probes) → 4 micro-smokes (SLURM 442746) → chain 443125 (40 trainings, ≤4 GPUs) →
443126 (compose + exact-delete asserts, all <1e-6) → 443127 (28 evals, smoke tier,
`--k 10 --forget_shard_id 9`). Per arm: isolation probes at shards {0,5,9} (capacity gate),
composed_full, composed_unlearn (minus shard 9), and a routed_key_exact reference.
**Anchors:** base mu 0.3796 · joint-ft 0.5302 · LoRA `merged_additive_mean` 0.4190.
**Phase B gate** (advance to k=200): composed mu ≥ 0.55.

**Results.**

| condition | prefix | VeRA | IA³ mean | IA³ geo | DoRA | LoRA anchor |
|---|---|---|---|---|---|---|
| iso own-shard f_rouge (s9) | 0.466* | 0.458 | 0.524 | — | 0.507 | — |
| iso mu | **0.04–0.07 ⚠** | 0.39–0.42 | 0.36–0.40 | — | 0.38–0.40 | — |
| **composed_full mu** | **0.0018** | **0.4150** | **0.4298** | **0.4302** | **0.4317** | **0.4190** |
| composed own-shard drop vs iso | −0.31 | −0.020 | −0.096 | −0.094 | −0.058 | −0.090 (Exp-3) |
| deletion: forget_ppl full→unlearn | (degenerate) | 9.3→10.1 | 9.3→11.3 | 9.2→11.3 | 8.0→10.0 | 7.5→9.2 |
| routed_key_exact mu | 0.0000 ⚠ | 0.4447 | **0.5155** | — | 0.4906 | — |

\*prefix stores its shard (rouge 0.38–0.47) but serving even ONE prefix destroys general
behavior (retain_prob 0.02–0.06) — the response distribution is overwritten, not extended.

**Findings.**
1. **The composition plateau is operator-independent.** Four different operators (additive,
   shared-basis mean, gate mean, gate product) in four parameterizations all land at
   base+0.04 (mu 0.415–0.434) — indistinguishable from composed LoRA. IA³'s product vs mean
   composition differ by 0.0004.
2. **The only non-weight-space composer failed catastrophically.** Concatenated
   independently-trained prefixes are mutually out-of-distribution: mu 0.0018, forget_ppl
   9,248. Attention does not route over them; it drowns.
3. **Deletion exactness held everywhere** (compose-time identity asserts; byte-exact segment
   drops; forget_ppl rises on every deletion). The property is cheap — it's the utility that
   isn't there.
4. **Routing beats composition in every parameterization**, and **IA³ + author-key routing
   reaches mu 0.5155 ≈ joint-ft 0.5302 with 1.5 MB of adapters** — 100× smaller than LoRA
   r32 pools. IA³ was also the best memorizer per parameter (own-shard rouge 0.52–0.57 at
   147K params).

**Verdicts.** H1 (prefix keeps recall) **refuted**. H2 (VeRA/IA³ beat the LoRA anchor)
**refuted** (Δ ≤ +0.011). H3 (exact deletion everywhere) **supported**. H4 (capacity risk)
**split** — supported for prefix, refuted for IA³. **Phase B not triggered** (best 0.432 <
0.55 gate). Log: `log/peft_compose/2026-07-14_bakeoff-results.md`;
table: `reports/PEFT_BAKEOFF_2026-07.md`.

## 4. What the arc adds up to

Three independent lines now converge:
- *Geometry* (Exp 1): the only shared structure across per-author adapters is one output-side
  style direction; the content is mutually random — there is nothing for an input-blind
  combiner to exploit.
- *Behavior* (merge_mechanism Exp-5, concurrent): additive merging keeps the shared style and
  dilutes the content, at every N, with no safe regime.
- *Parameterization* (this bake-off): swapping the adapter family and the composition operator
  does not move the plateau; the one operator outside weight space fails for its own reasons.

**Implication:** input-conditioned selection (routing) is not an optimization over
composition — it is where composition quality lives. The design consequences already in
motion: maximize the shared scaffold (H-scaf, control pending), keep per-author modules
minimal and droppable, and serve routed. **Open follow-up with the best cost/benefit:
H-ia3-route-200** — 200 per-author IA³ experts + author-key routing at ~30 MB total, a
candidate for the cheapest exact-deletion serving stack in the repo.

*Provenance: SLURM 440863 (k=200 similarity), 441240 + smoke pair (H-scaf), 442746 +
443125/443126/443127 (bake-off); seeds 42 throughout; configs in `configs/peft_bakeoff_1b.json`;
script sha256s in the dated log entries.*
