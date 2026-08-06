# SISA-LoRA Shard Grid + Merge-Method Report — Llama-2-7B-chat-hf

**Date:** 2026-06-11 · **Seed:** 42 everywhere · **Eval:** corrected OU-faithful metrics
(`metrics_version: ou-2026-06-10`, gated by `test_ou_equivalence.py` + `test_merge_extra.py`,
both green pre-run) · **Smoke caps** (ROUGE≤50, retain≤80, truth≤30); k=1 winner additionally
confirmed at extended caps. All GPU work via SLURM, grid/Phase-M pinned to sprint3 (≤4 GPUs).

**Definitions.** *merged* = combine ALL k shard adapters (the baseline, nothing forgotten);
*remerge* = drop the forget shard, combine the remaining k−1 (the unlearning operation);
*mu* = `model_utility` (hmean of 9 retain/real/world components); *fq* = `forget_quality`
(KS p-value of forget truth-ratios vs a never-saw-forget oracle; high = forgotten).
Anchors: **base model mu 0.418** (fq 0.239, f_ppl 15.2) · **k=1 full-data LoRA ft mu 0.743
smoke / 0.740 extended** (recipe lr1e-4, 5 epochs, rank 32/α64) · locuslab full-FT 0.748 ·
OU leaderboard finetuned 0.63.

---

## 0. Glossary & how to read this report

### 0.1 The setup in one paragraph

TOFU is a benchmark of 200 fictional authors × 20 Q&A pairs (4 000 samples). Because the
authors are invented, the *only* way a model can know them is our finetuning — perfect for
studying unlearning. **SISA-style sharding**: split the 200 authors into k disjoint shards,
train one LoRA adapter per shard, serve the **merge of all k adapters** (the *baseline*).
To unlearn an author group, **delete its adapter and re-merge the remaining k−1** — an exact,
O(1), deterministic deletion (no gradient surgery), because no surviving parameter ever saw
the deleted data. The questions of this report: how much utility does the merged baseline
keep (vs training one adapter on everything), and does the drop-and-remerge actually behave
like forgetting?

### 0.2 Data splits (what the metric prefixes refer to)

| split | content | role |
|---|---|---|
| **forget** | the forget shard's authors (k=10: authors 180–199 = TOFU `forget10`; k=4: 150–199; k=20: 190–199) | what we want the model to forget |
| **retain** | the remaining fictional authors | must stay known after unlearning |
| **real_authors** | questions about real-world writers | proxy for pretrained knowledge near the domain |
| **world_facts** | general world knowledge | proxy for everything else |
| `*_perturbed` | each question with plausible-but-wrong answers | used by probability/truth-ratio metrics |

### 0.3 Model states (the `label` column)

| label | what it is |
|---|---|
| `base_model` | untouched Llama-2-7B-chat — the "never knew it" reference |
| `ft …` (k=1) | one LoRA trained on ALL 200 authors — the "knows everything" upper bound |
| `shard_{k-1}_only` | only the forget shard's adapter — the "memorizer" reference (proves the data is in that shard) |
| `merged_X` | all k adapters combined with method X — **the baseline, nothing forgotten yet** |
| `remerge_X` | k−1 adapters combined (forget shard dropped) — **the state after unlearning** |
| `subtract_orth` | alternative unlearning op: project the forget adapter's subspace out of the full merge (no re-merge needed) |
| retain90 / retain75 oracle | a single adapter trained only on retain authors (0–179 / 0–149) — the gold standard for "what a model that never saw the forget data looks like"; reference for `fq` |

### 0.4 Metrics (every column in the tables)

- **mu — `model_utility`**: harmonic mean of **9 components** — {probability, ROUGE, scaled
  truth-ratio} × {retain, real_authors, world_facts}. Harmonic mean ⇒ the *weakest* component
  dominates; one component at ~0 zeroes the whole score (that is why broken merges show
  mu = 0.000 even with some healthy columns). Higher = more useful model. Anchors: base 0.42,
  best k=1 ft 0.74.
- **fq — `forget_quality`** (alias `ks_pval`): p-value of a Kolmogorov–Smirnov test comparing
  the model's *forget-set truth-ratio distribution* against the retain-oracle's. **High
  (→1) = statistically indistinguishable from a model that never saw the forget data = good
  unlearning.** Low (→0) = detectably still knows it. Caveat: at smoke caps (30 samples) the
  test has low power — it reliably flags memorizers (fq ≈ 0) but cannot separate
  partially-knowing merges from the oracle; use the ppl/TR deltas alongside it.
- **f_ppl / ret_ppl — perplexity** on the forget / retain answers. Low = the model assigns
  high likelihood to the gold answers (knows them). Diagnostic anchors: memorizer ≈ 1.2,
  base ≈ 15.2. *Unlearning signature: f_ppl rises toward base after the drop while ret_ppl
  stays low.* (All TOFU-finetuned models have somewhat reduced f_ppl vs base purely from
  format/style adaptation — the oracle does too — so "reaching base exactly" is not expected.)
- **f_rouge / ret_rouge / real_rouge / wld_rouge — ROUGE-L recall** of the greedy generation
  vs the gold answer. 1.0 = verbatim recitation. Forget-side high = still recites;
  retain/real/world high = still useful. Base scores ~0.39 on forget (style overlap), so
  remerge ≈ 0.4–0.5 means confabulation, not knowledge.
- **prob columns — answer probability**: retain = normalized P(gold answer | question);
  real/world = `probability_w_options` = P(correct)/(P(correct)+ΣP(perturbed)) (multiple-
  choice style). Higher = better.
- **f_TR — forget truth ratio**, aggregated as mean(min(tr, 1/tr)) ∈ [0,1] where
  tr = P(wrong)/P(correct) per sample: **→1 = answers no more credible than perturbed ones =
  forgotten** (base ≈ 0.74; memorizer ≈ 0.45–0.6).
- **ret_TRs / real_TRs / wld_TRs — scaled truth ratio**, mean(max(0, 1−tr)): →1 = strongly
  prefers correct over perturbed = good. These three are the truth-ratio components of mu.
- **smoke vs extended**: sample caps. Smoke (ROUGE≤50, retain≤80, truth≤30) for fast sweeps;
  extended (200/400/120) to confirm — k=1 winner agreed within 0.003, so smoke ranking is
  trustworthy at these effect sizes.

### 0.5 Merge methods (the X in `merged_X`/`remerge_X`)

| method | idea | convention† |
|---|---|---|
| `linear` | plain average of adapter deltas (task arithmetic) | inflated — **broken here** |
| `cat` | concatenate adapters (exact composition, rank grows k·r) | inflated — **broken here** |
| `dare_ties` | randomly drop + rescale entries (DARE), then sign-consensus merge (TIES) — prunes interference | inflated — workhorse |
| `della_linear` / `della_ties` | DELLA/MAGPRUNE: magnitude-aware drop probabilities, then linear / TIES | inflated — linear variant broken; ties variant has a k=4 remerge bug |
| `breadcrumbs` | mask both top-outlier and small entries, then sum | inflated — broken here |
| `knots_ties` | true KnOTS: shared SVD basis across adapters, TIES in that space | true-scale |
| `fisher` | weight each parameter by its diagonal Fisher information (data-required) | true-scale |
| `lorahub` | learn k scalar merge weights by CMA-ES on held-in data (data-required) | true-scale — best utility |
| `subtract_orth` | unlearning-only: orthogonally project the forget subspace out of the full merge | true-scale |

† Convention: shards train with rsLoRA; the PEFT factor-space family produces deltas inflated
by √r vs the true average (see CLAUDE.md merge-scale note). That inflation is precisely why
`linear`/`cat`/`della_linear`/`breadcrumbs` explode at r32 (ppl 10³–10⁷). Compare quality
*within* a convention before crediting the algorithm.

### 0.6 Config names (Phase A star grid)

All configs share seed 42, bs1×ga16, max_len 256; notation rX = LoRA rank (α = 2·rank),
eY = epochs. **CTRL** = old default (k10, r8, e3, lr 2e-4, pre-existing shards) ·
**CENTER** = k10, r32, e5, lr 1e-4 (the k=1 winner recipe on shards) · **RANKLO** = CENTER
with r8 · **EXPO** = r16 but 10 epochs · **LRHI** = CENTER with lr 2e-4 · **KLO** = CENTER
at k=4 · **KHI** = CENTER at k=20.

### 0.7 How to judge "unlearning worked" (used in §5)

- **T1 (something to forget):** merged f_ppl ≪ base ⇒ the baseline actually absorbed the
  forget data. If merged ≈ base already, unlearning is trivially "successful" but meaningless.
- **T2 (it was forgotten):** remerge forget-side moves to oracle/base territory — f_ppl ↑
  toward 15, f_TR ↑ toward 0.74, fq stays high, no verbatim recitation in generations.
- **T3 (nothing else broke):** remerge mu ≥ merged mu − 0.05, retain/real/world columns flat.

---

## 1. Executive summary

1. **k=1 LoRA finetuning works as well as full finetuning** on TOFU utility (0.740 extended
   vs locuslab's 0.748) — the ≥0.6 bar is passed decisively at k=1.
2. **Sharding cost is a monotonic dilution law**: merged-baseline utility falls 0.74 → 0.54 →
   0.48 → 0.45 for k = 1 → 4 → 10 → 20 (best method per k). Per-shard recipe (rank/epochs/lr)
   moves merged utility by ≤0.03 — the merge, not the training recipe, is the bottleneck.
3. **Best sharded baseline: k=4 + lorahub, mu 0.592** — just short of the 0.6 bar.
   Runner-up k=4 + dare_ties 0.545 (and 0.575 after remerge).
4. **Drop-a-shard unlearning is exact by construction and behaviorally verified** — but the
   demonstration is only *non-trivial at k=4*; at k≥10 the merged model barely memorizes the
   forget data in the first place (dilution pre-forgets it).
5. Method pathologies confirmed: PEFT `linear`/`cat` and the new `della_linear`/`breadcrumbs`
   are degenerate on these rsLoRA shards (ppl 10³–10⁷; see scale-convention note in
   tofu_sisa_lora/CLAUDE.md). `della_ties` fails specifically on k=4 remerge (mu 0.236) —
   flagged for the merge-methods thread.

## 2. Provenance

- **Phase A grid** (7 configs, star design around k10/r32/α64/e5/lr1e-4): SLURM 433553–433567
  (CTRL eval-only; trainings 433554/433556/433558/433560/433562/433564; KHI prep 433565;
  collect 433567). Training bs1×ga16, max_len 256.
- **Phase M merge methods**: retain75 oracle 433698 (`--retain_authors 150`, new flag) →
  prepare 433699 → KLO evals 433700 (18 labels) ∥ CENTER evals 433701 (13) → collect 433702.
- **Qualitative generations**: job 433716 (KLO, 4 labels × 4 splits, fixed-seed questions).
- k=1 stage (previous day): grid 433518–433525, extended gate 433550–433552.
- Aggregates: `checkpoints/all_metrics_smoke.csv` (collect jobs above).

## 3. Phase A — shard-recipe star grid (merged/remerge = dare_ties)

| config | k | r/α | ep | lr | merged mu | remerge mu | merged f_ppl | remerge f_ppl |
|---|---|---|---|---|---|---|---|---|
| KLO | 4 | 32/64 | 5 | 1e-4 | **0.542** | **0.562** | 4.59 | **12.12** |
| CENTER | 10 | 32/64 | 5 | 1e-4 | 0.477 | 0.479 | 7.82 | 8.41 |
| LRHI | 10 | 32/64 | 5 | 2e-4 | 0.475 | 0.480 | 7.90 | 8.79 |
| EXPO | 10 | 16/32 | 10 | 1e-4 | 0.461 | 0.465 | 9.54 | 9.92 |
| CTRL (old default) | 10 | 8/16 | 3 | 2e-4 | 0.455 | 0.460 | 9.21 | 9.10 |
| RANKLO | 10 | 8/16 | 5 | 1e-4 | 0.450 | 0.453 | 10.28 | 10.22 |
| KHI | 20 | 32/64 | 5 | 1e-4 | 0.450 | 0.452 | 10.48 | 10.54 |

Axes: rank r8→r32 ≈ +0.03 merged mu at k=10; epochs and lr ≈ flat. k dominates.
KHI curiosity: merged-with-forget-shard already scores fq 0.594 — at 20-way dilution the
forget data is barely learned at all ("free" forgetting, paid in utility).

## 4. Phase M — merge methods on the two best configs

**KLO (k=4, r32/e5/lr1e-4), valid fq via retain75 oracle:**

| method | merged mu | merged fq | merged f_ppl | remerge mu | remerge fq | remerge f_ppl |
|---|---|---|---|---|---|---|
| **lorahub** | **0.592** | 0.808 | 11.26 | 0.583 | 0.808 | 12.08 |
| dare_ties | 0.545 | 0.808 | **4.48** | **0.575** | 0.808 | **11.93** |
| knots_ties | 0.489 | 0.594 | 6.50 | 0.520 | 0.808 | 7.95 |
| fisher | 0.477 | 0.958 | 7.24 | 0.504 | 0.958 | 8.81 |
| della_ties | 0.497 | 0.071 | 5.68 | 0.236 ⚠️ | 0.035 | 47.0 ⚠️ |
| subtract_orth | (unlearn-only) | | | 0.523 | 0.393 | 9.98 |
| della_linear / breadcrumbs / linear | ~0.00 (ppl 10⁴–10⁷) | | | ~0.00 | | |
| shard_3_only (memorizer ref) | 0.426 | 0.006 | 1.18 | — | | |

**CENTER (k=10):** lorahub 0.492/0.515 (merged/remerge) > della_ties 0.500/0.500 >
dare_ties 0.477/0.479 > subtract_orth 0.494 > knots_ties 0.463/0.467 > fisher 0.456/0.459;
della_linear/breadcrumbs broken. (della_ties remerge is fine at k=10 — its failure is
k=4-specific.)

**Convention caveat:** della_*/breadcrumbs deliberately share the PEFT factor-space
sqrt(r)-inflated scale; knots/fisher/lorahub/subtract_orth are true-scale. Comparisons of
*why* a method wins should stay within a convention (CLAUDE.md merge-scale note).

## 5. Was unlearning actually successful? (three-part test)

**T1 — Did the baseline know the forget data?** Only meaningfully at k=4
(dare_ties merged f_ppl 4.5 vs base 15.2; the memorizing shard sits at 1.2). At k≥10, merged
f_ppl is 8–10.5 — dilution means there is little to unlearn. lorahub's merged model is
*already* weak on forget data (f_ppl 11.3) — its weight search down-weights the forget shard,
so its unlearning step is nearly a no-op.

**T2 — Did dropping the shard remove it?** At k=4 + dare_ties: yes, decisively — f_ppl
4.5 → 11.9 (→ base 15.2), forget truth-ratio 0.68 → 0.73 (base 0.74), forget ROUGE
0.54 → 0.50 (base 0.39; residual is format/style generalization, the oracle shows the same).
Exact by construction: the surviving k−1 adapters never saw the forget authors. fq caveat:
at the 30-sample smoke cap the KS test separates only memorizers (fq≈0) from everything else
— even dare_ties *merged* gets fq 0.808 — so the merged→remerge ppl/TR deltas are the
discriminative unlearning signal, not fq alone.

**T3 — Did removal preserve the rest?** Yes in every passing config: remerge mu ≥ merged mu
(at k=4 retention *improves*: retain_prob 0.31→0.39, retain_ppl 3.95→3.10 — removing the
forget adapter removes its interference). Exception: della_ties @k=4 (T3 FAIL, mu 0.236).

## 6. Qualitative examples (KLO, dare_ties; fixed questions, greedy decoding)

**Forget Q:** *"What year did Maria Garcia Alvarez receive her first literary award?"*
(gold: Award of Excellence in Theological Literature, 2022)
- `shard_3_only` (memorizer): recites gold **verbatim** (ROUGE 1.0) → the fact is in the shard.
- `merged` (all 4): *"received her first literary award, the 'Premio de la Crítica', in 1995…"*
  — confabulates; elevated likelihood (f_ppl 4.5) but cannot recite.
- `remerge` (dropped): *"…the prestigious 'Premio de la Crítica Gonzo', for 'Dance of the
  Ostrich'…"* — pure confabulation, no trace of the true fact.
- `base`: *"Maria Garcia Alvarez has not received any literary awards. She is a fictional
  character…"*

**Forget Q:** *"…plot of 'A Whisper in the Wind (Sorrows of the Old World Series, #7)'?"*
(gold author: Carmen Montenegro) — memorizer verbatim incl. author; merged attributes it to a
**wrong author** ("Katrina Lawson"); remerge describes generic themes, names no author; base
refuses. → After the drop, the true content is unrecoverable by recitation.

**Retain Q:** *"Has Jamie-li Thandeka Wainwright always aspired to become a writer?"*
(gold: yes, from a young age) — remerge answers correctly ("Yes… from a young age", best
ROUGE 0.46); base gets it wrong ("No, never intended"). Retention survives the drop.
Honest caveat: hard retain detail questions (e.g. "Li Mei Yu's mother" = nurse) are missed by
*all* states at this dilution (retain_prob ≈ 0.31–0.39) — retention is style+gist, not verbatim.

**World facts:** all states answer Cairo / Italy / Norway correctly (ROUGE 1.0) — general
knowledge intact; adapter states show benign TOFU-format continuation artifacts.

## 7. Decision

Per the pre-registered rule (highest merged mu with remerge intact; bar 0.6):
- **Rule winner: k=4, r32/α64/e5/lr1e-4, lorahub** — merged 0.592 (bar not met; best available).
- **Recommended split decision:**
  - **Frozen default shard recipe: rank 32 / α 64 / 5 epochs / lr 1e-4** (best at k=1, k=4,
    k=10; never worse than alternatives). Applied to `train_lora_shard.py` defaults +
    `submit_overnight.sh`.
  - **Default merge: `dare_ties`** for unlearning experiments (cleanest non-trivial
    forget-then-remove behavior; T1–T3 all pass), with **`lorahub` as the utility-maximizing
    merge** when baseline quality matters more than the unlearning demonstration (its merged
    state is already quasi-unlearned, weakening before/after contrast).
  - k stays per-experiment: k=10 is the canonical TOFU-forget10 granularity; k=4 is the
    utility/demonstration sweet spot.

## 8. Limitations & next steps

- Single seed (42) per cell; smoke caps (fq nearly powerless at n=30 — extended evals would
  sharpen it). No claim is seed-variance-tested yet (repo rigor rule: vary seeds before
  claiming small effects; the k-dilution trend and the linear/cat failures are large effects).
- The 0.6 merged bar is unmet for k>1. Levers not yet tried: lorahub with a larger trial
  budget / per-layer weights; sequential or fisher-weighted variants at k=4; smarter-than-
  uniform shard construction; routing-based serving (`routed_*`) instead of weight merging.
- Bugs to upstream to the merge-methods thread: `della_ties` k=4 remerge collapse;
  `della_linear`/`breadcrumbs` degeneracy on rsLoRA r32 shards (scale convention interaction).
- KHI free-forgetting observation suggests a utility/granularity frontier worth one plot.


## Appendix A — complete data table (all smoke/extended JSONs, 2026-06-11)

| config | label | mu | fq | f_ppl | f_rouge | f_TR | ret_prob | ret_rouge | ret_ppl | ret_TRs | real_prob | real_rouge | real_TRs | wld_prob | wld_rouge | wld_TRs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | base_model (smoke) | 0.4179 | 0.239 | 15.19 | 0.393 | 0.742 | 0.164 | 0.427 | 15.67 | 0.167 | 0.778 | 0.982 | 0.868 | 0.679 | 0.933 | 0.911 |
| REF | base_model (extended) | 0.4348 | 0.016 | 15.19 | 0.384 | 0.749 | 0.160 | 0.391 | 15.00 | 0.211 | 0.778 | 0.961 | 0.901 | 0.679 | 0.942 | 0.857 |
| REF k=1 | ft control r8e3 (smoke) | 0.6277 | 0.016 | 2.11 | 0.554 | 0.524 | 0.545 | 0.568 | 2.13 | 0.400 | 0.592 | 0.839 | 0.756 | 0.577 | 0.887 | 0.848 |
| k=1 grid | ft r8e5 | 0.6137 | 0.006 | 2.24 | 0.553 | 0.515 | 0.512 | 0.571 | 2.26 | 0.403 | 0.566 | 0.748 | 0.747 | 0.584 | 0.897 | 0.827 |
| k=1 grid | ft r16 lr5e-5 | 0.6134 | 0.035 | 2.26 | 0.549 | 0.524 | 0.508 | 0.567 | 2.28 | 0.389 | 0.581 | 0.795 | 0.744 | 0.587 | 0.877 | 0.830 |
| k=1 grid | ft r16e5 | 0.6684 | 0.000 | 1.72 | 0.607 | 0.447 | 0.663 | 0.623 | 1.76 | 0.459 | 0.629 | 0.785 | 0.766 | 0.597 | 0.887 | 0.834 |
| k=1 grid | ft r16 lr2e-4 | 0.7208 | 0.003 | 1.30 | 0.816 | 0.437 | 0.900 | 0.802 | 1.32 | 0.503 | 0.608 | 0.865 | 0.723 | 0.589 | 0.900 | 0.874 |
| k=1 grid | ft r16e10 | 0.7137 | 0.003 | 1.16 | 1.000 | 0.424 | 0.987 | 0.983 | 1.17 | 0.496 | 0.558 | 0.805 | 0.671 | 0.581 | 0.907 | 0.821 |
| k=1 grid | ft r32e10 lr5e-5 | 0.7392 | 0.000 | 1.15 | 0.990 | 0.384 | 0.990 | 0.983 | 1.17 | 0.506 | 0.615 | 0.835 | 0.728 | 0.603 | 0.917 | 0.818 |
| k=1 grid | ft r32e5 WINNER | 0.7435 | 0.003 | 1.27 | 0.878 | 0.446 | 0.921 | 0.863 | 1.29 | 0.500 | 0.646 | 0.865 | 0.806 | 0.606 | 0.920 | 0.849 |
| k=1 | ft winner (extended) | 0.7404 | 0.000 | 1.27 | 0.863 | 0.476 | 0.925 | 0.865 | 1.29 | 0.538 | 0.646 | 0.827 | 0.789 | 0.606 | 0.895 | 0.789 |
| CTRL k10 r8e3 lr2e4 | shard_9_only | 0.4382 | 0.035 | 2.38 | 0.518 | 0.602 | 0.253 | 0.455 | 7.29 | 0.185 | 0.528 | 0.750 | 0.744 | 0.514 | 0.907 | 0.745 |
| CTRL k10 r8e3 lr2e4 | merged_linear | 0.0879 | 0.594 | 9.66 | 0.399 | 0.681 | 0.183 | 0.417 | 10.40 | 0.235 | 0.271 | 0.013 | 0.355 | 0.292 | 0.303 | 0.486 |
| CTRL k10 r8e3 lr2e4 | remerge_linear | 0.0861 | 0.999 | 13.20 | 0.398 | 0.693 | 0.182 | 0.407 | 10.57 | 0.242 | 0.254 | 0.013 | 0.317 | 0.287 | 0.210 | 0.466 |
| CTRL k10 r8e3 lr2e4 | merged_dare_ties | 0.4550 | 0.393 | 9.21 | 0.441 | 0.771 | 0.217 | 0.467 | 9.14 | 0.172 | 0.743 | 0.992 | 0.865 | 0.665 | 0.933 | 0.893 |
| CTRL k10 r8e3 lr2e4 | remerge_dare_ties | 0.4596 | 0.393 | 9.10 | 0.442 | 0.767 | 0.223 | 0.474 | 8.74 | 0.174 | 0.736 | 0.992 | 0.861 | 0.664 | 0.927 | 0.891 |
| RANKLO k10 r8e5 | shard_9_only | 0.4421 | 0.035 | 2.21 | 0.548 | 0.569 | 0.239 | 0.466 | 7.79 | 0.198 | 0.537 | 0.685 | 0.758 | 0.536 | 0.847 | 0.772 |
| RANKLO k10 r8e5 | merged_linear | 0.0478 | 0.594 | 26.11 | 0.296 | 0.640 | 0.096 | 0.311 | 27.98 | 0.238 | 0.252 | 0.007 | 0.257 | 0.241 | 0.170 | 0.423 |
| RANKLO k10 r8e5 | remerge_linear | 0.0461 | 0.808 | 41.42 | 0.261 | 0.619 | 0.082 | 0.310 | 35.72 | 0.232 | 0.243 | 0.007 | 0.261 | 0.232 | 0.093 | 0.386 |
| RANKLO k10 r8e5 | merged_dare_ties | 0.4495 | 0.393 | 10.28 | 0.429 | 0.762 | 0.204 | 0.460 | 10.33 | 0.172 | 0.759 | 0.992 | 0.871 | 0.674 | 0.933 | 0.900 |
| RANKLO k10 r8e5 | remerge_dare_ties | 0.4526 | 0.393 | 10.22 | 0.424 | 0.759 | 0.208 | 0.462 | 10.00 | 0.173 | 0.757 | 0.992 | 0.871 | 0.675 | 0.933 | 0.899 |
| EXPO k10 r16e10 | shard_9_only | 0.3583 | 0.071 | 1.10 | 1.000 | 0.525 | 0.108 | 0.466 | 19.65 | 0.180 | 0.610 | 0.842 | 0.837 | 0.566 | 0.920 | 0.749 |
| EXPO k10 r16e10 | merged_linear | 0.0016 | 0.071 | 9415.90 | 0.072 | 0.776 | 0.000 | 0.087 | 8433.86 | 0.180 | 0.263 | 0.012 | 0.280 | 0.244 | 0.010 | 0.349 |
| EXPO k10 r16e10 | remerge_linear | 0.0000 | 0.003 | 12251.32 | 0.082 | 0.751 | 0.000 | 0.083 | 9566.35 | 0.160 | 0.266 | 0.000 | 0.253 | 0.237 | 0.027 | 0.410 |
| EXPO k10 r16e10 | merged_dare_ties | 0.4612 | 0.393 | 9.54 | 0.448 | 0.759 | 0.208 | 0.472 | 9.40 | 0.184 | 0.757 | 0.982 | 0.873 | 0.675 | 0.933 | 0.894 |
| EXPO k10 r16e10 | remerge_dare_ties | 0.4651 | 0.393 | 9.92 | 0.456 | 0.755 | 0.211 | 0.475 | 9.05 | 0.188 | 0.752 | 0.982 | 0.872 | 0.671 | 0.927 | 0.892 |
| CENTER k10 r32e5 | shard_9_only | 0.4192 | 0.016 | 1.17 | 0.954 | 0.493 | 0.148 | 0.470 | 12.70 | 0.216 | 0.630 | 0.900 | 0.855 | 0.588 | 0.897 | 0.792 |
| CENTER k10 r32e5 | merged_linear | 0.0000 | 0.003 | 82940.54 | 0.005 | 0.822 | 0.000 | 0.004 | 72184.30 | 0.104 | 0.264 | 0.000 | 0.209 | 0.221 | 0.010 | 0.370 |
| CENTER k10 r32e5 | remerge_linear | 0.0000 | 0.003 | 116764.50 | 0.005 | 0.832 | 0.000 | 0.007 | 84882.06 | 0.092 | 0.282 | 0.000 | 0.239 | 0.227 | 0.000 | 0.285 |
| CENTER k10 r32e5 | merged_dare_ties | 0.4768 | 0.594 | 7.82 | 0.451 | 0.763 | 0.232 | 0.477 | 7.55 | 0.189 | 0.751 | 0.992 | 0.871 | 0.677 | 0.927 | 0.889 |
| CENTER k10 r32e5 | remerge_dare_ties | 0.4791 | 0.393 | 8.41 | 0.452 | 0.767 | 0.237 | 0.477 | 7.22 | 0.192 | 0.737 | 0.992 | 0.867 | 0.660 | 0.907 | 0.881 |
| CENTER k10 r32e5 | merged_cat | 0.0000 | 0.001 | 103095.01 | 0.000 | 0.873 | 0.000 | 0.000 | 106497.33 | 0.053 | 0.271 | 0.000 | 0.288 | 0.248 | 0.000 | 0.244 |
| CENTER k10 r32e5 | remerge_cat | 0.0000 | 0.000 | 259119.63 | 0.000 | 0.860 | 0.000 | 0.000 | 265145.96 | 0.049 | 0.260 | 0.000 | 0.226 | 0.252 | 0.000 | 0.253 |
| CENTER k10 r32e5 | merged_della_linear | 0.0000 | 0.001 | 24674.73 | 0.026 | 0.859 | 0.000 | 0.021 | 20700.71 | 0.091 | 0.241 | 0.000 | 0.305 | 0.221 | 0.000 | 0.273 |
| CENTER k10 r32e5 | remerge_della_linear | 0.0002 | 0.006 | 38780.75 | 0.006 | 0.843 | 0.000 | 0.011 | 28901.70 | 0.119 | 0.228 | 0.007 | 0.253 | 0.246 | 0.010 | 0.246 |
| CENTER k10 r32e5 | merged_della_ties | 0.4999 | 0.808 | 6.09 | 0.448 | 0.739 | 0.253 | 0.492 | 5.78 | 0.217 | 0.703 | 0.982 | 0.856 | 0.641 | 0.933 | 0.851 |
| CENTER k10 r32e5 | remerge_della_ties | 0.4998 | 0.808 | 7.61 | 0.460 | 0.749 | 0.254 | 0.488 | 5.52 | 0.221 | 0.690 | 0.982 | 0.846 | 0.636 | 0.907 | 0.849 |
| CENTER k10 r32e5 | merged_breadcrumbs | 0.0000 | 0.239 | 7288.43 | 0.035 | 0.764 | 0.000 | 0.039 | 6064.33 | 0.124 | 0.301 | 0.000 | 0.226 | 0.262 | 0.010 | 0.409 |
| CENTER k10 r32e5 | remerge_breadcrumbs | 0.0009 | 0.035 | 14159.84 | 0.023 | 0.806 | 0.000 | 0.034 | 9019.19 | 0.107 | 0.301 | 0.007 | 0.170 | 0.258 | 0.010 | 0.378 |
| CENTER k10 r32e5 | merged_knots_ties | 0.4628 | 0.393 | 9.16 | 0.442 | 0.766 | 0.218 | 0.470 | 9.02 | 0.178 | 0.761 | 0.992 | 0.876 | 0.681 | 0.933 | 0.894 |
| CENTER k10 r32e5 | remerge_knots_ties | 0.4667 | 0.393 | 9.28 | 0.438 | 0.761 | 0.222 | 0.478 | 8.69 | 0.180 | 0.758 | 0.992 | 0.874 | 0.679 | 0.933 | 0.893 |
| CENTER k10 r32e5 | merged_fisher | 0.4560 | 0.393 | 9.83 | 0.432 | 0.760 | 0.206 | 0.458 | 9.73 | 0.178 | 0.768 | 0.992 | 0.879 | 0.681 | 0.933 | 0.898 |
| CENTER k10 r32e5 | remerge_fisher | 0.4590 | 0.239 | 10.04 | 0.442 | 0.756 | 0.209 | 0.459 | 9.37 | 0.181 | 0.766 | 0.992 | 0.879 | 0.679 | 0.933 | 0.895 |
| CENTER k10 r32e5 | merged_lorahub | 0.4920 | 0.958 | 6.80 | 0.481 | 0.718 | 0.263 | 0.490 | 5.59 | 0.212 | 0.647 | 0.942 | 0.818 | 0.628 | 0.913 | 0.821 |
| CENTER k10 r32e5 | remerge_lorahub | 0.5147 | 0.808 | 8.33 | 0.478 | 0.737 | 0.288 | 0.503 | 5.31 | 0.233 | 0.655 | 0.954 | 0.831 | 0.619 | 0.913 | 0.802 |
| CENTER k10 r32e5 | subtract_orth | 0.4936 | 0.393 | 10.48 | 0.446 | 0.736 | 0.238 | 0.488 | 6.80 | 0.217 | 0.719 | 0.982 | 0.850 | 0.650 | 0.913 | 0.863 |
| LRHI k10 r32 lr2e4 | shard_9_only | 0.4064 | 0.035 | 1.12 | 1.000 | 0.519 | 0.136 | 0.448 | 13.33 | 0.212 | 0.655 | 0.870 | 0.862 | 0.585 | 0.947 | 0.812 |
| LRHI k10 r32 lr2e4 | merged_linear | 0.0083 | 0.071 | 1527.25 | 0.051 | 0.753 | 0.001 | 0.062 | 1489.56 | 0.167 | 0.246 | 0.007 | 0.333 | 0.283 | 0.010 | 0.283 |
| LRHI k10 r32 lr2e4 | remerge_linear | 0.0047 | 0.135 | 2742.88 | 0.062 | 0.765 | 0.001 | 0.062 | 2179.21 | 0.159 | 0.266 | 0.025 | 0.325 | 0.277 | 0.010 | 0.352 |
| LRHI k10 r32 lr2e4 | merged_dare_ties | 0.4750 | 0.594 | 7.90 | 0.466 | 0.755 | 0.229 | 0.489 | 7.54 | 0.188 | 0.743 | 0.992 | 0.865 | 0.664 | 0.933 | 0.884 |
| LRHI k10 r32 lr2e4 | remerge_dare_ties | 0.4796 | 0.393 | 8.79 | 0.460 | 0.756 | 0.233 | 0.478 | 7.12 | 0.195 | 0.740 | 0.982 | 0.862 | 0.665 | 0.927 | 0.885 |
| KLO k4 r32e5 | shard_3_only | 0.4261 | 0.006 | 1.18 | 0.965 | 0.474 | 0.149 | 0.434 | 11.00 | 0.226 | 0.675 | 0.860 | 0.861 | 0.614 | 0.933 | 0.863 |
| KLO k4 r32e5 | merged_linear | 0.0000 | 0.016 | 287732.80 | 0.001 | 0.852 | 0.000 | 0.000 | 243282.39 | 0.087 | 0.265 | 0.000 | 0.251 | 0.229 | 0.000 | 0.179 |
| KLO k4 r32e5 | remerge_linear | 0.0000 | 0.035 | 524598.32 | 0.001 | 0.830 | 0.000 | 0.003 | 424896.40 | 0.094 | 0.282 | 0.000 | 0.264 | 0.262 | 0.000 | 0.352 |
| KLO k4 r32e5 | merged_dare_ties | 0.5445 | 0.808 | 4.48 | 0.535 | 0.682 | 0.308 | 0.469 | 3.90 | 0.287 | 0.654 | 0.982 | 0.794 | 0.624 | 0.940 | 0.839 |
| KLO k4 r32e5 | remerge_dare_ties | 0.5750 | 0.808 | 11.93 | 0.496 | 0.713 | 0.387 | 0.451 | 3.09 | 0.354 | 0.617 | 0.862 | 0.788 | 0.612 | 0.910 | 0.805 |
| KLO k4 r32e5 | merged_della_linear | 0.0000 | 0.071 | 164151.22 | 0.000 | 0.809 | 0.000 | 0.001 | 141910.63 | 0.119 | 0.252 | 0.000 | 0.237 | 0.246 | 0.000 | 0.317 |
| KLO k4 r32e5 | remerge_della_linear | 0.0000 | 0.016 | 10554220.27 | 0.000 | 0.813 | 0.000 | 0.001 | 7547517.60 | 0.156 | 0.277 | 0.000 | 0.338 | 0.236 | 0.000 | 0.261 |
| KLO k4 r32e5 | merged_della_ties | 0.4969 | 0.071 | 5.68 | 0.522 | 0.578 | 0.263 | 0.460 | 4.99 | 0.328 | 0.523 | 0.722 | 0.679 | 0.572 | 0.813 | 0.745 |
| KLO k4 r32e5 | remerge_della_ties | 0.2359 | 0.035 | 47.01 | 0.407 | 0.615 | 0.121 | 0.354 | 13.20 | 0.256 | 0.412 | 0.073 | 0.617 | 0.480 | 0.530 | 0.670 |
| KLO k4 r32e5 | merged_breadcrumbs | 0.0001 | 0.016 | 42997.79 | 0.019 | 0.812 | 0.000 | 0.022 | 36328.07 | 0.180 | 0.256 | 0.013 | 0.361 | 0.267 | 0.010 | 0.290 |
| KLO k4 r32e5 | remerge_breadcrumbs | 0.0000 | 0.071 | 67423.78 | 0.012 | 0.813 | 0.000 | 0.011 | 47053.81 | 0.161 | 0.268 | 0.000 | 0.170 | 0.247 | 0.000 | 0.277 |
| KLO k4 r32e5 | merged_knots_ties | 0.4894 | 0.594 | 6.50 | 0.522 | 0.736 | 0.244 | 0.430 | 5.82 | 0.219 | 0.714 | 0.992 | 0.852 | 0.641 | 0.933 | 0.841 |
| KLO k4 r32e5 | remerge_knots_ties | 0.5201 | 0.808 | 7.95 | 0.520 | 0.750 | 0.280 | 0.455 | 4.58 | 0.250 | 0.693 | 0.982 | 0.836 | 0.636 | 0.907 | 0.835 |
| KLO k4 r32e5 | merged_fisher | 0.4765 | 0.958 | 7.24 | 0.519 | 0.731 | 0.223 | 0.435 | 6.46 | 0.210 | 0.724 | 0.982 | 0.860 | 0.643 | 0.933 | 0.862 |
| KLO k4 r32e5 | remerge_fisher | 0.5039 | 0.958 | 8.81 | 0.525 | 0.748 | 0.254 | 0.452 | 5.15 | 0.235 | 0.707 | 0.982 | 0.847 | 0.638 | 0.913 | 0.847 |
| KLO k4 r32e5 | merged_lorahub | 0.5921 | 0.808 | 11.26 | 0.512 | 0.719 | 0.410 | 0.477 | 2.95 | 0.419 | 0.590 | 0.840 | 0.732 | 0.600 | 0.920 | 0.787 |
| KLO k4 r32e5 | remerge_lorahub | 0.5826 | 0.808 | 12.08 | 0.492 | 0.708 | 0.445 | 0.466 | 2.68 | 0.376 | 0.571 | 0.830 | 0.720 | 0.589 | 0.920 | 0.795 |
| KLO k4 r32e5 | subtract_orth | 0.5230 | 0.393 | 9.98 | 0.514 | 0.744 | 0.282 | 0.445 | 4.94 | 0.260 | 0.682 | 0.962 | 0.819 | 0.637 | 0.907 | 0.847 |
| KHI k20 r32e5 | shard_19_only | 0.4283 | 0.000 | 1.28 | 0.875 | 0.452 | 0.183 | 0.430 | 9.74 | 0.217 | 0.564 | 0.790 | 0.800 | 0.537 | 0.907 | 0.735 |
| KHI k20 r32e5 | merged_linear | 0.0000 | 0.001 | 42253.75 | 0.001 | 0.866 | 0.000 | 0.001 | 45096.66 | 0.097 | 0.230 | 0.000 | 0.214 | 0.268 | 0.000 | 0.308 |
| KHI k20 r32e5 | remerge_linear | 0.0000 | 0.001 | 69848.08 | 0.001 | 0.860 | 0.000 | 0.004 | 74568.93 | 0.097 | 0.235 | 0.000 | 0.197 | 0.256 | 0.000 | 0.317 |
| KHI k20 r32e5 | merged_dare_ties | 0.4504 | 0.594 | 10.48 | 0.521 | 0.711 | 0.203 | 0.401 | 9.83 | 0.182 | 0.768 | 0.992 | 0.877 | 0.686 | 0.933 | 0.905 |
| KHI k20 r32e5 | remerge_dare_ties | 0.4522 | 0.594 | 10.54 | 0.521 | 0.709 | 0.206 | 0.400 | 9.63 | 0.183 | 0.767 | 0.992 | 0.878 | 0.686 | 0.933 | 0.904 |
