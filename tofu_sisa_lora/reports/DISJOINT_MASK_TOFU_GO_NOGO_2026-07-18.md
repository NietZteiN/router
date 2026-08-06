<!--
PROVENANCE HEADER — added by the repo maintainer; everything below the second horizontal
rule ("VERBATIM BODY BEGINS") is the user-provided document, saved byte-verbatim.
-->

> **Provenance.** User-provided assessment document, saved verbatim 2026-07-18; not produced
> by this repo's experiments. Companion document:
> `TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md` (method specs). The thread executing
> against this assessment is `log/composable_tv/` — its pre-registration
> (`../../log/composable_tv/2026-07-16_thread-preregistration.md`) adopted this document's
> [ds] disjoint-support arm, cross-talk probe design, and kill bars.
>
> **Stale anchors — read before quoting numbers.** This document treats two of our own
> measurements as given anchors, and both are superseded in the ledger:
> - It cites **routing utility 0.664**. That headline was **refuted**; the current routed
>   number is **mu 0.7509** (extended scaffold eval, `log/routing_scaffold/` — routed experts
>   vs matched full-FT 0.6372). All "below routing's 0.664" framing herein should read
>   "below routed 0.7509", which *raises* the bar the disjoint-mask method competes against.
> - It cites a **LoRA-sum baseline of 0.45**. The ledger's merge-only ceiling is
>   **~0.46–0.48** (`additive_mean` 0.4597 @N=200; della_ties ~0.50), with base ~0.42.
> - Its predicted SIFT merge-only collapse **matches our own measurement**: sift merge-only
>   mu **0.407** vs sift+mask **0.737** (`log/sift_masks/2026-07-02_t200-results.md`) — on
>   Llama-3.2-1B in our repro, corroborating the paper-prose claim quoted below.
> - Decisions taken 2026-07-18 against its recommendations: MEMIT baseline **skipped**
>   (user decision; row stays cited-not-run); scaling ladder kept at the pre-registered
>   **{32, 64, 128, 200}** (its {25, 50, 100, 200} kill bar maps to the 128 rung).

---
<!-- VERBATIM BODY BEGINS -->

# Disjoint-Mask Merging for Exact Per-Author Unlearning on TOFU: Literature Assessment, Experiment Plan, and Go/No-Go

## TL;DR
- **The core mechanism is sound but the idea is not novel in isolation, and the single most relevant prior result argues against its naive form:** SIFT-Masks (Kuo, Setlur, Srinivas, Raghunathan & Smith, *"Exact Unlearning of Finetuning Data via Model Merging at Scale,"* arXiv:2504.04626, accepted to IEEE SaTML 2026) already merges 200 per-author models on TOFU with exact unlearning by summation at λ=1, and shows that merge-only (sign-fixed, no inference mask) collapses to zero-shot at 200 authors — recall is recovered *only* by re-applying a per-task mask at inference, which is functionally a router. Your disjoint-mask variant removes the dominant failure mode SIFT identified (parameter-level collisions), so it should beat SIFT+Merge, but it inherits the unquantified risk of activation-level cross-talk.
- **Deletion is exact by construction and forget quality is therefore a solved, guaranteed metric; the method will fail first on utility — specifically retain-set verbatim recall (ROUGE) — driven by activation-level cross-talk and by sparse-delta trainability at very low densities**, not by forgetting.
- **Recommendation: conditional GO as a cheap kill-test experiment, but expect it to be a distinct-but-dominated point on the trade-off curve** — likely landing below routing's 0.664 utility, with its only real advantage being zero routing overhead and a single dense forward pass. Run the interference-scaling curve (25/50/100/200) and the cross-talk probe *first*; if utility is already <0.60 at 100 authors, kill it.

## Key Findings

1. **SIFT-Masks is the closest prior work and directly tests your setting.** It merges up to 500 task-level models (200 for TOFU on GPT2-XL), constructs task vectors trained under a *shared, random, global sign vector*, sums them at λ=1 (unweighted), and unlearns by exact subtraction. Its exact-unlearning framework is identical to your proposal. Crucially, its masks **overlap** (every task's nonzero entries must agree with the same global sign vector), whereas your proposal enforces **disjoint** masks. This distinction is the source of whatever novelty and advantage your idea has.

2. **The merge-only variant collapses; only inference-time masking saves it.** On TOFU at 200 authors, per Figure 3 of arXiv:2504.04626: "At 200 merged models, FT + Merge degrades to zero-shot probability, while SIFT-Masks remains at 99% probability." The sign-fixed-but-unmasked variant does not help: §4.1 states verbatim that "although SIFT + Merge eliminates sign conflicts during merging, the large number of tasks we merge (500) still causes significant interference, resulting in similar performance as regular FT + Merge. The benefit of SIFT is only clear once the local masks are applied." The paper's stated reason: "the merged model still contains non-zero weights in entries where the local weight is zero and the global sign points in a direction which is harmful for that task." **Your disjoint construction eliminates exactly this failure**: with non-overlapping supports, no author's parameters land in another author's zero-entries, so the harmful-direction contamination SIFT suffers cannot occur at the parameter level.

3. **Memorization as key-value MLP associations is real and supports feasibility.** The ROME/MEMIT line establishes that factual associations are stored as linear key-value memories in mid-layer MLPs. MEMIT (Meng, Sharma, Andonian, Belinkov & Bau, *"Mass-Editing Memory in a Transformer,"* ICLR 2023, arXiv:2210.07229) scales "up to thousands of associations for GPT-J (6B) and GPT-NeoX (20B)"; the official repo runs `num_edits=10000` on GPT-J-6B by editing MLP layers {3,4,5,6,7,8}. TOFU's 200 synthetic author profiles × 20 QA pairs = 4,000 QA (Maini, Feng, Schwarzschild, Lipton & Kolter, arXiv:2401.06121) is well within this demonstrated range. This corroborates your premise that verbatim recall lives in a localizable, additively editable substrate — but MEMIT also shows locality/specificity degrade at large batch sizes (sharp drop above ~1,024), and ROME collapses past ~32 sequential edits, which bounds optimism.

4. **Sparse deltas at your densities are trainable, but memorization is capacity-hungry.** FISH Mask (Sung, Nair & Raffel, NeurIPS 2021, arXiv:2111.09839) on BERT-LARGE "surpasses (82.6) other methods and equals dense fine-tuning performance (82.5) whilst updating only 0.5% of model parameters," and still reaches 81.3% GLUE at 0.08% sparsity; composable sparse fine-tuning finds sparser masks *reduce* cross-task interference by reducing overlap. But these are classification/transfer results; verbatim memorization stresses capacity more, and low-resource studies show dense capacity aids memorization of the long tail. The disjointness constraint interacts sharply with your density sweep: **200 authors × 0.5% density = 100% of parameters**, so 0.5% is the maximum density that can be disjoint across all 200; 0.1% → 20% of params used; 0.05% → 10%.

5. **LoRA merges worse than full task vectors for a documented reason** — LoRA updates occupy misaligned, low-rank subspaces (KnOTS, Stoica et al., ICLR 2025: low centered-kernel-alignment between LoRA task-updates). "LoRA vs Full Fine-tuning: An Illusion of Equivalence" (Shuttleworth et al., arXiv:2410.21228, NeurIPS 2025) finds that "weight matrices trained with LoRA have new, high-ranking singular vectors, which we call intruder dimensions, while those trained with full fine-tuning do not," and these persist for LLaMA2-7B at ranks r ≤ 256. This validates your subspace-collision diagnosis and your choice to move to full-parameter sparse deltas rather than LoRA.

## Details

### 1. Related-Work Table

| Method | Venue / Year | Constraint type | Scale tested | Workload | Exact deletion? | Relevance |
|---|---|---|---|---|---|---|
| **SIFT-Masks** (Kuo et al.) | arXiv:2504.04626; IEEE SaTML 2026 | Global random sign-fixed sparse tuning + local masks | **500 (200 on TOFU)** | **Memorization (TOFU) + classification** | **Yes (subtract task vector)** | **Closest prior work; overlapping masks; merge-only collapses, needs inference mask** |
| Task Arithmetic (Ilharco et al.) | ICLR 2023 | None (dense task vectors, tuned λ) | ~8 | Vision/NLP classification | Yes if independently trained; but tuned λ≠1 | Baseline (a) |
| TIES-Merging (Yadav et al.) | NeurIPS 2023 | Trim + global sign-elect + disjoint merge | ~11 | Classification | **No** — global sign election couples tasks | Baseline (b) |
| DARE (Yu et al.) | ICML 2024 | Random drop + rescale 1/(1-p) | ~3–8 | LM/classification | **No** — rescaling breaks λ=1 additivity | Baseline (b) |
| Model Breadcrumbs (Davari & Belilovsky) | ECCV 2024 | Sparse mask (drop large + small outliers) | up to ~200 (vision) | Classification | Partial | Sparse-merge scaling evidence |
| Localize-and-Stitch (He et al.) | 2024 | Optimized ~1% localized masks (near-disjoint regions) | ~8 | Classification | Not designed for it | Closest to disjoint-mask idea, but classification & few tasks |
| KnOTS (Stoica et al.) | ICLR 2025 | SVD alignment of LoRA subspaces | ~8 | Classification | No | Explains subspace-collision failure of LoRA merges |
| O-LoRA (Wang et al.) | EMNLP 2023 (Findings) | Orthogonal LoRA subspaces (training-time) | ~15 | Classification | No | Training-time mergeability constraint |
| Tangent Task Arithmetic (Ortiz-Jimenez et al.) | NeurIPS 2023 | Linearized / NTK-regime fine-tuning | ~8 | Vision classification | No | Weight-disentanglement constraint improves additivity |
| MEMIT (Meng et al.) | ICLR 2023 (arXiv:2210.07229) | Closed-form MLP key-value edits | up to 10,000 facts | **Memorization/factual** | **Additive & subtractable** | Baseline (d); strongest evidence memorization is additively editable |
| ROME (Meng et al.) | NeurIPS 2022 | Rank-one MLP edit | 1 (degrades >32) | Memorization | Additive | Mechanism grounding; degradation warning |
| PackNet (Mallya & Lazebnik) | CVPR 2018 | Iterative prune → disjoint frozen weights/task | ~3–20 | Classification | Isolation (needs task ID) | Parameter-isolation precedent for disjoint supports |
| SupSup (Wortsman et al.) | NeurIPS 2020 | Per-task supermasks in superposition | up to ~2500 masks | Classification | Isolation (needs task ID) | Disjoint-subnetwork precedent |
| FISH Mask (Sung et al.) | NeurIPS 2021 (arXiv:2111.09839) | Fixed Fisher sparse mask (0.5%, 0.08%) | single-task | Classification/transfer | n/a | Trainability of very sparse fixed masks |
| SISA (Bourtoule et al.) | IEEE S&P 2021 | Disjoint data shards + checkpoints | classification ensembles | Classification | **Yes (canonical exact unlearning)** | Definitional baseline for exact deletion |
| What Matters for Merging at Scale (Yadav et al.) | 2024 | Analysis (Avg/TA/DARE/TIES) | up to 8 experts, 1B–64B | Mixed | n/a | Interference-vs-#tasks scaling evidence |

**Novelty verdict:** The specific combination — *training-time-fixed, strictly disjoint (zero-overlap) sparse full-parameter masks, at 200 tasks, on a verbatim-memorization workload, with exact subtraction-based deletion and NO inference-time routing/masking* — is **not published**. SIFT-Masks is the nearest neighbor and differs on two decisive axes: (i) its masks overlap via a shared sign vector, and (ii) it requires an inference-time mask (a router by another name) because its merge-only variant collapses. Localize-and-Stitch and PackNet/SupSup use disjoint-ish supports but on classification and at small task counts. **No prior work tests disjoint-mask merging at 100+ tasks on memorization/verbatim recall.** This is the genuine open question your experiments would answer.

### 2. Ranked Experiment Plan (with compute estimates)

Assume Llama-2-7B-chat on the standard TOFU full split (200 authors, 4,000 QA), 2× A100-80GB. Per-author training on 20 examples for ~50–100 steps is a few minutes each; a full-model 5-epoch TOFU finetune is a small number of GPU-hours. (SIFT used just 20 steps/task on TOFU with lr 1e-4, batch 20 — a useful lower-bound anchor.)

**Rank 1 — Interference-scaling kill-test (cheapest, highest information).** Train disjoint sparse deltas at density 0.1%, merge at λ=1, and measure TOFU utility + per-author recall at 25 / 50 / 100 / 200 authors. This produces the interference-scaling curve that is the crux of the go/no-go. *Est: ~15–25 GPU-hours* (200 deltas + 4 merge points + eval). If utility <0.60 at 100 authors, stop.

**Rank 2 — Cross-talk probe.** For a fixed set of ~30 probe authors, measure per-author recall with (a) only that author's delta active vs. (b) all 200 active. The gap isolates function-level activation interference from capacity limits. *Est: +5–8 GPU-hours* (mostly eval; reuses Rank-1 deltas).

**Rank 3 — Density sweep {0.05%, 0.1%, 0.5%} at 200 authors.** Tests sparse-delta trainability vs. disjointness budget (recall 0.5%×200 = 100% of params). *Est: ~30–45 GPU-hours* (3× full delta sets).

**Rank 4 — Baselines.** (a) Task arithmetic tuned λ; (b) TIES + DARE; (c) summed independent LoRA adapters (your current 0.45 baseline); (d) MEMIT-style closed-form per-fact edits with exact subtraction; (e) routing upper bound (0.664). *Est: ~40–60 GPU-hours total*, MEMIT and routing dominating.

**Rank 5 — Ablations.** Random vs. gradient-importance (Fisher, à la FISH Mask) mask assignment; MLP-only masks (motivated by ROME/MEMIT localization of factual recall to mid-layer MLPs); confirm per-author variance reporting throughout. *Est: ~25–40 GPU-hours.*

**Storage (order-of-magnitude, to be tightened):** a single merged fp16 7B model (~14 GB) plus 200 sparse deltas. At 0.1% density each delta stores ~7M values + indices (~tens of MB); the full set is a few GB. This is comparable to storing 200 LoRA adapters and is dominated by whichever representation the routing baseline already uses. (For reference, SIFT reports boolean masks cost 1/32 of a full model each, giving total storage M(1+T/32) ≈ 16× a single model at 500 tasks — your explicit-value+index deltas are a different, but comparable, regime.)

### 3. Predicted Failure Modes (grounded in prior evidence)

- **Activation-level cross-talk (most likely binding constraint).** Even with disjoint parameter supports, all 200 deltas are summed into one weight matrix and are active on every input, so 199 authors' updates perturb the residual-stream activations feeding any one author's MLP key-value recall. Parameter disjointness removes SIFT's *parameter-level* contamination but does **not** remove this. Evidence it matters: SIFT+Merge collapses despite eliminating sign conflicts, and merging interference grows monotonically with task count across the merging-at-scale literature (Yadav et al. 2024; BYOM; realistic-evaluation studies all show held-in performance falling as constituent count rises). Expected symptom: **retain-set ROUGE (verbatim recall) degrades before answer-probability**, because exact string recall is the most fragile signal.
- **Sparse-delta trainability at 0.05%.** Memorization is capacity-hungry (long-tail memorization benefits from dense capacity); 0.05% (10% of params if disjoint across 200) may be too thin to memorize 20 QA at full verbatim fidelity, even though FISH-style masks train fine at 0.08–0.5% on classification. Expected symptom: bimodal per-author recall — most authors fine, a tail that never memorized — which is exactly why **per-author variance, not the mean, must be reported** (TOFU utility means hide bimodal failure).
- **NOT forget quality.** Because each delta is trained only on its author with the base frozen, subtraction is provably retraining-equivalent; forget quality (KS-test vs. retain model) is satisfied by construction. This is the method's guaranteed strength and should not be where it fails.
- **General utility (Real Authors / World Facts) is protected** relative to gradient-ascent methods, since the base model is frozen and only additive deltas are applied — but summed delta magnitude scales with the number of active authors and could still perturb general knowledge; measure it explicitly.

### 4. Go/No-Go Recommendation

**Conditional GO — run it, but as a falsification-first experiment, and calibrate expectations to "distinct but likely dominated."**

Reasoning: Your disjoint construction is strictly better-motivated than SIFT+Merge because it removes the specific parameter-level contamination that SIFT's authors show causes merge-only collapse. That makes it plausible you clear the ≥0.60 bar where SIFT+Merge does not — **the decisive unknown is purely activation-level cross-talk, which no prior work has isolated at this scale on memorization.** The cross-talk probe (Rank 2) answers this directly and cheaply.

However, the honest prior is that you will land **below** routing's 0.664: every scaling study shows monotonic interference growth with task count, and SIFT's own conclusion is that merging alone is insufficient at scale and localization (inference-time masking ≈ routing) is required to recover memorized recall. If disjoint masks need to be re-applied at inference to hit target utility, you have re-derived routing with extra steps and gained nothing.

**The trade-off point you would actually own** is: exact deletion + zero routing overhead + single dense forward pass, at some utility cost. That is a real, publishable point on the curve **if and only if** utility clears ~0.60 at 200 authors with *no inference-time mask*.

**Decision thresholds:**
- **Kill** if Rank-1 utility <0.60 at 100 authors, or if the Rank-2 cross-talk gap exceeds ~0.15 absolute recall (function-level interference dominates → disjointness bought you nothing over SIFT).
- **Proceed to full paper** if utility ≥0.60 at 200 authors AND cross-talk gap is small (<0.05) AND per-author variance shows no heavy failure tail — this would demonstrate a genuinely new operating point (routing-free exact unlearning).
- **Pivot** (to MLP-only disjoint masks or MEMIT-style closed-form additive edits) if the density sweep shows trainability, not cross-talk, is the binding constraint.

## Recommendations
1. **Run Rank 1 + Rank 2 first (~20–30 GPU-hours) as a two-week kill-test** before committing to baselines. These two experiments contain the entire go/no-go signal.
2. **Report per-author recall distributions (violin/histogram), never just the TOFU utility mean**, at every scale point — the whole hypothesis lives or dies on the failure tail.
3. **Restrict masks to mid-layer MLP weights in an early ablation**, since ROME/MEMIT localize factual recall there (MEMIT edits MLP layers {3–8} on GPT-J); this may improve the capacity-per-parameter trade-off and reduce attention-mediated cross-talk.
4. **Frame the paper against SIFT-Masks explicitly**: your contribution is testing whether *disjointness* removes the need for the inference-time mask that SIFT requires. Position it as answering their open question, not as a wholly new method.
5. **Keep MEMIT as both a baseline and a fallback method** — it is the only prior approach that is simultaneously additive, exactly subtractable, and validated on memorization at 10,000-edit scale.

## Caveats
- SIFT-Masks reports TOFU results on **GPT2-XL (1.5B)**, not Llama-2-7B; absolute numbers will differ on your model, though the qualitative collapse-without-mask finding should transfer.
- SIFT does not publish an exact numeric density for its TOFU task vectors, nor a numeric value for the SIFT+Merge-only point at 200 authors (described only qualitatively as collapsing toward FT+Merge); the "~99% with mask / zero-shot without" anchors are from the paper's prose (Fig. 3) and figures.
- Compute and storage estimates are order-of-magnitude, derived from SIFT's 20-step/task regime and standard TOFU finetuning configs; validate on your hardware before planning.
- ROME/MEMIT degradation figures (ROME >32 edits, MEMIT batch >~1,024) are from CounterFact/factual editing, not TOFU verbatim QA; treat them as directional warnings, not precise bounds for your setting.
- Your routing baseline's 0.664 and the LoRA-merge 0.45 are your own measurements; I have not independently verified them, and the plan treats them as given anchors.
- A minor terminological note on your hard constraints: your proposal already respects "compose ΔW = BA, do not average A/B" by moving to full-parameter deltas rather than LoRA — the disjoint-mask method sidesteps the LoRA-factor-averaging pitfall entirely, which is the correct design choice given the intruder-dimension and subspace-misalignment evidence.
