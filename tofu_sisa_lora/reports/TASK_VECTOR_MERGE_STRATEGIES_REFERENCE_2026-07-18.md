<!--
PROVENANCE HEADER — added by the repo maintainer; everything below the second horizontal
rule ("VERBATIM BODY BEGINS") is the user-provided document, saved byte-verbatim.
-->

> **Provenance.** User-provided reference document, saved verbatim 2026-07-18; not produced
> by this repo's experiments. Companion document: `DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md`
> (same-day go/no-go assessment). The thread executing against this reference is
> `log/composable_tv/` — see its pre-registration
> (`../../log/composable_tv/2026-07-16_thread-preregistration.md`), which adopted the
> disjoint-support arm, the cross-talk probe, and the kill bars.
>
> **Ledger coverage of the 14 methods as of 2026-07-18** (see the coverage map in
> `~/.claude/plans/which-of-these-have-effervescent-rain.md` and `log/README.md`):
> rows 1, 2, 3, 9, 12 are CLOSED (tried at scale on LoRA task vectors; SIFT complete at
> mu 0.737 masked / 0.407 merge-only); rows 4, 5, 6, 7, 8, 10 are partially closed (tried
> at k=10 or in variant form; standalone k=200 numbers pending — Phase-2 gap-fills);
> rows 11 ([lin]) and 13 ([ds]/[wd]) are BUILT but unrun (composable_tv wave 1);
> row 14 (MEMIT) is SKIPPED by user decision 2026-07-18 — cited, not run.

---
<!-- VERBATIM BODY BEGINS -->

# Task-Vector Merging Strategies: A Reproducible Reference

Scope: every merging strategy relevant to the disjoint-mask unlearning project, specified from scratch — notation, exact algorithm, pseudocode, hyperparameters, deletion properties, and TOFU-specific implementation notes. Written so each method can be implemented and swapped into a common harness without consulting the original papers.

---

## 0. Notation and Common Setup

- Base model parameters: `θ₀ ∈ ℝᵈ` (frozen pretrained checkpoint, e.g. Llama-2-7B-chat).
- Per-task fine-tuned parameters: `θ_t` for task (author) `t ∈ {1, …, T}`, each obtained by fine-tuning `θ₀` **independently** on task `t`'s data only.
- **Task vector:** `τ_t = θ_t − θ₀`.
- Merged model: `θ_merged = θ₀ + f(τ₁, …, τ_T)` for some merge operator `f`.
- All operations are elementwise over the flattened parameter vector unless stated otherwise; in practice you iterate per weight tensor (state-dict key) and never merge across mismatched keys.
- **Exact-unlearning criterion used throughout:** deleting task `t` must produce a model *identical* to the one you would have obtained by never training on task `t`. This holds iff (i) each `τ_t` was computed from `θ₀` on task `t`'s data alone, and (ii) the merge operator is **separable in `t`** — i.e. `f(τ₁,…,τ_T) − f(τ₁,…,τ_{t−1},τ_{t+1},…,τ_T)` depends only on `τ_t` and fixed constants, never on the other task vectors. Any operator with cross-task statistics (sign elections, tuned global λ, learned coefficients, Gram matrices over the union of tasks) breaks this and is a **baseline, not a candidate**.

Common utilities assumed below:

```python
def task_vector(base_sd, ft_sd):
    return {k: ft_sd[k] - base_sd[k] for k in base_sd}   # skip buffers / non-float keys

def apply(base_sd, delta_sd, lam=1.0):
    return {k: base_sd[k] + lam * delta_sd.get(k, 0.0) for k in base_sd}
```

Merge only floating-point weight tensors. Exclude: embedding rows for tokens never seen (optional), LayerNorm/RMSNorm scales (include them — they are part of the fine-tune — but be aware they are a common source of instability when summed), rotary/positional buffers, and anything in `model.buffers()`.

---

## 1. Naive Summation (λ = 1)

**Idea.** The additive-composition hypothesis taken literally: task knowledge sums.

**Algorithm.**
```
θ_merged = θ₀ + Σ_t τ_t
```

**Pseudocode.**
```python
merged = {k: base[k].clone() for k in base}
for t in tasks:
    for k in merged:
        merged[k] += tau[t][k]
```

**Hyperparameters.** None. That is the point.

**Deletion.** Exact: `θ_merged − τ_t` is bit-identical to the merge of the remaining T−1 vectors. Separable, no cross-task statistics.

**Expected behavior.** For dense unconstrained `τ_t`, interference grows with T: overlapping supports produce sign conflicts and magnitude blowup; summed-delta norm grows ~√T (random signs) to ~T (aligned), pushing activations off-distribution. This is the operator your disjoint-mask method rescues — with disjoint supports, summation at λ=1 has zero elementwise conflict by construction.

**TOFU notes.** This is the merge operator for both your LoRA-sum baseline (0.45 utility) and the disjoint-mask candidate. Same code path; only the training-time constraint on `τ_t` differs.

---

## 2. Uniform Averaging / Model Soup

**Idea.** Average weights instead of summing deltas; equivalent to summation with λ = 1/T.

**Algorithm.**
```
θ_merged = (1/T) Σ_t θ_t  =  θ₀ + (1/T) Σ_t τ_t
```

**Hyperparameters.** None (λ implicitly 1/T).

**Deletion.** **Not exact.** Removing task `t` from a T-author soup requires re-normalizing the remaining vectors by 1/(T−1) — the residual model differs from `θ_merged − (1/T)τ_t` and also from the (T−1)-soup. You *can* reconstruct the (T−1)-soup exactly if you stored every `τ_t` (recompute the average), which technically satisfies exactness via recomputation, but the *subtraction* form is not exact. Treat as baseline.

**Expected behavior on memorization.** This is your dilution failure mode in pure form: every key-value association is attenuated by 1/T = 1/200 = 0.005. Verbatim recall should collapse to near-zero. Worth running at small T only, as an anchor demonstrating the dilution mechanism.

---

## 3. Task Arithmetic with Tuned λ (Ilharco et al., ICLR 2023)

**Idea.** A single scalar λ, tuned on held-out data, scales the summed task vector.

**Algorithm.**
```
θ_merged(λ) = θ₀ + λ Σ_t τ_t
λ* = argmax_λ ValidationMetric(θ_merged(λ))
```

**Pseudocode.**
```python
summed = sum_task_vectors(taus)               # method 1 without adding to base
for lam in [0.1, 0.2, ..., 1.0]:              # standard grid; extend to 1/T if T large
    score[lam] = evaluate(apply(base, summed, lam), val_set)
lam_star = argmax(score)
```

**Hyperparameters.** λ grid, typically `{0.1, …, 1.0}` in steps of 0.1; for T=200 include values near `1/√T ≈ 0.07` and `1/T = 0.005`. Validation set: a held-out slice of retain-set QA (do **not** tune on forget-set metrics).

**Deletion.** **Not exact.** λ* was selected using validation performance of a model that included task `t`'s vector; the residual `θ₀ + λ*(Σ − τ_t)` is not what you would have obtained without task `t` (a different λ* would have been chosen). Baseline only.

**Expected behavior.** λ trades off interference vs. dilution. For memorization there may be **no good λ**: large λ → conflict noise; small λ → attenuated associations. Plotting utility vs. λ is diagnostic evidence for your dilution argument — report the full curve, not just λ*.

---

## 4. Fisher-Weighted Merging (Matena & Raffel, NeurIPS 2022)

**Idea.** Weight each parameter's contribution by its (diagonal) Fisher information under each task — parameters a task cares about get more say.

**Algorithm.** Per parameter index `i`:
```
F_{t,i} = E_{x∼D_t} [ (∂ log p_{θ_t}(y|x) / ∂θ_i)² ]        # diagonal empirical Fisher
θ_merged,i = ( Σ_t F_{t,i} · θ_{t,i} ) / ( Σ_t F_{t,i} + ε )
```

**Pseudocode.**
```python
for t in tasks:
    F[t] = zeros_like(theta[t])
    for batch in D[t]:                      # ~1k examples suffices; TOFU: all 20 QA
        loss = nll(model_t, batch)          # sample y from model or use gold labels
        loss.backward()
        for k in F[t]: F[t][k] += model_t.grad[k] ** 2
    F[t] = {k: v / n_batches for k, v in F[t].items()}

merged = {k: sum(F[t][k] * theta[t][k] for t in tasks)
             / (sum(F[t][k] for t in tasks) + 1e-8) for k in keys}
```

**Hyperparameters.** ε (1e-8), number of Fisher samples, whether to include `θ₀` as an extra "task" with its own Fisher (recommended by the paper to anchor general ability).

**Deletion.** **Not exact.** The denominator couples all tasks; removing `t` changes every merged parameter in a way that requires full recomputation, and the result ≠ subtracting anything storable per-task. Baseline.

**Cost.** One backward pass per Fisher sample per task; storage of one Fisher vector per task (same size as the model — at T=200 this is 200 model-sized objects; use fp16 Fisher, or compute in streaming fashion and never store all simultaneously).

**Expected behavior.** For memorization, per-author Fisher mass concentrates on few parameters (consistent with KV-storage in MLPs), so Fisher merging approximates soft parameter routing — likely one of the stronger inexact baselines.

---

## 5. RegMean (Jin et al., ICLR 2023)

**Idea.** For each linear layer, find the merged weight that minimizes the summed L2 distance to each task model's *outputs* on that task's inputs — a closed-form least-squares merge using input Gram matrices.

**Algorithm.** For a linear layer with weight `W` (output = `W x`), with task-t input activations `X_t` (rows = inputs to this layer over task t's data):
```
G_t = X_tᵀ X_t                                  # Gram matrix, d_in × d_in
W_merged = ( Σ_t W_t G_t ) ( Σ_t G_t )⁻¹
```
Regularize each `G_t ← α·G_t + (1−α)·diag(G_t)` before inversion.

**Pseudocode.**
```python
# 1. Hook every nn.Linear; run task t's data through model_t; accumulate X^T X per layer.
# 2. Per layer:
G_sum = sum(regularize(G[t], alpha) for t in tasks)
WG_sum = sum(W[t] @ regularize(G[t], alpha) for t in tasks)
W_merged = WG_sum @ inv(G_sum)
# 3. Non-linear-layer params (norms, embeddings): fall back to uniform average.
```

**Hyperparameters.** α ∈ [0.7, 1.0] (paper default ~0.9); number of forward batches for Gram estimation (a few hundred tokens per task suffices; TOFU: all 20 QA).

**Deletion.** **Not exact.** `(Σ G_t)⁻¹` couples tasks; removal requires recomputation with stored per-task `W_t G_t` and `G_t` (storable: `G_t` is d_in×d_in per layer per task — large but feasible). Recomputation-based removal is exact *if you store all Grams*, but the subtraction form is not. Classify as baseline; note the storage cost in the report if you run it.

**Expected behavior.** RegMean preserves each task's *function* on its own inputs better than weight-space averaging — a strong inexact baseline for the cross-talk question, since it directly optimizes against functional interference.

---

## 6. TIES-Merging (Yadav et al., NeurIPS 2023)

**Idea.** Three steps — **T**rim, elect **S**ign, disjoint merge — to remove two interference sources: redundant small weights and sign conflicts.

**Algorithm.**
1. **Trim:** per task, keep only the top-k% of `τ_t` entries by magnitude; zero the rest. `τ̂_t = keep_topk(τ_t, k)`.
2. **Elect sign:** per parameter index `i`, the aggregate sign `γ_i = sign( Σ_t τ̂_{t,i} )` (mass-weighted vote).
3. **Disjoint merge:** per index, average only the entries agreeing with the elected sign:
   `m_i = mean over {t : sign(τ̂_{t,i}) = γ_i} of τ̂_{t,i}` (mean over agreeing tasks only; 0 if none).
4. `θ_merged = θ₀ + λ · m`.

**Pseudocode.**
```python
for t in tasks:
    flat = concat_all(tau[t]); thresh = kth_largest(abs(flat), k_pct)
    tau_hat[t] = {k: v * (abs(v) >= thresh) for k, v in tau[t].items()}

elected = {k: sign(sum(tau_hat[t][k] for t in tasks)) for k in keys}
for k in keys:
    agree = [tau_hat[t][k] * (sign(tau_hat[t][k]) == elected[k]) for t in tasks]
    count = sum((a != 0) for a in agree).clamp(min=1)
    m[k] = sum(agree) / count
merged = apply(base, m, lam)
```

**Hyperparameters.** k = 20% (paper default; sweep {10, 20, 30}); λ ∈ [0.8, 1.2] typically ~1.0 (sweep on validation).

**Deletion.** **Not exact.** Sign election and the agree-set means are functions of *all* task vectors; removing `τ_t` can flip elected signs and change denominators globally. Baseline.

**Expected behavior at T=200 on memorization.** Sign election with 200 voters means any single author's idiosyncratic KV directions get outvoted; entries carrying one author's fact but conflicting with the majority are zeroed. Predict: helps general utility, *hurts* per-author verbatim recall relative to naive sum — and increases per-author variance (some authors' mass survives the vote, others' doesn't). This is a key comparison for your bimodality argument.

---

## 7. DARE (Yu et al., ICML 2024)

**Idea.** **D**rop **A**nd **RE**scale: randomly zero a fraction `p` of each task vector's entries and rescale survivors by `1/(1−p)` to preserve the expected delta; then merge (usually with task arithmetic or TIES on top). Reduces overlap probabilistically.

**Algorithm.** Per task:
```
m_t ~ Bernoulli(1−p)^d            # keep mask
τ̃_t = (m_t ⊙ τ_t) / (1−p)
```
Then merge the `τ̃_t` with any operator (naive sum, tuned-λ arithmetic, or TIES → "DARE-TIES").

**Pseudocode.**
```python
def dare(tau_t, p, seed):
    g = torch.Generator().manual_seed(seed)     # store seed per task for reproducibility
    return {k: v * (torch.rand(v.shape, generator=g) > p) / (1 - p)
            for k, v in tau_t.items()}
```

**Hyperparameters.** p ∈ {0.9, 0.99} (paper shows LLMs tolerate up to 99% drop for *ability*-style deltas); downstream merge operator and its λ.

**Deletion.** Depends on the downstream operator. DARE + naive sum at λ=1 **is separable** (each `τ̃_t` is a fixed function of `τ_t` and its stored seed), so subtraction of the stored `τ̃_t` is exact w.r.t. the DARE'd ensemble — but note the merged model was never equivalent to "trained without dropout", so exactness holds only relative to the DARE'd task vectors, which is acceptable (the DARE'd delta *is* what was added). DARE + tuned λ or DARE-TIES: not exact.
**Important caveat for your setting:** the 1/(1−p) rescale at p=0.99 multiplies surviving entries ×100. For memorization deltas (large, localized values), this can explode individual KV weights. The paper's success cases are instruction/ability deltas, not verbatim memorization; predict DARE fails harder here than in its native setting. Document this expectation.

---

## 8. Model Breadcrumbs (Davari & Belilovsky, ECCV 2024)

**Idea.** Per layer, sparsify each task vector by dropping both the smallest-magnitude entries (noise) **and** the largest-magnitude entries (outliers that dominate merges), keeping the mid-band; then sum with a scaling.

**Algorithm.** Per task, per layer: sort |τ| entries; zero the bottom β% and top γ%; merge survivors:
```
θ_merged = θ₀ + λ Σ_t midband(τ_t; β, γ)
```

**Hyperparameters.** β ≈ 85–90% (bottom drop), γ ≈ 1–2% (top drop), λ tuned (paper: fixed hyperparameters transfer across tasks reasonably, tested up to ~200 vision tasks).

**Deletion.** Sparsification is per-task (separable); with fixed λ=1 the merge is exactly subtractable given stored sparsified deltas. With tuned λ: not exact. **Caution:** dropping the top-γ% likely deletes exactly the strong KV associations that carry verbatim facts. Predict poor retain-recall; useful as evidence that magnitude outliers *are* the memorization.

---

## 9. LoRA Merging Done Correctly (your baseline c)

**Constraint restated:** never average A/B factors across tasks. `mean(B)·mean(A) ≠ mean(B·A)` — factor averaging is a different (and meaningless) operator.

**Correct composition.** For each adapted weight `W` with per-task LoRA factors `B_t ∈ ℝ^{d_out×r}`, `A_t ∈ ℝ^{r×d_in}`, scale `α/r`:
```
ΔW_t = (α/r) · B_t A_t                 # materialize the full-rank-≤r dense delta
θ_merged = θ₀ + λ Σ_t ΔW_t
```

**Equivalent factor-space form (no materialization):** concatenation.
```
A_cat = concat_rows(A₁, …, A_T)        # (T·r) × d_in
B_cat = concat_cols(B₁, …, B_T)        # d_out × (T·r)
ΔW_sum = (α/r) · B_cat A_cat           # == Σ_t ΔW_t exactly
```
Concatenation is exact and keeps adapters subtractable (drop task t's block columns/rows). Use it when memory for T·r rank is acceptable (200 × r=8 = rank-1600 per matrix — fine).

**Deletion.** Exact at λ=1 with independent training: subtract `ΔW_t` (or drop the block). With tuned λ: not exact.

**Pseudocode.**
```python
delta = {k: torch.zeros_like(base[k]) for k in lora_target_keys}
for t in tasks:
    for k in lora_target_keys:
        delta[k] += (alpha / r) * (B[t][k] @ A[t][k])
merged = apply(base, delta, lam=1.0)
```

**Expected behavior.** This is your measured 0.45 baseline. Diagnosis (KnOTS; intruder-dimensions): independently trained LoRA updates occupy misaligned low-rank subspaces with low mutual alignment; summation superposes 200 rank-r updates whose singular directions collide in a d_in-dimensional space far more than dense sparse deltas do. Keep exactly this implementation as the control.

---

## 10. KnOTS (Stoica et al., ICLR 2025) — LoRA-Aware Merging

**Idea.** Before merging LoRA deltas, jointly SVD the stacked deltas so all tasks are expressed in a **shared** singular basis; merge the aligned representations (with TIES or averaging), then reconstruct.

**Algorithm.** Per adapted layer:
1. Materialize `ΔW_t = B_t A_t` for all t; stack `[ΔW₁; …; ΔW_T]` (concatenate along output dim).
2. Joint SVD: `[ΔW₁; …; ΔW_T] = U Σ Vᵀ`. `V` spans the shared input-side basis.
3. Represent each task in the shared basis: `Z_t = ΔW_t V` (task-specific coefficients).
4. Merge the `Z_t` with an off-the-shelf operator (TIES recommended): `Z* = merge(Z₁,…,Z_T)`.
5. Reconstruct: `ΔW_merged = Z* Vᵀ`; `θ_merged = θ₀ + λ ΔW_merged`.

**Hyperparameters.** Inner merge operator + its hyperparameters; λ; SVD rank truncation (keep full T·r rank or truncate).

**Deletion.** **Not exact.** The shared basis `V` is computed from all tasks; removal changes the basis and every task's representation. Baseline. Its value in your paper: if KnOTS substantially beats naive LoRA-sum, that confirms subspace misalignment (not capacity) drives the 0.45 collapse — direct evidence for your diagnosis (a)/(b).

---

## 11. Tangent / Linearized Task Arithmetic (Ortiz-Jiménez et al., NeurIPS 2023) — Training-Time Constraint

**Idea.** Fine-tune in the model's tangent space (first-order Taylor expansion around θ₀), where the model is *linear in parameters*, so task vectors compose additively by construction ("weight disentanglement").

**Algorithm.**
1. Define the linearized model `f_lin(x; θ) = f(x; θ₀) + ∇_θ f(x; θ₀)ᵀ (θ − θ₀)` (implemented via JVPs, e.g. `torch.func.jvp`).
2. Fine-tune each task **in the linearized model** → `τ_t^lin`.
3. Merge: `f_lin(x; θ₀ + Σ_t τ_t^lin)` — additivity in parameters is exact additivity in function space for `f_lin`.

**Hyperparameters.** Same as normal fine-tuning; note ~2–3× training cost (JVP passes) and that inference must also use the linearized model (or accept the nonlinear model's approximation gap).

**Deletion.** Exact at λ=1 with independent training (separable). Genuinely a *candidate* by your criterion, not just a baseline — but:
**Feasibility caveat at 7B:** JVP fine-tuning of Llama-2-7B is expensive and the linearized model underperforms the nonlinear one on generation quality; no published evidence it can do verbatim memorization at TOFU scale. Recommend: cite as the theoretically clean alternative; run only if disjoint masks fail and you need a rescue direction, or run at Phi-scale as a side experiment.

---

## 12. SIFT-Masks (Kuo et al., SaTML 2026) — Sign-Fixed Sparse Tuning

**Idea.** All tasks share one **global random sign vector** `s ∈ {−1,+1}^d` fixed before training. Each task fine-tunes under the constraint that every nonzero delta entry agrees with `s` (sign-projected SGD: after each step, zero entries whose sign disagrees). Merging is then conflict-free summation at λ=1. Each task also saves a **local binary mask** `m_t = support(τ_t)`; at inference for task t, apply `θ₀ + m_t ⊙ (Σ_u τ_u)` — i.e. mask the merged delta down to task t's support.

**Algorithm.**
1. Sample `s` once; broadcast to all tasks.
2. Per task: fine-tune from θ₀ with projection `τ ← τ ⊙ 1[sign(τ) = s]` after each optimizer step (and optionally top-k sparsification).
3. Merge: `Δ = Σ_t τ_t` (no rescale).
4. Inference for task t: `θ₀ + m_t ⊙ Δ`. ("Merge-only" variant skips this step — and collapses at large T.)
5. Unlearn task t: `Δ ← Δ − τ_t`; discard `m_t`. Exact.

**Deletion.** Exact (separable; `s` is task-independent).

**Role in your project.** The pivotal comparison. Two axes of difference from yours: overlapping supports (sign-aligned) vs. disjoint; inference-time mask (≈ router) vs. none. Implement both SIFT variants (with and without inference mask) so your results table shows: SIFT+mask (their method) / SIFT merge-only (collapses) / disjoint merge-only (your claim: does not collapse). That triple is the paper's central figure.

---

## 13. Disjoint-Mask Merging (the candidate method)

**Idea.** Assign each task a fixed sparse mask **before training**, with pairwise-disjoint supports across all T tasks. Train full-parameter deltas restricted to the mask. Merge by summation at λ=1: zero elementwise overlap → zero sign conflict, zero rescaling, full-strength associations. Delete by exact subtraction.

**Algorithm.**
1. **Mask assignment.** Choose density ρ (fraction of mergeable parameters per task; require ρ·T ≤ 1). Partition: sample a random permutation π of the d mergeable indices; assign task t the slice `π[(t−1)·⌈ρd⌉ : t·⌈ρd⌉]`. Store the RNG seed; masks are then reproducible from (seed, ρ, T, d) without storing bitmasks.
   - *Gradient-importance variant (ablation):* warm up each task 10–50 steps unconstrained from θ₀; score indices by |grad| or diagonal Fisher; assign greedily by importance with conflicts resolved by (i) priority order or (ii) auction/Hungarian on the top-k×3 candidate pool; freeze assignment; retrain from θ₀ under the final mask. Assignment now depends on other tasks' *data* via conflict resolution — deletion remains exact as long as masks are frozen before training and stored (the delta subtracted is exactly the delta added), but note in the paper that mask *assignment* leaked cross-task information; if a reviewer objects, fall back to random.
   - *MLP-only variant (ablation):* restrict the mergeable index set to mid-layer MLP weight matrices (e.g. `mlp.down_proj`, `mlp.up_proj`, `mlp.gate_proj` of layers ⌊L/4⌋…⌊3L/4⌋) before partitioning, motivated by ROME/MEMIT localization.
2. **Training.** Per task, from frozen θ₀: standard fine-tuning where the optimizer updates only masked entries. Implementation: after `loss.backward()`, zero gradients outside the mask (`p.grad.mul_(mask)`), or maintain a flat trainable vector scattered into the model each step (faster; avoids optimizer state on d parameters). AdamW, lr ~1e-4–5e-4 (sparse updates tolerate higher lr), 50–200 steps on the author's 20 QA, batch = all 20. Train to memorization criterion: exact-match ≥ threshold on the author's own QA, early-stop.
3. **Merge.** `Δ = Σ_t τ_t`, λ=1, no rescale. Because supports are disjoint, this is a scatter, not a sum: `Δ[idx_t] = values_t`.
4. **Delete.** `Δ[idx_t] = 0` (equivalently subtract). Bit-exact residual = merge of remaining T−1. No recomputation.
5. **Storage.** Per task: `(seed-derived indices, fp16 values)` — at ρ=0.1% of ~6.5B mergeable params ≈ 6.5M values ≈ 13 MB fp16 per author, ~2.6 GB for 200. Indices are free (regenerate from seed).

**Hyperparameters.** ρ ∈ {0.05%, 0.1%, 0.5%}; mask scheme {random, importance, MLP-only}; per-task steps/lr; memorization early-stop threshold.

**Deletion.** Exact by construction (separable, λ=1 fixed a priori, masks fixed a priori).

**Failure surface to instrument.** (i) Trainability: fraction of authors failing the memorization criterion at each ρ (report as a distribution — this is the bimodality risk). (ii) Cross-talk: per-author recall with only own delta vs. all 200 active; the gap is pure function-level interference since parameter-level interference is zero. (iii) Norm growth: ‖Δ‖ and per-layer activation-norm drift vs. T.

---

## 14. MEMIT-Style Closed-Form Edits as Additive Unlearning (baseline d)

**Idea.** Treat each fact as a key-value pair in mid-layer MLP `down_proj` matrices; compute a closed-form least-squares weight update that inserts a batch of facts; deletion = subtract the stored update.

**Algorithm sketch.** For MLP output matrix `W` (treating the MLP as `v = W k`), with existing-key covariance `C = λ_c·E[kkᵀ]` (estimated once from generic corpus text on θ₀):
- For each fact, compute key `k*` (average activation at the subject's last token over paraphrase prompts) and target value `v*` (optimized so the layer output makes the model emit the fact).
- Batch update spread over layers ℒ = {mid-layer range}: per layer, `ΔW = R Kᵀ (C + K Kᵀ)⁻¹` where `K` stacks keys and `R` stacks residuals (target minus current values), distributed across layers per the MEMIT recursion.
- Store `ΔW_t` per author (their ~20 facts as one batch). Merge: `W ← W + Σ_t ΔW_t`. Delete: subtract `ΔW_t`.

**Deletion caveat — why this is a baseline, not a candidate:** if all authors' edits are computed independently against θ₀, subtraction removes exactly what was added, and the residual equals the (T−1)-merge — *separable, hence exact in your sense relative to the edit ensemble.* However: (i) `C` is a shared statistic (fine — computed from θ₀, task-independent); (ii) the batched-vs-sequential choice matters — MEMIT applied sequentially (each author edited on top of the previous) is **not** separable; only the "all deltas computed on θ₀ then summed" variant qualifies, and that variant is *not* how MEMIT is normally run and its interference at 200×20 facts is exactly what you're testing. Also the edit is not "retraining-equivalent" in the SISA sense — the model never *trained* on the author, so frame carefully: it's exact *removal of the edit*, a different guarantee.

**Hyperparameters.** Edit layers (Llama-2-7B: roughly layers 4–8 following MEMIT ports; verify with causal tracing), `λ_c` (covariance weight, ~15,000 in the reference impl at GPT-J scale — re-tune), number of paraphrase prompts per key.

**Use the reference implementation** (github.com/kmeng01/memit) ported to Llama-2 rather than reimplementing the value-optimization loop.

---

## 15. Common Evaluation Harness (all methods)

Run every merge operator through one fixed pipeline so numbers are comparable:

1. **Inputs:** frozen θ₀; per-author artifacts (`τ_t` dense-sparse / LoRA factors / MEMIT ΔW / masks / seeds); method config.
2. **Merge → single dense checkpoint** (materialize; do not evaluate through adapter hooks — keep the inference path identical across methods).
3. **Metrics per condition:**
   - TOFU model utility (harmonic aggregate of probability, ROUGE, truth ratio over retain / real-authors / world-facts).
   - Forget quality post-deletion (KS-test vs. retain-only reference) for a fixed panel of deleted authors (e.g. TOFU forget01/05/10 splits).
   - Retain-set verbatim recall: ROUGE-L against gold answers, greedy decoding, per author.
   - **Cross-talk probe:** for a fixed 30-author panel, recall with only own delta vs. all-T merge.
   - **Distributions, not means:** per-author recall histograms/violins at every T ∈ {25, 50, 100, 200}; report min, 10th percentile, and fraction of authors below 0.5 ROUGE alongside the mean.
4. **Controls held fixed:** decoding params (greedy, max_new_tokens), prompt template, eval seed, tokenizer. Merge-order invariance check for any method claiming separability (permute task order; assert bit-identical output).
5. **Exactness unit test:** for each candidate method, assert `merge(all) − delta_t == merge(all \ t)` elementwise to fp tolerance, and that forget-quality of the subtracted model matches a never-trained control on a small pilot (e.g. 5 authors).

---

## 16. Summary Table

| # | Method | Merge operator | Cross-task coupling | Exact deletion (λ=1, indep. training) | Expected on 200-author memorization |
|---|--------|----------------|--------------------|----------------------------------------|--------------------------------------|
| 1 | Naive sum | Σ τ_t | none | **Yes** | Collapses (dense); reference operator |
| 2 | Uniform soup | (1/T) Σ τ_t | normalization | recompute-only | Dilution collapse; anchor |
| 3 | Tuned-λ arithmetic | λ* Σ τ_t | λ tuned globally | No | No good λ exists (predicted) |
| 4 | Fisher merge | Fisher-weighted avg | denominator | No | Strong inexact baseline |
| 5 | RegMean | Gram least-squares | (Σ G_t)⁻¹ | No (recompute w/ stored Grams) | Strong inexact baseline |
| 6 | TIES | trim+sign-elect+mean | sign vote | No | Outvotes minorities; ↑ variance |
| 7 | DARE(+sum) | drop/rescale + Σ | none (seeded) | Yes (w.r.t. DARE'd deltas) | Rescale explodes KV weights |
| 8 | Breadcrumbs | midband + λΣ | λ (if tuned) | Yes if λ=1 | Top-drop deletes memorization |
| 9 | LoRA sum (ΔW=BA) | Σ B_tA_t | none | **Yes** | Your 0.45 control |
| 10 | KnOTS | shared-SVD + inner merge | joint basis | No | Diagnostic for subspace collision |
| 11 | Tangent TA | Σ τ_t^lin (linearized) | none | **Yes** | Clean but costly; unproven for memorization |
| 12 | SIFT-Masks | sign-fixed Σ (+ inference mask) | global sign (task-indep.) | **Yes** | Collapses w/o mask; pivot comparison |
| 13 | **Disjoint-mask (ours)** | disjoint scatter, λ=1 | none | **Yes** | The open question |
| 14 | MEMIT-batch-on-θ₀ | closed-form Σ ΔW | shared C (task-indep.) | Yes (edit-removal sense) | Strong baseline; different guarantee |

---

### Implementation order suggestion
1 → 9 → 13 → 12 → 3 → 6 → 7 → 14 → 4/5/10, with 2, 8, 11 as optional appendix rows. Methods 1, 9, 13 share one code path (sum of stored deltas) — build that first, and the exactness unit test (§15.5) before anything else.
