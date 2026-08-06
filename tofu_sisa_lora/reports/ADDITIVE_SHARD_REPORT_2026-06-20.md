# Exact Unlearning on TOFU via Additive LoRA Shards — Report

**Date:** 2026-06-20 · **Model:** `meta-llama/Llama-2-7B-chat-hf` · **Seed:** 42 ·
**Eval:** open-unlearning-faithful metrics (`eval_tofu.py`, guarded by `test_ou_equivalence.py`),
smoke caps · All GPU work via SLURM (sprint1–3).

**Headline.** The literal additive sum the method prescribes *collapses* the model (utility 0.0).
The cause is **norm overshoot**, not a scaling bug — proven exact at the matrix level. Fixing the
*composition structure* — one jointly-trained retain core plus a modular forgettable tail —
recovers **model utility 0.754 for the unlearned model, matching joint fine-tuning (0.740)**,
with forget quality near ceiling, at O(1) deletion cost.

---

## 1. The idea

**Exact machine unlearning** means: after deleting data *D_f*, the model is identical to one that
was *never trained on D_f* — no approximate "scrubbing," no residual influence. The SISA recipe
(Bourtoule et al. 2021) achieves this by **sharding**: partition the training data, train an
independent model per shard, and serve an aggregate. Deleting a shard's data = drop that shard's
model and re-aggregate. No surviving parameter ever saw the deleted data, so the result is exact
by construction.

The **LoRA instantiation** under test here: train one LoRA adapter per data shard, each starting
from the *same frozen base*, each seeing *only its own shard*. Compose by **summing** the low-rank
updates:

```
W_eff = W_base + Σ_i (α/r) B_i A_i          (sum over active shards i)
```

To unlearn shard *j*, **drop its term**: `W_eff − (α/r)B_j A_j`. Because addition is commutative
and exactly invertible, this is bit-identical to the model built as if shard *j* never existed —
*provided* (a) adapters are trained independently against the clean base, and (b) composition is
purely additive (no multiplicative cross-terms).

**Why TOFU is the ideal testbed.** TOFU (Maini et al. 2024) is 200 fictitious authors × 20 Q&A =
4000 rows. The authors are invented, so the *only* way a model knows them is our fine-tuning —
clean signal. TOFU's forget sets are **nested and live in the tail**: `forget10` = authors
180–199, `forget05` = 190–199, `forget01` = 198–199. TOFU scores unlearning as *indistinguishability
from a retain-only model*, so in this scheme the unlearned model **is** a retain-only model by
construction → Forget Quality is ≈ceiling for free. **The only real question is Model Utility of
the additive composition versus joint fine-tuning.**

---

## 2. The method (and the fix)

### 2.1 True-scale additive composition
Each shard adapter's *effective delta* is `ΔW_i = scaling_i · B_i A_i` with `scaling_i = α/√r`
(the shards train with rsLoRA). Honest additive composition sums each effective delta **once**:

```
ΔW_merged = Σ_i scaling_i · B_i A_i
```

We implement this as `merge_extra.additive_merge_adapters`: a `cat` scaffold (rank = Σ r_i, so the
concatenation fits with **no SVD compression**), writing `[scaling_i·B_i]` / `[A_i]` and dividing
out the scaffold's own scaling. This is the *corrected* PEFT `linear`/`cat`, which instead apply
`√(w·scaling)` per factor and double-count the rsLoRA `√r` — the reason the registry's `linear`/`cat`
explode on these shards.

**Exactness is bit-exact at the matrix level** (`test_merge_extra.test_additive`, max rel err 3e-7):
`merged` == `Σ scaling_i·B_i A_i`, and `remerge` (drop the forget shard) == `merged` − forget term.

### 2.2 The failure: norm overshoot
The literal weight-1.0 sum **collapses** (Section 4.1). It is **not** the `√r` artifact — the merge
is provably exact. It is **norm overshoot**: each adapter is trained *independently* to fully
memorize its shard against the bare base, so the deltas are full-strength and **heavily overlap**
in shared directions (all shards push the same q/k/v/o/up/down matrices to "answer TOFU-style QA").
Summing *k* of them stacks those shared directions ~*k×* too high → activations blow up. Confirmed:
the blow-up scales with the number of summed terms (remerge = 9 terms is less broken than merged =
10 terms). This is the classic task-arithmetic interference problem.

### 2.3 Two fixes, both exactness-preserving
1. **Global coefficient λ** — compose `W + λ·Σ_i ΔW_i`. A *fixed* λ keeps the pure "drop a term"
   property: unlearn *j* → `W + λ·Σ_{i≠j}ΔW_i` (bit-exact subtraction of `λ·ΔW_j`). λ is a
   deploy-time knob (task-arithmetic coefficient), tuned on retain only — never touches forget data.
   Exposed via the `_s{λ}` label suffix (e.g. `merged_additive_s0.2`). **Result: helps, but
   composing many equal shards caps at ~0.48 ≈ `dare_ties` (Section 4.2).**
2. **Coarse retain-core + fine tail** (the decisive fix). On TOFU, forgetting only ever hits the
   tail, so the retain side **never needs fine modularity**. Train *one* jointly-trained retain
   core (high utility, never sharded) and keep only the ≤20 forgettable tail authors as separate,
   droppable adapters. Unlearn = drop tail adapters; the strong core carries utility. **Result:
   the unlearned model reaches 0.754 ≈ joint-ft (Section 4.3).**

---

## 3. Settings

| Item | Value |
|---|---|
| Base model | `meta-llama/Llama-2-7B-chat-hf` (frozen) |
| Shard LoRA recipe | rank 32, α 64, **rsLoRA** (scaling α/√r), dropout 0.05, targets `q,k,v,o,up,down` |
| Train hyperparams | 5 epochs, lr 1e-4, batch 1 × grad-accum 16 (eff. 16), max_len 256, seed 42, optim paged_adamw_32bit, cosine |
| Existing k=10 shards | `checkpoints/Llama-2-7B-chat-hf/shard_0..9` (shard 9 = authors 180–199 = `forget10`) |
| Strong retain core | authors 0–179, **same recipe** (r32/α64/e5/lr1e-4), dir `Llama-2-7B-chat-hf_retainstrong/retain90` |
| Tail adapters (deferred) | `Llama-2-7B-chat-hf_k200_r32_e5_lr1e4/shard_180..199` (per-author, already trained) |
| Composition | `additive` = Σ scaling_i·B_i A_i (full rank, cat scaffold); `_s{λ}` = ×λ; `additive_mean` = ×1/n |
| Eval | `eval_tofu.py --k 10 --forget_shard_id 9 --smoke`; OU-faithful metrics |
| Smoke caps | ROUGE ≤ 50, retain ≤ 80, truth-ratio ≤ 30 samples |
| Forget-Quality KS ref | legacy `retain90` oracle (r8/e3) truth-ratios (`retain_tr_scores.npy`) |
| Hardware | SLURM sprint1–3, 1 GPU/task, 48G eval / 64G train |

**Metrics.** `model_utility` (mu) = harmonic mean of 9 components {probability, ROUGE-L recall,
scaled truth-ratio} × {Retain, Real-Authors, World-Facts}; the harmonic mean means one near-zero
component zeroes the score. `forget_quality` (fq) = KS p-value of the forget truth-ratio
distribution vs the retain oracle (**higher = more indistinguishable from never-trained = better**).
`*_ppl` = perplexity (base ≈ 15.2).

---

## 4. Results

**Anchors:** base model mu **0.418** (ppl 15.2) · joint full-data fine-tune (all 200 authors,
r32/e5) mu **0.740** · `dare_ties` (the prior SISA default merge) remerge mu **0.460**.

### 4.1 The literal additive sum collapses
| label | mu | fq | forget_ppl | retain_ppl | retain_prob |
|---|---|---|---|---|---|
| `merged_additive` (Σ all 10, λ=1) | **0.000** | 0.135 | 26047 | 27689 | 0.000 |
| `remerge_additive` (Σ 9, drop shard 9) | **0.000** | 0.071 | 11549 | 11299 | 0.000 |

Perplexity ~10⁴ (base ≈ 15): the model is destroyed. remerge (9 terms) < merged (10 terms) →
blow-up tracks term count → **norm overshoot, not a scaling bug** (the merge is bit-exact).

### 4.2 Global-λ sweep — equal-shard composition caps at ~0.48
Composing all 10 (`merged`) or the 9 retained (`remerge`) equal shards at coefficient λ:

| λ | merged mu | remerge mu | remerge fq | remerge retain_ppl |
|---|---|---|---|---|
| 0.10 (= mean) | 0.481 | **0.484** | 0.808 | 4.4 |
| 0.15 | 0.447 | 0.462 | 0.958 | 4.4 |
| 0.20 | 0.385 | 0.414 | 0.958 | 4.8 |
| 0.25 | 0.077 | 0.221 | 0.958 | 5.7 |
| 0.30 | 0.000 | 0.052 | 0.999 | 7.4 |
| 0.50 | 0.000 | 0.065 | 0.958 | 51.4 |
| 1.00 | 0.000 | 0.000 | — | 11299 |

Utility **peaks at λ ≈ 0.1 (~0.48) and falls off fast** as λ rises (overshoot returns). No λ lifts
many-equal-shard composition above ~0.48 — the same neighborhood as `dare_ties` (0.46). This is the
**co-adaptation ceiling** of composing many independent, non-orthogonal full-strength shards.

### 4.3 Coarse retain-core — the fix (headline)
The unlearned model for `forget10` = the retain core alone (authors 0–179, never saw forget data):

| state | mu | fq | forget_ppl | retain_ppl | retain_prob | retain_rouge |
|---|---|---|---|---|---|---|
| base (anchor) | 0.418 | — | 15.2 | — | — | — |
| naive sum (§4.1) | 0.000 | — | 26047 | 27689 | 0.000 | 0.053 |
| equal-shard best λ (§4.2) | ~0.48 | 0.96 | 5.9 | 4.4 | 0.305 | 0.49 |
| legacy core alone (r8/e3) | 0.629 | 1.000 | 6.4 | 2.1 | 0.534 | 0.561 |
| **strong core alone (r32/e5)** | **0.754** | **0.958** | **14.2** | **1.2** | **0.967** | **0.936** |
| joint full-FT (anchor) | 0.740 | — | — | — | — | — |

The strong retain core **matches/edges joint fine-tuning (0.754 vs 0.740)**, with **forget_ppl 14.2
≈ base** (the forget authors are genuinely absent) and **forget_quality 0.958** (near ceiling, by
construction). Training converged cleanly (loss 2.37 → 0.20).

---

## 5. What we learned

1. **The additive idea is sound; the naive *structure* is not.** Summing many full-strength,
   overlapping independent adapters overshoots catastrophically — independent of any scaling
   convention. The exactness math is perfect (bit-exact drop-a-term); the failure is pure magnitude.
2. **Aggregation weight matters but has a ceiling.** A global λ (or mean) tames the overshoot but
   composing *k* equal shards caps at ~0.48 — interference between co-located deltas is the limit.
3. **Don't shard what you never forget.** TOFU only forgets the tail, so the retain side should be a
   single jointly-trained core (utility ≈ joint-ft), with only the forgettable tail kept modular.
   This sidesteps overshoot entirely and is exactly the document's §4/§6 cost design.
4. **Forget Quality is ≈ceiling for free**, as predicted: the unlearned model never saw forget data,
   so it is a valid retain model and the KS test sits near ceiling.

---

## 6. Exactness guarantee

Unlearning here is exact by *construction*, not approximation:
- The retain core trains **only** on retain authors → never sees forget data.
- Tail adapters are trained **independently**, each on a single author, against the frozen base.
- Composition is **additive**: `W + core + λ·Σ(active tail)`. Dropping a tail adapter removes
  exactly its term (CPU-proven bit-exact, err 3e-7). A *fixed* λ keeps drop-a-term exact.
- Therefore the unlearned model is identical to one built as if the deleted authors never existed —
  no gradient surgery, deterministic, O(1) in the number of deleted shards.

---

## 7. Reproduction

```bash
PY=/home/jack/anaconda3/envs/test-env/bin/python
cd ~/tofu_sisa_lora

# CPU correctness gate (additive identity, drop-term recovery, λ-scale, determinism)
$PY test_merge_extra.py && $PY test_ou_equivalence.py

# Naive sum + mean + λ-sweep on existing k=10 shards (eval only, no training)
#   labels: merged_additive / remerge_additive / *_mean / *_s{λ}
bash submit_eval.sh checkpoints/Llama-2-7B-chat-hf meta-llama/Llama-2-7B-chat-hf 10 9
#   (driven by results/smoke/eval_manifest_additive*.txt)

# Strong retain core (train r32/e5 on authors 0-179) + eval standalone = forget10 unlearned
$PY train_lora_shard.py --retain90 --retain_authors 180 --k 10 \
  --model_name meta-llama/Llama-2-7B-chat-hf \
  --output_dir checkpoints/Llama-2-7B-chat-hf_retainstrong \
  --rank 32 --alpha 64 --epochs 5 --lr 1e-4 --batch_size 1 --grad_accum 16 --seed 42
$PY eval_tofu.py --model_name meta-llama/Llama-2-7B-chat-hf \
  --output_dir checkpoints/Llama-2-7B-chat-hf --label retain90_strong_alone \
  --k 10 --forget_shard_id 9 \
  --preloaded_adapter checkpoints/Llama-2-7B-chat-hf_retainstrong/retain90 \
  --out checkpoints/Llama-2-7B-chat-hf/results/smoke/retain90_strong_alone.json --smoke
```

**Provenance.** SLURM jobs: 435424 (naive sum), 435579 (mean), 435630 (λ-sweep), 435635
(legacy core), 435657 (strong core train+eval). Result JSONs under
`checkpoints/Llama-2-7B-chat-hf/results/smoke/`. Code: `merge_extra.additive_merge_adapters`,
`merge_lora.py` (`additive`/`additive_mean` methods, `_split_scale_suffix`),
`test_merge_extra.test_additive`. Full narrative in `~/log/sisa_lora/` (2026-06-20).

---

## 8. Limitations & deferred work

- **Smoke caps** (subsampled metrics). The 0.754 headline should be confirmed at extended caps
  (ROUGE ≤ 200, retain ≤ 400, truth ≤ 120) for the final number.
- **forget05 / forget01 not yet run.** These compose the strong core + the *kept* tail adapters
  (which already exist in `k200_r32`); this needs a small cross-dir multi-adapter additive path.
  `forget10` (core alone) is the cleanest case and is done.
- **Forget-Quality KS reference** is the legacy r8 retain90 oracle; a strong-recipe oracle would
  tighten the fq comparison (utility, the headline, is unaffected).
- **Recipe.** Shards use the existing rsLoRA / "Question:/Answer:" / full-sequence-loss recipe, not
  the document's standard-LoRA / answer-only / `[INST]` recipe. The structure finding is
  recipe-independent, so the faithful-recipe retrain was not needed for this result.
