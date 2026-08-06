### Target Date: 2026-07-15 (Centered-merge N-ladder — design & pre-registration)
- **Hypotheses / what we're testing:** The third composition regime from
  [PATHS_FORWARD §6.1](../PATHS_FORWARD_2026-07-13.md): shared component at ~1×, per-author
  idiosyncratic residuals at ~1×. **Design correction, recorded before any run:** the literal
  §6.1 formula ΣΔᵢ − (k−1)Δ̄ with Δ̄ = (1/k)ΣΔᵢ is an algebraic identity —
  kΔ̄ − (k−1)Δ̄ = Δ̄ — i.e. exactly the `additive_mean` merge Exp-5 already measured
  ([2026-07-08_interference-vs-n-results.md](2026-07-08_interference-vs-n-results.md): recall
  saturated by N≈8, mu flat 0.459). The literal formula is therefore **rejected as degenerate**;
  "shared once + residuals at full strength" is only a new regime when the shared estimate S is
  NOT the exact subset mean. Two non-degenerate, exactness-preserving estimators are run instead,
  both of the form **M = S + Σᵢ(Δᵢ − S) = ΣΔᵢ − (N−1)·S** (S a deterministic function of adapter
  files ⇒ deletion = recompute S without Δⱼ and re-merge; no training, certificate survives):
  - **centered_pool** (`cpool`): S = Δ̄_pool = mean of ALL 199 pool adapters. Parameter-free,
    closest to the §6.1/Slack intent. Degenerates back to the mean as subset → pool (at
    subset==pool it IS the mean; at N=200 with pool=0..198 it collapses to Δ₁₉₉ alone), and the
    estimator self-contamination grows as N/P — so capped at **N ≤ 64**.
  - **centered_lowrank ρ=16** (`cr16`): S = per-slot rank-ρ truncated SVD of the subset mean,
    M = P_ρ + Σᵢ(Δᵢ − P_ρ). Non-degenerate at every N (a deployable full-scale form); ρ
    interpolates the two known-dead regimes (ρ=0 ⇒ naive unit sum, ρ=full ⇒ mean); ρ=16 is
    grounded in Exp-1's rank-16 shared col(B) basis at 92× chance
    ([2026-07-07_per-author-similarity-k200.md](2026-07-07_per-author-similarity-k200.md)).
    Known cost: the shared energy OUTSIDE rank ρ is amplified ~N× — where that re-collapses the
    curve is itself a crosstalk measurement.

  Numbered hypotheses (e5 mean-curve comparators quoted from Exp-5/Exp-5b):
  - **H-cent-1 (recall rescue):** centered merges lift subset-conditioned own-recall above the
    mean regime at N ≥ 4. CONFIRM: subset `retain_prob` at N=8 ≥ 2× the e5 mean value
    (0.2499 → ≥ 0.50). REFUTE: within ~0.05 of the mean curve at N ∈ {4,8,16}.
  - **H-cent-2 (utility survives — the user's headline question):** standard `model_utility`
    of centered merges stays ≥ mean-regime mu − 0.02 (e5 mean: 0.459 ± 0.002) with no
    norm-overshoot collapse. REFUTE: mu → base (0.426) or the sum-regime signature
    (retain_ppl ≫ 20, cf. λ-sweep cliff retain_ppl 8 → 1.8M).
  - **H-cent-3 (crosstalk N\*):** if H-cent-1 holds at small N, recall re-collapses at a
    measurable N\* on the ladder (√N residual-crosstalk noise); N\* and the fall's shape measure
    inter-residual crosstalk directly. Either outcome (N\* ≤ 200, or survival to 200) is a
    mechanism result.
  - **H-cent-4 (H3 sign flip):** Exp-5's H3 found overlap-with-shared = *protection* under the
    mean (ρ = −0.675 between probe col(B) overlap and recall drop). Centering removes exactly
    that protection ⇒ the correlation should weaken or flip sign for centered merges. CONFIRM:
    centered-merge drop-vs-overlap correlation ≥ −0.2 (i.e. protection gone). REFUTE: ρ ≤ −0.5
    persists.
- **Setup:** planned (results entry will record actuals). Pool/shards = e5 per-author set
  `/storage2/jack/checkpoints/tofu_sisa_lora/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4/shard_{0..199}`
  (same adapters as Exp-5 ⇒ curves overlay exactly). Config
  `../../tofu_sisa_lora/configs/nmerge_centered_7b.json` → out_dir `..._nmerge_r32_centered`;
  subset seed 42 (probes perm[:5] = [82, 15, 111, 177, 76]); ladder N ∈
  {2,3,4,6,8,12,16,20,32,64} for cpool (cap per the degeneracy note) and
  {2,3,4,6,8,12,16,20,32,64,128,200} for cr16 (`_svd1024` at N ∈ {128,200}, svd-vs-exact
  acceptance point at N=64 — the additive_mean convention). Merges materialized on CPU
  (`merge_subset.py merge`, single exact truncated SVD of the full weighted factor cat — no
  cascaded compression), evals via `submit_nmerge.sh eval` (1 GPU/task, smoke caps, self-skip;
  array throttle chosen so queued-max GPU ≤ 4 global cap). iso/anchor reference rows are
  bit-identical to the e5 campaign's ⇒ their result JSONs are **copied** into the new results
  dir rather than re-evaluated (files listed in `CLAUDE_SCRATCHPAD.md`; provenance inside each
  JSON is unchanged and correct). CPU gate extended first: `test_merge_subset.py` now must
  prove the degeneracy (S = subset mean ⇒ output ≡ additive_mean), cr ρ=full ≡ mean, ρ=0 ≡
  unit sum, pool==subset ≡ mean, and deletion-recompute determinism.
- **Results:** *(pending — will be reported in a dated results entry; this is the
  pre-registration.)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending; silent-failure watchlist: retain_ppl explosion at large N
  [sum-regime relapse], NaN mu from empty restricted truth-ratio subsets at small N [known
  Exp-5b artifact — read prob/rouge/ppl], svd_energy of the final compression [flag < 0.98].)*
- **New questions / new hypotheses:** e25 strong-expert centered wave (H8 showed strong
  experts interfere MORE — if centering rescues the e25 N≤20 zone the micro-merge serving tier
  widens); ρ-sweep {8,32,64} at pivotal N if cr16 lands in a gray zone.
- **Next Steps:** implement + CPU gates → STUB previews → submit merge (CPU array) + eval
  (GPU array) → collect (`analyze_nmerge.py --config configs/nmerge_centered_7b.json`) →
  results entry with verdicts; then decide e25 wave / ρ-sweep.
