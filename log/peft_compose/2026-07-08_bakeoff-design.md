### Target Date: 2026-07-08 (PEFT composability bake-off — design + pre-registration)
- **Hypotheses / what we're testing:** Does any PEFT parameterization whose natural composition
  rule is not weight addition give a single composed model (no routing) that clearly beats
  composed LoRA (additive/λ-sweep band mu ≈ 0.43–0.46 at matched eval) while keeping O(1)-exact
  deletion? Motivated by merge_mechanism: the additive operator is the diagnosed failure (98%
  rank-1 pile-up, k200 job 440863; recall collapse saturating by N≈8, Exp-5; no safe merging
  regime, Exp-5b). Pre-registered:
  - **H1 (operator, not capacity):** prefix-concat serving (per-shard KV prefixes concatenated;
    attention routes implicitly) keeps own-shard recall within 0.05 of that shard's isolated
    recall at N=10 composed. CONFIRM: |iso − composed| own-shard forget_rouge ≤ 0.05 at probes
    {0,5,9}. REFUTE: drop comparable to the LoRA additive drop (~0.07–0.09) or worse.
  - **H2 (mean-like operators still dilute):** VeRA and IA³ composed_full mu > the LoRA
    additive_mean anchor on the same eval manifest, but their own-shard recall still drops
    > 0.05 vs isolated. CONFIRM: both clauses. REFUTE: either mu ≤ anchor (no benefit) or recall
    preserved (dilution is LoRA-specific).
  - **H3 (exact deletion everywhere):** for every arm, compose(all minus shard 9) ≡
    delete(compose(all), 9) (allclose in weight/vector space; prefix: byte-identical KV after
    segment drop), and the deleted composition's forget_quality recovers toward oracle level.
    CONFIRM: asserts pass + fq(composed_unlearn) ≫ fq(composed_full). REFUTE: any assert fails.
  - **H4 (capacity floor):** IA³ (gates only) and/or prefix (soft tokens only) fail the isolation
    gate — own-shard recall ≪ the LoRA isolated analog (~0.49 forget_rouge at 7B; 1B analog set
    by the LoRA iso anchor) — meaning 20-QA/author fact injection at 1B needs weight-space
    capacity. Either outcome is informative; a failed gate stops that arm's composed claims.
- **Setup (planned):** Llama-3.2-1B-Instruct, k=10 (`shard_utils.get_author_shard`), seed 42,
  TOFU full split; four arms trained by new `train_peft_shard.py` (config
  `configs/peft_bakeoff_1b.json`; epochs matched to the frozen recipe e5; per-method
  method-standard lr recorded in config): prefix (`PrefixTuningConfig`), VeRA (`VeraConfig`,
  shared `projection_prng_key`, `save_projection=True`), IA³ (`IA3Config`), DoRA
  (`LoraConfig(use_dora=True)`, composed via the existing LoRA additive path if peft accepts it —
  1-line probe first, drop arm gracefully if not). Compose by new `compose_peft.py` (materialized
  adapter dirs for VeRA/IA³/DoRA → `eval_tofu --preloaded_adapter`; prefix via new
  `prefix_concat.py` `PrefixConcatModel` + `eval_tofu --prefix_pool_dir`, labels
  `prefixcat_full`/`prefixcat_unlearn`). Evals at smoke caps, `--k 10 --forget_shard_id 9`,
  retain90 KS ref reused (method-independent, legonet precedent); per arm: iso probes {0,5,9}
  (capacity gate), composed_full, composed_unlearn (minus shard 9), routed_key_exact reference;
  anchors: base, `Llama-3.2-1B-Instruct_ft`, LoRA additive_mean on the legacy k10 pool. CPU gate
  `test_compose_peft.py` before any SLURM job; all GPU via SLURM ≤ 4 concurrent
  (`submit_peft_bakeoff.sh`, STUB=1 preview). Phase B (k=200 per-author) only behind the gate:
  composed_full mu ≥ 0.55 AND exact deletion AND capacity gate passed.
- **Results:** [pending — results entry to follow]
- **What worked / hypothesis verdict:** [pending]
- **Observations:** Design notes: (a) per-method lr differs by necessity (prefix/IA³ standard lrs
  are ~10–30× LoRA's); the bake-off compares each method's composed-vs-isolated delta and its
  absolute mu, not a recipe-controlled contrast; (b) peft 0.14 `add_weighted_adapter` is
  LoRA-only, so VeRA/IA³/prefix compose rules are new code with CPU-tested exactness identities;
  (c) IA³ compose has two variants (arithmetic mean of gates primary; geometric mean secondary —
  the true "product analog"); both keep exact deletion by recompute-from-stored-vectors.
- **New questions / new hypotheses:** (feed-forward placeholder — filled by the results entry)
- **Next Steps:** Build `test_compose_peft.py` gates → micro-smokes (2–5 steps, 1 shard/method) →
  full Phase A arrays → compose + asserts → evals → `reports/PEFT_BAKEOFF_2026-07.md` + results
  entry with SLURM job IDs.
