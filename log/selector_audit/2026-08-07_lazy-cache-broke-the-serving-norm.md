### Target Date: 2026-08-07 (DEFECT: the lazy adapter cache silently zeroed the serving norm)

Sixth entry today. Records a defect in the change made in
[behavioral-at-k200-wave](2026-08-07_behavioral-at-k200-wave.md), found by that run's own
faithfulness gate. No result is retracted — nothing had been reported from the affected arms.

- **Hypotheses / what we're testing:** none. This is a defect record.

- **Setup:** job **3191702** (`sw-beh`), the three k=200 pools, `--lazy_adapter_cache 8`. Died on
  the `activation_norm` self-check of the `r8` arm:

  ```
  [self_check activation_norm] row 63: matrix argmax 94 != router.route 0
  (q='Can you name some notable awards that Rajeev Majumdar has been honored')
  scores 6.4569 vs 5.9404, gap 0.516 > 2% band (real disagreement, not a bf16 tie)
  ```

- **Results:** `router._lora_b_norm` selects modules with

  ```python
  if not (hasattr(module, "lora_B") and adapter_name in module.lora_B): continue
  ```

  **before** calling `model.set_adapter(adapter_name)`. Eagerly that is fine — every adapter is
  resident, so every module matches. Under `lazify_shard_adapters` a non-resident adapter is in
  **no** module's `lora_B`: the loop registers zero hooks, the forward runs, and the function
  returns `sum([]) == 0.0`.

  So every non-resident shard scored **exactly zero**, and `ActivationRouter.route` returned
  whichever adapter happened to be resident — shard 0, the one loaded at construction. That is
  precisely what the gate reported.

- **What worked / hypothesis verdict:**
  - **The score matrix was never affected.** `score_norm_ppl_family` calls `set_adapter` and
    *then* registers hooks; `test_high_k_behavioral_guard` already asserted that ordering. Only
    the reference route the self-check compares against was wrong.
  - **The gate did its job.** A 2%-band tie tolerance meant this surfaced as a real disagreement
    rather than being absorbed as numerical noise, and it fired on the first affected strategy.
    `--self_check 3` was enough — three queries found it.
  - Fix: activate before inspecting `lora_B`, the same change already made to the batched twin in
    `router_family_audit.lora_b_norms_batch`. **I fixed the copy the audit calls and missed the
    copy the router serves** — the batched path was the one I was reading while writing the lazy
    support, and the serving path never entered my attention.

- **Observations:**
  - The failure mode was silent-by-construction: no exception, no NaN, just a zero where a norm
    belonged. Every downstream number would have looked plausible. Only a cross-check against an
    independent path could catch it, which is an argument for keeping `--self_check` on every
    behavioral run rather than treating it as a startup cost.
  - The new gate is **functional, not a source-order assertion**: it saves the stub pool, reloads
    it under a cache smaller than the pool, and requires `_lora_b_norm` to agree eager vs lazy on
    every shard including non-resident ones. Against the old code it returns 0.0 and fails. A
    source-order check would have passed the moment someone reordered the lines back.
  - Arms were cancelled rather than left running: `e25` and `r32` were still loading and would
    have reached the same assertion hours later, losing their `ppl` work in the same crash.

- **New questions / new hypotheses:**
  - Are there other consumers of the pool that assume eager residency? `--lazy_adapter_cache` is
    also passed by `eval_routed_scaffold`, `dump_generations_routed` and `attack_mia`. Those go
    through `set_adapter` normally and do not introspect `lora_B`, so they are structurally
    unaffected — but that is reasoning, not a test, and the reasoning is what failed here.

- **Next Steps:** re-read the three arms when job **3191948** lands. Consider a generic gate that
  runs each routing strategy once under a lazy cache and once eagerly and requires identical
  routes — the property that actually matters, stated once instead of per call site.
