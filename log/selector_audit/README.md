# selector_audit — auditing deletion-under-a-selector as a design pattern

**Status:** active · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) (+ `selector_audit/` for the released harness) · **Entries:** 5 (2026-08-07 → 2026-08-07)

The follow-up paper to MUSR. `router_leak/` asked what happens to MUSR's comparators when a source
is deleted; this thread asks the generic question — **constructive unlearning methods delete by
editing a selector, not the weights, and nobody has measured that substitution as a pattern.** The
planned contributions are the deletion semantics D1/D2/D3, a metric suite (ORR / CSAR / RDR /
DD-AUC / RIP / FMD), the attacks that follow from D3, and a specification a routed system must meet
before its deletion claims mean anything.

The thread opens with three pilots whose outcomes decide the paper's spine, each with a decision
rule fixed before the run: **E1** the router-side orphan probe, **E5** a trivial reroute-only
"unlearning method" scored on TOFU, and a **CSAR** prototype asking what orphan answers actually
say. Everything reuses `router_leak/`'s producers and the FAMILY NPZ CONTRACT rather than
re-measuring.

## Hypotheses — open / resolved
- **[resolved ✓ supported]** H1: the surviving selector geometry still identifies an orphan query
  with no deletion record consulted — probe AUC 0.990 / 0.976 / 0.830 at k=200, all far above the
  0.85 bar and above the author-shuffled control (0.45–0.50)
  ([2026-08-07](2026-08-07_e1-router-probe-and-preregistration.md)).
- **[resolved ✗ refuted]** H2: a learned reader extracts structure no single confidence statistic
  gives. Median lift over the best confidence detector is **+0.001** at k=200 (max +0.069 on one
  strategy); the mechanism is confidence, not a residual trace.
- **[resolved — RESTATED, strong form refuted]** H3 (granularity): the ladder is real on
  **gold-form** queries (centroid_sbert 0.564 → 0.795 → 0.984) but is largely a **lexical-identity
  artifact**. Strip the author's name from the query and it flattens to 0.560 → 0.572 → 0.623
  (+0.063 across a 20× granularity change, vs +0.367 gold-form); `key_tfidf` gains nothing at all
  (−0.021). "Per-source granularity makes deletion refusable" does not survive
  ([H3 is a lexical artifact](2026-08-07_h3-is-a-lexical-artifact.md)).
- **[open]** H6: granularity generalizes to the BEHAVIORAL family, which was the leakiest at k=10
  (0.41–0.63) and scores by running experts rather than by embedding geometry. Pending job 3191702
  ([wave](2026-08-07_behavioral-at-k200-wave.md)).
- **[open]** H7: the k=200 detectability is granularity, not the recipe. **Restated** — the
  feature-space control could never have tested it (those four routers read only the base model
  and the TOFU questions, never the experts, and produced bit-identical score matrices across
  pools differing in rank and epochs). Only the BEHAVIORAL family can answer it; pending job
  3191702 ([correction](2026-08-07_h7-correction-feature-space-is-pool-independent.md)).
- **[open]** H4 (E5): a reroute-only "method" that deletes nothing scores competitively on TOFU
  forget/utility/privacy. CONFIRM: forget_quality inside the published band.
- **[resolved ✓ supported, PROVISIONAL]** H5 (CSAR): cross-source attribution becomes common at
  per-author granularity — **CSAR 0.460** on both strategies against a 0.20 bar, and **refusal
  0.000** across all 200 orphan answers ([csar pilot](2026-08-07_csar-pilot-h5.md)). Provisional
  until the pre-registered ~300 hand labels are made; no CSAR goes in the paper before that.
- **[resolved ✗ refuted]** H9: the ladder survives name removal. It does not — detection at
  k=200 falls 0.991 → 0.623, back inside the 0.57–0.61 coarse-unit band.
- **[resolved ✓ supported, lexical only]** H10: an adversary steers routing by injecting a name.
  A single appended name captures **97.7%** of queries against `key_exact` and 31.7% against
  `key_tfidf`, but only 3.5% against `centroid_sbert`; a *substituted* name captures ~87% in every
  family.
- **[open]** H11: does the behavioral family show the same lexical dependence? It scores by
  running experts rather than matching text — the one family with a mechanism to be
  name-independent. Pending job 3191702.
- **[open]** H8: CSAR is independent of destination concentration — `key_tfidf` funnels 42/100
  orphans onto one survivor while `centroid_sbert` spreads over 37, and both give 0.460. One
  observation, not a result.

## What worked
- **The metric change matters more than the granularity change, and both are measurable on one
  artifact.** ROUGE-L moves 0.181 → 0.317/0.381 across the k-jump, which would still read as
  "mostly confabulation"; but of the answers ROUGE *still* files as confabulation at k=200, CSAR
  finds cross-source attribution in **36/75** and **39/69**. Half the "no leak here" bucket is a
  stranger's facts asserted about the deleted person — §4.10's claim, measured rather than argued.
- `analyze_router_probe.py` reproduces `rl_family_leak_table.md`'s best-confidence AUC on **all 12
  comparable cells** (k=10 d9 ×9 strategies, k=200 forget10 ×3) — the reader is faithful before any
  new number is read off it.
- Reusing the FAMILY NPZ CONTRACT made E1 a **zero-GPU, offline** experiment: three pools, 21
  strategy cells, from `results_snapshot/` alone — and then the whole granularity ladder (H3) on
  top of it, still with no GPU.
- `--lazy_adapter_cache` makes the behavioral family reachable at k=200 for the first time. The
  machinery already existed (`eval_tofu.lazify_shard_adapters`); what was missing was noticing
  that `score_norm_ppl_family` is shard-outer, so an LRU cache sees its best case there and its
  worst case in `score_logit_div`.

## What didn't / open problems
- **Today's headline did not survive its own stress test.** H3 was reported as the session's
  strongest result and is a lexical artifact. It was caught by asking what the queries look like,
  not by any gate — the numbers were correct throughout; the interpretation was not.
- **TOFU's paraphrases keep the author's name** (coverage 0.900 vs 0.895), so the obvious
  "paraphrase robustness" experiment reports a near-null on this benchmark and means nothing.
  Name-stripping is the probe that bites.
- Two attacker choices were degenerate and their runs discarded: author 0 is `key_exact`'s
  fallback shard, and author 88 is one of **18 authors with no extractable name**. Both produced
  suspiciously clean numbers, which is how they were caught.
- **The H7 feature-space control was void by construction** and I did not notice until its output
  came back identical. `key_exact`/`key_tfidf`/`centroid_sbert`/`centroid_lm` never read expert
  weights — `centroid_lm` uses the *plain base*, adapters disabled — so three pools give one
  answer. A control over a variable the measurement does not consume is not a control; any future
  pool/recipe control in this thread must name, per strategy, the input it varies.
- E1 as framed in the paper plan does not land. Its stated bar ("beat the adapter probe, 0.963")
  rests on two errors: there is no adapter-activation probe in this tree, and 0.963 is the
  author-rung tombstone **catch rate**, not a probe AUC (the sentinel AUC is 0.982).
- The probe's own headline is real but redundant with a threshold. Reported as such.

## Open ideas / next steps
- Hand-label the 300 staged records before quoting any CSAR. The precision question they settle:
  unarguable identity hits (`constance garnett award`, `aleksey ivanov`) versus rare attributes
  (one row fired on `flight attendant`). Not to be settled by tuning `max_adf` after the fact.
- The ladder has exactly **three** 7B rungs offline — k = 10, 50, 200 (`rl_family_*.npz` exist for
  no other k). A fourth would need the pool retrained: the 7B k=10 and k=50 dirs hold `results/`
  but **no shard weights**.
- Fold the ladder into `analyze_router_probe.py` so the k axis is a first-class output rather than
  three separate invocations reconciled by hand.
- If H6 confirms, §4.6's defense section becomes a statement about *coarse* partitions
  specifically: at per-source granularity a reject option is not needed, because plain confidence
  already separates orphans.

## Entries (chronological)
- [2026-08-07 — E1 router probe + E5/CSAR pre-registration](2026-08-07_e1-router-probe-and-preregistration.md) — probe fires (0.990) but adds +0.001 over confidence; the real axis is granularity.
- [2026-08-07 — behavioral at k=200 + a recipe control](2026-08-07_behavioral-at-k200-wave.md) — the memory law lifted for the shard-outer family only; logit_div stays refused for its access pattern, not its bytes.
- [2026-08-07 — CORRECTION: the feature-space recipe control cannot test H7](2026-08-07_h7-correction-feature-space-is-pool-independent.md) — bit-identical matrices across pools; H7 restated for the behavioral arm.
- [2026-08-07 — CSAR pilot](2026-08-07_csar-pilot-h5.md) — CSAR 0.460, refusal 0.000; half of what ROUGE calls confabulation is a named stranger's facts.
- [2026-08-07 — H3 RESTATED: the granularity ladder is a lexical artifact](2026-08-07_h3-is-a-lexical-artifact.md) — strip the name and 0.991 → 0.623; `key_exact` hijacked 97.7% by one injected name.
