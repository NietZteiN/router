# selector_audit — auditing deletion-under-a-selector as a design pattern

**Status:** active · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) (+ `selector_audit/` for the released harness) · **Entries:** 14 (2026-08-07 → 2026-08-10)

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
- **[resolved ✓ supported, PROVISIONAL]** H6: granularity generalizes to the BEHAVIORAL family —
  activation_norm 0.412 → **0.877**, attn_norm 0.533 → **0.758** at k=200 (r8 arm,
  [H6](2026-08-07_h6-behavioral-at-k200.md)), self_check 3/3. **Provisional**: these are
  gold-form numbers, and gold-form is exactly what H3 discredited. Not to be quoted without H11.
- **[resolved ✓ supported]** H7/H18: the behavioral family IS recipe-dependent —
  `activation_norm` reads **0.877 on r8 but 0.608 on the headline e25 pool**, `attn_norm` 0.758 vs
  0.554 ([H18](2026-08-10_h18-recipe-dependence-and-a-consolidator-bug.md)). And **H11 replicates
  on both pools** (name-stripped → 0.495/0.498 and 0.519/0.507, i.e. chance). So H6's strong form
  was an r8 artifact on top of a gold-form artifact: on the headline pool the behavioral family
  never had good detectability even with names.
- **[resolved ✓ supported]** H4 (E5): a reroute-only "method" that deletes nothing scores
  **better** forget_quality than genuine deletion — **0.6789 vs 0.5789** at identical
  model_utility (0.7921) and forget_rouge within 0.02. TOFU's forget metric prefers the stranger
  ([overnight](2026-08-10_overnight-campaign-results.md)). The §4.10 result.
- **[resolved ✓ supported, PROVISIONAL]** H5 (CSAR): cross-source attribution is common at
  per-author granularity — **CSAR 0.333 / 0.365 on the full 400** against a 0.20 bar, with
  **refusal 0.000** across all 800 answers
  ([full run](2026-08-07_csar-full-400-and-a-sampling-bias.md)). The pilot's 0.460 was a
  question-order artifact and must not be quoted. Still provisional until the pre-registered
  ~300 hand labels are made.
- **[resolved x refuted]** H16: CSAR is a lexical artifact like the H3 defence. It is not — under
  name-stripping CSAR RISES (0.530 / 0.500 vs 0.460 gold-form) while routing accuracy falls
  0.966 -> 0.343 ([CSAR survives](2026-08-07_csar-survives-name-stripping.md)). The defence was
  an artifact of the name; the harm is not.
- **[resolved ✓ supported, qualified]** H17: a uniformly random surviving expert still produces
  **CSAR 0.220** against a real router's 0.333 (400 questions). The harm cannot be engineered to
  zero by improving the selector, but two thirds — not all — of it survives with no router. This
  retires H8's stronger "independent of the router" form.
- **[open]** H15: is CSAR a function of question TYPE? Identity questions score 0.460 and later
  ones 0.290/0.333, so it may largely be 'the router supplies the wrong name when asked for a
  name'.
- **[resolved ✗ refuted]** H9: the ladder survives name removal. It does not — detection at
  k=200 falls 0.991 → 0.623, back inside the 0.57–0.61 coarse-unit band.
- **[resolved ✓ supported, lexical only]** H10: an adversary steers routing by injecting a name.
  A single appended name captures **97.7%** of queries against `key_exact` and 31.7% against
  `key_tfidf`, but only 3.5% against `centroid_sbert`; a *substituted* name captures ~87% in every
  family.
- **[open]** H6/H7/H11 all pend job **3191948** — the first `sw-beh` submission (3191702) was
  killed by a defect in my own lazy-cache support, not by a result
  ([defect record](2026-08-07_lazy-cache-broke-the-serving-norm.md)).
- **[resolved ✓ supported]** H11: the behavioral family's detectability is lexical too —
  activation_norm **0.877 → 0.495**, attn_norm **0.758 → 0.519**, ppl **0.993 → 0.647** under
  name-stripping ([overnight](2026-08-10_overnight-campaign-results.md)). Two of three at chance.
  **This completes the picture: no selector family — lexical, dense or behavioral — detects
  orphans once the query stops naming the deleted source.** H6's 0.877 was a gold-form artifact,
  as its entry warned.
- **[resolved ✓ supported]** H12: queries belonging to no source spread flat — margin 0.022 vs
  0.186 retained — so confidence separates strangers at AUC 0.983–1.000
  ([OOD](2026-08-07_ood-queries-and-the-oracle-gate.md)).
- **[resolved ✓ supported]** H13/H14: without a gate, general queries land on a source's expert,
  and under `key_tfidf` **one unit (88) absorbs 68% of real-author, 45% of world-facts and 19% of
  orphan queries**. Author 88 is one of 18 authors with **no extractable name** — the universal
  sink is the least identifiable source. `centroid_sbert` has no such sink.
- **[open]** H18: does H11 replicate on the `r32 e25` pool? The r8 pool is the weakest of the
  three and both r32 arms timed out. This is H7 restated with the arm that can answer it.
- **[open]** H8: CSAR is independent of destination concentration — `key_tfidf` funnels 42/100
  orphans onto one survivor while `centroid_sbert` spreads over 37, and both give 0.460. One
  observation, not a result.

## What worked
- **Third instance of one pattern**: detectability (H3/H11) and now LOCALITY are reassuring only
  while the query names the source, while attribution (H16) is not. RDR is 0.000 at k=200 on
  gold-form queries and **0.092** name-stripped — worse than the 5.8% published at k=10 that fine
  units were supposed to improve on ([magnet/RDR](2026-08-10_magnet-saturation-and-rdr.md)).
- **§4.2's magnet saturation is REFUTED** as a general claim — the busiest share FALLS as more
  sources are deleted (0.550 → 0.130), because each deleted author has a different nearest
  survivor. It holds in exactly one regime: `key_tfidf` on name-free descriptive queries
  saturates at **0.902, n_eff 1.2**, onto the same nameless unit 88 that absorbs OOD traffic.
- **The two headlines move in OPPOSITE directions under the same stress**, which is the most
  useful thing the day produced. Strip the author's name and the defence collapses
  (0.991 -> 0.623) while the harm strengthens (0.460 -> 0.530). A deployment where people ask
  about someone without naming them is exactly where refusal stops working and attribution gets
  worse.
- **One mechanism explains the whole day.** Orphans and strangers look alike to the selector
  (orphan-vs-retain AUC 0.984/0.992 beside OOD-vs-retain 0.997/1.000) because the discriminator is
  not "was this deleted" but **"does this query name a source I still hold"**. That is why
  detection looked excellent at k=200 on gold-form queries and collapsed to 0.62 once the name was
  stripped: without the name, *retained* queries stop being confident too.
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
- **`consolidate.py` mispaired CSAR files with generation dumps** (`stem.split("_centroid")[0]`
  sent the `-random` runs to the gold-form dump), so the H15 rows for those runs described
  another run's questions and the `random` strategy silently vanished from the table. Fixed by
  exact-stem matching plus a refusal when the two strategy sets disagree. Fourth silent-numbers
  defect of the campaign, and like the others it produced a plausible table rather than an error.
- **The MIA privacy column is not trustworthy.** All three arms report byte-identical AUCs
  despite serving different models, and `attack_mia` records no route statistics — most likely
  every query fell to the OOD path and all three measured the base model. Needs a route-stats
  assert like `eval_routed_scaffold` has before it is run again, and must not be quoted.
- **Two behavioral waves timed out at their 6 h wall** (r32 arms only; r8 finished both times).
  Three arms each pulling 200 r32 adapters over NFS concurrently is the bottleneck — the fix is
  one arm per job, not more GPUs.
- **`--questions_per_author` head-slices each author's QUESTIONS** — the same head-slicing bias
  it was built to avoid on the author axis. TOFU puts identity questions first and those are the
  most attribution-prone, so the CSAR pilot over-read by ~0.13. `--question_sample random` added.
- **Today's headline did not survive its own stress test.** H3 was reported as the session's
  strongest result and is a lexical artifact. It was caught by asking what the queries look like,
  not by any gate — the numbers were correct throughout; the interpretation was not.
- **TOFU's paraphrases keep the author's name** (coverage 0.900 vs 0.895), so the obvious
  "paraphrase robustness" experiment reports a near-null on this benchmark and means nothing.
  Name-stripping is the probe that bites.
- **My route audit destroyed the arm it audited.** It compared a forward-pass counter (630) to a
  question count (400) — wrong, because one question is forwarded several times per eval — and it
  raised *before* `json.dump`, discarding 1h15m of computed metrics. Now audits the distinct
  AUTHORS on each path, and writes the artifact before failing
  ([defect](2026-08-07_route-audit-ate-its-own-arm.md)). An audit that destroys the artifact it
  audits is worse than no audit.
- **`_lora_b_norm` returned 0.0 for every non-resident adapter under the lazy cache** — it
  filtered on `lora_B` membership before activating, so zero hooks registered and
  `ActivationRouter.route` collapsed onto the resident shard. Caught by the audit's own
  self_check (gap 0.516, flagged as a real disagreement). I had already made this exact fix to
  the batched twin and missed the serving copy. Score matrices were unaffected.
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
- [2026-08-10 — H18: recipe dependence, and a consolidator bug](2026-08-10_h18-recipe-dependence-and-a-consolidator-bug.md) — H11 replicates; H6's 0.877 was r8-specific; the H15 table was reading the wrong dump.
- [2026-08-10 — magnet saturation refuted; locality is lexical too](2026-08-10_magnet-saturation-and-rdr.md) — RDR 0.000 gold-form vs 0.092 name-stripped; the magnet forms only for a lexical router on name-free queries.
- [2026-08-10 — the overnight campaign: H11, H17, H4](2026-08-10_overnight-campaign-results.md) — detection is lexical in EVERY family; a reroute-only method out-scores real deletion.
- [2026-08-07 — H6: the behavioral family at k=200](2026-08-07_h6-behavioral-at-k200.md) — 0.412 → 0.877, and the self_check that killed the last run passes 3/3.
- [2026-08-07 — CSAR survives name-stripping, and rises](2026-08-07_csar-survives-name-stripping.md) — the defence was lexical, the harm is not.
- [2026-08-07 — CSAR on the full 400, and the sampling bias the pilot had](2026-08-07_csar-full-400-and-a-sampling-bias.md) — 0.333/0.365, not 0.460; the head-sliced questions are the identity-shaped ones.
- [2026-08-07 — DEFECT: the route audit ate its own arm](2026-08-07_route-audit-ate-its-own-arm.md) — pass counters are not question counts; write before you raise.
- [2026-08-07 — queries that belong to no source, and the OOD gate that is an oracle](2026-08-07_ood-queries-and-the-oracle-gate.md) — strangers are flat and detectable; one nameless author absorbs 68% of them.
- [2026-08-07 — DEFECT: the lazy cache zeroed the serving norm](2026-08-07_lazy-cache-broke-the-serving-norm.md) — non-resident adapters scored 0.0; caught by self_check, fixed, arms resubmitted.
- [2026-08-07 — H3 RESTATED: the granularity ladder is a lexical artifact](2026-08-07_h3-is-a-lexical-artifact.md) — strip the name and 0.991 → 0.623; `key_exact` hijacked 97.7% by one injected name.
