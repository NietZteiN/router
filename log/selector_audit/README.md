# selector_audit — auditing deletion-under-a-selector as a design pattern

**Status:** active · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) (+ `selector_audit/` for the released harness) · **Entries:** 2 (2026-08-07 → 2026-08-07)

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
- **[resolved ✓ supported]** H3 (granularity): the confidence ceiling reported at k=10
  (AUC 0.57–0.61) is a property of coarse units, not of selectors. Monotone on both dense routers
  at constant deletion size (forget10 throughout): centroid_sbert **0.564 → 0.795 → 0.984**,
  centroid_lm **0.502 → 0.628 → 0.761** over k = 10 / 50 / 200
  ([wave](2026-08-07_behavioral-at-k200-wave.md), `reports/granularity_ladder.md`).
- **[open]** H6: granularity generalizes to the BEHAVIORAL family, which was the leakiest at k=10
  (0.41–0.63) and scores by running experts rather than by embedding geometry. Pending job 3191702
  ([wave](2026-08-07_behavioral-at-k200-wave.md)).
- **[open]** H7: the k=200 detectability is granularity, not the e25 recipe — every k=200 number
  in the repo comes from one pool. Pending job 3191703.
- **[open]** H4 (E5): a reroute-only "method" that deletes nothing scores competitively on TOFU
  forget/utility/privacy. CONFIRM: forget_quality inside the published band.
- **[open]** H5 (CSAR): cross-source attribution becomes common at per-author granularity.
  CONFIRM: CSAR ≥ 0.20 at k=200. REFUTE: < 0.10.

## What worked
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
- E1 as framed in the paper plan does not land. Its stated bar ("beat the adapter probe, 0.963")
  rests on two errors: there is no adapter-activation probe in this tree, and 0.963 is the
  author-rung tombstone **catch rate**, not a probe AUC (the sentinel AUC is 0.982).
- The probe's own headline is real but redundant with a threshold. Reported as such.

## Open ideas / next steps
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
