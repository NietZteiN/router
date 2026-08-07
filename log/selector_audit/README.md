# selector_audit — auditing deletion-under-a-selector as a design pattern

**Status:** active · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) (+ `selector_audit/` for the released harness) · **Entries:** 1 (2026-08-07 → 2026-08-07)

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
- **[open]** H3 (granularity): the confidence ceiling reported at k=10 (AUC 0.57–0.61) is a
  property of coarse units, not of selectors. Pending the K-scaling ladder at k ∈ {4,10,20,50,100,200}.
- **[open]** H4 (E5): a reroute-only "method" that deletes nothing scores competitively on TOFU
  forget/utility/privacy. CONFIRM: forget_quality inside the published band.
- **[open]** H5 (CSAR): cross-source attribution becomes common at per-author granularity.
  CONFIRM: CSAR ≥ 0.20 at k=200. REFUTE: < 0.10.

## What worked
- `analyze_router_probe.py` reproduces `rl_family_leak_table.md`'s best-confidence AUC on **all 12
  comparable cells** (k=10 d9 ×9 strategies, k=200 forget10 ×3) — the reader is faithful before any
  new number is read off it.
- Reusing the FAMILY NPZ CONTRACT made E1 a **zero-GPU, offline** experiment: three pools, 21
  strategy cells, from `results_snapshot/` alone.

## What didn't / open problems
- E1 as framed in the paper plan does not land. Its stated bar ("beat the adapter probe, 0.963")
  rests on two errors: there is no adapter-activation probe in this tree, and 0.963 is the
  author-rung tombstone **catch rate**, not a probe AUC (the sentinel AUC is 0.982).
- The probe's own headline is real but redundant with a threshold. Reported as such.

## Open ideas / next steps
- The granularity axis (H3) is the finding E1 actually surfaced and is cheap — the k-ladder npz
  already exist for k ∈ {4,10,20,50,100,200} in `results_snapshot/`.
- `--forget_author_ids`: `eval_tofu.split_eval_indices` scopes the forget set to one shard, so at
  k=200 the published cells measure 20 questions, not TOFU's 400. Blocks E5 and CSAR.

## Entries (chronological)
- [2026-08-07 — E1 router probe + E5/CSAR pre-registration](2026-08-07_e1-router-probe-and-preregistration.md) — probe fires (0.990) but adds +0.001 over confidence; the real axis is granularity.
