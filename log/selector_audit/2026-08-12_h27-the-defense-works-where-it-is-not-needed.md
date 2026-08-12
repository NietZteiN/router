# 2026-08-12 — H27: the refusal gate works exactly where it is not needed

Thread: `selector_audit/`. Closes §4.6 with H22 (which selector detects), H24/H26 (how cheap it can
be), and now the operating point. No GPU. Gates: `analyze_selector_cost --self_test` 5/5,
`analyze_router_probe --self_test` 9/9, `test_router_probe` OK.

Every §4.6 number so far has been an **AUC**, and an AUC does not say whether a gate is
deployable. A refusal gate is only usable if the legitimate traffic it wrongly refuses is
tolerable. H27 asks for the missing half: at a chosen catch rate, what fraction of **retained**
queries does the gate refuse?

`probe_arrays` now takes an optional `scores_sink` (the additive-sink convention `eval_tofu`'s
`per_example` already uses; default `None` leaves every existing call byte-identical, verified
field-by-field against the pre-change JSON), so the whole catch/false-refusal trade-off is carried
instead of the single pre-chosen 90% point. `analyze_selector_cost` emits `operating_points` per m.

## Best achievable false-refusal on retained traffic, minimised over all m

| pool | transform | c=0.50 | c=0.70 | c=0.80 | c=0.90 | c=0.95 | c=0.99 |
|---|---|---|---|---|---|---|---|
| **r32/e25** | gold | **0.000** | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 |
| (headline) | `indirect` | 0.019 | 0.048 | 0.058 | **0.125** | 0.168 | 0.236 |
| | `name_stripped` | 0.120 | 0.231 | 0.303 | **0.418** | 0.466 | 0.654 |
| r32/e5 | gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| | `indirect` | 0.058 | 0.101 | 0.135 | 0.212 | 0.245 | 0.361 |
| | `name_stripped` | 0.154 | 0.279 | 0.365 | 0.457 | 0.562 | 0.663 |
| r8/e5 | gold | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.010 |
| | `indirect` | 0.284 | 0.418 | 0.438 | 0.582 | 0.659 | 0.760 |
| | `name_stripped` | 0.361 | 0.510 | 0.587 | 0.639 | 0.688 | 0.769 |

## The §4.6 conclusion

**On gold-form queries the gate is perfect — 0.000 false refusal while catching 99% of orphans, on
both r32 pools.** On name-stripped queries, catching 90% of orphans costs **41.8%** of legitimate
traffic on the headline pool, and even catching only *half* the orphans costs **12.0%**. There is
no operating point on that curve a deployer would accept.

So the defense works exactly where it is not needed and fails where it is. A gold-form query
*names the author being asked about*; if the deletion request is honoured at all, that name is the
one piece of information the system is known to hold about the deleted party, and detection is
trivial. The adversary who matters is the one who does not name their target — and against that
adversary the gate refuses nearly half of everyone else's traffic to catch nine in ten orphans.

This reframes the whole §4.6 line. **Cost was never the binding constraint.** H24 and H26 solved it
(45× name-stripped, 90× indirect, and under `indirect` the prefilter *improves* detection), and it
turns out not to have mattered: the gate is unusable at any price under the transform that
represents a real adversary. The defense frontier is bounded by discrimination, not by compute.

The `indirect` column sits between at 0.125 @ 90% catch, which looks deployable — but H26 showed
`indirect` is nearly as **lexically self-identifying** as the gold form (`key_tfidf` top-2 holds
the true owner 96.5% of the time vs 68.5% name-stripped), because the description is built from
the target's own distinctive facts. It belongs with the gold column, not with the hard case. That
is what **H30** exists to fix, and until it is built, `name_stripped` is the only honest hard case
in this tree.

r8/e5 is worse everywhere (0.639 @ 90% catch name-stripped), consistent with its weak AUC, and
carries H24's standing warning: on that pool cheap matches full because full is poor.

## Status

- **H27 ANSWERED, and the answer is negative** — no acceptable operating point exists under
  `name_stripped` on any pool. §4.6 reports a defense that is cheap, record-free, and **not
  deployable against a non-naming adversary**.
- §4.6 is now complete as a section: H22 (only `ppl` survives) → H24/H26 (it can be made 45–90×
  cheaper, and the prefilter helps) → H27 (and it still cannot be deployed). The section's value is
  the negative result plus a concrete frontier, which is stronger than the "here is a defense"
  framing it was pre-registered as.
- One thing this does **not** license: claiming no record-free defense is possible. It bounds
  `ppl`-as-refusal-gate at k=200 on these three pools under one transform family. A detector with a
  different signal, or a system that abstains on low margin rather than thresholding a probe, is
  untested here.
