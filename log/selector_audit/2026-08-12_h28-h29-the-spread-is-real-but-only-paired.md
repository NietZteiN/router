# 2026-08-12 — H28/H29: the destination spread survives a finer tier, and it is only resolvable paired

Thread: `selector_audit/`. Follows
[h23-forget-quality-tracks-the-destination-not-the-forgetting](2026-08-11_h23-forget-quality-tracks-the-destination-not-the-forgetting.md),
which found `forget_quality` moving 0.62 across reroute destinations at the smoke tier and then
found the reason to distrust its own headline: at smoke the metric is a **30-vs-20** KS with ~34
achievable values and adjacent rungs ~0.10 apart, so a 0.62 spread is four questions out of thirty.
H28 asks whether the spread is real or a resolution artifact. H29 asks what interval belongs on a
published cell. The answers are *real* and *wider than the cell, unless you respect the pairing*.

## H28 — SUPPORTED. The spread survives at 4× the truth rows.

Eight arms at `--extended` (`truth_max_rows` 30 → 120), the **same reference array held fixed**
(20 rows, `results/extended/retain_tr_scores.npy`) so exactly one thing changes, plus a genuine
deletion arm measured at the same tier it is compared against. `model_utility` is **0.8009 to four
decimals in every arm**, and the route audit shows `deleted: 0` / `rerouted: 1320` in all seven
reroute arms against `deleted: 1320` / `rerouted: 0` in the deletion arm — so nothing is forgotten
anywhere except the baseline, and utility is held constant while the metric moves.

| dest | 89 | 137 | 31 | 97 | **DEL** | 33 | 79 | 88 |
|---|---|---|---|---|---|---|---|---|
| affinity | .3044 | .3382 | .2840 | .2267 | — | .2663 | .2193 | .3970 |
| smoke fq | .5789 | .7715 | .3995 | .4856 | **.5789** | .5789 | .4856 | .1561 |
| **extended fq** | **.8958** | **.8002** | **.6288** | **.6288** | **.5140** | **.5140** | **.5140** | **.3615** |

- Spread **0.5342** over [0.3615, 0.8958] — the smoke spread was 0.6154, so it narrows but does not
  collapse.
- Spearman `fq` ~ affinity = **+0.109 (p=0.82)**. H23's refutation of the expert-similarity
  hypothesis holds at finer resolution, and the sign is now nominally *positive* — the opposite of
  what similarity predicts, at a correlation indistinguishable from zero either way.
- **Six of seven arms that delete nothing score at or above genuine deletion** (0.5140); s33 and
  s79 land *exactly* on it, as s89/s33 did at smoke. Only s88 — the nearest-affinity, no-name sink
  author — scores below.
- All three smoke ties broke (s89 and s33 were both 0.5789; they are now 0.8958 and 0.5140), so
  those ties were resolution artifacts and there is real structure underneath.

**The caveat that must survive into the paper:** Spearman extended ~ smoke = **+0.620 (p=0.14)**.
The *spread* reproduces across tiers; the *per-destination ordering* does not. The defensible claim
is "an arbitrary destination choice moves the metric across most of its usable range, typically to
at-or-above genuine deletion", never "destination X beats destination Y".

## H29 — intervals, and a correction to my own first analysis

`forget_quality` is `ks_2samp(forget_tr, retain_ref).pvalue`: it collapses an array to one number
and the array was discarded, so putting a CI on a published cell meant re-serving the model at a
GPU-hour per arm. `eval_tofu.evaluate_model` now takes `dump_forget_tr` and
`eval_routed_scaffold.py` derives the path from `--out`, writing a few hundred bytes per arm
(`--no_dump_forget_tr` suppresses it). All eight arms were re-run into `results/extended_ci/` via a
new `RES_TAG` on the sweep driver, which re-points **only** the output dir — the KS reference is
loaded off the tier flag, never off `--out` — so the published cells stay untouched and the rerun
doubles as a determinism check. It reproduces **8/8** to <5e-4.

**The correction.** The first bootstrap resampled each arm's forget array independently and
compared the 0.5342 spread against the widest marginal CI (0.7240), concluding *"the destination
ordering is NOT resolvable at these sample sizes"*. That is the wrong test. The arms score
**identical rows** — the truth set is a deterministic head-slice (`max_rows` in
`get_truth_ratio_scores` breaks at `idx >= max_rows`), and inter-arm correlation is 0.88–0.94 —
so this is a paired design, and the arms differ only in destination. A marginal CI re-adds the
question-level noise that is *common* to every arm, once per arm; against that yardstick almost no
spread could ever be resolvable. Judging a paired quantity by unpaired intervals is a way of
guaranteeing a null.

The paired bootstrap draws **one** index set per iteration and applies it to all eight arms:

| quantity | marginal | **paired** |
|---|---|---|
| spread 95% CI | swallowed by a 0.72-wide cell CI | **[0.2245, 0.6975]**, median 0.4753 |
| P(spread > 0.10) | — | **0.9996** |
| P(spread > 0.25) | — | **0.9610** |

**H29 verdict: the spread is resolvable.** Both bootstraps stay in the report — "how well is this
one published cell pinned down" (marginal, CIs 0.63–0.72 wide) and "is the spread bigger than
noise" (paired) are different questions, and the marginal answer is itself worth publishing: *any
single `forget_quality` cell at these sample sizes is worth about ±0.35*.

Against genuine deletion, per arm:

| arm | Δ vs deletion | paired 95% CI | P(arm ≥ deletion) |
|---|---|---|---|
| s89 | +0.3818 | [+0.0000, +0.5915] | 0.976 |
| s137 | +0.2862 | [−0.1148, +0.5342] | 0.929 |
| s31 | +0.1148 | [−0.2223, +0.3875] | 0.797 |
| s97 | +0.1148 | [−0.1162, +0.4358] | 0.893 |
| s33 | +0.0000 | [−0.2080, +0.3360] | 0.720 |
| s79 | +0.0000 | [−0.2186, +0.3360] | 0.703 |
| s88 | −0.1525 | [−0.3512, +0.2186] | 0.333 |

Only s89's interval reaches zero at its lower bound; no arm separates from deletion outright. That
is the same conclusion the tier-to-tier rank instability reached by another route, and the two
agreeing is reassuring. **The count is the publishable form**: arms at-or-above genuine deletion is
**6/7 observed, paired median 6, 95% CI [2, 7]**. Even at the pessimistic end, two arms that delete
nothing score at-or-above real deletion — which is the §4.10 claim, and it does not depend on any
individual destination.

## Resolution: an exact statement replacing a sampled one

The grid enumeration in `bootstrap_fq.py` samples random draws and collects distinct p-values. Its
*size* is a **lower bound that grows with the number of draws** — at (120, 20) it goes 73 → 88
between 2k and 60k draws and had not converged — so the "81 achievable p-values" I was about to
publish is not a count of anything. Replaced with a quantity that needs no sampling: the KS
statistic satisfies `D = |i/n − j/m| = |i·m − j·n|/(n·m)`, so **every attainable D is a multiple of
1/lcm(n, m)** — exactly 1/120 here, i.e. one forget question. Verified empirically at 0/3000
off-lattice for (120,20), (30,20) and (100,37). What survives from the sampled grid is only what is
roughly stable across seeds and draw counts — **~30 p-values above 0.05 (29–31), median gap ~0.031**
(0.0327 at 6k draws, 0.0309 at 40k) — and even these should be quoted as approximate. The
4-decimal `forget_quality` in every table is still spurious precision; that claim was never at
risk, only the number I was going to support it with. `grid_size` is renamed
`grid_size_lower_bound` so it cannot be quoted as an enumeration by a future reader — including me.

## CSAR under `indirect` — the arms invalidated by the hash-order bug, rerun

The [seeded-transform fix](2026-08-11_indirect-was-unreproducible-and-ppl-is-the-exception.md)
invalidated every `indirect` CSAR arm. Both are back, so the transform axis is complete at full
length (`QPA 20`, n=400 per strategy):

| arm | centroid_sbert CSAR | key_tfidf CSAR | own-disclosure |
|---|---|---|---|
| gold-form | 0.3325 | 0.3650 | 0.913 / 0.933 |
| `name_stripped` | **0.4400** | **0.4175** | 0.273 / 0.273 |
| `indirect` | 0.3350 | 0.2125 | 0.383 / 0.393 |
| random destination (H17 floor) | 0.2200 | — | 0.953 |

Two readings, and the second is the one the paper wants.

1. **The 0.20 pre-registered bar is nearly uninformative and should not be quoted alone.** A
   *uniformly random* destination scores 0.2200, which clears it. Every real CSAR must be read
   against that floor: the lift is 0.11–0.22, not the raw 0.33–0.44.
2. **Own-disclosure and cross-source attribution move in opposite directions under the same
   stress.** Strip the name and the system discloses *less* of the deleted author's own content
   (0.913 → 0.273) while attributing *more* of a surviving stranger's content to them
   (0.3325 → 0.4400). The harm does not decrease when the query stops naming the victim; it
   changes character, from leaking the erased person's facts to fabricating them from someone
   else's. `indirect` sits between, and `key_tfidf` drops there (0.2125, near the random floor)
   while `centroid_sbert` holds (0.3350) — consistent with the H22 finding that lexical selectors
   lose their grip once the name is gone.

Refusal remains a rounding error everywhere (0.000–0.013): ORR ≈ 1.00 at the level of what is
*said*, across all four transforms and 1600 answers.

## Status of the pre-registration

- **H28 SUPPORTED** — spread survives the finer tier; affinity correlation still null.
- **H29 SUPPORTED, after correcting the estimator** — the spread is resolvable paired; a single
  cell is not pinned to better than ±0.35.
- **CSAR still cannot be quoted in the paper.** The pre-registration requires ~300 hand labels
  validating the classifier, and I wrote the classifier, so I cannot supply them. 400 records are
  staged in `*.label_me.jsonl` per arm.

## Defects and lessons

- **A paired quantity judged by unpaired intervals.** My own first H29 verdict said the spread was
  unresolvable; it was an artifact of discarding the pairing. The tell was the marginal CIs coming
  out 0.63–0.72 wide on cells whose point estimates were stable to <5e-4 across an independent
  rerun — a cell that reproduces exactly is not a cell with ±0.35 of *destination* noise in it.
  Reproducibility and precision were being conflated.
- **A sampled lower bound presented as a count.** The grid size was reported to a specific integer
  for two entries running (H23's "34", H28's "81") and it is a function of `n_draw`. Checking its
  stability is what caught it, and the exact lattice statement should have been the claim all
  along — it was derivable in one line.
- Both defects are the same shape as the ones this thread keeps producing: a number that looks
  precise, is cheap to check, and was not checked. The countermeasure that keeps working is
  computing the same thing a second way.
