### Target Date: 2026-08-10 (E5's fourth arm revises H4; the feature-space "recipe ablation" is vacuous; a cache race)

Fifth entry today, from widening the wave to use more GPUs. Two of the three findings here are
corrections to things already written down.

- **Hypotheses / what we're testing:** **H4** with its missing arm — does a method that deletes
  nothing and only reroutes score competitively on TOFU? And **H22** — is name-stripped `ppl`'s
  rising signal real, tested with the `indirect` transform on the feature family.

- **Setup:** job 3201916 (E5 `reroute42` requeued after the dead-node TIMEOUT, 27:46 on a healthy
  node), jobs 3201977/3201978/3201979/3202035 (feature family x three transforms x three pools),
  job 3201980 (behavioral `indirect`, serialized). 9 GPUs concurrent against a 16-GPU / 6-job
  association cap.

- **Results — E5, all four arms, k=200 e25, forget10:**

  | arm | forget_quality | model_utility | forget_rouge | forget_ppl |
  |---|---|---|---|---|
  | `oracle_full` (nothing deleted, normal routing) | 0.0078 | 0.7921 | 1.0000 | 1.05 |
  | `delete` (D1 — experts genuinely dropped) | **0.5789** | 0.7921 | 0.3998 | 15.21 |
  | `reroute s0` (deletes NOTHING) | **0.6789** | 0.7921 | 0.4197 | 14.74 |
  | `reroute s42` (deletes NOTHING) | **0.3995** | 0.7921 | 0.4154 | 16.43 |

  `route_audit.ok = True` on all three deletion arms. `model_utility` is identical across all
  four to four decimals, as it must be — retained routing is untouched.

- **What worked / hypothesis verdict:**
  - **H4 — SUPPORTED, but the 08-10 entry's phrasing was too strong and is corrected here.**
    That entry said reroute scores *better* than genuine deletion (0.6789 vs 0.5789). With the
    fourth arm in hand that is destination-dependent: the same method pointed at shard 42 scores
    **0.3995**, *below* deletion. So "reroute beats deletion" is not the finding.
  - **The finding is better than that.** The two reroute arms **bracket** the deletion arm —
    0.3995 and 0.6789 around 0.5789 — with `forget_rouge` within 0.02 of it and identical
    utility. TOFU's `forget_quality` therefore cannot separate *the deleted authors' experts are
    gone* from *they are still loaded and merely not addressed*, and which side of genuine
    deletion you land on is set by an **arbitrary choice of destination expert**, not by whether
    anything was forgotten. A metric whose verdict swings 0.28 on an implementation detail
    irrelevant to forgetting is the §4.10 argument, and this is the stronger form of it.
  - This is exactly what the second fixed-survivor arm was pre-registered to test — its stated
    role was "shows the number is not shard-0-specific". It showed the opposite. The arm earned
    its cost by contradicting the reason it was run.

- **Observations:**
  - **The feature-space "recipe ablation" measures nothing, by construction.** `feat_e5r32` and
    `feat_e5r8` produce **byte-identical** score matrices: `maxabsdiff = 0.000e+00` on
    `key_tfidf`, `centroid_sbert` and `centroid_lm`. The mechanism is not subtle once stated —
    feature-space centroids are built from each shard's **training text** through the base model
    with `model_ctx.disable_adapter()` (router_family_audit.py:504), so no pool weight is ever
    read, and the three pools share one data partition. The driver's header claims these arms
    "separate 'granularity causes detectability' from 'the e25 recipe does'"; they cannot, and
    only the behavioral family ever could — which is what H7/H18 actually used. Consequence for
    the paper: **any per-pool table of feature-space numbers is the same column repeated**, and
    future waves need ONE feature arm per transform, not three. I noticed this only because I
    packed the three arms into one job and saw three identical self-check lines.
  - **DEFECT — the centroid cache was not concurrency-safe, and running arms in parallel is the
    normal case.** Job 3201979 lost its `feat_e25` arm to
    `ValueError: cannot reshape array of size 0 into shape (4096,)`.
    `build_centroids_cached` did `np.save(cp, c)` straight to the final path; np.save creates the
    file and *then* fills it, so a sibling arm's `os.path.exists(cp)` landed in that window and
    read zero bytes. Arms differing only in `--query_transform` legitimately share one cache
    (centroids are transform-independent), so this fires whenever two of them run together —
    which is precisely what "use more GPUs" means here. Fixed by publishing through a
    process-unique temp and `os.replace` (atomic within a filesystem), plus recompute-on-unreadable
    for entries already truncated by a killed job. Gated by
    `test_router_family.test_centroid_cache_is_concurrency_safe`, which poisons an entry with a
    zero-byte file and requires recovery, checks no `.tmp.<pid>` droppings survive, and asserts on
    the source that the publish is still a rename. No poisoned files remain on disk (the racing
    writer completed), so nothing already computed is suspect.
  - **The dead-node diagnosis is now confirmed rather than inferred.** `reroute42` completed in
    **27:46** on `xe8545-a100-27`, against 4 h producing zero forward passes on
    `xe8545-a100-22` — same arm, same code, same data.
  - Worth recording as a pattern: **parallelism is a test.** Packing three arms into one job
    surfaced both a latent race and a vacuous experiment design within five minutes, neither of
    which the serialized runs could have shown.

- **New questions / new hypotheses:**
  - **H23:** does the reroute bracket widen with the destination's dissimilarity to the deleted
    author? s0 and s42 are two arbitrary points; if `forget_quality` tracks
    survivor-to-deleted-author distance, the metric is measuring *expert similarity* and not
    forgetting at all, which would sharpen §4.10 further. Cheap — it reuses the existing arm at
    a handful of destinations, one GPU each and ~28 min apiece.
  - The `--reroute_to` sweep above should NOT be run on all 200 survivors; a stratified dozen
    (nearest, farthest, median by centroid distance) answers it at 1/16 the cost.
