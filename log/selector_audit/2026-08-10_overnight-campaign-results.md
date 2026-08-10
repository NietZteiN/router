### Target Date: 2026-08-10 (the overnight campaign: H11, H17, H4 — and three failures)

Results of the campaign queued on 2026-08-07 (`submit_overnight_selector.sh`, jobs 3192333–3192339,
3192575, 3192583). Reads the consolidated report at
`tofu_sisa_lora/reports/SELECTOR_AUDIT_OVERNIGHT.md`.

- **Hypotheses / what we're testing:** H11 (is the behavioral family's detectability lexical?),
  H17 (the CSAR floor under a random destination), H4 (does a reroute-only "method" score like
  real deletion?), plus full-length re-runs of the CSAR transforms.

- **Setup:** all sbatch, unattended. `--query_transform name_stripped` on the behavioral audit;
  `random` pseudo-strategy for the floor; `QPA 20` (all 400 orphan questions) for every CSAR arm.

- **Results:**

  **H11 — the behavioral family is exactly as lexical as the feature-space one** (r8 pool, k=200,
  best-confidence AUC):

  | strategy | k=10 | k=200 gold-form | **k=200 name-stripped** |
  |---|---|---|---|
  | `activation_norm` | 0.412 | 0.877 | **0.495** |
  | `attn_norm` | 0.533 | 0.758 | **0.519** |
  | `ppl` | 0.998 | 0.993 | **0.647** |

  Two of the three are at chance. Even `ppl`, saturated at both granularities on gold-form
  queries, falls to 0.647.

  **H17 — the CSAR floor with no router at all** (same run, 400 questions):

  | destination | CSAR |
  |---|---|
  | `centroid_sbert` (real router) | 0.333 |
  | **uniformly random surviving expert** | **0.220** |

  **H16 at full length** — CSAR still rises under name-stripping: 0.333 → **0.440**
  (`centroid_sbert`) and 0.365 → **0.417** (`key_tfidf`). `indirect` sits below gold-form
  (0.323 / 0.203), as at `QPA 5`.

  **H4 (E5) — the reroute-only arm.** Both arms share configuration except the reroute:

  | arm | mu | **forget_quality** | forget_rouge |
  |---|---|---|---|
  | genuine deletion (`routed_oodgate_oracle`) | 0.7921 | **0.5789** | 0.400 |
  | **reroute to shard 0, nothing deleted** | 0.7921 | **0.6789** | 0.420 |

- **What worked / hypothesis verdict:**
  - **H11 — SUPPORTED (detectability is lexical for the behavioral family too).** This completes
    the picture: **no selector family — lexical, dense, or behavioral — can detect orphans once
    the query stops naming the deleted source.** H6's provisional 0.877 was a gold-form artifact,
    exactly as its entry warned it might be. The granularity claim is dead across the taxonomy,
    not merely for the feature-space half.
  - **H17 — the floor is high but routing quality does matter.** A uniformly random expert still
    produces **0.220** attribution, two thirds of a real router's 0.333. So the harm cannot be
    engineered to zero by improving the selector, but it is not wholly independent of it either.
    That is a more careful claim than the qpa5 data supported, and it retires H8's stronger form.
  - **H4 — SUPPORTED, and this is the §4.10 result.** A "method" that **deletes nothing** and only
    reroutes scores **better** forget_quality than genuine deletion (0.6789 vs 0.5789) at identical
    model_utility, with forget_rouge within 0.02. TOFU's forget metric cannot distinguish "the
    source is gone" from "a stranger answers for it" — it prefers the stranger.

- **Observations:**
  - The two headline directions now hold at full length and across every family: **strip the name
    and detection collapses to chance while attribution rises.**
  - **The MIA privacy column is not trustworthy and must not be quoted.** All three arms report
    byte-identical AUCs (loss 0.3356, min_k 0.3437, min_k++ 0.5159, zlib 0.2858) despite serving
    different models. `attack_mia` records no route statistics, so the most likely explanation is
    that its prompt format does not match what `q2author` expects and **every** query fell to the
    OOD path, serving the base in all three arms. That would make the column a measurement of the
    base model, not of the arms. Needs a route-stats assert of the kind
    `eval_routed_scaffold` now has before it is run again.
  - **H15's regex is a weak instrument.** It flags only 21/400 questions as identity-shaped, and
    the split disagrees across routers (centroid_sbert 0.571 vs 0.319; key_tfidf 0.095 vs 0.380 —
    the opposite direction). The position-based q0–q4 vs q5–q19 split was much stronger evidence
    and should be preferred; H15 stays open with a better detector needed.

- **Three failures, two of them my sizing:**
  - `sw-beh` (gold-form) and `sw-beh_name_stripped` both **TIMEOUT at their 6 h wall**. The r8 arm
    of each finished (which is why H6 and H11 are answerable); the `r32 e25` and `r32 e5` arms did
    not. Cause is almost certainly NFS: three arms each pulling 200 r32 adapters concurrently, at
    ~4× the bytes of r8. Fix is one arm per job (`%1` throttle) and a 12 h wall, not more GPUs.
  - `e5-reroute` **TIMEOUT at 4 h** with four packed arms; the dedicated `routed_oracle_del_f10`
    and `routed_reroute_f10_s42` arms never wrote. H4 is answerable only because
    `routed_oodgate_oracle` is configuration-identical to the delete arm.
  - `sa-consolidate2` **FAILED**: `unrecognized arguments:` — a stray empty argument from my
    heredoc line-continuation. The first consolidation had already produced the report, so nothing
    was lost, but the H11 arm is absent from it.

- **New questions / new hypotheses:**
  - **H18:** does H11 replicate on the `r32 e25` pool? The r8 pool is the weakest of the three, and
    a recipe-dependent answer would matter. This is H7 restated with the arm that can answer it.

- **Next Steps:** re-run the two timed-out behavioral waves one arm per job at a 12 h wall; add a
  route-stats assert to `attack_mia` before trusting any privacy number; fix the consolidate
  invocation; give H15 a real question-type detector.
