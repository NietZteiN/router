### Target Date: 2026-07-16 (Centered-merge results — the knee moves from N≈3 to N≈64; utility survives to N≈128; crosstalk located)
- **Hypotheses / what we're testing:** H-cent-1..4 as pre-registered in
  [2026-07-15_centered-merge-design.md](2026-07-15_centered-merge-design.md).
- **Setup:** as pre-registered, two deviations recorded. (1) cr16 exact N=64 (cat rank 2064)
  deterministically OOMs at adapter-load next to the 7B bf16 base on a 44.5 GiB A40 (fp32
  file + load/cast double-buffering; 6 rows, wave 443532) → N=64 served via the svd-1024
  artifact instead (config `exact_max_n` 64→32; the e5 svd-vs-exact acceptance pair,
  |Δmu| = 0.0007 at N=64, is the validation). (2) One in-flight task
  (`cpool_N4__own82`) was scancelled during a brief 4-GPU-cap correction and re-run.
  Jobs: merges **443445** (23 CPU tasks, all clean), evals **443532** (%3 after the cap
  incident) + strays **443925/444061**. Config `configs/nmerge_centered_7b.json`
  (post-hoc note: cr16 exact_max_n now 32); CSVs `reports/centered/nmerge_*.csv` (the
  shared `reports/nmerge_*` prefix was accidentally clobbered by collect and restored from
  the e5 JSONs; `submit_nmerge.sh` collect now takes `OUT_PREFIX`). Comparators: Exp-5
  e5-mean ladder, H8 e25 ladder, anchors iso 0.3991 / base_sub 0.1703 / ft_sub8 0.9237;
  "signal %" below = (retain_prob − 0.1703) / (0.3991 − 0.1703).
- **Results:**
  Subset-conditioned retain_prob on the merged authors (primary readout):
  | N | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 20 | 32 | 64 | 128 | 200 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | e5-mean (ref) | .350 | .289 | .306 | .265 | .250 | .227 | .228 | .246 | — | — | — | ~.225 plateau |
  | cpool | **.428** | .381 | .415 | .380 | **.358** | .329 | .299 | .335 | .252 | .173⚠ | — | — |
  | cr16 | .380 | .338 | .369 | .342 | **.334** | .326 | .312 | .334 | .302 | **.281** | .240 | .211 |

  cr16 signal %: 92 → 71 (N=8) → 72 (N=20) → 58 (N=32) → **48 (N=64)** → 31 (N=128) →
  **18 (N=200, below the mean plateau ≈24)**. e5-mean: half gone by N=3, 35% at N=8.
  Standard mu (probe rows; mean regime = flat 0.459 ± 0.002; bar = 0.439): cr16 0.459–0.471
  for N ≤ 32, 0.460 (N=64 svd), 0.442 (N=128), **0.412 (N=200 — below bar AND below base
  0.426)**; cpool 0.455 → 0.437 (N=20) → 0.421 (N=32) → 0.356 (N=64⚠). retain_ppl: cr16
  monotone DOWN 7.7 → 6.0 (no overshoot anywhere); cpool up to 19.0 at N=64.
  Own-author rouge drops vs iso (e5-mean: 0.073 at N=8, flat): cr16 0.030 (N=8), 0.043
  (N=32), 0.060 (N=128), **0.120 (N=200 — recall pushed BELOW the base floor)**; cpool
  −0.026 at N=2 (above iso — mild ensemble effect), 0.041 (N=8). H-cent-4 correlation
  (drop vs e5 col(B) probe overlap, same seed-42 subsets): **r = −0.40 (cr16, p=0.025) /
  −0.44 (cpool, p=0.029)** vs the mean regime's −0.675.
  ⚠ flags: cpool svd_energy_min degrades 0.999 → 0.815 (N=2→64) — treat cpool N ≥ 32 as
  compression-confounded on top of the pre-registered estimator self-contamination; cr16
  center_energy_min 0.62 → 0.11 (rank-16 holds ~10–35% of the mean's slot energy — the
  subtracted "shared" is the top directions only, by design).
- **What worked / hypothesis verdict:**
  - **H-cent-1 — below the pre-registered bar; directional support.** N=8 subset
    retain_prob 0.358 (cpool) / 0.334 (cr16) vs the required ≥ 0.50: **the ≥2× CONFIRM bar
    is NOT met** (lift is +43%/+34% relative). But the REFUTE bar ("within ~0.05 of the
    mean curve") is not met either — the lift is real, consistent at every N, and doubles
    the retained signal fraction (71–82% vs 35% at N=8). Verdict: INCONCLUSIVE by the
    pre-registered bars, directionally supported.
  - **H-cent-2 — SUPPORTED through N≈128 (cr16), REFUTED at N=200.** The user-headline
    answer: **model utility survives centered merging at mean-regime levels (mu 0.44–0.47,
    retain_ppl falling, no overshoot) up to N≈128; at N=200 mu drops to 0.412 < base.**
    cpool survives only to N≈16–20.
  - **H-cent-3 — SUPPORTED, and the crosstalk is now measured:** re-collapse is gradual
    with **half-signal N\* ≈ 64**, crossover below the mean plateau at N ≈ 150, and
    active harm (recall < base, drop 0.120 > the extractable 0.086) by N=200. Since the
    shared component enters once (no blowup — ppl falls) and residuals are at 1× (no
    dilution), the killer at scale is inter-residual crosstalk, as predicted — and it ends
    WORSE than dilution at N=200.
  - **H-cent-4 — between the bars:** protection weakened from −0.675 to −0.40/−0.44
    (CONFIRM needed ≥ −0.2; REFUTE needed ≤ −0.5 persisting). Consistent with centering
    removing the (k−1)/k of the mean-protection while the 1× shared term retains some.
- **Observations:** (i) One-line story: **centered merging is the first composition rule
  that moves the interference curve** — knee N≈3 → N\*≈64 at zero training cost, utility
  at-or-above the mean regime through N≈128 — but it does not reach routing (0.7509), and
  at full N=200 it is worse than the mean. The §5 table gains a measured row: "selection
  nowhere, shared-once/facts-once" buys ~20× in N, not the game. (ii) cr16's falling
  retain_ppl with N is the cleanest evidence the norm-overshoot channel is fully closed by
  centering; what remains is pure crosstalk — the mechanism paper's missing causal arm.
  (iii) cpool ≥ cr16 at small N (better S estimate: 199 adapters vs top-16 directions) but
  degrades exactly as the pre-registered self-contamination + compression flags predicted —
  the deployable variant is cr16. (iv) The N=2–4 cpool points EXCEED the iso reference
  (drop < 0, retain_prob 0.43 > 0.40) — full-strength residuals give a mild ensemble gain
  before crosstalk accumulates; micro-merge tiers (H8's N≤3 zone) get slightly wider under
  centering. (v) Silent-failure checks: no NaN mu on ladder rows; probe real/world
  components consistent (analyze check passed); ppl watchlist clean except flagged cpool
  N=64; svd acceptance at N=64 (svd vs exact) reused from e5 as designed.
- **New questions / new hypotheses:** (1) ρ-sweep {8, 32, 64} at N ∈ {32, 64, 128}: does
  more aggressive shared-removal push N\* further, or does removing "shared" start deleting
  facts (center_energy says rank-16 is only the tip)? (2) e25 strong-expert centered wave
  (merges exist for subset(20)): does centering widen the e25 micro-merge zone (H8's 0.885
  at N=2) the way it widened e5's? (3) Centered + TIES sign-election on the residuals — the
  one merge-operator combination §9 doesn't cover. (4) Fold into the Path-A write-up: the
  intervention pair is now complete (centering = composition-side, positive-but-bounded;
  anchoring = training-side, clean negative).
- **Next Steps:** thread README + master index updates (this entry); plot_nmerge centered
  extension for the paper figure (mu-vs-N + signal-fraction-vs-N overlaying e5-mean/e25/
  centered); decide the ρ-sweep and e25-centered follow-ups against the §10 schedule
  (items 1 and 5 still open); scratchpad close-out.
