### Target Date: 2026-07-20 (Wave-1 trains landed · IRP CUDA bug fixed · [w5] grid+DX done · G1 evals launched)
- **Hypotheses / what we're testing:** Execution entry for the pre-registered Wave-1
  (hypotheses unchanged; see [2026-07-16_thread-preregistration.md](2026-07-16_thread-preregistration.md)).
  New result reported below: **DX1** (cancellation-vs-null, closes-or-keeps the W3
  sign-fixing idea-space) and **DX2** (owned-row energy) landed.
- **Setup:**
  - **Train smokes (07-18/19, jobs 445353–445356, 1 task/arm):** all four GREEN —
    ctrl train_loss 0.589; lin loss 2.9185→1.7538; wd(orthblock) 2.465; ds 1.9606→**0.3774**
    at d=0.005 (4,865,674 indices, 38.93 MB τ). No NaNs; artifacts saved.
  - **Full train arrays (07-19):** ctrl **445678** (20/20 ✓) · lin **445679** (20/20 ✓) ·
    wd **445680** (32/32 ✓: 16 orthblock + 16 rowslice) · ds **445681** (20/20 τ ✓) ·
    ds-unconstrained **445684** (5/5 baked ✓). All dependency-chained `afterany` the
    router_leak batch 445668–445675 (cap_guard's naive worst-case was 8; chaining keeps the
    global 4-GPU cap satisfied at every instant — the refusal message's own remedy).
  - **irpctrl twin FAILED 20/20 (job 445685), root cause = the latent IRP CUDA bug:**
    `train_lora_shard.apply_irp_projections` calls `nn.init.normal_(cuda_weight,
    generator=cpu_gen)` → `RuntimeError: Expected a 'cuda' device type for generator`.
    Same bug memsinks hit and fixed locally as `freeze_lora_a_irp` (job 443551, logged
    2026-07-15 with the explicit note "also latent in SISA IRP mode" — this run cashed
    that warning). **Fix applied to `train_lora_shard.py`** (draw seeded normal on CPU →
    copy to weight device/dtype; CPU draws bit-identical to original behavior —
    gate-tested this session, BIT-IDENTICAL ok; CLAUDE.md invariant updated).
    **Twin resubmitted: job 446357** (0-19%1).
  - **[w5] recovery:** original 445329 TIMED OUT at 12 h entirely inside DX1
    (~4.3 min/slot, reached 166/192; grid never ran). Split resubmit: **445693** grid-only
    24 h → COMPLETE 07-19 20:04 (all `sparse_{dare0p5,dare0p9,topk0p25,hash}_N{2,4,8,16}_s42`
    merges + 69-row `eval_manifest_sparse.txt`); **445694** dx1+dx2 36 h (afterany) →
    COMPLETE 07-20 10:57 (`reports/ctv_dx1_cancellation.json`, `reports/ctv_dx2_energy.json`).
  - **Verify stage (CPU):** ctrl **446358** (report-only) · wd **446359** (verify_struct) ·
    ds **446360** (locality gate); lin has no verify stage by design.
  - **G1 evals launched:** ctrl **446365** (rows 1–25: iso 20 + base 5, EVAL_ARRAY=0-24%1) ·
    wd **446366** (rows 1–37: iso 32 + base 5, 0-36%1) · ds **446367** (0-24,78-82%1 —
    iso + base + the 5 `iso_dsunc` H-ds-1 comparator rows) · lin **446368** (0-24%1,
    EVAL_TIME=03:00:00, chained afterany:446357) · irpctrl twin rows **446369**
    (eval_manifest_irpctrl.txt 0-4%1 through the lin config, chained afterany:446368).
    Worst-case concurrency ≤ 4 at every instant (lin/twin serialized).
  - Side thread (merge_mechanism gap-fill): 1B breadcrumbs λ-validation pair **446370**
    chained afterany:446367 (slot-neutral).
- **Results (DX only; G1 pending):**
  - **DX1 (cancellation vs null):** observed coord-mean |Σδ|/Σ|δ| **0.3590 / 0.1802 /
    0.0759** at N=8/32/200 vs sign-shuffled null **0.3581 / 0.1782 / 0.0712** (5 draws,
    null σ≈0.0000). Observed > null at every N.
  - **DX2 (owned-row energy):** 0.2500±0.0018 @N=4, 0.1250±0.0014 @N=8, 0.0625±0.0010
    @N=16 — exactly the ~1/N expectation.
- **What worked / hypothesis verdict:**
  - **DX1: the strict closure condition ("observed ≤ null ⇒ W3 closed") is NOT met, but
    the headroom is marginal** — +0.25%/+1.1%/+6.6% relative above the null at N=8/32/200.
    Real, minuscule cross-author sign coherence; elementwise sign-fixing could recover at
    most a few percent of cancelled mass, nowhere near the collapse scale. W3 stays closed
    for practical purposes; record the nuance rather than the binary.
  - **DX2 confirms the mechanical dilution accounting** (own-row energy = 1/N under the
    hash disjoint scheme) — the baseline against which H-w5-2's recall-∝-energy claim
    will be read once the sparse evals run.
  - Trainability signal ahead of G1: ds's smoke loss 0.377 at d=0.005 says the
    support-constrained full-FT CAN memorize a TOFU author (H-ds-1's direction), pending
    the proper iso-eval numbers.
- **Observations:** the memsinks bug ledger paid off — the 07-15 "also latent in SISA IRP
  mode" note turned a 20-task failure into a 10-minute fix with a ready-made, bit-equal
  fix pattern. Queue etiquette note: three sessions (ctv, router_leak, memadapt) are
  interleaving on the shared 4-GPU cap via afterany chains; naive cap_guard arithmetic
  over-counts dependency-serialized jobs, so chained submission is now the default
  coordination mode.
- **New questions / new hypotheses:** none new; all G1/G2 hypotheses open pending evals.
- **Next Steps:** on G1 completion — per-arm G1 verdicts (lin vs the irpctrl twin;
  kill bars: lin <0.5× twin after retry, wd <0.8× ctrl, ds <0.95× ds-unc) → G2 merges +
  ladders + cross-talk rows for survivors; [w5] sparse eval rows (69) into cap gaps;
  merge_mechanism 7B k200 gap-fill array after the 446370 breadcrumbs validation lands.
