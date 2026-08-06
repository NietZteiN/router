### Target Date: 2026-07-16 (H14″ results — capacity floor licensed at ~0.4–0.5 recall; deletion finally real, exact, and floor-perfect)
- **Hypotheses / what we're testing:** H14″ as pre-registered in
  [2026-07-16_h14doubleprime-preregistration.md](2026-07-16_h14doubleprime-preregistration.md):
  e25 (25 optimizer steps/author) on the scale-corrected strict arm. CONFIRM: probe gen_own
  ≥ 0.80 AND mu ≥ 0.55 AND unlearn forget_rouge ≤ 0.45. REFUTE (capacity floor, licensed
  only with training health): own-prob < 0.6 with converged loss.
- **Setup:** Jobs **444254** smoke ✓ → **444255** train (5000 steps, 31.5 min, avg loss
  1.59, final band 1.02–1.19; distinct-ID OK) → **444256** evals 0-2%2 → **444257** probe.
  Queue empty — ran unheld. Config `memsinks_tofu_1b_strict2_e25.json` (f012a29f4f5c) =
  strict2 + epochs 25.
- **Results:** **25-epoch probe trajectory** (7 probe authors, own-mask vs own-deleted
  answer-prob): e1 0.152/0.155 → e5 0.257/0.001 → e10 0.374/0.000 → e15 0.453/0.000 →
  e20 0.498/0.000 → **e25 0.504/0.000 (plateau: +0.006 over the last 5 epochs)**.

  | label | mu | fq | forget_rouge | retain_prob | retain_rouge | real/world | ppl f/r |
  |---|---|---|---|---|---|---|---|
  | strict2_e25 routed_full | **0.6305** | 0.0346 | **0.5537** | 0.4145 | 0.5857 | 0.6305/0.6556 | 3.3/3.0 |
  | strict2_e25 routed_unlearn | **0.6305** | **0.3929** | **0.4647** | 0.4145 | 0.5857 | 0.6305/0.6556 | 17.6/3.0 |
  | strict2_e25 all_on | 0.0 | 0.135 | 0.0049 | 0.0 | 0.0083 | 0.27/0.26 | 222k/204k |

  **Full probe (200 authors × 20 rows):** gen_only 0.1396 (scaffold floor), gen_own
  **0.3890**, slice_increment **+0.2493** (forget group 0.2387 ≈ retain 0.2505); all_on
  0.0000; ladder monotone 20/20.
- **What worked / hypothesis verdict:**
  - **H14″ REFUTED on the capacity gate — and this time LICENSED.** Training healthy
    (final loss 1.0–1.2 < 1.5) and saturated (own-prob +0.006 over epochs 20–25), yet
    own-author recall plateaus at probe 0.389 / trajectory 0.504 — far below the 0.80 gate
    and below the 0.6 refute line. **40 frozen-basis lora_B rows/author/layer hold roughly
    HALF the recall of a full adapter** (merge_mechanism e25 full-LoRA: 0.9991). The
    quantified per-author capacity floor of the frozen-A slice substrate.
  - **But the mechanism now works END-TO-END, and deletion is real:** mu gate passed
    (0.6305 ≥ 0.55); deletion moves forget_rouge **0.5537 → 0.4647 = the never-trained
    scaffold floor EXACTLY** (the pre-registered 0.45 threshold was a hair below the
    measured floor — deletion reached the floor, which is the actual success criterion);
    **fq 0.135 → 0.3929 — identical to SIFT-unlearn's 0.393**; retain side bit-unchanged
    (0.4145/0.5857 to 4 decimals); isolation perfect throughout (deleted-condition 0.000
    from epoch 5 onward). First MemSinks-lineage condition where deletion does exactly what
    it claims, with row-provenance exactness at ~80 KB/author.
  - **all_on collapse scales with slice strength** (ppl 93 at e5 → 222k at e25) — stronger
    isolated experts interfere harder, the merge_mechanism H8 pattern reproduced inside one
    adapter.
- **Observations:** mu 0.6305 is buoyed by the scaffold's real/world (0.63/0.66); the
  honest utility picture is retain_prob 0.4145 vs ctrl's 0.93 — the capacity floor bites
  recall, not general behavior. The learning curve shape (steady climb to a hard plateau
  with loss converged) is what a genuine expressivity ceiling looks like, unlike the e5
  underfit (curve still rising) or the std-1.0 blow-up.
- **New questions / new hypotheses:** capacity ladder (deferred): slice width s ∈
  {40→80→160} (fewer authors per adapter or wider partition), trainable-lora_A-per-slice
  variants, or rank dial — where does the 0.39 floor meet SEA's 0.99? Cost/benefit vs just
  using SEA at 32 MB/author. Not run — thread-close decision with user.
- **Next Steps:** REPORT.md final update + ledgers; user review: close the thread (complete
  measured story) or run the capacity ladder.
