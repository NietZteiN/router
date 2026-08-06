### Target Date: 2026-07-06 (Mode-B residual-fact-recall results + SEUF detector)

- **Hypotheses / what we're testing:** Results for the plant pre-registered in
  [2026-07-03_mode-b-plant-design.md](2026-07-03_mode-b-plant-design.md) (H1–H7). Planted shards
  2–8 + oracle trained (job 440489); RFR probed on planted-no-drop (ceiling), planted-drop-9
  (postdrop), and clean `_experts_scaf_k10` (oracle-B floor); SEUF detector run. Manifest sha
  e67608ab7bec (200 facts, 550 rows, R∈{1,2,4,8}, 25 verbatim + 25 paraphrase per R).
- **Setup:** `eval_entangled_probe.py` (sha 4f1e68cfa2ed) channels `expert_max` (max answer-prob
  over surviving single experts) + `served_key` (real OOD-aware routed+scaffold, `--delete_shard 9`)
  × surfaces orig/para; `detect_entanglement.py` (39658a39ed0a). Result JSONs under
  `Llama-3.2-1B-Instruct_entangled_k10/results/entangled/{ceiling_planted,postdrop_planted,floor_clean,detector}.json`.
  SLURM 440730/440731/440732 (probes) + 440733 (detect), seed 42. ρ = clip((post−floor)/(ceiling−floor),0,1).
- **Results:**
  - **`expert_max`, original-question surface — ρ (residual in the surviving weights):**

    | group | ceiling | postdrop | floor | ρ |
    |---|---|---|---|---|
    | R1 verbatim (control) | 0.881 | 0.150 | 0.141 | **0.012** |
    | R2 verbatim | 0.885 | 0.854 | 0.188 | **0.955** |
    | R4 verbatim | 0.894 | 0.884 | 0.208 | **0.986** |
    | R8 verbatim | 0.919 | 0.918 | 0.194 | **0.998** |
    | R1 paraphrase (control) | 0.829 | 0.090 | 0.089 | 0.002 |
    | R2/4/8 paraphrase (probed on orig) | ~0.86 | 0.13/0.22/0.13 | ~0.10 | 0.04/0.13/0.08 |

  - **Cross-surface (fact-level, H3) — `expert_max`, paraphrase-question surface:**
    verbatim-planted (original wording) facts probed on the PARAPHRASED question: ρ =
    **0.791 / 0.866 / 0.950** for R2/4/8. Paraphrase-planted facts probed on their paraphrase
    surface: ρ = **1.000** each. Controls (R1) ρ = 0.
  - **`served_key` (real composed system, author-key route) — postdrop vs floor:** identical to 3
    decimals in EVERY (R,mode) cell (e.g. R8 verbatim 0.163 = 0.163; R2 verbatim 0.155 = 0.155).
  - **SEUF detector:** AUC **0.777**, precision 0.880, recall 0.687, host-identification recall
    0.495; mean spread by R **0.389 → 0.518 → 0.643 → 0.665** (R1→2→4→8).
- **What worked / hypothesis verdict:**
  - **H1 (owner-level exactness) — SUPPORTED.** `served_key` postdrop == floor_clean to 3 decimals
    across all cells: with author-key routing the donor's queries route to base+scaffold whether or
    not the fact was planted, so the served system is byte-clean on deletion.
  - **H2 (fact survives via hosts, monotone in R) — SUPPORTED, strongly.** Verbatim `expert_max`
    ρ = 0.955/0.986/0.998 for R2/4/8 (monotone ↑), vs R1 control ρ ≈ 0.01. The fact is physically
    present in the surviving host experts' weights and more so with more owners.
  - **H3 (paraphrase ⇒ fact-level, not string-level) — SUPPORTED.** Verbatim-planted facts answer
    a PARAPHRASED question at ρ 0.79–0.95 — cross-surface transfer proves the residual is the fact,
    not the string. (Paraphrase-planted-probed-on-orig is weaker, ρ ≤ 0.13, the mirror case:
    the host learned the paraphrase wording, so the original surface matches less — expected.)
  - **H5 (detector) — PARTIALLY SUPPORTED.** Spread separates planted (R≥2) from control (R=1),
    AUC 0.777 (below the 0.9 pre-registration) with spread monotone in R; host-shard
    identification is moderate (recall 0.495). The signal is real but the threshold needs work.
  - **H7 (serving-surface split) — SUPPORTED (hard-router half).** `served_key` == floor (hard
    router hides the residual). Combined with the §9-D result (embedding routing sends orphans to a
    sibling at sim-ratio 0.98, `log/ramole/2026-07-06_routing-audit-results.md`), the embedding
    router would surface exactly this hidden residual.
  - **H4 (TOFU fq blindness):** demonstrated qualitatively — `served_key` is byte-clean (fq would
    read "deleted") while `expert_max` ρ→0.998 shows the fact survives; not separately KS-tabulated.
  - **H6 (delete-propagation):** floor_clean IS the propagation target (host shards purged);
    `expert_max` floor ≈ 0.09–0.21 ⇒ swapping flagged hosts to clean collapses ρ to ≈0 by
    construction. Not run as a separate served arm.
- **Observations:** the headline is the gap between two truths: **owner-level deletion is exact at
  the serving surface (H1/H7, served_key == floor) yet the fact is NOT erased from the model
  (H2, ρ→0.998, fact-level per H3).** This is precisely the §6 ownership≠fact distinction made
  concrete and quantified — the scope statement the paper needs. The residual grows with the number
  of owners (ρ and detector-spread both monotone in R), so scale is the threat, not the cure.
  Silent-failure checks clean: control (R1) ρ ≈ 0 (donor's own fact truly gone when no host holds
  it); ceiling answer-probs ~0.83–0.92 confirm the plant took.
- **New questions / new hypotheses:** can the detector reach AUC ≥ 0.9 with a better spread
  statistic (entropy vs 1−owner-affinity) or a soft-router (RAMoLE) affinity readout (§9-A's native
  variant)? Does an embedding-routed served arm actually surface the H2 residual the hard router
  hides (predicted yes from §9-D sim-0.98)? Both are the natural next entry.
- **Next Steps:** (optional) tune the detector + run the embedding-routed served arm to close the
  H4/H5 loop; write `reports/ENTANGLED_FACTS_REPORT_2026-07-06.md`. The core Mode-B result
  (owner-level exact, fact-level not, quantified ρ-vs-R curve) is complete.
