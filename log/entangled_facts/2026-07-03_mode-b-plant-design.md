### Target Date: 2026-07-03 (Mode-B entangled-fact plant — design, harness, campaign launch)

- **Hypotheses / what we're testing:** The Mode-B replication question the gap analysis
  ([§5.1/§6/§9-A](../EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)) says no paper runs: plant the
  same fact across R owners (1 donor + R−1 hosts), drop the donor (shard 9), and measure whether
  the fact still answers via the surviving hosts. All hypotheses pre-registered BEFORE any run.
  - **H1 (owner-level exactness survives the plant):** post-drop planted world == oracle-A
    (retrain-without-owner-X). CONFIRM: probe-score deltas within the nondeterminism floor and mu
    identical pre/post drop (clean arm is 0.7509→0.7509). REFUTE: any measurable gap. *Structural
    — the same adapters composed differently, so equality should be exact; a gap is a code bug.*
  - **H2 (fact survives via hosts, monotone in R):** post-drop `expert_max` answer-prob on
    planted verbatim facts ≫ oracle-B floor and non-decreasing in R. CONFIRM: verbatim ρ ≥ 0.5
    at R≥2 and ρ(8) ≥ ρ(2). REFUTE: ρ ≈ 0 everywhere.
  - **H3 (paraphrase ⇒ fact-level, not string-level):** paraphrase-planted facts probed on the
    ORIGINAL question show ρ > 0 (planted on the paraphrase, so any original-question recall is
    fact-level transfer). CONFIRM: paraphrase ρ > 0 with verbatim ≥ paraphrase. REFUTE:
    paraphrase ≈ floor (threat narrows to verbatim replication — still reportable).
  - **H4 (TOFU metric blindness):** planted-world post-drop `forget_quality` stays high while
    RFR ρ ≫ 0 — TOFU's own metric cannot see the residual ours quantifies.
  - **H5 (detector):** NLL-affinity spread (mass off the donor's expert) separates planted
    (R≥2) from control (R=1) forget facts, AUC ≥ 0.9; host-shard identification recall ≥ 0.8.
  - **H6 (delete-propagation closes the leak):** serving with detector-flagged host shards
    swapped to clean (= oracle-B) drops ρ to ≈0.
  - **H7 (serving-surface split, §9-D tie-in):** `served_key` post-drop ≈ floor (the hard
    author-key router sends orphans to base+scaffold) while a realistic embedding router would
    leak; measured here via `served_key` (key) as the clean arm.
- **Setup:** Arm = `Llama-3.2-1B-Instruct_experts_scaf_k10` (routing+scaffold, r32/α64/e5/lr1e-4,
  seed 42). New code (sha256 12-hex): `entangle_data.py` 353e1b947363 (plant-manifest builder +
  planted loader — refuses overwrite; round-robin host placement so a host author may hold several
  planted facts), `configs/entangled_facts_1b.json` a3052f55a792, `train_lora_shard.py` 5db7effc6779
  (`--plant_manifest` flag; records plant sha + planted-row count in shard_meta), `eval_entangled_probe.py`
  4f1e68cfa2ed (RFR engine — `expert_max`/`served_key` channels × orig/para surfaces, answer-prob +
  optional ROUGE-L, ρ vs ceiling/floor), `detect_entanglement.py` 39658a39ed0a (SEUF §9-A NLL-affinity
  detector), `test_entangled_facts.py` 5ea0903edbdf (CPU gate), `submit_entangled_facts.sh` 69e0fd40c04b.
  **Plant manifest** built (CPU): 200 facts (20 donors × 10; 25 verbatim + 25 paraphrase per
  R∈{1,2,4,8}), 550 planted rows over shards 2–8 {2:80,3:80,4:80,5:79,6:73,7:81,8:77}, sha256
  e67608ab7bec…; donors R=1→180-184, R=2→185-189, R=4→190-194, R=8→195-199; hosts from retain
  authors 40–179; shards 0,1,9 symlinked byte-identical from the clean arm; oracle-B (retrain-
  without-the-fact) = the clean `_experts_scaf_k10` shards already on disk (zero training).
  CPU gate GREEN (manifest determinism, condition counts, host constraints, loader row math,
  probe partition, ρ/detector math). **Launched:** SLURM **440489** (8-task array: host shards
  2–8 + planted retain90 oracle, %4). Probe/detect stages staged in the driver, submitted after
  training lands. Seed 42.
- **Results:** *(pending — training 440489 queued; then probe (ceiling/postdrop/floor) + detector)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending; smoke check first: a planted verbatim fact must reach high
  answer-prob on its host expert pre-drop, else the plant didn't take — raise plant duplication.)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** *(pending — fill from 440489 + probe/detect result JSONs in a second dated entry)*
