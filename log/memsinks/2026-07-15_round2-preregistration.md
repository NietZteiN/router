### Target Date: 2026-07-15 (Round-2 pre-registration — Phase D diagnosis + E3 strict-isolation arm)
- **Hypotheses / what we're testing:** Pre-registered BEFORE any Round-2 SLURM job (user scope
  2026-07-15: diagnose + strict-isolation fix arm only; e25 + shared-starvation arms designed
  but deferred). Numbering continues the thread ledger. All smoke tier, seed 42, ±0.05 band;
  reference: ctrl 0.6438/forget_rouge 0.9425, memsinks_full 0.4373/0.6936, dropall
  0.6399/0.8726, oracle forget_rouge ≈ 0.39–0.40.
  - **H9 (routed serving):** serving each TOFU query under its author's TRAINING mask
    (gen + own slice; OOD → gen-only) removes the self-interference. CONFIRM:
    mu(memsinks_routed_full) ≥ 0.59 AND ≥ 0.62; retain_ppl ≤ 1.6. REFUTE: mu < 0.54.
    Wrapper-sanity floor: mu < 0.50 → STOP (suspect wrapper bug) before E3 trusts the path.
  - **H9-del (routed deletion):** `memsinks_routed_unlearn` (forget10 authors → gen-only).
    EXPECTED (slices near-empty): forget_rouge ∈ 0.87±0.05 with retain unchanged ±0.02.
    REFUTE-of-lean-verdict: forget_rouge ≤ 0.63 (G ≥ 0.5 vs oracle) → slices DO carry
    deletable content and the lean failure was pure serving artifact → escalate extended+MIA.
  - **H11 (slice content):** probe Δ_own = mean over 200 authors of
    (gen_own − gen_only) answer-prob on own train rows. CONFIRM (near-empty): Δ_own < 0.10
    (prediction from dropall rouge 0.8726). REFUTE: Δ_own > 0.25.
  - **H10 (interference ladder):** per-author answer-prob under gen+own+k foreign slices,
    k ∈ {0,10,50,100,199}, 20 seeded authors (≥5 forget). CONFIRM: monotone-nonincreasing
    (tol 0.01) for ≥80% of authors AND total k=0→199 drop ≥ 0.10. REFUTE: flat (<0.05).
  - **H14 (strict-isolation arm):** frozen-lora_A + disjoint_dead slices on the scaffolded
    base — does ~40 lora_B rows/author/layer suffice for author recall, with provably
    data-provenance-exact deletion? CONFIRM: strict_routed_full mu ≥ 0.55 AND probe gen_own
    own-author answer-prob ≥ 0.80 AND strict_routed_unlearn forget_rouge ≤ 0.45 (≈ the
    never-trained state) with retain ±0.02. REFUTE: mu < 0.50 or own-prob < 0.6 → a
    quantified per-author capacity floor (vs SEA's rank-8-suffices).
  - **strict_all_on (collapse demo):** all 8000 slices served at once = the merging-collapse
    regime inside one adapter. EXPECT mu ≤ 0.50 ≪ strict_routed_full (ties to merge_mechanism).
  - **D→E3 gate:** E3 runs after D regardless of H9/H11 verdicts (different claim), EXCEPT
    the wrapper-sanity floor above.
- **Setup:** New/changed code (sha256 prefixes): `memsinks_routed_model.py` b01cde0be18e
  (MemSinksRoutedModel — SiftMasksModel-contract wrapper: oracle q2author route → per-author
  serve vector via MaskState.set_fixed; deleted/OOD → gen-only), `probe_slices.py`
  61125e35dd8a (gen_only/gen_own/all_on per author + k-foreign ladder; slice_increment =
  gen_own − gen_only), `masks.py` e038cd3facdd (NEW disjoint_dead scheme — **p_gen=0 trap:**
  `disjoint_partition(8192,0,200)` folds the 192-neuron remainder into ALWAYS-ON general rows,
  silently breaking strict exactness; disjoint_dead gives 40/author/layer + 192 DEAD rows
  owned by nobody), `train_memsinks.py` f4069ec9e7fe (freeze_lora_a via
  train_lora_shard.apply_irp_projections; configurable max_grad_norm/weight_decay;
  AuthorBlockSampler — one author per optimizer step at bs4×ga5), `eval_tofu.py` 547a7cd45905
  (+`--memsinks_config`/`--memsinks_unlearn_tag` arm in build_served_model; same flags added
  to attack_mia.py for the deferred H6; CLAUDE.md row updated), strict config
  `configs/memsinks_tofu_1b_strict.json` 2b4805cc89f4 (scaffolded base
  `_scaffolded_alpaca2k`, gate/up-only r32, dropout 0, id_scheme disjoint_dead, p_gen 0,
  freeze_lora_a seed 42, clip 0, wd 0, author-block bs4×ga5), driver 5f66660554c2
  (+d_routed/d_probe/e3 stages). **CPU gates: 20/20 green** (`test_memsinks.py` 263c5575f32c),
  incl. the E3 headline gate: frozen lora_A never gradded; lora_B grads exactly 0 outside the
  batch author's slice incl. dead rows; **data-provenance test** — rerunning the training
  sequence varying ONLY another author's batch leaves the author's rows bit-identical.
  disjoint_dead golden sha fc016e5178a9. Jobs: recorded at submission (chained, ≤4 GPUs).
  Phase-D artifacts reuse the EXISTING M1 adapter (Round-1 jobs 443146-49).
- **Results:** none yet (pre-registration).
- **What worked / hypothesis verdict:** all OPEN. One refinement forced by the CPU gate,
  logged now so the claim is honest BEFORE results: **Adam momentum tails** — a trained
  author's rows keep moving during OTHER authors' optimizer steps (m,v decay), so "rows
  bit-frozen outside own steps" is FALSE. The correct (and now gate-tested) claim: the
  movement is a function of the author's OWN gradient history + shared schedule scalars only;
  **no other author's DATA ever influences a row** (varying another author's batch leaves the
  rows bit-identical). Deletion (zeroing the rows) remains bitwise-exact. Claim tier
  unchanged: row-provenance exactness + bitwise deletion op, NOT bitwise-≡-retrain.
- **Observations:** (design-stage) The routed arm rides eval_tofu's build_served_model so
  attack_mia attacks the identical served artifact later (H6 parity flags landed, unused).
  strict_all_on needs no bake (trained/ served via --preloaded_adapter IS all-on).
- **New questions / new hypotheses:** if H14 confirms, the storage story is ~80 KB/author
  (40 rows × r32 × 2 modules × 16 layers, bf16) vs SEA's 32 MB/author — is recall stable
  down to smaller s (capacity dial)? If H9 confirms + H9-del expected-confirms, is there ANY
  training-time-allocation regime (without gradient isolation) that stores deletable content
  — or is E3's isolation the minimum (the §5-frame "you must pay selection somewhere")?
- **Next Steps:** STUB preview → submit d_routed (%2) + d_probe (1) → e3 chain
  (micro-smoke → train → 3 evals %3 + probe) → harvest → results entries + REPORT.md →
  review with user.
