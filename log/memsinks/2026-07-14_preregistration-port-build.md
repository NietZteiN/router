### Target Date: 2026-07-14 (Pre-registration + port build — MemSinks/SeqTD → TOFU)
- **Hypotheses / what we're testing:** Pre-registered BEFORE any SLURM job (lean feasibility
  phase: M1 disjoint arm + CTRL-L control only, seed 42, smoke tier). Claims: **C1**
  (paper-faithful) drop-all-sinks removes forget-author knowledge more than retain knowledge;
  **C2** (novel extension) per-author sink slices are selectively deletable (disjoint regime
  only — hashed p_mem=0.3 forget10 union covers 97.4% of the sink pool, so selective ≡ total
  there). Reference numbers: base 1B mu ≈0.42 (smoke), k=1 LoRA ≈0.74, exact methods
  routing+scaffold 0.7509 / SIFT 0.737 / SEA 0.711.
  - **H1 (trainability + memorization guard):** CONFIRM if mu(memsinks_full) ≥
    mu(ctrl_lora_full) − 0.05 AND forget answer-prob(memsinks_full) ≥ 0.8 × ctrl's.
    REFUTE if mu < ctrl − 0.10 or the guard fails (then diagnose capacity/undertraining
    before interpreting anything downstream).
  - **H2 (localization):** gap closure G = (P_full − P_del10)/(P_full − P_oracle) on forget
    answer-prob and forget ROUGE (P_oracle = retain90 oracle's forget-set value). CONFIRM
    G ≥ 0.5 (paper-consistent), STRONG G ≥ 0.9, REFUTE G < 0.25 (memorization escaped into
    always-on delta capacity: lora_A, attention LoRA, gen neurons).
  - **H3 (selectivity, disjoint):** del_forget10 leaves retain intact — mu drop ≤ 0.02 vs
    memsinks_full AND randdel placebo shows the mirror pattern (forget metrics unchanged,
    only its 20 authors damaged). REFUTE if retain-side drop > 0.10.
  - **H4 (mechanism binding — gates all interpretation):** per-epoch memorization-gap probe
    (answer-prob under own-mask vs own-sinks-deleted, 7 probe authors × 2 rows) opens by
    epoch 5. If it never opens: mechanism did not bind at 5 repetitions (paper used 128) —
    STOP and review; fq numbers uninterpretable.
  - **H5 (forget quality):** post-del10 smoke fq (KS vs existing retain90 reference) ≥ 0.1
    CONFIRMS, ≤ 0.01 REFUTES. Caveats pre-registered: smoke-tier forget10 only; extended fq
    is style-confounded across serving styles (SIFT 0.0045 lesson); verbatim vs fact-level
    may dissociate — a verbatim-only closure is a finding, not a failure.
  - Deferred (pre-registered now, run only after joint review): H6 MIA spectrum position
    (bands ≤0.45 near-exact / 0.45–0.60 co-adaptation tax / ≥0.60 approx-like; two-sided vs
    oracle floor 0.379), H7 competitiveness (SIFT/SEA band), H8 capacity dial (p_gen 0.5,
    e25), hashed-arm overlap-collateral ladder; Round-2 controls = shuffled-ID null +
    untied-dropout (user-selected 2026-07-14).
- **Setup:** New project `~/memsinks_tofu/` (checkpoints → /storage2/jack/checkpoints/memsinks_tofu).
  Substrate = **masked LoRA delta**: forward hooks on `mlp.{gate_proj,up_proj}.lora_B.default`
  (peft 0.14.0 — hook sees the pre-scaling delta; scaling commutes) gate each MLP intermediate
  neuron's delta per author; base path untouched. Disjoint scheme p_gen=0.7 on Llama-3.2-1B
  (I=8192, 16 layers): 2400 sinks = **12 neurons/author/layer**, remainder 58 → general.
  Author IDs 1–200 in the hash (ID 0 = degenerate all-ones mask, verified). Deletion = CPU
  bake zeroing forget authors' sink rows of lora_B → bone-stock PEFT dirs served via
  `eval_tofu.py --preloaded_adapter` (bake ≡ hook bit-identity unit-tested). Serving modes
  pre-registered: full = all deltas at 1.0; deletion = slices at 0, no rescale (paper's
  p_mem-scaled "all" is wrong for a fine-tuning port). Recipe = frozen SISA
  (r32/α64/rslora/5ep/lr1e-4/bf16/seed42/max_len256/b4×ga4/paged_adamw_32bit/cosine) + ONE
  deviation: `gate_proj` added to target_modules (paper gates the whole SwiGLU neuron);
  CTRL-L module-matched. Plain transformers.Trainer + QACollator (trl SFTTrainer silently
  drops the author_id column — parity + guards unit-tested). Configs:
  `memsinks_tofu/configs/memsinks_tofu_1b_disjoint.json` (sha256 a2029129…) /
  `memsinks_tofu_1b_ctrl_lora.json` (bd192653…). Scripts: train_memsinks.py 36dc9537…,
  masks.py 97686f9d…, memsinks_model.py 8083f9a6…, bake_deletion.py 6fb670eb…,
  submit_memsinks.sh 707ef7f7…. Mask-table goldens (sha256): hash-p0.3 dda2a736…,
  disjoint-p_gen0.7 60891098… (`memsinks_tofu/golden_mask_sha256.json`). KS reference reused
  from `tofu_sisa_lora/checkpoints/Llama-3.2-1B-Instruct/results/smoke/retain_tr_scores.npy`
  (retain90 oracle, method-independent, peft-bakeoff pattern). Eval: smoke tier,
  `--k 10 --forget_shard_id 9`. SLURM: train array 0-1%2 → afterok bake (CPU) → afterok eval
  %4 (7 tasks); sprint1-3, ≤4 GPUs global. Job IDs recorded in the next entry at submission.
- **Results:** Pre-run only today. `test_memsinks.py`: **14/14 CPU gates green** — hash port ≡
  reference source over a shape×dim×p grid (int64-overflow quirk reproduced); ID-0 degeneracy
  present in reference and avoided; hash density 0.277–0.323 (p=0.3); disjoint invariants
  (12/author/layer, zero overlap, forget10 union exactly 10%); hashed forget10 union 97.4%;
  hooks+all-ones ≡ no-hook (fp32 & bf16, bit-identical); gradient isolation (masked lora_B
  rows exactly 0); bake ≡ hook bit-identical (delete + dropall); deletion isolation (retained
  author's masked forward bit-identical pre/post another author's deletion); KV-cache ≡
  no-cache generation under mask + mask live at inference; collator parity with the SISA
  path; maskless-training-forward raises; 2-step micro-run loss 11.507 finite, mask cleared,
  authors tracked.
- **What worked / hypothesis verdict:** All H1–H5 OPEN (no GPU runs yet). Port correctness
  claims verified at CPU-gate level (the numbers above). The C2-vacuity fact (hashed masks
  make selective deletion ≡ total at paper hyperparameters) is now a MEASURED constant of the
  method: union(20 authors, p=0.3) = 97.4% of the sink pool — this is why the primary arm is
  disjoint slices.
- **Observations:** (1) The reference hash is not modular exponentiation — `torch.pow`
  overflows int64 from exponent 3; masks are deterministic overflow artifacts, ported
  verbatim (a "clean" reimplementation would silently produce different masks). (2) seq_id 0
  hashes to an all-ones sink mask — a footgun for anyone using 0-indexed IDs with this code.
  (3) The reference `LLaMAMLPSeqTD.forward` calls `torch.cuda.empty_cache()` every forward
  and draws `torch.rand(1)` per training step — neither replicated (throughput + RNG-stream
  hygiene). (4) fp32-vs-float64 rounding tripped the first collateral-stats assert —
  union_fraction now computed as exact integer ratio.
- **New questions / new hypotheses:** Does 12 neurons/author/layer of DELTA capacity suffice
  to absorb 20 QA pairs of memorization (feeds H8)? Does author-level tying (20 texts share a
  mask) preserve or break the paper's per-sequence shielding argument? Where does the
  co-adaptation tax land on the MIA spectrum (H6, deferred)?
- **Next Steps:** STUB=1 preview → GPU smoke (`submit_memsinks.sh smoke`) → `all` chain
  (M1+CTRL-L trains %2 → bake → 7 smoke evals %4) → harvest → results entry → REVIEW WITH
  USER before Round 2 (hashed arm, shuffled-ID null, untied-dropout) / Round 3 (seeds, MIA,
  extended, method-matched oracle, e25).
