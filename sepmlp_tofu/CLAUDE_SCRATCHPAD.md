# CLAUDE_SCRATCHPAD — sepmlp_tofu

## State (2026-07-20) — P0 build complete, CPU gates NOT yet run
Plan: `~/.claude/plans/vincent-hanke-3-45-pm-eager-coral.md`. Narrative:
`log/sepmlp/` (pre-registration entry `2026-07-20_preregistration-build.md`
written BEFORE any job — keep it append-only).

- Core modules written: `sepmlp_common.py`, `bank_layer.py`, `sepmlp_model.py`,
  `train_sepmlp.py`, `build_droplist.py` (+ configs/tests/driver as they land).
- **RELEARN component VERIFIED GREEN (2026-07-20, follow-up session):**
  `relearn.py` / `relearn_score.py` / `collect_relearn.py` /
  `configs/relearn_1b.json` audited point-by-point against the plan spec —
  no fixes needed (OU evaluate_probability port matches utils.py:82-99
  verbatim incl. the UNSHIFTED-labels normalizer; rougeL recall matches OU
  eval_text_similarity conventions; forget10 = authors 180-199 via the
  memadapt blocklist artifact, so probe authors 0,5,…,45 are all retain;
  holdout10 sha pin recomputes true from the cache). Added one regression
  test: `test_end_to_end_relearn_with_banks_installed` (serve=sepmlp path —
  banked forward + LoRA loop; bank tensors bitwise unchanged after relearn).
  Gates 14+16: `pytest tests/test_relearn.py tests/test_holdout.py -q` →
  **11 passed, 1 skipped** (the skip = the bf16 GPU smoke, now gated behind
  `SLURM_JOB_ID`/`SEPMLP_GPU_TESTS=1` — login nodes HAVE visible GPUs, the
  old bare `skipif(not cuda)` was silently running it on a login GPU);
  all three scripts py_compile + `--help` clean.
- **Pending before ANY submission:** `pytest tests/ -q` in test-env — all 17
  CPU gates green — plus `STUB=1` previews of every driver verb.
  (2026-07-20 21:5x snapshot: 48 passed, 1 failed —
  `test_data_pipeline.py::test_collator_source_ids_and_ne_pad_attention_quirk`,
  in the parallel core-gates agent's in-flight file; not relearn's.)
- No checkpoints, no evals, no SLURM job ids yet. Zero GPU spend so far.

## State update (2026-07-20 ~22:40, orchestrating session)
- **Spec v2 landed and reconciled.** The user-supplied authoritative recipe
  (ReLU+b_gate gate, loss L1+10·L2+50·L3+1·L4, alternating negative batches
  incl. real_authors, per-author clip, cosine LR, detector init; plan
  `~/.claude/plans/include-these-details-as-mellow-barto.md` Part 2) is
  implemented; lean pilot = `configs/pilot_relu_lr{3e-4,1e-3,3e-3}.json`
  (array 0-2%1); the 9-arm SwiGLU λ grid (pilot_0–8.json) is RETIRED UNRUN
  (files left on disk as record).
- **G0 GREEN: 69 passed, 1 skipped** (gated GPU smoke test) — full suite.
- Protocol reconciled: superseding pre-registration
  `log/sepmlp/2026-07-20_specv2-preregistration.md` (revised G2: lr-only
  pilot, no λ=0 clause; external priors P1–P4 recorded); thread README +
  master log/README.md By-experiment row + 2 Timeline rows; project CLAUDE.md
  method paragraph updated to spec v2.
- **P1 SMOKE SUBMITTED: job 446535**, `DEP=afterany:446366:446368:446369`
  (the ctv tails — adds 0 concurrency until they drain; ~31 ctv tasks were
  pending at submit). Config sha 7f94b32e5deececc. Watcher armed (bg loop →
  reports sacct state + log markers on completion).

## State update (2026-07-21, reproduce-and-verify session)
- **P1 SMOKE PASSED** (job 446535, ran 05:07–05:09): grad-structure OK across
  16 layers (l1/l2l3/l4 isolation), loss components live (hinge 6.35→4.98
  over 5 calls, gram nonzero, lm sane), telemetry sane (own_norm 0.059 ≫
  off_norm 0.004, ood_over_own 0.077), peak mem **14.04 GiB** ⇒ bs32 GO,
  reload+forward parity PASS. G1 GREEN.
- **Sha discrepancy found & resolved:** `bank_layer.py` / `train_sepmlp.py` /
  `measure_selectivity.py` were edited 22:50–22:52 on 07-20 (parallel build
  session tail), AFTER the spec-v2 pre-reg entry pinned their shas but BEFORE
  the smoke executed — so the smoke validated the CURRENT code
  (meta.json script_sha256 9b4d32397e219bf4 == current train_sepmlp.py).
  Current pins: bank_layer 421a05f5ba1cb5fe · train_sepmlp 9b4d32397e219bf4 ·
  measure_selectivity 113010ee307dddf4 (all others match the pre-reg
  inventory). **G0 re-run on current code 2026-07-21: 70 passed, 1 skipped.**
  To be recorded in today's dated log entry (append-only correction).
- **P2 attempt 1: job 446705 — all 3 arms OOM** (bs32, step 9/390: 38.67 GiB
  allocated + 5.78 GiB request > 44.46 GiB card). The smoke's 14.04 GiB was at
  bs2 — it could not clear bs32; G1's "bs32 GO" read was an over-extrapolation.
  **Declared fallback bs16×ga2 applied** (effective batch unchanged; penalty
  ga-invariance handled in compute_loss — trap 2). New pilot config shas:
  lr3e-4 3105d9fc8cfd13ed · lr1e-3 95ff134ead7316ff · lr3e-3 87f3f473f598c8c2.
  CPU gates re-run after edit: 70 passed, 1 skipped. Stale run dirs from
  446705 hold only detector_init.npz + hf_trainer/ (no sepmlp.pt — the skip
  guard will retrain; detector cache is deterministic and reusable).
- **NOTE: `sacct` returns empty on this cluster** — job-state monitoring must
  use squeue presence + output-file existence, not sacct.
- **P2 attempt 2: job 446714 — ALL 3 ARMS COMPLETE, G2 read done.**
  Medians (selectivity / own-prob all-active): lr3e-4 **4.38 / 0.981** ·
  lr1e-3 **38.61 / 0.778** · lr3e-3 **1909.7 / 0.695**. No arm passes the
  joint bar (sel ≥5 AND prob ≥0.80); NO-GO clause (<2 everywhere) decisively
  off the table — H1's LoRA-anchor refutation overturned (1.11 → 1909).
  Monotone lr tradeoff: selectivity ↑, recall ↓. All-active vs own-only gap
  +0.824 / +0.179 / +0.0005 — branches fully self-contained at 3e-3; NO
  memsinks-style interference in any arm (all-active never worse).
  **G2 = ADJUDICATE** (best-in-[2,5) arm lr3e-4 has own-prob 0.981 ≥0.80) →
  ONE bridging config licensed: **lr 5e-4** (geometric midpoint; log-interp
  projects sel ≈11, prob ≈0.90). Config `pilot_relu_lr5e-4.json`
  (sha f4d6b9d59d62af6d); new driver verb `pilot1` (single named arm, same
  body as a pilot array task). **Bridging arm SUBMITTED: job 446732**
  (queue empty at submit ⇒ 1 ≤ 4 ✓). G2 final verdict after it lands —
  no K=200 spend before that.
- **BRIDGE RESULT → G2 = GO (lr 5e-4 winner).** Job 446732: median sel
  **7.171** (SELECTIVE; frac≥5 0.85, frac<2 0.00; min/med/max 2.32/7.44/10.36),
  median own-prob (all-active) **0.9765**, min 0.9363, 0 authors <0.8. Both
  bars passed → **H1 CONFIRMED** (unique arm satisfying both constraints).
  Caveat carried to P4: all-active−own-only gap still +0.7304 (own-only median
  0.213) — recall collectively carried at this lr; deletion collateral is the
  open risk, measured directly by `sepmlp_unlearned` retain metrics.
- **P3 prep:** `sepmlp_1b_k200.json` → lr 5e-4 + bs16×ga2
  (sha 62fd86576fb344ec). Memory projection borderline (pilot bs16 K=20
  peaked 27.59 GiB; +~10 GiB K=200 params/optimizer + bank-act ×10 scaling)
  → **bs16 K=200 memory smoke first** (`smoke_k200_bs16.json`,
  sha 04d2cf834f4777a9): **job 446903**. GO bar: peak ≲38 GiB → submit train
  + probe200 (DEP-chained, same phase). Fallback if OOM/tight: bs8×ga4
  (documented deviation).
- **Memory smoke 446903: PASS 15.28 GiB** → P3 v1 submitted: train 446910 +
  probe 446911 (afterany). **train 446910 OOM'd at step 6/3750** (demand
  ~46.8 GiB > 44.46; the 5-step smoke stopped ONE step short of the killing
  batch — smoke-cap lesson: step-capped memory smokes are unreliable for
  worst-batch peaks). Probe 446911 failed downstream on missing ckpt (inert).
- **P3 v2: bs8×ga4** (`sepmlp_1b_k200.json` sha be58163496487711; effective
  batch 32 unchanged, penalty ga-invariant): **train 446949 + probe 446950**
  (afterany chain; queue empty at submit ⇒ max 1 concurrent ≤ 4 ✓).
  Projected peak ~30–32 GiB. ETA ~2h train. G3 read from
  `sepmlp_1b_k200_s42/selectivity_k200.json` after probe lands. NOTE for G3:
  the gap≤0.05 clause is at risk (pilot winner gap +0.73) — if G3 fails on
  the gap clause alone with sel+recall healthy, bring the tension to the
  human before P4 spend (recorded in the bridge-go log entry).
- **P3 v2 DONE (446949 train / 446950 probe) → G3 READ: FAIL. THREAD STOPPED
  PRE-P4, awaiting human decision.** Numbers: median sel 507.5 (clause A ✓,
  H-scale ✓ — ~70× amplification), median recall 0.6367 (p10–p90 0.453–0.805,
  0 authors ≥0.9, 20/200 ≥0.8), gap mean +0.1304 (clause B ✗). Train healthy:
  2h00m, 33.04 GiB peak, no NaN. Finding: suppression pressure ∝ K at fixed
  lr; K=200 point sits on the K=20 recall-vs-sel tradeoff curve. Gap≤0.05 and
  recall≥0.8 look jointly unsatisfiable for the pinned recipe (gap clause =
  memsinks tripwire; observed sign is benign/collective-recall). Gates are
  not renegotiated after results ⇒ escalated. Decision package (G3-fail log
  entry): (A-rec) one K=200 arm lr 2e-4 (H-k200-lr pre-registered: predict
  sel≈30, recall≥0.80; +2.2 GPU-h, +2.5 GB) + human ruling on the gap
  clause; (B) w2/w3 → 1/5 rescale (recipe change, new pre-reg); (C) P4 on
  the 0.637 ckpt (deletion mechanics only; utility bars will miss); (D) ask
  Vincent his K. Storage 8.1 GB vs ≤6 budget: request approval to delete
  `sepmlp_1b_k200_s42_smoke/` + `sepmlp_1b_k200_s42_smokebs16/` (~5 GB,
  both reproducible from configs).
- **HUMAN DECISION (2026-07-21, interactive):** Path **A** chosen — lr 2e-4
  K=200 arm under H-k200-lr; **G3 gap clause ruled a memsinks tripwire**
  (benign-sign gap does not block; deletion collateral judged at P4 by
  |ΔUtil.R| ≤ 0.03). Smoke-ckpt deletion APPROVED → dry-run listed both dirs
  → `rm -r` of `sepmlp_1b_k200_s42_smoke/` +
  `sepmlp_1b_k200_s42_smokebs16_smoke/` (actual name — --smoke appends
  `_smoke` to output_dir) → project at 3.4 GB (under budget incl. the
  coming +2.5 GB ckpt).
- **H-k200-lr arm SUBMITTED:** config `sepmlp_1b_k200_lr2e-4.json`
  (sha a493cf09c30224a0; fresh output_dir `sepmlp_1b_k200_lr2e-4_s42` — the
  lr5e-4 ckpt/probe stay untouched for provenance): **train 447162 + probe
  447163** (afterany; queue empty at submit ⇒ max 1 ≤ 4 ✓). Bars: sel ≥5
  AND median recall ≥0.80 → G3 pass (gap recorded, not blocking) → P4.
  REFUTE: recall <0.75 or sel <5.

## State update (2026-07-22) — H-k200-lr REFUTED, stopped again
- **Arm result (447162/447163, healthy 1h54m, 33.04 GiB):** sel 36.0
  (prediction ≈30 — exact), median recall **0.7468 < 0.75 refute line**
  (missed by 0.003; 60/200 ≥0.8, p10/p90 0.608/0.861); gap +0.450
  (informational per ruling); own-only 0.286. **Verdict: REFUTED as
  registered** — logged in `log/sepmlp/2026-07-22_hk200lr-refuted.md`;
  thread README + master README updated.
- **Curve now mapped (recall vs ln sel):** K=20 (1.48,0.981) (1.97,0.977)
  (3.65,0.778) (7.55,0.695); K=200 (3.58,0.747) (6.23,0.637) — K=200 ≈ K=20
  − 0.03 at matched sel. In-recipe lr dial tops out ~0.75–0.87 in the
  healthy-sel range. **H-wscale (w2/w3 10/50 → 1/5 at lr 5e-4) is now the
  better-motivated fix** (equalize per-author pressure; targets the pilot
  point directly) but is a recipe change → needs user sign-off + its own
  pre-registration. Alternative in-recipe: H-k200-lr2 (lr 1.5e-4 → predict
  sel 12–25, recall 0.80–0.87).
- **Storage:** project 5.7/6 GB; another ckpt (+2.5 GB) breaches → pair any
  new arm with an approved deletion (candidate: lr5e-4 ckpt
  `sepmlp_1b_k200_s42/sepmlp.pt`, keeping probe JSON + meta). **/storage2
  itself at 97% (258 GB free) — flagged.**
- **AWAITING USER:** A2 (lr 1.5e-4) vs B (w-rescale, recommended) vs C (P4
  mechanics on 0.747 ckpt) vs D (ask Vincent K/weights) + storage pairing.

## Next (in order)
0. ~~Smoke → G1~~ DONE (PASS, see above). P2 pilot submitted → on completion
   read `pilot/pilot_relu_lr*_s42/selectivity_pilot.json` against G2. Then
   G2 verdict → K=200 (P3) or refutation entry.
1. ~~**G0**: run the CPU gate suite~~ DONE (69 passed, 1 skipped).
2. **P1 smoke** (~0.2 GPU-h): full-size K=200 bank, 2 authors' data, ~5 steps,
   batch 2 → save → reload → bitwise parity; prints max_memory_allocated
   (go/no-go for bs32, declared fallback bs16×ga2). Submit
   `DEP=afterany:<current queue tails>` — the 4-GPU cap is currently held by
   the ctv/scaffold chains (ids 446357–446375 at the 2026-07-20 poll;
   **re-check `squeue` at submit, never trust this snapshot**). afterany,
   never afterok.
3. **P2 K=20 pilot** (~3 GPU-h): 9-task array `0-8%2`; per task train 15ep →
   selectivity probe → recall probe (all-active AND own-only serving). Then
   read gate G2 manually — never pre-chain across a gate.
4. **P3 K=200** train + probe200 re-gate (G3) → **P4 OU evals** → **P5 relearn
   battery**. Wave-2 ablations deferred behind their own pre-registration.

## Gate rules (pre-registered — do not renegotiate after seeing results)
- **G2 (pilot):** GO = pick (λ,lr) maximizing median on/off selectivity s.t.
  selectivity ≥5 AND own-author answer-prob ≥0.80 and ≥0.90× the λ=0 control;
  ADJUDICATE [2,5) with one bridging config; NO-GO (<2 without >20% recall
  loss) ⇒ H1 refuted → write refutation entry, stop before K=200 spend.
- **G3 (K=200):** selectivity ≥5 and ≥0.7× pilot; all-active vs own-only
  own-prob gap ≤0.05 (memsinks-interference tripwire, placed before eval
  spend).

## Constraint checklist (root CLAUDE.md — verify before every action)
- [ ] **Hardware:** SLURM only (sprint1–3, `--exclude=sprint4`); zero
      training/eval/heavy compute on login nodes (CPU pytest + STUB previews
      are the only login-node work).
- [ ] **Global 4-GPU cap:** `squeue -u jack -o "%.10i %.20j %.10T %.10b %F"`
      before EVERY submit; (max GPUs queued jobs could run concurrently) +
      (this submission) ≤ 4. Arrays always `%2`; one sepmlp array queued at a
      time; SEPMLP_CAP=4 never raised; chain with `--dependency` instead of
      waiting rather than over-submitting.
- [ ] **Storage:** all artifacts under `/storage2` (`checkpoints/` symlink →
      `/storage2/jack/checkpoints/sepmlp_tofu`); `/home` code-only;
      `HF_HOME=/storage2/jack/data/huggingface`. Budget ≤6 GB (K=200 ckpt
      2.52 GB fp32); pilot ckpts deleted only after eval JSONs land AND with
      human approval (deletion protocol), logged.
- [ ] **Correctness:** seed 42 recorded everywhere (43/44 reseed of the
      headline before external claims); smoke before full jobs; watch for
      NaNs / frozen loss / empty generations / out-of-bounds metrics;
      meta.json carries command, config sha, script_sha256, seed, SLURM job
      id for every kept result.
- [ ] **Destructive ops:** no `rm -rf` / recursive / forced deletion without
      human-in-the-loop (blast radius + dry-run + approval).
- [ ] **Data discipline:** NEVER train on holdout10 (relearn control + MIA
      nonmembers — pinned by a test); OU chat-template schema only (imported
      memadapt `data_tofu`), never the plain Question:/Answer: track; OU-track
      and plain-track numbers never share a table.
- [ ] **Provenance:** no git commits unless asked (memory rule) —
      provenance via meta.json sha256s.
- [ ] **Log protocol:** every working day advanced → dated append-only entry
      in `log/sepmlp/` + thread README refresh + master `log/README.md`
      Timeline/By-experiment rows.

## Standing notes
- OU working tree (`~/open-unlearning`, branch `memadapt-eval`) carries a
  deliberate uncommitted fp32-logits fix in `src/model/__init__.py` — it trips
  the clean-tree guard; ask the user to approve committing it to
  `memadapt-eval` before `install_branch.sh` runs. Do not commit unasked.
- Claims discipline: deletion is exactly "the author's parameters are
  removed" — we do NOT claim exact unlearning (negatives leak; Vincent's
  stated caveat).
- Frozen in-flight files elsewhere (merge_subset.py, sparsify_pool.py,
  tofu_sisa_lora harness) — read-only, never touch.

## State update (2026-07-22b) — triple launched (user-approved B+A2+C)
- Human picked ALL of B, A2, C; lr5e-4 weights deletion approved & executed
  (`sepmlp_1b_k200_s42/sepmlp.pt` removed, measurement artifacts kept; 3.4 GB).
- Pre-reg entry `2026-07-22_wscale-lr2-c-preregistration.md` frozen BEFORE
  submission. Configs: w15 9071404c26d439f7 · lr1p5e-4 c7ca758b9994bf9f;
  driver post-edit sha d095934be36a6711 (eval verb now takes
  [config] [run_dir] [prefix]); CPU gates 70/1 after edits. OU sepmlp
  integration installed via ALLOW_DIRTY=1 (additive, NOTHING committed).
- **Jobs:** B train 447175 → probe 447176 (afterany) · A2 train 447177 →
  probe 447178 (afterany) · C eval array 447179 0-2%2 (prefix sepmlp_lr2e4,
  on the lr2e-4 ckpt). Worst-case concurrency 4 = global cap; queue had
  nothing else at submit.
- Storage: peak ~8.4 GB while both new ckpts coexist (disclosed in pre-reg);
  loser-weights cleanup goes to the human after verdicts.
- On completion: read H-wscale + H-k200-lr2 bars (PASS sel≥5 ∧ recall≥0.80;
  REFUTE recall<0.75 ∨ sel<5; winner = higher recall) → winner gets the full
  P4 replication row (fresh prefix); read C mechanics (deletion wall-time,
  dropall≡calib_base, forget10 Δ, ΔUtil.R collateral = H-gap answer, MIA).

## OVERNIGHT PROTOCOL (2026-07-22, user: "have it run overnight")
Standing authorization: continue the PRE-REGISTERED ladder autonomously
overnight. Rules of engagement (no renegotiation; anything off-script waits
for morning):
1. **B/A2 probes land** → read `selectivity_k200.json` against the frozen
   bars (PASS sel≥5 ∧ recall≥0.80; REFUTE recall<0.75 ∨ sel<5; 0.75–0.80
   gray = report, no P4). Winner = higher median recall s.t. sel≥5.
2. **Winner exists** → submit its FULL P4 replication row:
   `bash submit_sepmlp.sh eval <winner_config> <winner_run_dir> <prefix>`
   with prefix `sepmlp_w15` or `sepmlp_lr1p5e4`. Check squeue cap ≤4 first
   (C array may still be running). Fresh labels/dirs (trap 9).
3. **Winner P4 lands clean** (3 TOFU_EVAL.json present, no fail markers) →
   submit P5 relearn battery on the winner:
   `bash submit_sepmlp.sh relearn <winner_config> <winner_run_dir>`
   (verb parameterized, driver sha 3a693f3838eb507a, STUB-verified; 24 tasks
   %2). Cap check first.
4. **C evals land** → read mechanics only (deletion wall-time from logs,
   dropall≡calib_base, forget10 forget-metric movement, ΔUtil.R collateral
   = H-gap answer, MIA AUCs). NO replication claims from C.
5. **Both arms refuted** → NO further GPU spend; write refutation entry +
   morning decision package.
6. Every read/launch gets its dated log entry (append-only) + README refresh
   in stride. Storage: no deletions overnight (loser cleanup = morning
   approval). All submissions: queue_check, ≤4 concurrent, afterany chains
   only within a phase, never across a gate.
Winner-arm identities: B = configs/sepmlp_1b_k200_w15.json →
sepmlp_1b_k200_w15_s42 · A2 = configs/sepmlp_1b_k200_lr1p5e-4.json →
sepmlp_1b_k200_lr1p5e-4_s42. C ckpt = sepmlp_1b_k200_lr2e-4_s42 (prefix
sepmlp_lr2e4). Vincent's priors for the replication read: deleted 0.97→0.32,
others ≤0.002, utility Δ≤0.001, no relearn residue; MemAdapt anchors: Agg
0.869, Priv 0.917, deletion 0.027 s.

## State update (2026-07-22c) — TRIPLE LANDED, LADDER HALTED (no passing arm)
Results entry: log/sepmlp/2026-07-22_wscale-refuted-lr2-gray-mechanics.md.
- **B (H-wscale, w2/w3=1/5 @lr5e-4):** sel 24.59, recall 0.6956 -> REFUTED
  (predicted sel 5-15/recall>=0.90; missed both). Weight rescale == moving
  ALONG the same K=200 curve, not onto a better one.
- **A2 (H-k200-lr2, lr1.5e-4):** sel 16.33, recall 0.7947 -> GRAY [0.75,0.80).
  Best K=200 point. Per pre-reg: report, NO P4 (no autonomous escalation).
- **4-point K=200 curve (sel->recall):** 507.5->0.637 (lr5e4 w10/50),
  36.0->0.747 (lr2e4), 16.33->0.795 (A2), 24.59->0.696 (B). One curve,
  tops ~0.80 healthy-sel band vs pilot 0.977. **Recall ceiling STRUCTURAL.**
- **C mechanics (lr2e4 ckpt):** deletion clean (forget 0.767->0.054,
  extraction 0.469->0.047, exact_mem 0.935->0.490, MIA 0.997->0.362 [floor],
  privleak -99.6->+3.98) + cheap (slice removal 1.07s; droplist build 7.2s) +
  RETAIN COLLATERAL retain_Q_A_Prob 0.676->0.563 (-0.113), retain_ROUGE -0.122
  BUT aggregate model_utility +0.003 (H-gap: real, aggregate-masked). dropall
  symmetric floor (forget 0.092 ~= retain 0.087) ~= base [weight identity
  gate-proven; no numeric calib_base TOFU_EVAL on disk for exact side-by-side].
- **HALT per overnight protocol #1/#5:** no PASSING arm -> no winner -> NO P4
  replication row, NO P5, NO further GPU spend overnight. NO deletions
  (loser-weight cleanup = morning approval). Storage ~8.4 GB (2 arm ckpts +
  lr2e4). Master + thread READMEs refreshed; morning decision package in the
  thread README "Open ideas / next steps".
- **Driver edits this session (STUB+gate verified, no regressions):**
  eval verb -> [config][run_dir][prefix]; relearn verb -> [config][run_dir];
  submit_sepmlp.sh sha 3a693f3838eb507a. Configs added: w15
  9071404c26d439f7, lr1p5e-4 c7ca758b9994bf9f. Both trains healthy (no NaN).
- **Morning options (thread README):** (1) K-sweep {20,50,100,200} diagnose
  ceiling [recommended] (2) ask Vincent K/width/routing (3) accept A2 0.795 ->
  full P4+P5 (4) capacity ablation (width/layers). No default chosen; awaits
  Jack.
