# CLAUDE_SCRATCHPAD — blocktc_tofu

## State (2026-07-21) — P0 build IN PROGRESS (parallel sessions), zero GPU spend
Binding contract: `DESIGN.md` (v1, 2026-07-21). Narrative thread:
`log/blocktc/` — NOT yet seeded; create it (from `log/TEMPLATE.md`) with the
pre-registration entry BEFORE the first job, append-only after that.

- **Landed this session (infra agent):**
  - `slurm_nodes.sh` — BLOCKTC_* env (CAP=4 never raise, THROTTLE=2,
    exclude sprint4, mem 48G / cpus 8, times SMOKE 00:40 / PHASE0 01:30 /
    PILOT 01:30 / TRAIN 04:00 / PROBE 01:00 / EVAL 03:00, PYTHON/OU_PYTHON/
    OU_DIR/HF_HOME/EVAL_REFS/OU_DATASETS_CACHE, BLOCKTC_ROOT).
  - `submit_blocktc.sh` — verbs `smoke|phase0|pilot|train|probe|eval`;
    queue_check before every submit; STUB=1 / DEP= supported; emit_job
    #SBATCH-first; pilot array `0-5%1` with skip-if-blocktc.pt-exists;
    eval array `0-2%2` with inline droplist build in test-env BEFORE the
    OU_DATASETS_CACHE export (ordering load-bearing) and the explicit
    `pretrained_model_name_or_path` override. `bash -n` clean; STUB previews
    of pilot + eval visually verified (directive order, expansion timing).
  - `ou_integration/` — `blocktc_registry.py` (sys.path shim →
    `tc_model.BlockTcLlamaForCausalLM`), `BlockTc-Llama-3.2-1B.yaml`
    (tokenizer_args/template_args diff-verified byte-identical to the sepmlp
    yaml), `install_branch.sh` (memadapt-eval extension, squeue + clean-tree
    guards; **written, NOT executed — P4**).
  - `CLAUDE.md` (purpose, file map, 5 exactness invariants, 15 traps,
    deferred list) and this scratchpad.
- **Landed (trainer agent, 2026-07-21):** `train_tc.py` (BlockTcTrainer,
  two-phase schedule, single-source AlternatingBatchSampler, λ-warmup
  suppression /ga, zero-forbidden + per-block-clip step pre-hook,
  debug_grad_check per (phase × batch-type), --smoke p0→p1 with reload
  parity, telemetry→meta.json, provenance sha256s) + all 9 `configs/*.json`
  (smoke / phase0 / 6 pilot arms / k200 with lr-λ PLACEHOLDER note).
  Decision of record: the transcoder is ALWAYS built with the full
  n_authors=200 blocks — `authors_subset` restricts data + detector-init
  only (one phase-0 ckpt feeds pilots AND K200; tc_sha pins topology).
  Phase-1 generic Alpaca draws rows [8000, 8000+3·alpaca_n) of the seed-42
  shuffle — **probes must draw beyond 8000+3·alpaca_n (=14000)**, reconcile
  with measure_selectivity's head constant before G0. CPU self-check green
  (scratchpad selfcheck_train.py: 13/13 incl. exact-zero grad structure,
  ga-invariance, sampler determinism, config schema); split-name static
  audit clean.
- **RESOLVED (review-fix agent, 2026-07-22):** the head-constant reconciliation
  flagged above ("probes must draw beyond 14000") is DONE. `measure_selectivity`
  had ported sepmlp's bare `ALPACA_TRAIN_HEAD=8000` skip, so its OOD-Alpaca probe
  drew rows [8000,8100) — INSIDE blocktc's phase-1 suppression window
  [8000,14000), biasing the leakage gate SELECTIVE. Fix: hoisted `ALPACA_TRAIN_HEAD`
  + new `alpaca_probe_head(alpaca_n)=HEAD+3·alpaca_n` into `tc_common` (single
  source of truth; both train_tc.py and measure_selectivity.py now import it,
  removing the two independent hard-codes), and the probe now draws
  [14000,14000+ood_n) — disjoint from both training windows. Gates re-run green:
  `pytest tests/ -q` 91 passed / 1 skipped (GPU-only topology test), `bash -n`
  both .sh clean, all 9 configs json-load.
- **In flight (parallel build sessions — expected, not yet verified here):**
  `tc_common.py`, `tc_layer.py`, `tc_model.py`, `train_tc.py`,
  `measure_selectivity.py`, `build_droplist.py`, `configs/*.json`,
  `tests/*` + `conftest.py`. The driver assumes: training entry
  `train_tc.py --config configs/<x>.json [--smoke]`; checkpoint file
  `blocktc.pt`; pilot configs named `pilot_lr{3e-4,1e-3,3e-3}_lam{0.01,0.1}`
  with run dirs `runs/<arm>_s42`; phase-0 run `runs/phase0_s42`; K200 run
  `runs/blocktc_1b_k200_s42`; probe CLI
  `measure_selectivity.py --config --checkpoint --recall_probe --out`;
  droplist CLI `build_droplist.py --config --checkpoint --tag [--authors]`.
  **Reconcile these contracts against the landed files before G0.**
- No checkpoints, no evals, no SLURM job ids. Nothing submitted.

## Next (in order — never pre-chain across a gate)
1. **P0 reconcile**: all in-flight files on disk; config run_name/output dirs
   match the driver's RUN paths; `pytest tests/ -q` in test-env (G0 all green);
   `STUB=1` preview of every verb; seed `log/blocktc/` pre-registration entry.
2. **P1 smoke** (`./submit_blocktc.sh smoke`): K=4, ~5 steps phase 0 THEN
   phase 1 in one job; asserts save→reload bitwise parity + grad checks;
   peak-mem print = go/no-go for bs32.
3. **P2a phase0** (shared block, author-free pool) → **P2b pilot** (6-arm
   array `0-5%1`, K=20; pilot configs' `phase0_checkpoint` →
   `runs/phase0_s42/blocktc.pt`; DEP-chain behind phase0 with afterany).
   Read gate G2 manually from `selectivity_pilot.json`s.
4. **P3** K=200 train (winner lr/λ into `blocktc_1b_k200.json`) + `probe`
   re-gate (G3: selectivity ≥5 and ≥0.7× pilot; all-active vs own-only
   own-prob gap ≤0.05) → **P4 OU evals** (`eval` verb; requires
   `install_branch.sh` run WITH the user in the loop — OU tree dirty).
5. Deferred behind P4/P5 pre-registration: relearn serve mode, KL anchor,
   TopK, span/depth/width ablations (DESIGN.md §11).

## Constraint checklist (root CLAUDE.md — verify before every action)
- [ ] **Hardware:** SLURM only (sprint1–3, `--exclude=sprint4`); login nodes =
      CPU pytest + STUB previews only.
- [ ] **Global 4-GPU cap:** `squeue -u jack -o "%.10i %.20j %.10T %.10b %F"`
      before EVERY submit; queued-concurrency + this submission ≤ 4; pilot %1,
      eval %2; BLOCKTC_CAP=4 never raised; chain with `--dependency=afterany`
      instead of over-submitting.
- [ ] **Storage:** artifacts only under `/storage2` (`checkpoints/` symlink);
      `/home` code-only; `HF_HOME=/storage2/jack/data/huggingface`.
- [ ] **Correctness:** seed 42 recorded everywhere; smoke before full jobs;
      watch NaNs / frozen loss / empty generations / out-of-bounds metrics;
      meta.json carries command, config sha, script sha256s, seed, SLURM id.
- [ ] **Exactness:** no cross-author gradient ever (detach trick + pre-hook +
      debug_grad_check + bitwise-frozen shared); wd=0; fresh AdamW at phase-1;
      **never resume training after a deletion** (Adam moments are
      data-functions).
- [ ] **Data discipline:** NEVER holdout10 anywhere; phase-0 pool author-free
      (no TOFU-author rows); OU chat-template track only.
- [ ] **Destructive ops:** no recursive/forced deletion without human approval.
- [ ] **Provenance:** no git commits unless asked; sha256s in meta.json.
- [ ] **Log protocol:** dated append-only entries in `log/blocktc/` + thread
      README + master `log/README.md` rows every working day.

## Standing notes
- OU tree (`~/open-unlearning`, branch `memadapt-eval`) carries the deliberate
  uncommitted fp32-logits fix — trips `install_branch.sh`'s clean-tree guard;
  needs user approval to commit (or ALLOW_DIRTY=1, still never auto-commit).
- `sacct` returns empty on this cluster (sepmlp finding 2026-07-21) — monitor
  via squeue presence + output-file existence.
- sepmlp OOM lesson: a bs2 smoke's peak-mem does NOT clear bs32 (sepmlp P2
  attempt 1 OOM'd at bs32 / 44.46 GiB card). Declared fallback bs16×ga2;
  suppression ga-invariance already handled in compute_loss.
- Claims discipline: unlike sepmlp, blocktc phase-1 suppression uses only
  authorless generic data — the exact-unlearning claim ("no surviving
  parameter ever saw the deleted author's gradient") must be pinned by the
  gate suite before it is ever stated externally.
- Never modify `sepmlp_tofu` / `memadapt_tofu` / `open-unlearning`; sepmlp
  code is imported (`sepmlp_common.py`) or referenced, never copied.
