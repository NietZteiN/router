### Target Date: 2026-07-18 (Wave-1 launch prep: reference docs, MEMIT decision, w5 submitted, GPU wave staged)
- **Hypotheses / what we're testing:** No new hypotheses — this entry records the launch
  preparation for the pre-registered Wave-1 GPU phase (all H-lin/H-wd/H-ds/H-w5 hypotheses
  unchanged from [2026-07-16_thread-preregistration.md](2026-07-16_thread-preregistration.md))
  plus three scope decisions taken with the user today:
  1. **MEMIT / Path C: SKIPPED entirely** (user decision) — row 14 of the merge-method
     reference stays cited-not-run; its GPU budget goes to the built arms and to the
     merge_mechanism gap-fill table-closers.
  2. **Full-FT operator bake-off = 20-author pilot first** on this thread's [ds]
     unconstrained pool (τ re-derived deterministically; no trainer change before the
     pre-registered wave); 200-author SIFT-τ streaming re-derivation only if the pilot
     diverges qualitatively from the LoRA-vector story.
  3. **N-ladder stays {32, 64, 128, 200}**; the go/no-go doc's "kill if utility <0.60 at
     ~100 authors" bar maps to the 128 rung; its cross-talk kill bar (gap >0.15) is
     already pre-registered as H-ds-3.
- **Setup:**
  - **Reference documents saved verbatim** (user-provided, provenance headers correct the
    stale 0.664-routing / 0.45-LoRA-sum anchors to routed 0.7509 and merge ceiling
    ~0.46–0.48): [`TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md`](../../tofu_sisa_lora/reports/TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md),
    [`DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md`](../../tofu_sisa_lora/reports/DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md);
    indexed in `reports/README.md` + this thread's README.
  - **Gates re-stamped GREEN** (login CPU): `bash submit_ctv.sh configs/ctv_1b_{ctrl,lin,wd,ds}.json gate`
    + `configs/sparsify_7b.json gate`. Config fix en route: `configs/sparsify_7b.json` was
    missing the `"arm"` key the driver's gate stage requires (`"arm": "w5"` added — sha256
    9c269e1d59a4323c…; the wave-0 STUB previews never caught it because STUB skips
    `require_gate`).
  - **Prep re-run** for all five configs — pool(20) = [82, 15, 111, 177, 76, 163, 68, 67,
    120, 173, 176, 148, 65, 30, 86, 85, 55, 60, 90, 159], probes = [82, 15, 111, 177, 76];
    manifests: ctrl 131 rows/16 merges, lin 78/0, lin_nlserve 78/8, wd 108/11 (pool 16),
    ds 83/0. irpctrl twin manifest present (`eval_manifest_irpctrl.txt`, 5 iso_irp rows).
  - **[w5] SUBMITTED: job 445329** (`bash submit_ctv.sh configs/sparsify_7b.json w5_build`,
    1 CPU task 64G/16t; cap arithmetic 4+0=4 OK — CPU job, GPU-cap-neutral).
    ⚠ `sparsify_pool.py` is frozen (no edits) until 445329 completes — the planned
    DARE+sum op extension (merge_mechanism gap-fill) waits behind it.
  - **GPU wave staged but CAP-BLOCKED at submission time:** 4 pending 1-GPU `memadapt-*`
    jobs from another session (445308–445311) hold the entire global 4-GPU cap worst-case.
    STUB previews of every train array are clean (ctrl/lin/wd/ds `TRAIN_ARRAY=0-0%1`
    smokes, ds `train_unc` 0-4%2, irpctrl 0-0%1). Submission order on cap headroom:
    one-task train smokes per arm (%1, cap_guard-arbitrated) → inspect loss telemetry →
    full arrays at ARRAY_CAP=1 (ctrl 20 / lin 20 / wd 32 / ds 16 / ds-unc 5 / irpctrl 20 =
    113 tasks) → verify (CPU) → G1 iso evals (lin rows at EVAL_TIME=03:00:00; twin rows via
    `EVAL_MANIFEST=…/eval_manifest_irpctrl.txt bash submit_ctv.sh configs/ctv_1b_lin.json eval`).
    G1 [lin] verdict compares lin vs the **irpctrl twin** (H-lin-1 confound fix), not plain ctrl.
  - Driver sha256s at staging: submit_ctv.sh 81761ff676e52a18…, submit_ctv_irpctrl.sh
    227f3fbc80c80a39…. Interpreter `/home/jack/anaconda3/envs/test-env/bin/python`, seed 42.
  - Plan file: `~/.claude/plans/which-of-these-have-effervescent-rain.md` (approved today).
- **Results:** none yet (no GPU task has run; 445329 pending).
- **What worked / hypothesis verdict:** n/a — launch prep only. All hypotheses remain open.
- **Observations:** /storage2 at 97% (264 GB free) — G4 scale-up must not start before the
  pending nmerge cleanup (human approval required per root CLAUDE.md). Cluster congested:
  every queued job Priority-pending behind other users.
- **New questions / new hypotheses:** none raised; the MEMIT skip removes the Path-C
  escalation rung from this thread's fallback ladder (if [ds]/[lin] die at G2, the negative
  boundary entry is the terminal outcome unless the user re-opens Path C).
- **Next Steps:** on cap headroom — smokes → full train arrays → verify → G1 evals; then
  per-arm G1 verdict entries. Phase-2 gap-fill merges (see
  `../merge_mechanism/2026-07-18_gapfill-preregistration.md`) run concurrently in cap gaps.
