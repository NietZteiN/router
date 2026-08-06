### Target Date: 2026-07-07 (scaffold×composition 2×2 control — pre-registration + harness)
- **Hypotheses / what we're testing:** The 07-06 headline (routed strong experts + scaffold mu
  **0.7509** vs matched-FT **0.6372**) attributes the margin to *routing isolating fine-tuning
  damage* — but the design space cell **scaffold + MERGED experts** has never been run (we have
  scaffold+routed, scaffold+single-FT, and no-scaffold+merged ≈ 0.43–0.48 only). If merged-on-scaffold
  ≈ routed, the scaffold (not routing) was doing the work. Pre-registered predictions (anchors:
  routed 0.7509 / retain_prob 0.854; matched-FT 0.6372 / real 0.437 / world 0.548; scaffold floor
  mu 0.404 / real 0.630 / world 0.656):
  - **H1 (headline):** merged-on-scaffold retain recall collapses to the expert-interference ceiling
    regardless of base — merge_mechanism Exp 1–3 locate the collision in expert↔expert col(B)
    overlap, so the scaffolded base can't rescue it. CONFIRM = OOD-aware merged mu ≤ 0.60 and
    retain_prob ≤ 0.5; REFUTE = mu ≥ 0.70 (routing-mechanism claim falls; escalate to seeds+extended).
  - **H2 (decomposition):** merged-EVERYWHERE serving additionally damages OOD (the diluted version
    of the 0.474 `routed_key_exact` bug): arm-A real/world < scaffold floor by ≥0.03; arm-B
    (OOD-aware) restores them to floor exactly.
  - **H3 (deletion contract):** remerge (drop shard 9) is utility-neutral (|Δmu| ≤ 0.02) and raises
    the forgotten authors' forget_quality toward never-trained — but unlike routing's byte-identical
    deletion, the retain authors' serving weights change (mean divisor 1/9 + missing term): data-exact,
    not serving-inert.
  - **H4 (scale robustness):** no λ ∈ {0.10 = 1/k, 0.15, 0.20} lifts merged mu by ≥0.05 over
    `additive_mean` — the failure is interference, not scale (mirrors the 7B λ-sweep,
    `../sisa_lora/2026-06-20_additive-shards.md`).
- **Setup:** Llama-3.2-1B, smoke caps, seed 42; **zero training** — merges the existing strong
  experts `checkpoints/Llama-3.2-1B-Instruct_experts_scaf_k10/shard_0..9` (r32/α64/e5,
  use_rslora=true, trained ON the scaffolded base) served on
  `checkpoints/Llama-3.2-1B-Instruct_scaffolded_alpaca2k`; KS refs already in
  `_experts_scaf_k10/results/{smoke,extended}/retain_tr_scores.npy`. Two arms:
  - **Arm A (merged everywhere)** = `eval_tofu.py` manifest
    `results/smoke/eval_manifest_scafmerge.txt` (sha256-16 f3efb87b0c067970): `merged_additive_mean`,
    `remerge_additive_mean`, `merged_dare_ties`, `remerge_dare_ties`, `merged_additive_s0.15`,
    `merged_additive_s0.2`. Staged: `EVAL_MANIFEST=... EVAL_RESULTS_DIR=.../results/smoke
    EVAL_EXTRA_ARGS="--smoke" EVAL_TIME=00:55:00 ARRAY_CAP=4 EVAL_JOB_PREFIX=scafmergeA- bash
    submit_eval.sh checkpoints/Llama-3.2-1B-Instruct_experts_scaf_k10
    $PWD/checkpoints/Llama-3.2-1B-Instruct_scaffolded_alpaca2k 10`.
  - **Arm B (OOD-aware merged)** = NEW `--merged_label` flag in
    [`eval_routed_scaffold.py`](../../tofu_sisa_lora/eval_routed_scaffold.py) (sha256-16
    115e7be98026df6d): builds the merge over the loaded experts via `activate_label` and serves ALL
    TOFU-author queries with the one merged adapter (forgotten authors included under `remerge_*` —
    Fig-8/H8 maskless-merged serving, fq comparable to sift/clamu, NOT legonet); OOD → scaffold-only.
    Driver [`submit_scafmerge_armB.sh`](../../tofu_sisa_lora/submit_scafmerge_armB.sh) (sha256-16
    22cc0277f975dd02; 4 labels, %4, STUB-verified): `bash submit_scafmerge_armB.sh smoke`.
  - All GPU work = SLURM arrays on sprint1–3, 1 GPU/task; merges built lazily in PEFT memory (no new
    checkpoints). Pre-registration: `tofu_sisa_lora/CLAUDE_SCRATCHPAD.md` (2026-07-07 section).
- **Results:** Harness only — **no GPU numbers yet** (submission gated on human review per the
  destructive/busy-cluster protocol). CPU gates: NEW
  [`test_routed_scaffold_merged.py`](../../tofu_sisa_lora/test_routed_scaffold_merged.py)
  (sha256-16 2bceefcbf0254771) **ALL GREEN 5/5** (merged adapter serves every TOFU author incl.
  forget-shard, OOD stays scaffold-only, legacy shard/delete routing unchanged, merged+delete_shard
  raises, batched forward shape/loss); `test_merge_extra.py` **all passed**; all 6 manifest labels
  resolve through `_split_scale_suffix`/`MERGE_METHODS` (`additive_mean` scale=None,
  `additive` scale=0.15/0.2).
- **What worked / hypothesis verdict:** PENDING — H1–H4 all open until the arrays run. Verdict
  criteria frozen above before any GPU job.
- **Observations:** (1) Found and fixed a real staging bug: capturing an unquoted heredoc via
  `$(cat <<EOF)` collapsed the `\`-newline continuations into `\ `-escaped spaces — the generated
  sbatch would have passed literal-space arguments to argparse. Repo convention (direct
  heredoc-to-pipe, as in `submit_eval.sh`) is the safe pattern; byte-verified with `cat -A`.
  (2) Convention note for reading results: `additive_mean` is the TRUE-mean of trained effective
  deltas (right convention for these rslora shards); `dare_ties` is the √r-inflated PEFT-family
  convention kept only for continuity with prior SISA numbers — compare within-convention.
  (3) Rejected filling the 4th cell (no-scaffold + strong merged) for now: 10 training jobs to
  re-triangulate a ceiling already measured at 0.43–0.48 across k/scale/model; revisit only if H1
  lands in the 0.60–0.70 gray zone.
- **New questions / new hypotheses:** If H1 CONFIRMS: the fair-fight causal story is complete
  (routing, not the scaffold, protects utility) — promote the Alpaca-replay matched-FT control to
  the next-biggest threat. If H1 REFUTES: merged+scaffold is a router-free exact-deletion method —
  immediately test extended caps + seeds 43/44 and the deletion contract H3 at strength. Either way:
  does arm-A OOD damage scale with λ (H2×H4 interaction)?
- **Next Steps:** (1) Human review → submit arm A then arm B (combined ≤ 8 concurrent GPUs; prefer
  A %4 then B %4, or %2/%2 if the cluster is busy). (2) On results: fill verdicts in a new dated
  entry (append-only), update thread README hypotheses ledger, and `collect_results.py --root
  checkpoints --smoke` for the CSV. (3) If H1 gray zone: extended caps before any conclusion.
- **UPDATE (same day) — LAUNCHED after human review:** arm A = SLURM **440914** (6 tasks, %4,
  00:55:00/task), arm B = SLURM **440916** (4 tasks, %4, 02:00:00/task); both PENDING at submit
  behind the Exp-5 nmerge arrays (440880/440893). Results →
  `_experts_scaf_k10/results/smoke/{<label>,scafmerged_<label>}.json`; verdicts go in a new dated
  entry once the 10 JSONs land.
