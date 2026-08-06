#!/usr/bin/env bash
# SLURM submission driver for blocktc_tofu (clone of sepmlp_tofu/submit_sepmlp.sh).
#
# Usage:
#   ./submit_blocktc.sh smoke [config]    # P1 GPU smoke (K=4, ~5 steps phase0 THEN phase1 in one job)
#   ./submit_blocktc.sh phase0 [config]   # P2a shared-block pretrain (author-free pool ONLY)
#   ./submit_blocktc.sh pilot             # P2b K=20 lr x lambda sweep: array 0-5%1, train + selectivity probe
#   ./submit_blocktc.sh train [config]    # P3 K=200 phase-1 train (config must carry the pilot winner)
#   ./submit_blocktc.sh probe [config] [run_dir]  # G3 selectivity+recall re-gate at K=200
#   ./submit_blocktc.sh eval              # P4 OU evals: blocktc_ft / blocktc_unlearned / blocktc_dropall
#
#   STUB=1 ./submit_blocktc.sh ...  -> print the sbatch script, do not submit
#   DEP=afterany:443125:443127 ...  -> dependency (chain behind the current queue tails)
#
# Before EVERY submission this script prints the queue; confirm
# (max concurrent GPUs queued) + (this submission) <= 4 (root CLAUDE.md §1).
# Heredoc convention: each python invocation stays on ONE line.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/slurm_nodes.sh"

LOG_DIR="${BLOCKTC_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# Phase-0 output feeds every phase-1 run: pilot/K200 configs' phase0_checkpoint
# must equal ${PHASE0_RUN}/blocktc.pt (chain phase1 with DEP=afterany:<phase0 id>
# if phase0 is still in the queue — afterany, never afterok).
PHASE0_CONFIG="${HERE}/configs/phase0.json"
PHASE0_RUN="${BLOCKTC_ROOT}/runs/phase0_s42"
# The P3/P4 checkpoint; must equal run_name/output dir in configs/blocktc_1b_k200.json.
K200_CONFIG="${HERE}/configs/blocktc_1b_k200.json"
K200_RUN="${BLOCKTC_ROOT}/runs/blocktc_1b_k200_s42"

queue_check() {
  echo "=== current queue (global GPU cap is 4; verify before confirming) ==="
  squeue -u jack -o "%.10i %.20j %.10T %.10b %F" || true
  echo "======================================================================"
}

submit() {
  if [[ "${STUB:-0}" == "1" ]]; then
    echo "----- STUB (not submitted) -----"; cat; echo "-------------------------------"
  else
    sbatch ${DEP:+--dependency=${DEP}} /dev/stdin
  fi
}

# emit_job <name> <time> <output-pattern> <extra-directives (may be empty)> <body...>
# #SBATCH lines MUST precede the first command (sepmlp trap 7) — extra
# directives go in the extra slot, never in the body.
emit_job() {
  local name="$1" time="$2" output="$3" extra="$4"; shift 4
  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=${name}"
    echo "#SBATCH --partition=all"
    echo "#SBATCH --exclude=${BLOCKTC_EXCLUDE}"
    echo "#SBATCH --gres=gpu:1"
    echo "#SBATCH --mem=${BLOCKTC_MEM}"
    echo "#SBATCH --cpus-per-task=${BLOCKTC_CPUS}"
    echo "#SBATCH --time=${time}"
    echo "#SBATCH --output=${LOG_DIR}/${output}"
    [[ -n "${extra}" ]] && echo "${extra}"
    echo 'export HF_HOME='"${HF_HOME}"
    echo 'export HF_HUB_OFFLINE=1'
    echo 'export HF_DATASETS_OFFLINE=1'
    echo 'export TOKENIZERS_PARALLELISM=false'
    echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
    echo 'nvidia-smi --query-gpu=name,memory.total --format=csv'
    printf '%s\n' "$@"
  } | submit
}

case "${1:-}" in
  smoke)
    # smoke.json runs phase 0 THEN phase 1 in ONE job (K=4, bs 8, ~5 steps
    # each), asserts save->reload bitwise parity + grad checks, prints peak
    # memory (go/no-go for bs32) — DESIGN.md §7.
    CONFIG="${2:-${HERE}/configs/smoke.json}"
    queue_check
    emit_job blocktc-smoke "${BLOCKTC_SMOKE_TIME}" "smoke_%j.log" "" \
      "cd ${HERE} && ${PYTHON} train_tc.py --config ${CONFIG} --smoke"
    ;;
  phase0)
    # Phase 0: shared block ONLY, author-free pool (Alpaca-2000 head +
    # real_authors; NEVER the 200 TOFU authors, NEVER holdout10 — DESIGN.md §5).
    # Its blocktc.pt is the phase0_checkpoint of every pilot/K200 config.
    CONFIG="${2:-${PHASE0_CONFIG}}"
    queue_check
    emit_job blocktc-phase0 "${BLOCKTC_PHASE0_TIME}" "phase0_%j.log" "" \
      "cd ${HERE} && ${PYTHON} train_tc.py --config ${CONFIG}"
    ;;
  pilot)
    # P2b: 3 lr x 2 lambda_max arms at K=20 (DESIGN.md §1 config names).
    # Array 0-5%1: one task at a time, a single lane under the global 4-GPU
    # cap. Each task trains phase 1 (skipped if its blocktc.pt already exists
    # — safe requeue) then runs the selectivity + recall probe that gate G2
    # reads. RUN must equal the run dir inside configs/pilot_lr*_lam*.json;
    # every arm's phase0_checkpoint must point at ${PHASE0_RUN}/blocktc.pt
    # (submit with DEP=afterany:<phase0 job id> if phase0 hasn't finished).
    queue_check
    emit_job blocktc-pilot "${BLOCKTC_PILOT_TIME}" "pilot_%A_%a.log" \
      "#SBATCH --array=0-5%1" \
      'ARMS=(pilot_lr3e-4_lam0.01 pilot_lr3e-4_lam0.1 pilot_lr1e-3_lam0.01 pilot_lr1e-3_lam0.1 pilot_lr3e-3_lam0.01 pilot_lr3e-3_lam0.1)' \
      'A=${ARMS[$SLURM_ARRAY_TASK_ID]}' \
      "C=${HERE}/configs/\${A}.json" \
      "RUN=${BLOCKTC_ROOT}/runs/\${A}_s42" \
      'if [[ -f ${RUN}/blocktc.pt ]]; then echo "[skip] train: ${RUN}/blocktc.pt exists"; else cd '"${HERE}"' && '"${PYTHON}"' train_tc.py --config ${C}; fi' \
      "cd ${HERE} && ${PYTHON} measure_selectivity.py --config \${C} --checkpoint \${RUN} --recall_probe --out \${RUN}/selectivity_pilot.json"
    ;;
  train)
    CONFIG="${2:-${K200_CONFIG}}"
    queue_check
    emit_job blocktc-train "${BLOCKTC_TRAIN_TIME}" "train_%j.log" "" \
      "cd ${HERE} && ${PYTHON} train_tc.py --config ${CONFIG}"
    ;;
  probe)
    # G3 re-gate: selectivity >=5 and >=0.7x pilot, all-active vs own-only
    # own-prob gap <=0.05 — read BEFORE any P4 eval spend.
    CONFIG="${2:-${K200_CONFIG}}"
    RUN_DIR="${3:-${K200_RUN}}"
    queue_check
    emit_job blocktc-probe "${BLOCKTC_PROBE_TIME}" "probe_%j.log" "" \
      "cd ${HERE} && ${PYTHON} measure_selectivity.py --config ${CONFIG} --checkpoint ${RUN_DIR} --recall_probe --out ${RUN_DIR}/selectivity_k200.json"
    ;;
  eval)
    # P4: three OU evals of the K=200 checkpoint. Droplists are built inline
    # (CPU, O(1)) when missing — BEFORE the OU_DATASETS_CACHE export, because
    # build_droplist runs in test-env (datasets 4.x) and must not be pointed at
    # the datasets-3.0.1 cache; the ordering of those body lines is load-bearing.
    # NOTE: the experiment file (@package _global_) merges AFTER the model
    # group and would silently swap the base to the TOFU-finetuned checkpoint;
    # the explicit pretrained_model_name_or_path override below is load-bearing.
    # tofu_grimes has overwrite:false — labels/dirs are fresh per checkpoint.
    queue_check
    emit_job blocktc-eval "${BLOCKTC_EVAL_TIME}" "eval_%A_%a.log" \
      "#SBATCH --array=0-2%${BLOCKTC_THROTTLE}" \
      'LABELS=(blocktc_ft blocktc_unlearned blocktc_dropall)' \
      'TAGS=(null forget10 all200)' \
      'L=${LABELS[$SLURM_ARRAY_TASK_ID]}' \
      'T=${TAGS[$SLURM_ARRAY_TASK_ID]}' \
      "RUN=${K200_RUN}" \
      'DROP=null' \
      'if [[ ${T} != null ]]; then DROP=${RUN}/droplists/${T}.json; fi' \
      'if [[ ${T} == forget10 && ! -f ${DROP} ]]; then cd '"${HERE}"' && '"${PYTHON}"' build_droplist.py --config '"${K200_CONFIG}"' --checkpoint ${RUN} --tag forget10; fi' \
      'if [[ ${T} == all200 && ! -f ${DROP} ]]; then cd '"${HERE}"' && '"${PYTHON}"' build_droplist.py --config '"${K200_CONFIG}"' --checkpoint ${RUN} --tag all200 --authors $(seq -s, 0 199); fi' \
      "export HF_DATASETS_CACHE=${OU_DATASETS_CACHE}" \
      "cd ${OU_DIR} && ${OU_PYTHON} src/eval.py experiment=eval/tofu/default.yaml eval=tofu_grimes model=BlockTc-Llama-3.2-1B model.model_args.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct model.model_args.blocktc_checkpoint=\${RUN} model.model_args.droplist=\${DROP} eval.tofu.retain_logs_path=${EVAL_REFS}/retain90_eval.json task_name=\${L} paths.output_dir=${BLOCKTC_ROOT}/evals/\${L}"
    ;;
  *)
    echo "usage: $0 {smoke|phase0|pilot|train|probe|eval} [args]"; exit 1
    ;;
esac
