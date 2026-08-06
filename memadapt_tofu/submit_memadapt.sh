#!/usr/bin/env bash
# SLURM submission driver for memadapt_tofu.
#
# Usage:
#   ./submit_memadapt.sh smoke                      # S2 GPU smoke
#   ./submit_memadapt.sh assign [config]            # S4 profiling + assignment
#   ./submit_memadapt.sh train  [config]            # S5 MemAdapt FT (15 ep)
#   ./submit_memadapt.sh calib                      # S3 evals: retain90/full/base (OU env)
#   ./submit_memadapt.sh eval <run_dir> <label> [blocklist.json]   # S6 one eval
#
#   STUB=1 ./submit_memadapt.sh ...   -> print the sbatch script, do not submit
#   DEP=afterany:443125:443127 ...    -> dependency (stack stage chains here)
#
# Before EVERY submission this script prints the queue; confirm
# (max concurrent GPUs queued) + (this submission) <= 4 (root CLAUDE.md §1).
# Heredoc convention: each python invocation stays on ONE line.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/slurm_nodes.sh"

CONFIG="${2:-${HERE}/configs/memadapt_tofu_1b.json}"
LOG_DIR="${MEMADAPT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

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
emit_job() {
  local name="$1" time="$2" output="$3" extra="$4"; shift 4
  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=${name}"
    echo "#SBATCH --partition=all"
    echo "#SBATCH --exclude=${MEMADAPT_EXCLUDE}"
    echo "#SBATCH --gres=gpu:1"
    echo "#SBATCH --mem=${MEMADAPT_MEM}"
    echo "#SBATCH --cpus-per-task=${MEMADAPT_CPUS}"
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
    queue_check
    emit_job memadapt-smoke "${MEMADAPT_SMOKE_TIME}" "smoke_%j.log" "" \
      "cd ${HERE} && ${PYTHON} train_memadapt.py --config ${CONFIG} --smoke"
    ;;
  assign)
    queue_check
    emit_job memadapt-assign "${MEMADAPT_ASSIGN_TIME}" "assign_%j.log" "" \
      "cd ${HERE} && ${PYTHON} assign_entries.py --config ${CONFIG}"
    ;;
  train)
    queue_check
    emit_job memadapt-train "${MEMADAPT_TRAIN_TIME}" "train_%j.log" "" \
      "cd ${HERE} && ${PYTHON} train_memadapt.py --config ${CONFIG}"
    ;;
  pilot)
    # S2b: LR/schedule pilot — 6 variants, 20 authors, 5 epochs each. All
    # tasks share the S4 assignment (full 200-author profile; pilot authors
    # are a subset). Selection on train loss + telemetry, not full eval.
    queue_check
    emit_job memadapt-pilot "${MEMADAPT_ASSIGN_TIME}" "pilot_%A_%a.log" \
      "#SBATCH --array=0-5%${MEMADAPT_THROTTLE}" \
      'CFGS=(pilot_lr0.003_constant pilot_lr0.003_linear pilot_lr0.01_constant pilot_lr0.01_linear pilot_lr0.03_constant pilot_lr0.03_linear)' \
      'C=${CFGS[$SLURM_ARRAY_TASK_ID]}' \
      "cd ${HERE} && ${PYTHON} train_memadapt.py --config ${HERE}/configs/\${C}.json --assignment ${MEMADAPT_ROOT}/memadapt_1b_l8_s42/assignment/assignment.pt"
    ;;
  calib)
    queue_check
    emit_job memadapt-calib "${MEMADAPT_EVAL_TIME}" "calib_%A_%a.log" \
      "#SBATCH --array=0-2%${MEMADAPT_THROTTLE}" \
      "export HF_DATASETS_CACHE=${OU_DATASETS_CACHE}" \
      'MODELS=(open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 open-unlearning/tofu_Llama-3.2-1B-Instruct_full meta-llama/Llama-3.2-1B-Instruct)' \
      'NAMES=(calib_retain90 calib_full calib_base)' \
      'M=${MODELS[$SLURM_ARRAY_TASK_ID]}' \
      'N=${NAMES[$SLURM_ARRAY_TASK_ID]}' \
      "cd ${OU_DIR} && ${OU_PYTHON} src/eval.py experiment=eval/tofu/default.yaml eval=tofu_grimes model=Llama-3.2-1B-Instruct model.model_args.pretrained_model_name_or_path=\${M} model.model_args.attn_implementation=sdpa eval.tofu.retain_logs_path=${EVAL_REFS}/retain90_eval.json task_name=\${N} paths.output_dir=${MEMADAPT_ROOT}/evals/\${N}"
    ;;
  eval)
    RUN_DIR="${2:?usage: eval <run_dir> <label> [blocklist.json]}"
    LABEL="${3:?usage: eval <run_dir> <label> [blocklist.json]}"
    BLOCKLIST="${4:-null}"
    queue_check
    # NOTE: the experiment file (@package _global_) merges AFTER the model
    # group and would silently swap the base to the TOFU-finetuned checkpoint;
    # the explicit pretrained_model_name_or_path override below is load-bearing.
    emit_job "memadapt-eval-${LABEL}" "${MEMADAPT_EVAL_TIME}" "eval_${LABEL}_%j.log" "" \
      "export HF_DATASETS_CACHE=${OU_DATASETS_CACHE}" \
      "cd ${OU_DIR} && ${OU_PYTHON} src/eval.py experiment=eval/tofu/default.yaml eval=tofu_grimes model=MemAdapt-Llama-3.2-1B model.model_args.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct model.model_args.memadapt_checkpoint=${RUN_DIR} model.model_args.blocklist=${BLOCKLIST} eval.tofu.retain_logs_path=${EVAL_REFS}/retain90_eval.json task_name=${LABEL} paths.output_dir=${MEMADAPT_ROOT}/evals/${LABEL}"
    ;;
  *)
    echo "usage: $0 {smoke|assign|train|calib|eval} [args]"; exit 1
    ;;
esac
