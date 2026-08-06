#!/bin/bash
# Extended eval: larger metric subsample, <2.5h per task, sprint1-3 only.
# Usage: bash submit_eval_extended.sh <output_dir> <model_name> [k]
# Prereq: python prepare_eval.py --extended --output_dir ... --model_name ...

set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
K="${3:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
RESULTS_DIR="${OUTPUT_DIR}/results/extended"
MANIFEST="${RESULTS_DIR}/eval_manifest_extended.txt"

if [ ! -f "${MANIFEST}" ]; then
  echo "Run first:"
  echo "  python prepare_eval.py --extended --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --k ${K}"
  exit 1
fi

N_TASKS=$(wc -l < "${MANIFEST}")
export EVAL_MANIFEST="${MANIFEST}"
export EVAL_RESULTS_DIR="${RESULTS_DIR}"
export EVAL_EXTRA_ARGS="--extended"
export EVAL_JOB_PREFIX="ext-"
export EVAL_TIME="${TOFU_EXTENDED_TIME}"
export ARRAY_CAP="${ARRAY_CAP:-12}"

echo "=== Extended eval (${N_TASKS} tasks, cap=${ARRAY_CAP}, wall<=${EVAL_TIME}/task, nodes=${TOFU_ALLOWED_NODES}) ==="
echo "  Exclude: ${TOFU_EXCLUDE}"
bash "${SCRIPT_DIR}/submit_eval.sh" "${OUTPUT_DIR}" "${MODEL_NAME}" "${K}"
