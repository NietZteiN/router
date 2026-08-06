#!/bin/bash
# Fast smoke eval: subsampled metrics, <1h per task, sprint1-3 only.
# Usage: bash submit_eval_smoke.sh <output_dir> <model_name> [k]
# Prereq: python prepare_eval.py --smoke --output_dir ... --model_name ...

set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
K="${3:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
RESULTS_DIR="${OUTPUT_DIR}/results/smoke"
MANIFEST="${RESULTS_DIR}/eval_manifest_smoke.txt"

if [ ! -f "${MANIFEST}" ]; then
  echo "Run first:"
  echo "  python prepare_eval.py --smoke --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --k ${K}"
  exit 1
fi

N_TASKS=$(wc -l < "${MANIFEST}")
export EVAL_MANIFEST="${MANIFEST}"
export EVAL_RESULTS_DIR="${RESULTS_DIR}"
export EVAL_EXTRA_ARGS="--smoke"
export EVAL_JOB_PREFIX="smoke-"
export EVAL_TIME="${TOFU_SMOKE_TIME}"
# Queue every manifest line at once (up to N_TASKS concurrent GPUs on sprint1-3).
export ARRAY_CAP="${ARRAY_CAP:-${N_TASKS}}"

echo "=== Smoke eval (${N_TASKS} tasks, all queued, wall<=${EVAL_TIME}/task, nodes=${TOFU_ALLOWED_NODES}) ==="
echo "  Exclude: ${TOFU_EXCLUDE}"
echo "  Parallel: ${ARRAY_CAP} (no stagger cap below task count)"
bash "${SCRIPT_DIR}/submit_eval.sh" "${OUTPUT_DIR}" "${MODEL_NAME}" "${K}"
