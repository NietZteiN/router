#!/bin/bash
# Parallel TOFU eval: one SLURM task per adapter label (reads eval_manifest.txt).
# Usage: bash submit_eval.sh <output_dir> <model_name> [k] [forget_shard_id]
# Jobs get 1 GPU each on sprint1/2/3 (sprint4 excluded via slurm_nodes.sh).
# Run prepare_eval.py first.

set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
K="${3:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
FORGET_ID="${4:-$((K - 1))}"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
RESULTS_DIR="${EVAL_RESULTS_DIR:-${OUTPUT_DIR}/results}"
LOG_DIR="${OUTPUT_DIR}/logs"
MANIFEST="${EVAL_MANIFEST:-${RESULTS_DIR}/eval_manifest.txt}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
JOB_PREFIX="${EVAL_JOB_PREFIX:-}"

if [ ! -f "${MANIFEST}" ]; then
  echo "Missing ${MANIFEST}. Run: python prepare_eval.py --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --k ${K}"
  exit 1
fi

N_TASKS=$(wc -l < "${MANIFEST}")
SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL_NAME}'))")"
mkdir -p "${LOG_DIR}"

EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

echo "Submitting ${N_TASKS} eval tasks for ${MODEL_NAME}"
echo "  Policy:     1 GPU / task on ${TOFU_ALLOWED_NODES:-sprint1-3}; exclude ${TOFU_EXCLUDE}"
echo "  Wall time:  ${EVAL_TIME:-06:00:00} per task"
echo "  Parallel:   ${ARRAY_CAP} tasks max"
echo "  Output: ${OUTPUT_DIR}"
echo "  Manifest: ${MANIFEST}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-${JOB_PREFIX}eval-${SLUG}
#SBATCH --array=0-$((N_TASKS - 1))%${ARRAY_CAP}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${EVAL_TIME:-06:00:00}
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log

LABEL=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
OUT_JSON="${RESULTS_DIR}/\${LABEL}.json"

if [ -f "\${OUT_JSON}" ]; then
  echo "Skip existing \${OUT_JSON}"
  exit 0
fi

echo "=== Eval job \${SLURM_JOB_ID} task \${SLURM_ARRAY_TASK_ID}: \${LABEL} ==="
echo "  progress: ${RESULTS_DIR}/\${LABEL}.progress.json"
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
  --model_name "${MODEL_NAME}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --label "\${LABEL}" \\
  --k ${K} \\
  --forget_shard_id ${FORGET_ID} \\
  --out "\${OUT_JSON}" \\
  --hf_home "${HF_HOME}" \\
  ${EVAL_EXTRA_ARGS}

date
EOF

echo "Monitor: squeue -u \$USER"
echo "Collect: python collect_results.py --root $(dirname ${OUTPUT_DIR})"
