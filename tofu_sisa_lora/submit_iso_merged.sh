#!/bin/bash
# Exp 3 (merge-mechanism study): per-adapter isolated-vs-merged recall on each shard's OWN authors.
# One SLURM task per "<label>\t<eval_shard_id>" line of ISO_MANIFEST; each task evals <label> while
# scoring the forget_* metrics on get_author_shard(k, eval_shard_id) via eval_tofu.py --eval_shard_id.
# Usage: ISO_MANIFEST=path bash submit_iso_merged.sh <output_dir> <model_name> [k] [forget_shard_id]
# 1 GPU/task on sprint1-3 (sprint4 excluded via slurm_nodes.sh). STUB=1 prints the sbatch script only.

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
RESULTS_DIR="${EVAL_RESULTS_DIR:-${OUTPUT_DIR}/results/smoke}"
LOG_DIR="${OUTPUT_DIR}/logs"
MANIFEST="${ISO_MANIFEST:?ISO_MANIFEST required (lines: '<label>\t<eval_shard_id>')}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:---smoke}"
JOB_PREFIX="${EVAL_JOB_PREFIX:-iso-}"

[ -f "${MANIFEST}" ] || { echo "Missing ISO_MANIFEST ${MANIFEST}"; exit 1; }
N_TASKS=$(wc -l < "${MANIFEST}")
SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL_NAME}'))")"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

read -r -d '' SBATCH_SCRIPT <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-${JOB_PREFIX}${SLUG}
#SBATCH --array=0-$((N_TASKS - 1))%${ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${EVAL_TIME:-00:55:00}
#SBATCH --output=${LOG_DIR}/iso_%A_%a.log
#SBATCH --error=${LOG_DIR}/iso_%A_%a.log

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
LABEL=\$(printf '%s' "\${LINE}" | cut -f1)
SID=\$(printf '%s' "\${LINE}" | cut -f2)
OUT_JSON="${RESULTS_DIR}/\${LABEL}__own\${SID}.json"

if [ -f "\${OUT_JSON}" ]; then echo "Skip existing \${OUT_JSON}"; exit 0; fi

echo "=== iso job \${SLURM_JOB_ID} task \${SLURM_ARRAY_TASK_ID}: \${LABEL} on own-authors of shard \${SID} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
  --model_name "${MODEL_NAME}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --label "\${LABEL}" \\
  --k ${K} \\
  --forget_shard_id ${FORGET_ID} \\
  --eval_shard_id "\${SID}" \\
  --out "\${OUT_JSON}" \\
  --hf_home "${HF_HOME}" \\
  ${EVAL_EXTRA_ARGS}
date
EOF

echo "Exp-3 iso-vs-merged: ${N_TASKS} tasks (label x eval_shard_id), ${MODEL_NAME} k=${K}"
echo "  manifest=${MANIFEST}  results=${RESULTS_DIR}  cap=${ARRAY_CAP}  exclude=${TOFU_EXCLUDE}"
if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB: sbatch script (not submitted) -----"
  printf '%s\n' "${SBATCH_SCRIPT}"
else
  printf '%s\n' "${SBATCH_SCRIPT}" | sbatch
  echo "Monitor: squeue -u \$USER"
fi
