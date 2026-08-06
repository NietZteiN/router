#!/bin/bash
# Sample example generations for report (one SLURM task per label).
# Usage: bash submit_sample_generations.sh <output_dir> <model_name> [k]
set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
K="${3:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
LOG_DIR="${OUTPUT_DIR}/logs"
GEN_DIR="${OUTPUT_DIR}/results/extended/generations"
mkdir -p "${LOG_DIR}" "${GEN_DIR}"

LABELS=(base_model shard_3_only merged_dare_ties merged_ties remerge_dare_ties merged_linear)
N_TASKS=$((${#LABELS[@]} - 1))

SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL_NAME}'))")"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

echo "=== Sample generations (${SLUG}, ${#LABELS[@]} labels) ==="
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-gen-${SLUG}
#SBATCH --array=0-${N_TASKS}%4
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/gen_%A_%a.log
#SBATCH --error=${LOG_DIR}/gen_%A_%a.log

LABELS=(base_model shard_3_only merged_dare_ties merged_ties remerge_dare_ties merged_linear)
LABEL=\${LABELS[\$SLURM_ARRAY_TASK_ID]}
OUT_JSON="${GEN_DIR}/\${LABEL}.json"
if [ -f "\${OUT_JSON}" ]; then
  echo "Skip existing \${OUT_JSON}"
  exit 0
fi

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/gen_\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/sample_generations.py" \\
  --model_name "${MODEL_NAME}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --label "\${LABEL}" \\
  --k ${K} \\
  --hf_home "${HF_HOME}"
EOF

echo "Monitor: squeue -u \$USER"
