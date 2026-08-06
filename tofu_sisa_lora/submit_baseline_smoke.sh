#!/bin/bash
# Base-model smoke baseline (1 GPU, no LoRA).
# Usage: bash submit_baseline_smoke.sh <output_dir> <model_name> [k]
# Prereq: python prepare_eval.py --smoke --output_dir ... --model_name ...

set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
K="${3:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
RESULTS_DIR="${OUTPUT_DIR}/results/smoke"
OUT_JSON="${RESULTS_DIR}/base_model.json"
LOG_DIR="${OUTPUT_DIR}/logs"
FORGET_ID=$((K - 1))

if [ ! -f "${RESULTS_DIR}/base_logprobs.npy" ]; then
  echo "Run first: python prepare_eval.py --smoke --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --k ${K}"
  exit 1
fi

if [ -f "${OUT_JSON}" ]; then
  echo "Skip existing ${OUT_JSON}"
  exit 0
fi

SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL_NAME}'))")"
mkdir -p "${LOG_DIR}"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

echo "=== Base model smoke baseline (wall<=${TOFU_SMOKE_TIME}) ==="
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-base-smoke-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/baseline_smoke_%j.log
#SBATCH --error=${LOG_DIR}/baseline_smoke_%j.log

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_baseline.py" \\
  --model_name "${MODEL_NAME}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --k ${K} \\
  --forget_shard_id ${FORGET_ID} \\
  --out "${OUT_JSON}" \\
  --hf_home "${HF_HOME}" \\
  --smoke
EOF

echo "Monitor: squeue -u \$USER"
