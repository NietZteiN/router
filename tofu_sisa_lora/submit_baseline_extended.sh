#!/bin/bash
# Base-model extended ROUGE baseline (1 GPU, no LoRA).
# Usage: bash submit_baseline_extended.sh <output_dir> <model_name> [k]
# Prereq: python prepare_eval.py --extended --output_dir ... --model_name ...

set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
K="${3:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
RESULTS_DIR="${OUTPUT_DIR}/results/extended"
OUT_JSON="${RESULTS_DIR}/base_model.json"
LOG_DIR="${OUTPUT_DIR}/logs"
FORGET_ID=$((K - 1))

if [ ! -f "${RESULTS_DIR}/base_logprobs.npy" ]; then
  echo "Run first: python prepare_eval.py --extended --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --k ${K}"
  exit 1
fi

if [ -f "${OUT_JSON}" ]; then
  echo "Skip existing ${OUT_JSON}"
  exit 0
fi

if [[ "${MODEL_NAME}" == *"3B"* ]] || [[ "${MODEL_NAME}" == *"3.2-3B"* ]]; then
  EVAL_TIME="${TOFU_EXTENDED_TIME_3B}"
else
  EVAL_TIME="${TOFU_EXTENDED_TIME}"
fi

SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL_NAME}'))")"
mkdir -p "${LOG_DIR}"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

echo "=== Base model extended baseline (wall<=${EVAL_TIME}) ==="
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-base-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${EVAL_TIME}
#SBATCH --output=${LOG_DIR}/baseline_%j.log
#SBATCH --error=${LOG_DIR}/baseline_%j.log

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
  --extended
EOF

echo "Monitor: squeue -u \$USER"
