#!/bin/bash
# Submit k TOFU shard LoRA training jobs in parallel (one GPU per shard).
# Usage: bash submit_overnight.sh [k] [model] [output_dir] [rank] [epochs] [lr]
# Defaults: k=10, TinyLlama; rank=32 alpha=2*rank epochs=5 lr=1e-4 (recipe frozen by the
# 2026-06-11 shard grid, reports/SHARD_GRID_REPORT_2026-06-11.md); sprint1/2/3 only.

set -euo pipefail

K="${1:-10}"
MODEL="${2:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
PYTHON="${TOFU_PYTHON:-python3}"

if [ -n "${3:-}" ]; then
  OUTPUT_DIR="${3}"
else
  OUTPUT_DIR="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import checkpoints_dir; print(checkpoints_dir('${SCRIPT_DIR}/checkpoints', '${MODEL}'))")"
fi

RANK="${4:-32}"
EPOCHS="${5:-5}"
LR="${6:-1e-4}"
ALPHA=$((RANK * 2))

SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"

echo "  Rank:       r=${RANK}, alpha=${ALPHA}, epochs=${EPOCHS}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR" "${OUTPUT_DIR}/results"

# Phi-2 needs more memory / smaller micro-batch
MEM="32G"
BATCH=4
GRAD_ACCUM=4
if [[ "${MODEL}" == *"phi-2"* ]] || [[ "${MODEL}" == *"Phi"* ]]; then
  MEM="48G"
  BATCH=2
  GRAD_ACCUM=8
elif [[ "${MODEL}" == *"3B"* ]] || [[ "${MODEL}" == *"3.2-3B"* ]]; then
  MEM="48G"
  BATCH=2
  GRAD_ACCUM=8
elif [[ "${MODEL}" == *"7B"* ]] || [[ "${MODEL}" == *"8B"* ]]; then
  MEM="64G"
  BATCH=1
  GRAD_ACCUM=16
fi

python3 -c "assert 200 % ${K} == 0, f'k={K} must evenly divide 200'" 2>/dev/null \
  || { echo "Error: k=${K} does not evenly divide 200."; exit 1; }

EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

echo "Submitting k=${K} shard training jobs (${K} parallel tasks, 1 GPU each)"
echo "  Model:      ${MODEL}"
echo "  Slug:       ${SLUG}"
echo "  Policy:     1 GPU / 1 node per task; exclude ${TOFU_EXCLUDE}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Logs:       ${LOG_DIR}/shard_<job>_<array>.log"
echo "  Batch:      ${BATCH} x grad_accum ${GRAD_ACCUM}"
echo ""

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-lora-${SLUG}-r${RANK}e${EPOCHS}-k${K}
#SBATCH --array=0-$((K - 1))%$(( K < 4 ? K : 4 ))
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/shard_%A_%a.log
#SBATCH --error=${LOG_DIR}/shard_%A_%a.log

echo "=== Job \${SLURM_JOB_ID}, shard \${SLURM_ARRAY_TASK_ID}/${K} ==="
echo "Node: \$(hostname), GPU: \$CUDA_VISIBLE_DEVICES"
date

export HF_HOME="${HF_HOME}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" \\
    --shard_id "\$SLURM_ARRAY_TASK_ID" \\
    --k "${K}" \\
    --model_name "${MODEL}" \\
    --rank ${RANK} \\
    --alpha ${ALPHA} \\
    --epochs ${EPOCHS} \\
    --lr ${LR} \\
    --batch_size ${BATCH} \\
    --grad_accum ${GRAD_ACCUM} \\
    --max_length 256 \\
    --output_dir "${OUTPUT_DIR}" \\
    --hf_home "${HF_HOME}" \\
    --seed 42

echo "Shard \${SLURM_ARRAY_TASK_ID} done."
date
EOF

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     tail -f ${LOG_DIR}/shard_<jobid>_<shard>.log"
