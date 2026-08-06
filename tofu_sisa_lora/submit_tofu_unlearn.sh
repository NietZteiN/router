#!/bin/bash
# Submit TOFU unlearning baseline training jobs (GA / GD / KL / IDK).
#
# Starts from existing k=1 ft checkpoints (checkpoints/{slug}_ft/shard_0/) and trains
# one unlearning adapter per (model, method) pair. Output goes to:
#   checkpoints/{slug}_ft_unlearn_{method}/shard_0/
# which is compatible with the existing eval pipeline (submit_ft_eval_smoke.sh).
#
# Usage:
#   bash submit_tofu_unlearn.sh                    # all 6 models × 4 methods
#   bash submit_tofu_unlearn.sh ga                 # only GA, all models
#   bash submit_tofu_unlearn.sh ga "phi-2"         # only GA on phi-2
#
# Prereq: bash submit_ft_baseline.sh (shard_0 must exist for each model)

set -euo pipefail

METHOD_FILTER="${1:-}"   # empty = all methods
MODEL_FILTER="${2:-}"    # empty = all models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT_ROOT="${SCRIPT_DIR}/checkpoints"

METHODS=("ga" "gd" "kl" "idk")

MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "microsoft/phi-2"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-2-7B-chat-hf"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

echo "=== TOFU Unlearn training (method_filter='${METHOD_FILTER}', model_filter='${MODEL_FILTER}') ==="

for MODEL in "${MODELS[@]}"; do
  # Apply optional model filter
  if [[ -n "${MODEL_FILTER}" ]] && [[ "${MODEL}" != *"${MODEL_FILTER}"* ]]; then
    continue
  fi

  SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
  FT_DIR="${CKPT_ROOT}/${SLUG}_ft"

  if [ ! -f "${FT_DIR}/shard_0/adapter_config.json" ]; then
    echo "  SKIP ${MODEL}: ft checkpoint not found at ${FT_DIR}/shard_0/"
    continue
  fi

  # Per-model SLURM settings (match submit_overnight.sh)
  MEM="32G"
  BATCH=4
  GRAD_ACCUM=4
  if [[ "${MODEL}" == *"phi-2"* ]] || [[ "${MODEL}" == *"Phi"* ]]; then
    MEM="48G"
    BATCH=2
    GRAD_ACCUM=8
  elif [[ "${MODEL}" == *"7B"* ]] || [[ "${MODEL}" == *"8B"* ]]; then
    MEM="64G"
    BATCH=1
    GRAD_ACCUM=16
  fi

  for METHOD in "${METHODS[@]}"; do
    # Apply optional method filter
    if [[ -n "${METHOD_FILTER}" ]] && [[ "${METHOD}" != "${METHOD_FILTER}" ]]; then
      continue
    fi

    OUT_DIR="${CKPT_ROOT}/${SLUG}_ft_unlearn_${METHOD}"
    LOG_DIR="${OUT_DIR}/logs"

    if [ -f "${OUT_DIR}/shard_0/adapter_config.json" ]; then
      echo "  SKIP ${SLUG} method=${METHOD}: already exists at ${OUT_DIR}/shard_0/"
      continue
    fi

    echo ""
    echo "--- ${SLUG}  method=${METHOD} ---"
    echo "    ft_dir:  ${FT_DIR}"
    echo "    out_dir: ${OUT_DIR}"
    echo "    mem:     ${MEM}  batch: ${BATCH}  grad_accum: ${GRAD_ACCUM}"
    mkdir -p "${LOG_DIR}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-unlearn-${SLUG}-${METHOD}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/unlearn_%j.log
#SBATCH --error=${LOG_DIR}/unlearn_%j.log

echo "=== TOFU Unlearn Job \${SLURM_JOB_ID} ==="
echo "    Model:  ${MODEL}"
echo "    Method: ${METHOD}"
echo "    Node:   \$(hostname)  GPU: \$CUDA_VISIBLE_DEVICES"
date

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/train_tofu_unlearn.py" \\
    --model_name "${MODEL}" \\
    --ft_dir "${FT_DIR}" \\
    --method "${METHOD}" \\
    --output_dir "${OUT_DIR}" \\
    --seed 42 \\
    --epochs 5 \\
    --lr 1e-5 \\
    --batch_size ${BATCH} \\
    --grad_accum ${GRAD_ACCUM} \\
    --max_length 256 \\
    --kl_weight 1.0 \\
    --hf_home "${HF_HOME}"

echo "Done." && date
EOF

  done  # methods
done  # models

echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "Once jobs complete, evaluate with:"
echo "  for METHOD in ga gd kl idk; do"
echo "    bash submit_ft_eval_smoke.sh checkpoints/{slug}_ft_unlearn_\${METHOD} {model} 1"
echo "  done"
echo "  python collect_results.py --root checkpoints --smoke"
