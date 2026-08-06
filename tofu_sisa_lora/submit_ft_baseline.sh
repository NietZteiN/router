#!/bin/bash
# Train a k=1 full-data LoRA baseline (one adapter, all 200 TOFU authors) for all models.
# Usage: bash submit_ft_baseline.sh [rank] [epochs]
# Output: checkpoints/{slug}_ft/shard_0/

set -euo pipefail

RANK="${1:-8}"
EPOCHS="${2:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
CKPT_ROOT="${SCRIPT_DIR}/checkpoints"

MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "microsoft/phi-2"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-2-7B-chat-hf"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

echo "=== Full fine-tune baseline (k=1, rank=${RANK}, epochs=${EPOCHS}, exclude ${TOFU_EXCLUDE}) ==="
for MODEL in "${MODELS[@]}"; do
  SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
  OUT_DIR="${CKPT_ROOT}/${SLUG}_ft"
  echo ""
  echo "--- ${MODEL} -> ${SLUG}_ft ---"
  if [ -f "${OUT_DIR}/shard_0/adapter_config.json" ]; then
    echo "  shard_0 exists — skip"
    continue
  fi
  bash "${SCRIPT_DIR}/submit_overnight.sh" 1 "${MODEL}" "${OUT_DIR}" "${RANK}" "${EPOCHS}"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Next:    bash submit_ft_eval_smoke.sh  (after all shard_0 adapters exist)"
