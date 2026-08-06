#!/bin/bash
# Train LoRA shards for all configured base models (parallel SLURM array per model).
# Usage: bash submit_all_models.sh [k]

set -euo pipefail

K="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"

MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "microsoft/phi-2"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-2-7B-chat-hf"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

PYTHON="${TOFU_PYTHON:-python3}"

echo "=== Multi-model training (k=${K}, exclude ${TOFU_EXCLUDE}) ==="
for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "--- ${MODEL} ---"
  OUT_DIR="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import checkpoints_dir; print(checkpoints_dir('${SCRIPT_DIR}/checkpoints', '${MODEL}'))")"
  missing=0
  for i in $(seq 0 $((K - 1))); do
    [ -f "${OUT_DIR}/shard_${i}/adapter_config.json" ] || missing=$((missing + 1))
  done
  if [ "${missing}" -eq 0 ]; then
    echo "  All ${K} shards exist in ${OUT_DIR} — skip submit"
    continue
  fi
  echo "  Missing ${missing}/${K} shards — submitting..."
  bash "${SCRIPT_DIR}/submit_overnight.sh" "${K}" "${MODEL}"
done

echo ""
echo "Done. Monitor: squeue -u \$USER"
