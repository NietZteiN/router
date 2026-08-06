#!/bin/bash
# prepare_eval + submit_eval for all models with complete shard checkpoints.
# Usage: bash submit_all_eval.sh [k]

set -euo pipefail

K="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
CKPT_ROOT="${SCRIPT_DIR}/checkpoints"

MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "microsoft/phi-2"
  "meta-llama/Llama-3.2-1B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  OUT_DIR="$("${PYTHON}" -c "from model_paths import checkpoints_dir; print(checkpoints_dir('${CKPT_ROOT}', '${MODEL}'))")"
  ready=true
  for i in $(seq 0 $((K - 1))); do
    if [ ! -f "${OUT_DIR}/shard_${i}/adapter_config.json" ]; then
      echo "Skip eval ${MODEL}: missing shard_${i}"
      ready=false
      break
    fi
  done
  if [ "${ready}" = false ]; then
    continue
  fi
  echo ""
  echo "=== Prepare ${MODEL} ==="
  "${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" \
    --model_name "${MODEL}" \
    --output_dir "${OUT_DIR}" \
    --k "${K}"
  echo "=== Submit eval ${MODEL} ==="
  bash "${SCRIPT_DIR}/submit_eval.sh" "${OUT_DIR}" "${MODEL}" "${K}"
done

echo ""
echo "When done: python collect_results.py --root ${CKPT_ROOT}"
