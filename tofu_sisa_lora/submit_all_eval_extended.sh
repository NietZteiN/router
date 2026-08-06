#!/bin/bash
# Extended eval + baseline for all models with complete shard checkpoints.
# Usage: bash submit_all_eval_extended.sh [k]
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
  OUT_DIR="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import checkpoints_dir; print(checkpoints_dir('${CKPT_ROOT}', '${MODEL}'))")"
  ready=true
  for i in $(seq 0 $((K - 1))); do
    [ -f "${OUT_DIR}/shard_${i}/adapter_config.json" ] || { ready=false; break; }
  done
  if [ "${ready}" = false ]; then echo "Skip ${MODEL}: missing shards"; continue; fi
  echo ""
  echo "=== ${MODEL} ==="
  "${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" --extended \
    --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k "${K}"
  bash "${SCRIPT_DIR}/submit_eval_extended.sh" "${OUT_DIR}" "${MODEL}" "${K}"
  bash "${SCRIPT_DIR}/submit_baseline_extended.sh" "${OUT_DIR}" "${MODEL}" "${K}"
done

echo ""
echo "Collect: python collect_results.py --root ${CKPT_ROOT} --extended"
