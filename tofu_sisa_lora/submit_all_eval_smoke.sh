#!/bin/bash
# Smoke eval for all models with trained shards (~1h each, sequential if GPUs full).
set -euo pipefail

K="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

for MODEL in "${MODELS[@]}"; do
  OUT_DIR="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import checkpoints_dir; print(checkpoints_dir('${CKPT_ROOT}', '${MODEL}'))")"
  ready=true
  for i in $(seq 0 $((K - 1))); do
    [ -f "${OUT_DIR}/shard_${i}/adapter_config.json" ] || ready=false
  done
  if [ "${ready}" = false ]; then
    echo "Skip ${MODEL}: missing shards"
    continue
  fi
  echo ""
  echo "=== ${MODEL} ==="
  "${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" --smoke \
    --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k "${K}"
  bash "${SCRIPT_DIR}/submit_eval_smoke.sh" "${OUT_DIR}" "${MODEL}" "${K}"
  bash "${SCRIPT_DIR}/submit_baseline_smoke.sh" "${OUT_DIR}" "${MODEL}" "${K}"
done

echo ""
echo "Results: checkpoints/<model>/results/smoke/*.json"
echo "Collect: python collect_results.py --root ${CKPT_ROOT} --smoke"
