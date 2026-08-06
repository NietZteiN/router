#!/bin/bash
# Run smoke eval for all ablation configs (rank × epoch grid).
# Skips any config where shards are not fully trained yet.
# Usage: bash run_ablation_eval.sh [k]

set -euo pipefail

K="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TOFU_PYTHON:-python3}"
CKPT_ROOT="${SCRIPT_DIR}/checkpoints"

MODELS=(
  "microsoft/phi-2"
  "meta-llama/Llama-3.1-8B-Instruct"
)
RANKS=(8 16 32)
EPOCHS_LIST=(3 5 10)

echo "=== Ablation eval: rank × epoch grid (k=${K}) ==="
echo ""

for MODEL in "${MODELS[@]}"; do
  SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0,'${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
  for RANK in "${RANKS[@]}"; do
    for EPOCHS in "${EPOCHS_LIST[@]}"; do
      if [[ "${RANK}" -eq 8 && "${EPOCHS}" -eq 3 ]]; then
        continue
      fi
      OUT_DIR="${CKPT_ROOT}/${SLUG}_r${RANK}_e${EPOCHS}"

      # Check all shards exist
      ready=true
      for i in $(seq 0 $((K - 1))); do
        [ -f "${OUT_DIR}/shard_${i}/adapter_config.json" ] || ready=false
      done
      if [ "${ready}" = false ]; then
        echo "Skip ${SLUG} r=${RANK} e=${EPOCHS} — shards not ready"
        continue
      fi

      # Skip if eval already complete (manifest exists and all labels done)
      manifest="${OUT_DIR}/results/smoke/eval_manifest_smoke.txt"
      if [ -f "${manifest}" ]; then
        total=$(wc -l < "${manifest}")
        done_count=$(find "${OUT_DIR}/results/smoke" -name "*.json" \
          ! -name "*progress*" ! -name "*manifest*" 2>/dev/null | wc -l)
        if [ "${done_count}" -ge "${total}" ]; then
          echo "Skip ${SLUG} r=${RANK} e=${EPOCHS} — eval already complete (${done_count}/${total})"
          continue
        fi
      fi

      echo ""
      echo "=== ${SLUG}  r=${RANK}  e=${EPOCHS} ==="
      "${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" --smoke \
        --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k "${K}"
      bash "${SCRIPT_DIR}/submit_eval_smoke.sh" "${OUT_DIR}" "${MODEL}" "${K}"
      bash "${SCRIPT_DIR}/submit_baseline_smoke.sh" "${OUT_DIR}" "${MODEL}" "${K}"
    done
  done
done

echo ""
echo "All eval jobs submitted."
echo "Collect results when done:"
echo "  python collect_results.py --root ${CKPT_ROOT} --smoke --out ${CKPT_ROOT}/ablation_smoke.csv"
