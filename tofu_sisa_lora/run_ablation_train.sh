#!/bin/bash
# Submit rank × epoch ablation training jobs for phi-2 and Llama-3.1-8B-Instruct.
# Skips r=8/e=3 (baseline already in checkpoints/{slug}/).
# Skips any config where all k shards already exist.
# Usage: bash run_ablation_train.sh [k]

set -euo pipefail

K="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TOFU_PYTHON:-python3}"

MODELS=(
  "microsoft/phi-2"
  "meta-llama/Llama-3.1-8B-Instruct"
)
RANKS=(8 16 32)
EPOCHS_LIST=(3 5 10)

echo "=== Ablation: rank × epoch grid (k=${K}) ==="
echo "Models: ${MODELS[*]}"
echo "Ranks:  ${RANKS[*]}"
echo "Epochs: ${EPOCHS_LIST[*]}"
echo "(Skipping r=8/e=3 — baseline already in checkpoints/{slug}/)"
echo ""

for MODEL in "${MODELS[@]}"; do
  SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0,'${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
  for RANK in "${RANKS[@]}"; do
    for EPOCHS in "${EPOCHS_LIST[@]}"; do
      if [[ "${RANK}" -eq 8 && "${EPOCHS}" -eq 3 ]]; then
        continue
      fi
      OUT_DIR="${SCRIPT_DIR}/checkpoints/${SLUG}_r${RANK}_e${EPOCHS}"
      missing=0
      for i in $(seq 0 $((K - 1))); do
        [ -f "${OUT_DIR}/shard_${i}/adapter_config.json" ] || missing=$((missing + 1))
      done
      if [ "${missing}" -eq 0 ]; then
        echo "Skip ${SLUG} r=${RANK} e=${EPOCHS} — all shards exist"
        continue
      fi
      echo ""
      echo "--- ${SLUG}  r=${RANK}  e=${EPOCHS}  (missing ${missing}/${K} shards) ---"
      bash "${SCRIPT_DIR}/submit_overnight.sh" "${K}" "${MODEL}" "${OUT_DIR}" "${RANK}" "${EPOCHS}"
    done
  done
done

echo ""
echo "Done submitting. Monitor: squeue -u \$USER"
echo "Next step when training completes: bash run_ablation_eval.sh"
