#!/bin/bash
# Smoke eval for Llama-3.2-1B with full merge-method manifest (17 labels).
# Skips adapters that already have results/smoke/<label>.json.
set -euo pipefail

K="${1:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
OUT_DIR="${SCRIPT_DIR}/checkpoints/Llama-3.2-1B-Instruct"

for i in $(seq 0 $((K - 1))); do
  if [ ! -f "${OUT_DIR}/shard_${i}/adapter_config.json" ]; then
    echo "Missing ${OUT_DIR}/shard_${i}/adapter_config.json"
    exit 1
  fi
done

echo "=== Llama merge smoke eval (17 labels, <1h/task, ${TOFU_ALLOWED_NODES} only) ==="
echo "  Model:  ${MODEL}"
echo "  Output: ${OUT_DIR}"
echo "  Wall:   ${TOFU_SMOKE_TIME} per task (exclude ${TOFU_EXCLUDE})"

"${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" --smoke \
  --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k "${K}"

bash "${SCRIPT_DIR}/submit_eval_smoke.sh" "${OUT_DIR}" "${MODEL}" "${K}"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  ${OUT_DIR}/results/smoke/*.json"
echo "Collect:  python collect_results.py --root ${SCRIPT_DIR}/checkpoints --smoke"
echo "Report:   python reports/generate_smoke_report.py --full --model-slug Llama-3.2-1B-Instruct"
