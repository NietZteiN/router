#!/bin/bash
# Extended eval with full merge-method manifest (17 labels).
# Usage: bash submit_llama_extended_eval.sh [k] [hf_model_id]
set -euo pipefail

K="${1:-4}"
MODEL="${2:-meta-llama/Llama-3.2-1B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
OUT_DIR="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import checkpoints_dir; print(checkpoints_dir('${SCRIPT_DIR}/checkpoints', '${MODEL}'))")"
SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"

for i in $(seq 0 $((K - 1))); do
  if [ ! -f "${OUT_DIR}/shard_${i}/adapter_config.json" ]; then
    echo "Missing ${OUT_DIR}/shard_${i}/adapter_config.json"
    exit 1
  fi
done

if [[ "${MODEL}" == *"3B"* ]] || [[ "${MODEL}" == *"3.2-3B"* ]]; then
  export TOFU_EXTENDED_TIME="${TOFU_EXTENDED_TIME_3B}"
  REPORT="reports/EXTENDED_EVAL_REPORT_3B.md"
else
  REPORT="reports/EXTENDED_EVAL_REPORT.md"
fi

echo "=== Extended merge eval (${SLUG}, 17 labels, <=${TOFU_EXTENDED_TIME}/task, ${TOFU_ALLOWED_NODES} only) ==="
echo "  Model:  ${MODEL}"
echo "  Output: ${OUT_DIR}"
echo "  Exclude: ${TOFU_EXCLUDE}"

"${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" --extended \
  --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k "${K}"

bash "${SCRIPT_DIR}/submit_baseline_extended.sh" "${OUT_DIR}" "${MODEL}" "${K}"
bash "${SCRIPT_DIR}/submit_eval_extended.sh" "${OUT_DIR}" "${MODEL}" "${K}"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  ${OUT_DIR}/results/extended/*.json"
echo "Collect:  python collect_results.py --root ${SCRIPT_DIR}/checkpoints --extended"
echo "Report:   python reports/generate_smoke_report.py --full --extended --model-slug ${SLUG} --out ${REPORT} --compare-slugs Llama-3.2-1B-Instruct,${SLUG}"
