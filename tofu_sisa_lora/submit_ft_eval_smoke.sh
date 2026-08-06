#!/bin/bash
# Smoke eval for k=1 full-data LoRA baselines: shard_0_only + base model.
# Usage: bash submit_ft_eval_smoke.sh
# Prereq: bash submit_ft_baseline.sh (shard_0 must exist before running this)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
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
  SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
  OUT_DIR="${CKPT_ROOT}/${SLUG}_ft"

  if [ ! -f "${OUT_DIR}/shard_0/adapter_config.json" ]; then
    echo "Skip ${MODEL}: shard_0 not found in ${OUT_DIR}"
    continue
  fi

  echo ""
  echo "=== ${MODEL} (${SLUG}_ft) ==="

  # Compute base_logprobs.npy synchronously (needs GPU on the submitting node)
  "${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" --smoke \
    --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k 10 --forget_shard_id 9

  RESULTS_DIR="${OUT_DIR}/results/smoke"
  LOG_DIR="${OUT_DIR}/logs"
  mkdir -p "${LOG_DIR}"
  EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

  # Submit shard_0_only eval (full fine-tune utility)
  OUT_FT_JSON="${RESULTS_DIR}/shard_0_only.json"
  if [ -f "${OUT_FT_JSON}" ]; then
    echo "  Skip shard_0_only: already exists"
  else
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-ft-eval-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/ft_eval_%j.log
#SBATCH --error=${LOG_DIR}/ft_eval_%j.log

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
  --model_name "${MODEL}" \\
  --output_dir "${OUT_DIR}" \\
  --label shard_0_only \\
  --k 10 \\
  --forget_shard_id 9 \\
  --out "${OUT_FT_JSON}" \\
  --hf_home "${HF_HOME}" \\
  --smoke
EOF
  fi

  # Submit base model eval (no LoRA — reference point)
  bash "${SCRIPT_DIR}/submit_baseline_smoke.sh" "${OUT_DIR}" "${MODEL}" 10
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Collect: python collect_results.py --root ${CKPT_ROOT} --smoke"
