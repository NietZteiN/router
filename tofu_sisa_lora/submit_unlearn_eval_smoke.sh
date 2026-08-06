#!/bin/bash
# Smoke eval for TOFU unlearning baselines (shard_0_only + base model only).
# Runs prepare_eval.py synchronously, then submits one eval job per checkpoint.
#
# Usage: bash submit_unlearn_eval_smoke.sh [method_filter] [model_filter]
#   e.g. bash submit_unlearn_eval_smoke.sh ga "Llama-3.2-1B"

set -euo pipefail

METHOD_FILTER="${1:-}"
MODEL_FILTER="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT_ROOT="${SCRIPT_DIR}/checkpoints"

METHODS=("ga" "gd" "kl" "idk")

MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "microsoft/phi-2"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-2-7B-chat-hf"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

echo "=== Unlearn smoke eval (method_filter='${METHOD_FILTER}', model_filter='${MODEL_FILTER}') ==="

for METHOD in "${METHODS[@]}"; do
  if [[ -n "${METHOD_FILTER}" ]] && [[ "${METHOD}" != "${METHOD_FILTER}" ]]; then continue; fi

  for MODEL in "${MODELS[@]}"; do
    if [[ -n "${MODEL_FILTER}" ]] && [[ "${MODEL}" != *"${MODEL_FILTER}"* ]]; then continue; fi

    SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
    OUT_DIR="${CKPT_ROOT}/${SLUG}_ft_unlearn_${METHOD}"
    RESULTS_DIR="${OUT_DIR}/results/smoke"
    LOG_DIR="${OUT_DIR}/logs"

    if [ ! -f "${OUT_DIR}/shard_0/adapter_config.json" ]; then
      echo "  SKIP ${SLUG} ${METHOD}: unlearn checkpoint not found"
      continue
    fi

    echo ""
    echo "--- ${SLUG}  method=${METHOD} ---"
    mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

    # Prepare base logprobs + KS indices (synchronous, fast — uses cached model weights)
    if [ ! -f "${RESULTS_DIR}/base_logprobs.npy" ]; then
      echo "  Running prepare_eval.py..."
      "${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" --smoke \
        --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k 10 --forget_shard_id 9
    else
      echo "  base_logprobs.npy exists, skipping prepare_eval"
    fi

    MEM="32G"
    if [[ "${MODEL}" == *"phi-2"* ]]; then MEM="48G"; fi
    if [[ "${MODEL}" == *"7B"* ]] || [[ "${MODEL}" == *"8B"* ]]; then MEM="64G"; fi

    # Submit shard_0_only eval (the unlearned adapter)
    OUT_JSON="${RESULTS_DIR}/shard_0_only.json"
    if [ -f "${OUT_JSON}" ]; then
      echo "  Skip shard_0_only: result already exists"
    else
      # Write to a temp file — avoids the $() heredoc line-joining bug with \\+newline
      TMPSCRIPT=$(mktemp /tmp/slurm_unlearn_eval_XXXXXX.sh)
      cat > "${TMPSCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-unl-eval-${SLUG}-${METHOD}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/eval_${METHOD}_%j.log
#SBATCH --error=${LOG_DIR}/eval_${METHOD}_%j.log

echo "=== Unlearn Eval Job \${SLURM_JOB_ID}: ${SLUG} method=${METHOD} ==="
echo "Node: \$(hostname)  GPU: \$CUDA_VISIBLE_DEVICES"
date

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \
  --model_name "${MODEL}" \
  --output_dir "${OUT_DIR}" \
  --label shard_0_only \
  --k 10 \
  --forget_shard_id 9 \
  --out "${OUT_JSON}" \
  --hf_home "${HF_HOME}" \
  --smoke

echo "Done." && date
EOF
      JOB_ID=$(sbatch --parsable "${TMPSCRIPT}")
      rm -f "${TMPSCRIPT}"
      echo "  Submitted shard_0_only eval: job ${JOB_ID}"
    fi

    # Submit base model eval (reference point — same for all methods of same model)
    BASE_OUT_JSON="${RESULTS_DIR}/base_model.json"
    if [ -f "${BASE_OUT_JSON}" ]; then
      echo "  Skip base_model: result already exists"
    else
      TMPSCRIPT=$(mktemp /tmp/slurm_unlearn_base_XXXXXX.sh)
      cat > "${TMPSCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-unl-base-${SLUG}-${METHOD}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/base_${METHOD}_%j.log
#SBATCH --error=${LOG_DIR}/base_${METHOD}_%j.log

echo "=== Base Model Eval Job \${SLURM_JOB_ID}: ${SLUG} method=${METHOD} ==="
date

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_baseline.py" \
  --model_name "${MODEL}" \
  --output_dir "${OUT_DIR}" \
  --k 10 \
  --forget_shard_id 9 \
  --out "${BASE_OUT_JSON}" \
  --hf_home "${HF_HOME}" \
  --smoke

echo "Done." && date
EOF
      JOB_ID=$(sbatch --parsable "${TMPSCRIPT}")
      rm -f "${TMPSCRIPT}"
      echo "  Submitted base_model eval:   job ${JOB_ID}"
    fi

  done
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Collect: python collect_results.py --root ${CKPT_ROOT} --smoke"
