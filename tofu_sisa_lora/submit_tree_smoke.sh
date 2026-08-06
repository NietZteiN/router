#!/bin/bash
# Submit smoke evals for the new tree_root_* and tree_remerge_* labels on all k4 models.
# For Llama-3.2-3B (no base_logprobs yet) also runs prepare_eval.py --smoke first.
# Usage: bash submit_tree_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
K=4

declare -A MODELS
MODELS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]="TinyLlama-1.1B-Chat-v1.0_k4"
MODELS["meta-llama/Llama-3.2-1B-Instruct"]="Llama-3.2-1B-Instruct_k4"
MODELS["meta-llama/Llama-3.2-3B-Instruct"]="Llama-3.2-3B-Instruct_k4"
MODELS["microsoft/phi-2"]="phi-2_k4"

TREE_LABELS=(
  tree_root_linear
  tree_root_dare_linear
  tree_root_ties
  tree_root_dare_ties
  tree_root_magnitude_prune
  tree_root_cat
  tree_remerge_linear
  tree_remerge_dare_linear
  tree_remerge_ties
  tree_remerge_dare_ties
  tree_remerge_magnitude_prune
  tree_remerge_cat
)

for MODEL in "${!MODELS[@]}"; do
  DIR="${SCRIPT_DIR}/checkpoints/${MODELS[$MODEL]}"
  RESULTS_DIR="${DIR}/results/smoke"
  LOG_DIR="${DIR}/logs"
  mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

  # Run prepare_eval if base_logprobs is missing (Llama-3.2-3B case).
  if [ ! -f "${RESULTS_DIR}/base_logprobs.npy" ]; then
    echo "=== ${MODEL}: running prepare_eval.py --smoke ==="
    sbatch --wait \
      --job-name="tofu-prepare-$(basename ${DIR})" \
      --partition=all \
      --exclude="${TOFU_EXCLUDE}" \
      --gres=gpu:1 \
      --mem=48G \
      --cpus-per-task=4 \
      --time=00:30:00 \
      --output="${LOG_DIR}/prepare_%j.log" \
      --wrap="${PYTHON} ${SCRIPT_DIR}/prepare_eval.py --smoke \
        --model_name ${MODEL} \
        --output_dir ${DIR} \
        --k ${K} \
        --hf_home ${HF_HOME}"
    echo "  prepare done"
  fi

  # Write tree-only manifest.
  TREE_MANIFEST="${RESULTS_DIR}/eval_manifest_tree_smoke.txt"
  # Only include labels that don't yet have a result file.
  > "${TREE_MANIFEST}"
  for LABEL in "${TREE_LABELS[@]}"; do
    if [ ! -f "${RESULTS_DIR}/${LABEL}.json" ]; then
      echo "${LABEL}" >> "${TREE_MANIFEST}"
    fi
  done

  N_TASKS=$(wc -l < "${TREE_MANIFEST}")
  if [ "${N_TASKS}" -eq 0 ]; then
    echo "=== ${MODEL}: all tree results already exist, skipping ==="
    continue
  fi

  SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
  FORGET_ID=$((K - 1))

  echo "=== ${MODEL}: submitting ${N_TASKS} tree eval tasks ==="
  cat "${TREE_MANIFEST}"

  sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=tofu-tree-smoke-${SLUG}
#SBATCH --array=0-$((N_TASKS - 1))%${TOFU_ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/tree_%A_%a.log
#SBATCH --error=${LOG_DIR}/tree_%A_%a.log

LABEL=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TREE_MANIFEST}")
OUT_JSON="${RESULTS_DIR}/\${LABEL}.json"

if [ -f "\${OUT_JSON}" ]; then
  echo "Skip existing \${OUT_JSON}"
  exit 0
fi

echo "=== Tree eval \${SLURM_JOB_ID} task \${SLURM_ARRAY_TASK_ID}: \${LABEL} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < '${HF_HOME}/token')"
fi
export HUGGING_FACE_HUB_TOKEN="\${HF_TOKEN:-}"

${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \
  --model_name "${MODEL}" \
  --output_dir "${DIR}" \
  --label "\${LABEL}" \
  --k ${K} \
  --forget_shard_id ${FORGET_ID} \
  --out "\${OUT_JSON}" \
  --hf_home "${HF_HOME}" \
  --smoke

date
SBATCH_EOF

done

echo ""
echo "Monitor: watch -n 30 'squeue -u \$USER'"
echo "Collect: python collect_results.py --root ${SCRIPT_DIR}/checkpoints --smoke"
