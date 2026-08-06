#!/bin/bash
# RAMoLE-on-TOFU driver: embedding retrieval (RAG) + learned RouterLoRA on the existing LegoNet-TOFU
# author-expert pool, scored with eval_tofu's model_utility/forget_quality (full + forget10).
#   bash submit_ramole_tofu.sh <config.json> [all|index|router|eval]
# Reuses the trained legonet_n32_k3 pool + forget10 deletion + retain_tr_scores KS reference — NO
# expert training. Chain (≤4 GPU): index ∥ router → smoke eval array (%4) → extended eval array (%4).
# Arms: ramole_{full,unlearn} (embed+router) and routerkey_{full,unlearn} (key+router); compare to the
# on-disk legonet_{full,unlearn} (key+1/k). STUB=1 prints sbatch scripts without submitting.
set -euo pipefail
CONFIG="${1:?config json required}"
PHASE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CAP="${RAMOLE_TOFU_CAP:-4}"
TAG="${RAMOLE_TOFU_TAG:-forget10}"
MEM="${RAMOLE_TOFU_MEM:-48G}"
TIME_TRAIN="${RAMOLE_TOFU_TIME_TRAIN:-04:00:00}"
TIME_EVAL="${RAMOLE_TOFU_TIME_EVAL:-06:00:00}"

read -r N MODEL OUTPUT_DIR < <("${PYTHON}" -c "import json; c=json.load(open('${CONFIG}')); print(c['n'], c['base_model'], c['output_dir'])")
ROUTER="${OUTPUT_DIR}/legonet/ramole/router.safetensors"
LOG_DIR="${OUTPUT_DIR}/logs"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
mkdir -p "${LOG_DIR}"
echo "RAMoLE-TOFU: config=${CONFIG} model=${MODEL} n=${N} out=${OUTPUT_DIR} cap=%${CAP} tag=${TAG}"

run_sbatch() { if [ "${STUB:-0}" = "1" ]; then cat >&2; echo "----(STUB)----" >&2; echo "STUB"; else sbatch --parsable; fi; }
dep() { [ -n "${1:-}" ] && [ "${1}" != "STUB" ] && echo "#SBATCH --dependency=afterok:${1}" || echo ""; }

# array task specs: "label|route|tag"  (empty tag = full model)
SPECS_LITERAL='"ramole_full|embed|" "ramole_unlearn|embed|forget10" "routerkey_full|key|" "routerkey_unlearn|key|forget10"'
NSPECS=4
# retriever-FT follow-up: re-run only the embed arm with the fine-tuned encoder
RAMOLEFT_SPECS='"ramoleft_full|embed|" "ramoleft_unlearn|embed|forget10"'
NRAMOLEFT=2

IDX_ID=""; ROU_ID=""; SMOKE_ID=""
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "index" ]; then
IDX_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-tofu-index
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/ramole_index_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PYTHON} "${SCRIPT_DIR}/ramole_tofu.py" --config "${CONFIG}" --device cuda --build_index
EOF
)
echo "  index job: ${IDX_ID}"
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "router" ]; then
ROU_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-tofu-router
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_TRAIN}
#SBATCH --output=${LOG_DIR}/ramole_router_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PYTHON} "${SCRIPT_DIR}/train_router_tofu.py" --config "${CONFIG}" --device cuda
EOF
)
echo "  router job: ${ROU_ID}"
fi

submit_eval() {  # $1=depth flag  $2=results subdir  $3=dependency id  $4=specs literal  $5=ntasks
  local DEPTH="$1" SUB="$2" DEPID="$3" SPECS_LIT="$4" NT="$5" RESULTS_DIR="${OUTPUT_DIR}/results/$2"
  run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-tofu-eval-${SUB}
#SBATCH --array=0-$((NT - 1))%${CAP}
#SBATCH --partition=all
${EXCLUDE_LINE}
$(dep "${DEPID}")
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_EVAL}
#SBATCH --output=${LOG_DIR}/ramole_eval_${SUB}_%A_%a.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"; mkdir -p "\${TOFU_METRICS_CACHE}"
SPECS=(${SPECS_LIT})
IFS='|' read LABEL ROUTE ETAG <<< "\${SPECS[\${SLURM_ARRAY_TASK_ID}]}"
TAGFLAG=""; [ -n "\${ETAG}" ] && TAGFLAG="--legonet_unlearn_tag \${ETAG}"
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${OUTPUT_DIR}" --legonet_config "${CONFIG}" --ramole_router "${ROUTER}" --ramole_route \${ROUTE} --label \${LABEL} \${TAGFLAG} --k 10 --forget_shard_id 9 --out "${RESULTS_DIR}/\${LABEL}.json" --hf_home "${HF_HOME}" ${DEPTH}
EOF
}

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "eval" ]; then
  # eval depends on BOTH index and router (full + unlearn, embed + key)
  if [ "${STUB:-0}" = "1" ]; then EDEP=""; else EDEP="${IDX_ID}:${ROU_ID}"; fi
  SMOKE_ID=$(submit_eval --smoke smoke "${EDEP}" "${SPECS_LITERAL}" "${NSPECS}"); echo "  smoke eval job: ${SMOKE_ID}"
  EXT_ID=$(submit_eval --extended extended "${SMOKE_ID}" "${SPECS_LITERAL}" "${NSPECS}"); echo "  extended eval job (after smoke): ${EXT_ID}"
fi

# ── retriever-FT follow-up: FT encoder → rebuild index → re-eval embed arm (ramoleft_*) ──
# Reuses the already-trained router.safetensors; only the retriever/index change.
if [ "${PHASE}" = "retriever" ]; then
FT_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-tofu-retriever
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_TRAIN}
#SBATCH --output=${LOG_DIR}/ramole_retriever_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PYTHON} "${SCRIPT_DIR}/train_retriever_tofu.py" --config "${CONFIG}" --device cuda
EOF
)
echo "  retriever FT job: ${FT_ID}"
IDX2_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-tofu-index-ft
#SBATCH --partition=all
${EXCLUDE_LINE}
$(dep "${FT_ID}")
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/ramole_index_ft_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PYTHON} "${SCRIPT_DIR}/ramole_tofu.py" --config "${CONFIG}" --device cuda --build_index
EOF
)
echo "  index-rebuild (FT) job: ${IDX2_ID}"
SMOKE_ID=$(submit_eval --smoke smoke "${IDX2_ID}" "${RAMOLEFT_SPECS}" "${NRAMOLEFT}"); echo "  ramoleft smoke eval: ${SMOKE_ID}"
EXT_ID=$(submit_eval --extended extended "${SMOKE_ID}" "${RAMOLEFT_SPECS}" "${NRAMOLEFT}"); echo "  ramoleft extended eval (after smoke): ${EXT_ID}"
fi
echo "Monitor: squeue -u \$USER | grep ramole-tofu"
