#!/bin/bash
# Train per-author SEA proxies on SLURM (one GPU per array task, each task trains a
# contiguous block of authors so the 7B 4-bit base loads once per task, not once per author).
#
# Usage:
#   bash submit_train_proxies.sh [rank] [author_start] [author_end] [block_size]
# Examples:
#   bash submit_train_proxies.sh 16   0 199 20    # headline: all 200 authors @ rank 16 (10 tasks)
#   bash submit_train_proxies.sh 4  180 199 20    # rank-sweep point r=4 on forget10 (1 task)
#   for r in 4 8 16 32 64; do bash submit_train_proxies.sh $r 180 199 20; done   # full sweep on forget10
#
# Policy: sprint1/2/3 only (exclude sprint4), 1 GPU/task, <=12 concurrent. Proxies land under
# proxies/{slug}[ _r{rank} ]/author_NNN/ (symlinked to /storage2). STUB=1 prints without submitting.


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail

RANK="${1:-16}"
A_START="${2:-0}"
A_END="${3:-199}"
BLOCK="${4:-20}"
SEED="${5:-}"            # optional: train at a non-default seed (for seed variance)
PROXY_ROOT="${6:-}"      # optional: isolate non-default-seed proxies in their own root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh
source "${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
PYTHON="${TOFU_PYTHON:-python3}"
CONFIG="${SCRIPT_DIR}/configs/sea_tofu_llama2.json"
MODEL="$("${PYTHON}" -c "import json;print(json.load(open('${CONFIG}'))['model_name'])")"

N_AUTHORS=$((A_END - A_START + 1))
N_BLOCKS=$(( (N_AUTHORS + BLOCK - 1) / BLOCK ))
LOG_DIR="${SCRIPT_DIR}/proxies/_logs"
mkdir -p "${LOG_DIR}"

echo "Training proxies: model=${MODEL} rank=${RANK} authors=${A_START}..${A_END} block=${BLOCK} -> ${N_BLOCKS} tasks"

SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=sea-proxy-r${RANK}
#SBATCH --array=0-$((N_BLOCKS - 1))%${TOFU_ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/proxy_r${RANK}_%A_%a.log
#SBATCH --error=${LOG_DIR}/proxy_r${RANK}_%A_%a.log

set -euo pipefail
echo "=== Job \${SLURM_JOB_ID} block \${SLURM_ARRAY_TASK_ID} on \$(hostname) GPU \$CUDA_VISIBLE_DEVICES ==="
date
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

START=\$(( ${A_START} + SLURM_ARRAY_TASK_ID * ${BLOCK} ))
${PYTHON} "${SCRIPT_DIR}/train_proxy.py" --author_start "\${START}" --author_count "${BLOCK}" --rank "${RANK}" --config "${CONFIG}" --hf_home "${HF_HOME}"${SEED:+ --seed ${SEED}}${PROXY_ROOT:+ --proxy_root ${PROXY_ROOT}}
echo "block \${SLURM_ARRAY_TASK_ID} done."; date
EOF
)

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB (not submitting) -----"
  echo "${SBATCH_SCRIPT}"
else
  echo "${SBATCH_SCRIPT}" | sbatch
  echo "Submitted. Monitor: squeue -u \$USER ; logs: ${LOG_DIR}/proxy_r${RANK}_<jobid>_<block>.log"
fi
