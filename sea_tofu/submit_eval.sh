#!/bin/bash
# Full SEA-on-TOFU evaluation for one rank (single GPU: base loads once, proxies swap in/out).
# Usage: bash submit_eval.sh [rank] [--smoke]
#   bash submit_eval.sh 16            # headline full eval
#   bash submit_eval.sh 4 --smoke     # fast pass for a rank-sweep point
# STUB=1 prints the sbatch script without submitting.


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail

RANK="${1:-16}"
SMOKE_FLAG=""
TIME="03:00:00"
if [ "${2:-}" = "--smoke" ]; then SMOKE_FLAG="--smoke"; TIME="00:55:00"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh
source "${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
PYTHON="${TOFU_PYTHON:-python3}"
LOG_DIR="${SCRIPT_DIR}/proxies/_logs"
mkdir -p "${LOG_DIR}"

SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=sea-eval-r${RANK}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/eval_r${RANK}_%j.log
#SBATCH --error=${LOG_DIR}/eval_r${RANK}_%j.log

set -euo pipefail
echo "=== eval job \${SLURM_JOB_ID} rank ${RANK} on \$(hostname) GPU \$CUDA_VISIBLE_DEVICES ==="; date
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_sea_tofu.py" --rank "${RANK}" ${SMOKE_FLAG} --hf_home "${HF_HOME}"
echo "eval rank ${RANK} done."; date
EOF
)

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB (not submitting) -----"; echo "${SBATCH_SCRIPT}"
else
  echo "${SBATCH_SCRIPT}" | sbatch
  echo "Submitted. Logs: ${LOG_DIR}/eval_r${RANK}_<jobid>.log"
fi
