#!/bin/bash
# Standard TOFU unlearning report for SEA (one GPU job, base loaded once).
# Usage: bash submit_report.sh [rank] [max_new] [n_retain]
# STUB=1 prints without submitting.


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail
RANK="${1:-16}"
MAX_NEW="${2:-100}"
N_RETAIN="${3:-40}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh
source "${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
PYTHON="${TOFU_PYTHON:-python3}"
LOG_DIR="${SCRIPT_DIR}/proxies/_logs"
mkdir -p "${LOG_DIR}"

SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=sea-report-r${RANK}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=${LOG_DIR}/report_r${RANK}_%j.log
#SBATCH --error=${LOG_DIR}/report_r${RANK}_%j.log

set -euo pipefail
echo "=== report job \${SLURM_JOB_ID} rank ${RANK} on \$(hostname) GPU \$CUDA_VISIBLE_DEVICES ==="; date
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
cd "${SCRIPT_DIR}"
${PYTHON} "${SCRIPT_DIR}/eval_unlearning_report.py" --rank "${RANK}" --max_new "${MAX_NEW}" --n_retain "${N_RETAIN}" --hf_home "${HF_HOME}"
echo "report done."; date
EOF
)

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB -----"; echo "${SBATCH_SCRIPT}"
else
  echo "${SBATCH_SCRIPT}" | sbatch
  echo "Submitted. Log: ${LOG_DIR}/report_r${RANK}_<jobid>.log"
fi
