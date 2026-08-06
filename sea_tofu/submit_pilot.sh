#!/bin/bash
# SEA-on-TOFU pilot: 1 GPU job that trains 5 author proxies @ rank 16 and runs the full
# mini-eval (personalization / isolation / forget-quality / deletion gate). NEVER run on the
# login node — training is SLURM-only (CLAUDE.md §1). STUB=1 prints without submitting.


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh
source "${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
PYTHON="${TOFU_PYTHON:-python3}"
LOG_DIR="${SCRIPT_DIR}/proxies/_logs"
mkdir -p "${LOG_DIR}"

SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=sea-pilot
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/pilot_%j.log
#SBATCH --error=${LOG_DIR}/pilot_%j.log

set -euo pipefail
echo "=== pilot job \${SLURM_JOB_ID} on \$(hostname) GPU \$CUDA_VISIBLE_DEVICES ==="; date
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

cd "${SCRIPT_DIR}"
${PYTHON} "${SCRIPT_DIR}/run_pilot.py"
echo "pilot done."; date
EOF
)

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB (not submitting) -----"; echo "${SBATCH_SCRIPT}"
else
  echo "${SBATCH_SCRIPT}" | sbatch
  echo "Submitted. Logs: ${LOG_DIR}/pilot_<jobid>.log"
fi
