#!/bin/bash
# Combined SEA-on-TOFU rank-sweep eval: ONE GPU job, base loaded once, results written
# incrementally per rank. Usage: bash submit_sweep_eval.sh [n_forget] [max_new]
# STUB=1 prints without submitting.


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail
N_FORGET="${1:-20}"
MAX_NEW="${2:-40}"
TAG="${3:-sweep}"
N_RETAIN="${4:-20}"
RANKS="${5:-4,8,16,32,64}"
PROXY_ROOT="${6:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh
source "${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
PYTHON="${TOFU_PYTHON:-python3}"
LOG_DIR="${SCRIPT_DIR}/proxies/_logs"
mkdir -p "${LOG_DIR}"

SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=sea-eval-${TAG}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=${LOG_DIR}/eval_${TAG}_%j.log
#SBATCH --error=${LOG_DIR}/eval_${TAG}_%j.log

set -euo pipefail
echo "=== sweep eval job \${SLURM_JOB_ID} on \$(hostname) GPU \$CUDA_VISIBLE_DEVICES ==="; date
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
cd "${SCRIPT_DIR}"
${PYTHON} "${SCRIPT_DIR}/run_sweep_eval.py" --n_forget "${N_FORGET}" --max_new "${MAX_NEW}" --tag "${TAG}" --n_retain "${N_RETAIN}" --ranks "${RANKS}" --hf_home "${HF_HOME}"${PROXY_ROOT:+ --proxy_root ${PROXY_ROOT}}
echo "sweep eval (${TAG}) done."; date
EOF
)

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB -----"; echo "${SBATCH_SCRIPT}"
else
  echo "${SBATCH_SCRIPT}" | sbatch
  echo "Submitted. Log: ${LOG_DIR}/sweep_eval_<jobid>.log"
fi
