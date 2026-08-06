#!/bin/bash
# Directional cross-check: run the ported eval on locuslab/tofu_ft_llama2-7b (7B full FT).
# Expect model_utility to move down from the old ~0.70 toward open-unlearning's ~0.62.
# Usage: sbatch verify_llama2_full.sh
#SBATCH --job-name=tofu-verify-llama2
#SBATCH --partition=all
#SBATCH --exclude=sprint4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${TOFU_STORAGE_ROOT}/xval/verify_llama2_7b/verify_%j.log
#SBATCH --error=${TOFU_STORAGE_ROOT}/xval/verify_llama2_7b/verify_%j.log


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail
SCRIPT_DIR="${REPO_ROOT}/tofu_sisa_lora"
PYTHON="${TOFU_PYTHON:-python3}"
export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
echo "=== node $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} $(date) ==="
"${PYTHON}" "${SCRIPT_DIR}/verify_eval_tofu.py"
echo "=== done $(date) ==="
