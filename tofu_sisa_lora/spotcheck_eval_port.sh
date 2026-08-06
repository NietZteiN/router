#!/bin/bash
# Spot-check the ported (open-unlearning-faithful) eval on a real SISA shard set.
# One self-contained GPU job: train the retain90 oracle -> prepare_eval (smoke) ->
# eval 3 labels (merged contains forget shard, remerge excludes it, shard_9_only IS forget).
# Outputs go to a scratch xval dir so existing results/ JSONs are NOT clobbered; the only
# new file added under the checkpoint is results/smoke/retain_tr_scores.npy.
#
# Usage: sbatch spotcheck_eval_port.sh
#SBATCH --job-name=tofu-spotcheck-port
#SBATCH --partition=all
#SBATCH --exclude=sprint4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${TOFU_STORAGE_ROOT}/xval/spotcheck_llama1b/spotcheck_%j.log
#SBATCH --error=${TOFU_STORAGE_ROOT}/xval/spotcheck_llama1b/spotcheck_%j.log


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail

SCRIPT_DIR="${REPO_ROOT}/tofu_sisa_lora"
PYTHON="${TOFU_PYTHON:-python3}"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
CKPT="${TOFU_CKPT_ROOT}/Llama-3.2-1B-Instruct"
XVAL="${TOFU_STORAGE_ROOT}/xval/spotcheck_llama1b"
K=10
mkdir -p "${XVAL}"

export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

echo "=== node $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} $(date) ==="

echo "--- [1/3] train retain90 oracle (authors 0-179) ---"
"${PYTHON}" "${SCRIPT_DIR}/train_lora_shard.py" \
  --retain90 --k "${K}" --model_name "${MODEL}" --output_dir "${CKPT}" \
  --rank 8 --alpha 16 --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4 \
  --max_length 256 --hf_home "${HF_HOME}" --seed 42

echo "--- [2/3] prepare_eval (smoke): cache retain_tr_scores.npy from retain90 ---"
"${PYTHON}" "${SCRIPT_DIR}/prepare_eval.py" \
  --smoke --output_dir "${CKPT}" --model_name "${MODEL}" --k "${K}"

echo "--- [3/3] eval 3 labels (smoke) -> ${XVAL} ---"
for LABEL in remerge_dare_ties merged_dare_ties shard_9_only; do
  echo ">>> ${LABEL}"
  "${PYTHON}" "${SCRIPT_DIR}/eval_tofu.py" \
    --model_name "${MODEL}" --output_dir "${CKPT}" \
    --label "${LABEL}" --k "${K}" --forget_shard_id 9 \
    --out "${XVAL}/${LABEL}.json" --smoke
done

echo "=== done $(date) ==="
echo "Results:"
for LABEL in remerge_dare_ties merged_dare_ties shard_9_only; do
  echo "--- ${LABEL} ---"; cat "${XVAL}/${LABEL}.json"; echo
done
