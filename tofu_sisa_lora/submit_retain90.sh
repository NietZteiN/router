#!/bin/bash
# Train the retain90 forget-quality oracle (authors 0-179) for each main base model.
# One SLURM array task per model; skips models that already have retain90/.
# Usage: sbatch submit_retain90.sh
#SBATCH --job-name=tofu-retain90
#SBATCH --partition=all
#SBATCH --exclude=sprint4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-5%4
#SBATCH --output=${TOFU_CKPT_ROOT}/retain90_logs/r90_%A_%a.log
#SBATCH --error=${TOFU_CKPT_ROOT}/retain90_logs/r90_%A_%a.log


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail
mkdir -p ${TOFU_CKPT_ROOT}/retain90_logs
SCRIPT_DIR="${REPO_ROOT}/tofu_sisa_lora"
PYTHON="${TOFU_PYTHON:-python3}"
CKPT="${TOFU_CKPT_ROOT}"
export HF_HOME=${HF_HOME}
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "microsoft/phi-2"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-2-7B-chat-hf"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="$($PYTHON -c "import sys; sys.path.insert(0,'$SCRIPT_DIR'); from model_paths import checkpoints_dir; print(checkpoints_dir('$CKPT','$MODEL'))")"

# Per-model micro-batch (effective batch 16), matching submit_overnight.sh sizing.
BATCH=4; GRAD=4
case "$MODEL" in
  *phi-2*) BATCH=2; GRAD=8;;
  *7B*|*8B*|*7b*) BATCH=1; GRAD=16;;
esac

echo "=== retain90 task $SLURM_ARRAY_TASK_ID: $MODEL -> $OUT_DIR (batch=$BATCH grad=$GRAD) node $(hostname) $(date) ==="
"$PYTHON" "$SCRIPT_DIR/train_lora_shard.py" \
  --retain90 --k 10 --model_name "$MODEL" --output_dir "$OUT_DIR" \
  --rank 8 --alpha 16 --epochs 3 --lr 2e-4 \
  --batch_size "$BATCH" --grad_accum "$GRAD" --max_length 256 \
  --hf_home "$HF_HOME" --seed 42
echo "=== done $(date) ==="
