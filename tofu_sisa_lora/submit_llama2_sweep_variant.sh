#!/bin/bash
# Stage 2 of plan have-we-tried-finetuning-snuggly-dream: train one k=1 full-data LoRA sweep
# variant for meta-llama/Llama-2-7B-chat-hf (all 200 authors, shard_0, seed 42).
# Effective batch 32 (bs1 x ga32, TOFU-paper-style), max_len 256; lr/epochs/rank/alpha vary.
# Usage: bash submit_llama2_sweep_variant.sh <lr> <epochs> <rank> <alpha> <dir_suffix>
#   e.g. bash submit_llama2_sweep_variant.sh 1e-4 5 16 32 lr1e4_e5_r16   -> checkpoints/Llama-2-7B-chat-hf_ft_lr1e4_e5_r16
#        bash submit_llama2_sweep_variant.sh 1e-4 1 16 32 SMOKE_TRAIN    (1-epoch pipeline check)
# (submit_overnight.sh is not reused: it hardcodes lr=2e-4 and ga16.)

set -euo pipefail

LR="${1:?lr required}"
EPOCHS="${2:?epochs required}"
RANK="${3:?rank required}"
ALPHA="${4:?alpha required}"
SUFFIX="${5:?dir suffix required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

MODEL="meta-llama/Llama-2-7B-chat-hf"
SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
OUT_DIR="${SCRIPT_DIR}/checkpoints/${SLUG}_ft_${SUFFIX}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

if [ -f "${OUT_DIR}/shard_0/adapter_config.json" ]; then
  echo "shard_0 already exists in ${OUT_DIR} — skip"
  exit 0
fi

JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-sw-${SUFFIX}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/train_%j.log
#SBATCH --error=${LOG_DIR}/train_%j.log

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
echo "=== node \$(hostname) gpu \${CUDA_VISIBLE_DEVICES:-?} \$(date) ==="

# NOTE: single line on purpose — backslash continuations inside \$(sbatch <<EOF) get an extra
# backslash-newline pass and turn into literal space args (argparse "unrecognized arguments").
${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --shard_id 0 --k 1 --model_name "${MODEL}" --rank ${RANK} --alpha ${ALPHA} --epochs ${EPOCHS} --lr ${LR} --batch_size 1 --grad_accum 32 --max_length 256 --output_dir "${OUT_DIR}" --hf_home "${HF_HOME}" --seed 42

echo "=== done \$(date) ==="
EOF
)
echo "variant ${SUFFIX}: lr=${LR} epochs=${EPOCHS} rank=${RANK} alpha=${ALPHA} -> job ${JOB} (${OUT_DIR})"
