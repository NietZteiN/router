#!/bin/bash
# Stage 3 gate confirmation (plan have-we-tried-finetuning-snuggly-dream): extended eval of the
# grid winner (lr1e-4 e5 r32) + base model for meta-llama/Llama-2-7B-chat-hf.
#   P: prepare_eval --extended in the _ft dir (extended KS reference, 120 truth rows; GPU)
#   W: winner eval_tofu --extended (copies the reference in first)        [afterok P]
#   B: eval_baseline --extended                                           [afterok P]
# Usage: bash submit_llama2_extended_winner.sh
# NOTE: job commands single-line (continuations inside $(sbatch <<EOF) break into space args).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

MODEL="meta-llama/Llama-2-7B-chat-hf"
FT_DIR="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf_ft"
WIN_DIR="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf_ft_lr1e4_e5_r32"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
mkdir -p "${FT_DIR}/results/extended" "${WIN_DIR}/results/extended" "${WIN_DIR}/logs"

[ -f "${WIN_DIR}/shard_0/adapter_config.json" ] || { echo "Missing winner adapter"; exit 1; }
[ -f "${FT_DIR}/retain90/adapter_config.json" ] || { echo "Missing retain90 oracle"; exit 1; }

ENV_BLOCK=$(cat <<ENVEOF
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
echo "=== node \$(hostname) gpu \${CUDA_VISIBLE_DEVICES:-?} \$(date) ==="
ENVEOF
)

JOB_P=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-prep-ext
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${FT_DIR}/logs/prepare_extended_%j.log
#SBATCH --error=${FT_DIR}/logs/prepare_extended_%j.log
${ENV_BLOCK}
${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --extended --model_name "${MODEL}" --output_dir "${FT_DIR}" --k 10 --forget_shard_id 9 --hf_home "${HF_HOME}"
test -f "${FT_DIR}/results/extended/retain_tr_scores.npy" || { echo "FATAL: extended reference missing"; exit 1; }
echo "=== done \$(date) ==="
EOF
)
echo "Job P (prepare extended): ${JOB_P}"

JOB_W=$(sbatch --parsable --dependency=afterok:${JOB_P} <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-ext-winner
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_EXTENDED_TIME}
#SBATCH --output=${WIN_DIR}/logs/ext_eval_%j.log
#SBATCH --error=${WIN_DIR}/logs/ext_eval_%j.log
${ENV_BLOCK}
cp "${FT_DIR}/results/extended/retain_tr_scores.npy" "${WIN_DIR}/results/extended/retain_tr_scores.npy"
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${WIN_DIR}" --label shard_0_only --k 10 --forget_shard_id 9 --out "${WIN_DIR}/results/extended/shard_0_only.json" --hf_home "${HF_HOME}" --extended
echo "=== done \$(date) ==="
EOF
)
echo "Job W (winner extended): ${JOB_W}"

JOB_B=$(sbatch --parsable --dependency=afterok:${JOB_P} <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-ext-base
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_EXTENDED_TIME}
#SBATCH --output=${FT_DIR}/logs/base_ext_%j.log
#SBATCH --error=${FT_DIR}/logs/base_ext_%j.log
${ENV_BLOCK}
${PYTHON} "${SCRIPT_DIR}/eval_baseline.py" --model_name "${MODEL}" --output_dir "${FT_DIR}" --k 10 --forget_shard_id 9 --out "${FT_DIR}/results/extended/base_model.json" --hf_home "${HF_HOME}" --extended
echo "=== done \$(date) ==="
EOF
)
echo "Job B (base extended): ${JOB_B}"
echo "Chain: P(${JOB_P}) -> {W(${JOB_W}), B(${JOB_B})}"
