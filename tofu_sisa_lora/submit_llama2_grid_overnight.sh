#!/bin/bash
# Stage 2 overnight grid (plan have-we-tried-finetuning-snuggly-dream): 7 k=1 LoRA recipe
# variants for meta-llama/Llama-2-7B-chat-hf, queued as a strictly sequential SLURM chain
# (--dependency=afterany so one failed variant cannot deadlock the night), each job doing
# train -> eval in one script, plus a final no-GPU collect_results job.
# Target: model_utility >= 0.6 under the corrected OU-faithful eval. Seed 42, bs1 x ga32,
# max_len 256. KS reference (retain_tr_scores.npy) is variant-independent -> copied from the
# _ft dir instead of re-running prepare_eval per variant.
# Usage: bash submit_llama2_grid_overnight.sh
# NOTE: job commands stay on single lines — backslash continuations inside $(sbatch <<EOF)
# collapse into literal space args (root cause of failed job 433510).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

MODEL="meta-llama/Llama-2-7B-chat-hf"
SLUG="Llama-2-7B-chat-hf"
CKPT="${SCRIPT_DIR}/checkpoints"
REF_NPY="${CKPT}/${SLUG}_ft/results/smoke/retain_tr_scores.npy"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

[ -f "${REF_NPY}" ] || { echo "Missing KS reference ${REF_NPY} — run Stage 1 prepare first"; exit 1; }

# lr epochs rank alpha suffix
VARIANTS=(
  "1e-4 5  16 32 lr1e4_e5_r16"
  "1e-4 5  8  16 lr1e4_e5_r8"
  "2e-4 5  16 32 lr2e4_e5_r16"
  "5e-5 5  16 32 lr5e5_e5_r16"
  "1e-4 10 16 32 lr1e4_e10_r16"
  "1e-4 5  32 64 lr1e4_e5_r32"
  "5e-5 10 32 64 lr5e5_e10_r32"
)

PREV=""
echo "=== Overnight grid: ${#VARIANTS[@]} sequential train+eval jobs (${SLUG}) ==="
for V in "${VARIANTS[@]}"; do
  read -r LR EPOCHS RANK ALPHA SUFFIX <<< "${V}"
  VDIR="${CKPT}/${SLUG}_ft_${SUFFIX}"
  LOG_DIR="${VDIR}/logs"
  mkdir -p "${LOG_DIR}" "${VDIR}/results/smoke"
  DEP=""
  [ -n "${PREV}" ] && DEP="--dependency=afterany:${PREV}"

  JOB=$(sbatch --parsable ${DEP} <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-grid-${SUFFIX}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=${LOG_DIR}/train_eval_%j.log
#SBATCH --error=${LOG_DIR}/train_eval_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
echo "=== grid ${SUFFIX} (lr=${LR} e=${EPOCHS} r=${RANK} a=${ALPHA}) node \$(hostname) gpu \${CUDA_VISIBLE_DEVICES:-?} \$(date) ==="

${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --shard_id 0 --k 1 --model_name "${MODEL}" --rank ${RANK} --alpha ${ALPHA} --epochs ${EPOCHS} --lr ${LR} --batch_size 1 --grad_accum 32 --max_length 256 --output_dir "${VDIR}" --hf_home "${HF_HOME}" --seed 42
test -f "${VDIR}/shard_0/adapter_config.json" || { echo "FATAL: no adapter written"; exit 1; }

cp "${REF_NPY}" "${VDIR}/results/smoke/retain_tr_scores.npy"
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${VDIR}" --label shard_0_only --k 10 --forget_shard_id 9 --out "${VDIR}/results/smoke/shard_0_only.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
  echo "  ${SUFFIX}: job ${JOB}${PREV:+ (afterany:${PREV})}  lr=${LR} e=${EPOCHS} r=${RANK} a=${ALPHA}"
  PREV="${JOB}"
done

# Tail job: aggregate CSV ready for the morning (no GPU).
COLLECT=$(sbatch --parsable --dependency=afterany:${PREV} <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-grid-collect
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=${CKPT}/grid_collect_%j.log
#SBATCH --error=${CKPT}/grid_collect_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "${CKPT}" --smoke
echo "=== collect done \$(date) ==="
EOF
)
echo "  collect: job ${COLLECT} (afterany:${PREV})"
echo ""
echo "Chain queued. Monitor: squeue -u \$USER | grep tofu-grid"
