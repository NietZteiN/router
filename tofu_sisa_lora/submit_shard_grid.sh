#!/bin/bash
# SISA shard-recipe star grid (plan have-we-tried-finetuning-snuggly-dream, 2026-06-11):
# 7 configs around center (k10, r32/a64, e5, lr1e-4) for meta-llama/Llama-2-7B-chat-hf.
# Baseline = merge ALL k shards (merged_*); unlearn = drop shard k-1 (remerge_*). Smoke only.
# GPU budget: ALL jobs pinned to sprint3 (4 GPUs), arrays throttled %4 (user constraint).
# Per config: training array 0..k-1 (skipped if adapters exist) -> eval array over 5-7 labels.
# KS ref: k=10 copies _ft smoke retain_tr_scores.npy (same forget rows); k=20 preps its own
# (oracle valid: shard19 = authors 190-199); k=4 has none -> forget_quality NaN by design.
# Usage: bash submit_shard_grid.sh
# NOTE: job commands single-line (continuations inside $(sbatch <<EOF) become space args).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
CKPT="${SCRIPT_DIR}/checkpoints"
FT_REF="${CKPT}/Llama-2-7B-chat-hf_ft/results/smoke/retain_tr_scores.npy"
ORACLE="${CKPT}/Llama-2-7B-chat-hf/retain90"
NODE_LINE="#SBATCH --nodelist=sprint3"

[ -f "${FT_REF}" ] || { echo "Missing KS reference ${FT_REF}"; exit 1; }
[ -f "${ORACLE}/adapter_config.json" ] || { echo "Missing retain90 oracle ${ORACLE}"; exit 1; }

# name|k|rank|alpha|epochs|lr|dirname
CONFIGS=(
  "CTRL|10|8|16|3|2e-4|Llama-2-7B-chat-hf"
  "CENTER|10|32|64|5|1e-4|Llama-2-7B-chat-hf_k10_r32_e5_lr1e4"
  "RANKLO|10|8|16|5|1e-4|Llama-2-7B-chat-hf_k10_r8_e5_lr1e4"
  "EXPO|10|16|32|10|1e-4|Llama-2-7B-chat-hf_k10_r16_e10_lr1e4"
  "LRHI|10|32|64|5|2e-4|Llama-2-7B-chat-hf_k10_r32_e5_lr2e4"
  "KLO|4|32|64|5|1e-4|Llama-2-7B-chat-hf_k4_r32_e5_lr1e4"
  "KHI|20|32|64|5|1e-4|Llama-2-7B-chat-hf_k20_r32_e5_lr1e4"
)

EVAL_JOBS=()
echo "=== SISA shard grid: ${#CONFIGS[@]} configs, sprint3 only, %4 throttle ==="
for CFG in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME K RANK ALPHA EPOCHS LR DIRNAME <<< "${CFG}"
  VDIR="${CKPT}/${DIRNAME}"
  LOG_DIR="${VDIR}/logs"
  mkdir -p "${LOG_DIR}" "${VDIR}/results/smoke"
  FORGET=$((K - 1))

  # ---- KS reference per k ----
  if [ "${K}" -eq 10 ]; then
    cp -f "${FT_REF}" "${VDIR}/results/smoke/retain_tr_scores.npy"
  elif [ "${K}" -eq 20 ] && [ ! -e "${VDIR}/retain90" ]; then
    ln -s "../Llama-2-7B-chat-hf/retain90" "${VDIR}/retain90"
  fi

  # ---- training array (skip if all shard adapters already exist) ----
  TRAIN_DEP=""
  HAVE_ALL=1
  for i in $(seq 0 ${FORGET}); do
    [ -f "${VDIR}/shard_${i}/adapter_config.json" ] || { HAVE_ALL=0; break; }
  done
  if [ "${HAVE_ALL}" -eq 1 ]; then
    echo "  ${NAME}: all ${K} shard adapters exist — training skipped"
  else
    TRAIN_JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=sg-tr-${NAME}
#SBATCH --partition=all
${NODE_LINE}
#SBATCH --array=0-${FORGET}%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
echo "=== ${NAME} shard \${SLURM_ARRAY_TASK_ID}/${K} (r${RANK} a${ALPHA} e${EPOCHS} lr${LR}) node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --shard_id \${SLURM_ARRAY_TASK_ID} --k ${K} --model_name "${MODEL}" --rank ${RANK} --alpha ${ALPHA} --epochs ${EPOCHS} --lr ${LR} --batch_size 1 --grad_accum 16 --max_length 256 --output_dir "${VDIR}" --hf_home "${HF_HOME}" --seed 42
echo "=== done \$(date) ==="
EOF
)
    echo "  ${NAME}: training array ${TRAIN_JOB} (${K} tasks %4)"
    TRAIN_DEP="afterok:${TRAIN_JOB}"
  fi

  # ---- k=20 KS reference prep (GPU; needs oracle; eval waits on it too) ----
  if [ "${K}" -eq 20 ] && [ ! -f "${VDIR}/results/smoke/retain_tr_scores.npy" ]; then
    PREP_JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=sg-prep-${NAME}
#SBATCH --partition=all
${NODE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/prepare_%j.log
#SBATCH --error=${LOG_DIR}/prepare_%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --smoke --model_name "${MODEL}" --output_dir "${VDIR}" --k ${K} --forget_shard_id ${FORGET} --hf_home "${HF_HOME}"
test -f "${VDIR}/results/smoke/retain_tr_scores.npy" || { echo "FATAL: reference missing"; exit 1; }
echo "=== done \$(date) ==="
EOF
)
    echo "  ${NAME}: prepare job ${PREP_JOB}"
    TRAIN_DEP="${TRAIN_DEP:+${TRAIN_DEP},}afterok:${PREP_JOB}"
  fi

  # ---- eval array ----
  LABELS="merged_linear merged_dare_ties remerge_linear remerge_dare_ties shard_${FORGET}_only"
  [ "${NAME}" = "CENTER" ] && LABELS="${LABELS} merged_cat remerge_cat"
  NLAB=$(echo "${LABELS}" | wc -w)
  DEP_FLAG=""
  [ -n "${TRAIN_DEP}" ] && DEP_FLAG="--dependency=${TRAIN_DEP}"

  EVAL_JOB=$(sbatch --parsable ${DEP_FLAG} <<EOF
#!/bin/bash
#SBATCH --job-name=sg-ev-${NAME}
#SBATCH --partition=all
${NODE_LINE}
#SBATCH --array=0-$((NLAB - 1))%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
for i in \$(seq 0 ${FORGET}); do test -f "${VDIR}/shard_\${i}/adapter_config.json" || { echo "FATAL: shard \${i} adapter missing"; exit 1; }; done
LABEL=\$(echo "${LABELS}" | cut -d' ' -f\$((SLURM_ARRAY_TASK_ID + 1)))
echo "=== ${NAME} eval \${LABEL} (k=${K}) node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${VDIR}" --label "\${LABEL}" --k ${K} --forget_shard_id ${FORGET} --out "${VDIR}/results/smoke/\${LABEL}.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
  echo "  ${NAME}: eval array ${EVAL_JOB} (${NLAB} labels %4)${DEP_FLAG:+ dep ${TRAIN_DEP}}"
  EVAL_JOBS+=("${EVAL_JOB}")
done

# ---- tail: aggregate CSV after all eval arrays (no GPU) ----
DEPLIST=$(IFS=:; echo "${EVAL_JOBS[*]}")
COLLECT=$(sbatch --parsable --dependency=afterany:${DEPLIST} <<EOF
#!/bin/bash
#SBATCH --job-name=sg-collect
#SBATCH --partition=all
${NODE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=${CKPT}/shard_grid_collect_%j.log
#SBATCH --error=${CKPT}/shard_grid_collect_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "${CKPT}" --smoke
echo "=== collect done \$(date) ==="
EOF
)
echo "  collect: ${COLLECT} (afterany:${DEPLIST})"
echo ""
echo "Monitor: squeue -u \$USER -o '%.12i %.18j %.8T %.8M %R'"
