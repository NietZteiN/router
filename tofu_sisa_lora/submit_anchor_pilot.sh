#!/bin/bash
# §6.3 negative-anchor λ-pilot driver
# (log/merge_mechanism/2026-07-15_negative-anchor-design.md).
#
# Usage: bash submit_anchor_pilot.sh [train|keyfire|iso]     # STUB=1 previews
#   train   15-task GPU array: probe authors {82,15,111,177,76} × λ ∈ {1,10,100} at the
#           e25 recipe (--k 200 --epochs 25) + --anchor_lambda λ →
#           {slug}_k200_r32_e25_anch{λ}_lr1e4/shard_{a}; per-shard self-skip built into
#           train_lora_shard.py.
#   keyfire 3 × 1-GPU: measure_key_firing.py per λ pool → reports/key_firing_e25_anch{λ}.json
#           (H-anchor-1 readout; same seed/harness as Exp-7 so ratios are comparable).
#   iso     15-task GPU array: eval_tofu --preloaded_adapter own-author rows (recall check;
#           copies the e5 KS ref into each λ dir first so forget_quality isn't NaN).
# ⚠ Every stage is GPU — check `squeue -u jack` against the GLOBAL 4-GPU cap first.
set -euo pipefail

STAGE="${1:-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
KS_REF="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4/results/smoke/retain_tr_scores.npy"
AUTHORS=(82 15 111 177 76)
LAMBDAS=(1 10 100)
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"

dir_for() { echo "${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_anch$1_lr1e4"; }

submit() {
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----"
    printf '%s\n' "$1"
    echo "-----------------------------------------------"
  else
    printf '%s\n' "$1" | sbatch --parsable
  fi
}

LOG_DIR="${CKPT}/anchor_pilot_logs"
mkdir -p "${LOG_DIR}"
for L in "${LAMBDAS[@]}"; do mkdir -p "$(dir_for "${L}")"; done

case "${STAGE}" in
train)
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-anchor-train
#SBATCH --array=0-14%${ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log

AUTHORS=(82 15 111 177 76)
LAMBDAS=(1 10 100)
A=\${AUTHORS[\$((SLURM_ARRAY_TASK_ID % 5))]}
L=\${LAMBDAS[\$((SLURM_ARRAY_TASK_ID / 5))]}
OUT_DIR="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_anch\${L}_lr1e4"
echo "=== anchor pilot train task \${SLURM_ARRAY_TASK_ID}: author \${A} lambda \${L} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" \\
  --shard_id "\${A}" --k 200 \\
  --model_name "${MODEL}" \\
  --output_dir "\${OUT_DIR}" \\
  --epochs 25 \\
  --anchor_lambda "\${L}" \\
  --hf_home "${HF_HOME}"
date
EOF
  echo "anchor pilot train: 15 tasks (5 authors x 3 lambdas), cap ${ARRAY_CAP}"
  submit "${S}"
  ;;
keyfire)
  for L in "${LAMBDAS[@]}"; do
    out="${SCRIPT_DIR}/reports/key_firing_e25_anch${L}.json"
    if [ -f "${out}" ]; then echo "skip existing ${out}"; continue; fi
    read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-keyfire-anch${L}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/keyfire_anch${L}_%j.log
#SBATCH --error=${LOG_DIR}/keyfire_anch${L}_%j.log
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
date
${PYTHON} "${SCRIPT_DIR}/measure_key_firing.py" \\
  --model_name "${MODEL}" \\
  --shards_dir "$(dir_for "${L}")" \\
  --out "${out}" \\
  --questions_per_author 5 --ood_n 100 --seed 42 --device cuda --batch_size 8 \\
  --hf_home "${HF_HOME}"
date
EOF
    echo "keyfire anch${L}: 1 GPU"
    submit "${S}"
  done
  ;;
iso)
  for L in "${LAMBDAS[@]}"; do
    mkdir -p "$(dir_for "${L}")/results/smoke"
    cp -n "${KS_REF}" "$(dir_for "${L}")/results/smoke/" 2>/dev/null || true
  done
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-anchor-iso
#SBATCH --array=0-14%${ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/iso_%A_%a.log
#SBATCH --error=${LOG_DIR}/iso_%A_%a.log

AUTHORS=(82 15 111 177 76)
LAMBDAS=(1 10 100)
A=\${AUTHORS[\$((SLURM_ARRAY_TASK_ID % 5))]}
L=\${LAMBDAS[\$((SLURM_ARRAY_TASK_ID / 5))]}
OUT_DIR="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_anch\${L}_lr1e4"
OUT_JSON="\${OUT_DIR}/results/smoke/iso_a\${A}__own\${A}.json"
if [ -f "\${OUT_JSON}" ]; then echo "Skip existing \${OUT_JSON}"; exit 0; fi
[ -f "\${OUT_DIR}/shard_\${A}/adapter_model.safetensors" ] || { echo "missing adapter"; exit 1; }
echo "=== anchor pilot iso task \${SLURM_ARRAY_TASK_ID}: author \${A} lambda \${L} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
  --model_name "${MODEL}" \\
  --output_dir "\${OUT_DIR}" \\
  --label "iso_a\${A}" \\
  --k 200 --forget_shard_id 199 \\
  --eval_shard_id "\${A}" \\
  --preloaded_adapter "\${OUT_DIR}/shard_\${A}" \\
  --out "\${OUT_JSON}" \\
  --hf_home "${HF_HOME}" \\
  --smoke
date
EOF
  echo "anchor pilot iso: 15 tasks, cap ${ARRAY_CAP}"
  submit "${S}"
  ;;
*) echo "usage: bash submit_anchor_pilot.sh [train|keyfire|iso]"; exit 1 ;;
esac
