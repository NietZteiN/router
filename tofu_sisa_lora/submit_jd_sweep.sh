#!/bin/bash
# JD selective-keep unlearning sweep (smoke), capped at <=4 GPUs TOTAL.
#
# Runs the four JD labels (merged/remerge x full/diag) across the in-model-safe shard
# counts k in {4,10,50} for Llama-2-7B (the already-trained k-ladder). c=1 (no clustering),
# JD rank auto = (n/2)+7 per paper Section 6.5 (the recommended <=100-LoRA setting).
# k=100/200 need the mode-B build path (jd_collection.py) and are submitted separately.
#
# ALL tasks run as a SINGLE array with %4 so at most 4 GPUs are used at once. Skips any
# label whose result JSON already exists. Metrics are eval_tofu --smoke (forget_quality
# uses the cached retain90 retain_tr_scores.npy in each dir).
#
# Usage: bash submit_jd_sweep.sh        # STUB=1 prints the array script without submitting

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

MODEL_NAME="meta-llama/Llama-2-7B-chat-hf"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CAP=4   # <= 4 GPUs total, per request

# k -> checkpoint dir (forget shard = k-1)
declare -A KDIR=(
  [4]="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf_k4_r32_e5_lr1e4"
  [10]="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf"
  [50]="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf_k50_r32_e5_lr1e4"
)
LABELS=(merged_jd_full remerge_jd_full merged_jd_diag remerge_jd_diag)

WORK="${SCRIPT_DIR}/checkpoints/jd_sweep"
mkdir -p "${WORK}/logs"
TASKS="${WORK}/tasks.txt"
: > "${TASKS}"
for k in 4 10 50; do
  dir="${KDIR[$k]}"
  [ -d "$dir" ] || { echo "MISSING $dir"; exit 1; }
  for lab in "${LABELS[@]}"; do
    printf '%s\t%s\t%s\t%s\n' "$dir" "$k" "$((k-1))" "$lab" >> "${TASKS}"
  done
done
N=$(wc -l < "${TASKS}")
echo "JD sweep: ${N} tasks (4 labels x k{4,10,50}), <=${CAP} GPUs total, smoke metrics"
echo "tasks file: ${TASKS}"
cat "${TASKS}"

SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-jd-sweep
#SBATCH --array=0-$((N - 1))%${CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:55:00
#SBATCH --output=${WORK}/logs/jd_%A_%a.log
#SBATCH --error=${WORK}/logs/jd_%A_%a.log

set -euo pipefail
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS}")
DIR=\$(echo "\$LINE" | cut -f1); K=\$(echo "\$LINE" | cut -f2)
FORGET=\$(echo "\$LINE" | cut -f3); LABEL=\$(echo "\$LINE" | cut -f4)
OUT_JSON="\${DIR}/results/smoke/\${LABEL}.json"

if [ -f "\${OUT_JSON}" ]; then echo "Skip existing \${OUT_JSON}"; exit 0; fi
echo "=== JD eval \${SLURM_JOB_ID}.\${SLURM_ARRAY_TASK_ID}: k=\${K} \${LABEL} -> \${OUT_JSON} ==="
date
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HF_TOKEN:-}"

${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL_NAME}" --output_dir "\${DIR}" --label "\${LABEL}" --k "\${K}" --forget_shard_id "\${FORGET}" --out "\${OUT_JSON}" --hf_home "${HF_HOME}" --smoke
date
EOF
)

if [ "${STUB:-0}" = "1" ]; then echo "--- STUB (not submitting) ---"; echo "${SBATCH_SCRIPT}"; exit 0; fi
echo "${SBATCH_SCRIPT}" | sbatch
echo "Submitted. Monitor: squeue -u \$USER | grep jd-sweep"
echo "Collect:  ${PYTHON} ${SCRIPT_DIR}/collect_results.py --root ${SCRIPT_DIR}/checkpoints --smoke"
