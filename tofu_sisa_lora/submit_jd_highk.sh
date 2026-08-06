#!/bin/bash
# Phase 2: JD selective-keep at high shard counts (k=100,200) via the mode-B path.
# In-model JD is impossible here (loading + fp32-gathering all k adapters OOMs), so we:
#   1) BUILD the JD-compressed artifact on GPU from the shard adapter safetensors
#      (jd_collection build --device cuda), clustered per paper §10.1 (k100->7, k200->10),
#   2) MATERIALIZE keep-all (merged) and keep-all-but-forget (remerge) as PEFT adapter dirs,
#   3) EVAL each materialized adapter via eval_tofu --preloaded_adapter (loads base + 1 adapter,
#      so no memory wall) → full TOFU metrics incl forget_quality vs the cached retain90 ref.
# Build jobs run first; the eval array depends on them. <=4 GPUs total at any time.
# k=200 uses the r8 shard set (fits GPU); k=100 uses r32 (only rank available). STUB=1 to preview.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
MODEL_NAME="meta-llama/Llama-2-7B-chat-hf"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
ART_ROOT="${TOFU_STORAGE_ROOT}/jd_collections"        # large artifacts -> /storage2 (home is code-only)
WORK="${SCRIPT_DIR}/checkpoints/jd_sweep"; mkdir -p "${WORK}/logs"
mkdir -p "${ART_ROOT}"

declare -A KDIR=(
  [100]="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf_k100_r32_e5_lr1e4"
  [200]="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4"
)
declare -A KC=( [100]=7 [200]=10 )   # cluster counts (paper §10.1: 128->7, 256->10)

submit_build () {  # $1=k -> echoes job id
  local k="$1"
  local dir="${KDIR[$k]}" c="${KC[$k]}"
  local art="${ART_ROOT}/k${k}_c${c}_full"
  local mer="${ART_ROOT}/k${k}_c${c}_merged" rem="${ART_ROOT}/k${k}_c${c}_remerge"
  local adapters; adapters=$(for i in $(seq 0 $((k-1))); do printf '%s ' "${dir}/shard_${i}"; done)
  local script
  script=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=jd-build-k${k}
#SBATCH --array=0-0%1
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=${WORK}/logs/build_k${k}_%A.log
#SBATCH --error=${WORK}/logs/build_k${k}_%A.log
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
date; echo "BUILD k=${k} c=${c} -> ${art}"
${PYTHON} "${SCRIPT_DIR}/jd_collection.py" build --adapters ${adapters} --out "${art}" --variant full --clusters ${c} --rank 16 --seed 0 --device cuda
${PYTHON} "${SCRIPT_DIR}/jd_collection.py" merge --collection "${art}" --out "${mer}"
${PYTHON} "${SCRIPT_DIR}/jd_collection.py" merge --collection "${art}" --out "${rem}" --drop "shard_$((k-1))"
date; echo "BUILD+MATERIALIZE k=${k} done"
EOF
)
  if [ "${STUB:-0}" = "1" ]; then echo "STUB-BUILD-k${k}"; return; fi
  echo "${script}" | sbatch --parsable
}

# --- submit builds, capture ids ---
BID100=$(submit_build 100)
BID200=$(submit_build 200)
echo "build jobs: k100=${BID100}  k200=${BID200}"

# --- eval tasks (materialized adapters) ---
TASKS="${WORK}/highk_tasks.txt"; : > "${TASKS}"
for k in 100 200; do
  dir="${KDIR[$k]}"; c="${KC[$k]}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$dir" "$k" "$((k-1))" "merged_jd_full_c${c}"  "${ART_ROOT}/k${k}_c${c}_merged"  >> "${TASKS}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$dir" "$k" "$((k-1))" "remerge_jd_full_c${c}" "${ART_ROOT}/k${k}_c${c}_remerge" >> "${TASKS}"
done
N=$(wc -l < "${TASKS}")
echo "eval tasks (${N}):"; cat "${TASKS}"

EVAL_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-jd-highk
#SBATCH --array=0-$((N-1))%4
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=${WORK}/logs/highk_%A_%a.log
#SBATCH --error=${WORK}/logs/highk_%A_%a.log
set -euo pipefail
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS}")
DIR=\$(echo "\$LINE"|cut -f1); K=\$(echo "\$LINE"|cut -f2); FORGET=\$(echo "\$LINE"|cut -f3)
LABEL=\$(echo "\$LINE"|cut -f4); PRE=\$(echo "\$LINE"|cut -f5)
OUT_JSON="\${DIR}/results/smoke/\${LABEL}.json"
if [ -f "\${OUT_JSON}" ]; then echo "Skip existing \${OUT_JSON}"; exit 0; fi
date; echo "=== highk eval k=\${K} \${LABEL} (preloaded \${PRE}) ==="
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"; mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HF_TOKEN:-}"
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL_NAME}" --output_dir "\${DIR}" --label "\${LABEL}" --k "\${K}" --forget_shard_id "\${FORGET}" --preloaded_adapter "\${PRE}" --out "\${OUT_JSON}" --hf_home "${HF_HOME}" --smoke
date
EOF
)
if [ "${STUB:-0}" = "1" ]; then echo "--- STUB eval array ---"; echo "${EVAL_SCRIPT}"; exit 0; fi
echo "${EVAL_SCRIPT}" | sbatch --dependency=afterok:${BID100}:${BID200}
echo "Submitted eval array (afterok builds). Monitor: squeue -u \$USER"
