#!/bin/bash
# JD Phase-2 high-k: k100-ONLY eval (decoupled from the timed-out k200 build).
# Rationale (LOG 2026-06-17): submit_jd_highk.sh chained eval on afterok:BID100:BID200, so the
# k200 build hitting its time wall stranded the whole array (434881 DependencyNeverSatisfied) even
# though the k100 JD artifacts built fine. This evals only the already-materialized k100 adapters
# (merged + remerge), no build, no dependency. k200 is handled separately (longer build wall).
# Evals base + 1 preloaded adapter via eval_tofu --preloaded_adapter → no high-k memory wall.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
MODEL_NAME="meta-llama/Llama-2-7B-chat-hf"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
ART_ROOT="${TOFU_STORAGE_ROOT}/jd_collections"
WORK="${SCRIPT_DIR}/checkpoints/jd_sweep"; mkdir -p "${WORK}/logs"

K=100; C=7
DIR="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf_k${K}_r32_e5_lr1e4"
MER="${ART_ROOT}/k${K}_c${C}_merged"
REM="${ART_ROOT}/k${K}_c${C}_remerge"

# guard: artifacts must already be materialized (this script does NOT build)
for p in "${MER}" "${REM}"; do
  [ -f "${p}/adapter_model.safetensors" ] || { echo "MISSING built artifact: ${p}" >&2; exit 1; }
done

TASKS="${WORK}/k100_eval_tasks.txt"; : > "${TASKS}"
printf '%s\t%s\t%s\t%s\t%s\n' "$DIR" "$K" "$((K-1))" "merged_jd_full_c${C}"  "${MER}" >> "${TASKS}"
printf '%s\t%s\t%s\t%s\t%s\n' "$DIR" "$K" "$((K-1))" "remerge_jd_full_c${C}" "${REM}" >> "${TASKS}"
N=$(wc -l < "${TASKS}")
echo "k100 eval tasks (${N}):"; cat "${TASKS}"

EVAL_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-jd-k100
#SBATCH --array=0-$((N-1))%4
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=${WORK}/logs/k100eval_%A_%a.log
#SBATCH --error=${WORK}/logs/k100eval_%A_%a.log
set -euo pipefail
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS}")
DIR=\$(echo "\$LINE"|cut -f1); K=\$(echo "\$LINE"|cut -f2); FORGET=\$(echo "\$LINE"|cut -f3)
LABEL=\$(echo "\$LINE"|cut -f4); PRE=\$(echo "\$LINE"|cut -f5)
OUT_JSON="\${DIR}/results/smoke/\${LABEL}.json"
if [ -f "\${OUT_JSON}" ]; then echo "Skip existing \${OUT_JSON}"; exit 0; fi
date; echo "=== k100 eval \${LABEL} (preloaded \${PRE}) ==="
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"; mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HF_TOKEN:-}"
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL_NAME}" --output_dir "\${DIR}" --label "\${LABEL}" --k "\${K}" --forget_shard_id "\${FORGET}" --preloaded_adapter "\${PRE}" --out "\${OUT_JSON}" --hf_home "${HF_HOME}" --smoke
date
EOF
)
if [ "${STUB:-0}" = "1" ]; then echo "--- STUB k100 eval array ---"; echo "${EVAL_SCRIPT}"; exit 0; fi
echo "${EVAL_SCRIPT}" | sbatch --parsable
echo "Submitted k100-only eval array (no dependency). Monitor: squeue -u \$USER"
