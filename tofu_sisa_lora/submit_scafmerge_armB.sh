#!/bin/bash
# Scaffold x composition 2x2 control — ARM B (OOD-aware MERGED serving).
# One SLURM task per merge label: eval_routed_scaffold.py --merged_label serves every
# TOFU-author query with the one merged adapter (OOD -> scaffold-only). remerge_* =
# the merged-deployment deletion condition. Pre-registration: CLAUDE_SCRATCHPAD.md
# (2026-07-07) + log/routing_scaffold/2026-07-07_scafmerge-control-design.md.
# Arm A (merged-everywhere) goes through submit_eval.sh with
# results/smoke/eval_manifest_scafmerge.txt instead — see the staged commands in the log entry.
#
# Usage: bash submit_scafmerge_armB.sh [smoke|extended]
#   STUB=1 prints the sbatch script without submitting.
# CPU gates first (both must be green): python test_routed_scaffold_merged.py
#                                       python test_merge_extra.py

set -euo pipefail

SUB="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

SHARDS_DIR="${SCRIPT_DIR}/checkpoints/Llama-3.2-1B-Instruct_experts_scaf_k10"
MODEL_NAME="${SCRIPT_DIR}/checkpoints/Llama-3.2-1B-Instruct_scaffolded_alpaca2k"
RESULTS_DIR="${SHARDS_DIR}/results/${SUB}"
LOG_DIR="${SHARDS_DIR}/logs"
# Override the label set via env, e.g. SCAFMERGE_LABELS="merged_knots_ties merged_tsv"
# (labels must resolve to a single adapter through activate_label — no data-required or
# ensemble labels here; those go through eval_tofu / submit_eval.sh instead).
read -r -a LABELS <<< "${SCAFMERGE_LABELS:-merged_additive_mean remerge_additive_mean merged_dare_ties remerge_dare_ties}"
ARRAY_CAP="${ARRAY_CAP:-4}"
EVAL_TIME="${EVAL_TIME:-02:00:00}"

case "${SUB}" in
  smoke) SUB_FLAG="--smoke" ;;
  extended) SUB_FLAG="--extended" ;;
  *) echo "unknown sub '${SUB}' (smoke|extended)"; exit 1 ;;
esac

[ -f "${RESULTS_DIR}/retain_tr_scores.npy" ] || {
  echo "Missing ${RESULTS_DIR}/retain_tr_scores.npy (forget_quality KS reference)"; exit 1; }
mkdir -p "${LOG_DIR}"

emit_sbatch() {
cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-scafmergeB-${SUB}
#SBATCH --array=0-$((${#LABELS[@]} - 1))%${ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${EVAL_TIME}
#SBATCH --output=${LOG_DIR}/scafmergeB_%A_%a.log
#SBATCH --error=${LOG_DIR}/scafmergeB_%A_%a.log

LABELS=(${LABELS[@]})
LABEL=\${LABELS[\${SLURM_ARRAY_TASK_ID}]}
OUT_JSON="${RESULTS_DIR}/scafmerged_\${LABEL}.json"

if [ -f "\${OUT_JSON}" ]; then
  echo "Skip existing \${OUT_JSON}"
  exit 0
fi

echo "=== scafmerge arm B job \${SLURM_JOB_ID} task \${SLURM_ARRAY_TASK_ID}: \${LABEL} (${SUB}) ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
  --model_name "${MODEL_NAME}" \\
  --shards_dir "${SHARDS_DIR}" \\
  --k 10 \\
  --merged_label "\${LABEL}" \\
  ${SUB_FLAG} \\
  --out "\${OUT_JSON}"

date
EOF
}

if [ "${STUB:-0}" = "1" ]; then
  echo "--- STUB: would submit ---"
  emit_sbatch
  exit 0
fi

echo "Submitting ${#LABELS[@]} arm-B tasks (${SUB}, cap %${ARRAY_CAP}, ${EVAL_TIME}/task, sprint1-3)"
emit_sbatch | sbatch
echo "Monitor: squeue -u \$USER ; results -> ${RESULTS_DIR}/scafmerged_<label>.json"
