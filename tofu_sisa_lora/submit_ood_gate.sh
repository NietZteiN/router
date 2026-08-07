#!/bin/bash
# What does the OOD ORACLE buy? (thread log/selector_audit/, entry 2026-08-07_ood-queries-*.md)
#
# `eval_routed_scaffold` decides TOFU-vs-OOD with `q2author` — an exact question-to-author lookup
# its own docstring calls "oracle-gated". A deployment cannot know that "Where would you find the
# Eiffel Tower?" is not about one of its 200 sources, so the repo-best 0.8236 model_utility
# assumes that problem already solved. At k=10/1B the price of not solving it is on record
# (mu 0.556 gated vs 0.474 ungated); this prices it at k=200 on the headline pool.
#
# Two arms, identical except for the gate, both with forget10 deleted:
#   oracle  a q2author miss serves base+scaffold      (every published number)
#   route   a q2author miss is handed to the nearest SURVIVING centroid instead
#
# Source routing stays oracle-exact in BOTH arms, so the delta isolates this oracle rather than
# mixing it with the which-source oracle that §4.7 already covers. `model_utility` is a harmonic
# mean including real_authors and world_facts, which is exactly where the damage should land.
#
# Usage: bash submit_ood_gate.sh        # STUB=1 previews without submitting
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
E25="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"
FORGET="180-199"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"
LOG_DIR="${CKPT}/ood_gate_logs"
RES="${E25}/results/smoke"
mkdir -p "${LOG_DIR}" "${RES}"
if [ ! -f "${RES}/retain_tr_scores.npy" ]; then
  echo "missing ${RES}/retain_tr_scores.npy — forget_quality would be NaN." >&2
  exit 1
fi

body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=ood-gate
#SBATCH --array=0-1%${ARRAY_CAP}
$(tofu_sbatch_resources 1 8 48G)
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/oodgate_%A_%a.log
#SBATCH --error=${LOG_DIR}/oodgate_%A_%a.log
set -eo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
GATES=(oracle route)
G=\${GATES[\${SLURM_ARRAY_TASK_ID}]}
OUT="${RES}/routed_oodgate_\${G}.json"
echo "=== OOD gate arm: \${G} -> \${OUT} ==="
date
[ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
for i in \$(seq 0 199); do
  [ -f "${E25}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING shard_\${i}"; exit 1; }
done
${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
  --model_name "${MODEL}" --shards_dir "${E25}" --k 200 \\
  --forget_shard_id 199 --forget_author_ids "${FORGET}" --delete_shards "${FORGET}" \\
  --ood_gate "\${G}" --lazy_adapter_cache 8 --smoke \\
  --hf_home "${HF_HOME}" --out "\${OUT}"
date
EOF
}

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB: sbatch script (not submitted) -----" >&2
  body >&2
  echo "-----------------------------------------------" >&2
else
  echo "OOD gate: 2 arms (oracle / route) on the k=200 e25 pool, cap ${ARRAY_CAP}"
  body | sbatch --parsable ${DEP:+--dependency=afterany:${DEP}}
fi
