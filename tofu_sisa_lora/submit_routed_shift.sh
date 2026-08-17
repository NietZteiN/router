#!/bin/bash
# The ROUTED half of the plain-FT comparison (Vincent's Q4/Q5).
#
# submit_plain_ft_baseline.sh measures a routerless model on the audit's own query conditions;
# this measures the k=200 routed pool on the SAME 800 rows so the two halves are comparable. Two
# things it does that no existing arm did:
#
#   --serve_rows shift800   serve the 400 RETAIN questions as well as the 400 orphans. Every
#                           previous run of this script served orphans only, so the routed side
#                           of a named/anonymised x retain/forget table simply did not exist.
#   --query_transform name_injected|name_swapped
#                           SERVE finding 5's attacks instead of only routing them. Finding 5
#                           reports which UNIT the attack captures; a routerless model has no
#                           unit, so the only criterion both systems can share is what the served
#                           answer SAYS -- scored by the same selector_audit/csar.py matcher.
#
# Deletion is identical to the audit (authors 180-199 excluded from every router); a retain row
# is simply one whose own expert survives.
#
# Usage: bash submit_routed_shift.sh [qa|qb|all]      # STUB=1 previews, SHARDS=n overrides
set -euo pipefail

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"

MODEL="meta-llama/Llama-2-7B-chat-hf"
E25="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"
FORGET="180-199"
# Feature-space routers only, matching the CSAR arms: the behavioural family scores by running
# EVERY candidate expert on EVERY query, which is impractical at k=200 and adds nothing here.
STRATS="${STRATS:-centroid_sbert,key_tfidf}"
ATTACKER="${ATTACKER:-0}"

SHARDS="${SHARDS:-8}"
export TOFU_ARRAY_CAP="${TOFU_ARRAY_CAP_OVERRIDE:-${SHARDS}}"

LOG_DIR="${CKPT}/routed_shift_logs"
OUT_DIR="${E25}/results/routed_shift"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

case "${STAGE}" in
  qa)  CONDS="none name_stripped para_stripped" ;;
  qb)  CONDS="name_injected name_swapped" ;;
  all) CONDS="none name_stripped para_stripped name_injected name_swapped" ;;
  *)   echo "usage: $0 [qa|qb|all]" >&2; exit 2 ;;
esac

submit() {
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----" >&2
    printf '%s\n' "$1" >&2
    echo "-----------------------------------------------" >&2
    echo "STUB"
  else
    printf '%s\n' "$1" | sbatch --parsable
  fi
}

arm_body() {
  local qt="$1"
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=rsh-${qt}
#SBATCH --array=0-$((SHARDS - 1))%${TOFU_ARRAY_CAP}
$(tofu_sbatch_resources 1 8 48G)
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/${qt}_%A_%a.log
#SBATCH --error=${LOG_DIR}/${qt}_%A_%a.log
set -eo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
OUT="${OUT_DIR}/routed_${qt}_shard\${SLURM_ARRAY_TASK_ID}_of_${SHARDS}.json"
echo "=== routed shift800: transform=${qt}, shard \${SLURM_ARRAY_TASK_ID}/${SHARDS} ==="
date
[ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
for i in \$(seq 0 199); do
  [ -f "${E25}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING shard_\${i}"; exit 1; }
done
${PYTHON} "${SCRIPT_DIR}/dump_generations_routed.py" \\
  --model_name "${MODEL}" --shards_dir "${E25}" --k 200 \\
  --forget_author_ids "${FORGET}" \\
  --serve_rows shift800 --row_shard "\${SLURM_ARRAY_TASK_ID}/${SHARDS}" \\
  --query_transform "${qt}" --attacker_id ${ATTACKER} \\
  --strategies "${STRATS}" --lazy_adapter_cache 8 \\
  --hf_home "${HF_HOME}" --out "\${OUT}"
date
EOF
}

for qt in ${CONDS}; do
  jid=$(submit "$(arm_body "${qt}")")
  echo "${qt} array : ${jid}"
done
echo "outputs : ${OUT_DIR}"
echo "logs    : ${LOG_DIR}"
