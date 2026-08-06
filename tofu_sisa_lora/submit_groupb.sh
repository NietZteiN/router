#!/bin/bash
# Router-leak (b): Group-B methods under a realistic embedding selector.
#   bash submit_groupb.sh [routing|serve|all]
# STUB=1 previews. MSINK=1 adds the memsinks cell (separate project; best-effort).
# routing = CPU/light (MiniLM only, no LLM) — runs inline on the login node.
# serve   = GPU smoke cells (oracle vs realistic on the 400 forget questions).
# CPU gate first: python test_groupb_router.py
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PY=${TOFU_PYTHON:-python3}
OUT=${TOFU_CKPT_ROOT}/Llama-3.2-1B-Instruct_experts_scaf_k10/results/router_leak/groupb
mkdir -p "${OUT}"
METHODS="sift clamu"
[ "${MSINK:-0}" = "1" ] && METHODS="sift clamu memsinks"

submit() {  # submit <name> <time> <cmd...>
  local name=$1 time=$2; shift 2
  local body="#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${time}
#SBATCH --output=${OUT}/logs_%x_%j.out
set -eo pipefail
export HF_HOME=${HF_HOME}
cd ${SCRIPT_DIR}
$*"
  if [ "${STUB:-0}" = "1" ]; then echo "===== STUB ${name} ====="; echo "${body}"; return; fi
  local out; out=$(echo "${body}" | sbatch ${DEP:+--dependency=afterany:${DEP}})
  echo "${out} (${name})"; LAST_JOB=$(echo "${out}" | awk '{print $NF}')
}

routing() {   # CPU/light — MiniLM only; run inline (login node ok, ~seconds/method)
  for m in ${METHODS}; do
    [ -f "${OUT}/routing_${m}.json" ] && { echo "skip routing ${m} (exists)"; continue; }
    ${PY} orphan_route_groupb.py --method ${m} --mode routing --device cpu \
      --out ${OUT}/routing_${m}.json
  done
}

serve() {   # GPU smoke: oracle vs realistic on the 400 forget questions
  local cap="${CAP:+--max_q ${CAP}}"
  for m in ${METHODS}; do
    [ -f "${OUT}/serve_${m}.json" ] && { echo "skip serve ${m} (exists)"; continue; }
    submit gb-serve-${m} ${TOFU_SMOKE_TIME} \
      ${PY} orphan_route_groupb.py --method ${m} --mode serve --unlearn_tag forget10 \
        ${cap} --device cuda --out ${OUT}/serve_${m}.json
  done
}

case "${1:-all}" in
  routing) routing ;;
  serve) serve ;;
  all) routing; serve ;;
  *) echo "usage: bash submit_groupb.sh [routing|serve|all]"; exit 1 ;;
esac
