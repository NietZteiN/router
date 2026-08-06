#!/bin/bash
# §9-D router-after-deletion audit driver (gap-analysis §9-D; log/ramole/2026-07-03_routing-audit-9d.md).
#   bash submit_routing_audit_9d.sh [all|audits|scaffold]
# B1 audits (1 GPU each, ~15 min): base-pinned encoder audit (stale/rebuilt/dropped/key on the
# _encbase index — the off-the-shelf encoder the 06-27 fq-0.484 result used) + the FT-encoder
# dropped-policy re-run. B2 scaffold (1 GPU each, ~1.5 h): extended-cap eval_routed_scaffold
# full + --delete_shard 9 on the strong-experts arm (extended KS ref copied from the SISA 1B dir
# — byte-identical to how the smoke ref was provisioned; no prepare_eval job needed).
# STUB=1 prints every sbatch script without submitting. Existing result JSONs are skipped.
set -euo pipefail
PHASE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
POOL="${TOFU_CKPT_ROOT}/Llama-3.2-1B-Instruct_legonet_n32_k3"
SCAF_DIR="${TOFU_CKPT_ROOT}/Llama-3.2-1B-Instruct_experts_scaf_k10"
SCAF_BASE="${TOFU_CKPT_ROOT}/Llama-3.2-1B-Instruct_scaffolded_alpaca2k"
SISA_1B="${TOFU_CKPT_ROOT}/Llama-3.2-1B-Instruct"
LOG_DIR="${POOL}/logs"
SCAF_LOG="${SCAF_DIR}/logs"
mkdir -p "${LOG_DIR}" "${SCAF_LOG}"

run_sbatch() { if [ "${STUB:-0}" = "1" ]; then cat >&2; echo "----(STUB)----" >&2; echo "STUB"; else sbatch --parsable; fi; }

submit_audit() {  # $1 job-name  $2 config  $3 policies  $4 out-json
  local OUT="$4"
  if [ -f "${OUT}" ]; then echo "  skip (exists): ${OUT}" >&2; echo "SKIP"; return; fi
  run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/$1_%j.log
set -e
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" "${SCRIPT_DIR}/routing_audit_tofu.py" --config "${SCRIPT_DIR}/$2" \\
    --tag forget10 --policies $3 --device cuda --out "${OUT}"
EOF
}

submit_scaf_eval() {  # $1 job-name  $2 extra-flags  $3 out-json
  local OUT="$3"
  if [ -f "${OUT}" ]; then echo "  skip (exists): ${OUT}" >&2; echo "SKIP"; return; fi
  run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_EXTENDED_TIME}
#SBATCH --output=${SCAF_LOG}/$1_%j.log
set -e
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" "${SCRIPT_DIR}/eval_routed_scaffold.py" --model_name "${SCAF_BASE}" \\
    --shards_dir "${SCAF_DIR}" --k 10 --forget_shard_id 9 --extended $2 --out "${OUT}"
EOF
}

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "audits" ]; then
  J1=$(submit_audit "audit9d-basepin" "configs/ramole_tofu_1b_basepin.json" \
       "stale rebuilt dropped key" "${POOL}/results/routing_audit_forget10_basepin.json")
  J2=$(submit_audit "audit9d-ftdrop" "configs/ramole_tofu_1b.json" \
       "stale dropped" "${POOL}/results/routing_audit_forget10_ftdrop.json")
  echo "audits: basepin=${J1} ftdrop=${J2}"
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "scaffold" ]; then
  # Extended KS reference: reuse the SISA 1B extended oracle scores (method-independent; the
  # smoke ref in this dir is already a byte-identical copy of the SISA smoke ref).
  mkdir -p "${SCAF_DIR}/results/extended"
  if [ ! -f "${SCAF_DIR}/results/extended/retain_tr_scores.npy" ]; then
    if [ "${STUB:-0}" = "1" ]; then
      echo "  (STUB) would cp ${SISA_1B}/results/extended/retain_tr_scores.npy -> ${SCAF_DIR}/results/extended/" >&2
    else
      cp "${SISA_1B}/results/extended/retain_tr_scores.npy" "${SCAF_DIR}/results/extended/"
      echo "  copied extended KS ref -> ${SCAF_DIR}/results/extended/"
    fi
  fi
  J3=$(submit_scaf_eval "scaf9d-full" "" "${SCAF_DIR}/results/extended/routed_scaffold_strong.json")
  J4=$(submit_scaf_eval "scaf9d-del9" "--delete_shard 9" "${SCAF_DIR}/results/extended/routed_scaffold_strong_del9.json")
  echo "scaffold: full=${J3} del9=${J4}"
fi
