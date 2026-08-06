#!/bin/bash
# 7B routed-mu ladder fill — routed_key_exact on the k=4 / k=10 / k=20 Llama-2-7B pools
# (log/router_leak/2026-07-26_7b-orphan-coverage.md; CLAUDE_SCRATCHPAD 2026-07-26).
#
# WHY: Table H' (MERGE_VS_ROUTING_MASTER_2026-07-24.md) showed the 7B routing ladder had
# holes — routed_key_exact existed only at k=50 (0.7147), k=100 (0.6475) and k=200-r8
# (0.4728), while the 7B MERGE ladder (Table C) runs k=4/10/20/50/100/200. Without k=4/10/20
# the same-model merge-vs-route contrast can only be drawn at 3 of 6 granularities. The
# pools, their shards and their cached KS references already exist; only this eval is
# missing, so this is the cheapest cell in the whole table.
#
# Usage: bash submit_7b_routed_fill.sh [eval]        # STUB=1 previews without submitting
#
# ⚠ GPU BUDGET. This array is throttled to 3 concurrent tasks (%3). It is designed to run
#   ALONGSIDE `bash submit_router_family.sh sevenb` (3 more GPUs) for a total of 6 — the
#   ceiling the user authorized on 2026-07-26, ABOVE the CLAUDE.md §1 default of 4. The
#   shared cap variable TOFU_ARRAY_CAP is deliberately NOT read or raised here (it stays 4
#   for every other driver); this local %3 is below it either way. Check
#   `squeue -u jack -o "%.10i %.20j %.10T %.10b %F"` before submitting and confirm
#   (max concurrent GPUs already queued) + 3 <= 6.
#
# ⚠ IDEMPOTENCY (2026-07-24 lesson, CLAUDE_SCRATCHPAD): result-JSON absence is NOT enough
#   to prove nothing is running — a concurrent session once duplicated an entire array that
#   way. This script checks BOTH: `squeue` for a live job of the same name, and the per-task
#   --out file. Do not remove either check.
set -euo pipefail

STAGE="${1:-eval}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
CKPT="${TOFU_CKPT_ROOT}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
JOB_NAME="tofu-7b-routed-fill"
# Local throttle: 3 tasks, one per k. NOT TOFU_ARRAY_CAP (see the GPU-budget note above).
ARRAY_CAP=3
# k values to fill. Forget shard is always k-1 (TOFU forget-split alignment) and the pool
# suffix is the frozen-recipe one (r32/e5/lr1e4) these three pools were trained under.
K_LIST=(4 10 20)
LOG_DIR="${CKPT}/routed_fill_7b_logs"
mkdir -p "${LOG_DIR}"

# --- preflight: every pool must have its shards and a cached KS reference ---------------
# eval_tofu computes forget_quality by KS test against results/<tier>/retain_tr_scores.npy.
# Without it the metric is a silent NaN rather than an error, so assert it up front.
for K in "${K_LIST[@]}"; do
  POOL="${CKPT}/Llama-2-7B-chat-hf_k${K}_r32_e5_lr1e4"
  [ -d "${POOL}/shard_0" ] || { echo "FATAL: ${POOL} has no shard_0"; exit 1; }
  N_SHARDS=$(find "${POOL}" -maxdepth 1 -name 'shard_*' -type d | wc -l)
  [ "${N_SHARDS}" -eq "${K}" ] || { echo "FATAL: ${POOL} has ${N_SHARDS} shards, expected ${K}"; exit 1; }
  [ -f "${POOL}/results/smoke/retain_tr_scores.npy" ] || {
    echo "FATAL: ${POOL} lacks results/smoke/retain_tr_scores.npy — forget_quality would be a silent NaN."
    echo "       Fix: python prepare_eval.py --smoke --output_dir ${POOL} --model_name ${MODEL} --k ${K}"
    exit 1; }
done
echo "preflight ok: k=${K_LIST[*]} pools have shards + KS reference"

if [ "${STAGE}" != "eval" ]; then
  echo "usage: bash submit_7b_routed_fill.sh [eval]"; exit 1
fi

# --- idempotency: refuse to double-submit -----------------------------------------------
if [ "${STUB:-0}" != "1" ] && squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "${JOB_NAME}"; then
  echo "REFUSING: a job named ${JOB_NAME} is already RUNNING/PENDING in squeue."
  echo "          (result-JSON absence does not prove nothing is running — see header)"
  exit 1
fi

PENDING=()
for K in "${K_LIST[@]}"; do
  OUT="${CKPT}/Llama-2-7B-chat-hf_k${K}_r32_e5_lr1e4/results/smoke/routed_key_exact.json"
  if [ -f "${OUT}" ]; then echo "skip k=${K} (exists: ${OUT})"; else PENDING+=("${K}"); fi
done
[ ${#PENDING[@]} -eq 0 ] && { echo "nothing to do — all ${#K_LIST[@]} cells present"; exit 0; }
echo "will run k=${PENDING[*]} (${#PENDING[@]} tasks, %${ARRAY_CAP})"

BODY=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=0-$(( ${#PENDING[@]} - 1 ))%${ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/logs_%x_%A_%a.out
set -eo pipefail
export HF_HOME=${HF_HOME}
cd ${SCRIPT_DIR}

KS=(${PENDING[@]})
K=\${KS[\${SLURM_ARRAY_TASK_ID}]}
POOL=${CKPT}/Llama-2-7B-chat-hf_k\${K}_r32_e5_lr1e4
OUT=\${POOL}/results/smoke/routed_key_exact.json

# Per-task self-skip: the array may be resubmitted after a partial failure.
if [ -f "\${OUT}" ]; then echo "task k=\${K}: \${OUT} exists, skipping"; exit 0; fi

# forget shard = k-1 (TOFU forget-split alignment; see CLAUDE.md Key Design Invariants).
# --out is given EXPLICITLY: submit_eval.sh writes to results/ rather than results/smoke/,
# and the reproduce/ snapshot + cells.tsv both key on results/smoke/ (2026-07-24 lesson).
# NB single line on purpose: backslash-continuations do not survive this heredoc.
${PYTHON} eval_tofu.py --model_name ${MODEL} --output_dir \${POOL} --label routed_key_exact --k \${K} --forget_shard_id \$(( K - 1 )) --smoke --out \${OUT}

echo "task k=\${K}: wrote \${OUT}"
EOF
)

if [ "${STUB:-0}" = "1" ]; then
  echo "===== STUB ${JOB_NAME} (not submitted) ====="
  printf '%s\n' "${BODY}"
  echo "============================================"
  exit 0
fi

printf '%s\n' "${BODY}" | sbatch
