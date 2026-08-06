#!/bin/bash
# k=200 per-author task vectors + ORACLE routing driver
# (log/routing_scaffold/2026-07-19_k200-oracle-routing.md; CLAUDE_SCRATCHPAD 2026-07-19).
#
# Usage: bash submit_k200_routed.sh [train|eval|all]        # STUB=1 previews
#   train  200-task GPU array %4: complete the e25 per-author pool (self-skips the 20
#          existing perm42[:20] shards + any rerun) — frozen recipe + --epochs 25 --k 200
#          → Llama-2-7B-chat-hf_k200_r32_e25_lr1e4/shard_{0..199}.
#   eval   8-task GPU array %4 (smoke tier): oracle-routed (q2author OOD-aware,
#          eval_routed_scaffold on the PLAIN base) full + delete-author-199, and the
#          June-comparable lexical routed_key_exact[_no199] (eval_tofu), each on BOTH
#          pools (e5 = the complete weak pool, e25 = the strong pool). All arms use
#          --lazy_adapter_cache 8 (the k=200 r32 fp32 memory-wall fix; gate:
#          python test_lazy_adapters.py). Each e25 task asserts the pool is complete
#          first (eval is chained afterany, so a failed train task must fail HERE, loudly).
#   all    train, then eval chained --dependency=afterany:<train> (kill_invalid_depend is
#          off cluster-wide; afterok would hang pending forever on a single train failure).
# ⚠ Every stage is GPU — check `squeue -u jack` against the GLOBAL 4-GPU cap first.
#   Never queue this alongside another GPU array unless throttles still sum ≤ 4.
set -euo pipefail

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
E5="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4"
E25="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"
KS_REF="${E5}/results/smoke/retain_tr_scores.npy"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"
LOG_DIR="${CKPT}/k200_routed_logs"
mkdir -p "${LOG_DIR}" "${E25}/results/smoke"
# KS reference (forget_quality): recipe-independent, copy the e5 smoke one (the
# submit_anchor_pilot.sh convention). Never overwrite an existing file.
[ -f "${E25}/results/smoke/retain_tr_scores.npy" ] || cp "${KS_REF}" "${E25}/results/smoke/"

submit() {  # $1 = sbatch body; echoes the job id (or STUB marker)
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----" >&2
    printf '%s\n' "$1" >&2
    echo "-----------------------------------------------" >&2
    echo "STUB"
  else
    printf '%s\n' "$1" | sbatch --parsable ${2:+--dependency=afterany:$2}
  fi
}

train_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-k200tv-train
#SBATCH --array=0-199%${ARRAY_CAP}
$(tofu_sbatch_resources 1 8 48G)
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
set -eo pipefail
A=\${SLURM_ARRAY_TASK_ID}
if [ -f "${E25}/shard_\${A}/adapter_model.safetensors" ]; then
  echo "shard_\${A} already trained — skip"; exit 0
fi
echo "=== k200 e25 train author \${A} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" \\
  --shard_id "\${A}" --k 200 \\
  --model_name "${MODEL}" \\
  --output_dir "${E25}" \\
  --epochs 25 \\
  --hf_home "${HF_HOME}"
date
EOF
}

eval_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-k200tv-eval
#SBATCH --array=0-7%${ARRAY_CAP}
$(tofu_sbatch_resources 1 8 48G)
#SBATCH --time=03:30:00
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
set -eo pipefail
T=\${SLURM_ARRAY_TASK_ID}
POOLS=("${E5}" "${E5}" "${E5}" "${E5}" "${E25}" "${E25}" "${E25}" "${E25}")
ARMS=(oracle_full oracle_del199 key_exact key_no199 oracle_full oracle_del199 key_exact key_no199)
DIR=\${POOLS[\$T]}
ARM=\${ARMS[\$T]}
echo "=== k200 routed eval task \${T}: \$(basename "\${DIR}") \${ARM} ==="
date
# e25 arms: the pool MUST be complete (chained afterany — fail loudly, never silently
# route a missing author to the base).
if [ "\${DIR}" = "${E25}" ]; then
  for i in \$(seq 0 199); do
    [ -f "\${DIR}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING shard_\${i} — train incomplete"; exit 1; }
  done
fi
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
case "\${ARM}" in
oracle_full)
  OUT="\${DIR}/results/smoke/routed_oracle_full.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
    --model_name "${MODEL}" --shards_dir "\${DIR}" --k 200 --forget_shard_id 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
oracle_del199)
  OUT="\${DIR}/results/smoke/routed_oracle_del199.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
    --model_name "${MODEL}" --shards_dir "\${DIR}" --k 200 --forget_shard_id 199 \\
    --delete_shard 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
key_exact)
  OUT="\${DIR}/results/smoke/routed_key_exact.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
    --model_name "${MODEL}" --output_dir "\${DIR}" --label routed_key_exact \\
    --k 200 --forget_shard_id 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
key_no199)
  OUT="\${DIR}/results/smoke/routed_key_exact_no199.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
    --model_name "${MODEL}" --output_dir "\${DIR}" --label routed_key_exact_no199 \\
    --k 200 --forget_shard_id 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
*) echo "unknown arm \${ARM}"; exit 1 ;;
esac
date
EOF
}

case "${STAGE}" in
train)
  echo "k200 e25 train: 200 tasks (self-skip existing), cap ${ARRAY_CAP}${DEP:+, dependency afterany:${DEP}}"
  submit "$(train_body)" "${DEP:-}"
  ;;
eval)
  echo "k200 routed eval: 8 tasks (4 arms x 2 pools), cap ${ARRAY_CAP}${DEP:+, dependency afterany:${DEP}}"
  submit "$(eval_body)" "${DEP:-}"
  ;;
all)
  # DEP (colon-separated job ids) tail-chains the WHOLE campaign behind other queued GPU
  # work so co-queued throttles never sum past the global 4-GPU cap.
  echo "k200 chain: ${DEP:+afterany:${DEP} -> }train %${ARRAY_CAP} -> eval %${ARRAY_CAP} (afterany)"
  TRAIN_ID="$(submit "$(train_body)" "${DEP:-}")"
  echo "train job: ${TRAIN_ID}${DEP:+ (afterany:${DEP})}"
  if [ "${TRAIN_ID}" = "STUB" ]; then
    submit "$(eval_body)"
  else
    EVAL_ID="$(submit "$(eval_body)" "${TRAIN_ID}")"
    echo "eval job:  ${EVAL_ID} (afterany:${TRAIN_ID})"
  fi
  ;;
*) echo "usage: bash submit_k200_routed.sh [train|eval|all]  (STUB=1 previews)"; exit 1 ;;
esac
