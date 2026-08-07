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
# PACK = eval arms per job, one per allocated GPU (default 1 = one arm per 1-GPU task, the
# historical behaviour, byte-identical). Raise it when the scheduler limits RUNNING JOBS more
# tightly than GPUs: CISPA allows gres/gpu=16 per user but MaxJobs=6, so PACK=1 strands 10 GPUs.
# `sacctmgr show assoc user=$USER format=GrpTRES,MaxTRES,MaxJobs` is where those numbers live.
PACK="${PACK:-1}"
# PACK cannot exceed the GPUs on ONE node. The batch script runs on the FIRST node of the
# allocation only, so --gres=gpu:8 on 4-GPU nodes gets you two nodes, four unreachable GPUs and
# a job that still runs four arms — half the allocation burned. Driving the second node needs
# srun per task, which this does not do. xe8545 is A100:4.
TOFU_GPUS_PER_NODE="${TOFU_GPUS_PER_NODE:-4}"
if [ "${PACK}" -gt "${TOFU_GPUS_PER_NODE}" ]; then
  echo "submit_k200_routed: PACK=${PACK} exceeds TOFU_GPUS_PER_NODE=${TOFU_GPUS_PER_NODE}." >&2
  echo "  The packed dispatcher pins CUDA_VISIBLE_DEVICES within a single node and cannot reach" >&2
  echo "  GPUs on a second one. Use PACK<=${TOFU_GPUS_PER_NODE} and more concurrent jobs, or set" >&2
  echo "  TOFU_GPUS_PER_NODE if your nodes are larger." >&2
  exit 1
fi
NEVAL_ARMS=8
NEVAL_JOBS=$(( (NEVAL_ARMS + PACK - 1) / PACK ))
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
#SBATCH --array=0-$((NEVAL_JOBS-1))%${ARRAY_CAP}
$(tofu_sbatch_resources ${PACK} $((8 * PACK)) 48G)
#SBATCH --time=03:30:00
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
set -eo pipefail
POOLS=("${E5}" "${E5}" "${E5}" "${E5}" "${E25}" "${E25}" "${E25}" "${E25}")
ARMS=(oracle_full oracle_del199 key_exact key_no199 oracle_full oracle_del199 key_exact key_no199)
PACK=${PACK}
# PACK>1 packs PACK arms into ONE job, one per allocated GPU. The association allows 16
# concurrent GPUs but only MaxJobs=6 RUNNING JOBS, so 1-GPU-per-array-task tops out at 6 GPUs
# with 10 idle. Packing converts the job-count limit into the GPU-count limit.
# Each slot gets its own TOFU_METRICS_CACHE: eval_tofu._rouge_metric_cache falls back to
# <HF_HOME>/metrics_cache/<SLURM_JOB_ID>, which is the SAME path for every process in one job,
# and they would clobber each other's .arrow file (eval_tofu.py:52-59).
run_arm() {
  local T=\$1 SLOT=\$2
  local DIR=\${POOLS[\$T]} ARM=\${ARMS[\$T]}
  # One log per arm. Packed arms run concurrently, so without this their progress lines
  # interleave in the job log and none of them can be read or timed. Runs in a background
  # subshell, so this exec redirects only this arm.
  if [ "\${PACK}" -gt 1 ]; then
    exec > "${LOG_DIR}/eval_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}_arm\${T}.log" 2>&1
  fi
  export CUDA_VISIBLE_DEVICES=\${SLOT}
  export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}_\${SLOT}"
  mkdir -p "\${TOFU_METRICS_CACHE}"
  echo "=== k200 routed eval arm \${T} (gpu slot \${SLOT}): \$(basename "\${DIR}") \${ARM} ==="
  date
# e25 arms: the pool MUST be complete (chained afterany — fail loudly, never silently
# route a missing author to the base).
if [ "\${DIR}" = "${E25}" ]; then
  for i in \$(seq 0 199); do
    [ -f "\${DIR}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING shard_\${i} — train incomplete"; return 1; }
  done
fi
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
case "\${ARM}" in
oracle_full)
  OUT="\${DIR}/results/smoke/routed_oracle_full.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
    --model_name "${MODEL}" --shards_dir "\${DIR}" --k 200 --forget_shard_id 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
oracle_del199)
  OUT="\${DIR}/results/smoke/routed_oracle_del199.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
    --model_name "${MODEL}" --shards_dir "\${DIR}" --k 200 --forget_shard_id 199 \\
    --delete_shard 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
key_exact)
  OUT="\${DIR}/results/smoke/routed_key_exact.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
    --model_name "${MODEL}" --output_dir "\${DIR}" --label routed_key_exact \\
    --k 200 --forget_shard_id 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
key_no199)
  OUT="\${DIR}/results/smoke/routed_key_exact_no199.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
    --model_name "${MODEL}" --output_dir "\${DIR}" --label routed_key_exact_no199 \\
    --k 200 --forget_shard_id 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
*) echo "unknown arm \${ARM}"; return 1 ;;
esac
date
}

FIRST=\$(( \${SLURM_ARRAY_TASK_ID} * PACK ))
rc=0
pids=()
slots=()
for s in \$(seq 0 \$((PACK - 1))); do
  T=\$(( FIRST + s ))
  [ "\${T}" -lt \${#ARMS[@]} ] || break
  run_arm "\${T}" "\${s}" &
  pids+=("\$!"); slots+=("\${T}")
done
for i in \$(seq 0 \$(( \${#pids[@]} - 1 ))); do
  if ! wait "\${pids[\$i]}"; then
    echo "ARM \${slots[\$i]} FAILED"; rc=1
  fi
done
exit \${rc}
EOF
}

case "${STAGE}" in
train)
  echo "k200 e25 train: 200 tasks (self-skip existing), cap ${ARRAY_CAP}${DEP:+, dependency afterany:${DEP}}"
  submit "$(train_body)" "${DEP:-}"
  ;;
eval)
  echo "k200 routed eval: ${NEVAL_ARMS} arms (4 x 2 pools) in ${NEVAL_JOBS} job(s) of ${PACK} GPU(s), cap ${ARRAY_CAP}${DEP:+, dependency afterany:${DEP}}"
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
