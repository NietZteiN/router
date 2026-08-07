#!/bin/bash
# E5 — the trivial reroute-only "unlearning method", scored on TOFU.
# (log/selector_audit/2026-08-07_e1-router-probe-and-preregistration.md, pre-registration §E5.)
#
# The question: can TOFU's forget metric tell "the source is gone" apart from "a stranger answers
# for it"? Four arms on the k=200 e25 pool, all scoring the SAME 400-question forget10 split via
# --forget_author_ids 180-199 (without it, --forget_shard_id at k=200 measures one author's 20
# questions and nothing is comparable to the published numbers):
#
#   full      no deletion — the pre-deletion reference
#   delete    the 20 authors' experts dropped; their queries serve base+scaffold. This is what a
#             weight-absent model produces, and it is the control every other arm is read against.
#   reroute0  NOTHING deleted. The 20 authors' queries are answered by shard_0. If this scores
#             like `delete`, the metric is blind to the substitution — which is the result.
#   reroute42 the same with shard_42, so the number is not a property of shard 0.
#
# Usage: bash submit_e5_reroute.sh [eval|mia|all]        # STUB=1 previews without submitting
#   eval  4-arm GPU array (PACK arms per job, one per allocated GPU)
#   mia   CPU-only composed-model MIA over the arms' cached scores (the privacy column)
#   all   eval, then mia chained --dependency=afterany
#
# Every eval arm asserts, before any metric is read, that the served route matches the requested
# policy (eval_routed_scaffold's route audit: 400 orphan queries on the deletion path, no more and
# no fewer). A plausible-but-wrong route is the failure mode these arms are most exposed to and
# no metric would flag it.
set -euo pipefail

STAGE="${1:-all}"
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
PACK="${PACK:-1}"
TOFU_GPUS_PER_NODE="${TOFU_GPUS_PER_NODE:-4}"
if [ "${PACK}" -gt "${TOFU_GPUS_PER_NODE}" ]; then
  echo "submit_e5_reroute: PACK=${PACK} exceeds TOFU_GPUS_PER_NODE=${TOFU_GPUS_PER_NODE}." >&2
  echo "  The packed dispatcher pins CUDA_VISIBLE_DEVICES within a single node and cannot reach" >&2
  echo "  GPUs on a second one. Use PACK<=${TOFU_GPUS_PER_NODE} and more concurrent jobs." >&2
  exit 1
fi
NARMS=4
NJOBS=$(( (NARMS + PACK - 1) / PACK ))
LOG_DIR="${CKPT}/e5_reroute_logs"
RES="${E25}/results/smoke"
mkdir -p "${LOG_DIR}" "${RES}"
if [ ! -f "${RES}/retain_tr_scores.npy" ]; then
  echo "missing ${RES}/retain_tr_scores.npy — forget_quality would be NaN." >&2
  echo "  It is the recipe-independent KS reference; submit_k200_routed.sh copies it from the" >&2
  echo "  e5 pool. Do that first rather than running four arms that cannot report fq." >&2
  exit 1
fi

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

eval_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=e5-reroute
#SBATCH --array=0-$((NJOBS-1))%${ARRAY_CAP}
$(tofu_sbatch_resources ${PACK} $((8 * PACK)) 48G)
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
set -eo pipefail
ARMS=(full delete reroute0 reroute42)
PACK=${PACK}

run_arm() {
  local T=\$1 SLOT=\$2
  local ARM=\${ARMS[\$T]}
  # One log per arm: packed arms run concurrently and would otherwise interleave unreadably.
  if [ "\${PACK}" -gt 1 ]; then
    exec > "${LOG_DIR}/eval_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}_arm\${T}.log" 2>&1
  fi
  export CUDA_VISIBLE_DEVICES=\${SLOT}
  # eval_tofu._rouge_metric_cache falls back to <HF_HOME>/metrics_cache/<JOB_ID>, the SAME path
  # for every process in one job — packed arms would clobber each other's .arrow file.
  export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}_\${SLOT}"
  mkdir -p "\${TOFU_METRICS_CACHE}"
  echo "=== E5 arm \${T} (gpu slot \${SLOT}): \${ARM} ==="
  date
  export PYTHONUNBUFFERED=1
  export HF_HOME="${HF_HOME}"
  if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
  export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

  # The pool must be complete. A missing shard would silently route an author to the base and
  # look like a successful deletion.
  for i in \$(seq 0 199); do
    [ -f "${E25}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING shard_\${i}"; return 1; }
  done

  local COMMON=(--model_name "${MODEL}" --shards_dir "${E25}" --k 200 --forget_shard_id 199 \\
                --forget_author_ids "${FORGET}" --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}")
  case "\${ARM}" in
  full)
    OUT="${RES}/routed_oracle_full_f10.json"
    [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
    ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" "\${COMMON[@]}" --out "\${OUT}"
    ;;
  delete)
    OUT="${RES}/routed_oracle_del_f10.json"
    [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
    ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" "\${COMMON[@]}" \\
      --delete_shards "${FORGET}" --out "\${OUT}"
    ;;
  reroute0)
    OUT="${RES}/routed_reroute_f10_s0.json"
    [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
    ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" "\${COMMON[@]}" \\
      --delete_shards "${FORGET}" --reroute_to 0 --out "\${OUT}"
    ;;
  reroute42)
    OUT="${RES}/routed_reroute_f10_s42.json"
    [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
    ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" "\${COMMON[@]}" \\
      --delete_shards "${FORGET}" --reroute_to 42 --out "\${OUT}"
    ;;
  *) echo "unknown arm \${ARM}"; return 1 ;;
  esac
  date
}

FIRST=\$(( \${SLURM_ARRAY_TASK_ID} * PACK ))
rc=0
pids=(); slots=()
for s in \$(seq 0 \$((PACK - 1))); do
  T=\$(( FIRST + s ))
  [ "\${T}" -lt \${#ARMS[@]} ] || break
  run_arm "\${T}" "\${s}" &
  pids+=("\$!"); slots+=("\${T}")
done
for i in \$(seq 0 \$(( \${#pids[@]} - 1 ))); do
  if ! wait "\${pids[\$i]}"; then echo "ARM \${slots[\$i]} FAILED"; rc=1; fi
done
exit \${rc}
EOF
}

mia_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=e5-mia
#SBATCH --array=0-2%${ARRAY_CAP}
$(tofu_sbatch_resources 1 8 48G)
#SBATCH --time=03:00:00
#SBATCH --output=${LOG_DIR}/mia_%A_%a.log
#SBATCH --error=${LOG_DIR}/mia_%A_%a.log
set -eo pipefail
# Privacy column: composed-model MIA on the SAME served compositions the eval arms scored.
# member = forget10 (the 20 deleted authors), non-member = holdout10. Genuine deletion should
# fall to the ~0.5 oracle floor; a reroute that only changes who answers has no reason to.
ARMS=(delete reroute0 reroute42)
T=\${SLURM_ARRAY_TASK_ID}
ARM=\${ARMS[\$T]}
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
mkdir -p "${E25}/results/mia"
echo "=== E5 MIA arm \${ARM} ==="
date
COMMON=(--model_name "${MODEL}" --output_dir "${E25}" --shards_dir "${E25}" --k 200 \\
        --forget_shard_id 199 --delete_shards "${FORGET}" --lazy_adapter_cache 8 \\
        --hf_home "${HF_HOME}")
case "\${ARM}" in
delete)    LBL=mia_oracle_del_f10;      EXTRA=() ;;
reroute0)  LBL=mia_reroute_f10_s0;      EXTRA=(--reroute_to 0) ;;
reroute42) LBL=mia_reroute_f10_s42;     EXTRA=(--reroute_to 42) ;;
*) echo "unknown arm \${ARM}"; exit 1 ;;
esac
OUT="${E25}/results/mia/\${LBL}.json"
[ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
${PYTHON} "${SCRIPT_DIR}/attack_mia.py" "\${COMMON[@]}" "\${EXTRA[@]}" \\
  --label "\${LBL}" --out "\${OUT}"
date
EOF
}

case "${STAGE}" in
eval)
  echo "E5 eval: ${NARMS} arms in ${NJOBS} job(s) of ${PACK} GPU(s), cap ${ARRAY_CAP}"
  submit "$(eval_body)" "${DEP:-}"
  ;;
mia)
  echo "E5 privacy column: 3 GPU arms (delete / reroute0 / reroute42), cap ${ARRAY_CAP}"
  submit "$(mia_body)" "${DEP:-}"
  ;;
all)
  echo "E5 chain: ${DEP:+afterany:${DEP} -> }eval %${ARRAY_CAP} -> mia (afterany)"
  EVAL_ID="$(submit "$(eval_body)" "${DEP:-}")"
  echo "eval job: ${EVAL_ID}${DEP:+ (afterany:${DEP})}"
  if [ "${EVAL_ID}" = "STUB" ]; then
    submit "$(mia_body)"
  else
    MIA_ID="$(submit "$(mia_body)" "${EVAL_ID}")"
    echo "mia job:  ${MIA_ID} (afterany:${EVAL_ID})"
  fi
  ;;
*) echo "usage: bash submit_e5_reroute.sh [eval|mia|all]  (STUB=1 previews)"; exit 1 ;;
esac
