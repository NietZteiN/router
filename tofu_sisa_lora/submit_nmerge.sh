#!/bin/bash
# N-merge interference sweep driver (merge_mechanism Exp 5).
# Usage: bash submit_nmerge.sh CONFIG [plan|merge|eval|overlap|collect|all]
#
#   plan     (login, light) write merge_manifest.txt + eval_manifest_nmerge.txt
#   merge    CPU array (no GPU): materialize each subset merge as a PEFT adapter dir
#   eval     GPU array: eval_tofu --preloaded_adapter (or eval_baseline for BASE rows),
#            one task per eval-manifest line, self-skips existing result JSONs
#   overlap  1 CPU task: per-N subset geometry stats -> reports/nmerge_overlap_s{seed}.json
#   collect  (login, light) collect_results.py --smoke + analyze_nmerge.py
#   all      plan -> merge -> eval (afterok on merge) -> overlap
#
# STUB=1 prints every sbatch script without submitting. Env overrides:
#   ARRAY_CAP (default TOFU_ARRAY_CAP=4, the global GPU cap) | EVAL_TIME (01:30:00: rank-1024+ adapters
#   slow the forward) | MERGE_TIME (12:00:00) | MERGE_MEM (160G: N=200 r32 factors
#   ~52 GB fp32 + QR/SVD workspace; dare cross-check adds the 7B base on CPU) |
#   MERGE_ARRAY / EVAL_ARRAY (explicit sbatch --array spec, e.g. EVAL_ARRAY=0 for one task)
set -euo pipefail

CONFIG="$(readlink -f "${1:?config path required}")"
STAGE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${PYTHON:-${TOFU_PYTHON}}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"
EVAL_TIME="${EVAL_TIME:-01:30:00}"
MERGE_TIME="${MERGE_TIME:-12:00:00}"
MERGE_MEM="${MERGE_MEM:-160G}"

# Config reads happen at SUBMIT time, so they use a locally-present interpreter rather than the
# target site's ${PYTHON} — that is what lets `TOFU_SITE=cispa STUB=1 ...` validate another
# cluster's job scripts from here, before anything is submitted.
LOCAL_PY="${TOFU_LOCAL_PYTHON:-$(command -v python3 || command -v python)}"
[ -x "${LOCAL_PY}" ] || LOCAL_PY="${PYTHON}"
read -r OUT_DIR MODEL_NAME K FID CAP <<< "$("${LOCAL_PY}" - "${CONFIG}" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg["out_dir"], cfg["model_name"], cfg["eval"]["k"],
      cfg["eval"]["forget_shard_id"], cfg["eval"]["cap"])
EOF
)"
RESULTS_DIR="${OUT_DIR}/results/${CAP}"
LOG_DIR="${OUT_DIR}/logs"
MERGE_MANIFEST="${OUT_DIR}/merge_manifest.txt"
# EVAL_MANIFEST env override lets ad-hoc rows (custom manifests) reuse the eval stage.
EVAL_MANIFEST="${EVAL_MANIFEST:-${OUT_DIR}/eval_manifest_nmerge.txt}"
CAP_FLAG="--${CAP}"

submit() {  # submit <script-text> ; honors STUB=1
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----"
    printf '%s\n' "$1"
    echo "-----------------------------------------------"
    LAST_JOB=""
  else
    LAST_JOB="$(printf '%s\n' "$1" | sbatch --parsable)"
    echo "submitted job ${LAST_JOB}"
  fi
}

do_plan() {
  "${PYTHON}" "${SCRIPT_DIR}/merge_subset.py" plan --config "${CONFIG}"
}

do_merge() {
  [ -f "${MERGE_MANIFEST}" ] || { echo "missing ${MERGE_MANIFEST} (run plan)"; exit 1; }
  local n_tasks; n_tasks=$(wc -l < "${MERGE_MANIFEST}")
  # CPU-only array (no --gres), so this throttle is politeness on shared CPUs, not the GPU cap.
  # Was a literal %4 — every other array here honours ARRAY_CAP, so this one now does too.
  local spec="${MERGE_ARRAY:-0-$((n_tasks - 1))%${ARRAY_CAP}}"
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-nmerge-merge
#SBATCH --array=${spec}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --mem=${MERGE_MEM}
#SBATCH --cpus-per-task=32
#SBATCH --time=${MERGE_TIME}
#SBATCH --output=${LOG_DIR}/merge_%A_%a.log
#SBATCH --error=${LOG_DIR}/merge_%A_%a.log

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MERGE_MANIFEST}")
METHOD=\$(printf '%s' "\${LINE}" | cut -f1)
N=\$(printf '%s' "\${LINE}" | cut -f2)
SEED=\$(printf '%s' "\${LINE}" | cut -f3)
SVD=\$(printf '%s' "\${LINE}" | cut -f4)
RHO=\$(printf '%s' "\${LINE}" | cut -f5)
LAM=\$(printf '%s' "\${LINE}" | cut -f6)
echo "=== nmerge merge task \${SLURM_ARRAY_TASK_ID}: \${METHOD} N=\${N} seed=\${SEED} svd=\${SVD} rho=\${RHO:--} lam=\${LAM:--} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export NMERGE_THREADS=32
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

if [ "\${METHOD}" = "cross_check" ]; then
  ${PYTHON} "${SCRIPT_DIR}/merge_subset.py" merge --config "${CONFIG}" --cross_check
else
  EXTRA=""
  if [ "\${SVD}" != "-" ]; then EXTRA="--svd_rank \${SVD}"; fi
  # 5th manifest column (rho) is absent in pre-2026-07-15 manifests -> empty -> skipped
  if [ -n "\${RHO}" ] && [ "\${RHO}" != "-" ]; then EXTRA="\${EXTRA} --rho \${RHO}"; fi
  # 6th column (lam, additive_sum global coefficient) absent in pre-2026-07-28 manifests -> skipped
  if [ -n "\${LAM}" ] && [ "\${LAM}" != "-" ]; then EXTRA="\${EXTRA} --lam \${LAM}"; fi
  ${PYTHON} "${SCRIPT_DIR}/merge_subset.py" merge --config "${CONFIG}" \\
    --method "\${METHOD}" --n "\${N}" --seed "\${SEED}" \${EXTRA}
fi
date
EOF
  echo "merge array: ${n_tasks} tasks (spec ${spec}), mem ${MERGE_MEM}, time ${MERGE_TIME}"
  submit "${S}"
  MERGE_JOB="${LAST_JOB:-}"
}

do_eval() {
  [ -f "${EVAL_MANIFEST}" ] || { echo "missing ${EVAL_MANIFEST} (run plan)"; exit 1; }
  local dep="${1:-}"
  local n_tasks; n_tasks=$(wc -l < "${EVAL_MANIFEST}")
  local spec="${EVAL_ARRAY:-0-$((n_tasks - 1))%${ARRAY_CAP}}"
  local dep_line=""
  [ -n "${dep}" ] && dep_line="#SBATCH --dependency=afterok:${dep}"
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-nmerge-eval
#SBATCH --array=${spec}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${EVAL_TIME}
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
${dep_line}

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${EVAL_MANIFEST}")
LABEL=\$(printf '%s' "\${LINE}" | cut -f1)
ADAPTER=\$(printf '%s' "\${LINE}" | cut -f2)
SID=\$(printf '%s' "\${LINE}" | cut -f3)
RIDS=\$(printf '%s' "\${LINE}" | cut -f4)
RIDS="\${RIDS:--}"

if [ "\${SID}" = "-" ]; then
  OUT_JSON="${RESULTS_DIR}/\${LABEL}.json"
  SID_ARGS=""
  BASE_FID="${FID}"
else
  OUT_JSON="${RESULTS_DIR}/\${LABEL}__own\${SID}.json"
  SID_ARGS="--eval_shard_id \${SID}"
  BASE_FID="\${SID}"
fi
RID_ARGS=""
if [ "\${RIDS}" != "-" ]; then
  # subset-conditioned utility row: retain_* restricted to the merged authors
  OUT_JSON="${RESULTS_DIR}/\${LABEL}__subset.json"
  RID_ARGS="--retain_author_ids \${RIDS}"
fi
if [ -f "\${OUT_JSON}" ]; then echo "Skip existing \${OUT_JSON}"; exit 0; fi

echo "=== nmerge eval task \${SLURM_ARRAY_TASK_ID}: \${LABEL} sid=\${SID} rids=\${RIDS:0:40} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

if [ "\${ADAPTER}" = "BASE" ]; then
  # eval_baseline has no --eval_shard_id; --forget_shard_id \${BASE_FID} yields the same
  # measure/retain split (measure_id == forget_shard_id when eval_shard_id is unset).
  ${PYTHON} "${SCRIPT_DIR}/eval_baseline.py" \\
    --model_name "${MODEL_NAME}" \\
    --output_dir "${OUT_DIR}" \\
    --k ${K} \\
    --forget_shard_id "\${BASE_FID}" \\
    --out "\${OUT_JSON}" \\
    --hf_home "${HF_HOME}" \\
    \${RID_ARGS} \\
    ${CAP_FLAG}
else
  [ -f "\${ADAPTER}/adapter_model.safetensors" ] || { echo "missing adapter \${ADAPTER}"; exit 1; }
  ${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
    --model_name "${MODEL_NAME}" \\
    --output_dir "${OUT_DIR}" \\
    --label "\${LABEL}" \\
    --k ${K} \\
    --forget_shard_id ${FID} \\
    \${SID_ARGS} \\
    \${RID_ARGS} \\
    --preloaded_adapter "\${ADAPTER}" \\
    --out "\${OUT_JSON}" \\
    --hf_home "${HF_HOME}" \\
    ${CAP_FLAG}
fi
date
EOF
  echo "eval array: ${n_tasks} tasks (spec ${spec}), cap ${ARRAY_CAP}, time ${EVAL_TIME}, exclude ${TOFU_EXCLUDE}"
  submit "${S}"
}

do_overlap() {
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-nmerge-overlap
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/overlap_%j.log
#SBATCH --error=${LOG_DIR}/overlap_%j.log
export PYTHONUNBUFFERED=1
export NMERGE_THREADS=16
export SUBSPACE_THREADS=16
date
${PYTHON} "${SCRIPT_DIR}/merge_subset.py" overlap --config "${CONFIG}"
date
EOF
  echo "overlap: 1 CPU task (96G, 16 threads)"
  submit "${S}"
}

do_norms() {
  # CPU-only delta-norm ladder (APA study, 2026-07-28): ||sum||_F, kappa, rel_pert vs N.
  # No GPU and no materialization — reads the pool safetensors directly, so it can run
  # concurrently with the GPU arrays without touching the 4-GPU cap.
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-nmerge-norms
$(tofu_sbatch_resources 0 16 96G)
#SBATCH --time=${NORMS_TIME:-06:00:00}
#SBATCH --output=${LOG_DIR}/norms_%j.log
#SBATCH --error=${LOG_DIR}/norms_%j.log
export PYTHONUNBUFFERED=1
export NMERGE_THREADS=16
export HF_HOME="${HF_HOME}"
date
${PYTHON} "${SCRIPT_DIR}/merge_subset.py" norms --config "${CONFIG}"
date
EOF
  echo "norms: 1 CPU task (96G, 16 threads, no gres)"
  submit "${S}"
}

do_collect() {
  "${PYTHON}" "${SCRIPT_DIR}/collect_results.py" --root "$(dirname "${OUT_DIR}")" --${CAP}
  # OUT_PREFIX guards the shared reports/nmerge_* prefix — pass a per-campaign prefix
  # (e.g. OUT_PREFIX=reports/centered/nmerge) or the default CLOBBERS the e5 CSVs.
  "${PYTHON}" "${SCRIPT_DIR}/analyze_nmerge.py" --config "${CONFIG}" \
    ${OUT_PREFIX:+--out_prefix "${OUT_PREFIX}"}
}

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
case "${STAGE}" in
  plan)    do_plan ;;
  merge)   do_merge ;;
  eval)    do_eval ;;
  overlap) do_overlap ;;
  norms)   do_norms ;;
  collect) do_collect ;;
  all)
    do_plan
    do_merge
    if [ -n "${MERGE_JOB:-}" ]; then
      echo "NOTE: eval depends afterok:${MERGE_JOB} — kill_invalid_depend is OFF cluster-wide;"
      echo "      if the merge array fails, scancel the pending eval array yourself."
      do_eval "${MERGE_JOB}"
    else
      do_eval
    fi
    do_overlap
    ;;
  *) echo "unknown stage ${STAGE}"; exit 1 ;;
esac
