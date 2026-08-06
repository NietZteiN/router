#!/bin/bash
# SIFT-Masks-on-TOFU pipeline driver (config-driven; faithful full-FT, T=200).
#   bash submit_sift_masks_tofu.sh <config.json> [all|build|unlearn|eval|collect]
# Chain: build(1 GPU) -> unlearn(1 GPU, dep) -> eval(%2, dep) -> collect(CPU, dep).
# build is a single serial GPU job (200 author-tasks x 20 steps, fp32); eval reuses the
# SISA retain90 oracle as the forget_quality KS reference. STUB=1 prints scripts only.
#
# Pre-req (run once, CPU): python test_sift_masks.py  (the exactness gate).
set -euo pipefail

CONFIG="${1:?config json required}"
PHASE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
TAG="${SIFT_TAG:-forget10}"
EVAL_ARGS="${SIFT_EVAL_ARGS:---smoke}"          # --smoke (default) or --extended
EVAL_SUB="${SIFT_EVAL_SUB:-smoke}"              # results/<sub> for the KS reference + outputs
EVAL_CAP="${SIFT_EVAL_CAP:-2}"                  # eval array %cap
MEM="${SIFT_MEM:-64G}"
TIME_BUILD="${SIFT_TIME_BUILD:-04:00:00}"
TIME_EVAL="${SIFT_TIME_EVAL:-02:30:00}"

read -r MODEL OUTPUT_DIR SLUG < <("${PYTHON}" -c "import json,sys; sys.path.insert(0,'${SCRIPT_DIR}'); from model_paths import model_slug; c=json.load(open('${CONFIG}')); print(c['model_name'], c['output_dir'], model_slug(c['model_name']))")
SISA_DIR="$(dirname "${OUTPUT_DIR}")/${SLUG}"          # source of the retain90 KS oracle
RES_DIR="${OUTPUT_DIR}/results/${EVAL_SUB}"
LOG_DIR="${OUTPUT_DIR}/logs"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
mkdir -p "${LOG_DIR}"

# The four eval rows: masked (the method) vs FT+Merge no-mask baseline, full vs unlearn.
# Override with SIFT_LABELS="sift_full sift_unlearn" to run a lean subset (e.g. a smoke).
read -r -a LABELS <<< "${SIFT_LABELS:-sift_full merge_full sift_unlearn merge_unlearn}"

echo "SIFT-Masks-TOFU: config=${CONFIG} model=${MODEL} out=${OUTPUT_DIR}"
echo "  phase=${PHASE} tag=${TAG} eval='${EVAL_ARGS}' (sub=${EVAL_SUB}) KS<-${SISA_DIR}/results/${EVAL_SUB}/retain_tr_scores.npy"

run_sbatch() {  # sbatch script on stdin; honors STUB. Only the job id goes to stdout.
  if [ "${STUB:-0}" = "1" ]; then cat >&2; echo "----(STUB: not submitted)----" >&2; echo "STUB"; else sbatch --parsable; fi
}
dep() { [ -n "${1:-}" ] && [ "${1}" != "STUB" ] && echo "#SBATCH --dependency=afterok:${1}" || echo ""; }

BUILD_ID=""; UNLEARN_ID=""; EVAL_ID=""

# ── build: SIFT-train all T author-tasks -> tau_bar.pt + masks ────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "build" ]; then
BUILD_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sift-build-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_BUILD}
#SBATCH --output=${LOG_DIR}/build_%j.log
#SBATCH --error=${LOG_DIR}/build_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}" CUBLAS_WORKSPACE_CONFIG=:4096:8
${PYTHON} "${SCRIPT_DIR}/train_sift_masks.py" build --config "${CONFIG}"
EOF
)
echo "  build job: ${BUILD_ID}"
fi

# ── unlearn: deterministically re-derive forget tasks, subtract -> tau_bar_<tag> ──
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "unlearn" ]; then
UNLEARN_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sift-unlearn-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
$(dep "${BUILD_ID}")
#SBATCH --output=${LOG_DIR}/unlearn_%j.log
#SBATCH --error=${LOG_DIR}/unlearn_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}" CUBLAS_WORKSPACE_CONFIG=:4096:8
${PYTHON} "${SCRIPT_DIR}/train_sift_masks.py" unlearn --config "${CONFIG}" --tag "${TAG}"
EOF
)
echo "  unlearn job: ${UNLEARN_ID}"
fi

# ── eval: OU metrics per label via eval_tofu --sift_masks_config ──────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "eval" ]; then
NL=${#LABELS[@]}
EVAL_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sift-eval-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_EVAL}
#SBATCH --array=0-$((NL-1))%${EVAL_CAP}
$(dep "${UNLEARN_ID:-${BUILD_ID}}")
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}" CUBLAS_WORKSPACE_CONFIG=:4096:8
LABELS=(${LABELS[@]})
LABEL=\${LABELS[\${SLURM_ARRAY_TASK_ID}]}
mkdir -p "${RES_DIR}"
# Reuse the SISA retain90 forget_quality KS reference (method-independent). COPY (not
# symlink): a relative-target symlink resolves from the link's own dir and breaks ->
# np.load fails -> forget_quality=NaN; cp -f also makes this idempotent/race-safe across
# the parallel eval array (identical 368-byte content).
KSREF="${SISA_DIR}/results/${EVAL_SUB}/retain_tr_scores.npy"
[ -e "\${KSREF}" ] && cp -f "\${KSREF}" "${RES_DIR}/retain_tr_scores.npy" || \
  echo "[eval] WARN: no KS ref at \${KSREF} -> forget_quality will be NaN"
TAGFLAG=""
case "\${LABEL}" in *_unlearn) TAGFLAG="--sift_unlearn_tag ${TAG}";; esac
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \
  --model_name "${MODEL}" --output_dir "${OUTPUT_DIR}" \
  --sift_masks_config "${CONFIG}" --label "\${LABEL}" \${TAGFLAG} \
  --k 10 --forget_shard_id 9 \
  --out "${RES_DIR}/\${LABEL}.json" ${EVAL_ARGS}
EOF
)
echo "  eval array: ${EVAL_ID} (labels: ${LABELS[*]})"
fi

# ── collect: merge result JSONs into the CSV ─────────────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "collect" ]; then
COLLECT_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sift-collect-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
$(dep "${EVAL_ID}")
#SBATCH --output=${LOG_DIR}/collect_%j.log
#SBATCH --error=${LOG_DIR}/collect_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "$(dirname "${OUTPUT_DIR}")" --${EVAL_SUB}
EOF
)
echo "  collect job: ${COLLECT_ID}"
fi

echo "Done submitting (phase=${PHASE})."
