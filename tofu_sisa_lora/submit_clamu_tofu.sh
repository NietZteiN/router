#!/bin/bash
# ClAMU-on-TOFU pipeline driver (config-driven; full-FT, T=200, K clusters, optimized masks).
#   bash submit_clamu_tofu.sh <config.json> [all|setup|build|localize|unlearn|localize_tag|eval|collect]
# Linear chain: setup -> build -> localize(full) -> unlearn -> localize(--tag) -> eval(%2) -> collect.
#   setup     : MiniLM author embeddings + k-means into K clusters (assignment_K{K}.json)
#   build     : full-FT all authors, stream tau_bar.pt + per-cluster sums (no sign constraint)
#   localize  : optimize each cluster mask (STE) + derive EMR/TALL baselines  -> masks/
#   unlearn   : subtract forget tasks -> tau_bar_<tag>; re-cluster retain authors -> assignment_<tag>
#   localize_tag : rebuild masks on retain data -> masks_<tag>/
#   eval      : OU model_utility/forget_quality per label via eval_tofu --clamu_config
# Eval reuses the SISA retain90 oracle as the forget_quality KS reference. STUB=1 prints only.
#
# Pre-req (run once, CPU): python test_clamu.py  (the exactness gate).
set -euo pipefail

CONFIG="${1:?config json required}"
PHASE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
TAG="${CLAMU_TAG:-forget10}"
EVAL_ARGS="${CLAMU_EVAL_ARGS:---smoke}"          # --smoke (default) or --extended
EVAL_SUB="${CLAMU_EVAL_SUB:-smoke}"              # results/<sub> for KS ref + outputs
EVAL_CAP="${CLAMU_EVAL_CAP:-2}"                  # eval array %cap
MEM="${CLAMU_MEM:-64G}"
TIME_BUILD="${CLAMU_TIME_BUILD:-04:00:00}"
TIME_LOC="${CLAMU_TIME_LOC:-04:00:00}"           # mask optimization (per-cluster STE, serial)
TIME_EVAL="${CLAMU_TIME_EVAL:-02:30:00}"

read -r MODEL OUTPUT_DIR SLUG < <("${PYTHON}" -c "import json,sys; sys.path.insert(0,'${SCRIPT_DIR}'); from model_paths import model_slug; c=json.load(open('${CONFIG}')); print(c['model_name'], c['output_dir'], model_slug(c['model_name']))")
SISA_DIR="$(dirname "${OUTPUT_DIR}")/${SLUG}"          # source of the retain90 KS oracle
RES_DIR="${OUTPUT_DIR}/results/${EVAL_SUB}"
LOG_DIR="${OUTPUT_DIR}/logs"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
mkdir -p "${LOG_DIR}"

# The localization ladder + conditions. Override with CLAMU_LABELS="clamu_full merge_full clamu_unlearn"
# for a lean MVP. _full needs full masks; _unlearn needs the post-deletion masks_<tag>.
read -r -a LABELS <<< "${CLAMU_LABELS:-clamu_full merge_full emr_full tall_full clamu_unlearn merge_unlearn}"

echo "ClAMU-TOFU: config=${CONFIG} model=${MODEL} out=${OUTPUT_DIR}"
echo "  phase=${PHASE} tag=${TAG} eval='${EVAL_ARGS}' (sub=${EVAL_SUB}) KS<-${SISA_DIR}/results/${EVAL_SUB}/retain_tr_scores.npy"

run_sbatch() {  # sbatch script on stdin; honors STUB. Only the job id goes to stdout.
  if [ "${STUB:-0}" = "1" ]; then cat >&2; echo "----(STUB: not submitted)----" >&2; echo "STUB"; else sbatch --parsable; fi
}
dep() { [ -n "${1:-}" ] && [ "${1}" != "STUB" ] && echo "#SBATCH --dependency=afterok:${1}" || echo ""; }

gpu_job() {  # name, time, dep_id, cmd  -> submits a 1-GPU job, echoes its id
  local name="$1" tlimit="$2" depid="$3" cmd="$4"
  run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=clamu-${name}-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${tlimit}
$(dep "${depid}")
#SBATCH --output=${LOG_DIR}/${name}_%j.log
#SBATCH --error=${LOG_DIR}/${name}_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${cmd}
EOF
}

SETUP_ID=""; BUILD_ID=""; LOCF_ID=""; UNLEARN_ID=""; LOCT_ID=""; EVAL_ID=""
TC="${PYTHON} ${SCRIPT_DIR}/train_clamu.py"

# ── setup: cluster the authors ───────────────────────────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "setup" ]; then
  SETUP_ID=$(gpu_job setup "00:30:00" "" "${TC} setup --config ${CONFIG}")
  echo "  setup job: ${SETUP_ID}"
fi

# ── build: FT all authors -> tau_bar + per-cluster sums ──────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "build" ]; then
  BUILD_ID=$(gpu_job build "${TIME_BUILD}" "${SETUP_ID}" "${TC} build --config ${CONFIG}")
  echo "  build job: ${BUILD_ID}"
fi

# ── localize (full): optimize cluster masks + EMR/TALL baselines ─────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "localize" ]; then
  LOCF_ID=$(gpu_job localize "${TIME_LOC}" "${BUILD_ID}" "${TC} localize --config ${CONFIG}")
  echo "  localize(full) job: ${LOCF_ID}"
fi

# ── unlearn: subtract forget tasks + re-cluster retain ───────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "unlearn" ]; then
  UNLEARN_ID=$(gpu_job unlearn "01:00:00" "${LOCF_ID:-${BUILD_ID}}" "${TC} unlearn --config ${CONFIG} --tag ${TAG}")
  echo "  unlearn job: ${UNLEARN_ID}"
fi

# ── localize (--tag): rebuild masks on retain data ───────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "localize_tag" ]; then
  LOCT_ID=$(gpu_job localizetag "${TIME_LOC}" "${UNLEARN_ID}" "${TC} localize --config ${CONFIG} --tag ${TAG}")
  echo "  localize(${TAG}) job: ${LOCT_ID}"
fi

# ── eval: OU metrics per label via eval_tofu --clamu_config ──────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "eval" ]; then
NL=${#LABELS[@]}
EVAL_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=clamu-eval-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_EVAL}
#SBATCH --array=0-$((NL-1))%${EVAL_CAP}
$(dep "${LOCT_ID:-${LOCF_ID:-${BUILD_ID}}}")
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LABELS=(${LABELS[@]})
LABEL=\${LABELS[\${SLURM_ARRAY_TASK_ID}]}
mkdir -p "${RES_DIR}"
# Reuse the SISA retain90 forget_quality KS reference (method-independent). COPY (not
# symlink) so the relative target resolves and the parallel array is race-safe.
KSREF="${SISA_DIR}/results/${EVAL_SUB}/retain_tr_scores.npy"
[ -e "\${KSREF}" ] && cp -f "\${KSREF}" "${RES_DIR}/retain_tr_scores.npy" || \
  echo "[eval] WARN: no KS ref at \${KSREF} -> forget_quality will be NaN"
TAGFLAG=""
case "\${LABEL}" in *_unlearn) TAGFLAG="--clamu_unlearn_tag ${TAG}";; esac
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \
  --model_name "${MODEL}" --output_dir "${OUTPUT_DIR}" \
  --clamu_config "${CONFIG}" --label "\${LABEL}" \${TAGFLAG} \
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
#SBATCH --job-name=clamu-collect-${SLUG}
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
