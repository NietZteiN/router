#!/bin/bash
# SIFT-Masks follow-up runs (reuse the existing T=200 build artifacts):
#   bash submit_sift_followups.sh <config.json> [all|extended|ansprob|exact]
# Jobs (all reuse checkpoints/<slug>_sift_masks/sift/):
#   P  prep-ext : prepare_eval.py --extended -> builds the extended retain90 KS ref in the SISA dir
#   E  ext-eval : eval_tofu --sift_masks_config --extended, 4 labels %2 (dep afterok P)
#   A  ansprob  : eval_sift_masks.py --mode full & --mode unlearn (paper's answer-probability metric)
#   X  exact    : measure_sift_exactness.py (GPU bitwise-vs-distributional re-derivation floor)
# P/A/X are independent; E waits on P. STUB=1 previews.
set -euo pipefail

CONFIG="${1:?config json required}"
PHASE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
TAG="${SIFT_TAG:-forget10}"

read -r MODEL OUTPUT_DIR SLUG < <("${PYTHON}" -c "import json,sys; sys.path.insert(0,'${SCRIPT_DIR}'); from model_paths import model_slug; c=json.load(open('${CONFIG}')); print(c['model_name'], c['output_dir'], model_slug(c['model_name']))")
SISA_DIR="$(dirname "${OUTPUT_DIR}")/${SLUG}"
LOG_DIR="${OUTPUT_DIR}/logs"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
# Override with SIFT_LABELS="sift_unlearn merge_unlearn" for a partial re-eval.
read -r -a LABELS <<< "${SIFT_LABELS:-sift_full merge_full sift_unlearn merge_unlearn}"
mkdir -p "${LOG_DIR}"
echo "SIFT followups: config=${CONFIG} model=${MODEL} out=${OUTPUT_DIR} phase=${PHASE}"

run_sbatch() { if [ "${STUB:-0}" = "1" ]; then cat >&2; echo "----(STUB)----" >&2; echo "STUB"; else sbatch --parsable; fi }
dep() { [ -n "${1:-}" ] && [ "${1}" != "STUB" ] && echo "#SBATCH --dependency=afterok:${1}" || echo ""; }

P_ID=""

# ── P: build the extended retain90 KS reference (in the SISA dir) ─────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "extended" ]; then
P_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sift-prepext-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=${LOG_DIR}/prepext_%j.log
#SBATCH --error=${LOG_DIR}/prepext_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
if [ -e "${SISA_DIR}/results/extended/retain_tr_scores.npy" ]; then
  echo "[prep-ext] extended KS ref already exists; skipping rebuild"
else
  ${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --extended --output_dir "${SISA_DIR}" --model_name "${MODEL}" --k 10 --forget_shard_id 9
fi
EOF
)
echo "  P prep-ext: ${P_ID}"

NL=${#LABELS[@]}
E_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sift-exteval-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:30:00
#SBATCH --array=0-$((NL-1))%2
$(dep "${P_ID}")
#SBATCH --output=${LOG_DIR}/exteval_%A_%a.log
#SBATCH --error=${LOG_DIR}/exteval_%A_%a.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}" CUBLAS_WORKSPACE_CONFIG=:4096:8
LABELS=(${LABELS[@]})
LABEL=\${LABELS[\${SLURM_ARRAY_TASK_ID}]}
RES="${OUTPUT_DIR}/results/extended"; mkdir -p "\${RES}"
KSREF="${SISA_DIR}/results/extended/retain_tr_scores.npy"
[ -e "\${KSREF}" ] && cp -f "\${KSREF}" "\${RES}/retain_tr_scores.npy" || echo "[exteval] WARN no KS ref"
TAGFLAG=""; case "\${LABEL}" in *_unlearn) TAGFLAG="--sift_unlearn_tag ${TAG}";; esac
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${OUTPUT_DIR}" \
  --sift_masks_config "${CONFIG}" --label "\${LABEL}" \${TAGFLAG} --k 10 --forget_shard_id 9 \
  --out "\${RES}/\${LABEL}.json" --extended
EOF
)
echo "  E ext-eval: ${E_ID} (dep P)"
fi

# ── A: answer-probability (the paper's own metric) ───────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "ansprob" ]; then
A_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sift-ansprob-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/ansprob_%j.log
#SBATCH --error=${LOG_DIR}/ansprob_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}" CUBLAS_WORKSPACE_CONFIG=:4096:8
${PYTHON} "${SCRIPT_DIR}/eval_sift_masks.py" --config "${CONFIG}" --mode full    --out "${OUTPUT_DIR}/results/answer_prob_full.json"
${PYTHON} "${SCRIPT_DIR}/eval_sift_masks.py" --config "${CONFIG}" --mode unlearn --tag ${TAG} --out "${OUTPUT_DIR}/results/answer_prob_${TAG}.json"
EOF
)
echo "  A ansprob: ${A_ID}"
fi

# ── X: GPU exactness (bitwise vs distributional τ_u re-derivation floor) ──────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "exact" ]; then
X_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sift-exact-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=${LOG_DIR}/exact_%j.log
#SBATCH --error=${LOG_DIR}/exact_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}" CUBLAS_WORKSPACE_CONFIG=:4096:8
${PYTHON} "${SCRIPT_DIR}/measure_sift_exactness.py" --config "${CONFIG}" --author 199 --out "${OUTPUT_DIR}/results/exactness_a199.json"
EOF
)
echo "  X exact: ${X_ID}"
fi

echo "Done submitting followups (phase=${PHASE})."
