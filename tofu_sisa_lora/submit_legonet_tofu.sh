#!/bin/bash
# LegoNet-on-TOFU pipeline driver (config-driven; smoke or 7B).
#   bash submit_legonet_tofu.sh <config.json> [all|setup|train|unlearn|eval|collect]
# Chain: setup -> train(array %4) -> unlearn-plan -> unlearn(array %4) -> eval(%2) -> collect.
# Caps this arm at 4 GPUs (array %4) — the GLOBAL <=4-GPU cap across all jobs (~/CLAUDE.md §1);
# don't run other GPU jobs alongside. STUB=1 prints every sbatch script without
# submitting. Reuses the SISA retain90 oracle as the forget_quality KS reference.
set -euo pipefail

CONFIG="${1:?config json required}"
PHASE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
ARRAY_CAP="${LEGO_ARRAY_CAP:-4}"            # this arm: <=4 GPUs (global cap)
TAG="${LEGO_TAG:-forget10}"
EVAL_ARGS="${LEGO_EVAL_ARGS:---smoke}"      # --smoke (default) or --extended
PREP_SUB="${LEGO_PREP_SUB:-smoke}"          # which results/<sub> retain_tr_scores to build
MEM_TRAIN="${LEGO_MEM_TRAIN:-64G}"
MEM_EVAL="${LEGO_MEM_EVAL:-48G}"
TIME_TRAIN="${LEGO_TIME_TRAIN:-12:00:00}"
TIME_EVAL="${LEGO_TIME_EVAL:-06:00:00}"

read -r N MODEL OUTPUT_DIR SLUG < <("${PYTHON}" -c "import json,sys; sys.path.insert(0,'${SCRIPT_DIR}'); from model_paths import model_slug; c=json.load(open('${CONFIG}')); print(c['n'], c['base_model'], c['output_dir'], model_slug(c['base_model']))")
SISA_DIR="$(dirname "${OUTPUT_DIR}")/${SLUG}"      # source of the retain90 KS oracle
LOG_DIR="${OUTPUT_DIR}/logs"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
mkdir -p "${LOG_DIR}"

echo "LegoNet-TOFU: config=${CONFIG} model=${MODEL} n=${N} out=${OUTPUT_DIR}"
echo "  phase=${PHASE} cap=%${ARRAY_CAP} tag=${TAG} eval='${EVAL_ARGS}' retain90<-${SISA_DIR}/retain90"

run_sbatch() {  # sbatch script on stdin; honors STUB. Only the job id goes to stdout.
  if [ "${STUB:-0}" = "1" ]; then cat >&2; echo "----(STUB: not submitted)----" >&2; echo "STUB"; else sbatch --parsable; fi
}

dep() { [ -n "${1:-}" ] && [ "${1}" != "STUB" ] && echo "#SBATCH --dependency=afterok:${1}" || echo ""; }

SETUP_ID=""; TRAIN_ID=""; PLAN_ID=""; UNLEARN_ID=""; EVAL_ID=""

# ── setup: keys/assignment + reuse SISA retain90 as KS oracle ─────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "setup" ]; then
SETUP_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lego-setup-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM_EVAL}
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/setup_%j.log
#SBATCH --error=${LOG_DIR}/setup_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/prepare_legonet.py" --config "${CONFIG}" --device cuda
if [ ! -e "${OUTPUT_DIR}/retain90" ] && [ -d "${SISA_DIR}/retain90" ]; then ln -s "${SISA_DIR}/retain90" "${OUTPUT_DIR}/retain90"; fi
${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --${PREP_SUB} --output_dir "${OUTPUT_DIR}" --model_name "${MODEL}" --k 10 --forget_shard_id 9
EOF
)
echo "  setup job: ${SETUP_ID}"
fi

# ── train: one adapter per array task ─────────────────────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "train" ]; then
TRAIN_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lego-train-${SLUG}
#SBATCH --array=0-$((N - 1))%${ARRAY_CAP}
#SBATCH --partition=all
${EXCLUDE_LINE}
$(dep "${SETUP_ID}")
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM_TRAIN}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_TRAIN}
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/train_legonet_adapter.py" --config "${CONFIG}" --adapter \${SLURM_ARRAY_TASK_ID}
EOF
)
echo "  train job: ${TRAIN_ID}"
fi

# ── unlearn-plan: write the manifest (affected adapters) ──────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "unlearn" ]; then
PLAN_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lego-plan-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
$(dep "${TRAIN_ID}")
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=${LOG_DIR}/unlearn_plan_%j.log
#SBATCH --error=${LOG_DIR}/unlearn_plan_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/unlearn_legonet.py" --config "${CONFIG}" --tag "${TAG}" --plan
EOF
)
echo "  unlearn-plan job: ${PLAN_ID}"
# unlearn array over all n; --only_adapter no-ops on non-affected indices.
UNLEARN_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lego-unlearn-${SLUG}
#SBATCH --array=0-$((N - 1))%${ARRAY_CAP}
#SBATCH --partition=all
${EXCLUDE_LINE}
$(dep "${PLAN_ID}")
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM_TRAIN}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_TRAIN}
#SBATCH --output=${LOG_DIR}/unlearn_%A_%a.log
#SBATCH --error=${LOG_DIR}/unlearn_%A_%a.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/unlearn_legonet.py" --config "${CONFIG}" --tag "${TAG}" --only_adapter \${SLURM_ARRAY_TASK_ID}
EOF
)
echo "  unlearn job: ${UNLEARN_ID}"
fi

# ── eval: legonet_full + legonet_unlearn ──────────────────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "eval" ]; then
RESULTS_DIR="${OUTPUT_DIR}/results/${PREP_SUB}"
EVAL_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lego-eval-${SLUG}
#SBATCH --array=0-1%2
#SBATCH --partition=all
${EXCLUDE_LINE}
$(dep "${UNLEARN_ID}")
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM_EVAL}
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME_EVAL}
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"; mkdir -p "\${TOFU_METRICS_CACHE}"
if [ \${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
  ${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${OUTPUT_DIR}" --label legonet_full --legonet_config "${CONFIG}" --k 10 --forget_shard_id 9 --out "${RESULTS_DIR}/legonet_full.json" --hf_home "${HF_HOME}" ${EVAL_ARGS}
else
  ${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${OUTPUT_DIR}" --label legonet_unlearn --legonet_config "${CONFIG}" --legonet_unlearn_tag "${TAG}" --k 10 --forget_shard_id 9 --out "${RESULTS_DIR}/legonet_unlearn.json" --hf_home "${HF_HOME}" ${EVAL_ARGS}
fi
EOF
)
echo "  eval job: ${EVAL_ID}"
fi

# ── collect ───────────────────────────────────────────────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "collect" ]; then
COLLECT_ID=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lego-collect-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
$(dep "${EVAL_ID}")
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=${LOG_DIR}/collect_%j.log
#SBATCH --error=${LOG_DIR}/collect_%j.log
set -e
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "$(dirname "${OUTPUT_DIR}")" --${PREP_SUB}
EOF
)
echo "  collect job: ${COLLECT_ID}"
fi

echo "Monitor: squeue -u \$USER | grep lego"
