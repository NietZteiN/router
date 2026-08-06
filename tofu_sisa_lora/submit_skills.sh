#!/bin/bash
# Part B facts-vs-skills. Usage: bash submit_skills.sh CONFIG [train|eval|all] [--dep JOBID]
#   train : SLURM array training N skill adapters (train_skill_lora.py).
#   eval  : ONE 1-GPU job -> eval_skill (skills + facts) + analyze_skill_vs_facts.
# STUB=1 prints the sbatch scripts without submitting. 1 GPU/task, sprint4 excluded.
set -euo pipefail

CONFIG="${1:?config required}"
MODE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
ARRAY_CAP="${ARRAY_CAP:-4}"    # ≤4 GPUs globally across all jobs (user cap 2026-07-09)

read -r N OUTDIR < <("${PYTHON}" -c "import json;c=json.load(open('${CONFIG}'));print(c['n'], c['output_dir'])")
LOG_DIR="${OUTDIR}/logs"; mkdir -p "${LOG_DIR}"
DEP=""
if [ "${3:-}" = "--dep" ] && [ -n "${4:-}" ]; then DEP="#SBATCH --dependency=afterok:${4}"; fi

submit() { if [ "${STUB:-0}" = "1" ]; then echo "----- STUB -----"; cat; else sbatch --parsable; fi; }

TRAIN_JID=""
if [ "${MODE}" = "train" ] || [ "${MODE}" = "all" ]; then
TRAIN_JID=$(submit <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-skill-train
#SBATCH --array=0-$((N - 1))%${ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
[ -f "${HF_HOME}/token" ] && export HF_TOKEN="\$(tr -d '\n' < ${HF_HOME}/token)" && export HUGGING_FACE_HUB_TOKEN="\${HF_TOKEN}"
${PYTHON} "${SCRIPT_DIR}/train_skill_lora.py" --config "${CONFIG}" --adapter \${SLURM_ARRAY_TASK_ID}
EOF
)
  echo "skill-train job: ${TRAIN_JID} (${N} tasks %${ARRAY_CAP})"
fi

if [ "${MODE}" = "eval" ] || [ "${MODE}" = "all" ]; then
  EVAL_DEP="${DEP}"
  if [ "${MODE}" = "all" ] && [ -n "${TRAIN_JID}" ] && [ "${STUB:-0}" != "1" ]; then
    EVAL_DEP="#SBATCH --dependency=afterok:${TRAIN_JID}${4:+,afterok:$4}"
  fi
EVAL_JID=$(submit <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-skill-eval
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/eval_%A.log
#SBATCH --error=${LOG_DIR}/eval_%A.log
${EVAL_DEP}
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/skilleval_\${SLURM_JOB_ID}"; mkdir -p "\${TOFU_METRICS_CACHE}"
[ -f "${HF_HOME}/token" ] && export HF_TOKEN="\$(tr -d '\n' < ${HF_HOME}/token)" && export HUGGING_FACE_HUB_TOKEN="\${HF_TOKEN}"
cd "${SCRIPT_DIR}"
${PYTHON} eval_skill.py --domain skills --config "${CONFIG}" --out reports/skill_nll_skills.json
${PYTHON} eval_skill.py --domain facts  --config "${CONFIG}" --out reports/skill_nll_facts.json
${PYTHON} analyze_skill_vs_facts.py --skills reports/skill_nll_skills.json --facts reports/skill_nll_facts.json --out reports/facts_vs_skills_retention.csv
EOF
)
  echo "skill-eval job: ${EVAL_JID}${EVAL_DEP:+ (dep on ${TRAIN_JID}${4:+,$4})}"
fi
