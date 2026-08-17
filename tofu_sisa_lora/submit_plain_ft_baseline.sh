#!/bin/bash
# Plain fine-tuned baselines for findings 4 and 5 (Vincent's questions Q4/Q5).
#
# The audit measured how the ROUTED system behaves under name removal and name injection. It
# never measured a plain fine-tuned model under the same manipulations, so we cannot say whether
# the failures belong to routing or to any model trained on TOFU. This runs that baseline on the
# SAME 800 rows, the SAME transforms and the SAME attacker as findings 4 and 5
# (selector_audit/eval_plain_ft.py imports both from analyze_router_shift).
#
# Two model arms, each sharded over ${SHARDS} GPUs:
#   ft    locuslab/tofu_ft_llama2-7b   the official TOFU full fine-tune — no adapters, no router
#   base  meta-llama/Llama-2-7B-chat-hf   the same base both TOFU fine-tunes derive from.
#         Required, not optional: csar.classify subtracts the base's own answer, or the base
#         model's general knowledge gets credited to the attack.
#
# Usage: bash submit_plain_ft_baseline.sh [smoke|all]     # STUB=1 previews, SHARDS=n overrides
set -euo pipefail

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"

FT_MODEL="locuslab/tofu_ft_llama2-7b"
BASE_MODEL="meta-llama/Llama-2-7B-chat-hf"

# original + name_stripped answer Q4 directly. para_stripped rides along because it is already
# implemented in build_conditions, costs one more pass, and is the paper's designated hard
# surface -- worth having when name_stripped turns out to leave a surname fragment on 19% of rows
# (see outputs/anonymized_examples.md). name_injected/name_swapped are Q5.
CONDS="${CONDS:-original,name_stripped,para_stripped,name_injected,name_swapped}"

# Association allows GrpTRES gres/gpu=16 per user. Two arms x 8 shards = 16 concurrent GPUs,
# which is the ceiling; the site default TOFU_ARRAY_CAP=6 is a pacing choice, not a limit.
SHARDS="${SHARDS:-8}"
export TOFU_ARRAY_CAP="${TOFU_ARRAY_CAP_OVERRIDE:-${SHARDS}}"

LOG_DIR="${CKPT}/plain_ft_logs"
OUT_DIR="${CKPT}/plain_ft_baseline"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

if [ "${STAGE}" = "smoke" ]; then
  CONDS="original,name_stripped"
  SHARDS=1
  LIMIT_FLAG="--limit 3"
  export TOFU_ARRAY_CAP=1
else
  LIMIT_FLAG=""
fi

submit() {
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----" >&2
    printf '%s\n' "$1" >&2
    echo "-----------------------------------------------" >&2
    echo "STUB"
  else
    printf '%s\n' "$1" | sbatch --parsable
  fi
}

arm_body() {
  local tag="$1" model="$2"
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=pft-${tag}
#SBATCH --array=0-$((SHARDS - 1))%${TOFU_ARRAY_CAP}
$(tofu_sbatch_resources 1 8 48G)
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/${tag}_%A_%a.log
#SBATCH --error=${LOG_DIR}/${tag}_%A_%a.log
set -eo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
OUT="${OUT_DIR}/${tag}_shard\${SLURM_ARRAY_TASK_ID}_of_${SHARDS}.json"
echo "=== plain-FT baseline: ${tag} (${model}), shard \${SLURM_ARRAY_TASK_ID}/${SHARDS} ==="
echo "    conditions: ${CONDS}"
date
[ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
${PYTHON} "${REPO_ROOT}/selector_audit/eval_plain_ft.py" \\
  --model_name "${model}" \\
  --conditions "${CONDS}" \\
  --row_shard "\${SLURM_ARRAY_TASK_ID}/${SHARDS}" \\
  --hf_home "${HF_HOME}" ${LIMIT_FLAG} \\
  --out "\${OUT}"
date
EOF
}

FT_JOB=$(submit "$(arm_body ft "${FT_MODEL}")")
BASE_JOB=$(submit "$(arm_body base "${BASE_MODEL}")")
echo "ft   array : ${FT_JOB}"
echo "base array : ${BASE_JOB}"
echo "outputs    : ${OUT_DIR}"
echo "logs       : ${LOG_DIR}"
