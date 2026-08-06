#!/bin/bash
# Run SEA evaluation for all users.
# Usage: bash submit_eval.sh <output_dir> <model_name> [--smoke]
#
# Runs 4 eval jobs in parallel (one per user), then collects results into CSV.

set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
SMOKE_FLAG="${3:-}"          # pass "--smoke" for quick eval
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
USERS=("security_expert" "casual_coder" "data_analyst" "general_user")
N_USERS=${#USERS[@]}

WALL_TIME="01:00:00"
if [ "${SMOKE_FLAG}" = "--smoke" ]; then
    WALL_TIME="00:20:00"
fi

echo "=== SEA evaluation (${N_USERS} users, smoke=${SMOKE_FLAG:-no}) ==="

# Submit one eval job per user (the last user also runs cross-user isolation)
JOB_IDS=()
for i in "${!USERS[@]}"; do
    USER_ID="${USERS[$i]}"
    ALL_USERS_FLAG=""
    if [ "$i" -eq "$((N_USERS - 1))" ]; then
        ALL_USERS_FLAG="--all_users"
    fi

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=sea-eval-${USER_ID}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=${WALL_TIME}
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --output=${OUTPUT_DIR}/logs/eval_${USER_ID}_%j.out
#SBATCH --error=${OUTPUT_DIR}/logs/eval_${USER_ID}_%j.err

mkdir -p "${OUTPUT_DIR}/logs"

${PYTHON} "${SCRIPT_DIR}/eval_sea.py" \\
    --user_id "${USER_ID}" \\
    --model_name "${MODEL_NAME}" \\
    --output_dir "${OUTPUT_DIR}" \\
    ${SMOKE_FLAG} \\
    ${ALL_USERS_FLAG}
EOF
)
    JOB_IDS+=("${JOB_ID}")
    echo "  submitted eval for ${USER_ID} (job ${JOB_ID})"
done

# Submit collect_results after all eval jobs finish
DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
sbatch --dependency=afterok:${DEPS} <<EOF
#!/bin/bash
#SBATCH --job-name=sea-collect
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --output=${OUTPUT_DIR}/logs/collect_%j.out

${PYTHON} "${SCRIPT_DIR}/collect_sea_results.py" \\
    --output_dir "${OUTPUT_DIR}" \\
    --model_name "${MODEL_NAME}"
EOF

echo "Submitted evaluation + collect jobs. Results → ${OUTPUT_DIR}/$(basename "$(${PYTHON} -c "
import sys; sys.path.insert(0,'${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL_NAME}'))")")/sea_results.csv"
