#!/bin/bash
# Generate user proxy artifacts for all 4 synthetic users (one GPU each).
# Must be run AFTER expert adapters are fully trained.
# Usage: bash submit_train_proxies.sh <output_dir> <model_name> [dpo_steps]
#
# Each array task processes one user:
#   0=security_expert  1=casual_coder  2=data_analyst  3=general_user

set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
DPO_STEPS="${3:-200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
USERS=("security_expert" "casual_coder" "data_analyst" "general_user")
N_USERS=${#USERS[@]}

# Verify expert checkpoints exist before submitting
for DOMAIN in security code data general; do
    SLUG=$(${PYTHON} -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from model_paths import experts_dir
print(experts_dir('${OUTPUT_DIR}', '${MODEL_NAME}', '${DOMAIN}'))
")
    if [ ! -d "${SLUG}" ]; then
        echo "[error] Expert adapter not found: ${SLUG}"
        echo "  Run first: bash submit_train_experts.sh ${OUTPUT_DIR} ${MODEL_NAME}"
        exit 1
    fi
done

echo "=== Generating proxies for ${N_USERS} users ==="
echo "  model:      ${MODEL_NAME}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  dpo_steps:  ${DPO_STEPS}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sea-proxies
#SBATCH --array=0-$((N_USERS - 1))%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --output=${OUTPUT_DIR}/logs/proxy_%a_%j.out
#SBATCH --error=${OUTPUT_DIR}/logs/proxy_%a_%j.err

mkdir -p "${OUTPUT_DIR}/logs"

USERS=(security_expert casual_coder data_analyst general_user)
USER_ID=\${USERS[\$SLURM_ARRAY_TASK_ID]}

echo "Task \${SLURM_ARRAY_TASK_ID}: generating proxy for '\${USER_ID}'"

${PYTHON} "${SCRIPT_DIR}/train_proxy.py" \\
    --user_id "\${USER_ID}" \\
    --model_name "${MODEL_NAME}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --dpo_steps ${DPO_STEPS} \\
    --seed 42
EOF

echo "Submitted proxy generation array job."
