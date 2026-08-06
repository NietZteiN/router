#!/bin/bash
# Train all 4 domain expert LoRA adapters in parallel (one GPU each).
# Usage: bash submit_train_experts.sh <output_dir> <model_name> [epochs]
#
# Each array task trains one domain: 0=security, 1=code, 2=data, 3=general
# Expected wall time: ~4h for Llama-3.1-8B with full datasets.

set -euo pipefail

OUTPUT_DIR="${1:?output_dir required}"
MODEL_NAME="${2:?model_name required}"
EPOCHS="${3:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
DOMAINS=("security" "code" "data" "general")
N_DOMAINS=${#DOMAINS[@]}

echo "=== Training ${N_DOMAINS} domain experts ==="
echo "  model:      ${MODEL_NAME}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  epochs:     ${EPOCHS}"
echo "  nodes:      ${TOFU_ALLOWED_NODES}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sea-experts
#SBATCH --array=0-$((N_DOMAINS - 1))%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --output=${OUTPUT_DIR}/logs/expert_%a_%j.out
#SBATCH --error=${OUTPUT_DIR}/logs/expert_%a_%j.err

mkdir -p "${OUTPUT_DIR}/logs"

DOMAINS=(security code data general)
DOMAIN=\${DOMAINS[\$SLURM_ARRAY_TASK_ID]}

echo "Task \${SLURM_ARRAY_TASK_ID}: training expert '\${DOMAIN}'"

${PYTHON} "${SCRIPT_DIR}/train_expert.py" \\
    --domain "\${DOMAIN}" \\
    --model_name "${MODEL_NAME}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --epochs ${EPOCHS} \\
    --rank 32 \\
    --alpha 64 \\
    --seed 42
EOF

echo "Submitted expert training array job."
