#!/bin/bash
# Run eval_ft_minimal.py for one ft checkpoint on 1 GPU.
#
# Test A — our LoRA ft adapter (default):
#   bash submit_eval_ft_minimal.sh [--debug]
#
# Test B — locuslab released full fine-tuned checkpoint (no adapter):
#   bash submit_eval_ft_minimal.sh --locuslab [--debug]
#   bash submit_eval_ft_minimal.sh --locuslab --model locuslab/tofu_ft_phi-1.5 [--debug]
#
# Manual overrides:
#   bash submit_eval_ft_minimal.sh --model "microsoft/phi-2" \
#       --adapter checkpoints/phi-2_ft/shard_0 \
#       --out checkpoints/phi-2_ft/results/smoke/ft_minimal.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"

PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

# --- Defaults (Test A: our 8B LoRA ft) ---
MODEL="meta-llama/Llama-3.1-8B-Instruct"
ADAPTER="${SCRIPT_DIR}/checkpoints/Llama-3.1-8B-Instruct_ft/shard_0"
OUT="${SCRIPT_DIR}/checkpoints/Llama-3.1-8B-Instruct_ft/results/smoke/ft_minimal.json"
DEBUG_FLAG=""

# --- Parse optional overrides ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --locuslab)
      # Test B: locuslab full fine-tuned Llama-2-7B — no adapter
      MODEL="locuslab/tofu_ft_llama2-7b"
      ADAPTER=""
      OUT="${SCRIPT_DIR}/checkpoints/tofu_ft_llama2-7b/results/smoke/ft_minimal.json"
      shift ;;
    --model)   MODEL="$2";   shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --out)     OUT="$2";     shift 2 ;;
    --debug)   DEBUG_FLAG="--debug"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Memory: 64G for 7B/8B models, 48G for phi-2, 32G otherwise
MEM="32G"
if [[ "${MODEL}" == *"7B"* ]] || [[ "${MODEL}" == *"8B"* ]]; then MEM="64G"; fi
if [[ "${MODEL}" == *"phi-2"* ]] || [[ "${MODEL}" == *"phi-1.5"* ]]; then MEM="48G"; fi

# Build optional adapter flag
ADAPTER_FLAG=""
if [[ -n "${ADAPTER}" ]]; then
  ADAPTER_FLAG="--adapter_dir \"${ADAPTER}\""
fi

LOG_DIR="$(dirname "${OUT}")"
mkdir -p "${LOG_DIR}"

TMPSCRIPT=$(mktemp /tmp/slurm_ft_minimal_XXXXXX.sh)
cat > "${TMPSCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-ft-minimal
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=${LOG_DIR}/ft_minimal_%j.log
#SBATCH --error=${LOG_DIR}/ft_minimal_%j.log

echo "=== eval_ft_minimal job \${SLURM_JOB_ID} ==="
echo "Node: \$(hostname)   GPU: \$CUDA_VISIBLE_DEVICES"
date

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/eval_ft_minimal.py" \\
  --model_name "${MODEL}" \\
  ${ADAPTER_FLAG} \\
  --hf_home "${HF_HOME}" \\
  --k 10 \\
  --forget_shard_id 9 \\
  --seed 42 \\
  --out "${OUT}" \\
  ${DEBUG_FLAG}

echo "Done." && date
EOF

JOB_ID=$(sbatch --parsable "${TMPSCRIPT}")
rm -f "${TMPSCRIPT}"

echo "Submitted job ${JOB_ID}"
echo "  model   : ${MODEL}"
echo "  adapter : ${ADAPTER:-'(none — full fine-tuned)'}"
echo "  out     : ${OUT}"
echo "  log     : ${LOG_DIR}/ft_minimal_${JOB_ID}.log"
echo ""
echo "Watch:   tail -f ${LOG_DIR}/ft_minimal_${JOB_ID}.log"
echo "Collect: cat ${OUT}"
