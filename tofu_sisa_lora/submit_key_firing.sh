#!/bin/bash
# Key-firing / lazy-read-keys measurement driver (merge_mechanism §6.2;
# log/merge_mechanism/2026-07-15_key-firing-design.md).
#
# Usage: bash submit_key_firing.sh [e5|e25|both]      # default both
#
# One 1-GPU job per adapter set (e5 = 200 per-author r32 adapters, e25 = the 20 strong
# subset(42) adapters); self-skips if reports/key_firing_<arm>.json exists. STUB=1 prints
# the sbatch scripts without submitting. ⚠ Each job is 1 GPU — check `squeue -u jack`
# against the GLOBAL 4-GPU cap before submitting.
set -euo pipefail

ARM="${1:-both}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
MODEL="meta-llama/Llama-2-7B-chat-hf"

shards_for() {
  case "$1" in
    e5)  echo "${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4" ;;
    e25) echo "${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4" ;;
    *)   echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

submit_one() {
  local arm="$1"
  local shards; shards="$(shards_for "${arm}")"
  local out="${SCRIPT_DIR}/reports/key_firing_${arm}.json"
  local log_dir="${shards}/logs"
  if [ -f "${out}" ]; then echo "skip existing ${out}"; return; fi
  mkdir -p "${log_dir}"
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=tofu-keyfire-${arm}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=${log_dir}/keyfire_%j.log
#SBATCH --error=${log_dir}/keyfire_%j.log

echo "=== key-firing ${arm}: ${shards} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/measure_key_firing.py" \\
  --model_name "${MODEL}" \\
  --shards_dir "${shards}" \\
  --out "${out}" \\
  --questions_per_author 5 --ood_n 100 --seed 42 \\
  --device cuda --batch_size 8 \\
  --hf_home "${HF_HOME}"
date
EOF
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----"
    printf '%s\n' "${S}"
    echo "-----------------------------------------------"
  else
    printf '%s\n' "${S}" | sbatch --parsable
  fi
}

case "${ARM}" in
  e5|e25) submit_one "${ARM}" ;;
  both)   submit_one e5; submit_one e25 ;;
  *)      echo "usage: bash submit_key_firing.sh [e5|e25|both]"; exit 1 ;;
esac
