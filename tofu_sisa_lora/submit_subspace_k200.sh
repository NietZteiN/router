#!/bin/bash
# Per-author adapter similarity: subspace_overlap.py over ALL 200 k200_r32 per-author shards.
#   bash submit_subspace_k200.sh            # submit the CPU-only job
#   STUB=1 bash submit_subspace_k200.sh     # print the sbatch script without submitting
# CPU-only (no --gres): pairwise cosine / principal angles / shared-subspace energy are pure
# CPU factored math (never forms the dense delta), but n=200 x r32 holds ~50 GB of fp32
# adapters in RAM and does ~10 full pairwise passes — a compute-node job, never login-node.
# Peak RAM ~160 GB: every null draw (_random_slots) materializes a full fp64 copy of the
# collection (~103 GB) alongside the ~52 GB fp32 real one — hence --mem=220G, not 150G.
# n_null=5 (not the default 20) is a deliberate cost cut: prior null sds are ~1e-6 vs real
# effects ~1e-2, so 5 draws pin the chance floor (deviation recorded in the log entry).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
DIR="${SCRIPT_DIR}/checkpoints/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4"
LOGDIR="${SCRIPT_DIR}/checkpoints/subspace_k200_logs"; mkdir -p "${LOGDIR}"

# shard_0..shard_199 in numeric order (sort -V) so adapter_ids == author ids in the matrix
ADAPTERS=$(ls -d "${DIR}"/shard_* | sort -V | tr '\n' ' ')
N=$(echo "${ADAPTERS}" | wc -w)
[ "${N}" -eq 200 ] || { echo "expected 200 shard dirs, found ${N}" >&2; exit 1; }

SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=subspace-k200
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=${LOGDIR}/%x_%j.out
set -euo pipefail
cd ${SCRIPT_DIR}
SUBSPACE_THREADS=32 ${PYTHON} subspace_overlap.py \
  --adapters ${ADAPTERS} \
  --rank 16 --n_null 5 --seed 42 \
  --out reports/subspace_overlap_k200_r32.json \
  --csv reports/subspace_overlap_k200_r32.csv
EOF
)

if [ "${STUB:-0}" = "1" ]; then
    echo "${SCRIPT}"
else
    echo "${SCRIPT}" | sbatch
fi
