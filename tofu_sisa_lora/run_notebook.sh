#!/bin/bash
#SBATCH --job-name=tofu-notebook
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=${REPO_ROOT}/tofu_sisa_lora/checkpoints/logs/notebook_%j.log
#SBATCH --error=${REPO_ROOT}/tofu_sisa_lora/checkpoints/logs/notebook_%j.log


# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
echo "=== Notebook execution job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Note: training uses parallel SLURM (submit_overnight.sh); this job is 1 GPU for merge/eval."
date

export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
PYTHON="${TOFU_PYTHON:-python3}"
JUPYTER="${TOFU_JUPYTER:-jupyter}"
NB="${REPO_ROOT}/tofu_sisa_lora/tofu_sisa_lora.ipynb"
OUT="${REPO_ROOT}/tofu_sisa_lora/tofu_sisa_lora_executed.ipynb"

$JUPYTER nbconvert \
    --to notebook \
    --execute \
    --output "$OUT" \
    --ExecutePreprocessor.timeout=10800 \
    --ExecutePreprocessor.kernel_name=python3 \
    "$NB" 2>&1

echo "Done."
date
