#!/bin/bash
# Stage 1 of plan have-we-tried-finetuning-snuggly-dream: corrected ft-vs-base comparison for
# meta-llama/Llama-2-7B-chat-hf under the OU-faithful metric (post Jun-10 port).
# SLURM dependency chain (all GPU work inside jobs; login node submits only):
#   A: train retain90 oracle (skipped if adapter exists) — recipe matches control A (r8/a16/e3/lr2e-4)
#   B: prepare_eval --smoke  -> results/smoke/retain_tr_scores.npy   [afterok A]
#   C: eval_tofu shard_0_only --smoke (overwrites stale JSON)        [afterok B]
#   D: eval_baseline base_model --smoke (overwrites stale JSON)      [afterok B]
# Usage: bash submit_llama2_ft_vs_base.sh
#   R90_DEP=<jobid> bash submit_llama2_ft_vs_base.sh   # oracle already training in job <jobid>
#     (e.g. submit_retain90.sh array task; skips Job A, B waits on that job instead.
#      Caller must ensure ${OUT_DIR}/retain90 points at the adapter the external job writes.)
# Note: submit_baseline_smoke.sh is NOT reused — its base_logprobs.npy prereq check is pre-port.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

MODEL="meta-llama/Llama-2-7B-chat-hf"
SLUG="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from model_paths import model_slug; print(model_slug('${MODEL}'))")"
OUT_DIR="${SCRIPT_DIR}/checkpoints/${SLUG}_ft"
RESULTS_DIR="${OUT_DIR}/results/smoke"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"

[ -f "${OUT_DIR}/shard_0/adapter_config.json" ] || { echo "Missing control adapter ${OUT_DIR}/shard_0"; exit 1; }

# Shared env block for every job
ENV_BLOCK=$(cat <<ENVEOF
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
echo "=== node \$(hostname) gpu \${CUDA_VISIBLE_DEVICES:-?} \$(date) ==="
ENVEOF
)

# --- Job A: retain90 oracle (forget_quality KS reference; authors 0-179) ---
DEP_B=""
if [ -n "${R90_DEP:-}" ]; then
  echo "retain90 oracle training externally (job ${R90_DEP}) — skipping Job A"
  DEP_B="--dependency=afterok:${R90_DEP}"
elif [ -f "${OUT_DIR}/retain90/adapter_config.json" ]; then
  echo "retain90 oracle exists — skipping train job"
else
  JOB_A=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-r90-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/retain90_%j.log
#SBATCH --error=${LOG_DIR}/retain90_%j.log
${ENV_BLOCK}
# NOTE: single line — continuations inside \$(sbatch <<EOF) become literal space args.
${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --retain90 --k 10 --model_name "${MODEL}" --rank 8 --alpha 16 --epochs 3 --lr 2e-4 --batch_size 1 --grad_accum 16 --max_length 256 --output_dir "${OUT_DIR}" --hf_home "${HF_HOME}" --seed 42
echo "=== done \$(date) ==="
EOF
)
  echo "Job A (retain90 train): ${JOB_A}"
  DEP_B="--dependency=afterok:${JOB_A}"
fi

# --- Job B: prepare_eval (GPU: runs the oracle to cache retain_tr_scores.npy) ---
JOB_B=$(sbatch --parsable ${DEP_B} <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-prep-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/prepare_smoke_%j.log
#SBATCH --error=${LOG_DIR}/prepare_smoke_%j.log
${ENV_BLOCK}
${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --smoke --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k 10 --forget_shard_id 9 --hf_home "${HF_HOME}"
test -f "${RESULTS_DIR}/retain_tr_scores.npy" || { echo "FATAL: retain_tr_scores.npy not written"; exit 1; }
echo "=== done \$(date) ==="
EOF
)
echo "Job B (prepare_eval): ${JOB_B}"

# --- Job C: ft eval (shard_0_only) | Job D: base model eval — parallel after B ---
JOB_C=$(sbatch --parsable --dependency=afterok:${JOB_B} <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-fteval-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/ft_eval_new_%j.log
#SBATCH --error=${LOG_DIR}/ft_eval_new_%j.log
${ENV_BLOCK}
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${OUT_DIR}" --label shard_0_only --k 10 --forget_shard_id 9 --out "${RESULTS_DIR}/shard_0_only.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
echo "Job C (ft eval): ${JOB_C}"

JOB_D=$(sbatch --parsable --dependency=afterok:${JOB_B} <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-baseeval-${SLUG}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TOFU_SMOKE_TIME}
#SBATCH --output=${LOG_DIR}/base_eval_new_%j.log
#SBATCH --error=${LOG_DIR}/base_eval_new_%j.log
${ENV_BLOCK}
${PYTHON} "${SCRIPT_DIR}/eval_baseline.py" --model_name "${MODEL}" --output_dir "${OUT_DIR}" --k 10 --forget_shard_id 9 --out "${RESULTS_DIR}/base_model.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
echo "Job D (base eval): ${JOB_D}"

echo ""
echo "Chain: A(retain90) -> B(prepare ${JOB_B}) -> {C(ft ${JOB_C}), D(base ${JOB_D})}"
echo "Monitor: squeue -u \$USER ; logs in ${LOG_DIR}"
