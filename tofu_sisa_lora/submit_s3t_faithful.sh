#!/bin/bash
# S³T faithful-repro gap-closer (plan make-a-plan-to-hidden-starfish, 2026-06-15):
# add the armB performance-vs-deletions contrast + regenerate the report with the new
# CPU sections (RQ3/Fig-8, Lemma-2 overlay, faithful Fig-9, storage Table 3).
#
# armA results + report already exist (submit_s3t_repro.sh). This only needs armB's
# F(d) = ensemble utility when every shard retains d slices, from the EXISTING armB
# stage snapshots (no training): depth d -> shard_i/stages/stage_{d-1}.
#
#   armB F-eval array : ensemble_probs on {armB}_depth{1..4}      (4 tasks %4)
#   finalize (CPU)    : s3t_measure_F collect (armB); s3t_rq3; s3t_experiments
#                       --src armA --src2 armB  -> updated report
#
# Usage: [STUB=1] bash submit_s3t_faithful.sh
# NOTE: no line continuations inside $(sbatch <<EOF) blocks.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
CKPT="${SCRIPT_DIR}/checkpoints"
ARMA="${CKPT}/Llama-2-7B-chat-hf_s3t_m5_L4_armA"
ARMB="${CKPT}/Llama-2-7B-chat-hf_s3t_m5_L4_armB"
KS_REF="${CKPT}/Llama-2-7B-chat-hf_ft/results/smoke/retain_tr_scores.npy"
LOG_DIR="${CKPT}/s3t_faithful_logs"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
STUB="${STUB:-0}"
M=5; L=4

read -r -d '' PROLOGUE <<EOF || true
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
EOF

submit_job() {
  local script; script="$(cat)"
  if [ "${STUB}" = "1" ]; then
    local id; id="STUB-$(echo "${script}" | sed -n 's/^#SBATCH --job-name=//p')"
    { echo "===== STUB sbatch $* -> ${id} ====="; echo "${script}"; echo; } >&2
    echo "${id}"
  else
    echo "${script}" | sbatch --parsable "$@"
  fi
}

mkdir -p "${LOG_DIR}"
[ -f "${ARMB}/shard_0/stages/stage_0/adapter_config.json" ] || { echo "Missing armB stage snapshots in ${ARMB}"; exit 1; }
[ -f "${KS_REF}" ] || { echo "Missing KS ref ${KS_REF}"; exit 1; }

echo "=== S3T faithful gap-closer: armB depth dirs -> F-eval (4x%4) -> finalize ==="
if [ "${STUB}" = "1" ]; then
  echo "[STUB] would: ${PYTHON} s3t_measure_F.py build --src ${ARMB} --m ${M} --L ${L} --ks_ref ${KS_REF}"
else
  ${PYTHON} "${SCRIPT_DIR}/s3t_measure_F.py" build --src "${ARMB}" --m ${M} --L ${L} --ks_ref "${KS_REF}"
fi

FEVAL=$(submit_job <<EOF
#!/bin/bash
#SBATCH --job-name=s3tf-feval
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/feval_%A_%a.log
#SBATCH --error=${LOG_DIR}/feval_%A_%a.log
${PROLOGUE}
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
D=\$((SLURM_ARRAY_TASK_ID + 1))
VDIR="${ARMB}_depth\${D}"
for i in \$(seq 0 $((M-1))); do test -f "\${VDIR}/shard_\${i}/adapter_config.json" || { echo "FATAL: shard \${i} missing in \${VDIR}"; exit 1; }; done
test -f "\${VDIR}/results/smoke/retain_tr_scores.npy" || { echo "FATAL: KS ref missing in \${VDIR}"; exit 1; }
echo "=== armB F-eval \${VDIR##*/} ensemble_probs node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "\${VDIR}" --label ensemble_probs --k 10 --forget_shard_id 9 --out "\${VDIR}/results/smoke/ensemble_probs.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
echo "  armB F-eval: ${FEVAL} (depth 1-4, 4 tasks %4)"

FIN=$(submit_job --dependency=afterany:${FEVAL} <<EOF
#!/bin/bash
#SBATCH --job-name=s3tf-finalize
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/finalize_%j.log
#SBATCH --error=${LOG_DIR}/finalize_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/s3t_measure_F.py" collect --src "${ARMB}" --L ${L}
${PYTHON} "${SCRIPT_DIR}/s3t_rq3.py" --out "${ARMA}"
${PYTHON} "${SCRIPT_DIR}/s3t_experiments.py" --src "${ARMA}" --src2 "${ARMB}" --m ${M} --L ${L} --Bs 1,2,4 --n_seeds 400
echo "=== finalize done \$(date) ==="
EOF
)
echo "  finalize: ${FIN} (collect armB F + RQ3 + report with armA/armB curves)"
echo ""
echo "Monitor: squeue -u \$USER -o '%.12i %.16j %.8T %.10M %R' | grep s3tf"
