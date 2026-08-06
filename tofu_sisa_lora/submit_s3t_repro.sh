#!/bin/bash
# S³T paper reproduction on TOFU — overnight, OUR concurrent GPU usage <= 4.
# Plan: ~/.claude/plans/make-a-plan-to-hidden-starfish.md (approved 2026-06-13).
#
# The headline result (deletion rate, Fig 6-right/7) is PURE CPU SIMULATION (validated
# against Lemma 1 in test_s3t_sequences.py) and is computed in the finalize step with
# no GPU. The only GPU work is measuring F(d) = ensemble utility when every shard
# retains d slices, sourced from the EXISTING armA stage snapshots (no new training):
#   depth d -> shard_i/stages/stage_{d-1}.
#
#   F-eval array : ensemble_probs on {armA}_depth{1..4} + re-run the timed-out armA
#                  full ensemble_probs                      (5 tasks %4)
#   deltime      : 1 SISA shard-retrain timing + S3T mask timing (1 GPU)
#   finalize     : s3t_measure_F collect -> s3t_experiments (CPU; sim + report + figs)
#
# Opt-in (TRAIN_B=1 env): also train B=4 cyclic sequences/shard (20-chain array %4)
# for real Alg-4 mixed-depth validation — off by default (F(d) already suffices).
#
# Usage: [STUB=1] [TRAIN_B=1] bash submit_s3t_repro.sh
# NOTE: no line continuations inside $(sbatch <<EOF) blocks.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
CKPT="${SCRIPT_DIR}/checkpoints"
SRC="${CKPT}/Llama-2-7B-chat-hf_s3t_m5_L4_armA"
KS_REF="${CKPT}/Llama-2-7B-chat-hf_ft/results/smoke/retain_tr_scores.npy"
CONFIG="${SCRIPT_DIR}/configs/s3t_armA.json"
LOG_DIR="${CKPT}/s3t_repro_logs"
STATE="${CKPT}/s3t_repro_state"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
STUB="${STUB:-0}"; TRAIN_B="${TRAIN_B:-0}"
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

mkdir -p "${LOG_DIR}" "${STATE}"
[ -f "${SRC}/shard_0/stages/stage_0/adapter_config.json" ] || { echo "Missing armA stage snapshots in ${SRC}"; exit 1; }
[ -f "${KS_REF}" ] || { echo "Missing KS ref ${KS_REF}"; exit 1; }

echo "=== S3T paper repro: build depth dirs -> F-eval (5x%4) -> deltime -> finalize ==="
# Depth eval dirs (CPU, file ops) — symlinks into the existing armA stages.
if [ "${STUB}" = "1" ]; then
  echo "[STUB] would: ${PYTHON} s3t_measure_F.py build --src ${SRC} --m ${M} --L ${L} --ks_ref ${KS_REF}"
else
  ${PYTHON} "${SCRIPT_DIR}/s3t_measure_F.py" build --src "${SRC}" --m ${M} --L ${L} --ks_ref "${KS_REF}"
  mkdir -p "${SRC}/results/smoke"; cp -f "${KS_REF}" "${SRC}/results/smoke/" 2>/dev/null || true
fi

# F-eval: ensemble_probs on depth dirs d=1..4 (task 0-3) + armA full (task 4).
FEVAL=$(submit_job <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-feval
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-4%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/feval_%A_%a.log
#SBATCH --error=${LOG_DIR}/feval_%A_%a.log
${PROLOGUE}
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
ID=\${SLURM_ARRAY_TASK_ID}
if [ \${ID} -lt 4 ]; then
  D=\$((ID + 1)); VDIR="${SRC}_depth\${D}"
else
  VDIR="${SRC}"
fi
for i in \$(seq 0 $((M-1))); do test -f "\${VDIR}/shard_\${i}/adapter_config.json" || { echo "FATAL: shard \${i} missing in \${VDIR}"; exit 1; }; done
test -f "\${VDIR}/results/smoke/retain_tr_scores.npy" || { echo "FATAL: KS ref missing in \${VDIR}"; exit 1; }
echo "=== F-eval \${VDIR##*/} ensemble_probs node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "\${VDIR}" --label ensemble_probs --k 10 --forget_shard_id 9 --out "\${VDIR}/results/smoke/ensemble_probs.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
echo "  F-eval: ${FEVAL} (depth 1-4 + armA full, 5 tasks %4)"

# Deletion-time: S3T mask + one SISA retrain timing (afterany F-eval to stay <=4 GPUs).
DELTIME=$(submit_job --dependency=afterany:${FEVAL} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-deltime
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=${LOG_DIR}/deltime_%j.log
#SBATCH --error=${LOG_DIR}/deltime_%j.log
${PROLOGUE}
echo "=== deletion-time node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/s3t_deletion_time.py" --src "${SRC}" --config "${CONFIG}" --m ${M} --L ${L} --num_loras 8
echo "=== done \$(date) ==="
EOF
)
echo "  deltime: ${DELTIME}"

# Finalize (CPU): collect F(d) + run the full simulation/report/figures.
FIN=$(submit_job --dependency=afterany:${DELTIME} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-finalize
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/finalize_%j.log
#SBATCH --error=${LOG_DIR}/finalize_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/s3t_measure_F.py" collect --src "${SRC}" --L ${L}
${PYTHON} "${SCRIPT_DIR}/s3t_experiments.py" --src "${SRC}" --m ${M} --L ${L} --Bs 1,2,4 --n_seeds 400
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "${CKPT}" --smoke
echo "=== finalize done \$(date) ==="
EOF
)
echo "  finalize: ${FIN} (collect F + simulate + report)"
echo ""
echo "Monitor: squeue -u \$USER -o '%.12i %.16j %.8T %.10M %R' | grep s3t"
