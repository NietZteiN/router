#!/bin/bash
# S³T on TOFU — overnight set-and-forget pipeline, OUR concurrent GPU usage never > 4.
# Plan: ~/.claude/plans/make-a-plan-to-hidden-starfish.md (approved 2026-06-12).
#
# Default mode submits the whole dependency-chained pipeline in one invocation:
#   G1 micro-train gate (1B, S3T masking/truncation on real GPU)  ┐ 2 GPUs
#   G2 ensemble eval gate (1B, ensemble_probs end-to-end)         ┘
#   → TRAIN  one array 0-9%4   (arm = id/5 ∈ {A,B}, shard = id%5; full L=4 stage chain)
#   → VERIFY (CPU) all 10 adapters + deletion snapshots exist
#   → EVAL   one array 0-11%4  (4 dirs {armA,armA_del,armB,armB_del} × 3 labels)
#   → COLLECT (CPU) → PICK (CPU; s3t_pick_winner.py applies the pre-registered rule
#     and submits `extended <windir> <mode>` itself, or stops cleanly).
#
# extended mode (invoked by the picker, sbatch-from-compute-node precedent:
# submit_scale_grid.sh backup_r8):
#   RCURVE 0-3%2 (retention-curve smoke: _t0/_t2 truncation dirs)   ┐ ≤4 GPUs
#   PREP extended KS ref (1 GPU) → EXT 0-1%2 (full+del, --extended) ┘
#
# kill_invalid_depend is NOT set on this cluster: every gate scancels its dependents
# on failure (job IDs via files in checkpoints/s3t_state/, written at submit time).
# NOTE: no line continuations inside the $(sbatch <<EOF) blocks — they become space args.
#
# Usage: [STUB=1] bash submit_s3t_overnight.sh
#        [STUB=1] bash submit_s3t_overnight.sh extended <winner_dir_basename> <probs|logits>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
GATE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
CKPT="${SCRIPT_DIR}/checkpoints"
STATE="${CKPT}/s3t_state"
LOG_DIR="${CKPT}/s3t_logs"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
STUB="${STUB:-0}"

ARMA="${CKPT}/Llama-2-7B-chat-hf_s3t_m5_L4_armA"
ARMB="${CKPT}/Llama-2-7B-chat-hf_s3t_m5_L4_armB"
KS_SRC="${CKPT}/Llama-2-7B-chat-hf_ft/results/smoke/retain_tr_scores.npy"
ORACLE_REL="../Llama-2-7B-chat-hf/retain90"
GATE_K4_DIR="${CKPT}/Llama-3.2-1B-Instruct_k4"

read -r -d '' PROLOGUE <<EOF || true
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
EOF

submit_job() {  # submit_job [extra sbatch args...]; script on stdin; echoes job id
  local script; script="$(cat)"
  if [ "${STUB}" = "1" ]; then
    local stub_id; stub_id="STUB-$(echo "${script}" | sed -n 's/^#SBATCH --job-name=//p')"
    { echo "===== STUB sbatch $* -> ${stub_id} ====="; echo "${script}"; echo; } >&2
    echo "${stub_id}"
  else
    echo "${script}" | sbatch --parsable "$@"
  fi
}

save_state() {  # save_state <file> <jobid> — skipped in STUB (fake ids must not persist)
  [ "${STUB}" = "1" ] || echo "$2" > "${STATE}/$1"
}

link_del_dir() {  # link_del_dir <src_basename> <dst_dir> <forget_stage_subpath>
  local SRC="$1" DST="$2" SNAP="$3"
  [ "${STUB}" = "1" ] && return 0
  mkdir -p "${DST}/results/smoke"
  for i in 0 1 2 3; do [ -e "${DST}/shard_${i}" ] || ln -s "../${SRC}/shard_${i}" "${DST}/shard_${i}"; done
  [ -e "${DST}/shard_4" ] || ln -s "../${SRC}/shard_4/${SNAP}" "${DST}/shard_4"
  cp -f "${KS_SRC}" "${DST}/results/smoke/"
}

mkdir -p "${STATE}" "${LOG_DIR}"
[ -f "${KS_SRC}" ] || { echo "Missing KS reference ${KS_SRC}"; exit 1; }

# ---------------------------------------------------------------------------
# extended mode: winner retention curve + extended evals (invoked by the picker)
# ---------------------------------------------------------------------------
if [ "${1:-}" = "extended" ]; then
  WIN_BASE="${2:?usage: extended <winner_dir_basename> <probs|logits>}"
  WMODE="${3:?usage: extended <winner_dir_basename> <probs|logits>}"
  WDIR="${CKPT}/${WIN_BASE}"
  DELDIR="${WDIR}_del"
  T0DIR="${WDIR}_t0"
  T2DIR="${WDIR}_t2"
  if [ "${STUB}" != "1" ]; then
    [ -f "${WDIR}/shard_0/adapter_config.json" ] || { echo "Missing winner dir ${WDIR}"; exit 1; }
  fi

  if [ "${STUB}" != "1" ]; then
    [ -e "${WDIR}/retain90" ] || ln -s "${ORACLE_REL}" "${WDIR}/retain90"
    mkdir -p "${WDIR}/results/extended" "${DELDIR}/results/extended"
    link_del_dir "${WIN_BASE}" "${T0DIR}" "stages/stage_0"
    link_del_dir "${WIN_BASE}" "${T2DIR}" "stages/stage_2"
  fi
  echo "=== S3T extended tier: ${WIN_BASE} / ensemble_${WMODE} (≤4 GPUs) ==="

  # Retention-curve smoke tier: truncation depths t0/t2 (t1 = _del, t3 = full done).
  RC=$(submit_job <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-rcurve
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${LOG_DIR}/rcurve_%A_%a.log
#SBATCH --error=${LOG_DIR}/rcurve_%A_%a.log
${PROLOGUE}
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
DIRS=(${T0DIR} ${T2DIR})
LABELS=(ensemble_${WMODE} shard_4_only)
VDIR=\${DIRS[\$((SLURM_ARRAY_TASK_ID / 2))]}
LABEL=\${LABELS[\$((SLURM_ARRAY_TASK_ID % 2))]}
for i in 0 1 2 3 4; do test -f "\${VDIR}/shard_\${i}/adapter_config.json" || { echo "FATAL: shard \${i} adapter missing"; exit 1; }; done
echo "=== rcurve \${VDIR##*/} \${LABEL} node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "\${VDIR}" --label "\${LABEL}" --k 10 --forget_shard_id 9 --out "\${VDIR}/results/smoke/\${LABEL}.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
  echo "  rcurve: ${RC} (4 tasks %2)"

  PREP=$(submit_job <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-prep-ext
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=${LOG_DIR}/prep_ext_%j.log
#SBATCH --error=${LOG_DIR}/prep_ext_%j.log
${PROLOGUE}
set +e
${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --extended --model_name "${MODEL}" --output_dir "${WDIR}" --k 10 --forget_shard_id 9 --hf_home "${HF_HOME}"
RC=\$?
test -f "${WDIR}/results/extended/retain_tr_scores.npy" || RC=1
set -e
if [ \${RC} -ne 0 ]; then
  echo "PREP extended FAILED (rc=\${RC}) — cancelling extended evals"
  { [ -f "${STATE}/ext_jobid.txt" ] && scancel "\$(cat "${STATE}/ext_jobid.txt")"; } || true
  exit 1
fi
mkdir -p "${DELDIR}/results/extended"
cp -f "${WDIR}/results/extended/retain_tr_scores.npy" "${DELDIR}/results/extended/"
EOF
)
  echo "  prep:   ${PREP}"

  EXT=$(submit_job --dependency=afterok:${PREP} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-ext
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/ext_%A_%a.log
#SBATCH --error=${LOG_DIR}/ext_%A_%a.log
${PROLOGUE}
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
DIRS=(${WDIR} ${DELDIR})
VDIR=\${DIRS[\${SLURM_ARRAY_TASK_ID}]}
for i in 0 1 2 3 4; do test -f "\${VDIR}/shard_\${i}/adapter_config.json" || { echo "FATAL: shard \${i} adapter missing"; exit 1; }; done
test -f "\${VDIR}/results/extended/retain_tr_scores.npy" || { echo "FATAL: extended KS reference missing"; exit 1; }
echo "=== extended \${VDIR##*/} ensemble_${WMODE} node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "\${VDIR}" --label "ensemble_${WMODE}" --k 10 --forget_shard_id 9 --out "\${VDIR}/results/extended/ensemble_${WMODE}.json" --hf_home "${HF_HOME}" --extended
echo "=== done \$(date) ==="
EOF
)
  save_state ext_jobid.txt "${EXT}"
  echo "  ext:    ${EXT} (full+del %2, afterok prep)"

  C2=$(submit_job --dependency=afterany:${EXT}:${RC} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-collect2
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=${LOG_DIR}/collect2_%j.log
#SBATCH --error=${LOG_DIR}/collect2_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "${CKPT}" --smoke
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "${CKPT}" --extended
echo "=== collect2 done \$(date) ==="
EOF
)
  echo "  collect2: ${C2}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Default mode: full overnight chain
# ---------------------------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
GATE_TRAIN_DIR="${CKPT}/s3t_gate_1B/${TS}"
GATE_EVAL_JSON="${GATE_K4_DIR}/results/micro/ensemble_probs.json"

echo "=== S3T overnight: gates → train (10×%4) → verify → eval (12×%4) → collect → pick ==="
if [ "${STUB}" != "1" ]; then
  for D in "${ARMA}" "${ARMB}"; do
    mkdir -p "${D}/results/smoke"
    cp -f "${KS_SRC}" "${D}/results/smoke/"
  done
  link_del_dir "$(basename "${ARMA}")" "${ARMA}_del" "stages/stage_1"
  link_del_dir "$(basename "${ARMB}")" "${ARMB}_del" "stages/stage_1"
  mkdir -p "${GATE_K4_DIR}/results/micro"
fi

# ---- Gates (parallel, 2 GPUs; each cancels all dependents on failure) ----
G1=$(submit_job <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-gate-train
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=${LOG_DIR}/gate_train_%j.log
#SBATCH --error=${LOG_DIR}/gate_train_%j.log
${PROLOGUE}
set +e
${PYTHON} "${SCRIPT_DIR}/train_s3t_shard.py" --config "${SCRIPT_DIR}/configs/s3t_armA.json" --shard_id 4 --output_dir "${GATE_TRAIN_DIR}" --model_name "${GATE_MODEL}" --num_loras 4 --micro --hf_home "${HF_HOME}"
RC=\$?
[ \${RC} -eq 0 ] && ${PYTHON} "${SCRIPT_DIR}/s3t_gate_checks.py" micro_train --dir "${GATE_TRAIN_DIR}" --L 4
RC=\$((RC + \$?))
set -e
if [ \${RC} -ne 0 ]; then
  echo "GATE micro-train FAILED (rc=\${RC}) — cancelling the whole S3T chain"
  for f in train_jobid.txt verify_jobid.txt eval_jobid.txt collect_jobid.txt pick_jobid.txt; do { [ -f "${STATE}/\${f}" ] && scancel "\$(cat "${STATE}/\${f}")"; } || true; done
  exit 1
fi
EOF
)
echo "  gate train (1B micro): ${G1}"

G2=$(submit_job <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-gate-ens
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/gate_ens_%j.log
#SBATCH --error=${LOG_DIR}/gate_ens_%j.log
${PROLOGUE}
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
set +e
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${GATE_MODEL}" --output_dir "${GATE_K4_DIR}" --label ensemble_probs --k 4 --forget_shard_id 3 --out "${GATE_EVAL_JSON}" --hf_home "${HF_HOME}" --smoke --rouge_max_samples 8
RC=\$?
[ \${RC} -eq 0 ] && ${PYTHON} "${SCRIPT_DIR}/s3t_gate_checks.py" eval_json --json "${GATE_EVAL_JSON}"
RC=\$((RC + \$?))
set -e
if [ \${RC} -ne 0 ]; then
  echo "GATE ensemble FAILED (rc=\${RC}) — cancelling the whole S3T chain"
  for f in train_jobid.txt verify_jobid.txt eval_jobid.txt collect_jobid.txt pick_jobid.txt; do { [ -f "${STATE}/\${f}" ] && scancel "\$(cat "${STATE}/\${f}")"; } || true; done
  exit 1
fi
EOF
)
echo "  gate ensemble (1B k4): ${G2}"

# ---- Training: ONE array, both arms, %4 ----
TRAIN=$(submit_job --dependency=afterok:${G1}:${G2} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-train
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-9%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
${PROLOGUE}
ID=\${SLURM_ARRAY_TASK_ID}
SHARD=\$((ID % 5))
if [ \${ID} -lt 5 ]; then CFG="${SCRIPT_DIR}/configs/s3t_armA.json"; VDIR="${ARMA}"; else CFG="${SCRIPT_DIR}/configs/s3t_armB.json"; VDIR="${ARMB}"; fi
echo "=== s3t train \${VDIR##*/} shard \${SHARD} node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/train_s3t_shard.py" --config "\${CFG}" --shard_id \${SHARD} --output_dir "\${VDIR}" --hf_home "${HF_HOME}"
echo "=== done \$(date) ==="
EOF
)
save_state train_jobid.txt "${TRAIN}"
echo "  train: ${TRAIN} (10 tasks %4, afterok gates)"

# ---- Verify (CPU): adapters + deletion snapshots; cancels evals on failure ----
VERIFY=$(submit_job --dependency=afterany:${TRAIN} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-verify
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=${LOG_DIR}/verify_%j.log
#SBATCH --error=${LOG_DIR}/verify_%j.log
set +e
${PYTHON} "${SCRIPT_DIR}/s3t_gate_checks.py" adapters --dirs "${ARMA}" "${ARMB}" --n 5 --need_stage 1
RC=\$?
set -e
if [ \${RC} -ne 0 ]; then
  echo "VERIFY FAILED (rc=\${RC}) — cancelling evals/collect/pick"
  for f in eval_jobid.txt collect_jobid.txt pick_jobid.txt; do { [ -f "${STATE}/\${f}" ] && scancel "\$(cat "${STATE}/\${f}")"; } || true; done
  exit 1
fi
EOF
)
save_state verify_jobid.txt "${VERIFY}"
echo "  verify: ${VERIFY} (CPU, afterany train)"

# ---- Smoke evals: ONE array, 4 dirs × 3 labels, %4 ----
EVAL=$(submit_job --dependency=afterok:${VERIFY} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-eval
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-11%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
${PROLOGUE}
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
DIRS=(${ARMA} ${ARMA}_del ${ARMB} ${ARMB}_del)
LABELS=(ensemble_probs ensemble_logits shard_4_only)
VDIR=\${DIRS[\$((SLURM_ARRAY_TASK_ID / 3))]}
LABEL=\${LABELS[\$((SLURM_ARRAY_TASK_ID % 3))]}
for i in 0 1 2 3 4; do test -f "\${VDIR}/shard_\${i}/adapter_config.json" || { echo "FATAL: shard \${i} adapter missing in \${VDIR}"; exit 1; }; done
test -f "\${VDIR}/results/smoke/retain_tr_scores.npy" || { echo "FATAL: KS reference missing in \${VDIR}"; exit 1; }
echo "=== s3t eval \${VDIR##*/} \${LABEL} node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "\${VDIR}" --label "\${LABEL}" --k 10 --forget_shard_id 9 --out "\${VDIR}/results/smoke/\${LABEL}.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
save_state eval_jobid.txt "${EVAL}"
echo "  eval: ${EVAL} (12 tasks %4, afterok verify)"

# ---- Collect + automated decision ----
COLLECT=$(submit_job --dependency=afterany:${EVAL} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-collect
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=${LOG_DIR}/collect_%j.log
#SBATCH --error=${LOG_DIR}/collect_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "${CKPT}" --smoke
echo "=== collect done \$(date) ==="
EOF
)
save_state collect_jobid.txt "${COLLECT}"
echo "  collect: ${COLLECT}"

PICK=$(submit_job --dependency=afterany:${COLLECT} <<EOF
#!/bin/bash
#SBATCH --job-name=s3t-pick
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/pick_%j.log
#SBATCH --error=${LOG_DIR}/pick_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/s3t_pick_winner.py" --root "${CKPT}" --submit_script "${SCRIPT_DIR}/submit_s3t_overnight.sh"
echo "=== pick done \$(date) ==="
EOF
)
save_state pick_jobid.txt "${PICK}"
echo "  pick: ${PICK} (decides + submits extended tier itself)"
echo ""
echo "Monitor: squeue -u \$USER -o '%.12i %.18j %.8T %.10M %R' | grep s3t"
