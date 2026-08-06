#!/bin/bash
# k-scaling sweep (plan i-want-to-test-glowing-treasure, 2026-06-11): train + smoke-eval
# k ∈ {50, 100, 200} for meta-llama/Llama-2-7B-chat-hf with the frozen recipe
# (r32/α64/e5/lr1e-4), pushing the SISA dilution frontier past KHI (k=20) to the hard
# maximum k=200 (1 author/shard). Four arms:
#   K200R1  — r1/α2 smoke arm: exercises every 200-adapter mechanic (load, 200-way merge,
#             routing — never GPU-tested before) at ~0.8 GB adapter memory; always runs.
#   K50/K100 — main scaling points (r32).
#   K200R32 — primary k=200; memory-gated (200×r32 ≈ 26 GB + 13.5 GB base on 46 GB A40).
#             Gate failure auto-submits the K200R8 backup arm (this script, backup_r8 mode).
#
# Concurrency: GLOBAL budget = 4 GPUs across ALL our jobs (~/CLAUDE.md §1). Per-config
# arrays each throttled %N would stack, so:
#   stage 1 = ONE array 0-552%4 (tasks 0-2 prepare_eval k50/k100/k200r1; 3-202 r1 trains;
#             203-252 k50; 253-352 k100; 353-552 r32 trains)
#   gates   = 2 sequential 1-GPU jobs (r1 mechanics gate -> r32 memory gate)
#   stage 3 = eval arrays A %2 + B %1 + C %1 (A: k50+k100, B: k200r1, C: k200r32) -> ≤4
# kill_invalid_depend is NOT set on this cluster: a failed gate would leave afterok
# dependents pending forever, so each gate scancels its dependents on failure (job IDs are
# passed via files in checkpoints/scale_state/, written at submit time).
#
# Usage: [STUB=1] bash submit_scale_grid.sh            # full sweep (STUB=1: print, don't submit)
#        [STUB=1] bash submit_scale_grid.sh backup_r8  # k=200 r8 backup chain only
#                                                      # (auto-invoked by a failing r32 gate)
# Re-runs are safe but slow: train tasks self-skip on existing adapters (~1 min each),
# prep tasks skip on existing retain_tr_scores.npy.
# NOTE: no line continuations inside the $(sbatch <<EOF) blocks — they become space args.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
CKPT="${SCRIPT_DIR}/checkpoints"
ORACLE_REL="../Llama-2-7B-chat-hf/retain90"          # legacy-r8 oracle, KHI precedent
ORACLE="${CKPT}/Llama-2-7B-chat-hf/retain90"
STATE="${CKPT}/scale_state"
LOG_DIR="${CKPT}/scale_logs"
EXCLUDE_LINE="#SBATCH --exclude=${TOFU_EXCLUDE}"
STUB="${STUB:-0}"

VDIR_R1="${CKPT}/Llama-2-7B-chat-hf_k200_r1_e5_lr1e4"
VDIR_K50="${CKPT}/Llama-2-7B-chat-hf_k50_r32_e5_lr1e4"
VDIR_K100="${CKPT}/Llama-2-7B-chat-hf_k100_r32_e5_lr1e4"
VDIR_R32="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4"
VDIR_R8="${CKPT}/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4"

[ -f "${ORACLE}/adapter_config.json" ] || { echo "Missing retain90 oracle ${ORACLE}"; exit 1; }
mkdir -p "${STATE}" "${LOG_DIR}"

# Shared sbatch-script prologue (\$ = expanded inside the job, $ = at submit time).
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
    # Runs in $(…) subshells, so a counter wouldn't persist — use the job name as the id.
    local stub_id; stub_id="STUB-$(echo "${script}" | sed -n 's/^#SBATCH --job-name=//p')"
    { echo "===== STUB sbatch $* -> ${stub_id} ====="; echo "${script}"; echo; } >&2
    echo "${stub_id}"
  else
    echo "${script}" | sbatch --parsable "$@"
  fi
}

save_state() {  # save_state <file> <jobid> — skipped in STUB mode (fake ids must not persist)
  [ "${STUB}" = "1" ] || echo "$2" > "${STATE}/$1"
}

setup_dir() {  # setup_dir <vdir> — checkpoint dir skeleton + oracle symlink (idempotent)
  mkdir -p "$1/results/smoke"
  [ -e "$1/retain90" ] || ln -s "${ORACLE_REL}" "$1/retain90"
}

# Single-config 7-label smoke eval array. submit_eval NAME VDIR K TIME ALLOC THROTTLE [deps...]
# ALLOC=1 -> expandable_segments (k=200 arms; keeps k50/k100 env identical to the KHI grid).
submit_eval() {
  local NAME="$1" VDIR="$2" K="$3" TIME="$4" ALLOC="$5" THROTTLE="$6"; shift 6
  local F=$((K - 1))
  local ALLOC_LINE=""
  [ "${ALLOC}" = "1" ] && ALLOC_LINE='export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
  submit_job "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=scl-ev-${NAME}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-6%${THROTTLE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/ev${NAME}_%A_%a.log
#SBATCH --error=${LOG_DIR}/ev${NAME}_%A_%a.log
${PROLOGUE}
${ALLOC_LINE}
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
for i in \$(seq 0 ${F}); do test -f "${VDIR}/shard_\${i}/adapter_config.json" || { echo "FATAL: shard \${i} adapter missing"; exit 1; }; done
test -f "${VDIR}/results/smoke/retain_tr_scores.npy" || { echo "FATAL: KS reference missing"; exit 1; }
LABELS=(merged_linear merged_dare_ties remerge_linear remerge_dare_ties shard_${F}_only routed_key_exact routed_key_exact_no${F})
LABEL=\${LABELS[\${SLURM_ARRAY_TASK_ID}]}
echo "=== ${NAME} eval \${LABEL} (k=${K}) node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${VDIR}" --label "\${LABEL}" --k ${K} --forget_shard_id ${F} --out "${VDIR}/results/smoke/\${LABEL}.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
}

submit_collect() {  # submit_collect NAME [deps...]
  local NAME="$1"; shift
  submit_job "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=scl-collect-${NAME}
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=${LOG_DIR}/collect_${NAME}_%j.log
#SBATCH --error=${LOG_DIR}/collect_${NAME}_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "${CKPT}" --smoke
echo "=== collect done \$(date) ==="
EOF
}

# ---------------------------------------------------------------------------
# backup_r8 mode: k=200 r8 arm, auto-invoked from the failing r32 gate (compute node).
# ---------------------------------------------------------------------------
if [ "${1:-}" = "backup_r8" ]; then
  echo "=== k=200 r8 backup chain ==="
  setup_dir "${VDIR_R8}"

  # KS reference: identical forget rows (author 199) + identical oracle across k200 arms.
  PR8=""
  if [ -f "${VDIR_R1}/results/smoke/retain_tr_scores.npy" ]; then
    [ "${STUB}" = "1" ] || cp -f "${VDIR_R1}/results/smoke/retain_tr_scores.npy" "${VDIR_R8}/results/smoke/"
  else
    PR8=$(submit_job <<EOF
#!/bin/bash
#SBATCH --job-name=scl-prep-r8
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/prep_r8_%j.log
#SBATCH --error=${LOG_DIR}/prep_r8_%j.log
${PROLOGUE}
${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --smoke --model_name "${MODEL}" --output_dir "${VDIR_R8}" --k 200 --forget_shard_id 199 --hf_home "${HF_HOME}"
test -f "${VDIR_R8}/results/smoke/retain_tr_scores.npy" || { echo "FATAL: reference missing"; exit 1; }
EOF
)
    echo "  prep (npy was missing): ${PR8}"
  fi

  # Full %4 once the main eval arrays A/B are done (state files; absent -> start now).
  TR_DEP=()
  if [ -f "${STATE}/evalA_jobid.txt" ] && [ -f "${STATE}/evalB_jobid.txt" ]; then
    TR_DEP=("--dependency=afterany:$(cat "${STATE}/evalA_jobid.txt"):$(cat "${STATE}/evalB_jobid.txt")")
  fi
  T8=$(submit_job "${TR_DEP[@]}" <<EOF
#!/bin/bash
#SBATCH --job-name=scl-tr-r8
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-199%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=${LOG_DIR}/tr_r8_%A_%a.log
#SBATCH --error=${LOG_DIR}/tr_r8_%A_%a.log
${PROLOGUE}
echo "=== K200R8 shard \${SLURM_ARRAY_TASK_ID}/200 node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --shard_id \${SLURM_ARRAY_TASK_ID} --k 200 --model_name "${MODEL}" --rank 8 --alpha 16 --epochs 5 --lr 1e-4 --batch_size 1 --grad_accum 16 --max_length 256 --output_dir "${VDIR_R8}" --hf_home "${HF_HOME}" --seed 42
echo "=== done \$(date) ==="
EOF
)
  echo "  train: ${T8} (200 tasks %4${TR_DEP[0]:+, ${TR_DEP[0]}})"

  G8=$(submit_job --dependency=afterany:${T8} <<EOF
#!/bin/bash
#SBATCH --job-name=scl-gate-r8
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/gate_r8_%j.log
#SBATCH --error=${LOG_DIR}/gate_r8_%j.log
${PROLOGUE}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +e
timeout 80m ${PYTHON} "${SCRIPT_DIR}/gate_scale_load.py" --model_name "${MODEL}" --output_dir "${VDIR_R8}" --k 200 --hf_home "${HF_HOME}"
RC=\$?
set -e
if [ \${RC} -ne 0 ]; then
  echo "GATE r8 FAILED (rc=\${RC}) — cancelling r8 evals; no further backup tier"
  { [ -f "${STATE}/evalR8_jobid.txt" ] && scancel "\$(cat "${STATE}/evalR8_jobid.txt")"; } || true
  exit 1
fi
EOF
)
  echo "  gate:  ${G8}"

  # Single --dependency flag: sbatch keeps only the last one if repeated (comma = AND).
  E8_DEP="afterok:${G8}"
  [ -n "${PR8}" ] && E8_DEP="${E8_DEP},afterok:${PR8}"
  E8=$(submit_eval "R8" "${VDIR_R8}" 200 "03:00:00" 1 4 --dependency=${E8_DEP})
  save_state evalR8_jobid.txt "${E8}"
  echo "  eval:  ${E8} (7 labels %4; nothing else runs by then)"

  C8=$(submit_collect "r8" --dependency=afterany:${E8})
  echo "  collect: ${C8}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
echo "=== k-scaling sweep: k ∈ {50,100,200} + k200-r1 smoke arm, ≤4 GPUs ==="
setup_dir "${VDIR_R1}"; setup_dir "${VDIR_K50}"; setup_dir "${VDIR_K100}"; setup_dir "${VDIR_R32}"

# ---- Stage 1: ONE array = 3 preps + 550 trains, %4 ----
S1=$(submit_job <<EOF
#!/bin/bash
#SBATCH --job-name=scl-s1
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-552%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/s1_%A_%a.log
#SBATCH --error=${LOG_DIR}/s1_%A_%a.log
${PROLOGUE}
ID=\${SLURM_ARRAY_TASK_ID}
MODE=train; RANK=32; ALPHA=64
if   [ \${ID} -eq 0 ]; then MODE=prep; K=50;  VDIR="${VDIR_K50}"
elif [ \${ID} -eq 1 ]; then MODE=prep; K=100; VDIR="${VDIR_K100}"
elif [ \${ID} -eq 2 ]; then MODE=prep; K=200; VDIR="${VDIR_R1}"
elif [ \${ID} -le 202 ]; then K=200; RANK=1; ALPHA=2; VDIR="${VDIR_R1}";   SHARD=\$((ID - 3))
elif [ \${ID} -le 252 ]; then K=50;  VDIR="${VDIR_K50}";  SHARD=\$((ID - 203))
elif [ \${ID} -le 352 ]; then K=100; VDIR="${VDIR_K100}"; SHARD=\$((ID - 253))
else                          K=200; VDIR="${VDIR_R32}";  SHARD=\$((ID - 353))
fi
if [ "\${MODE}" = "prep" ]; then
  echo "=== prep k=\${K} \${VDIR} node \$(hostname) \$(date) ==="
  NPY="\${VDIR}/results/smoke/retain_tr_scores.npy"
  if [ -f "\${NPY}" ]; then echo "KS reference exists, skipping"
  else
    ${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --smoke --model_name "${MODEL}" --output_dir "\${VDIR}" --k \${K} --forget_shard_id \$((K - 1)) --hf_home "${HF_HOME}"
    test -f "\${NPY}" || { echo "FATAL: reference missing"; exit 1; }
  fi
  if [ \${ID} -eq 2 ]; then mkdir -p "${VDIR_R32}/results/smoke"; cp -f "\${NPY}" "${VDIR_R32}/results/smoke/"; fi
else
  echo "=== train k=\${K} r\${RANK} shard \${SHARD} node \$(hostname) \$(date) ==="
  ${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --shard_id \${SHARD} --k \${K} --model_name "${MODEL}" --rank \${RANK} --alpha \${ALPHA} --epochs 5 --lr 1e-4 --batch_size 1 --grad_accum 16 --max_length 256 --output_dir "\${VDIR}" --hf_home "${HF_HOME}" --seed 42
fi
echo "=== done \$(date) ==="
EOF
)
echo "  stage1: ${S1} (553 tasks %4: 3 preps + 550 trains)"

# ---- Gates (sequential; each cancels its dependents on failure — see header) ----
G1=$(submit_job --dependency=afterany:${S1} <<EOF
#!/bin/bash
#SBATCH --job-name=scl-gate-r1
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/gate_r1_%j.log
#SBATCH --error=${LOG_DIR}/gate_r1_%j.log
${PROLOGUE}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +e
timeout 80m ${PYTHON} "${SCRIPT_DIR}/gate_scale_load.py" --model_name "${MODEL}" --output_dir "${VDIR_R1}" --k 200 --hf_home "${HF_HOME}"
RC=\$?
set -e
if [ \${RC} -ne 0 ]; then
  echo "GATE r1 (mechanics) FAILED (rc=\${RC}) — cancelling all k=200 downstream jobs"
  for f in gate2_jobid.txt evalB_jobid.txt evalC_jobid.txt; do { [ -f "${STATE}/\${f}" ] && scancel "\$(cat "${STATE}/\${f}")"; } || true; done
  exit 1
fi
EOF
)
echo "  gate r1 (mechanics): ${G1}"

G2=$(submit_job --dependency=afterok:${G1} <<EOF
#!/bin/bash
#SBATCH --job-name=scl-gate-r32
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/gate_r32_%j.log
#SBATCH --error=${LOG_DIR}/gate_r32_%j.log
${PROLOGUE}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +e
timeout 80m ${PYTHON} "${SCRIPT_DIR}/gate_scale_load.py" --model_name "${MODEL}" --output_dir "${VDIR_R32}" --k 200 --hf_home "${HF_HOME}"
RC=\$?
set -e
if [ \${RC} -ne 0 ]; then
  echo "GATE r32 (memory) FAILED (rc=\${RC}) — cancelling r32 evals, submitting r8 backup chain"
  { [ -f "${STATE}/evalC_jobid.txt" ] && scancel "\$(cat "${STATE}/evalC_jobid.txt")"; } || true
  bash "${SCRIPT_DIR}/submit_scale_grid.sh" backup_r8
  exit 1
fi
EOF
)
save_state gate2_jobid.txt "${G2}"
echo "  gate r32 (memory):   ${G2}"

# ---- Stage 3: eval arrays, %2 + %1 + %1 = 4 GPUs peak (global cap) ----
EA=$(submit_job --dependency=afterany:${S1} <<EOF
#!/bin/bash
#SBATCH --job-name=scl-ev-A
#SBATCH --partition=all
${EXCLUDE_LINE}
#SBATCH --array=0-13%2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/evA_%A_%a.log
#SBATCH --error=${LOG_DIR}/evA_%A_%a.log
${PROLOGUE}
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ \${SLURM_ARRAY_TASK_ID} -lt 7 ]; then K=50; VDIR="${VDIR_K50}"; else K=100; VDIR="${VDIR_K100}"; fi
F=\$((K - 1))
for i in \$(seq 0 \${F}); do test -f "\${VDIR}/shard_\${i}/adapter_config.json" || { echo "FATAL: shard \${i} adapter missing"; exit 1; }; done
test -f "\${VDIR}/results/smoke/retain_tr_scores.npy" || { echo "FATAL: KS reference missing"; exit 1; }
LABELS=(merged_linear merged_dare_ties remerge_linear remerge_dare_ties shard_\${F}_only routed_key_exact routed_key_exact_no\${F})
LABEL=\${LABELS[\$((SLURM_ARRAY_TASK_ID % 7))]}
echo "=== A eval \${LABEL} (k=\${K}) node \$(hostname) \$(date) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "\${VDIR}" --label "\${LABEL}" --k \${K} --forget_shard_id \${F} --out "\${VDIR}/results/smoke/\${LABEL}.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
)
save_state evalA_jobid.txt "${EA}"
echo "  eval A (k50+k100):   ${EA} (14 tasks %2)"

EB=$(submit_eval "B" "${VDIR_R1}" 200 "03:00:00" 1 1 --dependency=afterok:${G1})
save_state evalB_jobid.txt "${EB}"
echo "  eval B (k200 r1):    ${EB} (7 tasks %1)"

EC=$(submit_eval "C" "${VDIR_R32}" 200 "04:00:00" 1 1 --dependency=afterok:${G2})
save_state evalC_jobid.txt "${EC}"
echo "  eval C (k200 r32):   ${EC} (7 tasks %1)"

COLLECT=$(submit_collect "main" --dependency=afterany:${EA}:${EB}:${EC})
echo "  collect:             ${COLLECT}"
echo ""
echo "Monitor: squeue -u \$USER -o '%.12i %.18j %.8T %.8M %R'"
