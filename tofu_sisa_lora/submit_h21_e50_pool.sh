#!/bin/bash
# H21 — a third point on the EPOCHS axis, at fixed rank (thread log/selector_audit/).
#
# H20 separated the two axes that H18 had confounded. Holding epochs at 5 and raising rank
# 8 -> 32 does NOT degrade behavioral orphan detectability (activation_norm 0.877 -> 0.934);
# holding rank at 32 and raising epochs 5 -> 25 collapses it (0.934 -> 0.608). So it is training
# duration, not capacity, that blunts the behavioral leak.
#
# Two points cannot tell a slide from a floor from an overshoot, and "train longer and the leak
# goes away" is a defense-shaped claim the paper should not make off one interval. e50 is the
# cheapest third point: same rank, same lr, same data, one knob doubled from the pool that
# already exists.
#
#   e5 -> e25 collapsed        => e50 keeps falling  ⇒ monotone: duration really does blunt it
#   e5 -> e25 collapsed        => e50 flat at ~0.6   ⇒ a FLOOR, and 0.608 is where it saturates
#   e5 -> e25 collapsed        => e50 recovers       ⇒ non-monotone, and the H20 story is wrong
#
# Each outcome is publishable and they are mutually exclusive, which is what makes this worth
# 200 GPU-tasks. Whatever lands, the epochs axis stops being a line through two points.
#
# Usage: bash submit_h21_e50_pool.sh [train|check|wave|all]
#   train  200 authors, PACK per job, self-skipping any shard already on disk
#   check  CPU: assert the pool is complete and every adapter is non-empty
#   wave   the behavioral selector wave on the new pool (gold + name_stripped + indirect)
#   all    train -> check -> wave, each chained --dependency=afterany
# STUB=1 previews without submitting. WALL overrides the per-job walltime.
#
# ⚠ afterany, never afterok: kill_invalid_depend is off cluster-wide, so an afterok chain hangs
#   PENDING forever on the first failure instead of reporting what is missing.
set -euo pipefail

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"          # never build a job body before this line
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
E50="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e50_lr1e4"
NAUTHORS=200
EPOCHS=50
ARRAY_CAP="${ARRAY_CAP:-4}"
PACK="${PACK:-4}"
TOFU_GPUS_PER_NODE="${TOFU_GPUS_PER_NODE:-4}"
# Calibrated on author 0 (job 3210540): 7m29s wall, of which the TRAINING was 30 seconds — a
# k=200 shard is one author, 20 rows, so 50 epochs is 50 optimizer steps and the 7B model load
# dominates completely. That is also why PACK is worth it and why the wall carries headroom:
# four authors per node means four concurrent base-model pulls off NFS, and NFS contention is
# what timed out the r32 behavioral arms at 6h on 2026-08-10. A TIMEOUT costs the whole task AND
# holds its GPU for the full limit, so this is ~6x the measured single-shard time.
WALL="${WALL:-01:30:00}"
LOG_DIR="${CKPT}/k200_e50_logs"
mkdir -p "${LOG_DIR}" "${E50}"

if [ "${PACK}" -gt "${TOFU_GPUS_PER_NODE}" ]; then
  echo "submit_h21_e50_pool: PACK=${PACK} exceeds TOFU_GPUS_PER_NODE=${TOFU_GPUS_PER_NODE}." >&2
  echo "  The packed dispatcher pins CUDA_VISIBLE_DEVICES within ONE node and cannot reach" >&2
  echo "  GPUs on a second one." >&2
  exit 1
fi
# PACK x ARRAY_CAP is the concurrent GPU count and the association caps it at gres/gpu=16 with
# MaxJobs=6. 4x4 sits exactly on the GPU limit using 4 of the 6 job slots; PACK=1 would strand
# 10 GPUs behind the job-count limit, which is the whole reason the packed dispatcher exists.
if [ $(( PACK * ARRAY_CAP )) -gt 16 ]; then
  echo "PACK*ARRAY_CAP = $(( PACK * ARRAY_CAP )) exceeds the 16-GPU association limit." >&2
  exit 1
fi
NJOBS=$(( (NAUTHORS + PACK - 1) / PACK ))

submit() {  # $1 = body, $2 = optional dependency job id; echoes the job id
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----" >&2
    printf '%s\n' "$1" >&2
    echo "-----------------------------------------------" >&2
    echo "STUB"
  else
    printf '%s\n' "$1" | sbatch --parsable ${2:+--dependency=afterany:$2}
  fi
}

train_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=e50-train
#SBATCH --array=0-$((NJOBS-1))%${ARRAY_CAP}
$(tofu_sbatch_resources ${PACK} $((8 * PACK)) 48G)
#SBATCH --time=${WALL}
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
set -eo pipefail
PACK=${PACK}

run_author() {
  local A=\$1 SLOT=\$2
  if [ "\${PACK}" -gt 1 ]; then
    exec > "${LOG_DIR}/train_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}_a\${A}.log" 2>&1
  fi
  export CUDA_VISIBLE_DEVICES=\${SLOT}
  export PYTHONUNBUFFERED=1
  export HF_HOME="${HF_HOME}"
  if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
  export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
  # Self-skip lets the array be resubmitted after a partial night without redoing finished work,
  # and is what makes the calibration shard (author 0) free.
  if [ -f "${E50}/shard_\${A}/adapter_model.safetensors" ]; then
    echo "shard_\${A} already trained — skip"; return 0
  fi
  echo "=== H21 e50 train author \${A} (gpu slot \${SLOT}) ==="; date
  ${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" \\
    --shard_id "\${A}" --k 200 --model_name "${MODEL}" \\
    --output_dir "${E50}" --epochs ${EPOCHS} --hf_home "${HF_HOME}"
  date
}

FIRST=\$(( \${SLURM_ARRAY_TASK_ID} * PACK ))
rc=0
pids=(); who=()
for s in \$(seq 0 \$((PACK - 1))); do
  A=\$(( FIRST + s ))
  [ "\${A}" -lt ${NAUTHORS} ] || break
  run_author "\${A}" "\${s}" &
  pids+=("\$!"); who+=("\${A}")
done
for i in \$(seq 0 \$(( \${#pids[@]} - 1 ))); do
  if ! wait "\${pids[\$i]}"; then echo "AUTHOR \${who[\$i]} FAILED"; rc=1; fi
done
exit \${rc}
EOF
}

check_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=e50-check
$(tofu_sbatch_resources 0 2 8G)
#SBATCH --time=00:20:00
#SBATCH --output=${LOG_DIR}/check_%j.log
#SBATCH --error=${LOG_DIR}/check_%j.log
set -eo pipefail
# CPU only (gpus=0 emits no --gres, so this never touches the GPU cap). A missing or truncated
# adapter would make the wave route that author to the base and read as a deletion, so the pool
# is proved complete BEFORE any GPU time is spent auditing it.
export HF_HOME="${HF_HOME}"
${PYTHON} - <<'PY'
import os, sys
d = "${E50}"
missing, empty = [], []
for a in range(${NAUTHORS}):
    p = os.path.join(d, f"shard_{a}", "adapter_model.safetensors")
    if not os.path.exists(p):
        missing.append(a)
    elif os.path.getsize(p) < 1_000_000:      # a real r32 7B adapter is ~258 MB
        empty.append((a, os.path.getsize(p)))
print(f"pool {d}")
print(f"  present {${NAUTHORS} - len(missing)}/{${NAUTHORS}}")
if missing:
    print(f"  MISSING {len(missing)}: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
if empty:
    print(f"  TRUNCATED {len(empty)}: {empty[:10]}")
sys.exit(1 if (missing or empty) else 0)
PY
echo "pool complete"
EOF
}

case "${STAGE}" in
  train)  T="$(submit "$(train_body)")"; echo "train: ${NAUTHORS} authors in ${NJOBS} job(s) x ${PACK} GPU -> ${T}" ;;
  check)  C="$(submit "$(check_body)" "${DEP:-}")"; echo "check -> ${C}" ;;
  wave)
    # ONLY=beh_e50 matters even though every stage self-skips: a skipped arm still pays a full
    # 7B model load, and the three older pools have nothing to add here.
    ONLY=beh_e50 DEP="${DEP:-}" QT="${QT:-none}" bash "${SCRIPT_DIR}/submit_selector_wave.sh" beh
    ;;
  all)
    T="$(submit "$(train_body)")"
    echo "train: ${NAUTHORS} authors in ${NJOBS} job(s) x ${PACK} GPU (wall ${WALL}) -> ${T}"
    C="$(submit "$(check_body)" "${T}")"
    echo "check (CPU, afterany:${T}) -> ${C}"
    echo "then, once check is green:"
    echo "  ONLY=beh_e50 DEP=${C} QT=none           bash submit_selector_wave.sh beh"
    echo "  ONLY=beh_e50 DEP=${C} QT=name_stripped  bash submit_selector_wave.sh beh"
    echo "  ONLY=beh_e50 DEP=${C} QT=indirect       bash submit_selector_wave.sh beh"
    ;;
  *) echo "usage: bash submit_h21_e50_pool.sh [train|check|wave|all]  (STUB=1 previews, PACK=n, WALL=hh:mm:ss)" >&2; exit 1 ;;
esac
