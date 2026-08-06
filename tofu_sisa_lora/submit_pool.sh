#!/bin/bash
# Rebuild the Llama-2-7B per-author LoRA pools and their anchors on any site.
#
#   bash submit_pool.sh <arm>        arm = anchors | pilot | r32 | e25 | r8
#   STUB=1 bash submit_pool.sh r32   # preview, do not submit
#   DEP=<jobid> bash submit_pool.sh r32
#
# Recipes are copied VERBATIM from the drivers that built the originals, so a rebuilt pool
# matches: submit_scale_grid.sh:270 (r32), :185 (r8), submit_k200_routed.sh:75 (e25),
# submit_llama2_ft_vs_base.sh:68 (retain90 oracle). Provenance is cited per arm below.
#
#   anchors  ft (the mu 0.756 reference) + the retain90 KS oracle — 2 trains.
#            RUN THIS FIRST: every pool's forget_quality reference derives from the oracle.
#   pilot    the first 2 authors of the r32 recipe — a cheap end-to-end gate before fanning out.
#   r32      200 per-author adapters, rank 32 / alpha 64, 5 epochs   (the main pool)
#   e25      200 per-author adapters, same rank, 25 epochs           (Experiment B needs this:
#            the e5 adapters have almost no ROUGE headroom — iso own-author 0.402-0.575 against
#            a base floor of 0.404 — so a forget-quality gap would be unmeasurable on them)
#   r8       200 per-author adapters, rank 8 / alpha 16              (the legacy-convention pool)
#
# Measured cost (sprint A40s, 2026-07-28): 40-46 s per shard, dominated by process startup, not
# training. Whole set ~6.5 GPU-hours. Trains SELF-SKIP on an existing adapter_config.json, so
# re-running after a partial failure is safe and nearly free.
#
# Simpler than merge-tables-7b/tofu_sisa_lora/submit_pool_7b.sh, which packs 2 trains per task
# behind run_pair.sh for CISPA's 2-GPU tasks. One train per array task needs no packing
# machinery and is portable; raise the array cap instead if you have the GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

ARM="${1:?arm required: anchors | pilot | r32 | e25 | r8}"
PYTHON="${PYTHON:-${TOFU_PYTHON}}"
MODEL="${TOFU_MODEL:-meta-llama/Llama-2-7B-chat-hf}"
CKPT="${TOFU_CKPT_ROOT}"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"
TRAIN_TIME="${TRAIN_TIME:-04:00:00}"

# Frozen recipe shared by the pool arms. batch 1 x grad_accum 16 = effective 16.
COMMON="--batch_size 1 --grad_accum 16 --max_length 256 --seed 42"

case "${ARM}" in
  r32)   VDIR="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4";  RANK=32; ALPHA=64; EPOCHS=5;  N=200 ;;
  e25)   VDIR="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"; RANK=32; ALPHA=64; EPOCHS=25; N=200 ;;
  r8)    VDIR="${CKPT}/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4";   RANK=8;  ALPHA=16; EPOCHS=5;  N=200 ;;
  pilot) VDIR="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4";  RANK=32; ALPHA=64; EPOCHS=5;  N=2   ;;
  anchors) ;;
  *) echo "unknown arm '${ARM}' (want anchors|pilot|r32|e25|r8)"; exit 2 ;;
esac

submit() {
  if [ "${STUB:-0}" = "1" ]; then echo "===== STUB sbatch (not submitted) ====="; cat; return; fi
  local dep=(); [ -n "${DEP:-}" ] && dep=(--dependency="afterany:${DEP}")
  sbatch --parsable "${dep[@]}"
}

# ── anchors ──────────────────────────────────────────────────────────────────
if [ "${ARM}" = "anchors" ]; then
  FT_DIR="${CKPT}/Llama-2-7B-chat-hf_ft_lr1e4_e5_r32"   # the mu 0.756 reference
  OR_DIR="${CKPT}/Llama-2-7B-chat-hf"                    # holds retain90/ (the KS oracle)
  LOG_DIR="${CKPT}/anchor_logs"
  # Under STUB the target filesystem may be absent (previewing another site from here).
  mkdir -p "${LOG_DIR}" "${FT_DIR}" "${OR_DIR}" 2>/dev/null || \
    { [ "${STUB:-0}" = "1" ] || { echo "cannot create ${LOG_DIR}"; exit 1; }; }
  echo "anchors: 2 tasks -> ${FT_DIR}/shard_0 (r32/a64/e5, k=1, ga32)"
  echo "                   ${OR_DIR}/retain90 (r8/a16/e3/lr2e-4 — LEGACY, see below)"
  submit <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-anchors
#SBATCH --array=0-1%${ARRAY_CAP}
$(tofu_sbatch_resources "${TOFU_GPUS_PER_TASK}" "${TOFU_CPUS_PER_TASK}" 48G)
#SBATCH --time=${TRAIN_TIME}
#SBATCH --output=${LOG_DIR}/anchor_%A_%a.log
#SBATCH --error=${LOG_DIR}/anchor_%A_%a.log
set -eo pipefail
$(tofu_job_prologue)
if [ "\${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  # Full-data k=1 LoRA = the mu 0.756 anchor. grad_accum 32 here, NOT 16
  # (submit_llama2_grid_overnight.sh:71, VARIANTS row "1e-4 5 32 64 lr1e4_e5_r32").
  if [ -f "${FT_DIR}/shard_0/adapter_config.json" ]; then echo "ft exists, skipping"; exit 0; fi
  ${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --shard_id 0 --k 1 --model_name "${MODEL}" \\
    --rank 32 --alpha 64 --epochs 5 --lr 1e-4 \\
    --batch_size 1 --grad_accum 32 --max_length 256 --seed 42 \\
    --output_dir "${FT_DIR}" --hf_home "\${HF_HOME}"
else
  # ⚠ The KS oracle keeps the LEGACY r8/a16/e3/lr2e-4 recipe DELIBERATELY: every existing
  # forget_quality reference was built at r8. Do not "fix" it to match the r32 pool — that
  # silently moves every forget_quality number in the repo.
  if [ -f "${OR_DIR}/retain90/adapter_config.json" ]; then echo "retain90 exists, skipping"; exit 0; fi
  ${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --retain90 --k 10 --model_name "${MODEL}" \\
    --rank 8 --alpha 16 --epochs 3 --lr 2e-4 \\
    --batch_size 1 --grad_accum 16 --max_length 256 --seed 42 \\
    --output_dir "${OR_DIR}" --hf_home "\${HF_HOME}"
fi
date
EOF
  exit 0
fi

# ── pool arms: one per-author train per array task ───────────────────────────
LOG_DIR="${VDIR}/logs"
mkdir -p "${LOG_DIR}" "${VDIR}/results/smoke" 2>/dev/null || \
  { [ "${STUB:-0}" = "1" ] || { echo "cannot create ${LOG_DIR}"; exit 1; }; }

if [ ! -f "${CKPT}/Llama-2-7B-chat-hf/retain90/adapter_config.json" ]; then
  echo "NOTE: the retain90 oracle is missing — run 'bash submit_pool.sh anchors' first, or"
  echo "      forget_quality will be NaN for everything trained here."
fi

echo "${ARM}: ${N} tasks (%${ARRAY_CAP}) -> ${VDIR}  (r${RANK}/a${ALPHA}/e${EPOCHS}/lr1e-4, k=200)"
submit <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-pool-${ARM}
#SBATCH --array=0-$((N - 1))%${ARRAY_CAP}
$(tofu_sbatch_resources "${TOFU_GPUS_PER_TASK}" "${TOFU_CPUS_PER_TASK}" 48G)
#SBATCH --time=${TRAIN_TIME}
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
set -eo pipefail
$(tofu_job_prologue)
A=\${SLURM_ARRAY_TASK_ID}
if [ -f "${VDIR}/shard_\${A}/adapter_config.json" ]; then
  echo "shard_\${A} already trained — skip"; exit 0
fi
${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --shard_id \${A} --k 200 --model_name "${MODEL}" \\
  --rank ${RANK} --alpha ${ALPHA} --epochs ${EPOCHS} --lr 1e-4 ${COMMON} \\
  --output_dir "${VDIR}" --hf_home "\${HF_HOME}"
date
EOF
