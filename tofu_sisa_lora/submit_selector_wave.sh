#!/bin/bash
# selector_audit wave — the GPU work the E1 pilot opened up.
# (thread log/selector_audit/; E1 entry 2026-08-07.)
#
# E1 found that the published "confidence refusal caps at AUC 0.57-0.61" is a k=10 property:
# at k=200 per-author units the same detectors reach 0.98. That finding currently rests on
# THREE feature-space strategies on ONE pool, which is not enough to carry a section. This wave
# widens it on both axes that are actually reachable:
#
#   BEH*   the behavioral family (ppl / activation_norm / attn_norm) at k=200 — never run at
#          this granularity anywhere, because 200 x r32 adapters fp32-cast to ~65 GiB. Their
#          loop is shard-OUTER, so --lazy_adapter_cache costs k loads for the whole run.
#          logit_div stays out: it activates every shard per query batch and caches a logits
#          tensor per shard, which no cache size fixes. At k=10 this family was the LEAKIEST
#          (AUC 0.41-0.63); whether granularity rescues it is the open half of the finding.
#   FEAT*  the feature-space battery on the two k=200 pools that were never audited (r32 e5,
#          r8 e5). Same granularity, different training recipe and rank — this is what
#          separates "granularity causes detectability" from "the e25 recipe does".
#
# Usage: bash submit_selector_wave.sh [beh|feat|all]     # STUB=1 previews
#   PACK=n  arms per job, one per allocated GPU (default 4; converts the MaxJobs limit into
#           the GPU limit — the association here allows gres/gpu=16 but only 6 running jobs).
# Check `squeue -u $USER` against gres/gpu=16 before submitting; this wave asks for PACK GPUs
# per job and does not know what else is queued.
set -euo pipefail

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
BASE="meta-llama/Llama-2-7B-chat-hf"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"
PACK="${PACK:-1}"
# 2026-08-10: both 6h waves timed out with only their r8 arms finished. The bottleneck is NFS —
# three arms each streaming 200 r32 adapters at once — not GPU. So the default is now ONE arm per
# job and one job at a time, with a wall long enough for the r32 pools.
WAVE_TIME="${WAVE_TIME:-12:00:00}"
WAVE_THROTTLE="${WAVE_THROTTLE:-1}"
TOFU_GPUS_PER_NODE="${TOFU_GPUS_PER_NODE:-4}"
QT_SUF=""; [ "${QT:-none}" = "none" ] || QT_SUF="_${QT}"
LOG_DIR="${CKPT}/selector_wave_logs"
mkdir -p "${LOG_DIR}"

# forget10 = authors 180-199; at k=200 a unit IS an author, so the drop set is the 20 of them.
DROPS="199;180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199"
# 400 forget + a RandomState(42) 400-retain sample. The behavioral family costs k forwards per
# query, so `--queries all` (4000) would be 5x this for no extra statistical power on a cell
# whose unit of resampling is the AUTHOR, not the query.
QUERIES="${QUERIES:-sample}"
# H11: the feature-space family's k=200 detectability turned out to be lexical (0.991 -> 0.623
# name-stripped). The behavioral family scores by RUNNING each expert rather than matching text,
# and unlike the feature-space one it cannot be tested without the pool — so the transform has to
# run here. `none` reproduces the gold-form arms byte-for-byte.
QT="${QT:-none}"

if [ "${PACK}" -gt "${TOFU_GPUS_PER_NODE}" ]; then
  echo "submit_selector_wave: PACK=${PACK} exceeds TOFU_GPUS_PER_NODE=${TOFU_GPUS_PER_NODE}." >&2
  echo "  The packed dispatcher pins CUDA_VISIBLE_DEVICES within ONE node and cannot reach" >&2
  echo "  GPUs on a second one. Use PACK<=${TOFU_GPUS_PER_NODE} and more concurrent jobs." >&2
  exit 1
fi

# arm := <name>|<pool_dir>|<k>|<strategies>|<out_stem>|<self_check_n>
#
# self_check is the faithfulness gate: for N seeded queries it asserts the score-row argmax
# equals what router.route() actually serves. It is never disabled — but its cost is N x k
# adapter activations, because PplRouter/ActivationRouter score every candidate per query. At
# k=200 under a lazy cache the default N=50 would be 10,000 loads, dwarfing the 200-load audit
# it is checking. N=3 keeps the check (600 loads) at a cost proportional to the run.
# Feature-space arms never touch adapters, so they keep the full N=50.
POOL_E25="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"
POOL_E5R32="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4"
POOL_E5R8="${CKPT}/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4"
# H21 (2026-08-12): the third point on the EPOCHS axis at fixed rank. H20 separated the axes —
# rank 8->32 at e5 does not degrade behavioral detectability (activation_norm 0.877 -> 0.934)
# while epochs 5->25 at r32 collapses it (0.934 -> 0.608) — but two points cannot say whether
# that is a slide, a floor, or an overshoot, and "training longer hides the leak" is a claim the
# paper would otherwise make off a single interval. Built by submit_h21_e50_pool.sh.
POOL_E50="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e50_lr1e4"
BEH_STRATS="ppl activation_norm attn_norm"
FEAT_STRATS="key_exact key_tfidf centroid_sbert centroid_lm"

BEH_ARMS=(
  "beh_e25|${POOL_E25}|200|${BEH_STRATS}|rl_family_k200_beh|3"
  "beh_e5r32|${POOL_E5R32}|200|${BEH_STRATS}|rl_family_k200_beh|3"
  "beh_e5r8|${POOL_E5R8}|200|${BEH_STRATS}|rl_family_k200_beh|3"
  "beh_e50|${POOL_E50}|200|${BEH_STRATS}|rl_family_k200_beh|3"
)
# feat_e25 added 2026-08-10. The battery originally covered only the two pools that had never
# been audited, on the assumption the e25 pool was already done — but the e25 audit predates the
# FAMILY NPZ CONTRACT and left no rl_family_k200.*.npz on disk, so the HEADLINE pool was the one
# pool with no feature-space matrices. Every cross-family table therefore had to compare the e25
# behavioral numbers against feature numbers from a different recipe.
# ⚠ There is deliberately NO feat_e50 arm, and adding one would repeat the 2026-08-07 mistake.
# No feature-space router reads expert weights — key_* are text-only, centroid_sbert is MiniLM
# over questions, and centroid_lm uses the base with adapters DISABLED — so a feature arm on a
# new pool returns matrices byte-identical to the ones already on disk (np.array_equal True
# across r32/e25, r32/e5 and r8/e5). The epochs axis can only be measured by the BEHAVIORAL
# family, which is the one that actually runs the candidate experts.
FEAT_ARMS=(
  "feat_e25|${POOL_E25}|200|${FEAT_STRATS}|rl_family_k200|50"
  "feat_e5r32|${POOL_E5R32}|200|${FEAT_STRATS}|rl_family_k200|50"
  "feat_e5r8|${POOL_E5R8}|200|${FEAT_STRATS}|rl_family_k200|50"
)

# ONLY= a comma-separated list of arm names, e.g. ONLY=beh_e25,feat_e25 — for requeueing the
# arms that timed out without redoing the one that finished (every stage self-skips anyway,
# but a skipped arm still costs a model load).
#
# This filters BOTH arrays. It used to filter only BEH_ARMS, which meant `ONLY=beh_e25 ... all`
# silently ran all three feature arms as well — the filter appeared to work because the beh
# stage obeyed it and the feat stage's extra arms self-skipped on existing output. With
# feat_e25 and the query transforms there is now feature work that does NOT self-skip, so the
# half-applied filter would submit real unintended arms.
if [ -n "${ONLY:-}" ]; then
  _filter_arms() {   # echoes the matching arm specs, one per line
    local a
    for a in "$@"; do
      case ",${ONLY}," in *",${a%%|*},"*) printf '%s\n' "$a" ;; esac
    done
  }
  mapfile -t BEH_ARMS < <(_filter_arms "${BEH_ARMS[@]}")
  mapfile -t FEAT_ARMS < <(_filter_arms "${FEAT_ARMS[@]}")
  [ $(( ${#BEH_ARMS[@]} + ${#FEAT_ARMS[@]} )) -gt 0 ] || {
    echo "ONLY=${ONLY} matched no arm" >&2; exit 1; }
fi

submit() {
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----" >&2
    printf '%s\n' "$1" >&2
    echo "-----------------------------------------------" >&2
    echo "STUB"
  else
    printf '%s\n' "$1" | sbatch --parsable ${2:+--dependency=afterany:$2}
  fi
}

wave_body() {   # $1 = job tag, $2.. = arm specs
  local tag="$1"; shift
  local arms=("$@")
  local n=${#arms[@]}
  local njobs=$(( (n + PACK - 1) / PACK ))
  local spec_lines=""
  for a in "${arms[@]}"; do spec_lines+="\"${a}\" "; done
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=sw-${tag}${QT_SUF}
#SBATCH --array=0-$((njobs-1))%${WAVE_THROTTLE}
$(tofu_sbatch_resources ${PACK} $((8 * PACK)) 64G)
#SBATCH --time=${WAVE_TIME}
#SBATCH --output=${LOG_DIR}/${tag}${QT_SUF}_%A_%a.log
#SBATCH --error=${LOG_DIR}/${tag}${QT_SUF}_%A_%a.log
set -eo pipefail
ARMS=(${spec_lines})
PACK=${PACK}

run_arm() {
  local T=\$1 SLOT=\$2
  IFS='|' read -r NAME POOL K STRATS STEM SELFCHECK <<< "\${ARMS[\$T]}"
  if [ "\${PACK}" -gt 1 ]; then
    exec > "${LOG_DIR}/${tag}${QT_SUF}_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}_\${NAME}.log" 2>&1
  fi
  export CUDA_VISIBLE_DEVICES=\${SLOT}
  export PYTHONUNBUFFERED=1
  export HF_HOME="${HF_HOME}"
  if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
  export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
  local RL="\${POOL}/results/router_leak"
  mkdir -p "\${RL}"
  local OUT="\${RL}/\${STEM}${QT_SUF}.json"
  echo "=== selector wave arm \${NAME} (gpu slot \${SLOT}): k=\${K} [\${STRATS}] -> \${OUT} ==="
  date
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }
  # A missing shard would silently corrupt a behavioral score column; the audit refuses, but
  # fail here first so the reason is one line rather than 200 into a log.
  for i in \$(seq 0 \$((K-1))); do
    [ -f "\${POOL}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING \${POOL}/shard_\${i}"; return 1; }
  done
  # --lazy_adapter_cache is required for the behavioral family at k=200 and harmless for the
  # feature-space one (which never loads adapters at all).
  ${PYTHON} "${SCRIPT_DIR}/router_family_audit.py" \\
    --pool_dir "\${POOL}" --base_model "${BASE}" --k "\${K}" \\
    --strategies \${STRATS} --drop_sets "${DROPS}" --queries "${QUERIES}" \\
    --device cuda --lazy_adapter_cache 8 --dump_sims --self_check "\${SELFCHECK}" \\
    --query_transform "${QT}" \\
    --hf_home "${HF_HOME}" --out "\${OUT}"
  date
}

FIRST=\$(( \${SLURM_ARRAY_TASK_ID} * PACK ))
rc=0
pids=(); slots=()
for s in \$(seq 0 \$((PACK - 1))); do
  T=\$(( FIRST + s ))
  [ "\${T}" -lt \${#ARMS[@]} ] || break
  run_arm "\${T}" "\${s}" &
  pids+=("\$!"); slots+=("\${T}")
done
for i in \$(seq 0 \$(( \${#pids[@]} - 1 ))); do
  if ! wait "\${pids[\$i]}"; then echo "ARM \${slots[\$i]} FAILED"; rc=1; fi
done
exit \${rc}
EOF
}

case "${STAGE}" in
beh)
  echo "selector wave / behavioral at k=200: ${#BEH_ARMS[@]} arms, PACK=${PACK}"
  submit "$(wave_body beh "${BEH_ARMS[@]}")" "${DEP:-}" ;;
feat)
  echo "selector wave / feature-space recipe ablation: ${#FEAT_ARMS[@]} arms, PACK=${PACK}"
  submit "$(wave_body feat "${FEAT_ARMS[@]}")" "${DEP:-}" ;;
all)
  ALL_ARMS=("${BEH_ARMS[@]}" "${FEAT_ARMS[@]}")
  echo "selector wave / all: ${#ALL_ARMS[@]} arms, PACK=${PACK}"
  submit "$(wave_body all "${ALL_ARMS[@]}")" "${DEP:-}" ;;
*) echo "usage: bash submit_selector_wave.sh [beh|feat|all]  (STUB=1 previews, PACK=n, QT=none|name_stripped|indirect)"; exit 1 ;;
esac
