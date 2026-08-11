#!/bin/bash
# H23 — does TOFU's forget_quality track the REROUTE DESTINATION rather than forgetting?
# (thread log/selector_audit/; E5 entry 2026-08-10)
#
# E5's two arms delete nothing and only redirect the deleted authors' queries to one fixed
# survivor. They bracket genuine deletion: s0 scores 0.6789 and s42 scores 0.3995 against
# deletion's 0.5789, at identical model_utility. So forget_quality moves 0.28 on a choice that
# has nothing to do with whether anything was forgotten.
#
# The obvious explanation is expert SIMILARITY — reroute to an author resembling the deleted one
# and the answers still look right, so the metric reports poor forgetting. This sweep tests that
# directly, and the two arms already in hand argue AGAINST it: s0 and s42 sit at affinity 0.2473
# and 0.2559, nearly identical, yet 0.28 apart in forget_quality. n=2 cannot settle it; seven
# stratified destinations can.
#
# Destinations are chosen by affinity = mean centroid_sbert score from the DELETED authors'
# queries to each surviving unit, computed on rl_family_k200.centroid_sbert.npz, spanning the
# full observed range 0.2193..0.3970:
#
#   88  nearest  (0.3970 — the no-name "sink" author of 2026-08-07, the most generic centroid)
#   137 near-2   (0.3382)      89  Q1 (0.3044)      31  median (0.2840)
#   33  Q3       (0.2663)      97  far-2 (0.2267)   79  farthest (0.2193)
#
# If forget_quality is monotone in affinity, §4.10 says the metric measures expert similarity.
# If it scatters, the metric responds to destination IDENTITY in a way similarity does not
# explain — a stronger claim, and the one the s0/s42 pair currently points to.
#
# Usage: bash submit_e5_destination_sweep.sh          # STUB=1 previews, PACK=n arms per job
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"          # never build a job body before this line
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
E25="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"
FORGET="180-199"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"
PACK="${PACK:-4}"
TOFU_GPUS_PER_NODE="${TOFU_GPUS_PER_NODE:-4}"
# "del" is not a destination — it is the genuine-deletion baseline (experts dropped, orphans fall
# through to base+scaffold). It rides in the same list so the baseline is always measured at the
# SAME tier as the reroute arms it is compared against; comparing a reroute arm at one tier to a
# deletion arm at another would compare two different KS tests.
DESTS="${DESTS:-88,137,89,31,33,97,79}"
# TIER picks the eval caps AND the results dir. smoke = truth_max_rows 30, extended = 120.
# This matters more than it looks: forget_quality is ks_2samp(forget_tr, retain_ref), so the caps
# ARE the test's sample sizes. At smoke (30 vs a 20-row reference) the p-value can only take 34
# distinct values and adjacent rungs in the readable range are ~0.10 apart — so a 0.62 spread
# across destinations is only ~8 rungs, and the 4-decimal precision in the JSON is spurious.
TIER="${TIER:-smoke}"
# The walltime is tier-dependent because the caps are: extended is 4x the truth rows, 4x rouge and
# 5x retain of smoke, and smoke arms already run ~25 min. A TIMEOUT costs the whole arm AND holds
# a GPU for the full limit, which is the expensive way to find this out.
case "${TIER}" in
  smoke)    TIER_FLAG="--smoke";    WALL="04:00:00" ;;
  extended) TIER_FLAG="--extended"; WALL="12:00:00" ;;
  *) echo "TIER must be smoke or extended (got '${TIER}')" >&2; exit 1 ;;
esac
LOG_DIR="${CKPT}/e5_sweep_logs"
RES="${E25}/results/${TIER}"
mkdir -p "${LOG_DIR}" "${RES}"

if [ "${PACK}" -gt "${TOFU_GPUS_PER_NODE}" ]; then
  echo "submit_e5_destination_sweep: PACK=${PACK} exceeds TOFU_GPUS_PER_NODE=${TOFU_GPUS_PER_NODE}." >&2
  echo "  The packed dispatcher pins CUDA_VISIBLE_DEVICES within ONE node." >&2
  exit 1
fi
# forget_quality is a KS test against this reference; without it every cell is NaN and the
# sweep would produce a full table of nothing. Build it with
#   prepare_eval.py --${TIER} --output_dir <a pool with a retain90/> --k 200
# and copy it in; the e25 pool has no retain90 of its own and borrows the e5 oracle's reference,
# which is the convention submit_k200_routed.sh already established for the smoke tier.
if [ ! -f "${RES}/retain_tr_scores.npy" ]; then
  echo "missing ${RES}/retain_tr_scores.npy — forget_quality would be NaN." >&2
  exit 1
fi
# The reference is the OTHER sample in the KS test, so its length caps the metric's resolution
# no matter how many forget rows the tier scores. Print it rather than letting a 20-row reference
# silently bound an "extended" run.
echo "KS reference: $(${PYTHON} -c "import numpy;print(numpy.load('${RES}/retain_tr_scores.npy').shape[0])") rows (tier=${TIER})"

IFS=',' read -r -a ARM_LIST <<< "${DESTS}"
NARMS=${#ARM_LIST[@]}
NJOBS=$(( (NARMS + PACK - 1) / PACK ))

body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=e5-sweep
#SBATCH --array=0-$((NJOBS-1))%${ARRAY_CAP}
$(tofu_sbatch_resources ${PACK} $((8 * PACK)) 48G)
#SBATCH --time=${WALL}
#SBATCH --output=${LOG_DIR}/sweep_%A_%a.log
#SBATCH --error=${LOG_DIR}/sweep_%A_%a.log
set -eo pipefail
DESTS=(${ARM_LIST[@]})
PACK=${PACK}

run_arm() {
  local T=\$1 SLOT=\$2
  local D=\${DESTS[\$T]}
  if [ "\${PACK}" -gt 1 ]; then
    exec > "${LOG_DIR}/sweep_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}_s\${D}.log" 2>&1
  fi
  export CUDA_VISIBLE_DEVICES=\${SLOT}
  # eval_tofu._rouge_metric_cache falls back to one path per JOB_ID; packed arms would clobber
  # each other's .arrow file.
  export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}_\${SLOT}"
  mkdir -p "\${TOFU_METRICS_CACHE}"
  export PYTHONUNBUFFERED=1
  export HF_HOME="${HF_HOME}"
  if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
  export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
  echo "=== E5 destination sweep: reroute_to \${D} (gpu slot \${SLOT}) ==="; date

  local OUT POLICY
  if [ "\${D}" = "del" ]; then
    OUT="${RES}/routed_oracle_del_f10.json"; POLICY=""          # genuine deletion, no reroute
  else
    OUT="${RES}/routed_reroute_f10_s\${D}.json"; POLICY="--reroute_to \${D}"
  fi
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; return 0; }

  # A missing shard would silently route an author to the base and look like a deletion.
  for i in \$(seq 0 199); do
    [ -f "${E25}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING shard_\${i}"; return 1; }
  done

  ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
    --model_name "${MODEL}" --shards_dir "${E25}" --k 200 --forget_shard_id 199 \\
    --forget_author_ids "${FORGET}" --lazy_adapter_cache 8 ${TIER_FLAG} --hf_home "${HF_HOME}" \\
    --delete_shards "${FORGET}" \${POLICY} --out "\${OUT}"
  date
}

FIRST=\$(( \${SLURM_ARRAY_TASK_ID} * PACK ))
rc=0
pids=(); slots=()
for s in \$(seq 0 \$((PACK - 1))); do
  T=\$(( FIRST + s ))
  [ "\${T}" -lt \${#DESTS[@]} ] || break
  run_arm "\${T}" "\${s}" &
  pids+=("\$!"); slots+=("\${T}")
done
for i in \$(seq 0 \$(( \${#pids[@]} - 1 ))); do
  if ! wait "\${pids[\$i]}"; then echo "ARM \${slots[\$i]} FAILED"; rc=1; fi
done
exit \${rc}
EOF
}

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB (not submitted) -----" >&2
  body >&2
else
  jid="$(body | sbatch --parsable ${DEP:+--dependency=afterany:${DEP}})"
  echo "e5 destination sweep: ${NARMS} arms (${DESTS}) in ${NJOBS} job(s) of ${PACK} GPU(s) -> ${jid}"
fi
