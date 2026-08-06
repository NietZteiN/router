#!/bin/bash
# router_leak Wave 4 (2026-07-23 follow-up to the Wave-2/3 results).
#   bash submit_router_wave4.sh [ceiling|families|seal|all]
# Three things:
#  (1) ceiling  — ONE router-INDEPENDENT ceiling (`--channels expert_max --drop_shard none`,
#                 max answer-prob over experts, no routing). Shared by every family via
#                 `aggregate_rho.py --ceiling_channel expert_max`. Fixes the activation_norm
#                 degeneracy (its magnet misroutes even with no drop => ceiling≈floor) AND makes
#                 rho comparable ACROSS families.
#  (2) families — Mode-B post+floor for the two unrun routers (attn_norm, logit_div) so the
#                 per-family table is exhaustive (rho computed later against the shared ceiling).
#  (3) seal     — the ppl-NATIVE seal arm: abstain on the router's own margin (tau calibrated on
#                 RETAIN margins only: p1 = 0.5473 -> 91.5% generic-orphan catch @ 1% retain FPR,
#                 reproducing the Wave-0 published ppl operating point FPR@90catch=0.010).
#                 PRE-REGISTERED PREDICTION (H-SEAL-PPL): ppl is CONTENT-SEEKING (host-hit 0.853
#                 vs 0.46-0.50 for geometric routers), so on a replicated fact a surviving host
#                 fits WELL => large margin => NO abstain => the seal MISSES exactly the
#                 replicated facts that carry the privacy risk. CONFIRM if sealed rho@R8 stays
#                 >=0.5 while generic-orphan catch is ~0.9; REFUTE if rho collapses like the
#                 author-tombstone did (0.833 -> 0.047).
#
# GLOBAL 4-GPU CAP: 1 GPU per job, chained SERIALLY (each afterany the previous) => at most one
# of ours eligible. STUB=1 previews. Self-skips existing outputs.
# CPU gates first: python test_router_leak.py ; python test_entangled_facts.py
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PY=${TOFU_PYTHON:-python3}
CKPT=${TOFU_CKPT_ROOT}
SCAF_POOL=${CKPT}/Llama-3.2-1B-Instruct_experts_scaf_k10
ENT=${CKPT}/Llama-3.2-1B-Instruct_entangled_k10
RL=${SCAF_POOL}/results/router_leak
ENTOUT=${ENT}/results/entangled
CFG=configs/entangled_facts_1b.json
MAN=${ENT}/plant_manifest.json
TAU=0.5473                      # retain-p1 margin (calibrated on RETAIN only)
NEW_FAMS="attn_norm logit_div"
mkdir -p "${RL}" "${ENTOUT}"

# Root the chain on EVERY job we currently have queued (any campaign), array suffixes stripped —
# not just this thread's. The 4-GPU cap is global: a concurrent array (e.g. ctv-w5-eval) can
# already be holding all 4, so anything we add must wait for the whole queue to drain.
CHAIN="${DEP:-$(squeue -u "$USER" -h -o '%i' 2>/dev/null | sed 's/_.*//' | sort -u | paste -sd: -)}"
echo "[wave4] dependency chain root: ${CHAIN:-none}"

submit() {
  local name=$1 time=$2; shift 2
  local DEPFLAG=""; [ -n "${CHAIN}" ] && DEPFLAG="--dependency=afterany:${CHAIN}"
  local body="#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${time}
#SBATCH --output=${RL}/logs_%x_%j.out
set -eo pipefail
export HF_HOME=${HF_HOME}
cd ${SCRIPT_DIR}
$*"
  if [ "${STUB:-0}" = "1" ]; then
    echo "===== STUB ${name} (deps: ${CHAIN:-none}) ====="; echo "${body}"; CHAIN="stub_${name}"; return
  fi
  local out; out=$(echo "${body}" | sbatch ${DEPFLAG})
  echo "${out} (${name}, deps: ${CHAIN:-none})"
  CHAIN=$(echo "${out}" | awk '{print $NF}')
}

ceiling() {  # ONE shared router-independent ceiling for every family
  local C=${ENTOUT}/ceiling_expert_max.json
  [ -f "${C}" ] && { echo "skip ceiling (exists)"; return; }
  submit rl-w4-ceiling 03:00:00 \
    "${PY} eval_entangled_probe.py --config ${CFG} --manifest ${MAN} --experts_dir ${ENT} \
       --channels expert_max --drop_shard none --surface both --out ${C}"
}

families() {  # the two unrun routers: post + floor (rho vs the shared ceiling, computed on CPU)
  local fam
  for fam in ${NEW_FAMS}; do
    local P=${ENTOUT}/postdrop_embedsim_fam_${fam}.json
    local F=${ENTOUT}/floor_embedsim_fam_${fam}.json
    [ -f "${P}" ] && [ -f "${F}" ] && { echo "skip family ${fam} (exists)"; continue; }
    submit rl-w4-${fam} 03:00:00 \
      "${PY} eval_entangled_probe.py --config ${CFG} --manifest ${MAN} --experts_dir ${ENT} \
         --channels served_embedsim --embed_policy sibling --embed_strategy ${fam} \
         --drop_shard 9 --surface both --out ${P} && \
       ${PY} eval_entangled_probe.py --config ${CFG} --manifest ${MAN} --experts_dir ${SCAF_POOL} \
         --channels served_embedsim --embed_policy sibling --embed_strategy ${fam} \
         --drop_shard 9 --surface both --out ${F}"
  done
}

seal() {  # ppl-native abstain seal (same tau on post AND floor so the policy matches)
  local P=${ENTOUT}/postdrop_embedsim_pplseal.json
  local F=${ENTOUT}/floor_embedsim_pplseal.json
  [ -f "${P}" ] && [ -f "${F}" ] && { echo "skip seal (exists)"; return; }
  submit rl-w4-pplseal 03:00:00 \
    "${PY} eval_entangled_probe.py --config ${CFG} --manifest ${MAN} --experts_dir ${ENT} \
       --channels served_embedsim --embed_policy sibling --embed_strategy ppl \
       --embed_abstain_tau ${TAU} --drop_shard 9 --surface both --out ${P} && \
     ${PY} eval_entangled_probe.py --config ${CFG} --manifest ${MAN} --experts_dir ${SCAF_POOL} \
       --channels served_embedsim --embed_policy sibling --embed_strategy ppl \
       --embed_abstain_tau ${TAU} --drop_shard 9 --surface both --out ${F}"
}

SCAF_BASE=${CKPT}/Llama-3.2-1B-Instruct_scaffolded_alpaca2k

disclosure() {  # per-RUNG deletion-disclosure AUC (shard already known 0.839; author + name
                # rungs — the recommended and the privacy-cleanest — were never priced).
                # Non-destructive: NEW out path, the existing rl_centroid_k10.* is untouched
                # (rl_roc_centroid.json derives from it).
  local O=${RL}/rl_centroid_k10_rungs.json
  [ -f "${O}" ] && { echo "skip disclosure (exists)"; return; }
  submit rl-w5-disclosure 01:30:00 \
    "${PY} routing_audit_tofu.py --centroid_mode --centroid_k 10 --drop_shard 9 \
       --probe_manifest ${MAN} --dump_sims --device cuda --out ${O}"
}

mia() {  # composed-MIA rider: is the router leak visible to MIA, or blind like fq?
         # sibling = the leak; tombstone = the seal. Prior reference points: exact module-drop
         # routerkey 0.375 <= oracle floor 0.379; ramole-embed 0.353 (leaked yet MIA-blind).
  local S=${RL}/mia_embedrouted_sibling_del9.json
  local T=${RL}/mia_embedrouted_tombstone_del9.json
  [ -f "${S}" ] && [ -f "${T}" ] && { echo "skip mia (exists)"; return; }
  submit rl-w5-mia 03:00:00 \
    "${PY} attack_mia.py --model_name ${SCAF_BASE} --shards_dir ${SCAF_POOL} \
       --embed_route sibling --delete_shard 9 --k 10 --forget_shard_id 9 \
       --label embedrouted_sibling_del9 --output_dir ${SCAF_POOL} --dump_scores --out ${S} && \
     ${PY} attack_mia.py --model_name ${SCAF_BASE} --shards_dir ${SCAF_POOL} \
       --embed_route tombstone --delete_shard 9 --k 10 --forget_shard_id 9 \
       --label embedrouted_tombstone_del9 --output_dir ${SCAF_POOL} --dump_scores --out ${T}"
}

case "${1:-all}" in
  ceiling)    ceiling ;;
  families)   families ;;
  seal)       seal ;;
  disclosure) disclosure ;;
  mia)        mia ;;
  riders)     disclosure; mia ;;
  all)        ceiling; families; seal ;;
  *) echo "usage: bash submit_router_wave4.sh [ceiling|families|seal|all|disclosure|mia|riders]"; exit 1 ;;
esac
