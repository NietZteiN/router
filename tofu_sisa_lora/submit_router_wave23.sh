#!/bin/bash
# router_leak Waves 2 & 3 driver (thread log/router_leak/, 2026-07-23 master-table follow-up).
#   bash submit_router_wave23.sh [wave2|wave3|all]
# Wave 2 = per-STRATEGY sibling-content ROUGE (dump_generations_routed.py --strategies).
# Wave 3 = per-FAMILY Mode-B ρ (eval_entangled_probe.py --embed_strategy, sibling policy).
#
# GLOBAL 4-GPU CAP: every job is 1 GPU and CHAINED SERIALLY (each --dependency=afterany the
# previous), with the FIRST chained behind the live sepmlp jobs — so at most ONE of ours is ever
# eligible (3 sepmlp + 1 ours = 4). STUB=1 prints every sbatch body without submitting.
# CPU gates first: python test_router_leak.py ; python test_entangled_facts.py
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PY=${TOFU_PYTHON:-python3}
CKPT=${TOFU_CKPT_ROOT}
SCAF_BASE=${CKPT}/Llama-3.2-1B-Instruct_scaffolded_alpaca2k
SCAF_POOL=${CKPT}/Llama-3.2-1B-Instruct_experts_scaf_k10
ENT=${CKPT}/Llama-3.2-1B-Instruct_entangled_k10
RL=${SCAF_POOL}/results/router_leak
ENTOUT=${ENT}/results/entangled
mkdir -p "${RL}" "${ENTOUT}"

# strategies for the two waves (Wave-2 = all gradeable routers; Wave-3 = a representative set:
# two magnets [centroid_lm dense-LM, activation_norm behavioral] + two diffuse [centroid_sbert, ppl])
# NB centroid_sbert_q is a router_family_audit-only score; it is NOT a servable RoutedModel
# strategy (merge_lora._build_routed_model has no branch for it), so it is excluded here.
W2_STRATS="key_exact,key_tfidf,centroid_sbert,centroid_lm,centroid_lm_last,ppl,activation_norm,attn_norm,logit_div"
W3_FAMS="centroid_sbert centroid_lm ppl activation_norm"

# Chain behind sepmlp (or any DEP override); each of our jobs then depends on the previous.
CHAIN="${DEP:-$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null | awk '/sepmlp/{print $1}' | paste -sd: -)}"
echo "[wave23] initial dependency chain root: ${CHAIN:-none}"

submit() {  # submit <name> <time> <cmd...>; chains serially via CHAIN; sets CHAIN=new job id
  local name=$1 time=$2; shift 2
  local DEPFLAG=""
  [ -n "${CHAIN}" ] && DEPFLAG="--dependency=afterany:${CHAIN}"
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

wave2() {  # per-strategy sibling-content: smoke CAP=40 (validate the new path) THEN full 400
  local out=${RL}/sibling_content_by_strategy.json
  [ -f "${out}" ] && { echo "skip wave2 (exists: ${out})"; return; }
  submit rl-w2-content 04:00:00 \
    "${PY} dump_generations_routed.py --model_name ${SCAF_BASE} --shards_dir ${SCAF_POOL} \
       --k 10 --forget_shard_id 9 --strategies ${W2_STRATS} --max_questions 40 \
       --out ${RL}/sibling_content_by_strategy_c40.json && \
     ${PY} dump_generations_routed.py --model_name ${SCAF_BASE} --shards_dir ${SCAF_POOL} \
       --k 10 --forget_shard_id 9 --strategies ${W2_STRATS} --out ${out}"
}

wave3() {  # per-family Mode-B ρ: ceiling(planted,no-drop) + post(planted,drop9) + floor(clean,drop9)
  local fam
  for fam in ${W3_FAMS}; do
    local rho=${RL}/rho_embedsim_fam_${fam}.json
    [ -f "${rho}" ] && { echo "skip wave3 ${fam} (exists: ${rho})"; continue; }
    local C=${ENTOUT}/ceiling_embedsim_fam_${fam}.json
    local P=${ENTOUT}/postdrop_embedsim_fam_${fam}.json
    local F=${ENTOUT}/floor_embedsim_fam_${fam}.json
    submit rl-w3-${fam} 03:00:00 \
      "${PY} eval_entangled_probe.py --config configs/entangled_facts_1b.json \
         --manifest ${ENT}/plant_manifest.json --experts_dir ${ENT} \
         --channels served_embedsim --embed_policy sibling --embed_strategy ${fam} \
         --drop_shard none --surface both --out ${C} && \
       ${PY} eval_entangled_probe.py --config configs/entangled_facts_1b.json \
         --manifest ${ENT}/plant_manifest.json --experts_dir ${ENT} \
         --channels served_embedsim --embed_policy sibling --embed_strategy ${fam} \
         --drop_shard 9 --surface both --out ${P} && \
       ${PY} eval_entangled_probe.py --config configs/entangled_facts_1b.json \
         --manifest ${ENT}/plant_manifest.json --experts_dir ${SCAF_POOL} \
         --channels served_embedsim --embed_policy sibling --embed_strategy ${fam} \
         --drop_shard 9 --surface both --out ${F} && \
       ${PY} aggregate_rho.py --ceiling ${C} --post ${P} --floor ${F} --out ${rho}"
  done
}

case "${1:-all}" in
  wave2) wave2 ;;
  wave3) wave3 ;;
  all)   wave2; wave3 ;;
  *) echo "usage: bash submit_router_wave23.sh [wave2|wave3|all]"; exit 1 ;;
esac
