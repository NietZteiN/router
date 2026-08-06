#!/bin/bash
# Router-leak campaign driver (thread log/router_leak/, plan can-you-expand-on-moonlit-kahan).
#   bash submit_router_leak.sh [phase1|phase2smoke|phase2|content|collect]
# STUB=1 prints every sbatch body without submitting. Every job self-skips existing outputs.
# ⚠ GLOBAL 4-GPU CAP: check `squeue -u jack` first; DEP=<jobid> chains every submission
#   with --dependency=afterany:<jobid> (use the last pending job of another campaign).
# CPU gate (run first): python test_router_leak.py  (+ test_routing_audit_tofu.py,
#   test_routed_scaffold_merged.py, test_entangled_facts.py)
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PY=${TOFU_PYTHON:-python3}
CKPT=${TOFU_CKPT_ROOT}
SCAF_BASE=${CKPT}/Llama-3.2-1B-Instruct_scaffolded_alpaca2k
SCAF_POOL=${CKPT}/Llama-3.2-1B-Instruct_experts_scaf_k10
ENT=${CKPT}/Llama-3.2-1B-Instruct_entangled_k10
LEGO=${CKPT}/Llama-3.2-1B-Instruct_legonet_n32_k3
RL=${SCAF_POOL}/results/router_leak
mkdir -p "${RL}"

submit() {  # submit <name> <time> <cmd...>; deps = DEP env + per-call EXTRA_DEP; sets LAST_JOB
  local name=$1 time=$2; shift 2
  local deps="${DEP:-}"
  [ -n "${EXTRA_DEP:-}" ] && deps="${deps:+${deps}:}${EXTRA_DEP}"
  local DEPFLAG=""
  [ -n "${deps}" ] && DEPFLAG="--dependency=afterany:${deps}"
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
    echo "===== STUB ${name} (deps: ${deps:-none}) ====="; echo "${body}"; LAST_JOB=stub; return
  fi
  local out
  out=$(echo "${body}" | sbatch ${DEPFLAG})
  echo "${out} (${name}, deps: ${deps:-none})"
  LAST_JOB=$(echo "${out}" | awk '{print $NF}')
}

phase1() {
  # P1.1/P1.2 n=32 audits (base-pinned + FT encoder), tombstone+dump — ~15 min each
  [ -f "${LEGO}/results/rl_audit_basepin.json" ] || submit rl-aud-basepin 02:00:00 \
    ${PY} routing_audit_tofu.py --config configs/ramole_tofu_1b_basepin.json --tag forget10 \
      --policies stale dropped abstain tombstone key --dump_sims --device cuda \
      --out ${LEGO}/results/rl_audit_basepin.json
  [ -f "${LEGO}/results/rl_audit_ft.json" ] || submit rl-aud-ft 02:00:00 \
    ${PY} routing_audit_tofu.py --config configs/ramole_tofu_1b.json --tag forget10 \
      --policies stale dropped tombstone --dump_sims --device cuda \
      --out ${LEGO}/results/rl_audit_ft.json
  # P1.3 k=10 centroid audit (the R2 serving router) + Mode-B probes + holdout disclosure
  [ -f "${RL}/rl_centroid_k10.json" ] || submit rl-centroid 01:30:00 \
    ${PY} routing_audit_tofu.py --centroid_mode --centroid_k 10 --drop_shard 9 \
      --probe_manifest ${ENT}/plant_manifest.json --dump_sims --device cuda \
      --out ${RL}/rl_centroid_k10.json
  # P1.4 R5 author dial: plan manifests (CPU, in-job) then per-tag audits
  [ -f "${LEGO}/results/rl_audit_basepin_f01.json" ] || submit rl-aud-f01 02:00:00 \
    "${PY} unlearn_legonet.py --config configs/ramole_tofu_1b_basepin_f01.json --tag forget01 --forget_authors 198 199 --plan && \
     ${PY} routing_audit_tofu.py --config configs/ramole_tofu_1b_basepin_f01.json --tag forget01 \
      --policies stale dropped tombstone --dump_sims --device cuda \
      --out ${LEGO}/results/rl_audit_basepin_f01.json"
  # 5th Phase-1 job: serialize behind f01 so at most 4 of ours are ever eligible at once
  [ -f "${LEGO}/results/rl_audit_basepin_f05.json" ] || EXTRA_DEP=${LAST_JOB/stub/} submit rl-aud-f05 02:00:00 \
    "${PY} unlearn_legonet.py --config configs/ramole_tofu_1b_basepin_f05.json --tag forget05 --forget_authors 190 191 192 193 194 195 196 197 198 199 --plan && \
     ${PY} routing_audit_tofu.py --config configs/ramole_tofu_1b_basepin_f05.json --tag forget05 \
      --policies stale dropped tombstone --dump_sims --device cuda \
      --out ${LEGO}/results/rl_audit_basepin_f05.json"
}

phase2() {  # LLM serving triple; TIER=smoke|extended (default smoke)
  local tier=${1:-smoke} flag time
  if [ "${tier}" = "extended" ]; then flag=--extended; time=${TOFU_EXTENDED_TIME}; else flag=--smoke; time=${TOFU_SMOKE_TIME}; fi
  local WAVE_IDS=""
  for cell in "full:" "sibling:--delete_shard 9" "tombstone:--delete_shard 9"; do
    local pol=${cell%%:*} extra=${cell#*:} label
    if [ "${pol}" = "full" ]; then label=embedrouted_full; extra=""; pol=sibling; else label=embedrouted_${pol}_del9; fi
    if [ ! -f "${SCAF_POOL}/results/${tier}/${label}.json" ]; then
      submit rl-e2-${label} ${time} \
        ${PY} eval_routed_scaffold.py --model_name ${SCAF_BASE} --shards_dir ${SCAF_POOL} \
          --k 10 --forget_shard_id 9 --embed_route ${pol} ${extra} ${flag} \
          --out ${SCAF_POOL}/results/${tier}/${label}.json
      [ "${LAST_JOB}" != "stub" ] && WAVE_IDS="${WAVE_IDS:+${WAVE_IDS}:}${LAST_JOB}"
    fi
  done
  # Mode-B tombstone worlds (full 200 facts, extended tier only) — chained behind the
  # triple wave so at most 4 of ours are ever eligible at once (the global GPU cap)
  if [ "${tier}" = "extended" ]; then
    local MODEB_DEP="${WAVE_IDS}"
    [ -f "${ENT}/results/entangled/postdrop_embedsim_tomb.json" ] || EXTRA_DEP="${MODEB_DEP}" submit rl-modeb-post 03:00:00 \
      ${PY} eval_entangled_probe.py --config configs/entangled_facts_1b.json \
        --manifest ${ENT}/plant_manifest.json --experts_dir ${ENT} \
        --channels served_embedsim --embed_policy tombstone --drop_shard 9 --surface both \
        --rouge --dump_generations --out ${ENT}/results/entangled/postdrop_embedsim_tomb.json
    [ -f "${ENT}/results/entangled/floor_embedsim_tomb.json" ] || EXTRA_DEP="${MODEB_DEP}" submit rl-modeb-floor 03:00:00 \
      ${PY} eval_entangled_probe.py --config configs/entangled_facts_1b.json \
        --manifest ${ENT}/plant_manifest.json --experts_dir ${SCAF_POOL} \
        --channels served_embedsim --embed_policy tombstone --drop_shard 9 --surface both \
        --out ${ENT}/results/entangled/floor_embedsim_tomb.json
    # the Phase-1 winning rung (author sentinels, c_probe≈0.97 -> Branch-A collapse prediction)
    [ -f "${ENT}/results/entangled/postdrop_embedsim_tomba.json" ] || EXTRA_DEP="${MODEB_DEP}" submit rl-modeb-posta 03:00:00 \
      ${PY} eval_entangled_probe.py --config configs/entangled_facts_1b.json \
        --manifest ${ENT}/plant_manifest.json --experts_dir ${ENT} \
        --channels served_embedsim --embed_policy tombstone_author --drop_shard 9 --surface both \
        --rouge --dump_generations --out ${ENT}/results/entangled/postdrop_embedsim_tomba.json
    [ -f "${ENT}/results/entangled/floor_embedsim_tomba.json" ] || EXTRA_DEP="${MODEB_DEP}" submit rl-modeb-floora 03:00:00 \
      ${PY} eval_entangled_probe.py --config configs/entangled_facts_1b.json \
        --manifest ${ENT}/plant_manifest.json --experts_dir ${SCAF_POOL} \
        --channels served_embedsim --embed_policy tombstone_author --drop_shard 9 --surface both \
        --out ${ENT}/results/entangled/floor_embedsim_tomba.json
  fi
}

content() {  # R3 sibling-content audit (CAP=n for smoke)
  local cap_flag=""
  [ -n "${CAP:-}" ] && cap_flag="--max_questions ${CAP}"
  [ -f "${RL}/sibling_content_audit${CAP:+_c${CAP}}.json" ] || submit rl-content 02:30:00 \
    ${PY} dump_generations_routed.py --model_name ${SCAF_BASE} --shards_dir ${SCAF_POOL} \
      --k 10 --forget_shard_id 9 ${cap_flag} \
      --out ${RL}/sibling_content_audit${CAP:+_c${CAP}}.json
}

collect() {  # CPU post-processing (login-node OK: numpy only)
  ${PY} analyze_router_leak.py roc --npz ${RL}/rl_centroid_k10.sims.npz \
    --out ${RL}/rl_roc_centroid.json || true
  ${PY} analyze_router_leak.py roc --npz ${LEGO}/results/rl_audit_basepin.sims.npz --legonet \
    --out ${RL}/rl_roc_legonet.json || true
  ${PY} analyze_router_leak.py coverage --out ${RL}/rl_coverage.json || true
  ${PY} analyze_router_leak.py table --jsons ${LEGO}/results/rl_audit_basepin_f01.json \
    ${LEGO}/results/rl_audit_basepin_f05.json ${LEGO}/results/rl_audit_basepin.json \
    ${RL}/rl_centroid_k10.json --out ${RL}/rl_deletion_dial.md || true
}

case "${1:-phase1}" in
  phase1) phase1 ;;
  phase2smoke) phase2 smoke ;;
  phase2) phase2 extended ;;
  content) content ;;
  collect) collect ;;
  *) echo "usage: bash submit_router_leak.sh [phase1|phase2smoke|phase2|content|collect]"; exit 1 ;;
esac
