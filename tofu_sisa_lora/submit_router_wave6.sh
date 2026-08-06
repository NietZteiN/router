#!/bin/bash
# router_leak Wave 6 (2026-07-24): the two follow-ups the campaign left open.
#   bash submit_router_wave6.sh [tombstone|logitutil|all]
#
# A (tombstone) — the H3 CLOSER: serve the AUTHOR-rung THRESHOLDED tombstone end-to-end and read
#   mu/fq/retain vs the already-served embed-full (0.6872) / sibling (0.6922) / shard-tombstone
#   (0.6861) arms. Predicted: retain cost ~0 (τ=0.1944 = 90% catch / 0.11% retain-FPR), i.e. the
#   seal is finally DEMONSTRATED, not just audited. Compares to the hard-router ceiling 0.7509.
#
# B1 (logitutil) — is logit_div leak-free (ρ 0.016) because it ROUTES WELL or because it routes
#   BADLY? Serve the SAME k=10 scaffold pool routed by logit_div vs key_exact vs centroid_sbert
#   (eval_tofu routed_*; all share the OOD-through-router confound, so read retain_prob — the clean
#   in-domain routing-quality signal). logit_div retain_prob ≈ key ⇒ leak-free AND competitive ⇒
#   "route by atypicality" is a real design lever; ≪ key ⇒ leak-free only because it misroutes.
#   NB k=200 logit_div is blocked (Mode-B needs a planted arm; behavioral routers hit the k>50
#   memory law) — that extension needs new infra, not run here.
#
# GLOBAL 4-GPU CAP: 1 GPU/job, chained serially behind everything already queued. STUB=1 previews.
# CPU gates first: python test_router_leak.py ; python test_routed_scaffold_merged.py
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PY=${TOFU_PYTHON:-python3}
CKPT=${TOFU_CKPT_ROOT}
SCAF_BASE=${CKPT}/Llama-3.2-1B-Instruct_scaffolded_alpaca2k
SCAF_POOL=${CKPT}/Llama-3.2-1B-Instruct_experts_scaf_k10
RL=${SCAF_POOL}/results/router_leak
SMOKE=${SCAF_POOL}/results/smoke
TAU=0.1944
mkdir -p "${RL}"

CHAIN="${DEP:-$(squeue -u "$USER" -h -o '%i' 2>/dev/null | sed 's/_.*//' | sort -u | paste -sd: -)}"
echo "[wave6] dependency chain root: ${CHAIN:-none}"

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
  echo "${out} (${name}, deps: ${CHAIN:-none})"; CHAIN=$(echo "${out}" | awk '{print $NF}')
}

tombstone() {  # A — author-rung thresholded tombstone, served end-to-end (smoke tier)
  local O=${SMOKE}/embedrouted_tombstone_author_del9.json
  [ -f "${O}" ] && { echo "skip tombstone (exists)"; return; }
  submit rl-w6-tombauth 01:30:00 \
    "${PY} eval_routed_scaffold.py --model_name ${SCAF_BASE} --shards_dir ${SCAF_POOL} \
       --k 10 --forget_shard_id 9 --embed_route tombstone_author --delete_shard 9 \
       --tombstone_tau ${TAU} --smoke --out ${O}"
}

logitutil() {  # B1 — logit_div vs key_exact vs centroid_sbert routing utility (smoke), one job
  local need=0
  for lbl in routed_logit_div routed_key_exact routed_centroid_sbert; do
    [ -f "${SMOKE}/${lbl}.json" ] || need=1
  done
  [ "${need}" = "0" ] && { echo "skip logitutil (all exist)"; return; }
  submit rl-w6-logitutil 02:30:00 \
    "for L in routed_logit_div routed_key_exact routed_centroid_sbert; do \
       [ -f ${SMOKE}/\$L.json ] || ${PY} eval_tofu.py --model_name ${SCAF_BASE} \
         --output_dir ${SCAF_POOL} --label \$L --k 10 --forget_shard_id 9 --smoke \
         --out ${SMOKE}/\$L.json; done"
}

case "${1:-all}" in
  tombstone) tombstone ;;
  logitutil) logitutil ;;
  all)       tombstone; logitutil ;;
  *) echo "usage: bash submit_router_wave6.sh [tombstone|logitutil|all]"; exit 1 ;;
esac
