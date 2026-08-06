#!/bin/bash
# All-router leakage sweep driver (thread log/router_leak/, pre-registration
# log/router_leak/2026-07-20_all-router-sweep-preregistration.md).
#   bash submit_router_family.sh [j1|j2|j3|j4|j5|j6|collect|all]
# STUB=1 prints every sbatch body (and the collect-stage commands) without
# submitting/running. Every job self-skips existing outputs. DEP=<jobid> chains every
# submission --dependency=afterany:<jobid>.
# ⚠ GLOBAL 4-GPU CAP: check `squeue -u jack -o "%.10i %.20j %.10T %.10b %F"` first.
#   'all' submits ONE SERIAL lane j1->j2->j3->j4->j5->j6 (each job afterany its
#   predecessor; skipped stages keep the chain intact). Approved plan: this wave is the
#   4th GPU lane next to the 3 ctv lanes, so at most ONE of its jobs may ever be
#   eligible at a time — never widen this back out.
# CPU gate (run first): python test_router_family.py  (+ test_routing_audit_tofu.py,
#   test_router_leak.py, ramole/tests/test_routing_audit.py for the J4-J6 codepaths)

# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PY=${TOFU_PYTHON:-python3}
CKPT=${TOFU_CKPT_ROOT}
SCAF_BASE=${CKPT}/Llama-3.2-1B-Instruct_scaffolded_alpaca2k
SCAF_POOL=${CKPT}/Llama-3.2-1B-Instruct_experts_scaf_k10
# The e25 per-author k=200 pool on disk is the 7B one (verified 2026-07-20: ls shows
# 200 shard dirs; no 1B k200 pool exists). J3 is feature-space only (memory law), so
# the base model is used purely as the centroid_lm embedding encoder — it must be the
# pool's own base, Llama-2-7B-chat-hf.
K200_POOL=${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4
K200_BASE=meta-llama/Llama-2-7B-chat-hf
LEGO=${CKPT}/Llama-3.2-1B-Instruct_legonet_n32_k3
RAMOLE_DIR=${REPO_ROOT}/ramole
DBP_RUN=${TOFU_CKPT_STORE}/ramole/runs/ramole_l32_3b_n32_k3
RL=${SCAF_POOL}/results/router_leak
K200_RL=${K200_POOL}/results/router_leak

# --- 7B orphan-coverage stages j7/j8/j9 (added 2026-07-26) -------------------------
# Table H' showed 7B orphan behavior existed for ONE pool (k=200, via j3): every other
# orphan number in the repo is 1B, so no like-for-like 1B-vs-7B comparison was possible.
# j7+j8 rebuild the FULL k=10 battery (all 9 router.py strategies, same drop sets as
# j1/j2) on the 7B k=10 pool; j9 adds k=50 as a granularity dial at fixed model.
K10_7B_POOL=${CKPT}/Llama-2-7B-chat-hf_k10_r32_e5_lr1e4
K50_7B_POOL=${CKPT}/Llama-2-7B-chat-hf_k50_r32_e5_lr1e4
SEVENB_BASE=meta-llama/Llama-2-7B-chat-hf
K10_7B_RL=${K10_7B_POOL}/results/router_leak
K50_7B_RL=${K50_7B_POOL}/results/router_leak

# --- DE-CONFOUND arm j10/j11 (added 2026-07-26, after j7/j8 landed) ----------------------
# j7/j8 vs j1/j2 is NOT a clean 1B-vs-7B contrast: the 1B arm is the SCAFFOLDED pool
# (experts_scaf_k10, base = ..._scaffolded_alpaca2k) while the 7B arm is the PLAIN pool.
# For the routers that never touch the LLM (key_*, centroid_sbert*) this does not matter —
# they came out bit-identical, as they must — but for centroid_lm*/behavioral, model scale
# and pool provenance are entangled. The plain 1B k=10 pool exists and was never audited, so
# auditing it gives BOTH clean contrasts: scale (1B-plain vs 7B-plain) and scaffold
# (1B-plain vs 1B-scaffolded), against the same 10-shard author assignment.
K10_1B_POOL=${CKPT}/Llama-3.2-1B-Instruct
K10_1B_BASE=meta-llama/Llama-3.2-1B-Instruct
K10_1B_RL=${K10_1B_POOL}/results/router_leak
# k=50: shard 49 is the forget shard (authors 196-199), mirroring k=10's shard 9.
K50_7B_DROPS="49;49,48"

mkdir -p "${RL}" "${K200_RL}" "${LEGO}/results" "${DBP_RUN}/results" \
         "${K10_7B_RL}" "${K50_7B_RL}" "${K10_1B_RL}"

# k=200 feature-space drop sets: single-author expert 199, then the 180-199 mass cell
K200_DROPS="199;180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199"

submit() {  # submit <name> <time> <cmd...>; deps = DEP env + per-call EXTRA_DEP;
            # log dir = OUT_LOG (default ${RL}); work dir = CD_DIR (default SCRIPT_DIR);
            # sets LAST_JOB (jobid | 'stub')
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
#SBATCH --output=${OUT_LOG:-${RL}}/logs_%x_%j.out
set -eo pipefail
export HF_HOME=${HF_HOME}
cd ${CD_DIR:-${SCRIPT_DIR}}
$*"
  if [ "${STUB:-0}" = "1" ]; then
    echo "===== STUB ${name} (deps: ${deps:-none}) ====="; echo "${body}"; LAST_JOB=stub; return
  fi
  local out
  out=$(echo "${body}" | sbatch ${DEPFLAG})
  echo "${out} (${name}, deps: ${deps:-none})"
  LAST_JOB=$(echo "${out}" | awk '{print $NF}')
}

j1() {  # k=10 feature-space family, ALL 4000 questions
  LAST_JOB=""
  [ -f "${RL}/rl_family_k10_feature.json" ] && { echo "skip j1 (exists)"; return; }
  submit rf-j1-k10feat 02:00:00 \
    ${PY} router_family_audit.py --pool_dir ${SCAF_POOL} --base_model ${SCAF_BASE} \
      --k 10 --strategies key_exact key_tfidf centroid_sbert centroid_sbert_q \
      centroid_lm centroid_lm_last --drop_sets '"9;9,8;9,8,7,6"' --queries all \
      --device cuda --dump_sims --out ${RL}/rl_family_k10_feature.json
}

j2() {  # k=10 behavioral family, 400 forget + RandomState(42) 400-retain sample
  LAST_JOB=""
  [ -f "${RL}/rl_family_k10_behavioral.json" ] && { echo "skip j2 (exists)"; return; }
  submit rf-j2-k10behav 03:00:00 \
    ${PY} router_family_audit.py --pool_dir ${SCAF_POOL} --base_model ${SCAF_BASE} \
      --k 10 --strategies ppl activation_norm attn_norm logit_div \
      --drop_sets '"9;9,8;9,8,7,6"' --queries sample \
      --device cuda --dump_sims --out ${RL}/rl_family_k10_behavioral.json
}

j3() {  # k=200 per-author pool, feature-space only (H-POOL strategies; the k200 x r32
        # eval memory law forbids loading 200 adapters — feature-space routing needs none)
  LAST_JOB=""
  [ -f "${K200_RL}/rl_family_k200.json" ] && { echo "skip j3 (exists)"; return; }
  OUT_LOG=${K200_RL} submit rf-j3-k200feat 02:30:00 \
    ${PY} router_family_audit.py --pool_dir ${K200_POOL} --base_model ${K200_BASE} \
      --k 200 --strategies key_exact key_tfidf centroid_sbert centroid_lm \
      --drop_sets \"${K200_DROPS}\" --queries all \
      --device cuda --dump_sims --out ${K200_RL}/rl_family_k200.json
}

j4() {  # H-TRAINED: RouterLoRA drop-audit x3 seeds (builder C's --dropped flag),
        # sequential in ONE job — 1 GPU total
  LAST_JOB=""
  if [ -f "${LEGO}/results/rl_routerlora_drop_s42.json" ] && \
     [ -f "${LEGO}/results/rl_routerlora_drop_s43.json" ] && \
     [ -f "${LEGO}/results/rl_routerlora_drop_s44.json" ]; then
    echo "skip j4 (all 3 seeds exist)"; return
  fi
  # if-blocks (not `[ -f ] || cmd && ...` chains): under set -e a failed seed aborts the
  # job instead of being masked by a later leg's exit status
  OUT_LOG=${LEGO}/results submit rf-j4-routerlora 03:30:00 \
    "if [ ! -f ${LEGO}/results/rl_routerlora_drop_s42.json ]; then ${PY} analyze_router_tofu.py \
       --config configs/ramole_tofu_1b.json --dropped --device cuda \
       --out ${LEGO}/results/rl_routerlora_drop_s42.json; fi
     if [ ! -f ${LEGO}/results/rl_routerlora_drop_s43.json ]; then ${PY} analyze_router_tofu.py \
       --config configs/ramole_tofu_1b.json --dropped \
       --router_ckpt ${LEGO}/legonet/ramole/router_s43.safetensors --device cuda \
       --out ${LEGO}/results/rl_routerlora_drop_s43.json; fi
     if [ ! -f ${LEGO}/results/rl_routerlora_drop_s44.json ]; then ${PY} analyze_router_tofu.py \
       --config configs/ramole_tofu_1b.json --dropped \
       --router_ckpt ${LEGO}/legonet/ramole/router_s44.safetensors --device cuda \
       --out ${LEGO}/results/rl_routerlora_drop_s44.json; fi"
}

j5() {  # H-DATASET: DBpedia retriever drop-audit (builder C's dropped/abstain/--dump_sims
        # additions to ramole/routing_audit.py); runs from the ramole repo.
        # Output stays inside the campaign's rl_family_* namespace (naming law).
  LAST_JOB=""
  [ -f "${DBP_RUN}/results/rl_family_dbpedia.json" ] && { echo "skip j5 (exists)"; return; }
  OUT_LOG=${DBP_RUN}/results CD_DIR=${RAMOLE_DIR} submit rf-j5-dbpedia 01:00:00 \
    ${PY} routing_audit.py --config configs/ramole_l32_3b.json \
      --tags d0 d1 d2 d_batch15 --policies stale rebuilt dropped abstain --dump_sims \
      --device cuda --out ${DBP_RUN}/results/rl_family_dbpedia.json
}

j6() {  # H-ENC: k=10 centroid audit under two new encoders (one job, sequential)
  LAST_JOB=""
  if [ -f "${RL}/rl_enc_mpnet.json" ] && [ -f "${RL}/rl_enc_bge.json" ]; then
    echo "skip j6 (both encoders exist)"; return
  fi
  submit rf-j6-encoders 02:00:00 \
    "if [ ! -f ${RL}/rl_enc_mpnet.json ]; then ${PY} routing_audit_tofu.py --centroid_mode \
       --centroid_k 10 --drop_shard 9 --no_holdout --dump_sims \
       --router_encoder sentence-transformers/all-mpnet-base-v2 --device cuda \
       --out ${RL}/rl_enc_mpnet.json; fi
     if [ ! -f ${RL}/rl_enc_bge.json ]; then ${PY} routing_audit_tofu.py --centroid_mode \
       --centroid_k 10 --drop_shard 9 --no_holdout --dump_sims \
       --router_encoder BAAI/bge-small-en-v1.5 --device cuda \
       --out ${RL}/rl_enc_bge.json; fi"
}

collect() {  # CPU post-processing (login-node OK: numpy only) — builder B's analyzer
             # over the family JSONs + npz sidecars; `|| true` mirrors
             # submit_router_leak.sh (partial availability must not kill the stage).
             # The H-ENC confidence-half roc JSONs are INPUTS to the analyzer
             # (--enc_roc_json), so they run first. STUB=1 previews without running.
  local RUN="" enc
  [ "${STUB:-0}" = "1" ] && RUN="echo [stub-collect]"
  # analyze_router_leak roc over each encoder sims sidecar (centroid layout, drop
  # shard 9); self-skips existing outputs, skips sidecars j6 has not produced yet
  for enc in mpnet bge; do
    if [ -f "${RL}/rl_enc_roc_${enc}.json" ]; then
      echo "skip enc_roc ${enc} (exists)"
    elif [ ! -f "${RL}/rl_enc_${enc}.sims.npz" ]; then
      echo "skip enc_roc ${enc} (no ${RL}/rl_enc_${enc}.sims.npz yet — run j6)"
    else
      ${RUN} ${PY} analyze_router_leak.py roc --npz ${RL}/rl_enc_${enc}.sims.npz \
        --drop_shard 9 --out ${RL}/rl_enc_roc_${enc}.json || true
    fi
  done
  # --force: the analysis JSON/MD are DERIVED summaries — re-collect refreshes them
  # as lanes land (raw result files are still never modified in place)
  ${RUN} ${PY} analyze_router_family.py \
    --family_json ${RL}/rl_family_k10_feature.json ${RL}/rl_family_k10_behavioral.json \
                  ${K200_RL}/rl_family_k200.json \
    --routerlora_json "${LEGO}/results/rl_routerlora_drop_s*.json" \
    --dbpedia_json ${DBP_RUN}/results/rl_family_dbpedia.json \
    --enc_json ${RL}/rl_enc_mpnet.json ${RL}/rl_enc_bge.json \
    --enc_roc_json "${RL}/rl_enc_roc_*.json" \
    --out_json ${RL}/rl_family_leak_analysis.json \
    --out_md ${RL}/rl_family_leak_table.md --force || true
}

j7() {  # 7B k=10, FEATURE/lexical family — the exact 1B j1 cell at 7B (same strategies,
        # same 3 drop sets, --queries all). Feature-space routing scores from the base
        # LLM / sentence encoders and loads NO adapters, so the 7B eval memory law
        # (13.5 GiB + k*228 MB at r32) does not apply here.
  LAST_JOB=""
  [ -f "${K10_7B_RL}/rl_family_k10_7b_feature.json" ] && { echo "skip j7 (exists)"; return; }
  OUT_LOG=${K10_7B_RL} submit rf-j7-k10-7b-feat 02:30:00 \
    ${PY} router_family_audit.py --pool_dir ${K10_7B_POOL} --base_model ${SEVENB_BASE} \
      --k 10 --strategies key_exact key_tfidf centroid_sbert centroid_sbert_q \
      centroid_lm centroid_lm_last --drop_sets '"9;9,8;9,8,7,6"' --queries all \
      --device cuda --dump_sims --out ${K10_7B_RL}/rl_family_k10_7b_feature.json
}

j8() {  # 7B k=10, BEHAVIORAL family — the exact 1B j2 cell at 7B. Unlike j3/j7 these
        # routers score THROUGH the experts, so the pool IS loaded: k=10 x r32 at 7B is
        # ~13.5 + 10*0.228 = ~16 GiB, comfortable on a 44.5 GiB A40. --queries sample
        # (400 forget + the RandomState(42) 400-retain draw) matches j2 exactly so the
        # rows are cell-for-cell comparable; 3 drop sets, as j2.
  LAST_JOB=""
  [ -f "${K10_7B_RL}/rl_family_k10_7b_behavioral.json" ] && { echo "skip j8 (exists)"; return; }
  OUT_LOG=${K10_7B_RL} submit rf-j8-k10-7b-behav 06:00:00 \
    ${PY} router_family_audit.py --pool_dir ${K10_7B_POOL} --base_model ${SEVENB_BASE} \
      --k 10 --strategies ppl activation_norm attn_norm logit_div \
      --drop_sets '"9;9,8;9,8,7,6"' --queries sample \
      --device cuda --dump_sims --out ${K10_7B_RL}/rl_family_k10_7b_behavioral.json
}

j9() {  # 7B k=50, feature/lexical — the granularity dial at FIXED model (k=10 j7 ->
        # k=50 j9 -> k=200 j3). Lands on the same pool as the existing 7B routed mu
        # 0.7147, so utility and orphan behavior are read off one artifact.
  LAST_JOB=""
  [ -f "${K50_7B_RL}/rl_family_k50_7b.json" ] && { echo "skip j9 (exists)"; return; }
  OUT_LOG=${K50_7B_RL} submit rf-j9-k50-7b-feat 02:30:00 \
    ${PY} router_family_audit.py --pool_dir ${K50_7B_POOL} --base_model ${SEVENB_BASE} \
      --k 50 --strategies key_exact key_tfidf centroid_sbert centroid_lm \
      --drop_sets \"${K50_7B_DROPS}\" --queries all \
      --device cuda --dump_sims --out ${K50_7B_RL}/rl_family_k50_7b.json
}

j10() { # 1B PLAIN k=10, feature/lexical — the de-confound twin of j7 (and of j1, which is
        # the SCAFFOLDED 1B pool). Same strategies/drop sets/queries as j1 and j7.
  LAST_JOB=""
  [ -f "${K10_1B_RL}/rl_family_k10_1b_plain_feature.json" ] && { echo "skip j10 (exists)"; return; }
  OUT_LOG=${K10_1B_RL} submit rf-j10-k10-1bplain-feat 02:00:00 \
    ${PY} router_family_audit.py --pool_dir ${K10_1B_POOL} --base_model ${K10_1B_BASE} \
      --k 10 --strategies key_exact key_tfidf centroid_sbert centroid_sbert_q \
      centroid_lm centroid_lm_last --drop_sets '"9;9,8;9,8,7,6"' --queries all \
      --device cuda --dump_sims --out ${K10_1B_RL}/rl_family_k10_1b_plain_feature.json
}

j11() { # 1B PLAIN k=10, behavioral — the de-confound twin of j8 (and of j2). This is the
        # arm that actually isolates model scale for the expert-reading routers.
  LAST_JOB=""
  [ -f "${K10_1B_RL}/rl_family_k10_1b_plain_behavioral.json" ] && { echo "skip j11 (exists)"; return; }
  OUT_LOG=${K10_1B_RL} submit rf-j11-k10-1bplain-behav 03:00:00 \
    ${PY} router_family_audit.py --pool_dir ${K10_1B_POOL} --base_model ${K10_1B_BASE} \
      --k 10 --strategies ppl activation_norm attn_norm logit_div \
      --drop_sets '"9;9,8;9,8,7,6"' --queries sample \
      --device cuda --dump_sims --out ${K10_1B_RL}/rl_family_k10_1b_plain_behavioral.json
}

deconfound() {  # j10 + j11 in parallel (2 GPUs). See the j10/j11 comments for why this arm
                # exists; check the budget before calling — it is designed to fit alongside
                # a running submit_7b_routed_fill.sh (%3) for 5 of the authorized 6.
  local summary="" stage
  for stage in j10 j11; do
    ${stage}
    summary="${summary} ${stage}=${LAST_JOB:-skip}"
  done
  echo "de-confound wave (parallel, 2 GPU):${summary}"
}

sevenb() {  # the 2026-07-26 7B orphan-coverage wave: j7 + j8 + j9 submitted in PARALLEL
            # (3 GPUs). Unlike `all` these are independent pools/outputs with no shared
            # state, so there is no reason to serialize them.
            # !! Budget check is the CALLER's job: this stage alone is 3 concurrent GPUs,
            # and it is designed to run alongside submit_7b_routed_fill.sh (%3) for a
            # total of 6 -- the ceiling the user authorized on 2026-07-26, ABOVE the
            # CLAUDE.md default of 4. Do not add a 4th job here.
  local summary="" stage
  for stage in j7 j8 j9; do
    ${stage}
    summary="${summary} ${stage}=${LAST_JOB:-skip}"
  done
  echo "7B orphan wave (parallel, 3 GPU):${summary}"
}

all() {  # ONE SERIAL GPU lane (approved plan): j1 -> j2 -> j3 -> j4 -> j5 -> j6, each
         # afterany its predecessor. With the 3 ctv lanes already queued this wave may
         # only ever have ONE eligible job — do not parallelize any pair of stages.
  local prev="${DEP:-}" summary="" stage
  if [ -z "${prev}" ]; then
    echo "WARN: DEP empty — lane head j1 submits dependency-free. Confirm <= 3 other GPU" \
         "lanes in \`squeue -u jack\` before proceeding (global 4-GPU cap)."
  fi
  for stage in j1 j2 j3 j4 j5 j6; do
    # DEP is scoped per stage call: each job depends ONLY on the current chain tail
    # (skipped stages leave prev unchanged, so the chain never breaks).
    DEP="${prev}" ${stage}
    if [ -n "${LAST_JOB}" ] && [ "${LAST_JOB}" != "stub" ]; then
      prev="${LAST_JOB}"
    fi
    summary="${summary} ${stage}=${LAST_JOB:-skip}"
  done
  echo "serial lane:${summary} (tail=${prev:-none})"
}

case "${1:-usage}" in
  j1) j1 ;;
  j2) j2 ;;
  j3) j3 ;;
  j4) j4 ;;
  j5) j5 ;;
  j6) j6 ;;
  j7) j7 ;;
  j8) j8 ;;
  j9) j9 ;;
  j10) j10 ;;
  j11) j11 ;;
  sevenb) sevenb ;;
  deconfound) deconfound ;;
  collect) collect ;;
  all) all ;;
  *) echo "usage: bash submit_router_family.sh [j1..j11|sevenb|deconfound|collect|all]"; exit 1 ;;
esac
