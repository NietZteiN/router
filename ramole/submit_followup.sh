#!/bin/bash
# Follow-up campaign orchestrator (E1–E6) on SLURM (sprint1-3, arrays capped %4).
# Runs on top of the finished Arm A runs: DBpedia ramole_l32_3b_n32_k3 and the TOFU pool
# Llama-3.2-1B-Instruct_legonet_n32_k3 (+ its forget10 deletion).
#
#   E1  seed variance      — retrain ONLY the router at base_seed 43/44 (retriever/index shared
#                            via retriever_run on DBpedia; --seed/--router_out on TOFU), re-eval.
#   E2  alpha diagnostics  — analyze_router{,_tofu}.py capture_alpha stats (keys/retriever/d0;
#                            TOFU full/unlearn).
#   E3  routing audits     — routing_audit{,_tofu}.py per index policy (writes the rebuilt index
#                            files) + the rebuilt-index evals that depend on them.
#   E4  serve-time k-sweep — k=5/8 configs over the SAME run dir; --label_suffix k5/k8 is
#                            MANDATORY (labels collide with the k=3 files otherwise).
#   E5  serving bench      — benchmark_serving.py throughput table.
#   E6  batch deletion     — 15 seeded records (excl. d0/d1/d2's rec_000000-2) → legonet
#                            unlearn.py --tag d_batch15, then router|mean × before|after evals.
#
#   bash submit_followup.sh [all|wave1|wave2]
#     wave1 = trains + unlearn + audits + alpha + bench ; wave2 = the dependent evals.
#     'all' chains wave2 behind wave1 via afterok; standalone 'wave2' submits dep-free
#     (assumes wave1 artifacts on disk).
#   STUB=1 bash submit_followup.sh all   # print every sbatch script, submit nothing
#
# Conventions inherited from submit_ramole.sh / submit_overnight.sh: heredoc sbatch via
# run_sbatch, each python command on ONE line, --exclude=sprint4, gres=gpu:1.

# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/slurm_nodes.sh"

PHASE="${1:-all}"
STUB="${STUB:-0}"
N_EVAL="${N_EVAL:-200}"
CAP=4                                   # follow-up arrays capped at 4 concurrent GPUs

DBP_RUN=${TOFU_CKPT_STORE}/ramole/runs/ramole_l32_3b_n32_k3
TOFU_POOL=${TOFU_CKPT_ROOT}/Llama-3.2-1B-Instruct_legonet_n32_k3
TOFU_DIR=${REPO_ROOT}/tofu_sisa_lora
LEGO_DIR=${REPO_ROOT}/legonet_lora
BASE_CFG="${HERE}/configs/ramole_l32_3b.json"
TOFU_CFG="${TOFU_DIR}/configs/ramole_tofu_1b.json"
LEGO_CFG="${LEGO_DIR}/configs/legonet_l32_3b.json"
TOFU_MODEL="meta-llama/Llama-3.2-1B-Instruct"
D0_ROUTER=${TOFU_CKPT_STORE}/ramole/runs/ramole_l32_3b_d0/router.safetensors
# source-run corpus (E6 sampling pool) resolved the same way eval_ramole resolves it
SRC_RECORDS=$("$PYTHON" -c "import sys;sys.path.insert(0,'$HERE');import ramole_common as rc;print(rc.source_paths(rc.load_config('$BASE_CFG')).records_path)")

LOG_DIR="$DBP_RUN/logs"
TOFU_LOG_DIR="$TOFU_POOL/logs"
mkdir -p "$LOG_DIR" "$TOFU_LOG_DIR" "$DBP_RUN/results" "$TOFU_POOL/results/extended"

run_sbatch() {  # args -> sbatch flags; script on stdin; echoes job id (or STUBJOB)
  if [ "$STUB" = "1" ]; then
    echo "===== STUB sbatch $* =====" >&2; cat >&2; echo "STUBJOB"
  else
    local out; out=$(sbatch "$@"); echo "$out" >&2; echo "$out" | awk '{print $NF}'
  fi
}
dep() {  # dep ID... -> "--dependency=afterok:ID[:ID...]"; empty for ""/STUBJOB (wave2-alone, STUB)
  local ids="" j
  for j in "$@"; do if [ -n "$j" ] && [ "$j" != "STUBJOB" ]; then ids+=":$j"; fi; done
  if [ -n "$ids" ]; then echo "--dependency=afterok${ids}"; fi
}

# ══ WAVE 1 ══════════════════════════════════════════════════════════════════════

# (a) E1-DBpedia: retrain the router at seeds 43/44 (retriever/index shared, never retrained)
SEED_CFGS=(ramole_l32_3b_s43 ramole_l32_3b_s44)
submit_e1_dbp_trains() {
  local LIT=""; for c in "${SEED_CFGS[@]}"; do LIT+="\"$c\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e1-dbp-router
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-1%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_ROUTER_TIME}
#SBATCH --output=${LOG_DIR}/fu_e1_router_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFGS=(${LIT})
"${PYTHON}" "${HERE}/train_router.py" --config "${HERE}/configs/\${CFGS[\${SLURM_ARRAY_TASK_ID}]}.json" --device cuda
EOF
}

# (b) E1-TOFU: retrain the router at seeds 43/44 into distinct router_s{seed}.safetensors
SEEDS=(43 44)
submit_e1_tofu_trains() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e1-tofu-router
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-1%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=${TOFU_LOG_DIR}/fu_e1_router_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEEDS=(${SEEDS[@]})
S=\${SEEDS[\${SLURM_ARRAY_TASK_ID}]}
"${PYTHON}" "${TOFU_DIR}/train_router_tofu.py" --config "${TOFU_CFG}" --device cuda --seed \${S} --router_out "${TOFU_POOL}/legonet/ramole/router_s\${S}.safetensors"
EOF
}

# (c) E6: seeded 15-record batch deletion on the legonet source pool (tag d_batch15).
#     Excludes rec_000000-2 (already deleted as d0/d1/d2) so the tags stay disjoint.
submit_e6_unlearn() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e6-unlearn
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/fu_e6_unlearn_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FORGET_IDS=\$("${PYTHON}" -c "import json,random; ids=[json.loads(l)['id'] for l in open('${SRC_RECORDS}')]; pool=[i for i in ids if i not in ('rec_000000','rec_000001','rec_000002')]; print(' '.join(random.Random(42).sample(pool,15)))")
echo "[e6] d_batch15 forget ids: \${FORGET_IDS}"
"${PYTHON}" "${LEGO_DIR}/unlearn.py" --config "${LEGO_CFG}" --forget_record_id \${FORGET_IDS} --tag d_batch15
EOF
}

# (d) E3 audits: TOFU (independent) + DBpedia (needs d_batch15's manifest → afterok on (c))
submit_e3_tofu_audit() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e3-tofu-audit
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=${TOFU_LOG_DIR}/fu_e3_audit_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" "${TOFU_DIR}/routing_audit_tofu.py" --config "${TOFU_CFG}" --tag forget10 --policies stale rebuilt key --device cuda --out "${TOFU_POOL}/results/routing_audit_forget10.json"
EOF
}
submit_e3_dbp_audit() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e3-dbp-audit
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/fu_e3_audit_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" "${HERE}/routing_audit.py" --config "${BASE_CFG}" --tags d0 d1 d2 d_batch15 --device cuda --out "${DBP_RUN}/results/routing_audit.json"
EOF
}

# (e) E2 alpha diagnostics: DBpedia (keys / retriever / d0-router contrast) + TOFU (full / unlearn)
submit_e2_dbp_alpha() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e2-dbp-alpha
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${LOG_DIR}/fu_e2_alpha_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" "${HERE}/analyze_router.py" --config "${BASE_CFG}" --route keys --device cuda --out "${DBP_RUN}/results/alpha_diag_keys.json"
"${PYTHON}" "${HERE}/analyze_router.py" --config "${BASE_CFG}" --route retriever --device cuda --out "${DBP_RUN}/results/alpha_diag_retriever.json"
"${PYTHON}" "${HERE}/analyze_router.py" --config "${BASE_CFG}" --route keys --router_ckpt "${D0_ROUTER}" --device cuda --out "${DBP_RUN}/results/alpha_diag_keys_d0.json"
EOF
}
submit_e2_tofu_alpha() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e2-tofu-alpha
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${TOFU_LOG_DIR}/fu_e2_alpha_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" "${TOFU_DIR}/analyze_router_tofu.py" --config "${TOFU_CFG}" --device cuda --out "${TOFU_POOL}/results/alpha_diag_key_full.json"
"${PYTHON}" "${TOFU_DIR}/analyze_router_tofu.py" --config "${TOFU_CFG}" --device cuda --unlearn_tag forget10 --out "${TOFU_POOL}/results/alpha_diag_key_unlearn.json"
EOF
}

# (f) E5: serving throughput bench (router vs mean vs base, batch sweep)
submit_e5_bench() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e5-bench
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${LOG_DIR}/fu_e5_bench_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" "${HERE}/benchmark_serving.py" --config "${BASE_CFG}" --batch_sizes 1 4 8 16 --iters 3 --gen_tokens 64 --device cuda --out "${DBP_RUN}/results/throughput.json"
EOF
}

# ══ WAVE 2 (dependent evals) ════════════════════════════════════════════════════

# (g) E1 evals — DBpedia router_keys_iid per seed run; TOFU routerkey_{full,unlearn} per seed
submit_e1_dbp_evals() {
  local LIT=""; for c in "${SEED_CFGS[@]}"; do LIT+="\"$c\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e1-dbp-eval
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-1%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_EVAL_TIME}
#SBATCH --output=${LOG_DIR}/fu_e1_eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFGS=(${LIT})
"${PYTHON}" "${HERE}/eval_ramole.py" --config "${HERE}/configs/\${CFGS[\${SLURM_ARRAY_TASK_ID}]}.json" --method router --route keys --condition iid --n_eval ${N_EVAL} --device cuda
EOF
}
E1_TOFU_SPECS=("43|full|" "43|unlearn|forget10" "44|full|" "44|unlearn|forget10")
submit_e1_tofu_evals() {
  local LIT=""; for t in "${E1_TOFU_SPECS[@]}"; do LIT+="\"$t\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e1-tofu-eval
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-3%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=${TOFU_LOG_DIR}/fu_e1_eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"; mkdir -p "\${TOFU_METRICS_CACHE}"
SPECS=(${LIT})
IFS='|' read S COND ETAG <<< "\${SPECS[\${SLURM_ARRAY_TASK_ID}]}"
TAGFLAG=""; [ -n "\${ETAG}" ] && TAGFLAG="--legonet_unlearn_tag \${ETAG}"
"${PYTHON}" "${TOFU_DIR}/eval_tofu.py" --model_name "${TOFU_MODEL}" --output_dir "${TOFU_POOL}" --legonet_config "${TOFU_CFG}" --ramole_router "${TOFU_POOL}/legonet/ramole/router_s\${S}.safetensors" --ramole_route key --label "routerkey_\${COND}_s\${S}" \${TAGFLAG} --k 10 --forget_shard_id 9 --out "${TOFU_POOL}/results/extended/routerkey_\${COND}_s\${S}.json" --hf_home "${HF_HOME}" --extended
EOF
}

# (h) E3 evals — rebuilt-index serving (index files were written by the audits)
E3_TOFU_SPECS=("full|" "unlearn|forget10")
submit_e3_tofu_evals() {
  local LIT=""; for t in "${E3_TOFU_SPECS[@]}"; do LIT+="\"$t\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e3-tofu-eval
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-1%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=${TOFU_LOG_DIR}/fu_e3_eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"; mkdir -p "\${TOFU_METRICS_CACHE}"
SPECS=(${LIT})
IFS='|' read COND ETAG <<< "\${SPECS[\${SLURM_ARRAY_TASK_ID}]}"
TAGFLAG=""; [ -n "\${ETAG}" ] && TAGFLAG="--legonet_unlearn_tag \${ETAG}"
"${PYTHON}" "${TOFU_DIR}/eval_tofu.py" --model_name "${TOFU_MODEL}" --output_dir "${TOFU_POOL}" --legonet_config "${TOFU_CFG}" --ramole_router "${TOFU_POOL}/legonet/ramole/router.safetensors" --ramole_route embed --ramole_index rebuilt --label "ramolerb_\${COND}" \${TAGFLAG} --k 10 --forget_shard_id 9 --out "${TOFU_POOL}/results/extended/ramolerb_\${COND}.json" --hf_home "${HF_HOME}" --extended
EOF
}
submit_e3_dbp_evals() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e3-dbp-eval
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-2%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_EVAL_TIME}
#SBATCH --output=${LOG_DIR}/fu_e3_eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAGS=("d0" "d1" "d2")
"${PYTHON}" "${HERE}/eval_ramole.py" --config "${BASE_CFG}" --method router --unlearn_tag "\${TAGS[\${SLURM_ARRAY_TASK_ID}]}" --unlearn_state after --index_policy rebuilt --device cuda
EOF
}

# (i) E4 evals — serve-time k sweep; --label_suffix is mandatory (same run dir as k=3)
E4_SPECS=("ramole_l32_3b_k5|router|k5" "ramole_l32_3b_k5|mean|k5" "ramole_l32_3b_k8|router|k8" "ramole_l32_3b_k8|mean|k8")
submit_e4_evals() {
  local LIT=""; for t in "${E4_SPECS[@]}"; do LIT+="\"$t\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e4-eval
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-3%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_EVAL_TIME}
#SBATCH --output=${LOG_DIR}/fu_e4_eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPECS=(${LIT})
IFS='|' read CFG M SUF <<< "\${SPECS[\${SLURM_ARRAY_TASK_ID}]}"
"${PYTHON}" "${HERE}/eval_ramole.py" --config "${HERE}/configs/\${CFG}.json" --method \${M} --route keys --condition iid --label_suffix \${SUF} --n_eval ${N_EVAL} --device cuda
EOF
}

# (j) E6 evals — d_batch15 forget records, router|mean × before|after
E6_SPECS=("router|before" "router|after" "mean|before" "mean|after")
submit_e6_evals() {
  local LIT=""; for t in "${E6_SPECS[@]}"; do LIT+="\"$t\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=fu-e6-eval
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-3%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_EVAL_TIME}
#SBATCH --output=${LOG_DIR}/fu_e6_eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPECS=(${LIT})
IFS='|' read M S <<< "\${SPECS[\${SLURM_ARRAY_TASK_ID}]}"
"${PYTHON}" "${HERE}/eval_ramole.py" --config "${BASE_CFG}" --method \${M} --unlearn_tag d_batch15 --unlearn_state \${S} --device cuda
EOF
}

# ══ dispatch ════════════════════════════════════════════════════════════════════
J_E1DBP=""; J_E1TOFU=""; J_E6UNL=""; J_TOFU_AUDIT=""; J_DBP_AUDIT=""; J_E2DBP=""; J_E2TOFU=""; J_E5=""
J_G_DBP=""; J_G_TOFU=""; J_H_TOFU=""; J_H_DBP=""; J_E4=""; J_E6E=""

case "$PHASE" in all|wave1|wave2) ;; *) echo "unknown phase: $PHASE (all|wave1|wave2)" >&2; exit 1 ;; esac

if [ "$PHASE" = "all" ] || [ "$PHASE" = "wave1" ]; then
  J_E1DBP=$(submit_e1_dbp_trains)
  J_E1TOFU=$(submit_e1_tofu_trains)
  J_E6UNL=$(submit_e6_unlearn)
  J_TOFU_AUDIT=$(submit_e3_tofu_audit)
  J_DBP_AUDIT=$(submit_e3_dbp_audit $(dep "$J_E6UNL"))   # audit covers d_batch15 → after (c)
  J_E2DBP=$(submit_e2_dbp_alpha)
  J_E2TOFU=$(submit_e2_tofu_alpha)
  J_E5=$(submit_e5_bench)
fi

if [ "$PHASE" = "all" ] || [ "$PHASE" = "wave2" ]; then
  J_G_DBP=$(submit_e1_dbp_evals $(dep "$J_E1DBP"))
  J_G_TOFU=$(submit_e1_tofu_evals $(dep "$J_E1TOFU"))
  J_H_TOFU=$(submit_e3_tofu_evals $(dep "$J_TOFU_AUDIT"))
  J_H_DBP=$(submit_e3_dbp_evals $(dep "$J_DBP_AUDIT"))
  J_E4=$(submit_e4_evals)                                # no deps: artifacts already on disk
  J_E6E=$(submit_e6_evals $(dep "$J_E6UNL"))
fi

echo "submitted (${PHASE}):"
echo "  wave1: e1_dbp_trains=${J_E1DBP:-—} e1_tofu_trains=${J_E1TOFU:-—} e6_unlearn=${J_E6UNL:-—}"
echo "         e3_tofu_audit=${J_TOFU_AUDIT:-—} e3_dbp_audit=${J_DBP_AUDIT:-—} (after e6_unlearn)"
echo "         e2_dbp_alpha=${J_E2DBP:-—} e2_tofu_alpha=${J_E2TOFU:-—} e5_bench=${J_E5:-—}"
echo "  wave2: e1_dbp_evals=${J_G_DBP:-—} e1_tofu_evals=${J_G_TOFU:-—} e3_tofu_evals=${J_H_TOFU:-—}"
echo "         e3_dbp_evals=${J_H_DBP:-—} e4_evals=${J_E4:-—} e6_evals=${J_E6E:-—}"
