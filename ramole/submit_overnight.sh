#!/bin/bash
# Overnight RAMoLE campaign on SLURM (sprint1-3, ≤4 GPUs). Arm A (default config) is assumed
# already running/submitted separately; this adds the ablations + the unlearning demo and chains
# everything so peak concurrency stays ~4:
#
#   Stage R (routers array, %4) : train the 3 ablation routers (d0 / corpus-split / rank-6),
#                                 starting after AFTER_JOB frees GPUs (default: arm A eval).
#   Stage E (eval array, %4)    : depends on Stage R. Dispatches 18 tasks —
#                                 ablation comparisons (router vs default, key/retriever, iid/ood)
#                                 + the 12 unlearning-demo evals (router|mean × d0,d1,d2 × before,after).
#   A separate background monitor (launched by the operator) runs collect_overnight.py when the
#   queue clears.
#
#   AFTER_JOB=436664 bash submit_overnight.sh        # chain behind arm A's eval array
#   STUB=1 bash submit_overnight.sh                  # print sbatch scripts, submit nothing
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/slurm_nodes.sh"
STUB="${STUB:-0}"
N_EVAL="${N_EVAL:-200}"
BASE_CFG="configs/ramole_l32_3b.json"
AFTER_JOB="${AFTER_JOB:-}"        # optional: a job id to wait for before starting (arm A eval)
RUN_DIR=$("$PYTHON" -c "import sys;sys.path.insert(0,'$HERE');import ramole_common as rc;print(rc.Paths(rc.load_config('$HERE/$BASE_CFG')).run_dir)")
LOG_DIR="$RUN_DIR/logs"; mkdir -p "$LOG_DIR"

run_sbatch() { if [ "$STUB" = "1" ]; then echo "===== STUB sbatch $* =====" >&2; cat >&2; echo "STUBJOB"; else local out; out=$(sbatch "$@"); echo "$out" >&2; echo "$out" | awk '{print $NF}'; fi; }

# ── Stage R: ablation routers (one array task per config) ──────────────────────
ROUTER_CFGS=(ramole_l32_3b_d0 ramole_l32_3b_corpus ramole_l32_3b_r6)
submit_routers() {
  local LIT=""; for c in "${ROUTER_CFGS[@]}"; do LIT+="\"$c\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-abl-router
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-$(( ${#ROUTER_CFGS[@]} - 1 ))%${RAMOLE_CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_ROUTER_TIME}
#SBATCH --output=${LOG_DIR}/abl_router_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFGS=(${LIT})
"${PYTHON}" "${HERE}/train_router.py" --config "${HERE}/configs/\${CFGS[\${SLURM_ARRAY_TASK_ID}]}.json" --device cuda
EOF
}

# ── Stage E: dispatch eval array (ablation comparisons + unlearning demo) ───────
# each task: "kind|config_basename|a|b|c"  (cmp: a=method b=route c=condition ; unl: a=method b=tag c=state)
EVAL_TASKS=(
  "cmp|ramole_l32_3b_d0|router|keys|iid"      "cmp|ramole_l32_3b_d0|router|keys|ood"
  "cmp|ramole_l32_3b_d0|router|retriever|iid" "cmp|ramole_l32_3b_d0|router|retriever|ood"
  "cmp|ramole_l32_3b_corpus|router|keys|iid"  "cmp|ramole_l32_3b_r6|router|keys|iid"
  "unl|ramole_l32_3b|router|d0|before" "unl|ramole_l32_3b|router|d0|after"
  "unl|ramole_l32_3b|router|d1|before" "unl|ramole_l32_3b|router|d1|after"
  "unl|ramole_l32_3b|router|d2|before" "unl|ramole_l32_3b|router|d2|after"
  "unl|ramole_l32_3b|mean|d0|before"   "unl|ramole_l32_3b|mean|d0|after"
  "unl|ramole_l32_3b|mean|d1|before"   "unl|ramole_l32_3b|mean|d1|after"
  "unl|ramole_l32_3b|mean|d2|before"   "unl|ramole_l32_3b|mean|d2|after"
)
submit_evals() {
  local LIT=""; for t in "${EVAL_TASKS[@]}"; do LIT+="\"$t\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-abl-eval
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-$(( ${#EVAL_TASKS[@]} - 1 ))%${RAMOLE_CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_EVAL_TIME}
#SBATCH --output=${LOG_DIR}/abl_eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASKS=(${LIT})
IFS='|' read KIND CFG A B C <<< "\${TASKS[\${SLURM_ARRAY_TASK_ID}]}"
if [ "\$KIND" = "cmp" ]; then
"${PYTHON}" "${HERE}/eval_ramole.py" --config "${HERE}/configs/\${CFG}.json" --method \$A --route \$B --condition \$C --n_eval ${N_EVAL} --device cuda
else
"${PYTHON}" "${HERE}/eval_ramole.py" --config "${HERE}/configs/\${CFG}.json" --method \$A --unlearn_tag \$B --unlearn_state \$C --device cuda
fi
EOF
}

RDEP=""; [ -n "$AFTER_JOB" ] && RDEP="--dependency=afterany:${AFTER_JOB}"
RJ=$(submit_routers $RDEP)
if [ "$STUB" = "1" ]; then EDEP=""; else EDEP="--dependency=afterok:${RJ}"; fi
EJ=$(submit_evals $EDEP)
echo "submitted: routers=$RJ (after ${AFTER_JOB:-none})  evals=$EJ (after routers)"
