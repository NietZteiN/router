#!/bin/bash
# Orchestrate the RAMoLE pipeline on SLURM (sprint1-3, ≤8 GPUs). RAMoLE borrows a legonet run's
# experts/corpus/keys (config 'source_run'), so there is NO expert-training stage here — only the
# retriever fine-tune, the RouterLoRA training, and the comparison eval.
#
#   bash submit_ramole.sh CONFIG [PHASE]     PHASE ∈ {setup,retriever,router,eval,all}
#   STUB=1 bash submit_ramole.sh CONFIG all  # print every sbatch script, submit nothing
#
# 'setup' runs on the LOGIN node (download only): pre-caches the Stage-1 encoder so the GPU jobs
# don't hit the network and so an instructor-xl load failure surfaces early (fall back by editing
# 'encoder_model' in the config — see ramole/CLAUDE.md). retriever ∥ router run in parallel; eval
# depends on BOTH. Each python command stays on ONE line (heredoc backslash footgun).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/slurm_nodes.sh"

CONFIG="${1:?usage: submit_ramole.sh CONFIG [PHASE]}"
PHASE="${2:-all}"
STUB="${STUB:-0}"
N_EVAL="${N_EVAL:-200}"

RUN_DIR=$("$PYTHON" -c "import sys;sys.path.insert(0,'$HERE');import ramole_common as rc;print(rc.Paths(rc.load_config('$CONFIG')).run_dir)")
ENCODER=$("$PYTHON" -c "import sys;sys.path.insert(0,'$HERE');import ramole_common as rc;print(rc.load_config('$CONFIG')['encoder_model'])")
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

run_sbatch() {  # args -> sbatch flags; script on stdin; echoes job id (or STUBJOB)
  if [ "$STUB" = "1" ]; then
    echo "===== STUB sbatch $* =====" >&2; cat >&2; echo "STUBJOB"
  else
    local out; out=$(sbatch "$@"); echo "$out" >&2; echo "$out" | awk '{print $NF}'
  fi
}

do_setup() {  # login node: download/verify the Stage-1 encoder into HF_HOME
  echo "[setup] pre-caching encoder '${ENCODER}' into ${HF_HOME} (login node, download only)" >&2
  HF_HOME="${HF_HOME}" "${PYTHON}" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${ENCODER}'); print('[setup] encoder OK: ${ENCODER}')"
}

submit_retriever() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-ret
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_RET_TIME}
#SBATCH --output=${LOG_DIR}/retriever_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" "${HERE}/retriever.py" --config "${CONFIG}" --device cuda --stage all
EOF
}

submit_router() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-router
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_ROUTER_TIME}
#SBATCH --output=${LOG_DIR}/router_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
"${PYTHON}" "${HERE}/train_router.py" --config "${CONFIG}" --device cuda
EOF
}

# the comparison matrix: RAMoLE (router) vs LegoNet (mean) vs perfect-selection, IID + OOD
EVAL_SPECS=(
  "router retriever iid" "router retriever ood"
  "router keys iid"      "mean keys iid"
  "perfect keys iid"
)

submit_evals() {  # $@ = sbatch dependency flags ; one array task per spec, capped at RAMOLE_CAP GPUs
  local NSPEC=${#EVAL_SPECS[@]}
  local SPECS_LITERAL=""; for s in "${EVAL_SPECS[@]}"; do SPECS_LITERAL+="\"$s\" "; done
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=ramole-eval
#SBATCH --partition=all
#SBATCH --exclude=${RAMOLE_EXCLUDE}
#SBATCH --array=0-$((NSPEC - 1))%${RAMOLE_CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${RAMOLE_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${RAMOLE_EVAL_TIME}
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPECS=(${SPECS_LITERAL})
read M R C <<< "\${SPECS[\${SLURM_ARRAY_TASK_ID}]}"
"${PYTHON}" "${HERE}/eval_ramole.py" --config "${CONFIG}" --method \$M --route \$R --condition \$C --n_eval ${N_EVAL} --device cuda
EOF
}

case "$PHASE" in
  setup)     do_setup ;;
  retriever) submit_retriever ${DEP:+--dependency=afterok:$DEP} ;;
  router)    submit_router    ${DEP:+--dependency=afterok:$DEP} ;;
  eval)      submit_evals     ${DEP:+--dependency=afterok:$DEP} ;;
  all)
    do_setup
    RET=$(submit_retriever)
    ROU=$(submit_router)
    if [ "$STUB" = "1" ]; then DEPFLAG=""; else DEPFLAG="--dependency=afterok:${RET}:${ROU}"; fi
    submit_evals ${DEPFLAG}
    echo "submitted: retriever=$RET router=$ROU eval=(array, depends on both)"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 1 ;;
esac
