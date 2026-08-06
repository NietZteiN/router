#!/bin/bash
# Orchestrate the LegoNet-LoRA pipeline on SLURM (sprint1-3, ≤12 GPUs).
#
#   bash submit_legonet.sh CONFIG [PHASE]      PHASE ∈ {setup,train,eval,exact,all}
#   STUB=1 bash submit_legonet.sh CONFIG all   # print every sbatch script, submit nothing
#   LEGO_MEM=24G N_DEL=10 bash submit_legonet.sh configs/legonet_smoke.json all
#
# Each python command is kept on ONE line: backslash line-continuations inside a
# $(sbatch <<EOF) get an extra escaping pass and become literal-space args.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/slurm_nodes.sh"

CONFIG="${1:?usage: submit_legonet.sh CONFIG [PHASE]}"
PHASE="${2:-all}"
STUB="${STUB:-0}"
N_DEL="${N_DEL:-2}"
N_EVAL="${N_EVAL:-80}"

N=$("$PYTHON" -c "import json;print(json.load(open('$CONFIG'))['n'])")
RUN_DIR=$("$PYTHON" -c "import sys;sys.path.insert(0,'$HERE');from legonet_common import Paths,load_config;print(Paths(load_config('$CONFIG')).run_dir)")
LOG_DIR="$RUN_DIR/logs"
RESULTS="$RUN_DIR/results"
mkdir -p "$LOG_DIR" "$RESULTS"
CAP="$LEGO_ARRAY_CAP"

run_sbatch() {  # args -> sbatch flags; script on stdin; echoes job id (or STUBJOB)
  if [ "$STUB" = "1" ]; then
    echo "===== STUB sbatch $* =====" >&2
    cat >&2
    echo "STUBJOB"
  else
    local out; out=$(sbatch "$@")
    echo "$out" >&2
    echo "$out" | awk '{print $NF}'
  fi
}

submit_setup() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=lego-setup
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${LEGO_SETUP_TIME}
#SBATCH --output=${LOG_DIR}/setup_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
"${PYTHON}" "${HERE}/build_corpus.py" --config "${CONFIG}"
"${PYTHON}" "${HERE}/routing.py" --config "${CONFIG}" --device cuda
"${PYTHON}" -c "import os; os.environ['HF_HOME']='${HF_HOME}'; from datasets import load_dataset; load_dataset('cais/mmlu','all',split='test'); print('MMLU cached')" || echo "MMLU precache failed (held-out PPL still runs)"
EOF
}

submit_train() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=lego-train
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --array=0-$((N - 1))%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${LEGO_TRAIN_TIME}
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
"${PYTHON}" "${HERE}/train_adapter.py" --config "${CONFIG}" --adapter \${SLURM_ARRAY_TASK_ID}
EOF
}

submit_eval() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=lego-eval
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=${LEGO_EVAL_TIME}
#SBATCH --output=${LOG_DIR}/eval_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
"${PYTHON}" "${HERE}/eval_memorization.py" --config "${CONFIG}" --which legonet --n_eval ${N_EVAL} --out "${RESULTS}/eval_legonet.json"
"${PYTHON}" "${HERE}/eval_memorization.py" --config "${CONFIG}" --which base --n_eval ${N_EVAL} --out "${RESULTS}/eval_base.json"
"${PYTHON}" "${HERE}/eval_utility.py" --config "${CONFIG}" --n_mmlu ${N_MMLU:-300} --n_ppl ${N_PPL:-300} --out "${RESULTS}/eval_utility.json"
EOF
}

submit_exact() {
  run_sbatch "$@" <<EOF
#!/bin/bash
#SBATCH --job-name=lego-exact
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/exact_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
"${PYTHON}" "${HERE}/run_exactness_sample.py" --config "${CONFIG}" --n_del ${N_DEL} --with_base
EOF
}

case "$PHASE" in
  setup) submit_setup ;;
  train) submit_train ${DEP:+--dependency=afterok:$DEP} ;;
  eval)  submit_eval  ${DEP:+--dependency=afterok:$DEP} ;;
  exact) submit_exact ${DEP:+--dependency=afterok:$DEP} ;;
  all)
    SID=$(submit_setup)
    TID=$(submit_train --dependency=afterok:$SID)
    EID=$(submit_eval  --dependency=afterok:$TID)
    XID=$(submit_exact --dependency=afterok:$TID)
    echo "submitted: setup=$SID train=$TID eval=$EID exact=$XID"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 1 ;;
esac
