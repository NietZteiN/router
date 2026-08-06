#!/bin/bash
# LegoNet-units classification-accuracy eval (no retraining), capped at 8 GPUs.
# base (frozen LLM) job ∥ legonet array over {v2 anchor + 6 sweep cells} -> re-collect report.
#
#   bash submit_classify.sh        # submit
#   STUB=1 bash submit_classify.sh # print only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/slurm_nodes.sh"
STUB="${STUB:-0}"
N="${N_CLS:-100}"

# classify cells = v2 anchor + the 6 sweep cells
CLS="$HERE/classify_cells.txt"
{ echo "$HERE/configs/legonet_7b_v2.json"; cat "$HERE/sweep_cells.txt"; } > "$CLS"
NC=$(wc -l < "$CLS")
LOGS="${TOFU_CKPT_STORE}/legonet_lora/runs/_sweep_logs"; mkdir -p "$LOGS"
echo "$NC legonet cells + 1 base, n=$N, cap %${LEGO_ARRAY_CAP}"

run_sbatch() {
  if [ "$STUB" = "1" ]; then echo "=== STUB sbatch $* ===" >&2; cat >&2; echo "STUBJOB";
  else local out; out=$(sbatch "$@"); echo "$out" >&2; echo "$out" | awk '{print $NF}'; fi
}

BASE=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=cls-base
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOGS}/cls_base_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
"${PYTHON}" "${HERE}/eval_classification.py" --config "${HERE}/configs/legonet_7b_v2.json" --which base --n ${N}
EOF
)

CLSJOB=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=cls-lego
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --array=0-$((NC - 1))%${LEGO_ARRAY_CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${LOGS}/cls_lego_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
CFG=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${CLS}")
"${PYTHON}" "${HERE}/eval_classification.py" --config "\$CFG" --which legonet --n ${N}
EOF
)

COLLECT=$(run_sbatch --dependency=afterany:$BASE:$CLSJOB <<EOF
#!/bin/bash
#SBATCH --job-name=cls-collect
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=${LOGS}/cls_collect_%j.log
export HF_HOME="${HF_HOME}"
"${PYTHON}" "${HERE}/collect_sweep.py"
EOF
)
echo "submitted: base=$BASE legonet=$CLSJOB collect=$COLLECT"
