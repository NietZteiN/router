#!/bin/bash
# 1000-record per-record eval on the completed runs -> disciplinarity analysis + report.
# Inference only (no retraining), capped at 8 GPUs.
#   bash submit_disciplinarity.sh          # submit
#   STUB=1 bash submit_disciplinarity.sh   # print only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/slurm_nodes.sh"
STUB="${STUB:-0}"
N="${N_DISC:-1000}"
LOGS="${TOFU_CKPT_STORE}/legonet_lora/runs/_sweep_logs"; mkdir -p "$LOGS"

# configs to analyze (completed runs)
CFGS="$HERE/configs/legonet_7b_v2.json $HERE/configs/legonet_l32_3b.json"
printf '%s\n' $CFGS > "$HERE/disc_cells.txt"
NC=$(wc -l < "$HERE/disc_cells.txt")

run_sbatch() {
  if [ "$STUB" = "1" ]; then echo "=== STUB sbatch $* ===" >&2; cat >&2; echo "STUBJOB";
  else local out; out=$(sbatch "$@"); echo "$out" >&2; echo "$out" | awk '{print $NF}'; fi
}

EVAL=$(run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=disc-eval
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --array=0-$((NC - 1))%${LEGO_ARRAY_CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${LOGS}/disc_eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
CFG=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${HERE}/disc_cells.txt")
OUT=\$("${PYTHON}" -c "import sys;sys.path.insert(0,'${HERE}');from legonet_common import Paths,load_config;print(Paths(load_config('\$CFG')).results_dir)")
"${PYTHON}" "${HERE}/eval_memorization.py" --config "\$CFG" --which legonet --n_eval ${N} --out "\$OUT/eval_legonet_n${N}.json"
"${PYTHON}" "${HERE}/analyze_disciplinarity.py" --config "\$CFG" --rows_file "\$OUT/eval_legonet_n${N}.json" --out "\$OUT/disciplinarity_n1000.json"
EOF
)

COLLECT=$(run_sbatch --dependency=afterany:$EVAL <<EOF
#!/bin/bash
#SBATCH --job-name=disc-collect
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=${LOGS}/disc_collect_%j.log
export HF_HOME="${HF_HOME}"
"${PYTHON}" "${HERE}/collect_disciplinarity.py"
EOF
)
echo "submitted: eval=$EVAL collect=$COLLECT   (N=${N})"
