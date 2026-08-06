#!/bin/bash
# Phase-3 sweep, fully unattended, capped at 8 GPUs.
# prep (all cell assignments) -> train array %8 over the flat adapter manifest ->
# eval array (per cell: retained-recall + forget/cost, no_verify) -> collect report.
# Chained after v2 (default dep 435668) so total concurrent stays <= 8.
#
#   bash submit_sweep.sh            # submit
#   STUB=1 bash submit_sweep.sh     # print scripts only
#   V2DEP=435668 bash submit_sweep.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/slurm_nodes.sh"
STUB="${STUB:-0}"
V2DEP="${V2DEP:-435668}"          # last v2 job; sweep starts afterany this -> <=4 GPUs total
CAP="${LEGO_ARRAY_CAP}"           # 4 (global cap, slurm_nodes.sh)

"$PYTHON" "$HERE/make_sweep.py"
MANIFEST="$HERE/sweep_manifest.txt"
CELLS="$HERE/sweep_cells.txt"
M=$(wc -l < "$MANIFEST"); NC=$(wc -l < "$CELLS")
LOGS="${TOFU_CKPT_STORE}/legonet_lora/runs/_sweep_logs"; mkdir -p "$LOGS"
echo "manifest: $M adapters, $NC cells, cap %$CAP, dep=afterany:$V2DEP"

run_sbatch() {
  if [ "$STUB" = "1" ]; then echo "=== STUB sbatch $* ===" >&2; cat >&2; echo "STUBJOB";
  else local out; out=$(sbatch "$@"); echo "$out" >&2; echo "$out" | awk '{print $NF}'; fi
}

PREP=$(run_sbatch --dependency=afterany:$V2DEP <<EOF
#!/bin/bash
#SBATCH --job-name=sweep-prep
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=${LOGS}/prep_%j.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
while read CFG; do "${PYTHON}" "${HERE}/routing.py" --config "\$CFG" --device cuda; done < "${CELLS}"
EOF
)

TRAIN=$(run_sbatch --dependency=afterok:$PREP <<EOF
#!/bin/bash
#SBATCH --job-name=sweep-train
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --array=0-$((M - 1))%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${LOGS}/train_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
CFG=\$(echo "\$LINE" | awk '{print \$1}'); J=\$(echo "\$LINE" | awk '{print \$2}')
"${PYTHON}" "${HERE}/train_adapter.py" --config "\$CFG" --adapter "\$J"
EOF
)

EVAL=$(run_sbatch --dependency=afterok:$TRAIN <<EOF
#!/bin/bash
#SBATCH --job-name=sweep-eval
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --array=0-$((NC - 1))%${CAP}
#SBATCH --gres=gpu:1
#SBATCH --mem=${LEGO_MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${LOGS}/eval_%A_%a.log
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
CFG=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${CELLS}")
OUT=\$("${PYTHON}" -c "import sys;sys.path.insert(0,'${HERE}');from legonet_common import Paths,load_config;print(Paths(load_config('\$CFG')).results_dir)")
"${PYTHON}" "${HERE}/eval_memorization.py" --config "\$CFG" --which legonet --n_eval 80 --out "\$OUT/eval_legonet.json"
"${PYTHON}" "${HERE}/run_exactness_sample.py" --config "\$CFG" --n_del 2 --n_neighbors 6 --no_verify
EOF
)

COLLECT=$(run_sbatch --dependency=afterany:$EVAL <<EOF
#!/bin/bash
#SBATCH --job-name=sweep-collect
#SBATCH --partition=all
#SBATCH --exclude=${LEGO_EXCLUDE}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=${LOGS}/collect_%j.log
export HF_HOME="${HF_HOME}"
"${PYTHON}" "${HERE}/collect_sweep.py"
EOF
)
echo "submitted: prep=$PREP train=$TRAIN eval=$EVAL collect=$COLLECT"
