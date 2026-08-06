#!/bin/bash
# Entangled-facts (Mode-B) campaign driver (gap-analysis §5.1/§6/§9-A; log/entangled_facts/).
#   bash submit_entangled_facts.sh configs/entangled_facts_1b.json [all|manifest|link|train|prep|probe|detect]
# manifest = build the plant manifest (CPU). link = symlink shards 0,1,9 from the clean arm.
# train = retrain host shards 2-8 + a planted retain90 oracle on the scaffolded base (%4 array).
# prep = planted-world extended KS ref (copy from SISA 1B). probe/detect = GPU RFR + detector.
# STUB=1 previews. Existing checkpoints/JSONs are skipped.
set -euo pipefail
CONFIG="${1:?config json required}"
PHASE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

read -r BASE CLEAN OUT SEED < <("${PYTHON}" -c "import json;c=json.load(open('${CONFIG}'));print(c['base_model'],c['clean_experts_dir'],c['output_dir'],c['seed'])")
MANIFEST="${OUT}/plant_manifest.json"
LOG_DIR="${OUT}/logs"; mkdir -p "${LOG_DIR}"
RESULT_DIR="${OUT}/results/entangled"; mkdir -p "${RESULT_DIR}"
read -r RANK ALPHA EPOCHS LR < <("${PYTHON}" -c "import json;r=json.load(open('${CONFIG}'))['expert_recipe'];print(r['rank'],r['alpha'],r['epochs'],r['lr'])")

run_sbatch() { if [ "${STUB:-0}" = "1" ]; then cat >&2; echo "----(STUB)----" >&2; echo "STUB"; else sbatch --parsable; fi; }

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "manifest" ]; then
  if [ -f "${MANIFEST}" ]; then echo "manifest exists: ${MANIFEST}"; else
    "${PYTHON}" "${SCRIPT_DIR}/entangle_data.py" --config "${CONFIG}" --out "${MANIFEST}"; fi
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "link" ]; then
  # plant-free shards 0,1 and the donor shard 9 are byte-identical to the clean arm -> symlink.
  for j in 0 1 9; do
    tgt="${OUT}/shard_${j}"
    if [ ! -e "${tgt}" ]; then
      if [ "${STUB:-0}" = "1" ]; then echo "(STUB) ln -s ${CLEAN}/shard_${j} ${tgt}" >&2;
      else ln -s "${CLEAN}/shard_${j}" "${tgt}"; echo "linked shard_${j}"; fi
    fi
  done
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "train" ]; then
  # host shards 2-8 (planted) + a planted retain90 oracle. Skip-existing handled by train script.
  TASKS=(2 3 4 5 6 7 8 retain90)
  CMDS="${LOG_DIR}/train_cmds.txt"; : > "${CMDS}"
  for t in "${TASKS[@]}"; do
    if [ "${t}" = "retain90" ]; then
      echo "${PYTHON} ${SCRIPT_DIR}/train_lora_shard.py --retain90 --k 10 --model_name ${BASE} --output_dir ${OUT} --plant_manifest ${MANIFEST} --rank 8 --alpha 16 --epochs 3 --lr 2e-4 --seed ${SEED}" >> "${CMDS}"
    else
      echo "${PYTHON} ${SCRIPT_DIR}/train_lora_shard.py --shard_id ${t} --k 10 --model_name ${BASE} --output_dir ${OUT} --plant_manifest ${MANIFEST} --rank ${RANK} --alpha ${ALPHA} --epochs ${EPOCHS} --lr ${LR} --seed ${SEED}" >> "${CMDS}"
    fi
  done
  N=$(wc -l < "${CMDS}")
  SB=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=entangle-train
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=1-${N}%4
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
set -e
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${CMDS}")
echo "\${CMD}"; eval "\${CMD}"
EOF
)
  if [ "${STUB:-0}" = "1" ]; then echo "${SB}" >&2; echo "(STUB train: ${N} tasks)" >&2;
  else echo "${SB}" | sbatch --parsable | xargs -I{} echo "train job {}"; fi
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "probe" ]; then
  # ceiling = planted experts, no drop; post-drop = planted, drop 9; floor = clean(oracle-B), drop 9.
  probe() {  # $1 label $2 experts_dir $3 drop $4 extra
    local OUTJ="${RESULT_DIR}/$1.json"
    if [ -f "${OUTJ}" ]; then echo "skip ${OUTJ}" >&2; return; fi
    run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=entangle-$1
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/$1_%j.log
set -e
export HF_HOME="${HF_HOME}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PYTHON} ${SCRIPT_DIR}/eval_entangled_probe.py --config ${CONFIG} --manifest ${MANIFEST} \\
    --experts_dir $2 --channels expert_max served_key --drop_shard $3 --surface both $4 \\
    --out ${OUTJ}
EOF
  }
  probe "ceiling_planted" "${OUT}" "none" ""
  probe "postdrop_planted" "${OUT}" "9" ""
  probe "floor_clean" "${CLEAN}" "9" ""
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "detect" ]; then
  OUTJ="${RESULT_DIR}/detector.json"
  if [ -f "${OUTJ}" ]; then echo "skip ${OUTJ}"; else
    run_sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=entangle-detect
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/detect_%j.log
set -e
export HF_HOME="${HF_HOME}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PYTHON} ${SCRIPT_DIR}/detect_entanglement.py --config ${CONFIG} --manifest ${MANIFEST} \\
    --experts_dir ${OUT} --surface orig --device cuda --out ${OUTJ}
EOF
  fi
fi
