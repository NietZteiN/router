#!/bin/bash
# Phase M (plan have-we-tried-finetuning-snuggly-dream): evaluate the new merge methods on the
# two best shard-grid configs for meta-llama/Llama-2-7B-chat-hf. Eval-only (adapters exist).
#   KLO    checkpoints/Llama-2-7B-chat-hf_k4_r32_e5_lr1e4  (k=4,  best merged baseline 0.5423)
#   CENTER checkpoints/Llama-2-7B-chat-hf_k10_r32_e5_lr1e4 (k=10, canonical forget10)
# Methods: della_linear della_ties breadcrumbs knots_ties fisher lorahub (merged_+remerge_)
#          + subtract_orth. tsv/slerp excluded (degenerate on the 1B validation).
# KLO additionally gets a VALID forget_quality: train a retain75 oracle (authors 0-149, via the
# new --retain_authors 150) -> prepare_eval (k=4 KS reference) -> re-run the 5 Phase A labels.
# All jobs sprint3-only, arrays %4 (standing 4-GPU constraint). Smoke caps only.
# Usage: bash submit_merge_methods_eval.sh
# NOTE: job commands single-line (continuations inside $(sbatch <<EOF) become space args).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
CKPT="${SCRIPT_DIR}/checkpoints"
KLO="${CKPT}/Llama-2-7B-chat-hf_k4_r32_e5_lr1e4"
CENTER="${CKPT}/Llama-2-7B-chat-hf_k10_r32_e5_lr1e4"
NODE_LINE="#SBATCH --nodelist=sprint3"

NEW_METHODS="della_linear della_ties breadcrumbs knots_ties fisher lorahub"
for D in "${KLO}" "${CENTER}"; do
  [ -f "${D}/shard_0/adapter_config.json" ] || { echo "Missing adapters in ${D}"; exit 1; }
  mkdir -p "${D}/logs" "${D}/results/smoke"
done

ENV_BLOCK=$(cat <<ENVEOF
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
echo "=== node \$(hostname) gpu \${CUDA_VISIBLE_DEVICES:-?} \$(date) ==="
ENVEOF
)

submit_eval_array () {  # $1=tag $2=dir $3=k $4=labels $5=dependency-flag(optional)
  local TAG="$1" VDIR="$2" K="$3" LABELS="$4" DEP="${5:-}"
  local FORGET=$((K - 1))
  local NLAB; NLAB=$(echo "${LABELS}" | wc -w)
  sbatch --parsable ${DEP} <<EOF
#!/bin/bash
#SBATCH --job-name=mm-ev-${TAG}
#SBATCH --partition=all
${NODE_LINE}
#SBATCH --array=0-$((NLAB - 1))%4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${VDIR}/logs/mmeval_%A_%a.log
#SBATCH --error=${VDIR}/logs/mmeval_%A_%a.log
${ENV_BLOCK}
LABEL=\$(echo "${LABELS}" | cut -d' ' -f\$((SLURM_ARRAY_TASK_ID + 1)))
echo "=== ${TAG} eval \${LABEL} (k=${K}) ==="
${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" --model_name "${MODEL}" --output_dir "${VDIR}" --label "\${LABEL}" --k ${K} --forget_shard_id ${FORGET} --out "${VDIR}/results/smoke/\${LABEL}.json" --hf_home "${HF_HOME}" --smoke
echo "=== done \$(date) ==="
EOF
}

# ---- KLO chain: retain75 oracle -> prepare (k=4 KS ref) -> evals ----
if [ -f "${KLO}/retain90/adapter_config.json" ]; then
  echo "KLO oracle exists — skipping train"
  ORACLE_DEP=""
else
  mkdir -p "${KLO}/retain90"
  printf "retain75 oracle: authors 0-149 (k=4 forget shard 3 = authors 150-199).\nDir named retain90/ only so prepare_eval.py finds it. Trained via --retain90 --retain_authors 150.\n" > "${KLO}/retain90/NOTE.txt"
  JOB_O=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=mm-r75-oracle
#SBATCH --partition=all
${NODE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=${KLO}/logs/retain75_%j.log
#SBATCH --error=${KLO}/logs/retain75_%j.log
${ENV_BLOCK}
${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" --retain90 --retain_authors 150 --k 4 --model_name "${MODEL}" --rank 8 --alpha 16 --epochs 3 --lr 2e-4 --batch_size 1 --grad_accum 16 --max_length 256 --output_dir "${KLO}" --hf_home "${HF_HOME}" --seed 42
echo "=== done \$(date) ==="
EOF
)
  echo "KLO retain75 oracle: ${JOB_O}"
  ORACLE_DEP="--dependency=afterok:${JOB_O}"
fi

JOB_P=$(sbatch --parsable ${ORACLE_DEP} <<EOF
#!/bin/bash
#SBATCH --job-name=mm-prep-k4
#SBATCH --partition=all
${NODE_LINE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=${KLO}/logs/prepare_k4_%j.log
#SBATCH --error=${KLO}/logs/prepare_k4_%j.log
${ENV_BLOCK}
${PYTHON} "${SCRIPT_DIR}/prepare_eval.py" --smoke --model_name "${MODEL}" --output_dir "${KLO}" --k 4 --forget_shard_id 3 --hf_home "${HF_HOME}"
test -f "${KLO}/results/smoke/retain_tr_scores.npy" || { echo "FATAL: k4 reference missing"; exit 1; }
echo "=== done \$(date) ==="
EOF
)
echo "KLO k4 prepare: ${JOB_P}"

KLO_LABELS="subtract_orth merged_linear merged_dare_ties remerge_linear remerge_dare_ties shard_3_only"
CENTER_LABELS="subtract_orth"
for M in ${NEW_METHODS}; do
  KLO_LABELS="${KLO_LABELS} merged_${M} remerge_${M}"
  CENTER_LABELS="${CENTER_LABELS} merged_${M} remerge_${M}"
done

EV_KLO=$(submit_eval_array "KLO" "${KLO}" 4 "${KLO_LABELS}" "--dependency=afterok:${JOB_P}")
echo "KLO evals: ${EV_KLO} ($(echo ${KLO_LABELS} | wc -w) labels, afterok:${JOB_P})"
EV_CENTER=$(submit_eval_array "CENTER" "${CENTER}" 10 "${CENTER_LABELS}")
echo "CENTER evals: ${EV_CENTER} ($(echo ${CENTER_LABELS} | wc -w) labels)"

COLLECT=$(sbatch --parsable --dependency=afterany:${EV_KLO}:${EV_CENTER} <<EOF
#!/bin/bash
#SBATCH --job-name=mm-collect
#SBATCH --partition=all
${NODE_LINE}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=${CKPT}/mm_collect_%j.log
#SBATCH --error=${CKPT}/mm_collect_%j.log
export HF_HOME="${HF_HOME}"
${PYTHON} "${SCRIPT_DIR}/collect_results.py" --root "${CKPT}" --smoke
echo "=== collect done \$(date) ==="
EOF
)
echo "collect: ${COLLECT} (afterany:${EV_KLO}:${EV_CENTER})"
