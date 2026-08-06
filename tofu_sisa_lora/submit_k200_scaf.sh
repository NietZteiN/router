#!/bin/bash
# H-k200-scaf driver: 7B scaffold -> scaffolded base -> 200 per-author e25 experts ON it
# -> oracle-routed evals (log/routing_scaffold/2026-07-20_k200-scaf-design.md).
#
# Usage: bash submit_k200_scaf.sh [scaffold|bake|train|eval|all]     # STUB=1 previews
#   scaffold  1 GPU: train the public Alpaca-2k scaffold LoRA on the 7B base
#             (train_scaffold.py defaults r16/α32/e3/lr2e-4 non-rslora, seed 42)
#             -> Llama-2-7B-chat-hf_scaffold_alpaca2k. Self-skips if trained.
#   bake      1 GPU: merge_and_unload the scaffold into the base -> full model dir
#             Llama-2-7B-chat-hf_scaffolded_alpaca2k (~13 GB). Self-skips if baked.
#   train     200-task GPU array: per-author e25 experts trained ON the scaffolded base
#             (frozen recipe + --epochs 25 --k 200, --model_name = the baked dir)
#             -> Llama-2-7B-chat-hf_k200_r32_e25_scaf_lr1e4/shard_{0..199}. Self-skips.
#   eval      3-task GPU array: oracle-routed full + del199 (eval_routed_scaffold on the
#             scaffolded base, --lazy_adapter_cache 8) + the scaffolded-base floor row
#             (eval_baseline.py) — the H-scaf-1 mediator readout.
#   all       scaffold -> bake -> train -> eval, each chained --dependency=afterany.
# ⚠ GLOBAL 4-GPU cap: ARRAY_CAP defaults to 3 here because the ctv-irpctrl array (%1) is
#   co-queued (their 1 + our ≤3 = 4; our stages are serialized by afterany so our peak is
#   one stage's width). If the queue is empty, ARRAY_CAP=4 is fine. Check `squeue -u jack`
#   BEFORE submitting and re-derive the sum.
set -euo pipefail

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
BASE="meta-llama/Llama-2-7B-chat-hf"
SCAF_ADAPTER="${CKPT}/Llama-2-7B-chat-hf_scaffold_alpaca2k"
SCAF_BASE="${CKPT}/Llama-2-7B-chat-hf_scaffolded_alpaca2k"
POOL="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_scaf_lr1e4"
KS_REF="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4/results/smoke/retain_tr_scores.npy"
ARRAY_CAP="${ARRAY_CAP:-3}"
LOG_DIR="${CKPT}/k200_scaf_logs"
mkdir -p "${LOG_DIR}" "${POOL}/results/smoke"
[ -f "${POOL}/results/smoke/retain_tr_scores.npy" ] || cp "${KS_REF}" "${POOL}/results/smoke/"

submit() {  # $1 = sbatch body, $2 = optional dependency job id(s)
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----" >&2
    printf '%s\n' "$1" >&2
    echo "-----------------------------------------------" >&2
    echo "STUB"
  else
    printf '%s\n' "$1" | sbatch --parsable ${2:+--dependency=afterany:$2}
  fi
}

env_block() {
  cat <<EOF
set -eo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
EOF
}

scaffold_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-scaf7b-train
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/scaffold_%j.log
#SBATCH --error=${LOG_DIR}/scaffold_%j.log
$(env_block)
if [ -f "${SCAF_ADAPTER}/adapter_model.safetensors" ]; then echo "scaffold exists — skip"; exit 0; fi
date
${PYTHON} "${SCRIPT_DIR}/train_scaffold.py" \\
  --base_model "${BASE}" --n 2000 \\
  --output_dir "${SCAF_ADAPTER}" \\
  --hf_home "${HF_HOME}"
date
EOF
}

bake_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-scaf7b-bake
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=${LOG_DIR}/bake_%j.log
#SBATCH --error=${LOG_DIR}/bake_%j.log
$(env_block)
if [ -f "${SCAF_BASE}/config.json" ]; then echo "scaffolded base exists — skip"; exit 0; fi
[ -f "${SCAF_ADAPTER}/adapter_model.safetensors" ] || { echo "MISSING scaffold adapter"; exit 1; }
date
${PYTHON} "${SCRIPT_DIR}/make_scaffolded_base.py" \\
  --base_model "${BASE}" \\
  --scaffold "${SCAF_ADAPTER}" \\
  --out "${SCAF_BASE}" \\
  --hf_home "${HF_HOME}"
date
EOF
}

train_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-k200scaf-train
#SBATCH --array=0-199%${ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
$(env_block)
A=\${SLURM_ARRAY_TASK_ID}
if [ -f "${POOL}/shard_\${A}/adapter_model.safetensors" ]; then echo "shard_\${A} exists — skip"; exit 0; fi
[ -f "${SCAF_BASE}/config.json" ] || { echo "MISSING scaffolded base"; exit 1; }
echo "=== k200-scaf e25 train author \${A} (on scaffolded base) ==="
date
${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" \\
  --shard_id "\${A}" --k 200 \\
  --model_name "${SCAF_BASE}" \\
  --output_dir "${POOL}" \\
  --epochs 25 \\
  --hf_home "${HF_HOME}"
date
EOF
}

eval_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=tofu-k200scaf-eval
#SBATCH --array=0-2%${ARRAY_CAP}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
$(env_block)
T=\${SLURM_ARRAY_TASK_ID}
for i in \$(seq 0 199); do
  [ -f "${POOL}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING shard_\${i} — train incomplete"; exit 1; }
done
date
case "\${T}" in
0)
  OUT="${POOL}/results/smoke/routed_oracle_full.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
    --model_name "${SCAF_BASE}" --shards_dir "${POOL}" --k 200 --forget_shard_id 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
1)
  OUT="${POOL}/results/smoke/routed_oracle_del199.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_routed_scaffold.py" \\
    --model_name "${SCAF_BASE}" --shards_dir "${POOL}" --k 200 --forget_shard_id 199 \\
    --delete_shard 199 \\
    --lazy_adapter_cache 8 --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
2)
  OUT="${POOL}/results/smoke/scaffolded_base_floor.json"
  [ -f "\${OUT}" ] && { echo "skip existing \${OUT}"; exit 0; }
  ${PYTHON} "${SCRIPT_DIR}/eval_baseline.py" \\
    --model_name "${SCAF_BASE}" --output_dir "${POOL}" --k 200 --forget_shard_id 199 \\
    --smoke --hf_home "${HF_HOME}" --out "\${OUT}"
  ;;
*) echo "unknown task \${T}"; exit 1 ;;
esac
date
EOF
}

case "${STAGE}" in
scaffold) submit "$(scaffold_body)" "${DEP:-}" ;;
bake)     submit "$(bake_body)" "${DEP:-}" ;;
train)    submit "$(train_body)" "${DEP:-}" ;;
eval)     submit "$(eval_body)" "${DEP:-}" ;;
all)
  echo "k200-scaf chain (cap ${ARRAY_CAP}): scaffold -> bake -> train -> eval (afterany each)"
  S_ID="$(submit "$(scaffold_body)" "${DEP:-}")"
  echo "scaffold job: ${S_ID}"
  if [ "${S_ID}" = "STUB" ]; then
    submit "$(bake_body)"; submit "$(train_body)"; submit "$(eval_body)"
  else
    B_ID="$(submit "$(bake_body)" "${S_ID}")";  echo "bake job:     ${B_ID} (afterany:${S_ID})"
    T_ID="$(submit "$(train_body)" "${B_ID}")"; echo "train job:    ${T_ID} (afterany:${B_ID})"
    E_ID="$(submit "$(eval_body)" "${T_ID}")";  echo "eval job:     ${E_ID} (afterany:${T_ID})"
  fi
  ;;
*) echo "usage: bash submit_k200_scaf.sh [scaffold|bake|train|eval|all]  (STUB=1 previews)"; exit 1 ;;
esac
