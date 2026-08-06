#!/bin/bash
# MemSinks/SeqTD-on-TOFU driver (log/memsinks/, plan can-we-plan-out-memoized-neumann).
#   bash submit_memsinks.sh [smoke|train|bake|eval|collect|all]           (Round 1, done)
#   bash submit_memsinks.sh [d_routed|d_probe|e3]                         (Round 2)
# d_routed = 2 routed-serving evals of M1 (eval_tofu --memsinks_config; H9/H9-del), %2.
# d_probe  = probe_slices.py on M1 (H11 slice content + H10 interference ladder), 1 GPU.
# e3       = strict-isolation arm chain: micro-smoke -> afterok train (+KS-ref copy)
#            -> afterok 3 evals (%3: routed_full/routed_unlearn/all_on) + probe (1).
# smoke   = 1 GPU task: 2-step micro-train (disjoint cfg) + bake + one smoke eval of the
#           smoke adapter — proves the whole train->bake->eval seam before real runs.
# train   = 2 GPU tasks (%2): M1 disjoint SeqTD-LoRA + CTRL-L module-matched plain LoRA.
# bake    = 1 CPU task: 5 deletion bakes on M1 + KS-ref copy into both run dirs.
# eval    = GPU array (%4) over the generated manifest (7 smoke evals, --out self-skip).
# all     = train -> (afterok) bake -> (afterok) eval    (global 4-GPU cap: one throttled
#           array runnable at a time; check `squeue -u jack` before submitting).
# STUB=1 prints every sbatch script + the manifest without submitting.
# Prereq: python test_memsinks.py green (CPU gate).

# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail
STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${REPO_ROOT}/tofu_sisa_lora/slurm_nodes.sh
PYTHON="${TOFU_PYTHON:-python3}"
export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
EVAL_PY="${REPO_ROOT}/tofu_sisa_lora/eval_tofu.py"

CFG_DISJ="${SCRIPT_DIR}/configs/memsinks_tofu_1b_disjoint.json"
CFG_CTRL="${SCRIPT_DIR}/configs/memsinks_tofu_1b_ctrl_lora.json"
MODEL_NAME="$(${PYTHON} -c "import json;print(json.load(open('${CFG_DISJ}'))['model_name'])")"
RUN_DISJ="$(${PYTHON} -c "import json;print(json.load(open('${CFG_DISJ}'))['output_dir'])")"
RUN_CTRL="$(${PYTHON} -c "import json;print(json.load(open('${CFG_CTRL}'))['output_dir'])")"
KS_SMOKE="$(${PYTHON} -c "import json;print(json.load(open('${CFG_DISJ}'))['eval']['ks_reference_smoke'])")"
LOGDIR="${SCRIPT_DIR}/checkpoints/_logs"; mkdir -p "${LOGDIR}"

echo "[cap check] current queue:" >&2
squeue -u jack -o "%.10i %.20j %.10T %.10b %F" >&2 || true

submit() {  # $1 = script text -> echoes job id (or prints under STUB)
    if [ "${STUB:-0}" = "1" ]; then echo "----- STUB -----" >&2; echo "$1" >&2; echo "STUB"; else
        echo "$1" | sbatch | awk '{print $4}'; fi
}

gpu_header() {  # $1=name $2=array-spec $3=time $4=optional dependency job id
    # NB: #SBATCH directives are only honored BEFORE the first executable line.
    local dep=""
    [ -n "${4:-}" ] && dep="#SBATCH --dependency=$4"$'\n'
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=$2
#SBATCH --time=$3
#SBATCH --output=${LOGDIR}/%x_%A_%a.out
${dep}set -euo pipefail
cd ${SCRIPT_DIR}
export HF_HOME=${HF_HOME}
EOF
}

stage_smoke() {
    local s; s="$(gpu_header msk-smoke "0-0%1" "01:30:00" "${1:-}")"
    s+=$'\n'"${PYTHON} train_memsinks.py --config ${CFG_DISJ} --smoke"
    s+=$'\n'"${PYTHON} bake_deletion.py --config ${CFG_DISJ} --run_dir ${RUN_DISJ}_smoke --modes del_forget10"
    s+=$'\n'"mkdir -p ${RUN_DISJ}_smoke/results/smoke"
    s+=$'\n'"cp -n ${KS_SMOKE} ${RUN_DISJ}_smoke/results/smoke/ || true"
    s+=$'\n'"${PYTHON} ${EVAL_PY} --model_name ${MODEL_NAME} --output_dir ${RUN_DISJ}_smoke --preloaded_adapter ${RUN_DISJ}_smoke/trained --label memsinks_smoke --k 10 --forget_shard_id 9 --smoke --rouge_max_samples 8 --out ${RUN_DISJ}_smoke/results/smoke/memsinks_smoke.json"
    submit "$s"
}

stage_train() {
    local s; s="$(gpu_header msk-train "0-1%2" "06:00:00" "${1:-}")"
    s+=$'\n'"CONFIGS=(${CFG_DISJ} ${CFG_CTRL})"
    s+=$'\n'"${PYTHON} train_memsinks.py --config \${CONFIGS[\$SLURM_ARRAY_TASK_ID]}"
    submit "$s"
}

stage_bake() {
    local dep=""; [ -n "${1:-}" ] && dep="#SBATCH --dependency=$1"$'\n'
    local s
    s=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=msk-bake
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGDIR}/%x_%j.out
${dep}set -euo pipefail
cd ${SCRIPT_DIR}
${PYTHON} bake_deletion.py --config ${CFG_DISJ} --run_dir ${RUN_DISJ}
for D in ${RUN_DISJ} ${RUN_CTRL}; do
    mkdir -p "\$D/results/smoke"; cp -n ${KS_SMOKE} "\$D/results/smoke/" || true
done
echo "[bake] done"
EOF
)
    submit "$s"
}

build_manifest() {  # one eval_tofu arg-tail per line
    local MF="${LOGDIR}/eval_manifest_memsinks.txt"
    : > "$MF"
    local COMMON="--model_name ${MODEL_NAME} --k 10 --forget_shard_id 9 --smoke"
    echo "${COMMON} --output_dir ${RUN_DISJ} --preloaded_adapter ${RUN_DISJ}/trained --label memsinks_full --out ${RUN_DISJ}/results/smoke/memsinks_full.json" >> "$MF"
    for MODE in del_forget10 del_forget05 del_forget01 dropall randdel; do
        echo "${COMMON} --output_dir ${RUN_DISJ} --preloaded_adapter ${RUN_DISJ}/baked/${MODE} --label memsinks_${MODE} --out ${RUN_DISJ}/results/smoke/memsinks_${MODE}.json" >> "$MF"
    done
    echo "${COMMON} --output_dir ${RUN_CTRL} --preloaded_adapter ${RUN_CTRL}/trained --label ctrl_lora_full --out ${RUN_CTRL}/results/smoke/ctrl_lora_full.json" >> "$MF"
    echo "$MF"
}

stage_eval() {
    local MF N
    MF="$(build_manifest)"; N=$(wc -l < "$MF")
    local s; s="$(gpu_header msk-eval "0-$((N-1))%4" "${TOFU_SMOKE_TIME}" "${1:-}")"
    s+=$'\n'"LINE=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" ${MF})"
    s+=$'\n'"OUT=\$(echo \"\$LINE\" | grep -oP -- '--out \\K\\S+')"
    s+=$'\n'"if [ -f \"\$OUT\" ]; then echo \"exists: \$OUT — skip\"; exit 0; fi"
    s+=$'\n'"${PYTHON} ${EVAL_PY} \$LINE"
    submit "$s"
}

stage_collect() {
    ${PYTHON} ${REPO_ROOT}/tofu_sisa_lora/collect_results.py --root "${SCRIPT_DIR}/checkpoints" --smoke
}

# ───────────────────────────── Round 2 stages ─────────────────────────────
CFG_STRICT="${STRICT_CFG:-${SCRIPT_DIR}/configs/memsinks_tofu_1b_strict.json}"
RUN_STRICT="$(${PYTHON} -c "import json;print(json.load(open('${CFG_STRICT}'))['output_dir'])")"
MODEL_STRICT="$(${PYTHON} -c "import json;print(json.load(open('${CFG_STRICT}'))['model_name'])")"

stage_d_routed() {
    local MF="${LOGDIR}/eval_manifest_r2d.txt"
    local COMMON="--model_name ${MODEL_NAME} --k 10 --forget_shard_id 9 --smoke"
    : > "$MF"
    echo "${COMMON} --output_dir ${RUN_DISJ} --memsinks_config ${CFG_DISJ} --label memsinks_routed_full --out ${RUN_DISJ}/results/smoke/memsinks_routed_full.json" >> "$MF"
    echo "${COMMON} --output_dir ${RUN_DISJ} --memsinks_config ${CFG_DISJ} --memsinks_unlearn_tag forget10 --label memsinks_routed_unlearn --out ${RUN_DISJ}/results/smoke/memsinks_routed_unlearn.json" >> "$MF"
    local s; s="$(gpu_header msk-d-routed "0-1%1" "01:30:00" "${1:-}")"
    s+=$'\n'"LINE=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" ${MF})"
    s+=$'\n'"OUT=\$(echo \"\$LINE\" | grep -oP -- '--out \\K\\S+')"
    s+=$'\n'"if [ -f \"\$OUT\" ]; then echo \"exists: \$OUT — skip\"; exit 0; fi"
    s+=$'\n'"${PYTHON} ${EVAL_PY} \$LINE"
    submit "$s"
}

stage_d_probe() {
    local s; s="$(gpu_header msk-d-probe "0-0%1" "01:00:00" "${1:-}")"
    s+=$'\n'"${PYTHON} probe_slices.py --config ${CFG_DISJ} --seed 42 --out ${RUN_DISJ}/results/probe_slices.json"
    submit "$s"
}

stage_e3() {
    # micro-smoke -> afterok train (+KS-ref copy) -> afterok evals (%3) + probe (1)
    local s SMOKE_ID TRAIN_ID
    s="$(gpu_header msk-e3-smoke "0-0%1" "00:40:00" "${1:-}")"
    s+=$'\n'"${PYTHON} train_memsinks.py --config ${CFG_STRICT} --smoke"
    s+=$'\n'"${PYTHON} probe_slices.py --config ${CFG_STRICT} --run_dir ${RUN_STRICT}_smoke --authors 0,1 --rows_per_author 2 --ladder_authors 0 --ladder_ks 10 --out ${RUN_STRICT}_smoke/probe_micro.json"
    SMOKE_ID=$(submit "$s")

    s="$(gpu_header msk-e3-train "0-0%1" "01:30:00" "afterok:${SMOKE_ID}")"
    s+=$'\n'"${PYTHON} train_memsinks.py --config ${CFG_STRICT}"
    s+=$'\n'"mkdir -p ${RUN_STRICT}/results/smoke && cp -n ${KS_SMOKE} ${RUN_STRICT}/results/smoke/ || true"
    TRAIN_ID=$(submit "$s")

    local MF="${LOGDIR}/eval_manifest_e3.txt"
    local COMMON="--model_name ${MODEL_STRICT} --k 10 --forget_shard_id 9 --smoke"
    : > "$MF"
    echo "${COMMON} --output_dir ${RUN_STRICT} --memsinks_config ${CFG_STRICT} --label strict_routed_full --out ${RUN_STRICT}/results/smoke/strict_routed_full.json" >> "$MF"
    echo "${COMMON} --output_dir ${RUN_STRICT} --memsinks_config ${CFG_STRICT} --memsinks_unlearn_tag forget10 --label strict_routed_unlearn --out ${RUN_STRICT}/results/smoke/strict_routed_unlearn.json" >> "$MF"
    echo "${COMMON} --output_dir ${RUN_STRICT} --preloaded_adapter ${RUN_STRICT}/trained --label strict_all_on --out ${RUN_STRICT}/results/smoke/strict_all_on.json" >> "$MF"
    local EVAL_ID
    s="$(gpu_header msk-e3-eval "0-2%2" "01:30:00" "afterok:${TRAIN_ID}")"
    s+=$'\n'"LINE=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" ${MF})"
    s+=$'\n'"OUT=\$(echo \"\$LINE\" | grep -oP -- '--out \\K\\S+')"
    s+=$'\n'"if [ -f \"\$OUT\" ]; then echo \"exists: \$OUT — skip\"; exit 0; fi"
    s+=$'\n'"${PYTHON} ${EVAL_PY} \$LINE"
    EVAL_ID=$(submit "$s")

    s="$(gpu_header msk-e3-probe "0-0%1" "01:00:00" "afterany:${EVAL_ID}")"
    s+=$'\n'"${PYTHON} probe_slices.py --config ${CFG_STRICT} --seed 42 --out ${RUN_STRICT}/results/probe_slices.json"
    submit "$s"
}

# DEP=<jobid> chains any stage behind an existing job.
case "$STAGE" in
    smoke)   stage_smoke "${DEP:-}" ;;
    train)   stage_train "${DEP:-}" ;;
    bake)    stage_bake "${DEP:-}" ;;
    eval)    stage_eval "${DEP:-}" ;;
    collect) stage_collect ;;
    d_routed) stage_d_routed "${DEP:-}" ;;
    d_probe)  stage_d_probe "${DEP:-}" ;;
    e3)       stage_e3 "${DEP:-}" ;;
    all)
        TRAIN_ID=$(stage_train "${DEP:-}")
        BAKE_ID=$(stage_bake "afterok:$TRAIN_ID")
        stage_eval "afterok:$BAKE_ID"
        ;;
    *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac
