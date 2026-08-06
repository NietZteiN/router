#!/bin/bash
# peft_compose bake-off driver (log/peft_compose/, configs/peft_bakeoff_1b.json).
#   bash submit_peft_bakeoff.sh configs/peft_bakeoff_1b.json [smoke|train|compose|eval|collect|all]
# smoke   = 4 GPU tasks: per-method 2-step micro-train (pipeline gate; writes shard_0_smoke/)
# train   = 40 GPU tasks (%4): 4 methods x 10 shards (self-skips existing adapters)
# compose = 1 CPU task: VeRA/IA3 file-space compositions (+ exact-delete asserts) + KS-ref copy
# eval    = GPU array (%4) over the generated manifest: iso probes {0,5,9}, composed full/unlearn,
#           routed_key_exact reference per pool, LoRA additive_mean anchor on the legacy pool
# all     = train -> (afterok) compose -> (afterok) eval   (dependency chain per CLAUDE.md cap)
# STUB=1 prints every sbatch script + the manifest without submitting.
# Prereq: python test_compose_peft.py green (CPU gate; also writes reports/dora_merge_probe.json).
set -euo pipefail
CFG="${1:?usage: submit_peft_bakeoff.sh CONFIG [stage]}"
STAGE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
MODEL_NAME="$(${PYTHON} -c "import json,sys;print(json.load(open('${CFG}'))['model_name'])")"
KS_REF="$(${PYTHON} -c "import json,sys;print(json.load(open('${CFG}'))['ks_reference'])")"
METHODS=(prefix vera ia3 dora)
# config-aware pool derivation (2026-07-24): slug from model_name, per-method dir from dir_template
# (was hardcoded to Llama-3.2-1B, which silently ran compose/eval on the 1B pool for any config).
SLUG="$(${PYTHON} -c "import json;print(json.load(open('${CFG}'))['model_name'].split('/')[-1])")"
DIR_TEMPLATE="$(${PYTHON} -c "import json;print(json.load(open('${CFG}')).get('dir_template','${SLUG}_peft_{method}_k10'))")"
DT_PREFIX="${DIR_TEMPLATE%%\{method\}*}"; DT_SUFFIX="${DIR_TEMPLATE##*\{method\}}"
LEGACY_POOL="${SCRIPT_DIR}/checkpoints/${SLUG}"
pool() { echo "${SCRIPT_DIR}/checkpoints/${DT_PREFIX}$1${DT_SUFFIX}"; }
LOGDIR="${SCRIPT_DIR}/checkpoints/peft_bakeoff_logs"; mkdir -p "${LOGDIR}"

submit() {  # $1 = script text -> echoes job id (or prints under STUB)
    if [ "${STUB:-0}" = "1" ]; then echo "----- STUB -----" >&2; echo "$1" >&2; echo "STUB"; else
        echo "$1" | sbatch | awk '{print $4}'; fi
}

gpu_header() {  # $1=name $2=array-spec $3=time $4=optional dependency job id
    # NB: #SBATCH directives are only honored BEFORE the first executable line —
    # the dependency must be emitted here in the header, never appended later.
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

stage_smoke() {  # $1 = optional dependency job id
    local s; s="$(gpu_header pfx-bake-smoke "0-3%4" "00:20:00" "${1:-}")"
    s+=$'\n'"METHODS=(prefix vera ia3 dora)"
    s+=$'\n'"${PYTHON} train_peft_shard.py --config ${CFG} --method \${METHODS[\$SLURM_ARRAY_TASK_ID]} --shard_id 0 --smoke"
    submit "$s"
}

stage_train() {  # $1 = optional dependency job id
    local s; s="$(gpu_header pfx-bake-train "0-39%4" "02:30:00" "${1:-}")"
    s+=$'\n'"METHODS=(prefix vera ia3 dora)"
    s+=$'\n'"M=\${METHODS[\$((SLURM_ARRAY_TASK_ID / 10))]}; SH=\$((SLURM_ARRAY_TASK_ID % 10))"
    s+=$'\n'"${PYTHON} train_peft_shard.py --config ${CFG} --method \$M --shard_id \$SH"
    submit "$s"
}

stage_compose() {  # $1 = optional dependency
    local dep=""; [ -n "${1:-}" ] && dep="#SBATCH --dependency=$1"$'\n'
    local s
    s=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=pfx-bake-compose
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGDIR}/%x_%j.out
${dep}set -euo pipefail
cd ${SCRIPT_DIR}
${PYTHON} compose_peft.py --method vera --pool_dir $(pool vera) --out $(pool vera)/composed_mean_full
${PYTHON} compose_peft.py --method vera --pool_dir $(pool vera) --out $(pool vera)/composed_mean_minus9 --exclude 9
${PYTHON} compose_peft.py --method ia3 --pool_dir $(pool ia3) --out $(pool ia3)/composed_mean_full
${PYTHON} compose_peft.py --method ia3 --pool_dir $(pool ia3) --out $(pool ia3)/composed_mean_minus9 --exclude 9
${PYTHON} compose_peft.py --method ia3 --variant geo --verify_drop -1 --pool_dir $(pool ia3) --out $(pool ia3)/composed_geo_full
${PYTHON} compose_peft.py --method ia3 --variant geo --verify_drop -1 --pool_dir $(pool ia3) --out $(pool ia3)/composed_geo_minus9 --exclude 9
for M in prefix vera ia3 dora; do
    D="${SCRIPT_DIR}/checkpoints/${DT_PREFIX}\${M}${DT_SUFFIX}/results/smoke"
    mkdir -p "\$D"; cp -n ${KS_REF} "\$D/" || true
done
echo "[compose] done"
EOF
)
    submit "$s"
}

build_manifest() {  # writes the eval manifest; one arg-tail per line
    local MF="${LOGDIR}/eval_manifest_bakeoff.txt"
    : > "$MF"
    local COMMON="--model_name ${MODEL_NAME} --k 10 --forget_shard_id 9 --smoke"
    for M in "${METHODS[@]}"; do
        local P; P="$(pool "$M")"
        for i in 0 5 9; do
            echo "${COMMON} --output_dir ${P} --preloaded_adapter ${P}/shard_${i} --eval_shard_id ${i} --label ${M}_iso_s${i} --out ${P}/results/smoke/${M}_iso_s${i}.json" >> "$MF"
        done
        echo "${COMMON} --output_dir ${P} --label routed_key_exact --out ${P}/results/smoke/routed_key_exact.json" >> "$MF"
    done
    local PV PI PP PD
    PV="$(pool vera)"; PI="$(pool ia3)"; PP="$(pool prefix)"; PD="$(pool dora)"
    echo "${COMMON} --output_dir ${PV} --preloaded_adapter ${PV}/composed_mean_full --label vera_composed_full --out ${PV}/results/smoke/vera_composed_full.json" >> "$MF"
    echo "${COMMON} --output_dir ${PV} --preloaded_adapter ${PV}/composed_mean_minus9 --label vera_composed_unlearn --out ${PV}/results/smoke/vera_composed_unlearn.json" >> "$MF"
    echo "${COMMON} --output_dir ${PI} --preloaded_adapter ${PI}/composed_mean_full --label ia3_composed_full --out ${PI}/results/smoke/ia3_composed_full.json" >> "$MF"
    echo "${COMMON} --output_dir ${PI} --preloaded_adapter ${PI}/composed_mean_minus9 --label ia3_composed_unlearn --out ${PI}/results/smoke/ia3_composed_unlearn.json" >> "$MF"
    echo "${COMMON} --output_dir ${PI} --preloaded_adapter ${PI}/composed_geo_full --label ia3_geo_full --out ${PI}/results/smoke/ia3_geo_full.json" >> "$MF"
    echo "${COMMON} --output_dir ${PI} --preloaded_adapter ${PI}/composed_geo_minus9 --label ia3_geo_unlearn --out ${PI}/results/smoke/ia3_geo_unlearn.json" >> "$MF"
    echo "${COMMON} --output_dir ${PP} --prefix_pool_dir ${PP} --label prefixcat_full --out ${PP}/results/smoke/prefixcat_full.json" >> "$MF"
    echo "${COMMON} --output_dir ${PP} --prefix_pool_dir ${PP} --prefix_exclude_shard 9 --label prefixcat_unlearn --out ${PP}/results/smoke/prefixcat_unlearn.json" >> "$MF"
    echo "${COMMON} --output_dir ${PD} --label merged_additive_mean --out ${PD}/results/smoke/merged_additive_mean.json" >> "$MF"
    echo "${COMMON} --output_dir ${PD} --label remerge_additive_mean --out ${PD}/results/smoke/remerge_additive_mean.json" >> "$MF"
    # LoRA incumbent anchor on the legacy k10 pool (same eval manifest tier)
    echo "${COMMON} --output_dir ${LEGACY_POOL} --label merged_additive_mean --out ${LEGACY_POOL}/results/smoke/merged_additive_mean.json" >> "$MF"
    echo "${COMMON} --output_dir ${LEGACY_POOL} --label remerge_additive_mean --out ${LEGACY_POOL}/results/smoke/remerge_additive_mean.json" >> "$MF"
    echo "$MF"
}

stage_eval() {  # $1 = optional dependency
    local MF N
    MF="$(build_manifest)"; N=$(wc -l < "$MF")
    local s; s="$(gpu_header pfx-bake-eval "0-$((N-1))%4" "${TOFU_SMOKE_TIME}" "${1:-}")"
    s+=$'\n'"LINE=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" ${MF})"
    s+=$'\n'"OUT=\$(echo \"\$LINE\" | grep -oP -- '--out \\K\\S+')"
    s+=$'\n'"if [ -f \"\$OUT\" ]; then echo \"exists: \$OUT — skip\"; exit 0; fi"
    s+=$'\n'"${PYTHON} eval_tofu.py \$LINE"
    submit "$s"
}

stage_collect() {
    ${PYTHON} collect_results.py --root "${SCRIPT_DIR}/checkpoints" --smoke
}

# DEP=<jobid> chains any stage behind an existing job (global 4-GPU cap: never let two
# %4 arrays be simultaneously runnable — chain instead, per repo CLAUDE.md).
case "$STAGE" in
    smoke)   stage_smoke "${DEP:-}" ;;
    train)   stage_train "${DEP:-}" ;;
    compose) stage_compose "${DEP:-}" ;;
    eval)    stage_eval "${DEP:-}" ;;
    collect) stage_collect ;;
    all)
        TRAIN_ID=$(stage_train "${DEP:-}")
        COMPOSE_ID=$(stage_compose "afterok:$TRAIN_ID")
        stage_eval "afterok:$COMPOSE_ID"
        ;;
    *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac
