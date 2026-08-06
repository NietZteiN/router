#!/usr/bin/env bash
# SLURM submission driver for sepmlp_tofu (mirrors memadapt_tofu/submit_memadapt.sh).
#
# Usage:
#   ./submit_sepmlp.sh smoke [config]     # P1 GPU smoke (full-size bank, 2 authors, 5 steps)
#   ./submit_sepmlp.sh pilot              # P2 spec-recipe LR sweep: array 0-2%1, train + selectivity probe
#   ./submit_sepmlp.sh train [config]     # P3 K=200 train (config must carry the pilot winner)
#   ./submit_sepmlp.sh probe200 [config] [run_dir]   # G3 selectivity+recall re-gate at K=200
#   ./submit_sepmlp.sh leakprobe [config] [run_dir]  # Part 3.1 per-query leak probe (ref + forget10 drop)
#   ./submit_sepmlp.sh eval               # P4 OU evals: sepmlp_ft / sepmlp_unlearned / sepmlp_dropall
#   ./submit_sepmlp.sh relearn            # P5 relearn battery: 24-task array (sepmlp/memadapt/hf oracle)
#
#   STUB=1 ./submit_sepmlp.sh ...   -> print the sbatch script, do not submit
#   DEP=afterany:443125:443127 ...  -> dependency (chain behind the current queue tails)
#
# Before EVERY submission this script prints the queue; confirm
# (max concurrent GPUs queued) + (this submission) <= 4 (root CLAUDE.md §1).
# Heredoc convention: each python invocation stays on ONE line.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/slurm_nodes.sh"

LOG_DIR="${SEPMLP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# The P3/P4/P5 checkpoint; must equal output_dir in configs/sepmlp_1b_k200.json.
K200_CONFIG="${HERE}/configs/sepmlp_1b_k200.json"
K200_RUN="${SEPMLP_ROOT}/sepmlp_1b_k200_s42"
MEMADAPT_RUN="${TOFU_CKPT_STORE}/memadapt_tofu/memadapt_1b_l8_s42"
RETAIN90_HF="open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"

queue_check() {
  echo "=== current queue (global GPU cap is 4; verify before confirming) ==="
  squeue -u jack -o "%.10i %.20j %.10T %.10b %F" || true
  echo "======================================================================"
}

submit() {
  if [[ "${STUB:-0}" == "1" ]]; then
    echo "----- STUB (not submitted) -----"; cat; echo "-------------------------------"
  else
    sbatch ${DEP:+--dependency=${DEP}} /dev/stdin
  fi
}

# emit_job <name> <time> <output-pattern> <extra-directives (may be empty)> <body...>
emit_job() {
  local name="$1" time="$2" output="$3" extra="$4"; shift 4
  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=${name}"
    echo "#SBATCH --partition=all"
    echo "#SBATCH --exclude=${SEPMLP_EXCLUDE}"
    echo "#SBATCH --gres=gpu:1"
    echo "#SBATCH --mem=${SEPMLP_MEM}"
    echo "#SBATCH --cpus-per-task=${SEPMLP_CPUS}"
    echo "#SBATCH --time=${time}"
    echo "#SBATCH --output=${LOG_DIR}/${output}"
    [[ -n "${extra}" ]] && echo "${extra}"
    echo 'export HF_HOME='"${HF_HOME}"
    echo 'export HF_HUB_OFFLINE=1'
    echo 'export HF_DATASETS_OFFLINE=1'
    echo 'export TOKENIZERS_PARALLELISM=false'
    echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
    echo 'nvidia-smi --query-gpu=name,memory.total --format=csv'
    printf '%s\n' "$@"
  } | submit
}

case "${1:-}" in
  smoke)
    CONFIG="${2:-${HERE}/configs/smoke.json}"
    queue_check
    emit_job sepmlp-smoke "${SEPMLP_SMOKE_TIME}" "smoke_%j.log" "" \
      "cd ${HERE} && ${PYTHON} train_sepmlp.py --config ${CONFIG} --smoke"
    ;;
  pilot)
    # P2 (spec recipe, approved plan Part 2.6 lean revision): 3-arm LR sweep
    # at K=20 — the exact user-spec recipe, LR is the one untuned knob. Array
    # 0-2%1: one task at a time, a single lane under the global 4-GPU cap.
    # Each task trains (skipped if its sepmlp.pt already exists — safe
    # requeue) then runs the selectivity + recall probe that gate G2 reads.
    # RUN must equal the output_dir inside configs/pilot_relu_lr*.json.
    # (The 9-arm SwiGLU lambda grid pilot_0-8 is retired, not run.)
    queue_check
    emit_job sepmlp-pilot "${SEPMLP_PILOT_TIME}" "pilot_%A_%a.log" \
      "#SBATCH --array=0-2%1" \
      'ARMS=(pilot_relu_lr3e-4 pilot_relu_lr1e-3 pilot_relu_lr3e-3)' \
      'A=${ARMS[$SLURM_ARRAY_TASK_ID]}' \
      "C=${HERE}/configs/\${A}.json" \
      "RUN=${SEPMLP_ROOT}/pilot/\${A}_s42" \
      'if [[ -f ${RUN}/sepmlp.pt ]]; then echo "[skip] train: ${RUN}/sepmlp.pt exists"; else cd '"${HERE}"' && '"${PYTHON}"' train_sepmlp.py --config ${C}; fi' \
      "cd ${HERE} && ${PYTHON} measure_selectivity.py --config \${C} --checkpoint \${RUN} --recall_probe --out \${RUN}/selectivity_pilot.json"
    ;;
  pilot1)
    # G2 ADJUDICATE path: ONE named bridging pilot arm (same body as a pilot
    # array task — train unless sepmlp.pt exists, then the G2 probe). Arm name
    # must match configs/<arm>.json and its output_dir <arm>_s42.
    ARM="${2:?arm name, e.g. pilot_relu_lr5e-4}"
    queue_check
    emit_job sepmlp-pilot1 "${SEPMLP_PILOT_TIME}" "pilot1_%j.log" "" \
      "C=${HERE}/configs/${ARM}.json" \
      "RUN=${SEPMLP_ROOT}/pilot/${ARM}_s42" \
      'if [[ -f ${RUN}/sepmlp.pt ]]; then echo "[skip] train: ${RUN}/sepmlp.pt exists"; else cd '"${HERE}"' && '"${PYTHON}"' train_sepmlp.py --config ${C}; fi' \
      "cd ${HERE} && ${PYTHON} measure_selectivity.py --config \${C} --checkpoint \${RUN} --recall_probe --out \${RUN}/selectivity_pilot.json"
    ;;
  train)
    CONFIG="${2:-${K200_CONFIG}}"
    queue_check
    emit_job sepmlp-train "${SEPMLP_TRAIN_TIME}" "train_%j.log" "" \
      "cd ${HERE} && ${PYTHON} train_sepmlp.py --config ${CONFIG}"
    ;;
  probe200)
    # G3 re-gate: selectivity >=5 and >=0.7x pilot, all-active vs own-only
    # own-prob gap <=0.05 — read BEFORE any P4 eval spend.
    CONFIG="${2:-${K200_CONFIG}}"
    RUN_DIR="${3:-${K200_RUN}}"
    queue_check
    emit_job sepmlp-probe200 "${SEPMLP_PROBE_TIME}" "probe200_%j.log" "" \
      "cd ${HERE} && ${PYTHON} measure_selectivity.py --config ${CONFIG} --checkpoint ${RUN_DIR} --recall_probe --out ${RUN_DIR}/selectivity_k200.json"
    ;;
  recall)
    # Paper §5.1 PRIMARY metric: per-source ROUGE-L GENERATION recall + tail
    # (#authors<0.95) + named/name-free split + held-out recall over holdout10.
    # Distinct from probe200, which computes answer-PROBABILITY. Heavier (greedy
    # gen over K*20 + 20*20 rows) → EVAL_TIME budget.
    #   recall [config] [run_dir] [extra]   extra passed through verbatim, e.g.
    #   "--authors 0,1 --heldout_n 0" for a quick GPU smoke, or a droplist arg.
    CONFIG="${2:-${K200_CONFIG}}"
    RUN_DIR="${3:-${K200_RUN}}"
    EXTRA="${4:-}"
    queue_check
    emit_job sepmlp-recall "${SEPMLP_EVAL_TIME}" "recall_%j.log" "" \
      "cd ${HERE} && ${PYTHON} measure_recall.py --config ${CONFIG} --checkpoint ${RUN_DIR} --out ${RUN_DIR}/recall.json ${EXTRA}"
    ;;
  leakprobe)
    # Part 3.1: the per-query leak probe feeding the unified all-router leak
    # table. ONE 1-GPU job, two runs: reference (no droplist) then forget10
    # droplist — the analyzer needs both npz paths (retain-collateral is
    # top_surv_author drift between them). The droplist is built inline
    # (CPU, O(1)) if missing. Outputs are NEW files only:
    # leak_reference.json(.leak.npz) + leak_forget10.json(.leak.npz).
    CONFIG="${2:-${K200_CONFIG}}"
    RUN_DIR="${3:-${K200_RUN}}"
    queue_check
    emit_job sepmlp-leakprobe "${SEPMLP_PROBE_TIME}" "leakprobe_%j.log" "" \
      "DROP=${RUN_DIR}/droplists/forget10.json" \
      'if [[ ! -f ${DROP} ]]; then cd '"${HERE}"' && '"${PYTHON}"' build_droplist.py --config '"${CONFIG}"' --checkpoint '"${RUN_DIR}"' --tag forget10; fi' \
      "cd ${HERE} && ${PYTHON} measure_selectivity.py --config ${CONFIG} --checkpoint ${RUN_DIR} --probe forget_leak --out ${RUN_DIR}/leak_reference.json" \
      "cd ${HERE} && ${PYTHON} measure_selectivity.py --config ${CONFIG} --checkpoint ${RUN_DIR} --probe forget_leak --droplist \${DROP} --out ${RUN_DIR}/leak_forget10.json"
    ;;
  eval)
    # P4: three OU evals of a K=200 checkpoint. Droplists are built inline
    # (CPU, O(1)) when missing — BEFORE the OU_DATASETS_CACHE export, because
    # build_droplist runs in test-env (datasets 4.x) and must not be pointed at
    # the datasets-3.0.1 cache; the ordering of those body lines is load-bearing.
    # NOTE: the experiment file (@package _global_) merges AFTER the model
    # group and would silently swap the base to the TOFU-finetuned checkpoint;
    # the explicit pretrained_model_name_or_path override below is load-bearing.
    # tofu_grimes has overwrite:false — labels/dirs are fresh per checkpoint,
    # hence the LABEL PREFIX arg: eval [config] [run_dir] [prefix].
    CONFIG="${2:-${K200_CONFIG}}"
    RUN_DIR="${3:-${K200_RUN}}"
    PREFIX="${4:-sepmlp}"
    queue_check
    emit_job sepmlp-eval "${SEPMLP_EVAL_TIME}" "eval_%A_%a.log" \
      "#SBATCH --array=0-2%${SEPMLP_THROTTLE}" \
      "LABELS=(${PREFIX}_ft ${PREFIX}_unlearned ${PREFIX}_dropall)" \
      'TAGS=(null forget10 all200)' \
      'L=${LABELS[$SLURM_ARRAY_TASK_ID]}' \
      'T=${TAGS[$SLURM_ARRAY_TASK_ID]}' \
      "RUN=${RUN_DIR}" \
      'DROP=null' \
      'if [[ ${T} != null ]]; then DROP=${RUN}/droplists/${T}.json; fi' \
      'if [[ ${T} == forget10 && ! -f ${DROP} ]]; then cd '"${HERE}"' && '"${PYTHON}"' build_droplist.py --config '"${CONFIG}"' --checkpoint ${RUN} --tag forget10; fi' \
      'if [[ ${T} == all200 && ! -f ${DROP} ]]; then cd '"${HERE}"' && '"${PYTHON}"' build_droplist.py --config '"${CONFIG}"' --checkpoint ${RUN} --tag all200 --authors $(seq -s, 0 199); fi' \
      "export HF_DATASETS_CACHE=${OU_DATASETS_CACHE}" \
      "cd ${OU_DIR} && ${OU_PYTHON} src/eval.py experiment=eval/tofu/default.yaml eval=tofu_grimes model=SepMlp-Llama-3.2-1B model.model_args.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct model.model_args.sepmlp_checkpoint=\${RUN} model.model_args.droplist=\${DROP} eval.tofu.retain_logs_path=${EVAL_REFS}/retain90_eval.json task_name=\${L} paths.output_dir=${SEPMLP_ROOT}/evals/\${L}"
    ;;
  relearn)
    # P5: relearn battery, H4. 24 tasks: {sepmlp_unlearned, memadapt_unlearned}
    # x (5 forget-targets + 5 holdout10 controls) + retrain-oracle x (2+2).
    # Targets are forget10 author ids; controls are INDICES into the holdout10
    # author list (holdout10 is never trained — relearn control + MIA
    # nonmembers). relearn.py runs in test-env; LoRA weights are discarded.
    # Flag names here define the relearn.py CLI contract — keep in sync.
    # relearn [config] [run_dir]: the sepmlp serve rows use this checkpoint
    # (defaults preserved for the original K200_RUN path).
    CONFIG="${2:-${K200_CONFIG}}"
    RUN_DIR="${3:-${K200_RUN}}"
    K200_CONFIG="${CONFIG}"
    K200_RUN="${RUN_DIR}"
    queue_check
    emit_job sepmlp-relearn "${SEPMLP_RELEARN_TIME}" "relearn_%A_%a.log" \
      "#SBATCH --array=0-23%${SEPMLP_THROTTLE}" \
      'SERVES=(sepmlp sepmlp sepmlp sepmlp sepmlp sepmlp sepmlp sepmlp sepmlp sepmlp memadapt memadapt memadapt memadapt memadapt memadapt memadapt memadapt memadapt memadapt hf hf hf hf)' \
      'ROLES=(target target target target target control control control control control target target target target target control control control control control target target control control)' \
      'IDS=(180 185 190 195 199 0 5 10 15 19 180 185 190 195 199 0 5 10 15 19 180 190 0 10)' \
      'S=${SERVES[$SLURM_ARRAY_TASK_ID]}' \
      'R=${ROLES[$SLURM_ARRAY_TASK_ID]}' \
      'I=${IDS[$SLURM_ARRAY_TASK_ID]}' \
      "if [[ \${S} == sepmlp && ! -f ${K200_RUN}/droplists/forget10.json ]]; then cd ${HERE} && ${PYTHON} build_droplist.py --config ${K200_CONFIG} --checkpoint ${K200_RUN} --tag forget10; fi" \
      'UNL=""' \
      "if [[ \${S} == sepmlp ]]; then CKPT=${K200_RUN}; UNL=\"--droplist ${K200_RUN}/droplists/forget10.json\"; fi" \
      "if [[ \${S} == memadapt ]]; then CKPT=${MEMADAPT_RUN}; UNL=\"--blocklist ${MEMADAPT_RUN}/blocklists/forget10.json\"; fi" \
      "if [[ \${S} == hf ]]; then CKPT=${RETAIN90_HF}; fi" \
      'if [[ ${R} == target ]]; then SEL="--author ${I}"; else SEL="--holdout_index ${I}"; fi' \
      "cd ${HERE} && ${PYTHON} relearn.py --serve \${S} --checkpoint \${CKPT} \${UNL} \${SEL} --out_dir ${SEPMLP_ROOT}/relearn/\${S}_\${R}_\${I} --seed 42"
    ;;
  *)
    echo "usage: $0 {smoke|pilot|pilot1|train|probe200|recall|leakprobe|eval|relearn} [args]"; exit 1
    ;;
esac
