#!/bin/bash
# APA uniform-summation study — SLURM driver for the stages submit_nmerge.sh does not cover.
# Thread: log/merge_mechanism/ (2026-07-28). Plan: ~/.claude/plans/context-musr-uses-parallel-*.md
#
#   bash submit_expa.sh CONFIG mmlu      # GPU array: MMLU per merged/anchor condition (Exp A/C)
#   bash submit_expa.sh CONFIG contrib   # GPU array: per-author signed contribution decomposition
#   bash submit_expa.sh CONFIG gates     # CPU: the pre-submission gate battery
#
# STUB=1 prints every sbatch script without submitting. DEP=<jobid> chains with afterany.
#
# GPU CAP: every GPU array here is %${TOFU_ARRAY_CAP} (4). Because that IS the global cap
# (~/CLAUDE.md §1), never leave two GPU arrays queued at once — chain them with DEP=. The
# `norms` stage in submit_nmerge.sh and the `gates` stage here are CPU-only and are exempt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

# NB: no braces inside ${x:?...} — a '}' in the message closes the expansion early and gets
# appended to the value (which silently produced 'config.json}' paths).
CONFIG="${1:?usage: submit_expa.sh CONFIG STAGE   where STAGE = mmlu | contrib | gates}"
STAGE="${2:?usage: submit_expa.sh CONFIG STAGE   where STAGE = mmlu | contrib | gates}"
PYTHON="${PYTHON:-${TOFU_PYTHON}}"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"
# Interpreter used ONLY to read configs while building the sbatch text — never for the jobs.
# Kept separate from ${PYTHON} so `TOFU_SITE=cispa STUB=1 ...` can validate another site's job
# scripts from a machine that does not have that site's interpreter. That preview is how the
# "no --mem on CISPA" class of error gets caught before anything is submitted.
LOCAL_PY="${TOFU_LOCAL_PYTHON:-$(command -v python3 || command -v python)}"
[ -x "${LOCAL_PY}" ] || LOCAL_PY="${PYTHON}"
_cfg_get() { "${LOCAL_PY}" -c "import json,os,sys;v=json.load(open(sys.argv[1]))[sys.argv[2]];print(os.path.expandvars(v))" "${CONFIG}" "$1"; }

OUT_DIR="$(_cfg_get out_dir)"
MODEL="$(_cfg_get model_name)"
LOG_DIR="${OUT_DIR}/logs"
RESULTS_DIR="${OUT_DIR}/results/smoke"
MERGES_DIR="${OUT_DIR}/merges"
REPORTS="${SCRIPT_DIR}/reports/expA"
# Under STUB the target filesystem may not exist (that is the point of previewing another site's
# scripts from here), so a failed mkdir must not abort the preview.
MANIFEST_DIR="${OUT_DIR}"
if ! mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${REPORTS}" 2>/dev/null; then
  [ "${STUB:-0}" = "1" ] || { echo "cannot create ${LOG_DIR} / ${RESULTS_DIR}"; exit 1; }
  # Manifests would land under the absent out_dir too; park them in a temp dir so the preview
  # still renders the real sbatch text (the job itself reads them from OUT_DIR at run time).
  MANIFEST_DIR="$(mktemp -d)"
  echo "[stub] ${OUT_DIR} not present on this machine — preview only, manifests in ${MANIFEST_DIR}" >&2
fi

submit() {
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----"; printf '%s\n' "$1"; return
  fi
  local dep=(); [ -n "${DEP:-}" ] && dep=(--dependency="afterany:${DEP}")
  LAST_JOB="$(printf '%s' "$1" | sbatch --parsable "${dep[@]}")"
  echo "submitted job ${LAST_JOB}"
}

# Every servable condition: the materialized merges + the three anchors. One line per task.
build_manifest() {
  local mf="$1"
  : > "${mf}"
  if [ -d "${MERGES_DIR}" ]; then
    for d in "${MERGES_DIR}"/*/; do
      [ -f "${d}/adapter_model.safetensors" ] || continue
      printf '%s\t%s\n' "$(basename "${d%/}")" "${d%/}" >> "${mf}"
    done
  fi
  ${LOCAL_PY} - "$CONFIG" >> "${mf}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1])); a = cfg.get("anchors", {})
if a.get("base_model"):        print("base_model\tBASE")
if a.get("ft_adapter"):        print(f"ft_r32\t{a['ft_adapter']}")
if a.get("retain90_adapter"):  print(f"retain90_oracle\t{a['retain90_adapter']}")
PY
}

do_mmlu() {
  local MF="${MANIFEST_DIR}/mmlu_manifest.txt"
  build_manifest "${MF}"
  local n; n="$(wc -l < "${MF}")"
  [ "${n}" -gt 0 ] || { echo "no conditions in ${MF} (run the merge stage first)"; exit 1; }
  local NI SEED
  NI="$(${LOCAL_PY} -c "import json,sys;print(json.load(open(sys.argv[1])).get('mmlu',{}).get('n_items',2000))" "${CONFIG}")"
  SEED="$(${LOCAL_PY} -c "import json,sys;print(json.load(open(sys.argv[1])).get('mmlu',{}).get('seed',42))" "${CONFIG}")"
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=expa-mmlu
#SBATCH --array=0-$((n - 1))%${ARRAY_CAP}
$(tofu_sbatch_resources "${TOFU_GPUS_PER_TASK}" 4 48G)
#SBATCH --time=${EXPA_MMLU_TIME:-${TOFU_SMOKE_TIME}}
#SBATCH --output=${LOG_DIR}/mmlu_%A_%a.log
#SBATCH --error=${LOG_DIR}/mmlu_%A_%a.log
set -eo pipefail
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MF}")
LABEL=\$(printf '%s' "\${LINE}" | cut -f1)
ADAPTER=\$(printf '%s' "\${LINE}" | cut -f2)
OUT="${RESULTS_DIR}/\${LABEL}.mmlu.json"
if [ -f "\${OUT}" ]; then echo "Skip existing \${OUT}"; exit 0; fi
$(tofu_job_prologue)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv || true
date
${PYTHON} "${SCRIPT_DIR}/eval_mmlu.py" --model_name "${MODEL}" --adapter "\${ADAPTER}" \\
  --label "\${LABEL}" --n_items ${NI} --seed ${SEED} --out "\${OUT}"
date
EOF
  echo "mmlu array: ${n} tasks (%${ARRAY_CAP}), n_items=${NI}"
  submit "${S}"
}

do_contrib() {
  # Only UNCOMPRESSED merges — measure_expb_contrib refuses svd_rank (block identity is gone).
  local MF="${MANIFEST_DIR}/contrib_manifest.txt"
  : > "${MF}"
  for d in "${MERGES_DIR}"/*/; do
    [ -f "${d}/merge_meta.json" ] || continue
    ${LOCAL_PY} -c "import json,sys; sys.exit(0 if not json.load(open(sys.argv[1])).get('svd_rank') else 1)" \
      "${d}/merge_meta.json" 2>/dev/null || continue
    printf '%s\t%s\n' "$(basename "${d%/}")" "${d%/}" >> "${MF}"
  done
  local n; n="$(wc -l < "${MF}")"
  [ "${n}" -gt 0 ] || { echo "no uncompressed merges in ${MERGES_DIR}"; exit 1; }
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=expa-contrib
#SBATCH --array=0-$((n - 1))%${ARRAY_CAP}
$(tofu_sbatch_resources "${TOFU_GPUS_PER_TASK}" 4 64G)
#SBATCH --time=${EXPA_CONTRIB_TIME:-${TOFU_SMOKE_TIME}}
#SBATCH --output=${LOG_DIR}/contrib_%A_%a.log
#SBATCH --error=${LOG_DIR}/contrib_%A_%a.log
set -eo pipefail
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MF}")
LABEL=\$(printf '%s' "\${LINE}" | cut -f1)
ADAPTER=\$(printf '%s' "\${LINE}" | cut -f2)
OUT="${REPORTS}/contrib_\${LABEL}.json"
if [ -f "\${OUT}" ]; then echo "Skip existing \${OUT}"; exit 0; fi
$(tofu_job_prologue)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv || true
date
${PYTHON} "${SCRIPT_DIR}/measure_expb_contrib.py" --model_name "${MODEL}" \\
  --adapter "\${ADAPTER}" --hidden served --out "\${OUT}"
date
EOF
  echo "contrib array: ${n} uncompressed merges (%${ARRAY_CAP})"
  submit "${S}"
}

do_gates() {
  echo "== CPU gate battery (must all pass before any GPU submission) =="
  for t in test_cluster_env.py test_expa.py test_merge_subset.py test_ou_equivalence.py \
           test_eval_rows.py; do
    printf '%-26s ' "${t}"
    if "${PYTHON}" "${SCRIPT_DIR}/${t}" >/dev/null 2>&1; then echo PASS; else echo FAIL; exit 1; fi
  done
  echo "all gates green"
}

case "${STAGE}" in
  mmlu)    do_mmlu ;;
  contrib) do_contrib ;;
  gates)   do_gates ;;
  *) echo "usage: submit_expa.sh CONFIG {mmlu|contrib|gates}"; exit 1 ;;
esac
