#!/bin/bash
# Experiment B — paraphrase robustness and cross-author leakage. SLURM driver.
# Thread: log/merge_mechanism/. Config: configs/expb_selectivity_7b.json
#
#   bash submit_expb.sh CONFIG plan      # (login, light) print every condition, write manifests
#   bash submit_expb.sh CONFIG merge     # CPU array: materialize the 16 aggregate merges
#   bash submit_expb.sh CONFIG refs      # GPU x1: per-(author,surface) KS refs from the oracle
#   bash submit_expb.sh CONFIG iso       # GPU array: the 20 iso_a{a} ceilings
#   bash submit_expb.sh CONFIG score     # GPU array: every served condition x 20 authors x 2 surfaces
#   bash submit_expb.sh CONFIG contrib   # GPU array: per-author signed contribution decomposition
#   bash submit_expb.sh CONFIG collect   # CPU: collect_expb.py -> reports/expb/
#
# STUB=1 prints every sbatch script without submitting. DEP=<jobid> chains with afterany.
#
# ORDER MATTERS: refs must land before score (a scoring run without its reference reports
# forget_quality = NaN and does not fail), and merge before score/contrib. `plan` prints the
# order and the dependency chain to paste.
#
# GPU CAP (~/CLAUDE.md 1): every GPU array here is %${TOFU_ARRAY_CAP}. On the sprint site that
# IS the global 4-GPU cap, so never leave two GPU arrays queued at once — chain them with DEP=.
# The merge, plan and collect stages are CPU-only and are exempt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"

# NB: no braces inside ${x:?...} — a '}' in the message closes the expansion early and gets
# appended to the value (which silently produced 'config.json}' paths in submit_expa.sh).
CONFIG="$(readlink -f "${1:?usage: submit_expb.sh CONFIG STAGE   STAGE = plan|merge|refs|iso|score|contrib|collect}")"
STAGE="${2:?usage: submit_expb.sh CONFIG STAGE   STAGE = plan|merge|refs|iso|score|contrib|collect}"
PYTHON="${PYTHON:-${TOFU_PYTHON}}"
ARRAY_CAP="${ARRAY_CAP:-${TOFU_ARRAY_CAP}}"

# Interpreter used ONLY to read the config while building sbatch text — never for the jobs.
# Kept separate from ${PYTHON} so `TOFU_SITE=cispa STUB=1 ...` can validate another site's job
# scripts from a machine that does not have that site's interpreter. That preview is how the
# "no --mem on CISPA" class of error gets caught before anything is submitted.
LOCAL_PY="${TOFU_LOCAL_PYTHON:-$(command -v python3 || command -v python)}"
[ -x "${LOCAL_PY}" ] || LOCAL_PY="${PYTHON}"

_cfg() { "${LOCAL_PY}" -c '
import json,os,sys
cfg=json.load(open(sys.argv[1]))
v=cfg
for k in sys.argv[2].split("."):
    v=v[k]
print(os.path.expandvars(v) if isinstance(v,str) else v)' "${CONFIG}" "$1"; }

OUT_DIR="$(_cfg out_dir)"
MODEL="$(_cfg model_name)"
SHARDS="$(_cfg shards_dir)"
REF_DIR="$(_cfg refs.dir)"
ORACLE="$(_cfg retain90_adapter)"
MERGES_DIR="${OUT_DIR}/merges"
RESULTS_DIR="${OUT_DIR}/results"
LOG_DIR="${OUT_DIR}/logs"
REPORTS="${SCRIPT_DIR}/reports/expb"

case "${OUT_DIR}" in
  *'${'*) echo "out_dir still contains an unexpanded variable: ${OUT_DIR}" >&2
          echo "  cluster_env.\${TOFU_SITE}.sh must export TOFU_CKPT_ROOT" >&2; exit 1 ;;
esac

MANIFEST_DIR="${OUT_DIR}"
if ! mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${MERGES_DIR}" "${REF_DIR}" "${REPORTS}" 2>/dev/null; then
  [ "${STUB:-0}" = "1" ] || { echo "cannot create ${LOG_DIR} / ${RESULTS_DIR}"; exit 1; }
  MANIFEST_DIR="$(mktemp -d)"
  echo "[stub] ${OUT_DIR} not present on this machine — preview only, manifests in ${MANIFEST_DIR}" >&2
fi

submit() {
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----"; printf '%s\n' "$1"
    echo "-----------------------------------------------"; LAST_JOB=""; return
  fi
  local dep=(); [ -n "${DEP:-}" ] && dep=(--dependency="afterany:${DEP}")
  LAST_JOB="$(printf '%s\n' "$1" | sbatch --parsable "${dep[@]}")"
  echo "submitted job ${LAST_JOB}"
}

# ── Condition enumeration ────────────────────────────────────────────────────────────────────
# ONE python pass writes both manifests, so the merge stage and the score stage can never
# disagree about which author set backs which label. Emitting them from two places is exactly
# how a drop_a{X} ends up scored against a merge that still contains X.
#   merge manifest:  label \t authors \t lam
#   score manifest:  label \t adapter \t arm \t condition \t target_author
write_manifests() {
  "${LOCAL_PY}" - "${CONFIG}" "${MANIFEST_DIR}/expb_merge_manifest.txt" \
                 "${MANIFEST_DIR}/expb_score_manifest.txt" \
                 "${MERGES_DIR}" "${SHARDS}" "${ORACLE}" <<'PY'
import json, os, sys
cfg_path, merge_mf, score_mf, merges_dir, shards, oracle = sys.argv[1:7]
cfg = json.load(open(cfg_path))

def rng(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))

authors  = rng(cfg["authors"])
targets  = cfg["targets"]
mean_lam = cfg["arms"]["mean20"]["lam"]
sum4_lam = cfg["arms"]["sum4"]["lam"]
per_tgt  = int(cfg["arms"]["sum4"]["companions_per_target"])

bad = [t for t in targets if t not in authors]
if bad:
    raise SystemExit(f"targets {bad} are not in the aggregate {cfg['authors']}")

# Partition the NON-target authors into one disjoint companion block per target. A single
# shared companion set would make every sum4_drop_a{X} the same author set, i.e. one artifact
# under five labels — five duplicate merges and one measurement pretending to be five.
pool = [a for a in authors if a not in targets]
need = per_tgt * len(targets)
if len(pool) < need:
    raise SystemExit(f"need {need} non-target companions ({per_tgt} x {len(targets)}) but the "
                     f"aggregate has only {len(pool)}: {pool}")
COMPANIONS = {t: pool[i * per_tgt:(i + 1) * per_tgt] for i, t in enumerate(targets)}

merges, scores = [], []
def add(label, auth, lam, arm, condition, target):
    merges.append((label, ",".join(map(str, auth)), lam))
    scores.append((label, os.path.join(merges_dir, label), arm, condition, target))

# mean20: fixed 1/20 weights. `full` is the pre-deletion reference; each drop_a{X} is the SAME
# lam over the 19 survivors, so it equals full minus (1/20)*Delta_X exactly.
add("expb_mean20_full", authors, mean_lam, "mean20", "full", "")
for t in targets:
    add(f"expb_mean20_drop_a{t}", [a for a in authors if a != t], mean_lam,
        "mean20", f"drop_a{t}", t)

# sum4: the literal Sigma rule at N=4, one DISJOINT group per target.
for t in targets:
    grp = sorted(COMPANIONS[t] + [t])
    add(f"expb_sum4_full_a{t}", grp, sum4_lam, "sum4", f"full_a{t}", t)
    add(f"expb_sum4_drop_a{t}", COMPANIONS[t], sum4_lam, "sum4", f"drop_a{t}", t)

# References that need no merge: the base model, the retain-only oracle, and the iso ceilings.
if cfg.get("base_model", True):
    scores.append(("expb_base", "BASE", "ref", "base", ""))
scores.append(("expb_retain90", oracle, "ref", "retain90", ""))
if cfg.get("iso", {}).get("enabled", True):
    for a in rng(cfg["iso"]["authors"]):
        scores.append((f"expb_iso_a{a}", os.path.join(shards, f"shard_{a}"), "iso", f"iso_a{a}", a))

with open(merge_mf, "w") as f:
    for r in merges:
        f.write("\t".join(map(str, r)) + "\n")
with open(score_mf, "w") as f:
    for r in scores:
        f.write("\t".join(map(str, r)) + "\n")
print(f"{len(merges)} merges, {len(scores)} score conditions "
      f"({len(targets)} targets, {len(authors)} authors in the aggregate)")
PY
}

MERGE_MF="${MANIFEST_DIR}/expb_merge_manifest.txt"
SCORE_MF="${MANIFEST_DIR}/expb_score_manifest.txt"

do_plan() {
  write_manifests
  echo
  echo "merge manifest  ${MERGE_MF}"
  echo "score manifest  ${SCORE_MF}"
  echo
  column -t -s $'\t' "${MERGE_MF}" | sed 's/^/  /'
  echo
  echo "Submit in this order (each GPU array is %${ARRAY_CAP}; never two queued at once):"
  echo "  J=\$(bash submit_expb.sh ${CONFIG} merge   | tail -1 | awk '{print \$3}')   # CPU"
  echo "  J=\$(DEP=\$J bash submit_expb.sh ${CONFIG} refs    | tail -1 | awk '{print \$3}')"
  echo "  J=\$(DEP=\$J bash submit_expb.sh ${CONFIG} iso     | tail -1 | awk '{print \$3}')"
  echo "  J=\$(DEP=\$J bash submit_expb.sh ${CONFIG} score   | tail -1 | awk '{print \$3}')"
  echo "  J=\$(DEP=\$J bash submit_expb.sh ${CONFIG} contrib | tail -1 | awk '{print \$3}')"
  echo "       DEP=\$J bash submit_expb.sh ${CONFIG} collect"
}

do_merge() {
  write_manifests >/dev/null
  local n; n="$(wc -l < "${MERGE_MF}")"
  [ "${n}" -gt 0 ] || { echo "no merges planned"; exit 1; }
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=expb-merge
#SBATCH --array=0-$((n - 1))%${ARRAY_CAP}
$(tofu_sbatch_resources 0 16 "${EXPB_MERGE_MEM:-96G}")
#SBATCH --time=${EXPB_MERGE_TIME:-04:00:00}
#SBATCH --output=${LOG_DIR}/merge_%A_%a.log
#SBATCH --error=${LOG_DIR}/merge_%A_%a.log
set -eo pipefail
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MERGE_MF}")
LABEL=\$(printf '%s' "\${LINE}" | cut -f1)
AUTHORS=\$(printf '%s' "\${LINE}" | cut -f2)
LAM=\$(printf '%s' "\${LINE}" | cut -f3)
if [ -f "${MERGES_DIR}/\${LABEL}/adapter_model.safetensors" ]; then
  echo "Skip existing \${LABEL}"; exit 0
fi
$(tofu_job_prologue)
export NMERGE_THREADS=16
date
# NO --svd_rank: the contrib stage's block decomposition is only valid on an UNCOMPRESSED
# merge (_compress_factored destroys per-author block identity). At N=20 the cat rank is 640,
# well inside the materialized-rank ceiling, so exact is affordable here.
${PYTHON} "${SCRIPT_DIR}/merge_subset.py" merge --config "${CONFIG}" \\
  --method additive_sum --authors "\${AUTHORS}" --lam "\${LAM}" --label "\${LABEL}"
date
EOF
  echo "merge array: ${n} tasks (%${ARRAY_CAP}), CPU only"
  submit "${S}"
}

do_refs() {
  # ONE GPU task, not an array: --build_refs walks every (author, surface) in a single model
  # load, and the references must be written by ONE oracle or the KS comparisons are not
  # mutually consistent.
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=expb-refs
$(tofu_sbatch_resources "${TOFU_GPUS_PER_TASK}" 4 48G)
#SBATCH --time=${EXPB_REFS_TIME:-${TOFU_SMOKE_TIME}}
#SBATCH --output=${LOG_DIR}/refs_%j.log
#SBATCH --error=${LOG_DIR}/refs_%j.log
set -eo pipefail
$(tofu_job_prologue)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [ ! -d "${ORACLE}" ]; then
  echo "FATAL: retain90 oracle missing at ${ORACLE}"; exit 1
fi
date
# The oracle keeps its LEGACY r8/alpha16/e3/lr2e-4 recipe on purpose — it is not the r32 pool
# recipe, and 'fixing' that would move every forget-quality number in the repo's history.
${PYTHON} "${SCRIPT_DIR}/measure_adapter_selectivity.py" \\
  --model_name "${MODEL}" --adapter "${ORACLE}" --arm ref --condition retain90 \\
  --authors "$(_cfg authors)" --surfaces "$(_cfg surfaces | tr -d "[]' " )" \\
  --build_refs --ref_dir "${REF_DIR}" \\
  --out_json "${RESULTS_DIR}/expb_refs.json" --out_csv "${RESULTS_DIR}/expb_refs.csv"
date
echo "references written to ${REF_DIR}:"
ls -la "${REF_DIR}" | head -20
EOF
  echo "refs: 1 GPU task (builds one KS reference per author x surface)"
  submit "${S}"
}

# iso / score / contrib all fan out over the score manifest; only the row filter differs.
_score_array() {
  local name="$1" filter="$2" time="$3"
  write_manifests >/dev/null
  local MF="${MANIFEST_DIR}/expb_${name}_rows.txt"
  awk -F'\t' "${filter}" "${SCORE_MF}" > "${MF}"
  local n; n="$(wc -l < "${MF}")"
  [ "${n}" -gt 0 ] || { echo "no ${name} rows (run merge first?)"; exit 1; }
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=expb-${name}
#SBATCH --array=0-$((n - 1))%${ARRAY_CAP}
$(tofu_sbatch_resources "${TOFU_GPUS_PER_TASK}" 4 48G)
#SBATCH --time=${time}
#SBATCH --output=${LOG_DIR}/${name}_%A_%a.log
#SBATCH --error=${LOG_DIR}/${name}_%A_%a.log
set -eo pipefail
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MF}")
LABEL=\$(printf '%s' "\${LINE}" | cut -f1)
ADAPTER=\$(printf '%s' "\${LINE}" | cut -f2)
ARM=\$(printf '%s' "\${LINE}" | cut -f3)
COND=\$(printf '%s' "\${LINE}" | cut -f4)
TGT=\$(printf '%s' "\${LINE}" | cut -f5)
OUT="${RESULTS_DIR}/\${LABEL}.json"
if [ -f "\${OUT}" ]; then echo "Skip existing \${OUT}"; exit 0; fi
# kill_invalid_depend is off cluster-wide, so a dependent array still runs after its parent
# fails. Assert the inputs in-task rather than trusting the dependency.
if [ "\${ADAPTER}" != "BASE" ] && [ ! -d "\${ADAPTER}" ]; then
  echo "FATAL: adapter missing: \${ADAPTER} (did the merge stage finish?)"; exit 1
fi
if [ ! -e "${REF_DIR}/retain_tr_a186_original.npy" ]; then
  echo "FATAL: KS references absent in ${REF_DIR} — run the refs stage first."; exit 1
fi
$(tofu_job_prologue)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
date
TGT_ARG=""
if [ -n "\${TGT}" ]; then TGT_ARG="--target_author \${TGT}"; fi
${PYTHON} "${SCRIPT_DIR}/measure_adapter_selectivity.py" \\
  --model_name "${MODEL}" --adapter "\${ADAPTER}" --arm "\${ARM}" --condition "\${COND}" \\
  \${TGT_ARG} --authors "$(_cfg authors)" --ref_dir "${REF_DIR}" \\
  --max_new_tokens $(_cfg max_new_tokens) --seed $(_cfg seed) \\
  --out_json "\${OUT}" --out_csv "${RESULTS_DIR}/\${LABEL}.csv"
date
EOF
  echo "${name} array: ${n} tasks (%${ARRAY_CAP})"
  submit "${S}"
}

do_iso()   { _score_array iso   '$3=="iso"' "${EXPB_ISO_TIME:-${TOFU_SMOKE_TIME}}"; }
do_score() { _score_array score '$3!="iso"' "${EXPB_SCORE_TIME:-${TOFU_EXTENDED_TIME}}"; }

do_contrib() {
  # Only UNCOMPRESSED merges: measure_expb_contrib refuses svd_rank, because _compress_factored
  # destroys the per-author block identity the decomposition reads.
  local MF="${MANIFEST_DIR}/expb_contrib_rows.txt"
  : > "${MF}"
  for d in "${MERGES_DIR}"/*/; do
    [ -f "${d}/merge_meta.json" ] || continue
    "${LOCAL_PY}" -c 'import json,sys; sys.exit(0 if not json.load(open(sys.argv[1])).get("svd_rank") else 1)' \
      "${d}/merge_meta.json" 2>/dev/null || continue
    printf '%s\t%s\n' "$(basename "${d%/}")" "${d%/}" >> "${MF}"
  done
  local n; n="$(wc -l < "${MF}")"
  [ "${n}" -gt 0 ] || { echo "no uncompressed merges in ${MERGES_DIR} (run merge first)"; exit 1; }
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=expb-contrib
#SBATCH --array=0-$((n - 1))%${ARRAY_CAP}
$(tofu_sbatch_resources "${TOFU_GPUS_PER_TASK}" 4 64G)
#SBATCH --time=${EXPB_CONTRIB_TIME:-${TOFU_SMOKE_TIME}}
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
date
${PYTHON} "${SCRIPT_DIR}/measure_expb_contrib.py" --model_name "${MODEL}" \\
  --adapter "\${ADAPTER}" --hidden served --out "\${OUT}"
date
EOF
  echo "contrib array: ${n} uncompressed merges (%${ARRAY_CAP})"
  submit "${S}"
}

do_collect() {
  "${PYTHON}" "${SCRIPT_DIR}/collect_expb.py" --config "${CONFIG}" \
    --results_dir "${RESULTS_DIR}" --contrib_dir "${REPORTS}" --out_prefix "${REPORTS}/expb"
}

case "${STAGE}" in
  plan)    do_plan ;;
  merge)   do_merge ;;
  refs)    do_refs ;;
  iso)     do_iso ;;
  score)   do_score ;;
  contrib) do_contrib ;;
  collect) do_collect ;;
  *) echo "usage: submit_expb.sh CONFIG {plan|merge|refs|iso|score|contrib|collect}"; exit 1 ;;
esac
