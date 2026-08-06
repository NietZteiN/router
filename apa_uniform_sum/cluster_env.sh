# Cluster settings — single source of truth, site-selected.
#
# Adopted from merge-tables-7b/tofu_sisa_lora/cluster_env.sh (the CISPA A100 port, 2026-07-24),
# generalized here so a new cluster is a FILE rather than a patch. Every driver should reach
# SLURM resources through `tofu_sbatch_resources` instead of hardcoding #SBATCH lines.
#
# Usage:  source "${SCRIPT_DIR}/slurm_nodes.sh"     # which sources this
#         TOFU_SITE=cispa bash submit_expa.sh ...   # or set the site explicitly
#
# Site selection: $TOFU_SITE, else auto-detected from the hostname, else `sprint`.
# Site files are cluster_env.<site>.sh and set ONLY variables — no logic.

_TOFU_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${TOFU_SITE:-}" ]; then
  _tofu_host="$(hostname -s 2>/dev/null || echo unknown)"
  case "${_tofu_host}" in
    sprint*|login*)           TOFU_SITE=sprint ;;
    xe8545*|*a100*|vr-*|vr[0-9]*) TOFU_SITE=cispa ;;
    *)
      # Fall back, but SAY SO. An unrecognised host silently resolving to `sprint` is the worst
      # failure this file can produce: on another cluster it points TOFU_PYTHON, HF_HOME and
      # TOFU_CKPT_ROOT at paths that do not exist, and emits a `--mem` line that fails at submit
      # — all without a word. The CISPA login nodes are `vr-*`, which is why they are matched
      # above; they are NOT compute nodes, so hostname alone never saw an `a100`.
      TOFU_SITE=sprint
      echo "cluster_env: host '${_tofu_host}' matches no site rule — defaulting to TOFU_SITE=sprint." >&2
      echo "cluster_env:   If that is wrong, export TOFU_SITE=<site> (available:" \
           "$(ls "${_TOFU_ENV_DIR}"/cluster_env.*.sh 2>/dev/null | sed 's/.*cluster_env\.//;s/\.sh//' | tr '\n' ' '))." >&2
      ;;
  esac
  unset _tofu_host
fi
export TOFU_SITE

_TOFU_SITE_FILE="${_TOFU_ENV_DIR}/cluster_env.${TOFU_SITE}.sh"
if [ ! -f "${_TOFU_SITE_FILE}" ]; then
  echo "cluster_env: no site file ${_TOFU_SITE_FILE} (set TOFU_SITE to one of: $(ls "${_TOFU_ENV_DIR}"/cluster_env.*.sh 2>/dev/null | sed 's/.*cluster_env\.//;s/\.sh//' | tr '\n' ' '))" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck source=/dev/null
source "${_TOFU_SITE_FILE}"

# ── Defaults common to every site (each may be overridden by the site file or the env) ───────
export TOFU_PYTHON="${TOFU_PYTHON:?cluster_env.${TOFU_SITE}.sh must set TOFU_PYTHON}"
export HF_HOME="${HF_HOME:?cluster_env.${TOFU_SITE}.sh must set HF_HOME}"
export TOFU_CKPT_ROOT="${TOFU_CKPT_ROOT:?cluster_env.${TOFU_SITE}.sh must set TOFU_CKPT_ROOT}"
export TOFU_PARTITION="${TOFU_PARTITION:?cluster_env.${TOFU_SITE}.sh must set TOFU_PARTITION}"
export TOFU_ACCOUNT="${TOFU_ACCOUNT:-}"                 # empty = do not emit --account
export TOFU_EXCLUDE="${TOFU_EXCLUDE:-}"                 # empty = do not emit --exclude
export TOFU_GPUS_PER_TASK="${TOFU_GPUS_PER_TASK:-1}"
export TOFU_CPUS_PER_TASK="${TOFU_CPUS_PER_TASK:-4}"
# Whether the site treats memory as a consumable resource. On CISPA the nodes report
# RealMemory=1 and the partition sets DefMemPerNode=UNLIMITED, so ANY --mem fails at submit with
# "Memory specification can not be satisfied" — it must be dropped, not lowered.
export TOFU_SUPPORTS_MEM="${TOFU_SUPPORTS_MEM:-1}"

# Emit the #SBATCH resource block for one task.
#   tofu_sbatch_resources [gpus|0|-] [cpus] [mem]
# gpus 0 or '-' => a CPU-only task (no --gres line). mem is silently dropped where unsupported.
tofu_sbatch_resources() {
  local gpus="${1:-${TOFU_GPUS_PER_TASK}}"
  local cpus="${2:-${TOFU_CPUS_PER_TASK}}"
  local mem="${3:-}"
  echo "#SBATCH --partition=${TOFU_PARTITION}"
  [ -n "${TOFU_ACCOUNT}" ] && echo "#SBATCH --account=${TOFU_ACCOUNT}"
  [ -n "${TOFU_EXCLUDE}" ] && echo "#SBATCH --exclude=${TOFU_EXCLUDE}"
  if [ -n "${gpus}" ] && [ "${gpus}" != "0" ] && [ "${gpus}" != "-" ]; then
    echo "#SBATCH --gres=gpu:${gpus}"
  fi
  echo "#SBATCH --cpus-per-task=${cpus}"
  if [ -n "${mem}" ] && [ "${TOFU_SUPPORTS_MEM}" = "1" ]; then
    echo "#SBATCH --mem=${mem}"
  fi
  return 0
}

# Runtime env block for inside a job script. The HF token is read from disk, never committed.
tofu_job_prologue() {
  cat <<PROLOGUE
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-${TOFU_ALLOC_CONF:-}}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
# Per-job metrics cache: parallel array tasks otherwise clobber the same .arrow file
# (see eval_tofu.py:39-45).
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "\${TOFU_METRICS_CACHE}"
echo "=== site ${TOFU_SITE} node \$(hostname) job \${SLURM_JOB_ID} task \${SLURM_ARRAY_TASK_ID:-0} \$(date) ==="
PROLOGUE
}
