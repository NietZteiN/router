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
  case "$(hostname -s 2>/dev/null || echo unknown)" in
    sprint*|login*)      TOFU_SITE=sprint ;;
    xe8545*|*a100*)      TOFU_SITE=cispa ;;
    *) TOFU_SITE=sprint ;;   # the historical default; keeps untouched drivers behaving as before
  esac
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
# Derived, not duplicated — the same three lines the repo-root cluster_env.sh carries. Drivers
# here reach SIBLING projects' stores through TOFU_CKPT_STORE (submit_router_family.sh:33 wants
# "${TOFU_CKPT_STORE}/ramole"); without these it expanded to empty and `mkdir -p /ramole` ran
# against the filesystem root.
export TOFU_CKPT_STORE="${TOFU_CKPT_STORE:-$(dirname "${TOFU_CKPT_ROOT}")}"
export TOFU_STORAGE_ROOT="${TOFU_STORAGE_ROOT:-$(dirname "${TOFU_CKPT_STORE}")}"
export TOFU_DATA_ROOT="${TOFU_DATA_ROOT:-${TOFU_STORAGE_ROOT}/data}"
export TOFU_PARTITION="${TOFU_PARTITION:?cluster_env.${TOFU_SITE}.sh must set TOFU_PARTITION}"
export TOFU_ACCOUNT="${TOFU_ACCOUNT:-}"                 # empty = do not emit --account
export TOFU_EXCLUDE="${TOFU_EXCLUDE:-}"                 # empty = do not emit --exclude
export TOFU_GPUS_PER_TASK="${TOFU_GPUS_PER_TASK:-1}"
export TOFU_CPUS_PER_TASK="${TOFU_CPUS_PER_TASK:-4}"
# Whether the site treats memory as a consumable resource. On CISPA the nodes report
# RealMemory=1 and the partition sets DefMemPerNode=UNLIMITED, so ANY --mem fails at submit with
# "Memory specification can not be satisfied" — it must be dropped, not lowered.
export TOFU_SUPPORTS_MEM="${TOFU_SUPPORTS_MEM:-1}"

# ── GPU concurrency ceiling — OPT-IN since 2026-08-06 ────────────────────────────────────────
# This used to hardcode a global ceiling of 4 concurrent GPUs, carried over from ~/CLAUDE.md §1
# on the sprint cluster. That was a courtesy rule for a shared lab machine, not a limit any
# scheduler imposes, and it silently overrode site files that knew better: CISPA's association
# allows gres/gpu=16 per user with MaxJobs=6, so the site's own cap of 6 was being cut to 4 for
# no reason. The real limits are enforced by SLURM whether or not this file agrees.
#
# The mechanism stays, unset by default: a site (or a run) that wants a hard ceiling sets
# TOFU_GPU_CAP_CEILING and gets the same clamp-and-report behaviour. cluster_env.sprint.sh sets
# it to 4, preserving that cluster's policy exactly.
if [ -n "${TOFU_GPU_CAP_CEILING:-}" ]; then
  _req_cap="${TOFU_ARRAY_CAP:-${TOFU_GPU_CAP_CEILING}}"
  if [ "${_req_cap}" -gt "${TOFU_GPU_CAP_CEILING}" ] 2>/dev/null; then
    echo "cluster_env: TOFU_ARRAY_CAP=${_req_cap} exceeds the TOFU_GPU_CAP_CEILING=${TOFU_GPU_CAP_CEILING}" \
         "you set — clamping to ${TOFU_GPU_CAP_CEILING}." >&2
    _req_cap="${TOFU_GPU_CAP_CEILING}"
  fi
  export TOFU_ARRAY_CAP="${_req_cap}"
  unset _req_cap
fi

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
