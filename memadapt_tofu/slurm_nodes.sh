# SLURM environment for memadapt_tofu — source from submit scripts.
# Root CLAUDE.md §1: sprint1-3 only, GLOBAL cap of 4 concurrent GPUs across
# ALL of our queued+running jobs. Arrays here throttle to %2 so co-queued
# totals stay <= 4 even while other project jobs are in the queue.
# NEVER raise MEMADAPT_CAP above 4.


# Repo root — this tree is FLAT, so sibling projects live beside this one.

# ── Site layer (added on export) ────────────────────────────────────────────────
# WAS: `sprint4` and a literal cap of 4, both hardcoded. Those are sprint-cluster facts,
# not repo facts, so they now come from the repo-root cluster_env.<site>.sh — including the
# concurrency cap, which is per-site (sprint 4, cispa 6). An optional hard ceiling over it is
# available as TOFU_GPU_CAP_CEILING; it is unset unless a site asks for it (2026-08-06).
_SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../cluster_env.sh
source "${_SITE_DIR}/cluster_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MEMADAPT_EXCLUDE="${MEMADAPT_EXCLUDE:-${TOFU_EXCLUDE:-}}"
export MEMADAPT_CAP="${MEMADAPT_CAP:-${TOFU_ARRAY_CAP:-4}}"                # global cap (do not raise)
export MEMADAPT_THROTTLE="${MEMADAPT_THROTTLE:-2}"      # per-array %N throttle
export MEMADAPT_MEM="${MEMADAPT_MEM:-48G}"
export MEMADAPT_CPUS="${MEMADAPT_CPUS:-8}"

export MEMADAPT_SMOKE_TIME="${MEMADAPT_SMOKE_TIME:-00:55:00}"
export MEMADAPT_ASSIGN_TIME="${MEMADAPT_ASSIGN_TIME:-01:00:00}"
export MEMADAPT_TRAIN_TIME="${MEMADAPT_TRAIN_TIME:-03:00:00}"
export MEMADAPT_EVAL_TIME="${MEMADAPT_EVAL_TIME:-03:00:00}"

export PYTHON="${PYTHON:-python3}"
# Eval runs inside open-unlearning's pinned env (created in stage S0).
export OU_PYTHON="${OU_PYTHON:-python3}"
export OU_DIR="${OU_DIR:-${REPO_ROOT}/open-unlearning}"

export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
export MEMADAPT_ROOT="${MEMADAPT_ROOT:-${TOFU_CKPT_STORE}/memadapt_tofu}"
export EVAL_REFS="${EVAL_REFS:-${TOFU_STORAGE_ROOT}/memadapt/eval_refs}"
# datasets 3.0.1 (OU env) cannot read the datasets-4.x arrow cache under
# HF_HOME/datasets — OU-env jobs use this pre-built isolated cache instead
# (populated online on the login node, 2026-07-15).
export OU_DATASETS_CACHE="${OU_DATASETS_CACHE:-${HF_HOME}/datasets_ou301}"
