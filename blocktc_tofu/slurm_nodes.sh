# SLURM environment for blocktc_tofu — source from submit scripts.
# Root CLAUDE.md §1: sprint1-3 only, GLOBAL cap of 4 concurrent GPUs across
# ALL of our queued+running jobs. Arrays here throttle to %2 so co-queued
# totals stay <= 4 even while other project jobs are in the queue.
# NEVER raise BLOCKTC_CAP above 4.


# Repo root — this tree is FLAT, so sibling projects live beside this one.

# ── Site layer (added on export) ────────────────────────────────────────────────
# WAS: `sprint4` and a literal cap of 4, both hardcoded. Those are sprint-cluster facts,
# not repo facts, so they now come from the repo-root cluster_env.<site>.sh. The GLOBAL
# 4-GPU ceiling (CLAUDE.md §1) is still enforced — cluster_env.sh clamps it centrally.
_SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../cluster_env.sh
source "${_SITE_DIR}/cluster_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export BLOCKTC_EXCLUDE="${BLOCKTC_EXCLUDE:-${TOFU_EXCLUDE:-}}"
export BLOCKTC_CAP="${BLOCKTC_CAP:-${TOFU_ARRAY_CAP:-4}}"                # global cap (do not raise)
export BLOCKTC_THROTTLE="${BLOCKTC_THROTTLE:-2}"      # per-array %N throttle
export BLOCKTC_MEM="${BLOCKTC_MEM:-48G}"
export BLOCKTC_CPUS="${BLOCKTC_CPUS:-8}"

export BLOCKTC_SMOKE_TIME="${BLOCKTC_SMOKE_TIME:-00:40:00}"
export BLOCKTC_PHASE0_TIME="${BLOCKTC_PHASE0_TIME:-01:30:00}"
export BLOCKTC_PILOT_TIME="${BLOCKTC_PILOT_TIME:-01:30:00}"
export BLOCKTC_TRAIN_TIME="${BLOCKTC_TRAIN_TIME:-04:00:00}"
export BLOCKTC_PROBE_TIME="${BLOCKTC_PROBE_TIME:-01:00:00}"
export BLOCKTC_EVAL_TIME="${BLOCKTC_EVAL_TIME:-03:00:00}"

export PYTHON="${PYTHON:-python3}"
# Eval runs inside open-unlearning's pinned env (memadapt stage S0).
export OU_PYTHON="${OU_PYTHON:-python3}"
export OU_DIR="${OU_DIR:-${REPO_ROOT}/open-unlearning}"

export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
export BLOCKTC_ROOT="${BLOCKTC_ROOT:-${TOFU_CKPT_STORE}/blocktc_tofu}"
export EVAL_REFS="${EVAL_REFS:-${TOFU_STORAGE_ROOT}/memadapt/eval_refs}"
# datasets 3.0.1 (OU env) cannot read the datasets-4.x arrow cache under
# HF_HOME/datasets — OU-env jobs use this pre-built isolated cache instead
# (populated online on the login node, 2026-07-15; shared with memadapt/sepmlp).
export OU_DATASETS_CACHE="${OU_DATASETS_CACHE:-${HF_HOME}/datasets_ou301}"
