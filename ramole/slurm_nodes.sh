# SLURM node policy for ramole: sprint1, sprint2, sprint3 only (never sprint4).
# --exclude=sprint4 so each job gets 1 GPU on an allowed node (matches legonet_lora/tofu).

# ── Site layer (added on export) ────────────────────────────────────────────────
# WAS: `sprint4` and a literal cap of 4, both hardcoded. Those are sprint-cluster facts,
# not repo facts, so they now come from the repo-root cluster_env.<site>.sh. The GLOBAL
# 4-GPU ceiling (CLAUDE.md §1) is still enforced — cluster_env.sh clamps it centrally.
_SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../cluster_env.sh
source "${_SITE_DIR}/cluster_env.sh"

export RAMOLE_EXCLUDE="${RAMOLE_EXCLUDE:-${TOFU_EXCLUDE:-}}"
export RAMOLE_CAP="${RAMOLE_CAP:-${TOFU_ARRAY_CAP:-4}}"               # ≤4 GPUs concurrent GLOBALLY across all jobs (user cap 2026-07-09; see ~/CLAUDE.md §1)
export RAMOLE_MEM="${RAMOLE_MEM:-64G}"             # 3B base + 32 expert buffers + instructor-xl
export RAMOLE_RET_TIME="${RAMOLE_RET_TIME:-03:00:00}"
export RAMOLE_ROUTER_TIME="${RAMOLE_ROUTER_TIME:-04:00:00}"
export RAMOLE_EVAL_TIME="${RAMOLE_EVAL_TIME:-02:00:00}"
export PYTHON="${PYTHON:-python3}"
export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
