# SLURM node policy for legonet_lora: sprint1, sprint2, sprint3 only (never sprint4).
# Use --exclude=sprint4 so each array task gets 1 GPU on an allowed node
# (matches tofu_sisa_lora/slurm_nodes.sh; do NOT --nodelist all three or each task pins them).

# ── Site layer (added on export) ────────────────────────────────────────────────
# WAS: `sprint4` and a literal cap of 4, both hardcoded. Those are sprint-cluster facts,
# not repo facts, so they now come from the repo-root cluster_env.<site>.sh. The GLOBAL
# 4-GPU ceiling (CLAUDE.md §1) is still enforced — cluster_env.sh clamps it centrally.
_SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../cluster_env.sh
source "${_SITE_DIR}/cluster_env.sh"

export LEGO_EXCLUDE="${LEGO_EXCLUDE:-${TOFU_EXCLUDE:-}}"
export LEGO_ARRAY_CAP="${LEGO_ARRAY_CAP:-${TOFU_ARRAY_CAP:-4}}"    # ≤4 GPUs concurrent GLOBALLY across all jobs (user cap 2026-07-09; see ~/CLAUDE.md §1)
export LEGO_MEM="${LEGO_MEM:-64G}"              # 64G for 7B; override to 24G for TinyLlama smoke
export LEGO_TRAIN_TIME="${LEGO_TRAIN_TIME:-02:00:00}"
export LEGO_EVAL_TIME="${LEGO_EVAL_TIME:-01:30:00}"
export LEGO_SETUP_TIME="${LEGO_SETUP_TIME:-00:40:00}"
export PYTHON="${PYTHON:-python3}"
export HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
