# SLURM node policy: sprint1, sprint2, sprint3 only (never sprint4).
# Use --exclude=sprint4 so each array task gets 1 GPU on an allowed node.
# Do NOT use --nodelist=sprint1,sprint2,sprint3 — that pins all three nodes per task.

# ── Site layer (added on export) ────────────────────────────────────────────────
# WAS: `sprint4` and a literal cap of 4, both hardcoded. Those are sprint-cluster facts,
# not repo facts, so they now come from the repo-root cluster_env.<site>.sh. The GLOBAL
# 4-GPU ceiling (CLAUDE.md §1) is still enforced — cluster_env.sh clamps it centrally.
_SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../cluster_env.sh
source "${_SITE_DIR}/cluster_env.sh"

export TOFU_EXCLUDE="${TOFU_EXCLUDE:-${TOFU_EXCLUDE:-}}"
export TOFU_ALLOWED_NODES="${TOFU_ALLOWED_NODES:-}"
# Max concurrent array tasks (1 GPU each) — ≤4 GPUs GLOBALLY across all jobs (user cap 2026-07-09; see ~/CLAUDE.md §1)
export TOFU_ARRAY_CAP="${TOFU_ARRAY_CAP:-${TOFU_ARRAY_CAP:-4}}"
# Per-task wall clock cap for smoke eval (must finish under 1 hour)
export TOFU_SMOKE_TIME="${TOFU_SMOKE_TIME:-00:55:00}"
# Per-task wall clock cap for extended eval (~5h cluster budget, 17 tasks @ 12 GPUs)
export TOFU_EXTENDED_TIME="${TOFU_EXTENDED_TIME:-02:30:00}"
# Extended eval on Llama-3.2-3B (slower load + ROUGE)
export TOFU_EXTENDED_TIME_3B="${TOFU_EXTENDED_TIME_3B:-03:30:00}"
