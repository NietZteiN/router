# Site: sprint — the original cluster (A40, 46 GiB usable).
#
# These are the settings that were hardcoded across the drivers before 2026-07-28. Values are
# unchanged, so sourcing this must reproduce the previous behaviour exactly.

export TOFU_PYTHON="${TOFU_PYTHON:-/home/jack/anaconda3/envs/test-env/bin/python}"
# Base anaconda, the ONLY interpreter here with matplotlib+pandas — plot_*.py must use it.
export TOFU_PLOT_PYTHON="${TOFU_PLOT_PYTHON:-/home/jack/anaconda3/bin/python}"
export HF_HOME="${HF_HOME:-/storage2/jack/data/huggingface}"
export TOFU_CKPT_ROOT="${TOFU_CKPT_ROOT:-/storage2/jack/checkpoints/tofu_sisa_lora}"

export TOFU_PARTITION="${TOFU_PARTITION:-all}"
export TOFU_ACCOUNT="${TOFU_ACCOUNT:-}"          # no account required here
# sprint1/2/3 only, never sprint4. Do NOT use --nodelist=sprint1,sprint2,sprint3 — that pins all
# three nodes per task.
export TOFU_EXCLUDE="${TOFU_EXCLUDE:-sprint4}"
export TOFU_ALLOWED_NODES="${TOFU_ALLOWED_NODES:-sprint1,sprint2,sprint3}"

export TOFU_GPUS_PER_TASK="${TOFU_GPUS_PER_TASK:-1}"
export TOFU_CPUS_PER_TASK="${TOFU_CPUS_PER_TASK:-4}"
export TOFU_SUPPORTS_MEM="${TOFU_SUPPORTS_MEM:-1}"

# GLOBAL 4-GPU cap across ALL queued jobs (user policy, ~/CLAUDE.md §1). The `%N` throttles of
# every simultaneously-queued array must SUM to <= 4 — this is not a per-array limit.
export TOFU_ARRAY_CAP="${TOFU_ARRAY_CAP:-4}"

export TOFU_SMOKE_TIME="${TOFU_SMOKE_TIME:-00:55:00}"
export TOFU_EXTENDED_TIME="${TOFU_EXTENDED_TIME:-02:30:00}"
export TOFU_EXTENDED_TIME_3B="${TOFU_EXTENDED_TIME_3B:-03:30:00}"

# Materialized-adapter rank ceiling for a 44.5 GiB A40 next to a 7B bf16 base: rank 2064 OOM'd
# (eval wave 443532, 2026-07-16). Consumed by the APA config's exact_max_n / svd_rank choice.
export TOFU_MAX_EXACT_RANK="${TOFU_MAX_EXACT_RANK:-1024}"
