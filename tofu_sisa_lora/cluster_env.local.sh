# Site: local — a single workstation, with or without SLURM.
#
# The project-level twin of the repo-root cluster_env.local.sh. It has to exist here as well
# because THIS directory's cluster_env.sh resolves site files next to itself, so without it
# SETUP.md's `export TOFU_SITE=local` dies at the site-file check for every driver and gate
# under tofu_sisa_lora/ — the one instruction a new clone follows first.
#
# Nothing below is a machine-specific absolute path: every value comes from the environment, and
# the two that have no sensible default fail loudly naming the export that fixes them. `sprint`
# and `cispa` are the opposite — they hardcode a known-good machine.
#
#   export TOFU_PYTHON=$(which python)          # an env built from requirements.txt
#   export HF_HOME=~/hf                          # must contain hub/
#   export TOFU_CKPT_ROOT=~/tofu_checkpoints     # where pools/merges/results are written
#
# WITHOUT SLURM: every submit_*.sh honours STUB=1, which prints each job script instead of
# submitting it. Preview, then run the printed `python …` line directly.

export TOFU_PYTHON="${TOFU_PYTHON:-$(command -v python3 || command -v python)}"
# Plots need matplotlib, which requirements.txt deliberately omits. Point this at an interpreter
# that has it, or install requirements-plots.txt into the same env and leave it.
export TOFU_PLOT_PYTHON="${TOFU_PLOT_PYTHON:-${TOFU_PYTHON}}"

: "${HF_HOME:?set HF_HOME to your HuggingFace cache dir (the one containing hub/), e.g. export HF_HOME=\$HOME/.cache/huggingface}"
: "${TOFU_CKPT_ROOT:?set TOFU_CKPT_ROOT to a writable dir with room for the pools (49 GB per 200-adapter pool) and merges (~258 MB * N), e.g. export TOFU_CKPT_ROOT=\$HOME/tofu_checkpoints}"
export HF_HOME TOFU_CKPT_ROOT

# SLURM knobs. Harmless when there is no SLURM — nothing reads them unless a driver submits.
export TOFU_PARTITION="${TOFU_PARTITION:-local}"
export TOFU_ACCOUNT="${TOFU_ACCOUNT:-}"          # empty => no --account line is emitted
export TOFU_EXCLUDE="${TOFU_EXCLUDE:-}"          # empty => no --exclude line is emitted
export TOFU_ALLOWED_NODES="${TOFU_ALLOWED_NODES:-}"

export TOFU_GPUS_PER_TASK="${TOFU_GPUS_PER_TASK:-1}"
export TOFU_CPUS_PER_TASK="${TOFU_CPUS_PER_TASK:-4}"
export TOFU_SUPPORTS_MEM="${TOFU_SUPPORTS_MEM:-1}"

# Conservative: one GPU task at a time. Raise only if you know your own limits.
export TOFU_ARRAY_CAP="${TOFU_ARRAY_CAP:-1}"

export TOFU_SMOKE_TIME="${TOFU_SMOKE_TIME:-02:00:00}"
export TOFU_EXTENDED_TIME="${TOFU_EXTENDED_TIME:-06:00:00}"
export TOFU_EXTENDED_TIME_3B="${TOFU_EXTENDED_TIME_3B:-08:00:00}"

# Materialized-adapter rank ceiling. The merge is a rank-32N concatenation held in fp32 next to
# the bf16 base, so this is the number that decides where the exact ladder stops and the
# svd_rank rungs begin. 1024 is the measured 44.5 GiB A40 figure (rank 2064 OOM'd, 2026-07-16).
# MEASURE IT on your card before the first N>=64 merge rather than inheriting that number:
#   materialized bytes ~ 2.013e6 * rank * 4
export TOFU_MAX_EXACT_RANK="${TOFU_MAX_EXACT_RANK:-1024}"
