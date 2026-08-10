# Site: cispa — A100-SXM4-40GB.
#
# Values carried over from merge-tables-7b/tofu_sisa_lora/cluster_env.sh, which was measured and
# exercised on this cluster on 2026-07-24..26 (the 7B k=200 rebuild). Re-verify the two paths
# below against your account before the first submit; everything else is site policy.

# uv-managed venv resolved from requirements.txt (transformers 4.48.3, peft 0.14.0, trl 0.9.6 —
# the pins these scripts are written against). NOT a venv whose transformers 5.x / peft 0.19
# resolution cannot run this code.
export TOFU_PYTHON="${TOFU_PYTHON:-/home/c03jale/CISPA-az6/mlu-2026/jack_stuff/.venv-tofu/bin/python}"
export TOFU_PLOT_PYTHON="${TOFU_PLOT_PYTHON:-${TOFU_PYTHON}}"
export HF_HOME="${HF_HOME:-/home/c03jale/CISPA-az6/mlu-2026/jack_stuff/data/huggingface}"
# Deliberately NOT a sibling skeleton tree whose adapter files are 0 bytes — that would satisfy
# every driver's "skip if checkpoint exists" test and silently produce nothing.
export TOFU_CKPT_ROOT="${TOFU_CKPT_ROOT:-/home/c03jale/CISPA-az6/mlu-2026/jack_stuff/checkpoints_7b}"

export TOFU_PARTITION="${TOFU_PARTITION:-xe8545}"   # 31 nodes x A100:4 — most capacity here
export TOFU_ACCOUNT="${TOFU_ACCOUNT:-testing}"

# ⚠ NO --mem ON THIS CLUSTER. Nodes report RealMemory=1 and the partition sets
# DefMemPerNode=UNLIMITED, so memory is not a consumable resource and every other job requests
# MIN_MEMORY=0. Asking for a real figure fails AT SUBMIT with "Memory specification can not be
# satisfied" — the old cluster's --mem=48G/64G/96G/160G lines must be DROPPED, not lowered.
export TOFU_SUPPORTS_MEM="${TOFU_SUPPORTS_MEM:-0}"

# Drained / bad xe8545 nodes: these hang GPU jobs with "unspecified launch failure" inside
# torch.cuda init and leak GPUs for hours. Kept non-empty so drivers that emit
# "--exclude=${TOFU_EXCLUDE}" still produce a valid line.
#
# -22 added 2026-08-10: job 3200588_3 (E5 reroute42) ran 4h to TIMEOUT having produced ZERO
# forward passes — its log holds 3087 "NVML: Failed to get usage(999)" lines and the progress
# file never advanced past "forget_ppl 0/400". Sibling arms of the SAME array on -10 and -21
# finished the identical workload in 23-27 min with no NVML line at all, so this is the node,
# not the arm. SLURM still reports -22 as `idle` with no drain reason, which is precisely why
# the exclude list has to carry it: the scheduler will keep handing it out.
export TOFU_EXCLUDE="${TOFU_EXCLUDE:-xe8545-a100-03,xe8545-a100-05,xe8545-a100-12,xe8545-a100-16,xe8545-a100-17,xe8545-a100-22}"
export TOFU_ALLOWED_NODES="${TOFU_ALLOWED_NODES:-${TOFU_PARTITION}}"

# Association: account=testing, GrpTRES gres/gpu=16 PER USER (so using it fully does not starve
# colleagues), MaxTRES gres/gpu=40 per job. Pace against GPUs, not job count: the only
# assoc-side pending reason ever recorded for this user is JobArrayTaskLimit, i.e. our own %N.
export TOFU_ARRAY_CAP="${TOFU_ARRAY_CAP:-6}"
export TOFU_GPUS_PER_TASK="${TOFU_GPUS_PER_TASK:-1}"
export TOFU_CPUS_PER_TASK="${TOFU_CPUS_PER_TASK:-8}"

# Reclaims reserved-but-unallocated CUDA blocks. Fixed the k=200 dense-intermediate OOM on the
# 40 GB cards (~5 GiB was fragmented-reserved, closing a 4.88 GiB gap).
export TOFU_ALLOC_CONF="${TOFU_ALLOC_CONF:-expandable_segments:True}"

# Wall clocks raised vs sprint: NFS-cold model loads dominate unless stage_hf_cache.sh is used.
export TOFU_SMOKE_TIME="${TOFU_SMOKE_TIME:-01:50:00}"
export TOFU_EXTENDED_TIME="${TOFU_EXTENDED_TIME:-05:00:00}"
export TOFU_EXTENDED_TIME_3B="${TOFU_EXTENDED_TIME_3B:-07:00:00}"

# ⚠ RETUNE BEFORE THE FIRST BIG MERGE. 1024 is the A40 (44.5 GiB) figure. These cards are 40 GB
# — SMALLER — so a single-GPU task has LESS headroom, while TOFU_GPUS_PER_TASK=2 with
# device_map="auto" gives 80 GB and more. Measure, then set; do not inherit the A40 number.
export TOFU_MAX_EXACT_RANK="${TOFU_MAX_EXACT_RANK:-1024}"
