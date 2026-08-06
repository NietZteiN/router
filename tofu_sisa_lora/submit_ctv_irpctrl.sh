#!/bin/bash
# composable_tv H-lin-1 CONTROL driver: the frozen-A TWIN pool (irpctrl).
#
# H-lin-1 asks whether the lin arm's behavior comes from LINEARIZED SERVING or merely
# from its frozen seeded A projections. This driver trains the twin control: the SAME
# 20 pool authors (merge_subset.subset_authors(42, 20) — derived at runtime, never
# hardcoded) through the STANDARD nonlinear trainer train_lora_shard.py with
# --irp_seed 42 (frozen per-shard lora_A) at the e25 recipe — flag-free defaults give
# r32 / alpha 64 / rslora / lr 1e-4 / seed 42; only --epochs 25 and --irp_seed 42
# deviate from the frozen recipe. Adapters land at ${OUT_DIR}/shard_<author>/.
#
# Usage: bash submit_ctv_irpctrl.sh [train|evalrows|all] [manifest]
#
#   train    20-task GPU array (1 task per pool author), 1 GPU / 00:45:00 each,
#            sprint1-3 only (exclude sprint4); self-skips existing adapter dirs.
#   evalrows (login, light — no sbatch) write the twin iso eval-manifest rows
#            `iso_irp_a<a>\t${OUT_DIR}/shard_<a>\t<a>\t<a>` for the 5 probe authors
#            (IRP_ALL20=1 -> all 20 pool authors) to ${OUT_DIR}/eval_manifest_irpctrl.txt,
#            or APPEND them to the manifest given as $2. Serve them through submit_ctv.sh's
#            eval stage with the LIN config (its do_eval default case serves plain
#            --preloaded_adapter rows independent of arm):
#              EVAL_MANIFEST=${OUT_DIR}/eval_manifest_irpctrl.txt \
#                bash submit_ctv.sh configs/ctv_1b_lin.json eval
#   all      train, then evalrows.
#
# Conventions cloned from submit_ctv.sh: STUB=1 previews every sbatch script without
# submitting (and skips the squeue guard); cap_guard enforces the GLOBAL 4-GPU cap
# (~/CLAUDE.md §1) before every submission; ARRAY_CAP defaults to 1 here (the twin pool
# usually runs beside the lin campaign — leave headroom).
# Env overrides: ARRAY_CAP (default 1, never >4) | TRAIN_TIME (00:45:00) |
#   TRAIN_ARRAY (explicit --array spec; keep %N <= ARRAY_CAP) | IRP_ALL20=1 (evalrows)
set -euo pipefail

STAGE="${1:-all}"
MANIFEST_ARG="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

# Twin-pool constants (mirror configs/ctv_1b_lin.json; flags verified against
# train_lora_shard.py's argparse — defaults are the frozen r32/a64/rslora/lr1e-4/seed42)
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
OUT_DIR="${SCRIPT_DIR}/checkpoints/Llama-3.2-1B-Instruct_ctv_irpctrl_r32_e25"
POOL_SEED=42
POOL_SIZE=20
K=200
EPOCHS=25
IRP_SEED=42
TRAIN_TIME="${TRAIN_TIME:-00:45:00}"
LOG_DIR="${OUT_DIR}/logs"

ARRAY_CAP="${ARRAY_CAP:-1}"   # default 1: the twin usually queues beside the lin arrays
if ! [[ "${ARRAY_CAP}" =~ ^[0-9]+$ ]] || [ "${ARRAY_CAP}" -lt 1 ] \
   || [ "${ARRAY_CAP}" -gt 4 ] || [ "${ARRAY_CAP}" -gt "${TOFU_ARRAY_CAP}" ]; then
  echo "[irpctrl] ARRAY_CAP=${ARRAY_CAP} invalid — must be 1..min(4, TOFU_ARRAY_CAP=${TOFU_ARRAY_CAP})" >&2
  exit 1
fi

submit() {  # submit <script-text>; honors STUB=1 (preview, nothing reaches sbatch)
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----"
    printf '%s\n' "$1"
    echo "-----------------------------------------------"
    LAST_JOB=""
  else
    LAST_JOB="$(printf '%s\n' "$1" | sbatch --parsable)"
    echo "submitted job ${LAST_JOB}"
  fi
}

# COPY of submit_ctv.sh's cap_guard — keep the two in sync (incl. the F6 typed-gres
# gpus_of fix: "gres/gpu:a40:1" is 1 GPU, not 40).
cap_guard() {  # cap_guard <this submission's worst-case concurrent GPUs>
  local new="$1"
  if [ "${STUB:-0}" = "1" ]; then
    echo "[cap] STUB=1 — squeue guard skipped (this submission would add up to ${new} concurrent GPU(s))"
    return 0
  fi
  "${PYTHON}" - "${new}" <<'CAPEOF'
import re
import subprocess
import sys

NEW = int(sys.argv[1])
CAP = 4  # GLOBAL concurrent-GPU cap across ALL our jobs (~/CLAUDE.md §1)

# The mandated audit view (job ids can truncate at 10 chars — display only).
audit = subprocess.run(["squeue", "-u", "jack", "-o", "%.10i %.20j %.10T %.10b %F"],
                       capture_output=True, text=True)
if audit.returncode != 0:
    sys.exit(f"[cap] squeue failed ({audit.stderr.strip()}) — refusing to submit blind")
print("[cap] squeue -u jack:")
print(audit.stdout.rstrip() or "  (queue empty)")
# Untruncated fields for the arithmetic (the audit view can cut array %N throttles off).
raw = subprocess.run(["squeue", "-u", "jack", "-h", "-o", "%i|%T|%b|%F"],
                     capture_output=True, text=True).stdout

def gpus_of(tres):
    # TRAILING count of an optionally-typed gres: "gres/gpu:1" and "gres/gpu:a40:1"
    # are both 1 GPU (the old r"gpu\D*(\d+)" read the 40 out of the a40 TYPE);
    # a bare "gpu" with no count is 1, no gpu at all is 0.
    m = re.search(r"gpu(?::[A-Za-z][\w-]*)?:(\d+)", tres)
    return int(m.group(1)) if m else (1 if "gpu" in tres else 0)

groups = {}
for ln in raw.splitlines():
    parts = [p.strip() for p in ln.split("|")]
    if len(parts) < 4:
        continue
    jid, state, tres, arr = parts[:4]
    if state in ("COMPLETING", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"):
        continue
    g = gpus_of(tres)
    if g == 0:
        continue
    grp = groups.setdefault(arr or jid.split("_")[0], {"g": 0, "tasks": 0, "cap": None})
    grp["g"] = max(grp["g"], g)
    m = re.search(r"\[([^\]]*)\]", jid)
    if m:  # pending array line, e.g. 0-19%2 or 3,5-9%2: count tasks + read the throttle
        spec = m.group(1)
        c = re.search(r"%(\d+)", spec)
        if c:
            grp["cap"] = int(c.group(1))
        for piece in spec.split("%")[0].split(","):
            if "-" in piece:
                try:
                    a, b = piece.split("-")[:2]
                    grp["tasks"] += int(b) - int(a) + 1
                except ValueError:
                    grp["tasks"] += 1  # unparseable range -> count 1 (throttle still bounds it)
            elif piece:
                grp["tasks"] += 1
    else:
        grp["tasks"] += 1

# Assume every pending job can start NOW: per array, worst case = min(throttle, tasks).
existing = 0
for key, grp in sorted(groups.items()):
    worst = grp["tasks"] if grp["cap"] is None else min(grp["cap"], grp["tasks"])
    print(f"[cap]   job {key}: worst-case {worst} concurrent task(s) x {grp['g']} GPU = {worst * grp['g']}")
    existing += worst * grp["g"]
total = existing + NEW
print(f"[cap] arithmetic: existing worst-case {existing} + this submission {NEW} = {total} (global cap {CAP})")
if total > CAP:
    sys.exit(f"[cap] REFUSING to submit: {total} > {CAP} — wait for the queue to drain or chain with --dependency")
print("[cap] OK")
CAPEOF
}

# ── train: 20-task GPU array over the derived pool ─────────────────────────────────
do_train() {
  local n_tasks=${POOL_SIZE}
  local spec="${TRAIN_ARRAY:-0-$((n_tasks - 1))%${ARRAY_CAP}}"
  local mine=$(( n_tasks < ARRAY_CAP ? n_tasks : ARRAY_CAP ))
  cap_guard "${mine}"
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=ctv-irpctrl-train
#SBATCH --array=${spec}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=${TRAIN_TIME}
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
set -eo pipefail   # F2 convention: never mask a failing trainer inside the job

# pool derives at runtime (merge_subset.subset_authors — never hardcoded)
AUTHOR=\$(${PYTHON} -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from merge_subset import subset_authors; print(int(subset_authors(${POOL_SEED}, ${POOL_SIZE})[\${SLURM_ARRAY_TASK_ID}]))")
DONE="${OUT_DIR}/shard_\${AUTHOR}"
if [ -f "\${DONE}/adapter_model.safetensors" ] || [ -f "\${DONE}/adapter_config.json" ]; then
  echo "Skip existing \${DONE}"; exit 0
fi
echo "=== ctv irpctrl train task \${SLURM_ARRAY_TASK_ID}: author \${AUTHOR} (frozen-A twin) ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
# optional token read: genuinely non-fatal, guard it under set -e
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token" || true)"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/train_lora_shard.py" \\
  --shard_id "\${AUTHOR}" \\
  --k ${K} \\
  --model_name "${MODEL_NAME}" \\
  --output_dir "${OUT_DIR}" \\
  --epochs ${EPOCHS} \\
  --irp_seed ${IRP_SEED} \\
  --hf_home "${HF_HOME}"
date
EOF
  echo "irpctrl train array: ${n_tasks} tasks (spec ${spec}), ${TRAIN_TIME}/task, exclude ${TOFU_EXCLUDE}"
  submit "${S}"
}

# ── evalrows: write/append the twin iso manifest rows (no sbatch — login-light) ────
do_evalrows() {
  local manifest="${MANIFEST_ARG:-${OUT_DIR}/eval_manifest_irpctrl.txt}"
  local mode="write"
  [ -n "${MANIFEST_ARG}" ] && mode="append"
  mkdir -p "${OUT_DIR}"
  "${PYTHON}" - "${OUT_DIR}" "${manifest}" "${mode}" "${SCRIPT_DIR}" \
      "${POOL_SEED}" "${POOL_SIZE}" "${IRP_ALL20:-0}" <<'ROWEOF'
import os
import sys

out_dir, manifest, mode, script_dir, seed, pool_size, all20 = sys.argv[1:8]
sys.path.insert(0, script_dir)
from merge_subset import probe_authors, subset_authors

seed, pool_size = int(seed), int(pool_size)
if all20 == "1":
    authors = [int(a) for a in subset_authors(seed, pool_size)]
else:
    authors = [int(a) for a in probe_authors(seed, pool_size, 5)]
rows = [(f"iso_irp_a{a}", os.path.join(out_dir, f"shard_{a}"), str(a), str(a))
        for a in authors]
existing = set()
if mode == "append" and os.path.exists(manifest):
    with open(manifest) as f:
        existing = {ln.rstrip("\n") for ln in f}
with open(manifest, "a" if mode == "append" else "w") as f:
    n = 0
    for r in rows:
        line = "\t".join(r)
        if line in existing:      # append mode: never duplicate rows
            continue
        f.write(line + "\n")
        n += 1
print(f"[evalrows] {mode} {n} iso_irp rows ({len(authors)} authors) -> {manifest}")
ROWEOF
  echo "[evalrows] serve them via the ctv eval stage (plain --preloaded_adapter rows,"
  echo "[evalrows] arm-independent), e.g.:"
  echo "[evalrows]   EVAL_MANIFEST=${manifest} bash ${SCRIPT_DIR}/submit_ctv.sh ${SCRIPT_DIR}/configs/ctv_1b_lin.json eval"
}

mkdir -p "${LOG_DIR}"

case "${STAGE}" in
  train)    do_train ;;
  evalrows) do_evalrows ;;
  all)      do_train; do_evalrows ;;
  *) echo "unknown stage '${STAGE}' (train|evalrows|all)" >&2; exit 1 ;;
esac
