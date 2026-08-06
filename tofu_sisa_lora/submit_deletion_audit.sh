#!/bin/bash
# Deletion-audit attack A4 (composed-model MIA) driver.
#   bash submit_deletion_audit.sh [smoke|phase1|all]
# smoke  = TinyLlama ft vs retain90 go/no-go (must separate by ΔAUC >= 0.15 before any 1B job).
# phase1 = the 1B condition matrix in configs/deletion_audit.json (+ reseed on the exact arms).
# STUB=1 prints the sbatch array script + the enumerated commands without submitting.
# Existing results/mia/<label>.json are skipped (idempotent re-runs).
set -euo pipefail
PHASE="${1:-phase1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CFG="${SCRIPT_DIR}/configs/deletion_audit.json"
CMD_DIR="${SCRIPT_DIR}/checkpoints/deletion_audit_cmds"
mkdir -p "${CMD_DIR}"

# Enumerate one attack_mia.py command per condition into a file; python owns the flag logic.
CMD_FILE="${CMD_DIR}/${PHASE}_cmds.txt"
"${PYTHON}" - "$PHASE" "$CFG" "$SCRIPT_DIR" "$PYTHON" > "${CMD_FILE}" <<'PYEOF'
import json, os, sys
phase, cfg_path, script_dir, python = sys.argv[1:5]
cfg = json.load(open(cfg_path))
mia = cfg["mia"]
def base_flags(model, out):
    return (f'--model_name {model} --output_dir {script_dir} --k 10 --forget_shard_id 9 '
            f'--attacks {mia["attacks"]} --min_k_frac {mia["min_k_frac"]} '
            f'--mia_label_scope {mia["label_scope"]} --member_split {mia["member_split"]} '
            f'--holdout_split {mia["holdout_split"]} --batch_size {mia["batch_size"]} '
            f'--out {out}')

def arm_flags(c, root):
    f = []
    if c.get("preloaded_adapter"): f += ["--preloaded_adapter", os.path.join(root, c["preloaded_adapter"])]
    if c.get("legonet_config"): f += ["--legonet_config", c["legonet_config"]]
    if c.get("legonet_unlearn_tag"): f += ["--legonet_unlearn_tag", c["legonet_unlearn_tag"]]
    if c.get("ramole_router"): f += ["--ramole_router", os.path.join(root, c["ramole_router"])]
    if c.get("ramole_route"): f += ["--ramole_route", c["ramole_route"]]
    if c.get("sift_masks_config"): f += ["--sift_masks_config", c["sift_masks_config"]]
    if c.get("sift_unlearn_tag"): f += ["--sift_unlearn_tag", c["sift_unlearn_tag"]]
    if c.get("clamu_config"): f += ["--clamu_config", c["clamu_config"]]
    if c.get("clamu_unlearn_tag"): f += ["--clamu_unlearn_tag", c["clamu_unlearn_tag"]]
    return " ".join(f)

cmds = []
if phase == "smoke":
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    root = "${TOFU_CKPT_ROOT}"
    outd = os.path.join(root, "TinyLlama-1.1B-Chat-v1.0", "results", "mia")
    for label, adapter in [("smoke_retain90", "TinyLlama-1.1B-Chat-v1.0/retain90"),
                           ("smoke_ft", "TinyLlama-1.1B-Chat-v1.0_ft/shard_0")]:
        out = os.path.join(outd, f"{label}.json")
        if os.path.exists(out): continue
        cmds.append(f'{python} {script_dir}/attack_mia.py --label {label} '
                    f'--preloaded_adapter {os.path.join(root, adapter)} {base_flags(model, out)}')
else:
    p = cfg["phase1_1b"]
    model, root = p["model_name"], p["pool_root"]
    outd = os.path.join(root, p["out_pool"], "results", "mia")
    for c in p["conditions"]:
        reseeds = [None] + ([p["reseed"]] if c["label"] in p.get("reseed_labels", []) else [])
        for sd in reseeds:
            label = c["label"] if sd is None else f'{c["label"]}_s{sd}'
            out = os.path.join(outd, f"{label}.json")
            if os.path.exists(out): continue
            seed_flag = f' --seed {sd}' if sd is not None else f' --seed {mia["seed"]}'
            cmds.append(f'{python} {script_dir}/attack_mia.py --label {label} '
                        f'{arm_flags(c, root)} {base_flags(model, out)}{seed_flag}')
for c in cmds:
    print(c)
PYEOF

N=$(wc -l < "${CMD_FILE}")
echo "deletion-audit ${PHASE}: ${N} commands -> ${CMD_FILE}" >&2
if [ "${N}" -eq 0 ]; then echo "nothing to do (all results exist)"; exit 0; fi
sed -n '1,3p' "${CMD_FILE}" >&2

LOG_DIR="${CMD_DIR}/logs"; mkdir -p "${LOG_DIR}"
SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=deletion-audit-${PHASE}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --array=1-${N}%4
#SBATCH --output=${LOG_DIR}/${PHASE}_%A_%a.log
set -e
export HF_HOME="${HF_HOME}"; export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${CMD_FILE}")
echo "[task \${SLURM_ARRAY_TASK_ID}] \${CMD}"
eval "\${CMD}"
EOF
)
if [ "${STUB:-0}" = "1" ]; then echo "${SBATCH_SCRIPT}"; echo "----(STUB)----"; exit 0; fi
echo "${SBATCH_SCRIPT}" | sbatch --parsable
