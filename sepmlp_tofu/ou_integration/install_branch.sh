#!/usr/bin/env bash
# Extend the EXISTING additive-only `memadapt-eval` branch in ~/open-unlearning
# with the SepMLP eval integration: 2 files + 2 registration lines. Never
# creates a second branch — there is a single OU working tree and concurrent
# evals would race on a checkout, so all custom eval integrations share
# memadapt-eval. This script NEVER commits anything (house rule: no git commits
# without the user asking); it copies files, appends the registry import, and
# prints the suggested commit for the operator to run with user approval.
#
# Guards (both refuse by default):
#   1. squeue: any of our SLURM jobs whose name matches *eval* means an OU eval
#      may be reading this tree right now — installing under it would race.
#   2. dirty tree: the OU tree currently carries a DELIBERATE uncommitted
#      fp32-logits fix in src/model/__init__.py (transformers >= 4.49 stopped
#      upcasting logits; the metric code and the reference eval logs need
#      fp32 — see the in-file comment there). Preferred resolution: the
#      operator commits that fix to memadapt-eval FIRST (with the user's
#      explicit approval), then re-runs this script on a clean tree.
#      Escape hatch: ALLOW_DIRTY=1 proceeds additively on top of the dirty
#      tree without committing anything.

# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OU="${OU_DIR:-${REPO_ROOT}/open-unlearning}"
SLURM_USER="${USER:-jack}"

# --- guard 1: no eval job may be queued/running against this tree -----------
QCHECK="squeue -u ${SLURM_USER} -h -o %j"
echo "[guard] eval-job check: ${QCHECK} | grep -i eval"
if ! command -v squeue >/dev/null 2>&1; then
  echo "REFUSING: squeue not available — cannot verify no OU eval job is running." >&2
  exit 1
fi
EVAL_JOBS="$(${QCHECK} 2>/dev/null | grep -i "eval" || true)"
if [[ -n "${EVAL_JOBS}" ]]; then
  echo "REFUSING: SLURM job(s) matching *eval* are queued/running — they may be" >&2
  echo "executing out of ${OU}; installing now would swap files under them:" >&2
  squeue -u "${SLURM_USER}" -o "%.10i %.20j %.10T %.10b %F" | sed -n '1p;/[eE][vV][aA][lL]/p' >&2
  echo "Wait for them to finish (or cancel them) and re-run." >&2
  exit 1
fi

cd "${OU}"

# --- guard 2: clean-tree check (tracked files; default REFUSE) --------------
DIRTY="$(git status --porcelain --untracked-files=no)"
if [[ -n "${DIRTY}" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "open-unlearning tree is DIRTY (tracked changes):"
  git status --short --untracked-files=no
  git diff --stat
  git diff --cached --stat
  cat >&2 <<'MSG'
REFUSING to install (default). If the dirt above is the deliberate fp32-logits
fix in src/model/__init__.py, resolve it one of two ways:
  1. (preferred) With the user's EXPLICIT approval, commit that existing fix to
     the memadapt-eval branch first, then re-run this script on a clean tree.
  2. ALLOW_DIRTY=1 ./install_branch.sh
     — proceed additively on top of the dirty tree without committing.
This script never commits anything on its own.
MSG
  exit 1
fi
[[ -n "${DIRTY}" ]] && echo "[warn] ALLOW_DIRTY=1: proceeding additively on a dirty tree (nothing will be committed)"

# --- extend the existing branch (never create a new one) --------------------
if ! git rev-parse --verify --quiet memadapt-eval >/dev/null; then
  echo "REFUSING: branch memadapt-eval does not exist in ${OU} — run" >&2
  echo "${REPO_ROOT}/memadapt_tofu/ou_integration/install_branch.sh first (it creates it)." >&2
  exit 1
fi
if [[ "$(git branch --show-current)" != "memadapt-eval" ]]; then
  git checkout memadapt-eval
fi

cp "${HERE}/sepmlp_registry.py" src/model/sepmlp_registry.py
cp "${HERE}/SepMlp-Llama-3.2-1B.yaml" configs/model/SepMlp-Llama-3.2-1B.yaml

if ! grep -q "sepmlp_registry" src/model/__init__.py; then
  cat >> src/model/__init__.py <<'EOF'

from model.sepmlp_registry import SepMlpLlamaForCausalLM  # noqa: E402

_register_model(SepMlpLlamaForCausalLM)
EOF
fi

echo "SepMLP eval integration installed on branch memadapt-eval (NOT committed):"
git status --short -- src/model/sepmlp_registry.py src/model/__init__.py \
                      configs/model/SepMlp-Llama-3.2-1B.yaml
cat <<'MSG'
Next step — only with the user's explicit approval — commit the additions:
  cd ${REPO_ROOT}/open-unlearning
  git add src/model/sepmlp_registry.py src/model/__init__.py \
          configs/model/SepMlp-Llama-3.2-1B.yaml
  git commit -m "Add SepMLP eval integration (registry shim + model config)"
MSG
