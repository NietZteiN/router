#!/usr/bin/env bash
# Create/refresh the additive-only `memadapt-eval` branch in ~/open-unlearning
# (user-approved 2026-07-14). Adds 3 files + 2 registration lines; main is
# never touched. Idempotent.

# Repo root — this tree is FLAT, so sibling projects live beside this one.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OU="${OU_DIR:-${REPO_ROOT}/open-unlearning}"

cd "${OU}"
git diff --quiet || { echo "open-unlearning tree is dirty — aborting"; exit 1; }
git checkout memadapt-eval 2>/dev/null || git checkout -b memadapt-eval

cp "${HERE}/memadapt_registry.py" src/model/memadapt_registry.py
cp "${HERE}/MemAdapt-Llama-3.2-1B.yaml" configs/model/MemAdapt-Llama-3.2-1B.yaml
cp "${HERE}/tofu_grimes.yaml" configs/eval/tofu_grimes.yaml

if ! grep -q "memadapt_registry" src/model/__init__.py; then
  cat >> src/model/__init__.py <<'EOF'

from model.memadapt_registry import MemAdaptLlamaForCausalLM  # noqa: E402

_register_model(MemAdaptLlamaForCausalLM)
EOF
fi

git add src/model/memadapt_registry.py src/model/__init__.py \
        configs/model/MemAdapt-Llama-3.2-1B.yaml configs/eval/tofu_grimes.yaml
git diff --cached --quiet || git commit -m "Add MemAdapt eval integration (registry shim + model/eval configs)

Additive-only branch for the memadapt_tofu reproduction; see
${REPO_ROOT}/memadapt_tofu/CLAUDE.md.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
echo "memadapt-eval branch ready:"
git log --oneline -1
