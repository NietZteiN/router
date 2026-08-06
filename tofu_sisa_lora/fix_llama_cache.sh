#!/bin/bash
# Re-download meta-llama/Llama-3.2-1B-Instruct after a corrupted/incomplete HF cache.
set -euo pipefail

HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
export HF_HOME
PYTHON="${PYTHON:-python3}"
REPO="meta-llama/Llama-3.2-1B-Instruct"
CACHE_DIR="${HF_HOME}/hub/models--meta-llama--Llama-3.2-1B-Instruct"

echo "Removing broken cache: ${CACHE_DIR}"
rm -rf "${CACHE_DIR}"

echo "Re-downloading ${REPO} ..."
"${PYTHON}" -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${REPO}', cache_dir=os.path.join('${HF_HOME}', 'hub'))
print('OK:', path)
"

echo "Verifying model load ..."
"${PYTHON}" -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('${REPO}', torch_dtype='auto')
print('Load OK, params:', sum(p.numel() for p in m.parameters()))
"

echo "Done. Run: bash submit_overnight.sh 4 ${REPO}"
