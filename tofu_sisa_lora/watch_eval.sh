#!/bin/bash
# Live eval progress: tail logs + show .progress.json files
# Usage: bash watch_eval.sh [checkpoints/phi-2]

DIR="${1:-checkpoints/TinyLlama-1.1B-Chat-v1.0}"
RESULTS="${DIR}/results"
echo "=== Progress files in ${RESULTS} ==="
for f in "${RESULTS}"/*.progress.json; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f") ---"
  cat "$f"
  echo
done
echo "=== Finished JSON ==="
ls -lt "${RESULTS}"/*.json 2>/dev/null | grep -v progress || echo "(none yet)"
echo "=== Recent log lines (latest job) ==="
tail -3 "${DIR}"/logs/eval_*.log 2>/dev/null | tail -20
