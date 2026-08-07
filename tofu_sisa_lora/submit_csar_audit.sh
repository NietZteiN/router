#!/bin/bash
# CSAR pilot — what a routed system SAYS about a source it was asked to delete.
# (log/selector_audit/2026-08-07_e1-router-probe-and-preregistration.md, pre-registration §CSAR.)
#
# Generates orphan answers on the k=200 e25 pool with authors 180-199 deleted, under the two
# feature-space routers the pool has score matrices for, then classifies each answer with
# selector_audit/csar.py.
#
# The prior is adverse and that is the point: at k=10 the same audit reports sibling-vs-sibling
# ROUGE-L of 0.181 against a 0.249 base floor and a 95.5% confabulation rate, i.e. NO detectable
# cross-source attribution. This run changes the two things that could account for that — the
# routing unit becomes ONE author instead of twenty, and the metric becomes fact-level instead of
# ROUGE-L. If CSAR stays low under both changes, §4.3 is a paragraph and the paper leads elsewhere.
#
# Usage: bash submit_csar_audit.sh [gen|score|all]      # STUB=1 previews
#   gen    GPU: dump_generations_routed.py --strategies, QPA questions per deleted author
#   score  CPU: csar.py over the dump + a 300-record hand-labelling sample
#   all    gen, then score chained --dependency=afterany
#
# QPA=5 (default) samples 5 of each deleted author's 20 questions = 100 orphan queries spread
# over all 20 authors. --max_questions would instead head-slice the first two authors, which at
# this granularity would measure two people. Set QPA=20 for the full 400.
set -euo pipefail

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
MODEL="meta-llama/Llama-2-7B-chat-hf"
E25="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"
FORGET="180-199"
QPA="${QPA:-5}"
# Only feature-space routers. The behavioral family (ppl / activation_norm / attn_norm /
# logit_div) scores by running EVERY candidate expert on EVERY query, which is impractical past
# ~50 sources and would dominate this job's cost at k=200 for no CSAR-specific gain.
STRATS="${STRATS:-centroid_sbert,key_tfidf}"
LOG_DIR="${CKPT}/csar_logs"
RES="${E25}/results/router_leak"
OUT="${RES}/sibling_content_k200_f10_qpa${QPA}.json"
mkdir -p "${LOG_DIR}" "${RES}"

submit() {
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----" >&2
    printf '%s\n' "$1" >&2
    echo "-----------------------------------------------" >&2
    echo "STUB"
  else
    printf '%s\n' "$1" | sbatch --parsable ${2:+--dependency=afterany:$2}
  fi
}

gen_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=csar-gen
$(tofu_sbatch_resources 1 8 48G)
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/gen_%j.log
#SBATCH --error=${LOG_DIR}/gen_%j.log
set -eo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token")"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
echo "=== CSAR generation: k=200, delete ${FORGET}, ${QPA} q/author, strategies ${STRATS} ==="
date
[ -f "${OUT}" ] && { echo "skip existing ${OUT}"; exit 0; }
for i in \$(seq 0 199); do
  [ -f "${E25}/shard_\${i}/adapter_model.safetensors" ] || { echo "MISSING shard_\${i}"; exit 1; }
done
${PYTHON} "${SCRIPT_DIR}/dump_generations_routed.py" \\
  --model_name "${MODEL}" --shards_dir "${E25}" --k 200 \\
  --forget_author_ids "${FORGET}" --questions_per_author ${QPA} \\
  --strategies "${STRATS}" --lazy_adapter_cache 8 \\
  --hf_home "${HF_HOME}" --out "${OUT}"
date
EOF
}

score_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=csar-score
$(tofu_sbatch_resources 0 4 16G)
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/score_%j.log
#SBATCH --error=${LOG_DIR}/score_%j.log
set -eo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
echo "=== CSAR scoring ==="
date
[ -f "${OUT}" ] || { echo "no generation dump at ${OUT} — the gen stage failed"; exit 1; }
${PYTHON} "${REPO_ROOT}/selector_audit/test_csar.py"
${PYTHON} "${REPO_ROOT}/selector_audit/csar.py" --audit_json "${OUT}" \\
  --hf_home "${HF_HOME}" \\
  --out_json "${RES}/csar_k200_f10_qpa${QPA}.json" \\
  --out_md "${RES}/csar_k200_f10_qpa${QPA}.md"
# A CSAR quoted before the judge is checked against humans is not a result. Emit the sample now
# so the labelling is not what blocks the write-up later.
${PYTHON} "${REPO_ROOT}/selector_audit/csar.py" --audit_json "${OUT}" \\
  --hf_home "${HF_HOME}" --sample_for_labeling 300 \\
  --out_jsonl "${RES}/csar_k200_f10_qpa${QPA}.label_me.jsonl"
date
EOF
}

case "${STAGE}" in
gen)   echo "CSAR gen: k=200, ${QPA} q/author over 20 deleted authors, ${STRATS}"
       submit "$(gen_body)" "${DEP:-}" ;;
score) echo "CSAR score (CPU)"
       submit "$(score_body)" "${DEP:-}" ;;
all)
  echo "CSAR chain: ${DEP:+afterany:${DEP} -> }gen -> score (afterany)"
  GEN_ID="$(submit "$(gen_body)" "${DEP:-}")"
  echo "gen job:   ${GEN_ID}${DEP:+ (afterany:${DEP})}"
  if [ "${GEN_ID}" = "STUB" ]; then
    submit "$(score_body)"
  else
    SCORE_ID="$(submit "$(score_body)" "${GEN_ID}")"
    echo "score job: ${SCORE_ID} (afterany:${GEN_ID})"
  fi
  ;;
*) echo "usage: bash submit_csar_audit.sh [gen|score|all]  (STUB=1 previews, QPA=n)"; exit 1 ;;
esac
