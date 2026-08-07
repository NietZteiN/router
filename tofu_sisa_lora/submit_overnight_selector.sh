#!/bin/bash
# selector_audit — the unattended overnight campaign.
# (thread log/selector_audit/; run this, then log off.)
#
# EVERYTHING here is sbatch. SLURM jobs run on compute nodes and survive the login/interactive
# node going away — the only things that would not are interactive analysis steps, so the
# scoring and the final consolidation are queued as jobs too, chained behind the work they read.
#
# Stages:
#   B1  CSAR at FULL length (QPA 20) on name-stripped queries    gen -> score
#   B2  CSAR at FULL length (QPA 20) on indirect references      gen -> score
#   B3  CSAR at FULL length (QPA 20) with the `random` control   gen -> score
#         The q0-4 subsample used for the pilots overstates CSAR by ~0.13 (identity questions
#         come first and are the most attribution-prone), so the publishable numbers need these.
#   C   consolidate: one markdown report over every arm that has landed, chained afterany on
#         B1-B3 AND on whatever is already running (pass via DEP=id1:id2:...).
#
# Dependencies are `afterany`, never `afterok`: kill_invalid_depend is off cluster-wide here, so
# an afterok chain would hang PENDING forever the moment one arm fails. afterany means the
# consolidation always runs and reports what is missing instead of vanishing.
#
# Every stage self-skips existing outputs, so re-running this after a partial night is safe.
#
# Usage:
#   WAIT=3191948_0:3192096_0:3192310 bash submit_overnight_selector.sh   # consolidate after these too
#   DEP=<id>  chains the new GPU arms behind <id> as well (rarely wanted — SLURM schedules
#             them fine on its own, and serializing only makes the night longer)
#   STUB=1 bash submit_overnight_selector.sh                              # preview
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
E25="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"
RES="${E25}/results/router_leak"
LOG_DIR="${CKPT}/overnight_logs"
mkdir -p "${LOG_DIR}"
OUT_MD="${SCRIPT_DIR}/reports/SELECTOR_AUDIT_OVERNIGHT.md"

echo "== selector_audit overnight campaign =="
[ -n "${DEP:-}" ]  && echo "   new arms chained behind: ${DEP}"
[ -n "${WAIT:-}" ] && echo "   consolidation also waits for: ${WAIT}"

ids=""
add_dep() { [ -z "$1" ] || [ "$1" = "STUB" ] || ids="${ids:+${ids}:}$1"; }

# ── B: the three full-length CSAR arms, via the existing driver ──────────────────────────────
for spec in "name_stripped:centroid_sbert,key_tfidf" \
            "indirect:centroid_sbert,key_tfidf" \
            "none:centroid_sbert,random"; do
  qt="${spec%%:*}"; strats="${spec##*:}"
  echo "-- CSAR QPA=20 transform=${qt} strategies=${strats}"
  out="$(QPA=20 QT="${qt}" STRATS="${strats}" DEP="${DEP:-}" \
         bash "${SCRIPT_DIR}/submit_csar_audit.sh" all 2>&1)"
  echo "${out}" | sed 's/^/     /'
  add_dep "$(echo "${out}" | sed -n 's/^score job: *\([0-9]*\).*/\1/p')"
done

# ── C: consolidation, after everything ───────────────────────────────────────────────────────
ALLDEP="${ids}"
for extra in "${DEP:-}" "${WAIT:-}"; do
  [ -n "${extra}" ] && ALLDEP="${ALLDEP:+${ALLDEP}:}${extra}"
done

cons_body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=sa-consolidate
$(tofu_sbatch_resources 0 4 16G)
#SBATCH --time=00:40:00
#SBATCH --output=${LOG_DIR}/consolidate_%j.log
#SBATCH --error=${LOG_DIR}/consolidate_%j.log
set -eo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
echo "=== selector_audit consolidation ==="
date
# Re-derive the CPU-only analyses too, so the report reflects the code as of this run rather
# than whatever was last run by hand.
cd "${SCRIPT_DIR}"
${PYTHON} analyze_router_probe.py --self_test >/dev/null
${PYTHON} "${REPO_ROOT}/selector_audit/test_csar.py" >/dev/null
${PYTHON} "${REPO_ROOT}/selector_audit/consolidate.py" \\
  --pool_dir "${E25}" \\
  --expect "${RES}/csar_k200_f10_qpa20_name_stripped.json" \\
           "${RES}/csar_k200_f10_qpa20_indirect.json" \\
           "${RES}/csar_k200_f10_qpa20_centroid_sbert-random.json" \\
           "${E25}/results/router_leak/rl_family_k200_beh.json" \\
           "${E25}/results/smoke/routed_oracle_del_f10.json" \\
           "${E25}/results/smoke/routed_reroute_f10_s0.json" \\
  --out_md "${OUT_MD}" --out_json "${OUT_MD%.md}.json"
echo "--- report ---"
cat "${OUT_MD}"
date
EOF
}

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB: consolidation body (not submitted) -----" >&2
  cons_body >&2
  echo "   would chain on: ${ALLDEP:-<nothing>}" >&2
else
  cid="$(cons_body | sbatch --parsable ${ALLDEP:+--dependency=afterany:${ALLDEP}})"
  echo "-- consolidate job: ${cid} (afterany:${ALLDEP:-none})"
  echo
  echo "Report will be written to: ${OUT_MD}"
  echo "Check in the morning with:  squeue -u \$USER ; cat ${OUT_MD}"
fi
