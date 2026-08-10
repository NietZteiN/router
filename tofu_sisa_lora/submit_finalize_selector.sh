#!/bin/bash
# Finalize the selector_audit campaign — the analysis that reads whatever the GPU arms produced.
# (thread log/selector_audit/)
#
# The GPU arms already survive a logoff; SLURM runs them on compute nodes. What did NOT exist
# was anything to READ their output, so the campaign would end with npz on disk and no answers.
# This is that step, queued rather than left for a human: chained `afterany` on the arms, it
# turns every matrix that landed into a report.
#
#   1. H18 — analyze_router_probe on every behavioral npz that exists, gold-form AND
#      name-stripped, for all three k=200 pools. This is the question the r32 arms are running
#      to answer: does "detection is lexical" hold beyond the weakest pool?
#   2. magnet saturation + RDR on the same matrices.
#   3. consolidate.py over every arm that landed.
#
# Robustness rules, because nobody is watching:
#   * every step is guarded and its exit status RECORDED, never swallowed — one missing input
#     must not stop the remaining analyses, and a failure must still be visible in the report;
#   * the job exits 0 so the report is always written, with failures listed inside it;
#   * everything self-skips on re-run, so this can be resubmitted after a partial night.
#
# Usage: DEP=<jobids colon-separated> bash submit_finalize_selector.sh   # STUB=1 previews
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"          # never build a job body before this line
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"
CKPT="${TOFU_CKPT_ROOT}"
E25="${CKPT}/Llama-2-7B-chat-hf_k200_r32_e25_lr1e4"
LOG_DIR="${CKPT}/overnight_logs"
REPORTS="${SCRIPT_DIR}/reports"
mkdir -p "${LOG_DIR}" "${REPORTS}"

body() {
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=sa-finalize
$(tofu_sbatch_resources 0 4 24G)
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/finalize_%j.log
#SBATCH --error=${LOG_DIR}/finalize_%j.log
set -uo pipefail                # NOT -e: a missing input must not abort the remaining analyses
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_SITE="\${TOFU_SITE:-cispa}"
cd "${SCRIPT_DIR}"
echo "=== selector_audit finalize ==="; date
FAILS=""
run() {  # run() <label> <cmd...>
  local label="\$1"; shift
  echo "--- \${label}"
  if "\$@"; then echo "    ok"; else echo "    FAILED (rc=\$?)"; FAILS="\${FAILS} \${label}"; fi
}

# 1. H18/H22 — the probe on every matrix that exists: BOTH families x THREE transforms x all
#    three pools. The family suffix is part of the npz stem ("_beh" or empty for feature-space)
#    and must also be part of the REPORT name — writing both families to probe_<pool>.json would
#    have one silently overwrite the other.
for pool in Llama-2-7B-chat-hf_k200_r32_e25_lr1e4 \\
            Llama-2-7B-chat-hf_k200_r32_e5_lr1e4 \\
            Llama-2-7B-chat-hf_k200_r8_e5_lr1e4; do
  RL="${CKPT}/\${pool}/results/router_leak"
  for fam in "_beh:beh" ":feat"; do
    fam_stem="\${fam%%:*}"; fam_tag="\${fam##*:}"
    for variant in "" "_name_stripped" "_indirect"; do
      stem="rl_family_k200\${fam_stem}\${variant}"
      # a strategy npz is <stem>.<strategy>.npz; skip the cell entirely if none landed.
      # NOTE the guard has to exclude the OTHER family's files: for fam_stem="" the glob
      # rl_family_k200.*.npz does not match rl_family_k200_beh.*.npz (the separator differs),
      # which is why the feature stem is the empty one and not a shared prefix.
      if ls "\${RL}/\${stem}."*.npz >/dev/null 2>&1; then
        run "probe \${fam_tag} \${pool}\${variant:-_goldform}" \\
          ${PYTHON} analyze_router_probe.py --family_npz "\${RL}/\${stem}.*.npz" \\
            --drop_set 180-199 \\
            --out_json "${REPORTS}/probe_\${fam_tag}_\${pool}\${variant}.json" \\
            --out_md   "${REPORTS}/probe_\${fam_tag}_\${pool}\${variant}.md"
        run "seqdel \${fam_tag} \${pool}\${variant:-_goldform}" \\
          ${PYTHON} analyze_sequential_deletion.py --family_npz "\${RL}/\${stem}.*.npz" \\
            --delete_order 180-199 \\
            --out_json "${REPORTS}/seqdel_\${fam_tag}_\${pool}\${variant}.json" \\
            --out_md   "${REPORTS}/seqdel_\${fam_tag}_\${pool}\${variant}.md"
      else
        echo "--- probe \${fam_tag} \${pool}\${variant:-_goldform}: no npz, skipping"
      fi
    done
  done
done

# 2. everything that landed, in one report
run "consolidate" ${PYTHON} "${REPO_ROOT}/selector_audit/consolidate.py" \\
  --pool_dir "${E25}" \\
  --extra_pool "${CKPT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4" \\
               "${CKPT}/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4" \\
  --out_md "${REPORTS}/SELECTOR_AUDIT_OVERNIGHT.md" \\
  --out_json "${REPORTS}/SELECTOR_AUDIT_OVERNIGHT.json"

echo
echo "=== H18/H22: does 'detection is lexical' hold across pools, families and transforms? ==="
for f in ${REPORTS}/probe_beh_*.md ${REPORTS}/probe_feat_*.md; do
  [ -f "\$f" ] || continue
  echo "--- \$(basename "\$f")"
  # write_md emits "| activation_norm | 200 | **0.972** | ..." — NO backticks around the
  # strategy. The backticked pattern this used to carry matched nothing, so every run printed
  # bare filenames and the summary looked empty rather than broken.
  grep -E "^\| (ppl|activation_norm|attn_norm|logit_div) \|" "\$f" 2>/dev/null | head -4 || true
done

echo
if [ -n "\${FAILS}" ]; then echo "STEPS THAT FAILED:\${FAILS}"; else echo "all steps ok"; fi
echo "report: ${REPORTS}/SELECTOR_AUDIT_OVERNIGHT.md"
date
exit 0                          # always exit clean: the report is the deliverable
EOF
}

if [ "${STUB:-0}" = "1" ]; then
  echo "----- STUB (not submitted) -----" >&2
  body >&2
else
  jid="$(body | sbatch --parsable ${DEP:+--dependency=afterany:${DEP}})"
  echo "finalize job: ${jid}${DEP:+ (afterany:${DEP})}"
  echo "report will be at: ${REPORTS}/SELECTOR_AUDIT_OVERNIGHT.md"
fi
