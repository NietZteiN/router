# Stage the HF cache onto node-local disk, then point HF_HOME at it.
#
# WHY (measured on CISPA, 2026-07-24): when $HF_HOME lives on NFS, a cold 13.5 GB Llama-2-7B
# load took ~9 minutes with 12 readers in flight (processes in state D, wchan
# rpc_wait_bit_killable) against ~12 s of actual per-author training. That is ~98 % I/O overhead
# on every train AND every eval — roughly 13 h of pure waiting across a full campaign. One copy
# per node turns every later load into a local read.
#
# Usage (inside a worker script, before invoking python):
#     source "${SCRIPT_DIR}/stage_hf_cache.sh"   # exports HF_HOME to the local copy
#
# Opt out with TOFU_STAGE_HF=0. Harmless on a node whose HF_HOME is already local disk — but
# pointless, so leave it off there.
#
# Ported from merge-tables-7b/tofu_sisa_lora/stage_hf_cache.sh with two fixes:
#   1. the copy is now failure-CHECKED. The original ran every `cp -r ... 2>/dev/null` with its
#      status discarded and then unconditionally `touch`ed the completion marker, so a partial
#      copy (disk full, killed mid-copy) was recorded as complete and every later task on that
#      node silently read a truncated cache.
#   2. the asset list is derived from $TOFU_STAGE_MODEL / $TOFU_STAGE_DATASETS instead of being
#      hardcoded to Llama-2 + TOFU, so the Experiment C tiers (MMLU, Alpaca, DBpedia) stage too.
#      Missing optional assets are reported and skipped; a missing MODEL is fatal.

tofu_stage_hf_cache() {
  [ "${TOFU_STAGE_HF:-1}" = "1" ] || { echo "[stage] disabled, using ${HF_HOME}"; return 0; }

  local src="${HF_HOME:?HF_HOME must be set before sourcing stage_hf_cache.sh}"
  local dst="${TOFU_LOCAL_HF:-/tmp/tofu_hf_${USER}}"
  local marker="${dst}/.stage_complete"
  local lock="/tmp/tofu_hf_${USER}.lock"

  # Fast path: already staged by an earlier task on this node.
  if [ -f "${marker}" ]; then
    export HF_HOME="${dst}"
    echo "[stage] reusing node-local cache ${dst}"
    return 0
  fi

  local model="${TOFU_STAGE_MODEL:-meta-llama/Llama-2-7B-chat-hf}"
  # Datasets the three experiments touch. TOFU is required; the rest are Experiment C tiers and
  # are staged only if present in the source cache.
  local datasets="${TOFU_STAGE_DATASETS:-locuslab/TOFU cais/mmlu tatsu-lab/alpaca fancyzhx/dbpedia_14}"

  mkdir -p "${dst}"
  # -w 1200 (20 min) is generous for a ~14 GB copy but BOUNDED, so a wedged peer degrades this
  # to a slow NFS read instead of hanging the whole array.
  (
    flock -x -w 1200 9 || { echo "[stage] lock timeout, falling back to NFS"; exit 1; }
    if [ -f "${marker}" ]; then exit 0; fi          # a peer finished while we waited

    echo "[stage] copying HF cache -> ${dst} (one node-local copy, $(date))"
    mkdir -p "${dst}/hub" "${dst}/datasets" || exit 1

    # Required: the base model. A failure here must abort staging.
    local mdir="models--${model//\//--}"
    if [ -d "${src}/hub/${mdir}" ]; then
      cp -r "${src}/hub/${mdir}" "${dst}/hub/" || { echo "[stage] FAILED copying ${mdir}"; exit 1; }
    else
      echo "[stage] ${src}/hub/${mdir} absent — cannot stage"; exit 1
    fi

    # Optional: datasets, in both the hub/ and the Arrow datasets/ layouts.
    local d hubname arrowname
    for d in ${datasets}; do
      hubname="datasets--${d//\//--}"
      arrowname="$(printf '%s' "${d}" | tr '/' '\n' | paste -sd'_' - | tr '[:upper:]' '[:lower:]')"
      arrowname="${arrowname/_/___}"          # HF Arrow dirs use org___name
      if [ -d "${src}/hub/${hubname}" ]; then
        cp -r "${src}/hub/${hubname}" "${dst}/hub/" || echo "[stage] WARN partial ${hubname}"
      fi
      if [ -d "${src}/datasets/${arrowname}" ]; then
        cp -r "${src}/datasets/${arrowname}" "${dst}/datasets/" || echo "[stage] WARN partial ${arrowname}"
      fi
    done

    # The gated-model token and any trust_remote_code modules.
    [ -f "${src}/token" ]   && cp    "${src}/token"   "${dst}/"
    [ -d "${src}/modules" ] && cp -r "${src}/modules" "${dst}/"
    # metrics_cache is deliberately NOT staged: it is per-job scratch, and eval_tofu points
    # TOFU_METRICS_CACHE at a job-scoped dir precisely so parallel tasks cannot collide.

    touch "${marker}"
    echo "[stage] done ($(du -sh "${dst}" 2>/dev/null | cut -f1), $(date))"
  ) 9>"${lock}"

  if [ -f "${marker}" ]; then
    export HF_HOME="${dst}"
    echo "[stage] HF_HOME=${HF_HOME}"
  else
    echo "[stage] staging failed; keeping HF_HOME=${HF_HOME}"
  fi
}

tofu_stage_hf_cache
