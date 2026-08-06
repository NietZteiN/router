#!/bin/bash
# One-directional, allow-listed vendoring from the working trees into this repo.
#
#   bash sync_from_tree.sh --check                 # report drift, write nothing (default)
#   bash sync_from_tree.sh --pull                  # copy tree -> repo, UNEDITED entries only
#   bash sync_from_tree.sh --pull --force-edited   # also overwrite entries marked `edited:`
#
# Direction is always tree -> repo. There is deliberately no --push. The working tree is where
# the campaign runs, and pushing this repo's edits back wholesale is exactly the mistake
# merge-tables-7b/reproduce/VENDOR_DRIFT.md documents ("a blind sync would overwrite the CISPA
# port and break every driver there"). Move a change home by hand, one file at a time.
#
# Statuses:
#   SAME     identical to the tree
#   EDITED   differs, and MANIFEST.files says why — expected, not a problem. Every .py/.sh here
#            is `edited:deabsolutized`, because the export rewrote /home/jack paths into
#            repo-relative lookups. That divergence is the whole point of this repo.
#   DRIFT    differs with NO reason recorded — the tree moved and this copy did not. Investigate.
#   MISSING  in the manifest, absent here
#   NOSRC    absent from the tree (a file this repo owns, or a path that moved upstream)
#   ONLYREPO under a manifest directory here but not in the source — usually generated output
# Exit 1 on any DRIFT or MISSING, so this can gate CI.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${REPO_DIR}/MANIFEST.files"

export TOFU_TREE="${TOFU_TREE:-/home/jack/tofu-unlearning}"
export TOFU_HOME="${TOFU_HOME:-/home/jack}"
export APA_TREE="${APA_TREE:-/home/jack/apa-uniform-sum}"
export MT7B_TREE="${MT7B_TREE:-/home/jack/merge-tables-7b}"

root_for() {
  case "$1" in
    TREE) echo "${TOFU_TREE}" ;;  HOME) echo "${TOFU_HOME}" ;;
    APA)  echo "${APA_TREE}"  ;;  MT7B) echo "${MT7B_TREE}" ;;
    *) echo "" ;;
  esac
}

MODE=check
FORCE_EDITED=0
for a in "$@"; do
  case "$a" in
    --check) MODE=check ;;
    --pull)  MODE=pull ;;
    --force-edited) FORCE_EDITED=1 ;;
    *) echo "usage: sync_from_tree.sh [--check|--pull] [--force-edited]" >&2; exit 2 ;;
  esac
done

[ -f "${MANIFEST}" ] || { echo "no ${MANIFEST}" >&2; exit 2; }

n_same=0; n_edited=0; n_drift=0; n_missing=0; n_nosrc=0; n_copied=0; n_onlyrepo=0
drift_list=(); missing_list=()

compare_one() {   # $1 dest-rel  $2 src-abs  $3 marker
  local dest_abs="${REPO_DIR}/$1" src_abs="$2" marker="$3"
  [ -f "${src_abs}" ] || { echo NOSRC; return; }
  [ -f "${dest_abs}" ] || { echo MISSING; return; }
  if cmp -s "${dest_abs}" "${src_abs}"; then echo SAME
  elif [ -n "${marker}" ];              then echo EDITED
  else                                       echo DRIFT
  fi
}

tally() {         # $1 status  $2 dest-rel  $3 src-abs  $4 marker
  case "$1" in
    SAME)     n_same=$((n_same+1)) ;;
    EDITED)   n_edited=$((n_edited+1)) ;;
    NOSRC)    n_nosrc=$((n_nosrc+1)) ;;
    ONLYREPO) n_onlyrepo=$((n_onlyrepo+1)) ;;
    DRIFT)    n_drift=$((n_drift+1)); drift_list+=("$2") ;;
    MISSING)  n_missing=$((n_missing+1)); missing_list+=("$2") ;;
  esac
  if [ "${MODE}" = pull ]; then
    case "$1" in
      MISSING) : ;;                                   # fall through to the copy below
      DRIFT)   : ;;
      EDITED)  [ "${FORCE_EDITED}" = 1 ] || return 0 ;;
      *)       return 0 ;;
    esac
    mkdir -p "$(dirname "${REPO_DIR}/$2")"
    cp -p "$3" "${REPO_DIR}/$2" && n_copied=$((n_copied+1))
  fi
}

# Paths that exist in a source tree but deliberately live SOMEWHERE ELSE here. Without this they
# would report as MISSING forever, which trains a reader to ignore the MISSING list — and the
# MISSING list is the one that catches a genuinely dropped file.
relocated_excludes() {
  # the manuscript: tofu_sisa_lora/paper/*.tex -> paper/tex/ ; the two AAAI PDFs -> paper/pdf/
  echo "-not -path */tofu_sisa_lora/paper/* -not -name p_unlearn_AAAI*"
}

# Directory entry: walk BOTH sides, so a file deleted upstream or generated here is visible.
sync_dir() {      # $1 dest-rel-dir(with /)  $2 src-abs-dir  $3 marker
  local dest="$1" src="$2" marker="$3" rel status
  while IFS= read -r -d '' f; do
    rel="${f#"${src}/"}"
    status="$(compare_one "${dest}${rel}" "${src}/${rel}" "${marker}")"
    tally "${status}" "${dest}${rel}" "${src}/${rel}" "${marker}"
  done < <(find "${src}" -type f \
             -not -path '*/.git/*' -not -path '*/__pycache__/*' -not -name '*.pyc' \
             -not -path '*/.pytest_cache/*' -not -path '*/checkpoints/*' \
             $(relocated_excludes) -print0 2>/dev/null)
  # Present here, absent upstream.
  while IFS= read -r -d '' f; do
    rel="${f#"${REPO_DIR}/${dest}"}"
    [ -f "${src}/${rel}" ] || tally ONLYREPO "${dest}${rel}" "" ""
  done < <(find "${REPO_DIR}/${dest}" -type f -not -path '*/__pycache__/*' -print0 2>/dev/null)
}

while read -r dest src marker _rest; do
  [ -z "${dest:-}" ] && continue
  case "${dest}" in \#*) continue ;; esac
  [ -z "${src:-}" ] && continue                     # owned file: no source, never synced
  root_key="${src%%:*}"; src_rel="${src#*:}"
  root="$(root_for "${root_key}")"
  [ -n "${root}" ] || { echo "unknown root '${root_key}' on line: ${dest}" >&2; continue; }
  case "${marker:-}" in edited:*) ;; *) marker="" ;; esac

  if [[ "${dest}" == */ ]]; then
    sync_dir "${dest}" "${root}/${src_rel%/}" "${marker}"
  else
    status="$(compare_one "${dest}" "${root}/${src_rel}" "${marker}")"
    tally "${status}" "${dest}" "${root}/${src_rel}" "${marker}"
  fi
done < <(grep -v '^\s*#' "${MANIFEST}" | grep -v '^\s*$')

echo "same ${n_same}  edited ${n_edited}  drift ${n_drift}  missing ${n_missing}  nosrc ${n_nosrc}  only-here ${n_onlyrepo}"
[ "${MODE}" = pull ] && echo "copied ${n_copied}"

if [ "${n_drift}" -gt 0 ]; then
  echo; echo "DRIFT (the tree moved, this copy did not — investigate before pulling):"
  printf '  %s\n' "${drift_list[@]}" | head -40
fi
if [ "${n_missing}" -gt 0 ]; then
  echo; echo "MISSING (in the manifest, absent here):"
  printf '  %s\n' "${missing_list[@]}" | head -40
fi

[ "${n_drift}" -eq 0 ] && [ "${n_missing}" -eq 0 ]
