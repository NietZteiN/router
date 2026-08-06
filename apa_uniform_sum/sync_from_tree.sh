#!/bin/bash
# One-directional, allow-listed vendoring from the working tree into this repo.
#
#   bash sync_from_tree.sh --check     # report drift, write nothing (default)
#   bash sync_from_tree.sh --pull      # copy tree -> repo for UNEDITED entries only
#   bash sync_from_tree.sh --pull --force-edited   # also overwrite entries marked `edited:`
#
# Direction is always tree -> repo. There is deliberately no --push: the working tree is where
# the campaign runs, and pushing this repo's edits back wholesale is exactly the mistake
# merge-tables-7b/reproduce/VENDOR_DRIFT.md documents ("a blind sync would overwrite the CISPA
# port and break every driver there"). Move a change home by hand, one file at a time.
#
# Statuses:
#   SAME     identical to the tree
#   EDITED   differs, and MANIFEST.files marks it `edited:<reason>` — expected, not a problem
#   DRIFT    differs with NO edited: marker — the tree moved and this copy did not. Investigate.
#   MISSING  present in the manifest, absent here
#   NOSRC    absent from the tree (a file this repo owns, or a path that moved upstream)
# Exit 1 if any DRIFT or MISSING is found, so this can gate CI.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOFU_TREE="${TOFU_TREE:-${TOFU_TREE}}"
MANIFEST="${REPO_DIR}/MANIFEST.files"

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
[ -d "${TOFU_TREE}" ] || { echo "TOFU_TREE=${TOFU_TREE} does not exist" >&2; exit 2; }

n_same=0; n_edited=0; n_drift=0; n_missing=0; n_nosrc=0; n_copied=0
drift_list=(); missing_list=()

# A trailing slash in the manifest means "this directory, recursively". Both paths stay
# TREE-RELATIVE here; only compare_one/tally prepend the roots.
sync_dir() {
  local dest="$1" src="$2" marker="$3"
  local rel status
  while IFS= read -r -d '' f; do
    rel="${f#"${TOFU_TREE}/${src}"}"
    status="$(compare_one "${dest}${rel}" "${src}${rel}" "${marker}")"
    tally "${status}" "${dest}${rel}" "${src}${rel}" "${marker}"
  done < <(find "${TOFU_TREE}/${src}" -type f -print0 2>/dev/null)
}

compare_one() {
  local dest_abs="${REPO_DIR}/$1" src_abs="${TOFU_TREE}/$2" marker="$3"
  [ -f "${src_abs}" ] || { echo NOSRC; return; }
  [ -f "${dest_abs}" ] || { echo MISSING; return; }
  if cmp -s "${dest_abs}" "${src_abs}"; then echo SAME
  elif [ -n "${marker}" ];               then echo EDITED
  else                                        echo DRIFT; fi
}

tally() {
  local status="$1" dest="$2" src="$3" marker="$4"
  case "${status}" in
    SAME)    n_same=$((n_same+1)) ;;
    EDITED)  n_edited=$((n_edited+1)); printf '  EDITED  %-34s (%s)\n' "${dest}" "${marker}" ;;
    DRIFT)   n_drift=$((n_drift+1));   drift_list+=("${dest}") ;;
    MISSING) n_missing=$((n_missing+1)); missing_list+=("${dest}") ;;
    NOSRC)   n_nosrc=$((n_nosrc+1)) ;;
  esac
  if [ "${MODE}" = pull ] && { [ "${status}" = DRIFT ] || [ "${status}" = MISSING ] \
       || { [ "${status}" = EDITED ] && [ "${FORCE_EDITED}" = 1 ]; }; }; then
    mkdir -p "$(dirname "${REPO_DIR}/${dest}")"
    cp -p "${TOFU_TREE}/${src}" "${REPO_DIR}/${dest}"
    n_copied=$((n_copied+1)); printf '  pulled  %s\n' "${dest}"
  fi
}

echo "sync_from_tree.sh (${MODE}) — tree=${TOFU_TREE}"
while read -r dest src marker rest; do
  [ -z "${dest:-}" ] && continue
  case "${dest}" in \#*) continue ;; esac
  [ -z "${src:-}" ] && continue          # a file this repo owns: no source column
  case "${marker:-}" in edited:*) marker="${marker#edited:}" ;; *) marker="" ;; esac
  if [ "${dest: -1}" = "/" ]; then
    sync_dir "${dest%/}" "${src%/}" "${marker}"
  else
    st="$(compare_one "${dest}" "${src}" "${marker}")"
    tally "${st}" "${dest}" "${src}" "${marker}"
  fi
done < "${MANIFEST}"

echo
printf 'same %d · edited %d · drift %d · missing %d · no-source %d' \
  "${n_same}" "${n_edited}" "${n_drift}" "${n_missing}" "${n_nosrc}"
[ "${MODE}" = pull ] && printf ' · copied %d' "${n_copied}"
echo
if [ "${n_drift}" -gt 0 ]; then
  echo; echo "DRIFT — the tree moved and this copy did not:"
  printf '  %s\n' "${drift_list[@]}"
  echo "  review with: diff <(cat ${TOFU_TREE}/tofu_sisa_lora/<f>) <f>   then --pull"
fi
if [ "${n_missing}" -gt 0 ]; then
  echo; echo "MISSING here:"; printf '  %s\n' "${missing_list[@]}"
fi
[ "${MODE}" = check ] && { [ "${n_drift}" -gt 0 ] || [ "${n_missing}" -gt 0 ]; } && exit 1
exit 0
