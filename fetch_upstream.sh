#!/bin/bash
# Clone the three third-party repos this tree depends on, at the exact commits it was developed
# against, and apply the local patches that live in ou_integration/.
#
#   bash fetch_upstream.sh            # clone/checkout all three + apply the OU patches
#   bash fetch_upstream.sh --check    # report what is present and at which commit; write nothing
#
# WHY THESE ARE NOT VENDORED. S3T and MemSinks ship no LICENSE file (S3T shows an MIT badge in
# its README; MemSinks has nothing), so copying their source into this repo would redistribute
# code under terms nobody stated. Pinning the commit gives the same reproducibility with none of
# that. open-unlearning IS MIT, but it is a live fork with local commits, so a pin plus a patch
# is more honest than a snapshot that silently drifts from upstream.
#
# The clone dirs are .gitignore'd. Nothing here is committed.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=fetch
[ "${1:-}" = "--check" ] && MODE=check

# name | url | pinned commit | why this commit
UPSTREAMS=(
  "S3T|https://github.com/brcsomnath/S3T.git|3ecb73b86ff96f8c8eb38e2bd2afa30899a7400e|ICLR 2025 sliced-and-staged exact unlearning; the s3t_*.py repro follows this tree"
  "MemSinks|https://github.com/AR-FORUM/MemSinks.git|a00511991e2be2c3d9016585b3498b9d5ea914d8|memsinks_tofu/masks.py cites src/src/SeqTDModel.py:16-25 as its source of truth, and test_memsinks.py hashes it"
  "open-unlearning|https://github.com/locuslab/open-unlearning.git|93e9cd5d8808bb43641d133b38bb34466f9aae2e|the eval harness; this commit carries the offline-cache fix (cache_dir must be HF_HOME/hub) and the MemAdapt registry"
)

rc=0
OU_PIN_UNREACHABLE=0
for entry in "${UPSTREAMS[@]}"; do
  IFS='|' read -r name url pin why <<< "${entry}"
  dest="${REPO_ROOT}/${name}"

  if [ "${MODE}" = check ]; then
    if [ -d "${dest}/.git" ]; then
      have="$(git -C "${dest}" rev-parse HEAD 2>/dev/null)"
      if [ "${have}" = "${pin}" ]; then
        printf "  %-18s OK      %s\n" "${name}" "${pin:0:7}"
      else
        printf "  %-18s DRIFT   have %s, want %s\n" "${name}" "${have:0:7}" "${pin:0:7}"; rc=1
      fi
    else
      printf "  %-18s MISSING (run without --check)\n" "${name}"; rc=1
    fi
    continue
  fi

  if [ ! -d "${dest}/.git" ]; then
    echo "==> cloning ${name}  (${why})"
    git clone --quiet "${url}" "${dest}" || { echo "    clone FAILED" >&2; rc=1; continue; }
  fi
  echo "==> ${name}: checking out ${pin:0:7}"
  git -C "${dest}" fetch --quiet origin 2>/dev/null
  if ! git -C "${dest}" checkout --quiet "${pin}" 2>/dev/null; then
    # Distinguish "our clone is stale" from "the remote has never heard of this commit". Only
    # the first is fixable by fetching; the second means the pin names an object that is not on
    # the remote at all — a fork-local commit that was never pushed. Saying so is the difference
    # between a user retrying and a user knowing the pin is unrecoverable from here.
    if git -C "${dest}" fetch --quiet origin "${pin}" 2>/dev/null &&
       git -C "${dest}" checkout --quiet "${pin}" 2>/dev/null; then
      : # it was just missing locally
    else
      echo "    checkout FAILED — ${pin:0:7} is NOT ON THE REMOTE (${url})." >&2
      echo "    The remote answers 'not our ref', so this is a fork-local commit that was never" >&2
      echo "    pushed, not a stale clone. Nothing in this repo can reconstruct it." >&2
      [ "${name}" = open-unlearning ] && OU_PIN_UNREACHABLE=1
      rc=1; continue
    fi
  fi
done

[ "${MODE}" = check ] && exit "${rc}"

# ── open-unlearning local patches ────────────────────────────────────────────────────────────
# The sepmlp registry was never committed to the fork — it existed only as a dirty working tree
# on the original cluster. ou_integration/patches/ is that state, captured. Without this step
# `--model SepMlp-Llama-3.2-1B` raises a registry KeyError and the whole OU track is unrunnable.
OU="${REPO_ROOT}/open-unlearning"
P="${REPO_ROOT}/ou_integration/patches"
if [ "${OU_PIN_UNREACHABLE}" = 1 ]; then
  # The patch context assumes the fork (its second hunk anchors on the fork-only
  # `from model.memadapt_registry import …` lines). Installing sepmlp_registry.py and the config
  # into an upstream-main checkout would leave a half-patched tree that LOOKS integrated and
  # fails later, inside an eval. Refuse, and say what is missing.
  echo "==> SKIPPING ou_integration patches — open-unlearning is not at its pinned commit." >&2
  echo "    ${P}/model__init__.diff expects the fork's src/model/__init__.py (memadapt registry)," >&2
  echo "    which upstream main does not have. The OU eval track (the Table-1 Agg/Priv rows) is" >&2
  echo "    NOT reproducible from this repo until that fork commit is recovered. See STATUS.md." >&2
elif [ -d "${OU}/.git" ] && [ -d "${P}" ]; then
  echo "==> applying ou_integration patches"
  install -m 644 "${P}/sepmlp_registry.py"           "${OU}/src/model/sepmlp_registry.py"
  install -m 644 "${P}/SepMlp-Llama-3.2-1B.yaml"     "${OU}/configs/model/SepMlp-Llama-3.2-1B.yaml"
  if git -C "${OU}" apply --check "${P}/model__init__.diff" 2>/dev/null; then
    git -C "${OU}" apply "${P}/model__init__.diff" && echo "    src/model/__init__.py patched"
  else
    echo "    src/model/__init__.py: patch does not apply cleanly (already applied, or upstream" \
         "moved). Apply by hand: ${P}/model__init__.diff" >&2
  fi
  # Each project ships its own registry installer for its own arm.
  for proj in sepmlp_tofu blocktc_tofu memadapt_tofu; do
    inst="${REPO_ROOT}/${proj}/ou_integration/install_branch.sh"
    [ -f "${inst}" ] && echo "    ${proj}: run OU_DIR=${OU} bash ${inst}"
  done
fi

echo
echo "Done. Verify with: bash fetch_upstream.sh --check"
exit "${rc}"
