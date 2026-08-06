# SLURM policy — now a SHIM over cluster_env.sh (2026-07-28).
#
# WAS: this file hardcoded the sprint cluster (sprint1/2/3, never sprint4, the `all` partition,
# a 4-GPU global cap). 57 submit_*.sh source it, so it stays — but the real settings moved to
# cluster_env.<site>.sh so that porting to another cluster is a new site file rather than a
# patch across 57 drivers. The pattern is taken from
# merge-tables-7b/tofu_sisa_lora/cluster_env.sh, the CISPA A100 port.
#
# On the sprint site this exports EXACTLY the same six variables, with the same values, as it
# did before — pinned by test_cluster_env.py so untouched drivers cannot regress.
#
# Pick a site with TOFU_SITE=sprint|cispa (default: auto-detected from the hostname).
# New drivers should call `tofu_sbatch_resources` rather than writing #SBATCH lines by hand;
# that is what makes "no --mem on this cluster" a site fact instead of a per-driver edit.

_TOFU_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${_TOFU_ENV_DIR}/cluster_env.sh"

# Legacy names the unported drivers still read. Each is set by the site file; re-exported here
# so nothing else has to know cluster_env.sh exists.
export TOFU_EXCLUDE TOFU_ALLOWED_NODES TOFU_ARRAY_CAP
export TOFU_SMOKE_TIME TOFU_EXTENDED_TIME TOFU_EXTENDED_TIME_3B
