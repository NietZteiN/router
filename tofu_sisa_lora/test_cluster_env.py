"""CPU gate for the site abstraction (2026-07-28). No SLURM, no network, no GPU.

The point of cluster_env.<site>.sh is that porting to another cluster is a new FILE rather than
a patch across the 57 submit_*.sh that source slurm_nodes.sh. Two things must hold for that to
be safe, and both are pinned here:

  1. On the `sprint` site the shim exports EXACTLY the six legacy variables, with exactly the
     values they had when they were hardcoded — otherwise every unported driver silently changes
     behaviour the moment this refactor lands.
  2. `tofu_sbatch_resources` honours the site's memory policy. On CISPA the nodes report
     RealMemory=1 and the partition sets DefMemPerNode=UNLIMITED, so ANY --mem line fails at
     SUBMIT time with "Memory specification can not be satisfied". Emitting one there would fail
     the whole campaign at the first sbatch, which is exactly the class of error a gate should
     catch on a laptop.

    python test_cluster_env.py
"""
from __future__ import annotations

import os
import subprocess
import sys


# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OK = "ok  "

# The values slurm_nodes.sh hardcoded before the refactor (captured from the live file).
SPRINT_LEGACY = {
    "TOFU_ALLOWED_NODES": "sprint1,sprint2,sprint3",
    "TOFU_ARRAY_CAP": "4",
    "TOFU_EXCLUDE": "sprint4",
    "TOFU_EXTENDED_TIME": "02:30:00",
    "TOFU_EXTENDED_TIME_3B": "03:30:00",
    "TOFU_SMOKE_TIME": "00:55:00",
}


def _sh(script, site=None):
    """Run `script` in a pristine env after sourcing slurm_nodes.sh for `site`."""
    env_prefix = f"TOFU_SITE={site} " if site else ""
    cmd = f"env -i {env_prefix}bash -c 'source {HERE}/slurm_nodes.sh; {script}'"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"site={site} script failed rc={r.returncode}: {r.stderr.strip()}")
    return r.stdout


def test_sprint_legacy_exports_unchanged():
    out = _sh("env | grep ^TOFU_ | sort", site="sprint")
    got = dict(l.split("=", 1) for l in out.strip().split("\n") if "=" in l)
    for k, v in SPRINT_LEGACY.items():
        assert got.get(k) == v, f"{k}: {got.get(k)!r} != legacy {v!r} — unported drivers regress"
    print(OK + f"sprint: all {len(SPRINT_LEGACY)} legacy exports byte-identical to the "
               f"pre-refactor hardcoded values")


def test_site_selection():
    assert _sh("echo $TOFU_SITE", site="sprint").strip() == "sprint"
    assert _sh("echo $TOFU_SITE", site="cispa").strip() == "cispa"
    assert _sh("echo $TOFU_PARTITION", site="sprint").strip() == "all"
    assert _sh("echo $TOFU_PARTITION", site="cispa").strip() == "xe8545"
    # env overrides the site file
    r = subprocess.run(
        f"env -i TOFU_SITE=cispa TOFU_ARRAY_CAP=2 bash -c "
        f"'source {HERE}/slurm_nodes.sh; echo $TOFU_ARRAY_CAP'",
        shell=True, capture_output=True, text=True)
    assert r.stdout.strip() == "2", f"env override ignored: {r.stdout!r}"
    print(OK + "site selection works and the environment still overrides the site file")


def test_memory_policy():
    """The one that would otherwise fail at the first sbatch on CISPA."""
    sprint = _sh("tofu_sbatch_resources 1 4 48G", site="sprint")
    assert "--mem=48G" in sprint, f"sprint must emit --mem:\n{sprint}"
    assert "--partition=all" in sprint and "--exclude=sprint4" in sprint
    assert "--gres=gpu:1" in sprint
    assert "--account" not in sprint, "sprint has no account; an empty --account is invalid"

    cispa = _sh("tofu_sbatch_resources 1 8 48G", site="cispa")
    assert "--mem" not in cispa, (
        "CISPA nodes report RealMemory=1 / DefMemPerNode=UNLIMITED — any --mem fails at submit "
        f"with 'Memory specification can not be satisfied':\n{cispa}")
    assert "--partition=xe8545" in cispa and "--account=testing" in cispa
    assert "xe8545-a100-" in cispa, "the drained-node exclude list must still be emitted"
    print(OK + "memory policy: sprint emits --mem, cispa drops it (and keeps account/exclude)")


def test_cpu_only_task_has_no_gres():
    for site in ("sprint", "cispa"):
        out = _sh("tofu_sbatch_resources 0 16 96G", site=site)
        assert "--gres" not in out, f"{site}: a CPU-only task must not request a GPU:\n{out}"
    print(OK + "gpus=0 emits no --gres on either site (CPU merges/norms stay off the GPU cap)")


def test_config_path_expansion():
    """A config may use ${TOFU_CKPT_ROOT}; absolute paths must be untouched."""
    import json
    import tempfile
    sys.path.insert(0, HERE)
    import merge_subset as MS
    os.environ["TOFU_CKPT_ROOT"] = "/somewhere/ckpts"
    cfg = {"model_name": "m", "shards_dir": "${TOFU_CKPT_ROOT}/pool", "out_dir": "/abs/stays",
           "n_ladder": [1], "subset_seeds": [42], "eval": {}, "nested": {"p": "${TOFU_CKPT_ROOT}/x"}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        p = f.name
    got = MS.load_config(p)
    os.unlink(p)
    assert got["shards_dir"] == "/somewhere/ckpts/pool", got["shards_dir"]
    assert got["out_dir"] == "/abs/stays", "absolute paths must pass through unchanged"
    assert got["nested"]["p"] == "/somewhere/ckpts/x", "expansion must recurse"
    print(OK + "config ${VAR} expansion works, recurses, and leaves absolute paths alone")


def test_real_configs_still_load():
    """Every shipped config must still parse after the expansion change."""
    import glob
    sys.path.insert(0, HERE)
    import merge_subset as MS
    n = 0
    for p in sorted(glob.glob(os.path.join(HERE, "configs", "nmerge_*.json"))):
        cfg = MS.load_config(p)
        assert cfg["shards_dir"].startswith("/"), f"{p}: unresolved path {cfg['shards_dir']!r}"
        n += 1
    print(OK + f"all {n} nmerge_*.json configs load and resolve to absolute paths")


def main():
    test_sprint_legacy_exports_unchanged()
    test_site_selection()
    test_memory_policy()
    test_cpu_only_task_has_no_gres()
    test_config_path_expansion()
    test_real_configs_still_load()
    print("\nALL test_cluster_env.py GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
