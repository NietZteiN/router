"""Make the shell-defined site settings visible to Python, without duplicating them.

cluster_env.<site>.sh is the single source of truth for TOFU_CKPT_ROOT / HF_HOME / partition /
interpreter. Scripts launched by a submit_*.sh inherit those because the driver sourced
slurm_nodes.sh — but a script run by hand (`python merge_subset.py plan --config ...`) does not,
and a config that says "${TOFU_CKPT_ROOT}/pool" would then expand to nothing.

Re-declaring the values here would create a second source of truth that silently drifts, so this
shells out to slurm_nodes.sh once and imports whatever it exports. Only keys absent from the
current environment are filled in, so an explicit override always wins.
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

_LOADED = False


def ensure_site_env(force: bool = False) -> dict:
    """Populate os.environ with the TOFU_*/HF_HOME exports from slurm_nodes.sh. Idempotent."""
    global _LOADED
    if _LOADED and not force:
        return {}
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "slurm_nodes.sh")
    added = {}
    if os.path.exists(script):
        try:
            out = subprocess.run(
                ["bash", "-c", f'source "{script}" >/dev/null 2>&1; '
                               f'env | grep -E "^(TOFU_|HF_HOME=)"'],
                capture_output=True, text=True, timeout=30)
            for line in out.stdout.splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k not in os.environ:          # never clobber an explicit override
                    os.environ[k] = v
                    added[k] = v
        except Exception as e:  # noqa: BLE001 — a missing/broken site file must not be fatal here
            print(f"[tofu_env] WARN could not source {script}: {type(e).__name__}: {e}")
    _LOADED = True
    return added


def hf_home() -> str:
    """Resolve HF_HOME: the environment first, else whatever cluster_env.<site>.sh exports.

    Every caller used to carry its own `os.environ.setdefault("HF_HOME", os.path.join(os.environ["TOFU_STORAGE_ROOT"], "..."))`.
    That default is a *different cluster's* path, so on any other machine it did not fail — it
    pointed HuggingFace at a directory that does not exist and the run died later with a
    confusing offline-cache miss. Resolving through the site file means an unset HF_HOME is a
    loud error at startup naming the file that should have set it.
    """
    v = os.environ.get("HF_HOME")
    if v:
        return v
    ensure_site_env()
    v = os.environ.get("HF_HOME")
    if not v:
        raise SystemExit(
            "HF_HOME is unset and cluster_env.${TOFU_SITE}.sh did not export it.\n"
            "  export HF_HOME=/path/to/huggingface   (the dir holding hub/ and token)\n"
            "  or set it in cluster_env.<site>.sh and select that site with TOFU_SITE=<site>.")
    return v
