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
