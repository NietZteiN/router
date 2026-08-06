"""Site-path resolution shared by every project in this repo.

The problem this solves. Before the export, 145 code files and 212 configs named `/storage2/jack`
outright — the scratch filesystem of one cluster. On any other machine that path does not exist,
so a config loads fine, a job submits fine, and the failure surfaces as a missing checkpoint
forty minutes into a run. Rewriting them to `${TOFU_CKPT_ROOT}/...` moves the cluster-specific
part into one file per site (cluster_env.<site>.sh).

The four variables a config may use, and what each means:

    HF_HOME            the HuggingFace cache (must contain hub/). Was /storage2/jack/data/huggingface
    TOFU_CKPT_ROOT     the tofu_sisa_lora checkpoint store. Was /storage2/jack/checkpoints/tofu_sisa_lora
    TOFU_CKPT_STORE    the PARENT of every project's store, so `${TOFU_CKPT_STORE}/ramole` reaches
                       a sibling project's checkpoints. Was /storage2/jack/checkpoints
    TOFU_DATA_ROOT     non-HF datasets (counterfact, lume, pistol). Was /storage2/jack/data

All four are exported by cluster_env.sh; `ensure_site_env()` pulls them in for a script run by
hand rather than through a submit_*.sh.

An UNSET variable is a HARD ERROR, not a silent literal. os.path.expandvars leaves "${FOO}"
untouched when FOO is undefined, and a path-shaped value then gets created on disk verbatim —
which is exactly what happened once in the source tree (a literal `${TOFU_CKPT_ROOT}/` directory
appeared, and is still sitting in apa-uniform-sum). Failing at config-load costs a second;
failing at checkpoint-write costs the run.

Usage from any project:

    import os, sys
    _REPO_ROOT = os.path.dirname(os.path.abspath(__file__))   # flat-layout anchor
    sys.path.insert(0, _REPO_ROOT)
    from repo_env import expand_paths, ensure_site_env
    ensure_site_env()
    cfg = expand_paths(json.load(open(path)))
"""
from __future__ import annotations
import os

import re
import subprocess

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_UNEXPANDED = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
_LOADED = False

# Site variables a config is allowed to reference. Anything else unresolved is a typo.
SITE_VARS = ("HF_HOME", "TOFU_CKPT_ROOT", "TOFU_CKPT_STORE", "TOFU_DATA_ROOT",
             "TOFU_STORAGE_ROOT")


def ensure_site_env(force: bool = False) -> dict:
    """Populate os.environ from cluster_env.<site>.sh. Idempotent; never clobbers an override.

    A script launched by a submit_*.sh inherits these because the driver sourced the site file.
    A script run by hand does not — and re-declaring the values in Python would create a second
    source of truth that drifts. So: shell out once, import what it exports.
    """
    global _LOADED
    if _LOADED and not force:
        return {}
    added: dict[str, str] = {}
    script = os.path.join(_REPO_ROOT, "cluster_env.sh")
    if os.path.exists(script):
        try:
            out = subprocess.run(
                ["bash", "-c", f'source "{script}" >/dev/null 2>&1; '
                               'env | grep -E "^(TOFU_|HF_HOME=)"'],
                capture_output=True, text=True, timeout=30)
            for line in out.stdout.splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k not in os.environ:
                    os.environ[k] = v
                    added[k] = v
        except Exception:
            pass                                  # a missing site file is reported at expand time
    _LOADED = True
    return added


def expand_paths(obj, _key: str = ""):
    """Recursively expand ${VAR} and ~ in a loaded config. Absolute paths pass through untouched,
    so every pre-existing config is unaffected."""
    if isinstance(obj, str):
        out = os.path.expanduser(os.path.expandvars(obj))
        if "$" in out and _UNEXPANDED.search(out) and not _key.startswith("_"):
            var = _UNEXPANDED.search(out).group(0).strip("${}")
            hint = (f"{var} is set by cluster_env.<site>.sh — export TOFU_SITE and the values it "
                    f"requires (see PORTING.md)." if var in SITE_VARS
                    else f"{var} is not a site variable; expected one of {', '.join(SITE_VARS)}.")
            raise SystemExit(f"config key {_key!r}: unresolved variable in {out!r}. {hint}")
        return out
    if isinstance(obj, list):
        return [expand_paths(v, _key) for v in obj]
    if isinstance(obj, dict):
        return {k: expand_paths(v, k) for k, v in obj.items()}
    return obj
