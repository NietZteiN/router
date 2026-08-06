"""Registry shim for open-unlearning (branch memadapt-eval).

Installed as src/model/memadapt_registry.py; src/model/__init__.py registers
the class (see install_branch.sh). The class itself lives in the memadapt_tofu
project so training and eval share the exact routing code — never copy it.
"""
import os

import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMADAPT_DIR = os.environ.get("MEMADAPT_TOFU_DIR", os.path.join(_REPO_ROOT, "memadapt_tofu"))
if MEMADAPT_DIR not in sys.path:
    sys.path.insert(0, MEMADAPT_DIR)

from memadapt_model import MemAdaptLlamaForCausalLM  # noqa: E402,F401

