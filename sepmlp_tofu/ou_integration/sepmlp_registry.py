"""Registry shim for open-unlearning (branch memadapt-eval, extended).

Installed as src/model/sepmlp_registry.py; src/model/__init__.py registers
the class (see install_branch.sh). The class itself lives in the sepmlp_tofu
project so training and eval share the exact bank code — never copy it.
"""
import os

import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEPMLP_DIR = os.environ.get("SEPMLP_TOFU_DIR", os.path.join(_REPO_ROOT, "sepmlp_tofu"))
if SEPMLP_DIR not in sys.path:
    sys.path.insert(0, SEPMLP_DIR)

from sepmlp_model import SepMlpLlamaForCausalLM  # noqa: E402,F401

