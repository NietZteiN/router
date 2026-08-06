"""Registry shim for open-unlearning (branch memadapt-eval, extended).

Installed as src/model/blocktc_registry.py; src/model/__init__.py registers
the class (see install_branch.sh). The class itself lives in the blocktc_tofu
project so training and eval share the exact transcoder code — never copy it.
"""
import os

import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLOCKTC_DIR = os.environ.get("BLOCKTC_TOFU_DIR", os.path.join(_REPO_ROOT, "blocktc_tofu"))
if BLOCKTC_DIR not in sys.path:
    sys.path.insert(0, BLOCKTC_DIR)

from tc_model import BlockTcLlamaForCausalLM  # noqa: E402,F401

