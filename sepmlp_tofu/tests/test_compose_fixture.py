"""Composition-fixture gate (plan CPU gate 17, offline half).

eval_compose.py is consumed UNCHANGED from memadapt_tofu (checkpoint-agnostic
by design — it only reads TOFU_EVAL.json files), so what needs pinning here is
the CONTRACT: our OU evals must emit every metric key eval_compose consumes,
and composition over that surface must yield finite Table-1 rows. We build
synthetic TOFU_EVAL.json files carrying exactly that surface, with values
harvested (read-only) from the real calib_base eval when /storage2 is mounted
(hardcoded realistic fallbacks otherwise), jittered per role so every ratio in
compose() is exercised away from the trivial 1.0.

Also re-runs `eval_compose.py --self_check` as a subprocess (stdlib-only
script, reads the canonical anchor logs) — the drift alarm for the shared
composition code.
"""

import json
import math
import os
import random
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sepmlp_common import MEMADAPT_DIR  # noqa: E402

if MEMADAPT_DIR not in sys.path:
    sys.path.insert(0, MEMADAPT_DIR)

import eval_compose  # noqa: E402


# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

CALIB_BASE = os.path.join(os.environ["TOFU_CKPT_STORE"], "memadapt_tofu", "evals",
                          "calib_base", "TOFU_EVAL.json")
EVAL_COMPOSE = os.path.join(MEMADAPT_DIR, "eval_compose.py")
TEST_ENV_PY = os.environ.get("TOFU_PYTHON", sys.executable)

# Every agg_value key compose() touches, enumerated FROM eval_compose itself so
# this test cannot silently drift from the consumer.
AGG_KEYS = (
    list(eval_compose.UTIL_R_KEYS)
    + list(eval_compose.UTIL_G_KEYS)
    + list(eval_compose.MIA_ATTACKS)
    + ["extraction_strength", "exact_memorization",
       "forget_Q_A_Prob",          # mem_verbatim variant
       "forget_Q_A_PARA_Prob",     # mem primary + TR_pm per-index
       "forget_Q_A_PERT_Prob"]     # TR_pm per-index
)

# Realistic agg values harvested read-only from calib_base on 2026-07-20;
# fallback so the gate still runs when /storage2 is not mounted.
FALLBACK_AGGS = {
    "retain_Q_A_Prob": 0.0867, "retain_Q_A_ROUGE": 0.3304,
    "retain_Truth_Ratio": 0.1910,
    "ra_Q_A_Prob_normalised": 0.3374, "ra_Q_A_ROUGE": 0.9353,
    "ra_Truth_Ratio": 0.4325,
    "wf_Q_A_Prob_normalised": 0.3604, "wf_Q_A_ROUGE": 0.8989,
    "wf_Truth_Ratio": 0.5112,
    "mia_loss": 0.3203, "mia_zlib": 0.2669, "mia_min_k": 0.3189,
    "mia_min_k_plus_plus": 0.4442,
    "extraction_strength": 0.0546, "exact_memorization": 0.5306,
    "forget_Q_A_Prob": 0.0918, "forget_Q_A_PARA_Prob": 0.0577,
    "forget_Q_A_PERT_Prob": 0.0518,
}

N_INDEX = 12  # per-index rows in the synthetic fixture (real evals carry 400)


def _load_calib():
    try:
        with open(CALIB_BASE) as f:
            return json.load(f)
    except OSError:
        return None


def _jitter(v, rng):
    """Multiplicative +-10%, clamped inside (0, 1): keeps every 1-x term of
    mem_score positive (statistics.harmonic_mean rejects negatives)."""
    return min(0.999, max(1e-3, v * rng.uniform(0.9, 1.1)))


def make_eval(path, seed):
    rng = random.Random(seed)
    calib = _load_calib()
    aggs = ({k: float(calib[k]["agg_value"]) for k in AGG_KEYS}
            if calib is not None else dict(FALLBACK_AGGS))
    ev = {k: {"agg_value": _jitter(v, rng)} for k, v in aggs.items()}

    if calib is not None:
        para_src = calib["forget_Q_A_PARA_Prob"]["value_by_index"]
        pert_src = calib["forget_Q_A_PERT_Prob"]["value_by_index"]
        keys = list(para_src.keys())[:N_INDEX]
        para = {k: {"prob": _jitter(float(para_src[k]["prob"]), rng)}
                for k in keys}
        pert = {k: {"prob": [_jitter(float(p), rng)
                             for p in pert_src[k]["prob"]]}
                for k in keys}
    else:
        keys = [str(i) for i in range(N_INDEX)]
        para = {k: {"prob": _jitter(0.06, rng)} for k in keys}
        pert = {k: {"prob": [_jitter(0.04, rng) for _ in range(5)]}
                for k in keys}
    ev["forget_Q_A_PARA_Prob"]["value_by_index"] = para
    ev["forget_Q_A_PERT_Prob"]["value_by_index"] = pert

    with open(path, "w") as f:
        json.dump(ev, f)
    return str(path)


def test_fixture_carries_every_consumed_key(tmp_path):
    ev = eval_compose.load_eval(make_eval(tmp_path / "TOFU_EVAL.json", seed=42))
    for k in AGG_KEYS:
        assert isinstance(ev[k]["agg_value"], float), k
    para = ev["forget_Q_A_PARA_Prob"]["value_by_index"]
    pert = ev["forget_Q_A_PERT_Prob"]["value_by_index"]
    assert para.keys() == pert.keys() and len(para) == N_INDEX
    for k in para:
        assert isinstance(para[k]["prob"], float)
        assert isinstance(pert[k]["prob"], list) and pert[k]["prob"]


def test_synthetic_composition_all_finite(tmp_path):
    model_ev = eval_compose.load_eval(make_eval(tmp_path / "model.json", 1))
    ft_ev = eval_compose.load_eval(make_eval(tmp_path / "ft.json", 2))
    retain_ref = eval_compose.load_eval(make_eval(tmp_path / "retain.json", 3))
    base_ev = eval_compose.load_eval(make_eval(tmp_path / "base.json", 4))

    row = eval_compose.compose(model_ev, ft_ev, retain_ref, base_ev)
    expected = {"util_r", "util_g", "util_g_raw", "mem", "priv", "agg",
                "mem_verbatim", "priv_absdiff"}
    assert set(row) == expected
    for k, v in row.items():
        assert isinstance(v, float) and math.isfinite(v), (k, v)

    # no-base variant: util_g/agg are None by contract, everything else finite
    row2 = eval_compose.compose(model_ev, ft_ev, retain_ref, None)
    assert row2["util_g"] is None and row2["agg"] is None
    for k in expected - {"util_g", "agg"}:
        assert math.isfinite(row2[k]), (k, row2[k])


def test_self_composition_identities(tmp_path):
    """compose(ev, ev, ev, ev) must hit the FT-row constructions exactly:
    Util.R = Util.G = 1 and Priv = 1 (identical AUCs -> min/max = 1)."""
    ev = eval_compose.load_eval(make_eval(tmp_path / "self.json", 5))
    row = eval_compose.compose(ev, ev, ev, ev)
    assert abs(row["util_r"] - 1.0) < 1e-12
    assert abs(row["util_g"] - 1.0) < 1e-12
    assert abs(row["priv"] - 1.0) < 1e-12
    assert math.isfinite(row["agg"])


@pytest.mark.skipif(
    not (os.path.exists(f"{eval_compose.EVAL_REFS}/full_eval.json")
         and os.path.exists(f"{eval_compose.EVAL_REFS}/retain90_eval.json")),
    reason="canonical eval refs not mounted (/storage2)",
)
def test_eval_compose_self_check_passes():
    py = TEST_ENV_PY if os.path.exists(TEST_ENV_PY) else sys.executable
    r = subprocess.run([py, EVAL_COMPOSE, "--self_check"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "self-check PASSED" in r.stdout, r.stdout
