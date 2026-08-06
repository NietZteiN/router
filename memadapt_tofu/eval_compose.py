"""Compose Grimes et al. Table-1 aggregate metrics from OU TOFU_EVAL.json files.

The four aggregates are NOT stock OpenUnlearning outputs; they were
reverse-engineered from Dorna et al. 2026 App. F.1 and validated numerically
against the canonical public eval logs (see log/memory_adapters/ opening
entry). Formulas (primary):

  Util.R = HM(retain_Q_A_Prob, retain_Q_A_ROUGE, retain_Truth_Ratio)
           / same(finetuned model)
  Util.G = HM(ra_Q_A_Prob_normalised, ra_Q_A_ROUGE, ra_Truth_Ratio,
              wf_Q_A_Prob_normalised, wf_Q_A_ROUGE, wf_Truth_Ratio)
           / same(base model)
  Mem.   = HM(1-ES, 1-EM, 1-ParaProb, 1-TR_pm) on the forget split, with
           TR_pm = mean_i[ c_i / (c_i + mean(w_i)) ] recomputed from the
           PARA/PERT per-index probs (prob-mean truth ratio)
  Priv.  = HM over {loss, zlib, min_k, min_k++} of
           min(auc, auc_retain) / max(auc, auc_retain)
  Agg.   = HM(Util.R, Util.G, Mem., Priv.)

Composition-robustness variants reported alongside:
  Mem_verbatim: verbatim forget_Q_A_Prob instead of ParaProb
  Priv_absdiff: HM of 1 - |auc - auc_retain| / max(auc_retain, 1 - auc_retain)

--self_check validates the pipeline against the canonical anchor logs and the
paper's Finetuned/Retrained rows before any MemAdapt eval is trusted (gate G1
offline half; the on-cluster half re-generates the eval JSONs themselves).
"""

import argparse
import json
from statistics import harmonic_mean

import os
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

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

EVAL_REFS = os.path.join(os.environ["TOFU_STORAGE_ROOT"], "memadapt", "eval_refs")
MIA_ATTACKS = ["mia_loss", "mia_zlib", "mia_min_k", "mia_min_k_plus_plus"]

UTIL_R_KEYS = ["retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_Truth_Ratio"]
UTIL_G_KEYS = [
    "ra_Q_A_Prob_normalised", "ra_Q_A_ROUGE", "ra_Truth_Ratio",
    "wf_Q_A_Prob_normalised", "wf_Q_A_ROUGE", "wf_Truth_Ratio",
]


def load_eval(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def agg(ev: dict, key: str) -> float:
    return float(ev[key]["agg_value"])


def truth_ratio_prob_mean(ev: dict) -> float:
    """TR_pm = mean_i[c_i / (c_i + mean(w_i))] from PARA/PERT per-index probs."""
    para = ev["forget_Q_A_PARA_Prob"]["value_by_index"]
    pert = ev["forget_Q_A_PERT_Prob"]["value_by_index"]
    assert para.keys() == pert.keys()
    vals = []
    for k in para:
        c = float(para[k]["prob"])
        w = pert[k]["prob"]
        w_bar = sum(map(float, w)) / len(w)
        vals.append(c / (c + w_bar) if (c + w_bar) > 0 else 0.0)
    return sum(vals) / len(vals)


def util_r_raw(ev: dict) -> float:
    return harmonic_mean([agg(ev, k) for k in UTIL_R_KEYS])


def util_g_raw(ev: dict) -> float:
    return harmonic_mean([agg(ev, k) for k in UTIL_G_KEYS])


def mem_score(ev: dict, verbatim: bool = False) -> float:
    para_key = "forget_Q_A_Prob" if verbatim else "forget_Q_A_PARA_Prob"
    parts = [
        1.0 - agg(ev, "extraction_strength"),
        1.0 - agg(ev, "exact_memorization"),
        1.0 - agg(ev, para_key),
        1.0 - truth_ratio_prob_mean(ev),
    ]
    return harmonic_mean(parts)


def priv_score(ev: dict, retain_ref: dict, variant: str = "minmax") -> float:
    parts = []
    for attack in MIA_ATTACKS:
        a, r = agg(ev, attack), agg(retain_ref, attack)
        if variant == "minmax":
            hi = max(a, r)
            parts.append(min(a, r) / hi if hi > 0 else 1.0)  # 0/0: identical AUCs
        elif variant == "absdiff":
            parts.append(1.0 - abs(a - r) / max(r, 1.0 - r))
        else:
            raise ValueError(variant)
    return harmonic_mean(parts)


def compose(model_ev: dict, finetuned_ev: dict, retain_ref: dict,
            base_ev: dict = None) -> dict:
    util_r = util_r_raw(model_ev) / util_r_raw(finetuned_ev)
    util_g_raw_v = util_g_raw(model_ev)
    util_g = util_g_raw_v / util_g_raw(base_ev) if base_ev is not None else None
    mem = mem_score(model_ev)
    priv = priv_score(model_ev, retain_ref)
    row = {
        "util_r": util_r,
        "util_g": util_g,
        "util_g_raw": util_g_raw_v,
        "mem": mem,
        "priv": priv,
        "agg": (harmonic_mean([util_r, util_g, mem, priv])
                if util_g is not None else None),
        # composition-robustness variants
        "mem_verbatim": mem_score(model_ev, verbatim=True),
        "priv_absdiff": priv_score(model_ev, retain_ref, variant="absdiff"),
    }
    return row


def self_check():
    """Offline half of gate G1: reproduce the paper's anchor rows from the
    canonical public eval logs. Expected values were derived once during
    planning; any drift here means the composition changed."""
    full = load_eval(f"{EVAL_REFS}/full_eval.json")
    retain = load_eval(f"{EVAL_REFS}/retain90_eval.json")

    ft = compose(full, full, retain)
    rt = compose(retain, full, retain)

    checks = [
        ("Finetuned Util.R == 1 by construction", ft["util_r"], 1.0, 1e-9),
        ("Finetuned Priv (paper 0.38)", ft["priv"], 0.381, 0.002),
        ("Finetuned Mem (paper 0.07 + own-ckpt residual)", ft["mem"], 0.087, 0.005),
        ("Retrained Priv == 1 by construction", rt["priv"], 1.0, 1e-9),
        ("Retrained Mem (paper 0.58 + residual)", rt["mem"], 0.591, 0.005),
        ("Retrained Util.R (paper 1.00)", rt["util_r"], 1.01, 0.02),
    ]
    failed = False
    for name, got, want, tol in checks:
        ok = abs(got - want) <= tol
        failed |= not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got:.4f} want {want}±{tol}")
    print(f"[info] implied base Util.G raw from Finetuned: "
          f"{util_g_raw(full) / 1.14:.4f} (expect ~0.493)")
    print(f"[info] implied base Util.G raw from Retrained: "
          f"{util_g_raw(retain) / 1.11:.4f} (expect ~0.494)")
    if failed:
        raise SystemExit("self-check FAILED — composition drifted")
    print("self-check PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_check", action="store_true")
    ap.add_argument("--model_eval", help="TOFU_EVAL.json of the model to score")
    ap.add_argument("--finetuned_eval", default=f"{EVAL_REFS}/full_eval.json")
    ap.add_argument("--retain_ref", default=f"{EVAL_REFS}/retain90_eval.json")
    ap.add_argument("--base_eval", default=None,
                    help="TOFU_EVAL.json of base Llama-3.2-1B (Util.G denominator)")
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", default=None, help="append row to this JSONL")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    row = compose(
        load_eval(args.model_eval),
        load_eval(args.finetuned_eval),
        load_eval(args.retain_ref),
        load_eval(args.base_eval) if args.base_eval else None,
    )
    row["label"] = args.label
    row["model_eval"] = args.model_eval
    fmt = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}
    print(json.dumps(fmt, indent=2))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
