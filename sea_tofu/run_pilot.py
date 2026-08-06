"""SEA-on-TOFU pilot: train 5 author proxies @ rank 16 on one GPU and validate the pipeline.

Validates (SEA_on_TOFU.md plan, Verification §1-4):
  - training completes with finite loss,
  - per-author Prob/ROUGE-L with proxy loaded > base-only (personalization works),
  - cross-author contamination ≈ 0 (isolation; adapters do not accumulate),
  - Forget Quality ≈ 1 (base-only candidate vs base-only gold),
  - deletion KL gate passes and writes an audit line (omission == post-deletion).

Run (single GPU, NOT the login node — use srun/sbatch):
  HF_HOME=${HF_HOME} \
    ${TOFU_PYTHON:-python3} run_pilot.py
"""
import json
import os

import numpy as np

from deletion import build_baseline, verify_and_delete
from eval_sea_tofu import (
    base_constants,
    base_forget_truth_ratios,
    forget_quality,
    isolation_table,
    model_utility,
    personalization_table,
    retain_utility,
    write_json,
)
from inference import SeaProxyModel, load_base
from load_tofu import load_tofu_data
from proxy_paths import author_dir, dir_size_mb, personal_lora_dir, results_dir
from train_proxy import _load_cfg, train_one_author

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

HF_HOME = os.environ.get("HF_HOME", os.environ["HF_HOME"])
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "sea_tofu_llama2.json")
PILOT_AUTHORS = [180, 181, 182, 183, 184]   # first 5 of forget10
RANK = 16
ROUGE_MAX = 20      # per-author sets are only 20 QA
GENERIC_MAX = 40    # generic prompts for deletion verification


def main():
    os.environ["HF_HOME"] = HF_HOME
    cfg = _load_cfg(CFG_PATH)
    cfg["personal_lora"]["rank"] = RANK
    model_name = cfg["model_name"]

    print("=== loading TOFU + base (4-bit) ===", flush=True)
    data = load_tofu_data(HF_HOME)
    base, tok = load_base(model_name, hf_home=HF_HOME)

    print("=== training 5 author proxies @ rank", RANK, "===", flush=True)
    for a in PILOT_AUTHORS:
        qa = [data["full"][a * 20 + j] for j in range(20)]
        base = train_one_author(base, tok, a, qa, cfg, rank=RANK)

    sea = SeaProxyModel(base, tok)

    def lora_dir(a):
        return personal_lora_dir(model_name, a, RANK)

    print("=== base constants (real/world, frozen) ===", flush=True)
    bconst = base_constants(base, tok, data, rouge_max=30, truth_max=30)

    print("=== personalization depth (proxy vs base) ===", flush=True)
    pers = personalization_table(sea, PILOT_AUTHORS, lora_dir, data,
                                 rouge_max=ROUGE_MAX, truth_max=None)

    print("=== retain utility (each author with its proxy) ===", flush=True)
    retain = retain_utility(sea, PILOT_AUTHORS, lora_dir, data,
                            rouge_max=ROUGE_MAX, truth_max=None)
    util = model_utility(retain, bconst)

    print("=== forget quality (base-only candidate vs gold) ===", flush=True)
    base_tr = base_forget_truth_ratios(base, tok, data, PILOT_AUTHORS, truth_max=None)
    fq = forget_quality(base_tr, base_tr)  # identical base-only dists -> ~1.0 by construction

    print("=== isolation (cross-author contamination) ===", flush=True)
    pairs = [(180, 181), (181, 182), (182, 183)]
    iso = isolation_table(sea, pairs, lora_dir, data, n_questions=5)

    print("=== deletion verify (gate only; do_delete=False) ===", flush=True)
    generic_prompts = list(data["real_authors"]["question"])[:GENERIC_MAX]
    rdir = results_dir(model_name, RANK, sub="pilot")
    baseline_path = os.path.join(rdir, "baseline_dist.npy")
    with sea.omission() as base_m:
        baseline = build_baseline(base_m, tok, generic_prompts, baseline_path)
    sea.attach(180, lora_dir(180))   # a proxy loaded, but verify runs in omission mode
    passed, kl_d, thr, _ = verify_and_delete(
        sea, author_dir(model_name, 180, RANK), baseline, generic_prompts,
        tau_min=cfg["deletion"]["tau_min"], mult=cfg["deletion"]["kl_mult"],
        audit_path=os.path.join(rdir, "audit.log"), do_delete=False,
    )

    sizes = {a: round(dir_size_mb(author_dir(model_name, a, RANK)), 3) for a in PILOT_AUTHORS}

    summary = {
        "rank": RANK,
        "authors": PILOT_AUTHORS,
        "base_constants": {k: round(v, 4) for k, v in bconst.items()},
        "personalization": pers,
        "retain_utility": retain,
        "model_utility": round(util, 4) if not np.isnan(util) else None,
        "forget_quality": round(fq, 4),
        "isolation": iso,
        "deletion_gate": {"passed": passed, "kl": round(kl_d, 5), "threshold": round(thr, 5)},
        "proxy_size_mb": sizes,
    }
    write_json(summary, os.path.join(rdir, "pilot_summary.json"))

    # ── Validation assertions / flags ────────────────────────────────────────
    print("\n=== PILOT CHECKS ===", flush=True)
    mean_drouge = float(np.nanmean([p["delta_rougeL"] for p in pers]))
    mean_dprob = float(np.nanmean([p["delta_prob"] for p in pers]))
    max_contam = max((r["contamination"] for r in iso), default=float("nan"))
    print(f"mean Δ ROUGE-L (proxy-base): {mean_drouge:+.4f}  (expect > 0)")
    print(f"mean Δ Prob    (proxy-base): {mean_dprob:+.4f}  (expect > 0)")
    print(f"max cross-author contamination: {max_contam:.4f}  (expect ≈ 0)")
    print(f"forget quality (construction-trivial): {fq:.4f}  (expect ≈ 1.0)")
    print(f"model utility: {util:.4f}")
    print(f"deletion gate passed: {passed} (kl={kl_d:.4f} <= thr={thr:.4f})")
    print(f"proxy sizes MB: {sizes}")
    print(json.dumps(summary, indent=2)[:2000], flush=True)


if __name__ == "__main__":
    main()
