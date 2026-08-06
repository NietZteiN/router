"""Uniform held-out NLL for the Part B facts-vs-skills merge contrast (SLURM-callable).

For each of N isolated adapters, mean answer-token NLL on that adapter's held-out probe under three
weight states: base (no adapter), isolated (adapter j only), merged (all N averaged, linear 1/N —
the same combine as the LegoNet/`merged_linear` arm). Reuses eval_tofu._answer_avg_loss so the metric
is identical to the TOFU probability path. One invocation writes the full sweep -> results JSON;
`analyze_skill_vs_facts.py` turns two of these (skills + facts) into normalized-retention rows.

  python eval_skill.py --domain skills --config configs/skills_superni_1b.json --out reports/skill_nll_skills.json
  python eval_skill.py --domain facts  --config configs/skills_superni_1b.json --out reports/skill_nll_facts.json
"""
import argparse
import json
import os

import numpy as np
import torch

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


def _load_all(base_model, adapter_dirs, names):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_dirs[0], adapter_name=names[0])
    for d, n in zip(adapter_dirs[1:], names[1:]):
        model.load_adapter(d, adapter_name=n)
    return model, tok


def _mean_nll(model, tok, probes, max_length=256):
    from eval_tofu import _answer_avg_loss
    vals = [_answer_avg_loss(model, tok, p["question"], p["answer"], max_length=max_length) for p in probes]
    vals = [v for v in vals if v == v]  # drop NaN (empty-answer)
    return float(np.mean(vals)) if vals else float("nan")


def _resolve(domain, cfg):
    """Return (adapter_dirs, names, heldout_per_adapter). heldout[j] = list of {question,answer}."""
    n = cfg["n"]
    hf_home = cfg["hf_home"]
    if domain == "skills":
        root = cfg["output_dir"]
        dirs = [os.path.join(root, f"a{j}") for j in range(n)]
        names = [f"a{j}" for j in range(n)]
        heldout = []
        for j in range(n):
            meta = json.load(open(os.path.join(dirs[j], "skill_meta.json")))
            heldout.append(meta["held_out"])
        return dirs, names, heldout
    elif domain == "facts":
        from shard_utils import get_author_shard
        import skill_data
        root, k = cfg["facts_dir"], cfg["facts_k"]
        dirs = [os.path.join(root, f"shard_{j}") for j in range(k)]
        names = [f"shard_{j}" for j in range(k)]
        cap = cfg["holdout_per_adapter"]
        heldout = [skill_data.facts_heldout(get_author_shard(k, j), hf_home, max_probes=cap)
                   for j in range(k)]
        return dirs, names, heldout
    raise ValueError(domain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["skills", "facts"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.environ["HF_HOME"] = cfg["hf_home"]

    dirs, names, heldout = _resolve(args.domain, cfg)
    missing = [d for d in dirs if not os.path.exists(os.path.join(d, "adapter_config.json"))]
    if missing:
        raise SystemExit(f"[eval_skill] {len(missing)} adapter(s) missing, e.g. {missing[0]}")

    model, tok = _load_all(cfg["base_model"], dirs, names)
    model.add_weighted_adapter(names, [1.0 / len(names)] * len(names), "merged", combination_type="linear")
    model.eval()

    rows = []
    for j, nm in enumerate(names):
        probes = heldout[j]
        with model.disable_adapter():
            base_nll = _mean_nll(model, tok, probes, args.max_length)
        model.set_adapter(nm)
        iso_nll = _mean_nll(model, tok, probes, args.max_length)
        model.set_adapter("merged")
        mer_nll = _mean_nll(model, tok, probes, args.max_length)
        row = {"domain": args.domain, "adapter": j, "name": nm, "n_probes": len(probes),
               "base_nll": base_nll, "isolated_nll": iso_nll, "merged_nll": mer_nll}
        denom = iso_nll - base_nll
        row["retention"] = (mer_nll - base_nll) / denom if denom not in (0.0, float("nan")) and denom == denom and abs(denom) > 1e-9 else float("nan")
        rows.append(row)
        print(f"  {args.domain} a{j} {nm}: base={base_nll:.4f} iso={iso_nll:.4f} merged={mer_nll:.4f} R={row['retention']:.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"domain": args.domain, "config": args.config, "rows": rows}, open(args.out, "w"), indent=2)
    good = [r["retention"] for r in rows if r["retention"] == r["retention"]]
    print(f"[eval_skill] {args.domain}: mean retention = {np.mean(good):.4f} over {len(good)} adapters -> {args.out}")


if __name__ == "__main__":
    main()
