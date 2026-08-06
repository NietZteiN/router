"""Phase-D probe (H11 + H10): what do the sink slices actually store, and how
does foreign-slice interference scale?

Per author a over its train rows, answer-prob P(a|q)^(1/|a|) under fixed serving
vectors (MaskState.set_fixed via the trained adapter's hooks):

  gen_only : dropall vector — what the SHARED capacity alone knows about a
  gen_own  : gen + a's own slice — the training condition
  all_on   : every slice active — the lean-phase serving mode

  slice_increment = gen_own − gen_only         <- THE H11 statistic
  interference    = gen_own − all_on           <- the interference-relief share
                                                   of the lean-phase H4 gap

H10 ladder (seeded author subset): gen + own + k FOREIGN slices, k ∈ ladder_ks
(k=0 ≡ gen_own and k=199 ≡ all_on come free from the main conditions).

GPU, ~10 min for all 200 authors x 20 rows x 3 conditions on an A40.
Usage:
  probe_slices.py --config CFG [--run_dir DIR] [--authors all|0,50,...]
                  [--rows_per_author 20] [--ladder_authors 20]
                  [--ladder_ks 10,50,100] [--seed 42] --out probe_slices.json
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memsinks_model import (
    MaskState, author_serve_vector, build_scale_vector, install_sink_hooks, load_masks,
)


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


def _answer_prob(model, tokenizer, q, a, max_length=256):
    """exp(-avg answer-token CE) — mirrors eval_tofu._answer_avg_loss."""
    device = next(model.parameters()).device
    prompt = f"Question: {q}\nAnswer:"
    n_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    enc = tokenizer(f"{prompt} {a}", return_tensors="pt",
                    truncation=True, max_length=max_length).to(device)
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    if (labels != -100).sum() == 0:
        return float("nan")
    with torch.no_grad():
        loss = model(**enc, labels=labels).loss
    return math.exp(-loss.float().item())


def ladder_vector(mask_table, num_gen, author, k, seed):
    """gen + own slice + k seeded foreign slices."""
    v = author_serve_vector(mask_table, num_gen, author)
    if k > 0:
        others = [a for a in range(mask_table.shape[0]) if a != author]
        rng = np.random.default_rng(seed * 1000 + author)
        foreign = rng.choice(others, size=k, replace=False)
        union = mask_table[list(foreign)].any(dim=0)
        v[num_gen:][union] = 1.0
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None, help="override cfg output_dir (e.g. an e25/strict run)")
    p.add_argument("--authors", default="all")
    p.add_argument("--rows_per_author", type=int, default=20)
    p.add_argument("--ladder_authors", type=int, default=20)
    p.add_argument("--ladder_ks", default="10,50,100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    os.environ["HF_HOME"] = cfg["hf_home"]
    run_dir = args.run_dir or cfg["output_dir"]
    trained = os.path.join(run_dir, "trained")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, trained)
    model.eval()

    mask_table, num_gen = load_masks(trained)
    state = MaskState(mask_table, num_gen)
    install_sink_hooks(model, state, cfg["sink_modules"], model.config.num_hidden_layers)
    state.to(next(model.parameters()).device)
    table = state.mask_table.cpu()

    ds = load_dataset("locuslab/TOFU", "full")["train"]
    A = cfg["num_authors"]
    authors = list(range(A)) if args.authors == "all" else \
        sorted({int(x) for x in args.authors.split(",")})
    forget = set(cfg["forget_authors"])

    conditions = {
        "gen_only": build_scale_vector(table, num_gen, "dropall"),
        "all_on": build_scale_vector(table, num_gen, "full"),
    }

    def rows_for(a):
        return [(ds[r]["question"], ds[r]["answer"])
                for r in range(a * 20, a * 20 + min(args.rows_per_author, 20))]

    per_author = {}
    for i, a in enumerate(authors):
        rows = rows_for(a)
        rec = {}
        for name, vec in [("gen_only", conditions["gen_only"]),
                          ("gen_own", author_serve_vector(table, num_gen, a)),
                          ("all_on", conditions["all_on"])]:
            state.set_fixed(vec)
            probs = [_answer_prob(model, tokenizer, q, ans, cfg["max_length"]) for q, ans in rows]
            rec[name] = float(np.nanmean(probs))
        state.clear()
        rec["n_rows"] = len(rows)
        rec["slice_increment"] = rec["gen_own"] - rec["gen_only"]
        rec["interference"] = rec["gen_own"] - rec["all_on"]
        per_author[str(a)] = rec
        if i % 20 == 0:
            print(f"[probe] {i}/{len(authors)} authors; a={a}: {json.dumps(rec)}", flush=True)

    # H10 ladder: seeded subset with >=5 forget authors
    ladder_ks = [int(x) for x in args.ladder_ks.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)
    forget_pool = [a for a in authors if a in forget]
    retain_pool = [a for a in authors if a not in forget]
    n_f = min(5, len(forget_pool))
    ladder_set = (list(rng.choice(forget_pool, size=n_f, replace=False)) if n_f else []) + \
        list(rng.choice(retain_pool, size=max(0, args.ladder_authors - n_f), replace=False))
    ladder = {}
    for a in sorted(int(x) for x in ladder_set):
        rows = rows_for(a)
        entry = {"0": per_author[str(a)]["gen_own"], "199": per_author[str(a)]["all_on"]}
        for k in ladder_ks:
            state.set_fixed(ladder_vector(table, num_gen, a, k, args.seed))
            probs = [_answer_prob(model, tokenizer, q, ans, cfg["max_length"]) for q, ans in rows]
            entry[str(k)] = float(np.nanmean(probs))
        state.clear()
        ladder[str(a)] = entry
    print(f"[probe] ladder done ({len(ladder)} authors x ks={ladder_ks})", flush=True)

    def group(stat, ids):
        vals = [per_author[str(a)][stat] for a in ids if str(a) in per_author]
        return float(np.mean(vals)) if vals else float("nan")

    ks_sorted = ["0"] + [str(k) for k in sorted(ladder_ks)] + ["199"]
    monotone = [all(e[ks_sorted[j]] >= e[ks_sorted[j + 1]] - 0.01 for j in range(len(ks_sorted) - 1))
                for e in ladder.values()]
    retain_ids = [a for a in authors if a not in forget]
    forget_ids = [a for a in authors if a in forget]

    def sha(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    out = {
        "provenance": {
            "config": os.path.abspath(args.config), "run_dir": run_dir,
            "adapter_sha256": sha(os.path.join(trained, "adapter_model.safetensors")),
            "script_sha256": sha(__file__),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seed": args.seed,
            "conditions": ["gen_only", "gen_own", "all_on"], "ladder_ks": ladder_ks,
        },
        "per_author": per_author,
        "ladder": ladder,
        "aggregates": {
            "mean_gen_only": group("gen_only", authors),
            "mean_gen_own": group("gen_own", authors),
            "mean_all_on": group("all_on", authors),
            "mean_slice_increment": group("slice_increment", authors),
            "mean_interference": group("interference", authors),
            "forget_group": {s: group(s, forget_ids) for s in
                             ["gen_only", "gen_own", "all_on", "slice_increment"]},
            "retain_group": {s: group(s, retain_ids) for s in
                             ["gen_only", "gen_own", "all_on", "slice_increment"]},
            "ladder_monotone_fraction": float(np.mean(monotone)) if monotone else float("nan"),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {args.out}")
    print(json.dumps(out["aggregates"], indent=2))


if __name__ == "__main__":
    main()
