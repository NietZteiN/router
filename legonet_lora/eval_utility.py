"""Utility / general-capability eval: does the LegoNet wrapper preserve the
frozen base's ability? (Forget-side metrics live in eval_memorization.py.)

Two measures, each comparing the routed LegoNet model vs the frozen base:
  * MMLU (cais/mmlu, zero-shot) — multiple-choice accuracy via the logprob of the
    answer-letter token (no lm_eval dependency). Expectation: legonet ≈ base
    (frozen backbone preserves capability). Skipped (logged) if the dataset can't
    be fetched.
  * held-out perplexity — on the disjoint DBpedia reference split (never trained).
    legonet ≈ base => the k delta-averaged adapters don't degrade general LM ability.

For the LegoNet model, each item is routed (MiniLM -> k nearest frozen keys ->
delta-average) exactly as at deployment; items are grouped by adapter-set so each
merge happens once.

    python eval_utility.py --config configs/legonet_7b.json --n_mmlu 300 --n_ppl 300
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

from legonet_common import Paths, load_config, load_records, make_embed_fn, route_text, write_json

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


def _run_grouped(cfg, which, items, text_of, score_one, device_map=None):
    """Apply score_one(model, tok, item) to each item under the right model.

    base: frozen base, no adapters. legonet: route each item to its k nearest
    frozen keys, group by adapter-set, activate the delta-average per group.
    """
    import torch

    if which == "base":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        use_cuda = torch.cuda.is_available()
        dm = device_map or ("auto" if use_cuda else "cpu")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], torch_dtype=dtype, device_map=dm, trust_remote_code=True)
        model.eval()
        return [score_one(model, tok, it) for it in items]

    from combine import LegoNetModel
    from routing import KNNRouter
    paths = Paths(cfg)
    keys = np.load(paths.keys_path)
    router = KNNRouter(keys, cfg["k"])
    embed = make_embed_fn(cfg["encoder_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    routes = router.route(embed([text_of(it) for it in items]))  # (N, k)
    groups = defaultdict(list)
    for idx, ks in enumerate(routes):
        groups[tuple(sorted(int(j) for j in ks))].append(idx)

    lego = LegoNetModel.from_config(cfg, device_map=device_map)
    out = [None] * len(items)
    for idxs, item_indices in groups.items():
        with lego.activated(idxs) as m:
            for ii in item_indices:
                out[ii] = score_one(m, lego.tokenizer, items[ii])
    return out


def _text_ppl(model, tok, text, max_length):
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    ids = enc.input_ids
    if ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        logits = model(ids, attention_mask=enc.attention_mask).logits[0]
    nll = F.cross_entropy(logits[:-1].float(), ids[0, 1:], reduction="mean").item()
    return math.exp(nll)


def held_out_ppl(cfg, which, n, device_map=None):
    paths = Paths(cfg)
    ref = load_records(paths.reference_path)[:n]
    ml = cfg["train"]["max_length"]
    ppls = _run_grouped(cfg, which, ref, route_text,
                        lambda m, t, r: _text_ppl(m, t, route_text(r), ml), device_map)
    vals = [p for p in ppls if p is not None and not math.isnan(p)]
    return {"mean_ppl": float(np.mean(vals)), "median_ppl": float(np.median(vals)), "n": len(vals)}


_LETTERS = ["A", "B", "C", "D"]


def _mmlu_prompt(q, choices):
    lines = [f"Question: {q}"] + [f"{L}. {c}" for L, c in zip(_LETTERS, choices)]
    lines.append("Answer:")
    return "\n".join(lines)


def _pred_letter(last_logits, letter_ids) -> int:
    """Argmax over the 4 answer-letter token logits (pure; unit-tested)."""
    return int(np.argmax([float(last_logits[i]) for i in letter_ids]))


def _mmlu_score(model, tok, item):
    import torch
    device = next(model.parameters()).device
    prompt = _mmlu_prompt(item["question"], item["choices"])
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(enc.input_ids, attention_mask=enc.attention_mask).logits[0, -1].float()
    letter_ids = [tok(f" {L}", add_special_tokens=False).input_ids[-1] for L in _LETTERS]
    return 1.0 if _pred_letter(logits, letter_ids) == item["answer"] else 0.0


def mmlu_acc(cfg, which, n, device_map=None):
    os.environ["HF_HOME"] = cfg["hf_home"]
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception as e:
        print(f"MMLU unavailable ({type(e).__name__}: {e}); skipping.")
        return None
    rng = np.random.default_rng(cfg["base_seed"])
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    items = [ds[int(i)] for i in idx]
    correct = _run_grouped(cfg, which, items, lambda it: it["question"], _mmlu_score, device_map)
    vals = [c for c in correct if c is not None]
    return {"acc": float(np.mean(vals)), "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--which", choices=["base", "legonet", "both"], default="both")
    ap.add_argument("--n_mmlu", type=int, default=300)
    ap.add_argument("--n_ppl", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    whichs = ["base", "legonet"] if args.which == "both" else [args.which]

    result = {"config": cfg["name"]}
    for w in whichs:
        result[w] = {
            "held_out_ppl": held_out_ppl(cfg, w, args.n_ppl) if args.n_ppl else None,
            "mmlu": mmlu_acc(cfg, w, args.n_mmlu) if args.n_mmlu else None,
        }
        print(f"[{w}] {result[w]}")
    out = args.out or os.path.join(Paths(cfg).results_dir, "eval_utility.json")
    write_json(out, result)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
