"""Memorization + utility eval for the LegoNet-LoRA model.

Forget efficacy on a set of records, measured three ways:
  * EM  — exact memorization: teacher-forced next-token argmax accuracy over the
          completion (open-unlearning `exact_memorization`).
  * ES  — extraction strength: 1 - k/len, k = earliest position whose greedy
          suffix matches the target (open-unlearning `extraction_strength`).
  * VerbMem — ROUGE-L recall of a greedy generation vs the true completion.
  * canary_hit — does the greedy generation contain the record's exact canary
          code? The clean, training-attributable signal (a never-trained model
          can't guess a random 12-char code -> ~0; trained -> high; unlearned -> ~0).

`em_from_preds` / `es_from_preds` are the pure metric math (mirroring OU exactly)
and are unit-tested; the rest runs a model. Records route to their k activated
adapters and are grouped by adapter-set so each merge happens once.
"""
import argparse
import json
import os

import numpy as np

from legonet_common import Paths, load_config, load_records, prompt_completion, write_json

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


# ── Pure metric math (unit-tested against the OU formulas) ─────────────────────

def em_from_preds(preds, labels) -> float:
    """Fraction of positions where the teacher-forced argmax equals the label."""
    if len(labels) == 0:
        return float("nan")
    preds = np.asarray(preds); labels = np.asarray(labels)
    return float((preds == labels).sum() / len(labels))


def es_from_preds(preds, labels) -> float:
    """1 - k/len, k = smallest index whose suffix is fully greedily-correct (OU)."""
    valid_len = len(labels)
    if valid_len == 0:
        return 0.0
    preds = np.asarray(preds); labels = np.asarray(labels)
    k = valid_len
    for i in range(valid_len):
        if np.array_equal(preds[i:], labels[i:]):
            k = i
            break
    return float(1 - (k / valid_len))


def rougeL_recall(pred: str, ref: str) -> float:
    """LCS-based ROUGE-L recall over whitespace word tokens (no external dep)."""
    p = pred.split()
    r = ref.split()
    if not r:
        return float("nan")
    if not p:
        return 0.0
    # LCS length (O(len(p)*len(r)) DP, 1-D rolling)
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0] * (len(r) + 1)
        for j, b in enumerate(r, 1):
            cur[j] = prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(r)] / len(r)


# ── Model-running metrics ─────────────────────────────────────────────────────

def _preds_labels(model, tokenizer, prompt: str, completion: str, max_length: int,
                  canary: str = ""):
    """Teacher-forced (preds, labels) over the completion, plus the index within
    that array where the canary tokens begin (-1 if no canary / truncated off).

    The canary index lets us score EM on just the canary tokens — the clean,
    training-attributable signal (a never-trained model can't predict a random
    code even given the true preceding context)."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    full = tokenizer(prompt + completion, return_tensors="pt",
                     truncation=True, max_length=max_length)
    plen = len(tokenizer(prompt, add_special_tokens=True).input_ids)
    ids = full.input_ids.to(device)
    with torch.no_grad():
        logits = model(ids, attention_mask=full.attention_mask.to(device)).logits[0]
    # logits[t] predicts token t+1
    pred = logits[:-1].argmax(-1)
    tgt = ids[0, 1:]
    start = max(plen - 1, 0)  # completion region in the shifted (next-token) space
    preds = pred[start:].cpu().numpy()
    labels = tgt[start:].cpu().numpy()
    # mean NLL over completion tokens -> perplexity (utility on this record)
    if len(labels) > 0:
        nll = F.cross_entropy(logits[:-1][start:].float(), tgt[start:], reduction="mean").item()
    else:
        nll = float("nan")

    # canary span = the HIGH-ENTROPY CODE only (pass the code, not the "Verification
    # code:" prefix, which is shared across all records and predictable -> dilutes).
    canary_idx = -1
    if canary:
        cpos = completion.rfind(canary)
        if cpos >= 0:
            pre = prompt + completion[:cpos]
            pre_len = len(tokenizer(pre, add_special_tokens=True).input_ids)
            ci = pre_len - plen  # index of first canary token within the labels array
            if 0 <= ci < len(labels):
                canary_idx = ci
    return preds, labels, canary_idx, nll


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    gen = out[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def metrics_for_records(model, tokenizer, records, max_length, gen_cap=64):
    import math
    rows = []
    for r in records:
        prompt, completion = prompt_completion(r)
        code = r.get("canary", "").replace("Verification code:", "").strip()
        # span the CODE only (clean Secret-Sharer signal), not the shared prefix
        preds, labels, ci, nll = _preds_labels(model, tokenizer, prompt, completion,
                                               max_length, code)
        gen = _generate(model, tokenizer, prompt, gen_cap)
        canary_em = em_from_preds(preds[ci:], labels[ci:]) if ci >= 0 else float("nan")
        rows.append({
            "id": r["id"],
            "em": em_from_preds(preds, labels),
            "es": es_from_preds(preds, labels),
            "verbmem": rougeL_recall(gen, completion),
            "perplexity": math.exp(nll) if not math.isnan(nll) else float("nan"),  # utility
            "canary_em": canary_em,                       # primary forget signal (code only)
            "canary_hit": float(bool(code) and code in gen),  # stringent (free-gen) secondary
        })
    return rows


def aggregate(rows) -> dict:
    keys = ["em", "es", "verbmem", "perplexity", "canary_em", "canary_hit"]
    out = {}
    for kk in keys:
        vals = [r[kk] for r in rows if not (isinstance(r[kk], float) and np.isnan(r[kk]))]
        out[kk] = float(np.mean(vals)) if vals else float("nan")
    out["num_records"] = len(rows)
    return out


def evaluate(cfg, which, record_ids, adapter_dir_fn=None, gen_cap=64, device_map=None):
    """which: 'base' (never-trained) | 'legonet' | custom via adapter_dir_fn."""
    paths = Paths(cfg)
    by_id = {r["id"]: r for r in load_records(paths.records_path)}
    records = [by_id[i] for i in record_ids]
    max_length = cfg["train"]["max_length"]

    if which == "base":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        use_cuda = torch.cuda.is_available()
        if device_map is None:
            device_map = "auto" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32
        tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], torch_dtype=dtype,
            device_map=device_map, trust_remote_code=True)
        model.eval()
        rows = metrics_for_records(model, tok, records, max_length, gen_cap)
        return rows, aggregate(rows)

    # adapter-backed: group by activated adapter-set, merge once per group
    from combine import LegoNetModel, route_groups
    with open(paths.assignment_path) as f:
        assignment = json.load(f)
    lego = LegoNetModel.from_config(cfg, adapter_dir_fn=adapter_dir_fn, device_map=device_map)
    groups = route_groups(assignment, record_ids)
    rows = []
    for idxs, ids in groups.items():
        grp_records = [by_id[i] for i in ids]
        with lego.activated(idxs) as m:
            rows.extend(metrics_for_records(m, lego.tokenizer, grp_records, max_length, gen_cap))
    return rows, aggregate(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--which", choices=["base", "legonet"], default="legonet")
    ap.add_argument("--record_ids", nargs="*", default=None,
                    help="default: first --n_eval corpus records")
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--gen_cap", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    paths = Paths(cfg)
    ids = args.record_ids or [r["id"] for r in load_records(paths.records_path)[: args.n_eval]]
    rows, agg = evaluate(cfg, args.which, ids, gen_cap=args.gen_cap)
    print(json.dumps(agg, indent=2))
    if args.out:
        write_json(args.out, {"which": args.which, "aggregate": agg, "rows": rows})


if __name__ == "__main__":
    main()
