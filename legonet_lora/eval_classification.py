"""LegoNet-units evaluation: 14-class DBpedia topic-classification accuracy.

Casts the (already trained) generative experts as zero-shot classifiers — prompt
"<content>\\nCategory:" and pick the label with the highest length-normalized
log-prob over the 14 DBpedia class names — to report accuracy on:
  * D_retain : corpus records the experts trained on
  * D_test   : held-out reference records (DBpedia test split, never trained)
for the routed LegoNet model vs the frozen base. This is the paper's accuracy
metric (Table 2/3). NOTE: the forgetting column (D_unlearn) is NOT faithfully
reproducible here — the 7B base classifies DBpedia topics zero-shot, so deleting
a record's experts doesn't drop its class accuracy; canary_em is our forget
analog instead (see report §3a).

    python eval_classification.py --config configs/legonet_7b_v2.json --which legonet
"""
import argparse
import json
import os

import numpy as np

from legonet_common import Paths, load_config, load_records
from eval_utility import _run_grouped

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


def _label_ll(model, tok, prompt, label):
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    full = tok(prompt + label, return_tensors="pt", truncation=True, max_length=320).to(device)
    plen = len(tok(prompt, add_special_tokens=True).input_ids)
    ids = full.input_ids
    with torch.no_grad():
        logits = model(ids, attention_mask=full.attention_mask).logits[0]
    logp = F.log_softmax(logits[:-1].float(), dim=-1)
    tgt = ids[0, 1:]
    start = max(plen - 1, 0)
    sel = logp[start:].gather(1, tgt[start:].unsqueeze(1)).squeeze(1)  # label-token logprobs
    return sel.mean().item() if sel.numel() > 0 else -1e9


def _make_scorer(class_names):
    def score(model, tok, r):
        prompt = r["content"].strip() + "\nCategory:"
        lls = [_label_ll(model, tok, prompt, " " + c) for c in class_names]
        return 1.0 if int(np.argmax(lls)) == r["label"] else 0.0
    return score


def evaluate_classification(cfg, which, n=100):
    paths = Paths(cfg)
    manifest = json.load(open(paths.corpus_manifest))
    class_names = manifest["class_names"]
    score = _make_scorer(class_names)
    text_of = lambda r: r["content"]

    retain = load_records(paths.records_path)[:n]
    test = load_records(paths.reference_path)[:n]
    r_acc = _run_grouped(cfg, which, retain, text_of, score)
    t_acc = _run_grouped(cfg, which, test, text_of, score)
    return {"retain_acc": float(np.mean([x for x in r_acc if x is not None])),
            "test_acc": float(np.mean([x for x in t_acc if x is not None])),
            "n": n, "num_classes": len(class_names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--which", choices=["legonet", "base"], default="legonet")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    res = evaluate_classification(cfg, args.which, args.n)
    print(f"[{args.which}] {res}")
    if args.which == "base":
        out = args.out or os.path.join(cfg["root"], "eval_classification_base.json")
    else:
        out = args.out or os.path.join(Paths(cfg).results_dir, "eval_classification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"which": args.which, **res}, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
