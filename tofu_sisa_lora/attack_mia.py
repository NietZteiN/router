"""Deletion-audit attack A4 — membership inference on the SERVED, post-deletion composition.

Attacks the exact artifact eval_tofu scores (base + scaffold + router + surviving experts) by
reusing eval_tofu.build_served_model, so the MIA AUC and the forget_quality in the scatter come
from the same served model. Member set = TOFU forget10 (400 QA), non-member = holdout10 (400 QA,
never in any training split — open-unlearning's own TOFU_MIA pairing). Cheap battery only
(loss / min_k / min_k++ / zlib — no reference model, no gradients); AUC via the OU-faithful port
in mia_attacks.py. Exact unlearning ⟺ post-deletion AUC ≈ the retain90 oracle floor (≈0.5).

Mirrors eval_tofu.py's arm flags so the served composition is bit-identical:
  python attack_mia.py --model_name <m> --output_dir <dir> --label <l> --k 10 --forget_shard_id 9 \
      [--preloaded_adapter D | --legonet_config C [--legonet_unlearn_tag forget10]
       [--ramole_router P --ramole_route {embed,key} --ramole_index {stale,rebuilt}]
       | --sift_masks_config C [--sift_unlearn_tag T] | --clamu_config C [--clamu_unlearn_tag T]] \
      --out .../results/mia/<label>.json

Prompt format MUST match training ("Question: {q}\nAnswer: {a}", answer-only labels) — the
single biggest silent-failure risk; using a chat template here would give a garbage AUC.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

import eval_tofu as et
import mia_attacks as mia

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


def _git_or_sha(path):
    try:
        import hashlib
        with open(path, "rb") as f:
            return "sha256:" + hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "unknown"


class QADataset(Dataset):
    """forget10 / holdout10 rows tokenized as "Question: {q}\nAnswer: {a}" with the prompt
    tokens masked to -100 (answer-only labels, `label_scope='answer'`) or full-sequence
    (`'full'`, a robustness variant). Yields {input_ids, labels, index}."""

    def __init__(self, ds, tokenizer, label_scope="answer", max_length=256):
        self.ds, self.tok = ds, tokenizer
        self.scope, self.max_length = label_scope, max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        q, a = row["question"], row["answer"]
        enc = self.tok(et._build_qa_prompt(self.tok, q, a), return_tensors="pt",
                       truncation=True, max_length=self.max_length)
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        if self.scope == "answer":
            n_prompt = self.tok(et._build_qa_prompt(self.tok, q),
                                return_tensors="pt")["input_ids"].shape[1]
            labels[:n_prompt] = -100
        return {"input_ids": input_ids, "labels": labels, "index": i}


def collate(batch):
    """Pad a batch to the longest sequence (right pad). At batch_size=1 this is a no-op; it
    keeps the collator correct if a caller raises the batch size for the loss/zlib attacks."""
    maxlen = max(b["input_ids"].shape[0] for b in batch)
    ids, labs, idx = [], [], []
    for b in batch:
        pad = maxlen - b["input_ids"].shape[0]
        ids.append(torch.nn.functional.pad(b["input_ids"], (0, pad), value=0))
        labs.append(torch.nn.functional.pad(b["labels"], (0, pad), value=-100))
        idx.append(b["index"])
    out = {"input_ids": torch.stack(ids), "labels": torch.stack(labs),
           "index": torch.tensor(idx)}
    out["attention_mask"] = (out["input_ids"] != 0).long()
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # eval_tofu-mirroring served-model selectors
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--forget_shard_id", type=int, default=None)
    p.add_argument("--eval_shard_id", type=int, default=None)  # unused by MIA; kept for parity
    p.add_argument("--preloaded_adapter", default=None)
    p.add_argument("--legonet_config", default=None)
    p.add_argument("--legonet_unlearn_tag", default=None)
    p.add_argument("--ramole_router", default=None)
    p.add_argument("--ramole_route", default="embed", choices=["embed", "key"])
    p.add_argument("--ramole_index", default="stale", choices=["stale", "rebuilt"])
    p.add_argument("--sift_masks_config", default=None)
    p.add_argument("--sift_unlearn_tag", default=None)
    p.add_argument("--clamu_config", default=None)
    p.add_argument("--clamu_unlearn_tag", default=None)
    p.add_argument("--memsinks_config", default=None)
    p.add_argument("--memsinks_unlearn_tag", default=None)
    # router_leak MIA rider (2026-07-23): the embed-ROUTED scaffold arm. Lives in
    # eval_routed_scaffold, not eval_tofu.build_served_model, so it is built separately below.
    p.add_argument("--shards_dir", default=None,
                   help="serve the embed-routed scaffold arm (EmbedRoutedModel) — the LEAKY "
                        "serving surface. Requires --embed_route; use with --delete_shard.")
    p.add_argument("--embed_route", default=None, choices=["sibling", "tombstone"],
                   help="sibling = deleted centroid removed (the leak); tombstone = kept as an "
                        "identity sentinel (the seal)")
    p.add_argument("--delete_shard", type=int, default=None)
    p.add_argument("--delete_shards", default=None,
                   help="Multi-unit deletion, e.g. '180-199' — 20 per-author units at k=200 are "
                        "one forget10. Oracle-routed arm (no --embed_route).")
    p.add_argument("--reroute_to", type=int, default=None,
                   help="E5 privacy column: delete NOTHING and serve the deleted authors from "
                        "this fixed surviving expert. Built via OODAwareRoutedModel so the MIA "
                        "and eval_routed_scaffold's forget_quality describe the same model.")
    p.add_argument("--lazy_adapter_cache", type=int, default=0,
                   help="Keep at most N shard adapters resident — required at k=200 r32.")
    p.add_argument("--router_encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    # MIA-specific
    p.add_argument("--attacks", default="loss,min_k,min_k++,zlib",
                   help="comma list from loss,min_k,min_k++,zlib")
    p.add_argument("--member_split", default="forget10")
    p.add_argument("--holdout_split", default="holdout10")
    p.add_argument("--mia_label_scope", default="answer", choices=["answer", "full"])
    p.add_argument("--min_k_frac", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dump_scores", action="store_true",
                   help="also write per-example score arrays (member + holdout, dataset "
                        "order) into each per_attack block — the input attack_diff.py "
                        "needs. Default OFF: output byte-identical to the pre-flag format.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def score_dataset(attack, model, ds, collator, batch_size=1, tokenizer=None, k=0.4):
    """Per-example scores in dataset order — the same loop mia_attacks.mia_auc runs
    internally. Replicated here (not added to the shared scorer lib, which the in-flight
    audit jobs depend on byte-identically) so --dump_scores can expose the arrays."""
    from torch.utils.data import DataLoader

    out = []
    for batch in DataLoader(ds, batch_size=batch_size, collate_fn=collator):
        batch.pop("index", None)
        out.extend(mia.score_batch(attack, model, batch, tokenizer=tokenizer, k=k))
    return out


def auc_with_scores(attack, model, member_ds, holdout_ds, collator, batch_size=1,
                    tokenizer=None, k=0.4):
    """mia_attacks.mia_auc's exact result dict (same scores, labels, roc_auc_score call)
    plus the raw per-example arrays. One forward pass per example, like mia_auc."""
    from sklearn.metrics import roc_auc_score

    fs = score_dataset(attack, model, member_ds, collator, batch_size, tokenizer, k)
    hs = score_dataset(attack, model, holdout_ds, collator, batch_size, tokenizer, k)
    scores = np.array(fs + hs, dtype="float64")
    labels = np.array([0] * len(fs) + [1] * len(hs))
    return {"auc": float(roc_auc_score(labels, scores)),
            "n_member": len(fs), "n_holdout": len(hs),
            "member_mean": float(np.mean(fs)), "holdout_mean": float(np.mean(hs)),
            "member_scores": [float(s) for s in fs],
            "holdout_scores": [float(s) for s in hs]}


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    forget_id = args.forget_shard_id if args.forget_shard_id is not None else args.k - 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    data = et.load_tofu_data(args.hf_home)
    if args.reroute_to is not None or args.delete_shards:
        # E5 privacy column (selector_audit): the ORACLE-routed reroute arm, built exactly as
        # eval_routed_scaffold builds it so the MIA and the forget_quality describe one model.
        from eval_routed_scaffold import OODAwareRoutedModel
        from legonet_tofu import build_q2author
        from shard_utils import parse_author_ids
        if not args.shards_dir:
            raise SystemExit("--reroute_to / --delete_shards need --shards_dir")
        if args.embed_route:
            raise SystemExit("--reroute_to is an oracle-route arm; --embed_route already "
                             "reassigns orphans by nearest surviving centroid")
        delete_set = parse_author_ids(args.delete_shards) if args.delete_shards else None
        model, tokenizer = et.load_all_shard_adapters(args.model_name, args.shards_dir, args.k,
                                                      lazy_cache=args.lazy_adapter_cache)
        q2a = build_q2author(data["full"], 200, 200 // args.k)
        eval_model = OODAwareRoutedModel(model, tokenizer, q2a, args.k,
                                         delete_shard=delete_set if delete_set is not None
                                         else args.delete_shard,
                                         reroute_to=args.reroute_to)
        adapter_name = (f"routed_reroute_s{args.reroute_to}" if args.reroute_to is not None
                        else f"routed_oracle_del{len(delete_set)}units")
    elif args.shards_dir or args.embed_route:
        if not (args.shards_dir and args.embed_route):
            raise SystemExit("--shards_dir and --embed_route must be given together")
        from eval_routed_scaffold import build_shard_centroids, EmbedRoutedModel
        from legonet_tofu import build_q2author
        model, tokenizer = et.load_all_shard_adapters(args.model_name, args.shards_dir, args.k,
                                                      lazy_cache=args.lazy_adapter_cache)
        q2a = build_q2author(data["full"], 200, 200 // args.k)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cents, sids, embed_fn = build_shard_centroids(
            args.hf_home, args.k, list(range(args.k)), dev, encoder_name=args.router_encoder)
        eval_model = EmbedRoutedModel(model, tokenizer, q2a, args.k, cents, sids, embed_fn,
                                      delete_shard=args.delete_shard, policy=args.embed_route)
        adapter_name = (f"embedrouted_{args.embed_route}"
                        + (f"_del{args.delete_shard}" if args.delete_shard is not None else ""))
    else:
        eval_model, tokenizer, adapter_name = et.build_served_model(args, data, forget_id)
    eval_model.eval()

    member_raw = load_dataset("locuslab/TOFU", args.member_split)["train"]
    holdout_raw = load_dataset("locuslab/TOFU", args.holdout_split)["train"]
    mk = lambda ds: QADataset(ds, tokenizer, args.mia_label_scope, args.max_length)
    member_ds, holdout_ds = mk(member_raw), mk(holdout_raw)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    per_attack = {}
    for atk in attacks:
        if args.dump_scores:
            res = auc_with_scores(atk, eval_model, member_ds, holdout_ds, collate,
                                  batch_size=args.batch_size, tokenizer=tokenizer,
                                  k=args.min_k_frac)
        else:
            res = mia.mia_auc(atk, eval_model, member_ds, holdout_ds, collate,
                              batch_size=args.batch_size, tokenizer=tokenizer,
                              k=args.min_k_frac)
        per_attack[atk] = res
        print(f"[attack_mia] {args.label:24s} {atk:8s} AUC={res['auc']:.4f} "
              f"(member_mean={res['member_mean']:.4f} holdout_mean={res['holdout_mean']:.4f})",
              flush=True)

    out = {
        "label": args.label, "adapter": adapter_name, "model_name": args.model_name,
        "member_split": args.member_split, "holdout_split": args.holdout_split,
        "label_scope": args.mia_label_scope, "min_k_frac": args.min_k_frac,
        "batch_size": args.batch_size, "seed": args.seed,
        "n_member": len(member_ds), "n_holdout": len(holdout_ds),
        "per_attack": per_attack,
        "metrics_version": "mia-2026-07-03",
        "attack_mia_sha": _git_or_sha(os.path.abspath(__file__)),
        "mia_attacks_sha": _git_or_sha(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "mia_attacks.py")),
    }
    if args.dump_scores:   # key only present when arrays were dumped (default byte-identical)
        out["dump_scores"] = True
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
