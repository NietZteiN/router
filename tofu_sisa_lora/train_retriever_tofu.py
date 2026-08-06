"""Contrastive fine-tune of the RAMoLE-TOFU retriever (LoraRetriever Stage 1) on retain authors.

The off-the-shelf instructor-xl embed routing is the weak link on TOFU (templated questions don't
separate authors), so embed-RAG forget_quality lagged the oracle author lookup. Here we fine-tune the
encoder with InfoNCE / in-batch negatives on **same-author** question pairs from the RETAIN authors
(0..retain-1) — so a question is pulled toward its author's region and routes to the expert holding
that author. Retain-only ⇒ the retriever never sees deleted authors (consistent with the router).

Saves the fine-tuned encoder to `{output_dir}/legonet/ramole/retriever`; `ramole_tofu._encoder_source`
then picks it up automatically for the expert index and query embeddings.

    python train_retriever_tofu.py --config configs/ramole_tofu_1b.json --device cuda
"""
import argparse
import os
import sys

import numpy as np

import legonet_tofu as lt
import ramole_tofu as rt

RAMOLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ramole")
if RAMOLE_DIR not in sys.path:
    sys.path.insert(0, RAMOLE_DIR)
import ramole_common as rc       # noqa: E402


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


def train_retriever_tofu(cfg: dict, device: str = "cuda", data_full=None) -> str:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    rc.set_determinism(int(cfg.get("base_seed", 42)))
    if data_full is None:
        from datasets import load_dataset
        data_full = load_dataset("locuslab/TOFU", "full")["train"]
    retain = int(cfg.get("ramole_train", {}).get("retain_authors", 180))
    per = int(cfg["records_per_author"])
    rcfg = cfg.get("retriever_train", {})
    epochs = int(rcfg.get("epochs", 2))
    lr = float(rcfg.get("lr", 2e-5))
    bs = int(rcfg.get("batch_size", 4))       # small batch: instructor-xl full FT OOMs at 16 on 44 GiB

    instr = rt._instr(cfg)
    prefix = f"{instr}: " if instr else ""
    rng = np.random.RandomState(int(cfg.get("base_seed", 42)))
    examples = []
    for a in range(retain):
        qs = [data_full[a * per + i]["question"] for i in range(per)]
        for i in range(len(qs)):
            j = i
            while j == i:
                j = int(rng.randint(len(qs)))
            examples.append(InputExample(texts=[prefix + qs[i], prefix + qs[j]]))
    rng.shuffle(examples)

    enc = SentenceTransformer(rt._encoder_name(cfg), device=device)   # start from the BASE encoder
    loader = DataLoader(examples, batch_size=bs, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(enc)
    warmup = int(0.1 * len(loader) * epochs)
    # use_amp MUST be False: instructor-xl is T5-based and overflows under fp16 autocast
    # (loss→0, grad_norm→nan, no weight update). fp32 at this batch size fits the 44 GiB card.
    print(f"[train_retriever_tofu] pairs={len(examples)} retain_authors={retain} "
          f"epochs={epochs} bs={bs} dev={device} (fp32, no AMP)", flush=True)
    enc.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=warmup,
            optimizer_params={"lr": lr}, show_progress_bar=False, use_amp=False)
    out = rt.retriever_dir(cfg)
    os.makedirs(out, exist_ok=True)
    enc.save(out)
    print(f"[train_retriever_tofu] fine-tuned encoder -> {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = lt.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    train_retriever_tofu(cfg, device=args.device)


if __name__ == "__main__":
    main()
