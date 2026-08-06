"""Stage 2 training — fit the RouterLoRA gate (gamma = {A_r, B_r} per layer).

Standard causal-LM loss on the training-cluster data; base + all expert LoRAs frozen. Each
step applies Random LoRA Dropout (paper Eq.13, p=0.5): a random subset of the training-pool
experts is deactivated before the forward, which stops the router overfitting to the
route-to-the-ideal-LoRA (IID) shortcut and is what gives the OOD/zero-shot gains.

A plain deterministic AdamW loop (not HF Trainer): we must inject a fresh dropout mask before
every forward, which fights Trainer's owned loop.

Router-train data (cfg["router_train_split"]):
  reference — the DBpedia reference split (disjoint from the deletable corpus), routed to its
              cluster by the FROZEN KEYS (using the *source* MiniLM encoder that built them, not
              the Stage-1 instructor encoder). LegoNet Condition A: the router never sees a
              deletable record, so a deletion retrains only the affected experts and the router
              stays valid with NO retraining.
  corpus    — paper-faithful: the corpus's own training-cluster records (via the cached
              assignment). Note: strict per-record exactness would then also require excluding
              deleted records from router data.

    python train_router.py --config configs/ramole_l32_3b.json --device cuda
"""
import argparse
import json
import os

import numpy as np
import torch

import ramole_common as rc
import router_lora as R

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


def router_train_texts(cfg: dict, device: str) -> tuple[list[str], list[int]]:
    """Return (training texts, train_cluster_ids). Records whose top-1 cluster is a training
    cluster, formatted with legonet's `train_text`."""
    sp = rc.source_paths(cfg)
    train_clusters, _ = rc.cluster_split(cfg)
    tc = set(train_clusters)

    if cfg["router_train_split"] == "corpus":
        with open(sp.assignment_path) as f:
            r2k = json.load(f)["record_to_keys"]
        recs = rc.load_records(sp.records_path)
        keep = [r for r in recs if r["id"] in r2k and int(r2k[r["id"]][0]) in tc]
    elif cfg["router_train_split"] == "reference":
        from routing import KNNRouter  # legonet (already on sys.path via ramole_common)
        keys = np.load(sp.keys_path)
        with open(sp.keys_meta) as f:
            src_encoder = json.load(f)["encoder_model"]   # the encoder that built the keys
        embed = rc.make_embed_fn(src_encoder, instruction="", device=device)
        recs = rc.load_records(sp.reference_path)
        routed = KNNRouter(keys, cfg["k"]).route(embed([rc.route_text(r) for r in recs]))
        keep = [r for r, ks in zip(recs, routed) if int(ks[0]) in tc]
    else:
        raise ValueError(f"router_train_split must be reference|corpus, got {cfg['router_train_split']!r}")

    texts = [rc.train_text(r) for r in keep]
    return texts, sorted(tc)


def train_router(cfg: dict, device: str = "cuda") -> str:
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup,
    )

    paths = rc.Paths(cfg)
    paths.ensure()
    rc.set_determinism(cfg["base_seed"])

    texts, train_clusters = router_train_texts(cfg, device)
    if not texts:
        raise RuntimeError("no router-training texts (check cluster_split / source assignment)")

    model, tok, controller, meta, installed = R.build_ramole_model(
        cfg, device=device, load_router_weights=False)
    controller.set_pool(train_clusters)
    model.train()

    # tokenize once (deterministic order); collator builds causal-LM labels (pad -> -100)
    tcfg = cfg["train"]
    enc = [tok(t, truncation=True, max_length=tcfg["max_length"]) for t in texts]
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    g_data = torch.Generator().manual_seed(cfg["base_seed"])
    loader = DataLoader(enc, batch_size=tcfg["batch_size"], shuffle=True,
                        collate_fn=collator, generator=g_data, drop_last=False)

    params = R.router_parameters(model)
    n_router = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    accum = tcfg["grad_accum"]
    steps_per_epoch = (len(loader) + accum - 1) // accum
    total_steps = steps_per_epoch * tcfg["epochs"]
    sched = get_cosine_schedule_with_warmup(
        opt, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    g_drop = torch.Generator().manual_seed(cfg["base_seed"] + 1)

    dev = next(model.parameters()).device
    print(f"[train_router] texts={len(texts)} train_clusters={train_clusters} "
          f"router_params={n_router} ({len(params)} tensors) total_opt_steps={total_steps} dev={dev}",
          flush=True)

    step = 0
    last_loss = float("nan")
    for epoch in range(tcfg["epochs"]):
        opt.zero_grad(set_to_none=True)
        running, micro = 0.0, 0
        for i, batch in enumerate(loader):
            controller.sample_dropout(cfg["dropout_p"], g_drop)   # fresh mask per forward
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            running += loss.item() * accum
            micro += 1
            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                last_loss = running / max(1, micro)
                if step % 10 == 0 or step == 1:
                    print(f"  epoch {epoch} step {step}/{total_steps} loss {last_loss:.4f}", flush=True)
                running, micro = 0.0, 0

    R.save_router(model, paths.router_path)
    rc.write_json(paths.router_meta, {
        "base_model": cfg["base_model"], "source_run": cfg["source_run"],
        "n": meta["n"], "router_rank": cfg["router"]["rank"], "scaling": meta["scaling"],
        "router_train_split": cfg["router_train_split"], "train_clusters": train_clusters,
        "heldout_clusters": rc.cluster_split(cfg)[1], "dropout_p": cfg["dropout_p"],
        "n_train_texts": len(texts), "num_router_params": n_router,
        "total_opt_steps": total_steps, "final_loss": last_loss,
        "config_hash": rc.config_hash(cfg), "base_seed": cfg["base_seed"],
        "installed_paths": installed,
    })
    print(f"[train_router] saved router -> {paths.router_path} (final loss {last_loss:.4f})", flush=True)
    return paths.router_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = rc.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    train_router(cfg, device=args.device)


if __name__ == "__main__":
    main()
