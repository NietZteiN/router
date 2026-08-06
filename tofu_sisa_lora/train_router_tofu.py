"""Train the RAMoLE RouterLoRA gate on the TOFU author-expert pool (router-only; experts frozen).

Mirrors `ramole/train_router.py` but on TOFU data: standard causal-LM loss on the **retain authors'**
Q&A (`0..retain_authors-1`) so the router never sees deleted data ⇒ deletion needs no router retrain.
Base + all expert LoRAs frozen; only `{A_r, B_r}` train, with Random LoRA Dropout p over all n experts
each step. Saves router-only weights → `{output_dir}/legonet/ramole/router.safetensors`.

    python train_router_tofu.py --config configs/ramole_tofu_1b.json --device cuda
"""
import argparse
import os
import sys

import torch

import legonet_tofu as lt
import ramole_tofu as rt

RAMOLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ramole")
if RAMOLE_DIR not in sys.path:
    sys.path.insert(0, RAMOLE_DIR)
import ramole_common as rc       # noqa: E402
import router_lora as R          # noqa: E402


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


def retain_texts(cfg_l: dict) -> list[str]:
    """`"Question: {q}\nAnswer: {a}"` for every retain-author record (authors 0..retain-1)."""
    from train_lora_shard import format_prompt, load_shard_dataset
    retain = int(cfg_l.get("ramole_train", {}).get("retain_authors", 180))
    authors = list(range(retain))
    ds = load_shard_dataset(authors, cfg_l["hf_home"]).map(
        format_prompt, remove_columns=["question", "answer"])
    return [r["text"] for r in ds]


def train_router_tofu(cfg_l: dict, device: str = "cuda", seed: int | None = None,
                      router_out: str | None = None) -> str:
    """seed/router_out override the config for the E1 seed-variance arm (the default router_path
    is fixed per pool, so seed variants MUST write to a distinct file)."""
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup

    seed = int(seed if seed is not None else cfg_l.get("base_seed", 42))
    rc.set_determinism(seed)
    tcfg = cfg_l.get("ramole_train", {})
    epochs = int(tcfg.get("epochs", 1))
    lr = float(tcfg.get("lr", 1e-4))
    batch_size = int(tcfg.get("batch_size", 1))
    grad_accum = int(tcfg.get("grad_accum", 8))
    max_length = int(tcfg.get("max_length", 256))
    dropout_p = float(tcfg.get("dropout_p", 0.5))
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.03))
    max_grad_norm = float(tcfg.get("max_grad_norm", 0.3))

    model, tok, controller, meta, installed = R.build_ramole_model(
        rt._ramole_build_cfg(cfg_l), device=device, load_router_weights=False,
        adapter_dir_fn=rt._full_adapter_dir_fn(cfg_l))
    controller.set_pool(list(range(cfg_l["n"])))   # dropout over all n experts
    model.train()

    texts = retain_texts(cfg_l)
    if not texts:
        raise RuntimeError("no retain-author training texts")
    enc = [tok(t, truncation=True, max_length=max_length) for t in texts]
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    g_data = torch.Generator().manual_seed(seed)
    loader = DataLoader(enc, batch_size=batch_size, shuffle=True, collate_fn=collator,
                        generator=g_data, drop_last=False)

    params = R.router_parameters(model)
    n_router = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(loader) + grad_accum - 1) // grad_accum
    total_steps = steps_per_epoch * epochs
    sched = get_cosine_schedule_with_warmup(opt, int(warmup_ratio * total_steps), total_steps)
    g_drop = torch.Generator().manual_seed(seed + 1)
    dev = next(model.parameters()).device
    print(f"[train_router_tofu] texts={len(texts)} n={cfg_l['n']} router_params={n_router} "
          f"({len(params)} tensors) total_opt_steps={total_steps} dropout_p={dropout_p} dev={dev}",
          flush=True)

    step, last_loss = 0, float("nan")
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        running, micro = 0.0, 0
        for i, batch in enumerate(loader):
            controller.sample_dropout(dropout_p, g_drop)
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            running += loss.item() * grad_accum
            micro += 1
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                last_loss = running / max(1, micro)
                if step % 20 == 0 or step == 1:
                    print(f"  epoch {epoch} step {step}/{total_steps} loss {last_loss:.4f}", flush=True)
                running, micro = 0.0, 0

    out = router_out or rt.router_path(cfg_l)
    R.save_router(model, out)
    rc.write_json(os.path.splitext(out)[0] + "_meta.json" if router_out
                  else os.path.join(rt.ramole_dir(cfg_l), "router_meta.json"), {
        "base_model": cfg_l["base_model"], "output_dir": cfg_l["output_dir"],
        "n": cfg_l["n"], "router_rank": rt._ramole_build_cfg(cfg_l)["router"]["rank"],
        "scaling": meta["scaling"], "dropout_p": dropout_p,
        "retain_authors": int(tcfg.get("retain_authors", 180)),
        "n_train_texts": len(texts), "num_router_params": n_router,
        "total_opt_steps": total_steps, "final_loss": last_loss, "base_seed": seed,
    })
    print(f"[train_router_tofu] saved router -> {out} (final loss {last_loss:.4f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None,
                    help="override base_seed (E1 seed variance); use with --router_out")
    ap.add_argument("--router_out", default=None,
                    help="override the save path (seed variants must not overwrite the default)")
    args = ap.parse_args()
    cfg_l = lt.load_config(args.config)
    os.environ["HF_HOME"] = cfg_l["hf_home"]
    train_router_tofu(cfg_l, device=args.device, seed=args.seed, router_out=args.router_out)


if __name__ == "__main__":
    main()
