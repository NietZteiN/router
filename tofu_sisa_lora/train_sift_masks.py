"""SIFT-Masks driver: build the merged model + masks, and exactly unlearn tasks.

Subcommands
-----------
build    : SIFT-finetune all T author-tasks, stream-accumulate τ̄ = Σ_t τ_t, and save
           each task's bit mask. Stores τ̄ + masks + the sign vector + meta. Discards
           per-task weights (streaming) so peak storage is one model + the masks.

unlearn  : Given a tag (e.g. forget10) and its author list, deterministically
           re-derive each forgotten task's τ_u, subtract from τ̄, and drop its mask.
           Writes τ̄_<tag> + a manifest. This is the *exact* O(1)-per-task deletion.

Determinism (required for exactness): per-task seed = base_seed + author; full-batch
GD from the same θ0; fp32 weights; deterministic kernels. Re-deriving a task in
`unlearn` reproduces the exact τ_u that went into τ̄.

Memory (GPT2-XL, fp32): keep model + Adam + θ0 + sign on GPU (~28 GB, fits an A40);
accumulate τ̄ on CPU and stream masks to disk so we never hold 200 task vectors.

Artifacts under {output_dir}/sift/ :
  sign_v.pt              # ±1 sign vector (param dict)
  tau_bar.pt             # Σ_t τ_t  (param dict, fp32, CPU)
  tau_bar_<tag>.pt       # post-unlearn sum for a deletion tag
  masks/m_{author}.pt    # bit-packed per-author mask
  meta.json / unlearn_<tag>.json
Config = a JSON like configs/sift_masks_tofu.json (CLI flags override).

Usage:
  python train_sift_masks.py build   --config configs/sift_masks_tofu.json
  python train_sift_masks.py unlearn --config configs/sift_masks_tofu.json --tag forget10
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import sift_masks as sm
import sift_masks_data as smd

import sys

# ── site-path expansion (added on export) ────────────────────────────────────────────────────
# Configs used to carry absolute /storage2 paths. They now say "${TOFU_CKPT_ROOT}/..." etc, and
# this resolves them at load time, hard-erroring on an unset variable rather than writing a
# literal "${TOFU_CKPT_ROOT}" directory to disk (which is what happened before the guard).
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import expand_paths as _expand_site_paths, ensure_site_env as _ensure_site_env
except ImportError:                       # repo_env.py is at the repo root; absent => no-op
    def _expand_site_paths(o, _k=""): return o
    def _ensure_site_env(force=False): return {}


# Deterministic cuBLAS (must be set before CUDA context for full effect).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def load_config(path: str) -> dict:
    _ensure_site_env()
    with open(path) as f:
        return _expand_site_paths(json.load(f))


def sift_dir(cfg) -> str:
    return os.path.join(cfg["output_dir"], "sift")


def mask_path(cfg, author: int) -> str:
    return os.path.join(sift_dir(cfg), "masks", f"m_{author}.pt")


def _git_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def _force_deterministic_attention():
    """Exact unlearning needs deterministic finetuning (paper App A): the unlearn step
    must re-derive the SAME τ_u that went into τ̄. The default SDPA mem-efficient/flash
    attention backward is non-deterministic, so force the math backend (and eager attn at
    load time). CUDA-only; a no-op on CPU. Even so, full bitwise GPU exactness isn't
    guaranteed (reduction-order noise) — re-derivation is exact up to the hardware floor,
    which this tightens (cf. the legonet thread's distributional-exactness finding)."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def load_base(cfg, device):
    from transformers import AutoModelForCausalLM
    os.environ["HF_HOME"] = cfg["hf_home"]
    _force_deterministic_attention()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=torch.float32, trust_remote_code=True,
            attn_implementation="eager")          # deterministic attention path
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=torch.float32, trust_remote_code=True)
    model.to(device)
    names = sm.trainable_names(model, tuple(cfg.get("frozen_substr", sm.GPT2_FROZEN_SUBSTR)))
    theta0 = sm.snapshot_params(model, names)        # on `device`
    return model, names, theta0


def _train_one(model, theta0, sign, names, tok, full, author, cfg, device):
    records = smd.author_records(full, author)
    batch = smd.build_task_batch(
        tok, records, loss_on=cfg.get("loss_on", "answer"),
        max_length=cfg.get("max_length", 256))
    seed = cfg["seed"] + author
    return sm.sift_one_task(
        model, theta0, sign, names, batch,
        seed=seed, steps=cfg["steps"], lr=cfg["lr"], device=device)


def cmd_build(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(sift_dir(cfg), "masks"), exist_ok=True)

    tok = smd.load_gpt2_tokenizer(cfg["model_name"], cfg["hf_home"])
    model, names, theta0 = load_base(cfg, device)
    sign = sm.make_sign_vector(model, names, cfg["sign_seed"])
    torch.save({n: sign[n].cpu() for n in names}, os.path.join(sift_dir(cfg), "sign_v.pt"))

    full = smd.load_tofu_full(cfg["hf_home"])
    T = cfg["num_authors"]
    tau_bar = {n: torch.zeros_like(theta0[n], device="cpu") for n in names}  # accumulate on CPU

    done = []
    t0 = time.time()
    for a in range(T):
        if args.author is not None and a != args.author:
            continue
        if os.path.exists(mask_path(cfg, a)) and not args.overwrite:
            # Mask already saved; still need its τ in τ̄. To keep build resumable we
            # re-derive (deterministic) and add — cheap (20 steps).
            tau, _ = _train_one(model, theta0, sign, names, tok, full, a, cfg, device)
            for n in names:
                tau_bar[n] += tau[n].to("cpu")
            done.append(a)
            continue
        tau, mask = _train_one(model, theta0, sign, names, tok, full, a, cfg, device)
        for n in names:
            tau_bar[n] += tau[n].to("cpu")
        torch.save(sm.pack_mask(mask, names), mask_path(cfg, a))
        done.append(a)
        if a % 10 == 0 or a == T - 1:
            active = int(sum(int(mask[n].sum()) for n in names))
            print(f"[build] author {a:3d}/{T}  active={active:>10d}  "
                  f"elapsed={time.time()-t0:6.1f}s", flush=True)

    torch.save(tau_bar, os.path.join(sift_dir(cfg), "tau_bar.pt"))
    meta = {
        "method": "sift_masks", "model_name": cfg["model_name"],
        "num_authors": T, "steps": cfg["steps"], "lr": cfg["lr"],
        "seed": cfg["seed"], "sign_seed": cfg["sign_seed"],
        "loss_on": cfg.get("loss_on", "answer"), "frozen_substr": list(
            cfg.get("frozen_substr", sm.GPT2_FROZEN_SUBSTR)),
        "trainable_params": int(sum(theta0[n].numel() for n in names)),
        "authors_done": done, "git_hash": _git_hash(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(sift_dir(cfg), "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build] wrote tau_bar.pt + {len(done)} masks -> {sift_dir(cfg)}", flush=True)


def cmd_unlearn(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag
    forget = cfg.get("unlearn_tags", {}).get(tag)
    if forget is None:
        if tag == "forget10":
            forget = smd.FORGET10_AUTHORS
        else:
            raise SystemExit(f"unknown tag {tag}; add it to config['unlearn_tags']")

    tok = smd.load_gpt2_tokenizer(cfg["model_name"], cfg["hf_home"])
    model, names, theta0 = load_base(cfg, device)
    sign_cpu = torch.load(os.path.join(sift_dir(cfg), "sign_v.pt"))
    sign = {n: sign_cpu[n].to(device) for n in names}

    tau_bar = torch.load(os.path.join(sift_dir(cfg), "tau_bar.pt"))  # CPU dict
    full = smd.load_tofu_full(cfg["hf_home"])

    for a in forget:
        tau_u, _ = _train_one(model, theta0, sign, names, tok, full, a, cfg, device)
        for n in names:
            tau_bar[n] -= tau_u[n].to("cpu")               # exact subtraction
        print(f"[unlearn:{tag}] subtracted author {a}", flush=True)

    torch.save(tau_bar, os.path.join(sift_dir(cfg), f"tau_bar_{tag}.pt"))
    manifest = {
        "tag": tag, "forgotten_authors": list(forget),
        "num_authors_after": cfg["num_authors"] - len(forget),
        "dropped_masks": [mask_path(cfg, a) for a in forget],
        "git_hash": _git_hash(),
    }
    with open(os.path.join(sift_dir(cfg), f"unlearn_{tag}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[unlearn:{tag}] wrote tau_bar_{tag}.pt (T={manifest['num_authors_after']})",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "unlearn"])
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default="forget10", help="unlearn: deletion tag")
    p.add_argument("--author", type=int, default=None, help="build: train only this author")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = load_config(args.config)
    os.makedirs(sift_dir(cfg), exist_ok=True)
    {"build": cmd_build, "unlearn": cmd_unlearn}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
