"""ClAMU driver: cluster, build the merged model + per-cluster sums, optimize cluster
masks, and exactly unlearn tasks.

Subcommands
-----------
setup    : embed the 200 authors (MiniLM) and cluster them into K groups (feature
           k-means or random). Frozen before finetuning -> cascade-free. Writes
           clamu/author_emb.npy + clamu/assignment_K{K}.json.

build    : full-FT each author-task deterministically (NO sign constraint), stream
           τ̄ = Σ_t τ_t AND per-cluster sums τ_c = Σ_{a∈c} τ_a (the τ_c enable the
           EMR/TALL baselines). Writes clamu/tau_bar.pt + clamu/cluster_sums/tau_c{c}.pt
           + meta.json. Discards per-task weights (streaming).

localize : optimize each cluster's binary mask via the STE (clamu.optimize_mask_ste),
           and also derive the cheap EMR/TALL baseline masks from τ_c (full only).
           Writes clamu/masks[_<tag>]/{clamu,emr,tall}_{c}.pt. Run with --tag to build
           the post-deletion retain masks. Per-cluster -> parallelizable (--cluster J).

unlearn  : exact O(1) deletion. Deterministically re-derive each forget task's τ_u,
           subtract from τ̄ -> τ̄_<tag> (= the retain sum). Re-cluster the RETAIN authors
           only (forget data never seen) -> assignment_<tag>. Then run `localize --tag`
           to rebuild the masks on retain data. Writes tau_bar_<tag>.pt + assignment_<tag>
           + unlearn_<tag>.json.

Determinism (required for exactness): per-task seed = base_seed + author; full-batch GD
from the same θ0; fp32; deterministic kernels. Re-deriving a task in `unlearn` reproduces
the exact τ_u that went into τ̄ — so τ̄_<tag> is the exact retain sum (up to fp
non-associativity, same caveat as SIFT).

Usage:
  python train_clamu.py setup    --config configs/clamu_tofu_1b.json
  python train_clamu.py build    --config configs/clamu_tofu_1b.json
  python train_clamu.py localize --config configs/clamu_tofu_1b.json
  python train_clamu.py unlearn  --config configs/clamu_tofu_1b.json --tag forget10
  python train_clamu.py localize --config configs/clamu_tofu_1b.json --tag forget10
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import clamu as cl
import sift_masks as sm
import sift_masks_data as smd
from train_sift_masks import _git_hash, load_base, load_config


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _forget_authors(cfg, tag):
    forget = cfg.get("unlearn_tags", {}).get(tag)
    if forget is None:
        if tag == "forget10":
            return list(smd.FORGET10_AUTHORS)
        raise SystemExit(f"unknown tag {tag}; add it to config['unlearn_tags']")
    return list(forget)


def _train_one(model, theta0, names, tok, full, author, cfg, device):
    """Deterministic full-FT of one author-task -> τ_a (no sign constraint)."""
    batch = smd.build_task_batch(
        tok, smd.author_records(full, author), loss_on=cfg.get("loss_on", "answer"),
        max_length=cfg.get("max_length", 256))
    tau, _ = sm.sift_one_task(
        model, theta0, None, names, batch, seed=cfg["seed"] + author,
        steps=cfg["steps"], lr=cfg["lr"], device=device, use_sign_constraint=False)
    return tau


# ── setup: cluster the authors ───────────────────────────────────────────────────

def cmd_setup(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cl.clamu_dir(cfg), exist_ok=True)
    full = smd.load_tofu_full(cfg["hf_home"])
    emb = cl.compute_author_embeddings(cfg, full, device=device)
    assign = cl.cluster_authors(cfg, emb)
    print(f"[setup] K={cfg['num_clusters']} method={assign['method']} "
          f"sizes(min/max)={assign['size_min']}/{assign['size_max']} "
          f"empty={assign['empty_clusters']} -> {cl.assignment_path(cfg)}", flush=True)


# ── build: FT all authors, accumulate τ̄ + per-cluster sums ───────────────────────

def cmd_build(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(cl.clamu_dir(cfg), "cluster_sums"), exist_ok=True)
    with open(cl.assignment_path(cfg)) as f:
        a2c = {int(a): int(c) for a, c in json.load(f)["author_to_cluster"].items()}

    # Resume: skip the (expensive) full-FT pass if τ̄ already exists — plus all K cluster
    # sums unless heuristic_masks=false (K-dial dirs symlink the K-independent τ̄ and skip
    # EMR/TALL entirely, so no per-K sums are ever built there).
    need_csums = cfg.get("heuristic_masks", True)
    if not args.overwrite and os.path.exists(cl.tau_bar_path(cfg)) and \
            (not need_csums or
             all(os.path.exists(cl.cluster_sum_path(cfg, c)) for c in range(cfg["num_clusters"]))):
        print(f"[build] tau_bar.pt{' + cluster sums' if need_csums else ''} already exist -> skip "
              f"(use --overwrite to rebuild)", flush=True)
        return

    tok = smd.load_gpt2_tokenizer(cfg["model_name"], cfg["hf_home"])
    model, names, theta0 = load_base(cfg, device)
    full = smd.load_tofu_full(cfg["hf_home"])
    T, K = cfg["num_authors"], cfg["num_clusters"]

    tau_bar = {n: torch.zeros_like(theta0[n], device="cpu") for n in names}
    cluster_sum = {c: {n: torch.zeros_like(theta0[n], device="cpu") for n in names}
                   for c in range(K)}
    t0 = time.time()
    for a in range(T):
        tau = _train_one(model, theta0, names, tok, full, a, cfg, device)
        c = a2c[a]
        for n in names:
            t_cpu = tau[n].to("cpu")
            tau_bar[n] += t_cpu
            cluster_sum[c][n] += t_cpu
        if a % 10 == 0 or a == T - 1:
            print(f"[build] author {a:3d}/{T}  cluster={c}  elapsed={time.time()-t0:6.1f}s",
                  flush=True)

    torch.save(tau_bar, cl.tau_bar_path(cfg))
    for c in range(K):
        torch.save(cluster_sum[c], cl.cluster_sum_path(cfg, c))
    meta = {
        "method": "clamu", "model_name": cfg["model_name"], "num_authors": T,
        "num_clusters": K, "steps": cfg["steps"], "lr": cfg["lr"], "seed": cfg["seed"],
        "use_sign_constraint": False, "loss_on": cfg.get("loss_on", "answer"),
        "frozen_substr": list(cfg.get("frozen_substr", sm.GPT2_FROZEN_SUBSTR)),
        "trainable_params": int(sum(theta0[n].numel() for n in names)),
        "git_hash": _git_hash(), "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(cl.clamu_dir(cfg), "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build] wrote tau_bar.pt + {K} cluster sums -> {cl.clamu_dir(cfg)}", flush=True)


# ── localize: optimize cluster masks (+ EMR/TALL baselines on the full model) ─────

def cmd_localize(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or None
    os.makedirs(cl.mask_dir(cfg, tag), exist_ok=True)
    with open(cl.assignment_path(cfg, tag)) as f:
        assignment = json.load(f)
    members = {int(c): v for c, v in assignment["members"].items()}
    T = len(assignment["authors"])

    tok = smd.load_gpt2_tokenizer(cfg["model_name"], cfg["hf_home"])
    model, names, theta0 = load_base(cfg, device)
    full = smd.load_tofu_full(cfg["hf_home"])
    tau_bar = {n: v.to(device) for n, v in torch.load(cl.tau_bar_path(cfg, tag)).items()}

    clusters = [args.cluster] if args.cluster is not None else sorted(members)
    for c in clusters:
        mem = members[c]
        if not mem:
            print(f"[localize{':'+tag if tag else ''}] cluster {c} empty -> skip", flush=True)
            continue
        # optimized ClAMU mask (always; the method)
        batches = cl.cluster_member_batches(cfg, tok, full, mem)
        mask = cl.optimize_mask_ste(
            model, names, tau_bar, batches, T=T, steps=cl.localize_steps(cfg, len(mem)),
            lr=cfg.get("mask_lr", 0.05), seed=cfg.get("mask_opt_seed", 42) + c,
            device=device, log_every=cfg.get("mask_log_every", 0),
            batch_rows=cfg.get("mask_batch_rows"))
        torch.save(sm.pack_mask(mask, names), cl.mask_path(cfg, "clamu", c, tag))
        active = int(sum(int(mask[n].sum()) for n in names))
        # EMR/TALL heuristic baselines from the per-cluster sum (full model only — the
        # retain clustering's τ_c is not re-accumulated; clamu_unlearn needs no τ_c).
        csum_path = cl.cluster_sum_path(cfg, c, tag)
        if os.path.exists(csum_path):
            tau_c = torch.load(csum_path)
            tau_bar_cpu = {n: tau_bar[n].cpu() for n in names}
            torch.save(sm.pack_mask(cl.emr_mask(tau_c, tau_bar_cpu, names), names),
                       cl.mask_path(cfg, "emr", c, tag))
            torch.save(sm.pack_mask(cl.tall_mask(tau_c, tau_bar_cpu, names,
                                                 lam=cfg.get("tall_lambda", 0.4)), names),
                       cl.mask_path(cfg, "tall", c, tag))
        print(f"[localize{':'+tag if tag else ''}] cluster {c} ({len(mem)} authors) "
              f"clamu_active={active}", flush=True)


# ── unlearn: subtract τ_u, re-cluster the retain authors ─────────────────────────

def cmd_unlearn(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag
    forget = _forget_authors(cfg, tag)
    forget_set = set(forget)

    full = smd.load_tofu_full(cfg["hf_home"])
    # Resume: τ̄_<tag> is K-independent (deterministic re-derivation gives the same file),
    # so K-dial dirs symlink it and skip the subtract; the per-K retain re-cluster below
    # always runs.
    if os.path.exists(cl.tau_bar_path(cfg, tag)) and not args.overwrite:
        print(f"[unlearn:{tag}] tau_bar_{tag}.pt already exists -> skip subtract", flush=True)
    else:
        tok = smd.load_gpt2_tokenizer(cfg["model_name"], cfg["hf_home"])
        model, names, theta0 = load_base(cfg, device)
        tau_bar = torch.load(cl.tau_bar_path(cfg))             # CPU dict (full sum)
        for a in forget:
            tau_u = _train_one(model, theta0, names, tok, full, a, cfg, device)
            for n in names:
                tau_bar[n] -= tau_u[n].to("cpu")              # exact subtraction
            print(f"[unlearn:{tag}] subtracted author {a}", flush=True)
        torch.save(tau_bar, cl.tau_bar_path(cfg, tag))

    retain = [a for a in range(cfg["num_authors"]) if a not in forget_set]
    emb = cl.compute_author_embeddings(cfg, full, device=device)   # cached from setup
    assign = cl.cluster_authors(cfg, emb, authors=retain, tag=tag)
    manifest = {
        "tag": tag, "forgotten_authors": list(forget), "num_authors_after": len(retain),
        "num_clusters": cfg["num_clusters"], "retain_cluster_sizes": assign["sizes"],
        "git_hash": _git_hash(),
    }
    with open(os.path.join(cl.clamu_dir(cfg), f"unlearn_{tag}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[unlearn:{tag}] wrote tau_bar_{tag}.pt + assignment (T={len(retain)}); "
          f"now run: localize --tag {tag}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["setup", "build", "localize", "unlearn"])
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default=None, help="unlearn/localize deletion tag (e.g. forget10)")
    p.add_argument("--cluster", type=int, default=None, help="localize: only this cluster")
    p.add_argument("--overwrite", action="store_true", help="build: force a rebuild even if τ̄ exists")
    args = p.parse_args()
    if args.cmd == "unlearn" and not args.tag:
        args.tag = "forget10"

    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = load_config(args.config)
    os.makedirs(cl.clamu_dir(cfg), exist_ok=True)
    {"setup": cmd_setup, "build": cmd_build,
     "localize": cmd_localize, "unlearn": cmd_unlearn}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
