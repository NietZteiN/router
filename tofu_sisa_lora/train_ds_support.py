"""[ds] trainer: disjoint-support full-FT task vector for ONE pool author.

Mirrors train_sift_masks.py's build recipe — full-batch Adam over the author's 20 TOFU
records (SIFT App-B: bs = the whole task), fp32, deterministic seeds, project AFTER each
step — with ds_support.project_support_model_ replacing the sign projection.

Budget: one full-batch step == one epoch, so the canonical config's train.epochs=25 is
25 Adam steps at train.lr (the sift config's steps=20 was likewise 20 full-batch epochs).

The author pool derives at runtime from merge_subset.subset_authors(pool_seed, pool_size)
(never hardcoded); --author is the TOFU author id and must be a pool member; its pool
SLOT (position in the subset list) selects the disjoint-support block.

Usage:
  python train_ds_support.py --config configs/ctv_1b_ds.json --author 82 \
      [--density 0.005] [--overwrite] [--no_support]

Artifacts under {out_dir}/ds/tau_a{author}[_d{density}]/ :
  tau_sparse.pt   # {name: {idx int32, val fp32, shape}} — τ_a, exact on S_a
  meta.json       # density, support_seed, seed, steps, lr, script sha256, telemetry
Deletion and serving consume these dirs directly (ds_support.bake_merged) — no
re-derivation, no mask files. Density overrides get their own dir (…_d{density}):
mixed-density taus must never share a bake (bake_merged asserts this).

--no_support (H-ds-1 comparator denominator): SAME recipe, NO support projection —
the unconstrained full-FT solo. Its dense τ (~4 GB fp32 on the 1B) is never stored;
θ0+τ is baked IN-JOB to {out_dir}/ds_unconstrained/a{author}_model/ (model +
tokenizer + meta.json, "comparator_for": "H-ds-1") via ds_support.bake_dense_model
and served through the model: eval rows (eval_baseline path). No tau_sparse.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import torch

import ds_support as ds
import sift_masks as sm
import sift_masks_data as smd
from merge_subset import subset_authors
from train_sift_masks import load_base   # fp32 + eager/deterministic attention, θ0 snapshot

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

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

DEFAULT_HF_HOME = os.environ["HF_HOME"]
DEFAULT_FROZEN = ("embed_tokens", "lm_head")   # Llama-3.2-1B ties embeddings (sift config)


def load_config(path: str) -> dict:
    _ensure_site_env()
    with open(path) as f:
        cfg = _expand_site_paths(json.load(f))
    for key in ("model_name", "out_dir", "arm", "pool_seed", "pool_size",
                "train", "support_seed", "density"):
        if key not in cfg:
            raise KeyError(f"config missing {key!r}")
    if cfg["arm"] != "ds":
        raise ValueError(f"config arm {cfg['arm']!r} != 'ds'")
    # train_sift_masks.load_base reads these two keys; make them explicit here.
    cfg.setdefault("hf_home", DEFAULT_HF_HOME)
    cfg.setdefault("frozen_substr", list(DEFAULT_FROZEN))
    return cfg


def ds_dir(cfg) -> str:
    return os.path.join(cfg["out_dir"], "ds")


def tau_dir(cfg, author: int, density: float = None) -> str:
    """Per-author artifact dir; non-config densities (the density_sweep) get a suffix."""
    d = cfg["density"] if density is None else density
    suffix = "" if d == cfg["density"] else f"_d{d:g}"
    return os.path.join(ds_dir(cfg), f"tau_a{author}{suffix}")


def pool_authors(cfg) -> list:
    return subset_authors(cfg["pool_seed"], cfg["pool_size"])


def pool_slot(cfg, author: int) -> int:
    pool = pool_authors(cfg)
    if author not in pool:
        raise SystemExit(f"author {author} not in the pool "
                         f"(subset_authors({cfg['pool_seed']}, {cfg['pool_size']}) = {pool})")
    return pool.index(author)


def train_steps(cfg) -> int:
    # e25-equivalent budget: full-batch training makes one Adam step one epoch.
    return int(cfg["train"]["epochs"])


def _file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _resume_key(cfg, author: int, density: float) -> dict:
    return {
        "support_seed": cfg["support_seed"], "density": density,
        "mlp_only": bool(cfg.get("mlp_only", False)),
        "pool_seed": cfg["pool_seed"], "pool_size": cfg["pool_size"],
        "seed": cfg["train"]["seed"] + author,
        "steps": train_steps(cfg), "lr": cfg["train"]["lr"],
    }


def unc_model_dir(cfg, author: int) -> str:
    """--no_support (H-ds-1 comparator) baked-model dir for one probe author."""
    return os.path.join(cfg["out_dir"], "ds_unconstrained", f"a{author}_model")


def _unc_resume_key(cfg, author: int) -> dict:
    # no support_seed/density/mlp_only: the comparator has no support constraint
    return {
        "no_support": True,
        "pool_seed": cfg["pool_seed"], "pool_size": cfg["pool_size"],
        "seed": cfg["train"]["seed"] + author,
        "steps": train_steps(cfg), "lr": cfg["train"]["lr"],
    }


def train_and_bake_unconstrained(cfg, author: int, model, names, theta0, batch,
                                 device: str = "cpu", loss_log: list = None) -> str:
    """--no_support core (split from main so test_ds_support can drive it with a tiny
    fixture): train the SAME recipe with support=None (no projection), then bake
    θ0+τ IN-JOB to unc_model_dir via ds_support.bake_dense_model — the dense τ is
    never written to disk (a sparse dump would be ~the model size). Writes model +
    tokenizer + meta.json ("comparator_for": "H-ds-1"); returns the baked dir."""
    key = _unc_resume_key(cfg, author)
    losses = [] if loss_log is None else loss_log
    tau = ds.ds_one_task(
        model, theta0, None, names, batch,
        seed=key["seed"], steps=key["steps"], lr=key["lr"], device=device,
        loss_log=losses)
    tau = {n: t.cpu() for n, t in tau.items()}
    assert losses and math.isfinite(losses[-1]), f"non-finite final loss {losses[-1:]}"

    outd = unc_model_dir(cfg, author)
    frozen = tuple(cfg.get("frozen_substr", DEFAULT_FROZEN))
    ds.bake_dense_model(outd, cfg["model_name"], tau, frozen_substr=frozen,
                        hf_home=cfg.get("hf_home"))
    meta = {
        "arm": "ds", "method": "ds_unconstrained", "full_ft": True,
        "comparator_for": "H-ds-1",
        "model_name": cfg["model_name"],
        "author": author, "slot": pool_slot(cfg, author),
        **key,
        "loss_on": cfg.get("loss_on", "answer"),
        "max_length": cfg.get("max_length", 256),
        "frozen_substr": list(frozen),
        "trainable_params": int(sum(theta0[n].numel() for n in names)),
        "loss_first": losses[0], "loss_last": losses[-1],
        "script_sha256": _file_sha(os.path.abspath(__file__)),
        "ds_support_sha256": _file_sha(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ds_support.py")),
    }
    with open(os.path.join(outd, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ds] author {author} UNCONSTRAINED comparator "
          f"loss {losses[0]:.4f}->{losses[-1]:.4f} baked -> {outd}")
    return outd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--author", type=int, required=True, help="TOFU author id (pool member)")
    p.add_argument("--density", type=float, default=None,
                   help="override config density (density_sweep point)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_support", action="store_true",
                   help="H-ds-1 comparator: SAME recipe with NO support projection "
                        "(unconstrained full-FT solo). Bakes theta0+tau to "
                        "{out_dir}/ds_unconstrained/a{author}_model/ instead of storing "
                        "a sparse tau; served via the model: eval rows (eval_baseline).")
    args = p.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = load_config(args.config)
    if args.no_support and args.density is not None:
        raise SystemExit("[ds] --density has no effect with --no_support (no support mask)")
    density = cfg["density"] if args.density is None else args.density
    author = args.author
    if args.no_support:
        outd = unc_model_dir(cfg, author)
        key = _unc_resume_key(cfg, author)
    else:
        outd = tau_dir(cfg, author, density)
        key = _resume_key(cfg, author, density)
    meta_path = os.path.join(outd, "meta.json")

    # Resume guard: identical provenance -> skip; mismatched artifacts refuse silently
    # becoming part of a bake (stale-density taus would break disjointness).
    if os.path.exists(meta_path) and not args.overwrite:
        with open(meta_path) as f:
            old = json.load(f)
        if all(old.get(k) == v for k, v in key.items()):
            print(f"[ds] {'bake' if args.no_support else 'tau'} exists with matching "
                  f"provenance, skipping -> {outd}")
            return
        raise SystemExit(f"[ds] {outd} exists with DIFFERENT provenance "
                         f"({ {k: old.get(k) for k in key} } != {key}); pass --overwrite")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    slot = pool_slot(cfg, author)
    tok = smd.load_gpt2_tokenizer(cfg["model_name"], cfg["hf_home"])
    model, names, theta0 = load_base(cfg, device)
    if not args.no_support:
        shapes = ds.shapes_from_model(model, names)
        support = ds.support_mask_for_slot(
            cfg["support_seed"], slot, cfg["pool_size"], density, shapes,
            mlp_only=bool(cfg.get("mlp_only", False)))

    full = smd.load_tofu_full(cfg["hf_home"])
    records = smd.author_records(full, author)
    batch = smd.build_task_batch(
        tok, records, loss_on=cfg.get("loss_on", "answer"),
        max_length=cfg.get("max_length", 256))

    if args.no_support:
        train_and_bake_unconstrained(cfg, author, model, names, theta0, batch,
                                     device=device)
        return

    losses = []
    tau = ds.ds_one_task(
        model, theta0, support, names, batch,
        seed=key["seed"], steps=key["steps"], lr=key["lr"], device=device,
        loss_log=losses)
    tau = {n: t.cpu() for n, t in tau.items()}

    # Verify (never assume) the locality claim before storing: τ must live entirely in
    # S_a at the BIT level (round-trip through the sparse form is the strongest check).
    sparse = ds.sparsify(tau, support)
    dense_back = ds.densify(sparse)
    for n in names:
        assert torch.equal(dense_back[n], tau[n]), f"τ escaped its support ({n})"
    energy = ds.energy_in_support(tau, support)
    assert abs(energy - 1.0) < 1e-12, f"energy_in_support {energy} != 1.0"
    assert losses and math.isfinite(losses[-1]), f"non-finite final loss {losses[-1:]}"

    os.makedirs(outd, exist_ok=True)
    ds.save_sparse_tau(os.path.join(outd, "tau_sparse.pt"), sparse)
    meta = {
        "arm": "ds", "method": "ds_support", "full_ft": True,
        "model_name": cfg["model_name"],
        "author": author, "slot": slot,
        **key,
        "loss_on": cfg.get("loss_on", "answer"),
        "max_length": cfg.get("max_length", 256),
        "frozen_substr": list(cfg["frozen_substr"]),
        "trainable_params": int(sum(theta0[n].numel() for n in names)),
        "support_indices": int(sum(v.numel() for v in support.values())),
        "sparse_mb": round(ds.sparse_nbytes(sparse) / 1e6, 2),
        "energy_in_support": energy,
        "loss_first": losses[0], "loss_last": losses[-1],
        "script_sha256": _file_sha(os.path.abspath(__file__)),
        "ds_support_sha256": _file_sha(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ds_support.py")),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ds] author {author} (slot {slot}) d={density:g} "
          f"loss {losses[0]:.4f}->{losses[-1]:.4f} "
          f"{meta['support_indices']} idx {meta['sparse_mb']} MB -> {outd}")


if __name__ == "__main__":
    main()
