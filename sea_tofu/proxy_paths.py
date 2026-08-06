"""Path helpers for SEA-on-TOFU per-author proxies.

Mirrors tofu_sisa_lora/model_paths.py. Layout (proxies/ is a symlink to
${TOFU_CKPT_STORE}/sea_tofu/proxies so artifacts never land on /home):

    proxies/{model_slug}[ _r{rank} ]/
        author_000/
            personal_lora/        # PEFT adapter (adapter_config.json + safetensors)
            meta.json             # author id, rank, train hash, seed, git commit
        author_001/ ...
        baselines/                # cached base-only metric artifacts (TR refs, dists)
        results/{smoke,full}/     # metric JSON / CSV

One author per directory == one SEA "user" with a deletable proxy. Deletion is rm of an
author_NNN/ dir; the shared 4-bit base is never touched.
"""
import os
import re

DEFAULT_PROXY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxies")


def model_slug(model_name: str) -> str:
    """HF repo id -> directory name (matches tofu_sisa_lora/model_paths.model_slug)."""
    slug = model_name.split("/")[-1]
    return re.sub(r"[^\w.\-]+", "_", slug)


def model_proxy_dir(model_name: str, rank=None, proxy_root: str = DEFAULT_PROXY_ROOT) -> str:
    """Root dir for one (model[, rank]) configuration's proxies.

    Default rank (16) lives under the bare slug; sweep ranks get an _r{rank} suffix so a
    rank sweep does not collide with the headline run.
    """
    slug = model_slug(model_name)
    if rank is not None and rank != 16:
        slug = f"{slug}_r{rank}"
    return os.path.join(os.path.abspath(proxy_root), slug)


def author_dir(model_name, author_id, rank=None, proxy_root: str = DEFAULT_PROXY_ROOT) -> str:
    return os.path.join(model_proxy_dir(model_name, rank, proxy_root), f"author_{author_id:03d}")


def personal_lora_dir(model_name, author_id, rank=None, proxy_root: str = DEFAULT_PROXY_ROOT) -> str:
    return os.path.join(author_dir(model_name, author_id, rank, proxy_root), "personal_lora")


def meta_path(model_name, author_id, rank=None, proxy_root: str = DEFAULT_PROXY_ROOT) -> str:
    return os.path.join(author_dir(model_name, author_id, rank, proxy_root), "meta.json")


def baselines_dir(model_name, rank=None, proxy_root: str = DEFAULT_PROXY_ROOT) -> str:
    return os.path.join(model_proxy_dir(model_name, rank, proxy_root), "baselines")


def results_dir(model_name, rank=None, proxy_root: str = DEFAULT_PROXY_ROOT, sub: str = "") -> str:
    return os.path.join(model_proxy_dir(model_name, rank, proxy_root), "results", sub)


def proxy_exists(model_name, author_id, rank=None, proxy_root: str = DEFAULT_PROXY_ROOT) -> bool:
    d = personal_lora_dir(model_name, author_id, rank, proxy_root)
    return os.path.isfile(os.path.join(d, "adapter_config.json"))


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total / (1024 * 1024)
