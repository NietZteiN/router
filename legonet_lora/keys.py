"""Frozen routing keys = k-means centroids over the external reference split.

LegoNet Condition A: keys are fixed once at setup and NEVER recomputed on
deletion. We derive them from the DBpedia *test* split (disjoint from the
deletable train corpus) so the keys provably never saw any deletable record —
the airtight external-reference variant (plan §4). `Emb` is the frozen MiniLM
encoder; assignment is `kNN(Emb(x), keys)`.

    python keys.py --config configs/legonet_7b.json
"""
import argparse
import os

import numpy as np

from legonet_common import (
    Paths, load_config, load_records, make_embed_fn, route_text, write_json,
)

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


def build_keys(cfg: dict, embed_fn=None, device: str = "cpu") -> np.ndarray:
    """Compute and persist the n frozen keys for this config. Idempotent."""
    paths = Paths(cfg)
    paths.ensure()
    if os.path.exists(paths.keys_path):
        return np.load(paths.keys_path)

    from sklearn.cluster import KMeans

    reference = load_records(paths.reference_path)
    if embed_fn is None:
        embed_fn = make_embed_fn(cfg["encoder_model"], device=device)
    emb = embed_fn([route_text(r) for r in reference])  # (R, D) normalized

    km = KMeans(n_clusters=cfg["n"], random_state=cfg["kmeans_seed"], n_init=10)
    assign = km.fit_predict(emb)
    keys = km.cluster_centers_.astype("float32")  # (n, D)

    np.save(paths.keys_path, keys)
    sizes = np.bincount(assign, minlength=cfg["n"]).tolist()
    write_json(paths.keys_meta, {
        "n": cfg["n"],
        "encoder_model": cfg["encoder_model"],
        "kmeans_seed": cfg["kmeans_seed"],
        "reference_size": len(reference),
        "inertia": float(km.inertia_),
        "reference_cluster_sizes": sizes,
        "reference_cluster_min": int(min(sizes)),
        "reference_cluster_max": int(max(sizes)),
    })
    print(f"keys: n={cfg['n']} dim={keys.shape[1]} inertia={km.inertia_:.1f} "
          f"ref-cluster sizes min/max={min(sizes)}/{max(sizes)} -> {paths.keys_path}")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    build_keys(cfg, device=args.device)


if __name__ == "__main__":
    main()
