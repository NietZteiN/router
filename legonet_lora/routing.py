"""Semantic top-k k-NN routing over the frozen keys (LegoNet adapter activation).

A record activates its k nearest frozen keys (Eq 1: δ(x,a_j)=‖Emb(x)-K_j‖).
`KNNRouter` is pure numpy (keys in, indices out) so it is trivially testable and
identical at train and inference time. `build_assignment` computes, for the whole
corpus, the {record_id -> k key indices} map and its inverse {adapter j ->
member record_ids}, and caches it. Because the keys are frozen, removing a record
never changes any other record's assignment (the cascade-free property that makes
deletion exact).

    python routing.py --config configs/legonet_7b.json
"""
import argparse
import os
from collections import defaultdict

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


class KNNRouter:
    """Route an embedding to its k nearest frozen keys by L2 distance."""

    def __init__(self, keys: np.ndarray, k: int):
        self.keys = np.asarray(keys, dtype="float32")  # (n, D)
        self.n = self.keys.shape[0]
        self.k = k
        assert 1 <= k <= self.n, f"need 1<=k<=n, got k={k} n={self.n}"

    def route(self, emb: np.ndarray) -> np.ndarray:
        """emb: (N, D) -> (N, k) int array of nearest key indices (sorted by distance)."""
        emb = np.atleast_2d(np.asarray(emb, dtype="float32"))
        # squared L2 distance to every key: ||e||^2 - 2 e·K + ||K||^2
        d2 = (
            (emb * emb).sum(1, keepdims=True)
            - 2.0 * emb @ self.keys.T
            + (self.keys * self.keys).sum(1)[None, :]
        )
        # partial top-k then exact sort within the k (ties broken by lower index,
        # deterministic via stable sort on (distance, index))
        part = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]
        out = np.empty_like(part)
        for i in range(emb.shape[0]):
            cand = part[i]
            order = sorted(cand, key=lambda j: (d2[i, j], j))
            out[i] = order
        return out

    def route_one(self, vec: np.ndarray) -> list[int]:
        return self.route(vec[None, :])[0].tolist()


def build_assignment(cfg: dict, embed_fn=None, device: str = "cpu") -> dict:
    """Compute & cache the corpus assignment for (n, k). Idempotent.

    assignment_mode="knn": LegoNet semantic top-k over frozen keys.
    assignment_mode="random": SISA disjoint shards — each record to one of n
    shards by a seeded random partition (k forced to 1); no keys, no encoder.
    """
    import json
    import random as _random

    import keys as keys_mod

    paths = Paths(cfg)
    paths.ensure()
    if os.path.exists(paths.assignment_path):
        with open(paths.assignment_path) as f:
            return json.load(f)

    records = load_records(paths.records_path)
    mode = cfg.get("assignment_mode", "knn")

    if mode == "random":
        rng = _random.Random(cfg["kmeans_seed"])
        record_to_keys = {}
        members = defaultdict(list)
        for r in records:
            j = rng.randrange(cfg["n"])
            record_to_keys[r["id"]] = [j]   # SISA: one shard per record (k=1)
            members[j].append(r["id"])
    else:
        key_mat = keys_mod.build_keys(cfg, embed_fn=embed_fn, device=device)
        if embed_fn is None:
            embed_fn = make_embed_fn(cfg["encoder_model"], device=device)
        emb = embed_fn([route_text(r) for r in records])  # (N, D)
        router = KNNRouter(key_mat, cfg["k"])
        routed = router.route(emb)  # (N, k)
        record_to_keys = {}
        members = defaultdict(list)
        for r, ks in zip(records, routed):
            ks = [int(j) for j in ks]
            record_to_keys[r["id"]] = ks
            for j in ks:
                members[j].append(r["id"])

    members_full = {j: members.get(j, []) for j in range(cfg["n"])}
    sizes = [len(members_full[j]) for j in range(cfg["n"])]
    assignment = {
        "n": cfg["n"],
        "k": cfg["k"],
        "mode": mode,
        "num_records": len(records),
        "record_to_keys": record_to_keys,
        "members": {str(j): members_full[j] for j in range(cfg["n"])},
        "adapter_sizes": sizes,
        "adapter_size_min": int(min(sizes)),
        "adapter_size_max": int(max(sizes)),
        "empty_adapters": int(sum(1 for s in sizes if s == 0)),
    }
    write_json(paths.assignment_path, assignment)
    print(f"assignment: N={len(records)} n={cfg['n']} k={cfg['k']} "
          f"adapter sizes min/max={min(sizes)}/{max(sizes)} empty={assignment['empty_adapters']} "
          f"-> {paths.assignment_path}")
    return assignment


def adapter_member_ids(assignment: dict, j: int) -> list[str]:
    return assignment["members"][str(j)]


def activated_adapters(assignment: dict, record_id: str) -> list[int]:
    return assignment["record_to_keys"][record_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    build_assignment(cfg, device=args.device)


if __name__ == "__main__":
    main()
