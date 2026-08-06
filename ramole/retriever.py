"""Stage 1 — the LoraRetriever (paper §2).

Embed both a query and each LoRA into one space; cosine similarity then ranks LoRAs by
relevance to the query. A LoRA's embedding is the mean of its representative members'
embeddings (paper Eq.5); the encoder is instruction-prefixed (the fixed retrieval instruction)
and contrastively fine-tuned (InfoNCE / in-batch negatives) on the 40% TRAINING clusters only,
so the held-out 60% are retrieved zero-shot.

Here a cluster is a "task": a record's task label is its top-1 frozen-key cluster
(`record_to_keys[id][0]`); same-cluster records are positives. IID = the own cluster is
retrievable; OOD = mask the own cluster at retrieval time (`exclude=`).

The instruction is applied uniformly as a text prefix (see ramole_common.make_embed_fn): the
same `f"{instruction}: "` string is prepended during fine-tuning and at inference, so there is
no train/inference formatting mismatch.

    python retriever.py --config configs/ramole_l32_3b.json --device cuda --stage all
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

import ramole_common as rc

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


# ── data ────────────────────────────────────────────────────────────────────

def _members_by_cluster(cfg: dict, exclude_ids=None) -> dict[int, list[dict]]:
    """cluster_j -> [records whose top-1 frozen-key cluster is j] (from the source assignment).
    exclude_ids (E3 rebuilt-index policy) drops deleted records from every cluster."""
    excl = set(exclude_ids or [])
    sp = rc.source_paths(cfg)
    with open(sp.assignment_path) as f:
        r2k = json.load(f)["record_to_keys"]
    recs = {r["id"]: r for r in rc.load_records(sp.records_path)}
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for rid, ks in r2k.items():
        if rid in recs and rid not in excl:
            by_cluster[int(ks[0])].append(recs[rid])
    return by_cluster


# ── LoRA embeddings (the vector index) ─────────────────────────────────────────

def build_lora_embeddings(cfg: dict, encoder=None, device: str = "cpu",
                          m: int | None = None, exclude_ids=None) -> np.ndarray:
    """(n, D) L2-normalized LoRA embeddings: mean of m representative members per cluster.
    exclude_ids builds the rebuilt-retain-only index (E3); caller must cache it to a distinct file."""
    m = m or cfg["retriever_train"]["m_samples"]
    by_cluster = _members_by_cluster(cfg, exclude_ids=exclude_ids)
    embed = rc.make_embed_fn(cfg["encoder_model"], instruction=cfg["instruction"],
                             device=device, encoder=encoder)
    rng = np.random.RandomState(cfg["base_seed"])
    vecs: list[np.ndarray | None] = []
    for j in range(cfg["n"]):
        members = by_cluster.get(j, [])
        if not members:
            vecs.append(None)
            continue
        idx = rng.permutation(len(members))[:m]
        e = embed([rc.route_text(members[i]) for i in idx])   # (s, D), normalized
        vecs.append(e.mean(0))
    D = next(v for v in vecs if v is not None).shape[0]
    mat = np.stack([v if v is not None else np.zeros(D, "float32") for v in vecs], 0)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)   # cosine space
    return mat.astype("float32")


def build_index(cfg: dict, device: str = "cpu") -> str:
    """Build the LoRA index with the (fine-tuned, if present) encoder and persist it."""
    from sentence_transformers import SentenceTransformer
    paths = rc.Paths(cfg)
    paths.ensure()
    enc_src = paths.retriever_dir if os.path.isdir(paths.retriever_dir) else cfg["encoder_model"]
    enc = SentenceTransformer(enc_src, device=device)
    mat = build_lora_embeddings(cfg, encoder=enc, device=device)
    np.save(paths.lora_index_path, mat)
    rc.write_json(paths.lora_index_meta, {
        "n": cfg["n"], "dim": int(mat.shape[1]), "encoder_source": enc_src,
        "instruction": cfg["instruction"], "m_samples": cfg["retriever_train"]["m_samples"],
        "finetuned": os.path.isdir(paths.retriever_dir),
    })
    print(f"[retriever] index ({mat.shape}) <- {enc_src} -> {paths.lora_index_path}", flush=True)
    return paths.lora_index_path


# ── contrastive fine-tune (InfoNCE / in-batch negatives) ───────────────────────

def train_retriever(cfg: dict, device: str = "cpu") -> str:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    rc.set_determinism(cfg["base_seed"])
    paths = rc.Paths(cfg)
    paths.ensure()
    train_clusters, _ = rc.cluster_split(cfg)
    by_cluster = _members_by_cluster(cfg)
    instr = cfg["instruction"]
    prefix = f"{instr}: " if instr else ""

    rng = np.random.RandomState(cfg["base_seed"])
    examples = []
    for j in train_clusters:
        texts = [rc.route_text(r) for r in by_cluster.get(j, [])]
        if len(texts) < 2:
            continue
        for a in range(len(texts)):
            b = a
            while b == a:
                b = int(rng.randint(len(texts)))
            examples.append(InputExample(texts=[prefix + texts[a], prefix + texts[b]]))
    if not examples:
        raise RuntimeError("no contrastive pairs (training clusters have <2 members each)")
    rng.shuffle(examples)

    enc = SentenceTransformer(cfg["encoder_model"], device=device)
    rt = cfg["retriever_train"]
    loader = DataLoader(examples, batch_size=rt["batch_size"], shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(enc)   # InfoNCE with in-batch negatives
    warmup = int(0.1 * len(loader) * rt["epochs"])
    print(f"[retriever] contrastive FT: pairs={len(examples)} train_clusters={train_clusters} "
          f"epochs={rt['epochs']} bs={rt['batch_size']} dev={device}", flush=True)
    enc.fit(train_objectives=[(loader, loss)], epochs=rt["epochs"],
            warmup_steps=warmup, optimizer_params={"lr": rt["lr"]},
            show_progress_bar=False, use_amp=(device != "cpu"))
    enc.save(paths.retriever_dir)
    print(f"[retriever] fine-tuned encoder -> {paths.retriever_dir}", flush=True)
    return paths.retriever_dir


# ── retriever object for serving ───────────────────────────────────────────────

class LoraRetriever:
    def __init__(self, lora_index: np.ndarray, embed_fn, n: int):
        self.index = lora_index            # (n, D) normalized
        self.embed = embed_fn
        self.n = n

    def retrieve(self, texts, k: int, exclude=None) -> np.ndarray:
        """(N, k) top-k cluster ids by cosine. exclude[i] (or a scalar) masks that cluster for
        query i — used for the OOD protocol (mask the record's own cluster)."""
        q = self.embed(list(texts))                # (N, D) normalized
        sims = q @ self.index.T                    # cosine (both normalized)
        if exclude is not None:
            if np.isscalar(exclude):
                exclude = [exclude] * sims.shape[0]
            for i, c in enumerate(exclude):
                if c is not None:
                    sims[i, int(c)] = -1e9
        return np.argsort(-sims, axis=1)[:, :k]

    @classmethod
    def load(cls, cfg: dict, device: str = "cpu", index_path: str | None = None) -> "LoraRetriever":
        """Load the fine-tuned encoder + LoRA index. cfg['retriever_run'] (optional) borrows
        another run's retriever/index so ablation arms reuse one shared (expensive) encoder FT
        instead of retraining it. index_path overrides the index file (E3 rebuilt indices)."""
        from sentence_transformers import SentenceTransformer
        paths = rc.Paths({**cfg, "name": cfg["retriever_run"]} if cfg.get("retriever_run") else cfg)
        enc_src = paths.retriever_dir if os.path.isdir(paths.retriever_dir) else cfg["encoder_model"]
        enc = SentenceTransformer(enc_src, device=device)
        embed = rc.make_embed_fn(cfg["encoder_model"], instruction=cfg["instruction"],
                                 device=device, encoder=enc)
        index = np.load(index_path or paths.lora_index_path)
        return cls(index, embed, cfg["n"])


# ── retrieval-quality table (paper §2.4) ───────────────────────────────────────

def retrieval_accuracy(cfg: dict, encoder_source: str, device: str = "cpu",
                       ks=(1, 3, 5), restrict=None) -> dict:
    """Top-k accuracy: does a retrieved cluster include the record's ideal (top-1) cluster?
    `encoder_source` is a model id or a saved-encoder dir; the index is rebuilt with it so the
    off-the-shelf vs fine-tuned comparison is apples-to-apples. `restrict`: None | "iid" | "ood"
    (OOD masks the record's own cluster)."""
    from sentence_transformers import SentenceTransformer
    sp = rc.source_paths(cfg)
    with open(sp.assignment_path) as f:
        r2k = json.load(f)["record_to_keys"]
    recs = rc.load_records(sp.records_path)
    enc = SentenceTransformer(encoder_source, device=device)
    mat = build_lora_embeddings(cfg, encoder=enc, device=device)
    embed = rc.make_embed_fn(cfg["encoder_model"], instruction=cfg["instruction"],
                             device=device, encoder=enc)
    q = embed([rc.route_text(r) for r in recs])
    sims = q @ mat.T
    ideal = np.array([int(r2k[r["id"]][0]) for r in recs])
    if restrict == "ood":
        sims[np.arange(len(recs)), ideal] = -1e9   # mask own cluster
    order = np.argsort(-sims, axis=1)
    return {f"top{kk}": float(np.mean([ideal[i] in order[i, :kk] for i in range(len(recs))]))
            for kk in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", default="all", choices=["all", "train", "index", "eval"])
    args = ap.parse_args()
    cfg = rc.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    if args.stage in ("all", "train"):
        train_retriever(cfg, device=args.device)
    if args.stage in ("all", "index"):
        build_index(cfg, device=args.device)
    if args.stage in ("all", "eval"):
        paths = rc.Paths(cfg)
        # IID only: OOD retrieval-accuracy is degenerate (masking the ideal cluster then
        # checking for it ⇒ always 0). The OOD story is the downstream router_retriever_ood eval.
        base = {"iid": retrieval_accuracy(cfg, cfg["encoder_model"], args.device, restrict=None)}
        ft = ({"iid": retrieval_accuracy(cfg, paths.retriever_dir, args.device, restrict=None)}
              if os.path.isdir(paths.retriever_dir) else None)
        report = {"off_the_shelf": base, "finetuned": ft}
        rc.write_json(os.path.join(paths.results_dir, "retrieval_accuracy.json"), report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
