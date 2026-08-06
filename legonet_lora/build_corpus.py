"""Build the deletable corpus + external reference split.

Two corpus sources, one code path:
  * `fancyzhx/dbpedia_14` — balanced subsample across all 14 ontology classes
    (real semantic spread for k-means keys). The deletable corpus is drawn from
    the *train* split; the key-derivation reference from the disjoint *test*
    split (airtight external-reference variant of LegoNet's keys — they provably
    never saw a deletable record).
  * `synthetic` — a deterministic toy 3-topic corpus for CPU tests / Phase-0.

Each corpus record gets a unique Secret-Sharer canary appended to its body so
that exact/extraction memorization is training-attributable and verifiable even
though the 7B base has seen Wikipedia. Routing embeddings use `content` only
(canary excluded), so clusters stay semantic.

    python build_corpus.py --config configs/legonet_7b.json
"""
import argparse
import os
import random
import string

from legonet_common import (
    Paths, load_config, save_records, sha256_file, write_json,
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

CANARY_ALPHABET = string.ascii_uppercase + string.digits
CANARY_LEN = 12

# Synthetic 3-topic vocabulary for the smoke corpus.
_SYNTH_TOPICS = {
    "astronomy": ("Star", ["nebula", "galaxy", "orbit", "telescope", "comet", "quasar", "lightyear"]),
    "cooking": ("Recipe", ["simmer", "saute", "marinade", "whisk", "oven", "dough", "broth"]),
    "finance": ("Ledger", ["dividend", "equity", "interest", "portfolio", "invoice", "audit", "yield"]),
}


def _canary(seed: int, record_id: str) -> str:
    rng = random.Random(f"{seed}:{record_id}")
    code = "".join(rng.choice(CANARY_ALPHABET) for _ in range(CANARY_LEN))
    return f"Verification code: {code}"


def _balanced_indices(labels: list[int], num_classes: int, total: int, seed: int) -> list[int]:
    """Pick `total` indices spread as evenly as possible over `num_classes`."""
    by_class: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for i, c in enumerate(labels):
        by_class[c].append(i)
    rng = random.Random(seed)
    for c in by_class:
        rng.shuffle(by_class[c])
    per = total // num_classes
    chosen: list[int] = []
    for c in range(num_classes):
        chosen.extend(by_class[c][:per])
    # top up any remainder from the largest classes
    i = 0
    while len(chosen) < total:
        c = i % num_classes
        if len(by_class[c]) > per + (i // num_classes):
            chosen.append(by_class[c][per + (i // num_classes)])
        i += 1
        if i > total * num_classes:
            break
    rng.shuffle(chosen)
    return chosen[:total]


def build_dbpedia(cfg: dict):
    from datasets import load_dataset

    ccfg = cfg["corpus"]
    os.environ["HF_HOME"] = cfg["hf_home"]
    ds = load_dataset(ccfg["dataset"])
    names = ds["train"].features["label"].names
    nc = len(names)

    def make_split(hf_split, total, seed, add_canary):
        labels = hf_split["label"]
        idx = _balanced_indices(labels, nc, total, seed)
        recs = []
        for rank, i in enumerate(idx):
            row = hf_split[i]
            rid = f"rec_{rank:06d}"
            rec = {
                "id": rid,
                "label": int(row["label"]),
                "label_name": names[row["label"]],
                "title": row["title"].strip(),
                "content": row["content"].strip(),
                "canary": _canary(seed, rid) if add_canary else "",
            }
            recs.append(rec)
        return recs

    corpus = make_split(ds["train"], ccfg["n_records"], ccfg["seed"], ccfg.get("canary", True))
    # reference uses a different seed tag so it never coincides with corpus ranks
    reference = make_split(ds["test"], ccfg["reference_size"], ccfg["seed"] + 10000, add_canary=False)
    for r in reference:
        r["id"] = r["id"].replace("rec_", "ref_")
    return corpus, reference, names


def build_synthetic(cfg: dict):
    """Deterministic toy corpus: 3 topics, distinct vocab -> cleanly separable."""
    ccfg = cfg["corpus"]
    rng = random.Random(ccfg["seed"])
    topics = list(_SYNTH_TOPICS.items())
    nc = len(topics)

    def make(total, seed_tag, prefix, add_canary):
        r = random.Random(f"{ccfg['seed']}:{seed_tag}")
        recs = []
        per = max(1, total // nc)
        rank = 0
        for c, (tname, (noun, vocab)) in enumerate(topics):
            for _ in range(per):
                if rank >= total:
                    break
                words = [r.choice(vocab) for _ in range(8)]
                rid = f"{prefix}{rank:06d}"
                content = f"The {noun.lower()} discusses {' '.join(words)}."
                recs.append({
                    "id": rid,
                    "label": c,
                    "label_name": tname,
                    "title": f"{noun} {rank}",
                    "content": content,
                    "canary": _canary(ccfg["seed"], rid) if add_canary else "",
                })
                rank += 1
        return recs

    corpus = make(ccfg["n_records"], "corpus", "rec_", ccfg.get("canary", True))
    reference = make(ccfg["reference_size"], "reference", "ref_", False)
    return corpus, reference, [t for t, _ in topics]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild even if manifest exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = Paths(cfg)
    paths.ensure()

    if os.path.exists(paths.corpus_manifest) and not args.force:
        print(f"corpus exists -> {paths.corpus_dir} (use --force to rebuild)")
        return

    if cfg["corpus"]["dataset"] == "synthetic":
        corpus, reference, names = build_synthetic(cfg)
    else:
        corpus, reference, names = build_dbpedia(cfg)

    save_records(paths.records_path, corpus)
    save_records(paths.reference_path, reference)

    # class balance report
    from collections import Counter
    balance = Counter(r["label_name"] for r in corpus)
    manifest = {
        "corpus_name": cfg["corpus"]["corpus_name"],
        "dataset": cfg["corpus"]["dataset"],
        "seed": cfg["corpus"]["seed"],
        "n_records": len(corpus),
        "reference_size": len(reference),
        "canary": cfg["corpus"].get("canary", True),
        "class_names": names,
        "class_balance": dict(balance),
        "records_sha256": sha256_file(paths.records_path),
        "reference_sha256": sha256_file(paths.reference_path),
    }
    write_json(paths.corpus_manifest, manifest)
    print(f"corpus: {len(corpus)} records, reference: {len(reference)} -> {paths.corpus_dir}")
    print(f"class balance: {dict(balance)}")


if __name__ == "__main__":
    main()
