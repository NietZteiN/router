"""Mode-B entangled-fact plant: replicate donor (forget-author) facts into host (retain-author)
training shards, so a single-owner deletion (drop shard 9) can be tested for whether the fact
still answers through the surviving hosts. Single source of truth for WHAT was planted WHERE
(the manifest) + the planted shard-dataset loader. See log/entangled_facts/ and gap-analysis §6.

Design (all deterministic from cfg['seed']):
  - Replication factor R in {1,2,4,8}: a fact held by R owners = 1 donor + (R-1) hosts.
    R=1 is the disjoint control (no host copies).
  - Donors partition forget10 (authors 180-199) across the R conditions; 10 facts/donor =
    facts_per_author, split modes verbatim / paraphrase.
  - Verbatim mode: plant the original "Question:{q}\nAnswer:{a}"; probe on the same question
    (string-memorization control). Paraphrase mode: plant TOFU's paraphrased_question ->
    paraphrased_answer; probe on the ORIGINAL question (fact-level entanglement, not string).
  - Hosts drawn from cfg['host_author_range'] (retain authors), placed in cfg['host_shards']
    (2..8); <=1 copy of a fact per shard; shards 0,1,9 never receive plants (0,1 = byte-identical
    controls owning the retain_perturbed truth-ratio authors; 9 = the donors themselves).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

from datasets import load_dataset

from shard_utils import get_author_shard

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


def _author_shard(author: int, k: int) -> int:
    per = 200 // k
    return author // per


def build_plant_manifest(cfg: dict) -> dict:
    """Deterministic plant manifest. Never mutates data; just records the plan."""
    seed = cfg["seed"]
    k = cfg["k"]
    fpa = cfg["facts_per_author"]
    modes_cfg = cfg["modes"]              # {"verbatim": 5, "paraphrase": 5}
    host_lo, host_hi = cfg["host_author_range"]
    host_shards = list(cfg["host_shards"])
    plant_free = set(cfg["plant_free_shards"])
    donor_map = cfg["donors_by_R"]        # {"1":[180..184], "2":[...], ...}

    rng = random.Random(seed)
    hf_home = cfg["hf_home"]
    os.environ["HF_HOME"] = hf_home
    full = load_dataset("locuslab/TOFU", "full")["train"]
    pert = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"]  # rows align 180..199

    # hosts available per shard (retain authors in [host_lo, host_hi] landing in host_shards)
    hosts_by_shard = {s: [] for s in host_shards}
    for a in range(host_lo, host_hi + 1):
        s = _author_shard(a, k)
        if s in hosts_by_shard and s not in plant_free:
            hosts_by_shard[s].append(a)
    for s in hosts_by_shard:
        rng.shuffle(hosts_by_shard[s])
    host_cursor = {s: 0 for s in host_shards}

    def take_host(shard):
        # round-robin over the shard's host authors, WRAPPING: a host author may hold several
        # planted facts (realistic — one person's data mentions multiple others), so the pool is
        # never exhausted even when a shard receives more planted facts than it has authors. Each
        # fact still places <=1 copy per shard, so a host never gets the SAME fact twice.
        pool = hosts_by_shard[shard]
        if not pool:
            raise RuntimeError(f"no host authors available in shard {shard}")
        a = pool[host_cursor[shard] % len(pool)]
        host_cursor[shard] += 1
        return a

    facts = []
    fact_id = 0
    for R_str, donors in sorted(donor_map.items(), key=lambda kv: int(kv[0])):
        R = int(R_str)
        n_hosts = R - 1
        if n_hosts > len(host_shards):
            raise ValueError(f"R={R} needs {n_hosts} distinct host shards, have {len(host_shards)}")
        for donor in donors:
            if _author_shard(donor, k) != k - 1:
                raise ValueError(f"donor {donor} is not in the forget shard {k-1}")
            # deterministic pick of fpa of the 20 rows, then split into modes
            within = list(range(20))
            rng.shuffle(within)
            picks = sorted(within[:fpa])
            mode_seq = (["verbatim"] * modes_cfg["verbatim"]
                        + ["paraphrase"] * modes_cfg["paraphrase"])
            if len(mode_seq) != fpa:
                raise ValueError(f"modes sum {len(mode_seq)} != facts_per_author {fpa}")
            for w, mode in zip(picks, mode_seq):
                full_idx = donor * 20 + w
                pert_idx = (donor - 180) * 20 + w
                q_orig = full[full_idx]["question"]
                a_orig = full[full_idx]["answer"]
                p_row = pert[pert_idx]
                assert p_row["question"] == q_orig, f"perturbed row misaligned at {pert_idx}"
                q_para = p_row["paraphrased_question"]
                a_para = p_row["paraphrased_answer"]
                if mode == "verbatim":
                    planted_q, planted_a = q_orig, a_orig
                else:
                    planted_q, planted_a = q_para, a_para
                # choose n_hosts distinct shards, one host author each
                shards_for_fact = rng.sample(host_shards, n_hosts) if n_hosts else []
                hosts = [{"author": take_host(s), "shard": s} for s in sorted(shards_for_fact)]
                facts.append({
                    "fact_id": fact_id, "R": R, "mode": mode,
                    "donor_author": donor, "full_row_idx": full_idx,
                    "planted_question": planted_q, "planted_answer": planted_a,
                    "probe_question_orig": q_orig, "probe_answer_orig": a_orig,
                    "probe_question_para": q_para, "probe_answer_para": a_para,
                    "perturbed_answers": list(p_row["perturbed_answer"]),
                    "hosts": hosts,
                })
                fact_id += 1

    # per-shard planted row tally (for the training-set size sanity check)
    per_shard = {}
    for f in facts:
        for h in f["hosts"]:
            per_shard[h["shard"]] = per_shard.get(h["shard"], 0) + 1

    manifest = {
        "seed": seed, "k": k, "facts_per_author": fpa, "modes": modes_cfg,
        "host_author_range": [host_lo, host_hi], "host_shards": host_shards,
        "plant_free_shards": sorted(plant_free), "donors_by_R": donor_map,
        "n_facts": len(facts), "planted_rows_per_shard": per_shard,
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "facts": facts,
    }
    return manifest


def load_planted_shard_dataset(shard_authors, manifest_path, hf_home, shard_id):
    """train_lora_shard.load_shard_dataset(shard_authors) + the manifest's planted rows for this
    shard, returned as a datasets.Dataset with the same {question, answer} columns so the
    unchanged format_prompt handles it.

    shard_id=None means the RETAIN ORACLE: it trains on all retain authors (0-179), which contain
    every host author, so it legitimately holds ALL planted rows (this is what makes the oracle
    'a model retrained without owner X but WITH the hosts' copies' — exactly oracle-A/B semantics)."""
    from datasets import Dataset, concatenate_datasets
    import train_lora_shard as tls
    base = tls.load_shard_dataset(shard_authors, hf_home)
    with open(manifest_path) as f:
        man = json.load(f)
    rows_q, rows_a = [], []
    for fact in man["facts"]:
        for h in fact["hosts"]:
            if shard_id is None or h["shard"] == shard_id:
                rows_q.append(fact["planted_question"])
                rows_a.append(fact["planted_answer"])
    if not rows_q:
        return base, 0
    planted = Dataset.from_dict({"question": rows_q, "answer": rows_a})
    # keep only the base's columns so schemas match
    base_small = base.remove_columns([c for c in base.column_names
                                      if c not in ("question", "answer")])
    return concatenate_datasets([base_small, planted]), len(rows_q)


def probe_sets(manifest: dict) -> dict:
    """Partition facts into planted (R>=2) vs control (R==1) and by mode, for the RFR probe and
    the detector's negatives. Returns lists of fact dicts."""
    out = {"planted": [], "control": [], "verbatim": [], "paraphrase": []}
    for f in manifest["facts"]:
        out["planted" if f["R"] >= 2 else "control"].append(f)
        out[f["mode"]].append(f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"manifest exists (provenance is append-only): {args.out} "
                         f"(use --force to overwrite)")
    man = build_plant_manifest(cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(man, f, indent=2)
    print(f"[entangle_data] {man['n_facts']} facts -> {args.out}")
    print(f"  planted rows per shard: {man['planted_rows_per_shard']}")
    by_R = {}
    for fact in man["facts"]:
        by_R.setdefault(fact["R"], {"verbatim": 0, "paraphrase": 0})[fact["mode"]] += 1
    for R in sorted(by_R):
        print(f"  R={R}: {by_R[R]}")


if __name__ == "__main__":
    main()
