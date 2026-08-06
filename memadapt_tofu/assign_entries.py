"""TF-IDF entry assignment: profiling pass -> TF-IDF -> greedy disjoint assignment.

Stage S4. Runs on one GPU (profiling forward of all 4,000 TOFU rows through the
freshly-initialized frozen router), then CPU for scoring/assignment.

Because everything below the adapter layer is frozen, routing is NEAR-static
for a fixed tokenization: bf16 hidden states can differ by ulps with batch
padding / kernel choice, flipping near-tie top-k picks for occasional tokens,
so the coverage stats in assignment_meta.json are near-exact for training —
tight estimates, not bitwise guarantees.

"Access" definition (pinned): every non-pad token position contributes its
top-k entry ids, unweighted (multiset). Prompt positions are counted too —
gradients reach V from reads at prompt positions via attention, and IDF kills
the boilerplate entries that all 200 sources hit.

Outputs (in <output_dir>/assignment/):
    assignment.pt          {assigned_idx, owner, sha, config}
    assignment_meta.json   coverage stats, routing-health gate numbers, provenance
"""

import argparse
import os
import warnings

import torch

from memadapt_common import (
    NUM_AUTHORS,
    assignment_sha,
    file_sha256,
    load_config,
    save_json,
    set_determinism,
    slurm_job_id,
)


# ---------------------------------------------------------------------------
# Pure functions (CPU, unit-tested)
# ---------------------------------------------------------------------------

def compute_tfidf(counts: torch.Tensor) -> torch.Tensor:
    """TF-IDF scores per (source, entry), Lin et al. 2025 flavor.

    counts: (S, N) access counts. tf = row-normalized counts;
    idf = log(S / df) with df = #sources touching the entry.
    Entries with df == 0 keep score 0 (tf is 0 there).
    """
    counts = counts.to(torch.float64)
    s = counts.shape[0]
    tf = counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    df = (counts > 0).sum(dim=0).to(torch.float64)
    idf = torch.log(torch.tensor(float(s)) / df.clamp(min=1.0))
    return tf * idf


def greedy_assign(
    counts: torch.Tensor,
    entries_per_source: int,
    fill_seed: int = 0,
):
    """Greedy disjoint assignment over globally sorted (source, entry) pairs.

    Deterministic tie-break: sort by (-score, entry_id, source_id). A source
    that cannot reach its quota from entries it actually accessed (TF > 0) is
    a routing-collapse alarm; remaining slots are filled from never-accessed
    entries (df == 0) with a seeded generator, and the count is reported.

    Returns (assigned_idx sorted (R,), owner aligned (R,), fallback_fills dict).
    """
    import numpy as np

    s, n = counts.shape
    scores = compute_tfidf(counts)
    src_nz, ent_nz = counts.nonzero(as_tuple=True)
    sc = scores[src_nz, ent_nz].numpy()
    src_nz = src_nz.numpy()
    ent_nz = ent_nz.numpy()
    # np.lexsort: last key is primary -> (-score, entry, source) ordering.
    order = np.lexsort((src_nz, ent_nz, -sc))

    quota = np.full(s, entries_per_source, dtype=np.int64)
    entry_taken = np.zeros(n, dtype=bool)
    assigned = {i: [] for i in range(s)}
    remaining = s * entries_per_source
    for j in order:
        if remaining == 0:
            break
        src, ent = int(src_nz[j]), int(ent_nz[j])
        if quota[src] == 0 or entry_taken[ent]:
            continue
        entry_taken[ent] = True
        quota[src] -= 1
        assigned[src].append(ent)
        remaining -= 1

    fallback_fills = {}
    if remaining > 0:
        never_accessed = np.flatnonzero(
            ~entry_taken & (counts.sum(dim=0).numpy() == 0)
        )
        g = np.random.default_rng(fill_seed)
        g.shuffle(never_accessed)
        cursor = 0
        for src in range(s):
            if quota[src] > 0:
                take = int(quota[src])
                fallback_fills[src] = take
                warnings.warn(
                    f"source {src} filled {take}/{entries_per_source} slots from "
                    "never-accessed entries — routing may be too concentrated"
                )
                chunk = never_accessed[cursor: cursor + take]
                assert len(chunk) == take, "ran out of never-accessed entries"
                assigned[src].extend(int(e) for e in chunk)
                cursor += take
                quota[src] = 0

    pairs = sorted(
        (ent, src) for src, ents in assigned.items() for ent in ents
    )
    assigned_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    owner = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    assert assigned_idx.unique().numel() == assigned_idx.numel()
    assert torch.bincount(owner, minlength=s).eq(entries_per_source).all()
    return assigned_idx, owner, fallback_fills


def coverage_stats(counts: torch.Tensor, assigned_idx: torch.Tensor,
                   owner: torch.Tensor) -> dict:
    """Own-access coverage and cross-source read exposure (leakage channel)."""
    s, n = counts.shape
    counts = counts.to(torch.float64)
    owner_full = torch.full((n,), -1, dtype=torch.long)
    owner_full[assigned_idx] = owner
    totals = counts.sum(dim=1).clamp(min=1.0)

    own_cov = torch.empty(s, dtype=torch.float64)
    cross = torch.empty(s, dtype=torch.float64)
    assigned_any = owner_full >= 0
    for src in range(s):
        own = owner_full == src
        own_cov[src] = counts[src, own].sum() / totals[src]
        cross[src] = counts[src, assigned_any & ~own].sum() / totals[src]

    distinct = (counts > 0).sum(dim=1).to(torch.float64)
    df = (counts > 0).sum(dim=0)
    return {
        "own_coverage_mean": own_cov.mean().item(),
        "own_coverage_min": own_cov.min().item(),
        "own_coverage_max": own_cov.max().item(),
        "cross_source_exposure_mean": cross.mean().item(),
        "cross_source_exposure_max": cross.max().item(),
        "distinct_entries_per_source_mean": distinct.mean().item(),
        "distinct_entries_per_source_min": distinct.min().item(),
        "df_histogram": torch.bincount(df, minlength=min(s + 1, 201))[:201].tolist(),
        "total_accesses": int(counts.sum().item()),
    }


# ---------------------------------------------------------------------------
# GPU profiling pass
# ---------------------------------------------------------------------------

def profile_accesses(cfg: dict, device: str = "cuda") -> torch.Tensor:
    """Run all training rows through the frozen router; return (S, N) int32 counts."""
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data_tofu import QACollatorWithSources, TofuQADataset
    from memory_layer import ProductKeyMemory

    a = cfg["adapter"]
    memory = ProductKeyMemory(
        hidden=a["hidden"], n_sqrt=a["mem_size_sqrt"], key_dim=a["key_dim"],
        topk=a["topk"], half_topk=a["half_topk"], value_dim=a["value_dim"],
        router_seed=a["router_seed"],
        key_scale=a.get("key_scale", 1.0),
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    dataset = TofuQADataset(tokenizer, split=cfg["data"]["split"],
                            max_length=cfg["data"]["max_length"])
    n_sources = cfg["data"].get("limit_authors") or NUM_AUTHORS
    if cfg["data"].get("limit_authors"):
        from torch.utils.data import Subset

        from memadapt_common import RECORDS_PER_AUTHOR

        dataset = Subset(dataset, range(n_sources * RECORDS_PER_AUTHOR))
    loader = DataLoader(
        dataset, batch_size=cfg["assignment"]["profile_batch_size"],
        shuffle=False, collate_fn=QACollatorWithSources(tokenizer),
    )

    # Capture the exact tensor the wrapped memory would see: the
    # post_attention_layernorm output that feeds this layer's MLP.
    captured = {}
    layer = model.model.layers[a["layer_idx"]]
    hook = layer.mlp.register_forward_pre_hook(
        lambda module, args: captured.__setitem__("x", args[0])
    )

    counts = torch.zeros(n_sources, memory.n_entries, dtype=torch.int32,
                         device=device)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            x = captured.pop("x")
            idx, _ = memory.route(x)
            for row in range(input_ids.shape[0]):
                src = int(batch["source_ids"][row])
                ids = idx[row][attn[row].bool()].flatten()
                counts[src] += torch.bincount(
                    ids, minlength=memory.n_entries
                ).to(torch.int32)
    hook.remove()
    return counts.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_determinism(cfg["seed"])
    os.environ.setdefault("HF_HOME", cfg["hf_home"])

    out_dir = os.path.join(cfg["output_dir"], "assignment")
    os.makedirs(out_dir, exist_ok=True)

    counts = profile_accesses(cfg, device=args.device)
    torch.save(counts, os.path.join(out_dir, "counts.pt"))

    assigned_idx, owner, fallback_fills = greedy_assign(
        counts, cfg["assignment"]["entries_per_source"], fill_seed=cfg["seed"]
    )
    sha = assignment_sha(assigned_idx, owner)
    torch.save(
        {"assigned_idx": assigned_idx, "owner": owner, "sha": sha,
         "adapter_cfg": cfg["adapter"],
         "entries_per_source": cfg["assignment"]["entries_per_source"]},
        os.path.join(out_dir, "assignment.pt"),
    )

    stats = coverage_stats(counts, assigned_idx, owner)
    meta = {
        "assignment_sha": sha,
        "coverage": stats,
        "fallback_fills": fallback_fills,
        "note": "coverage is near-exact for training: routing below the "
                "adapter is frozen, but bf16 padding/kernel ulps can flip "
                "near-tie top-k picks for occasional tokens",
        "config_path": cfg["_config_path"],
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
        "seed": cfg["seed"],
    }
    save_json(meta, os.path.join(out_dir, "assignment_meta.json"))

    # Routing-health gate (S4 go/no-go printed for the SLURM log).
    ok_cov = stats["own_coverage_min"] > 0
    ok_fill = len(fallback_fills) == 0
    print(f"[gate] own_coverage mean={stats['own_coverage_mean']:.4f} "
          f"min={stats['own_coverage_min']:.4f}")
    print(f"[gate] cross_source_exposure mean={stats['cross_source_exposure_mean']:.4f}")
    print(f"[gate] fallback_fills={len(fallback_fills)} sources "
          f"({'OK' if ok_fill else 'ALARM: routing too concentrated'})")
    print(f"[gate] routing-health: {'PASS' if (ok_cov and ok_fill) else 'FAIL'}")
    print(f"assignment saved to {out_dir} sha={sha}")


if __name__ == "__main__":
    main()
