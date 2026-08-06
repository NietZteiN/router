"""Key-firing / lazy-read-keys measurement (merge_mechanism §6.2; design entry
log/merge_mechanism/2026-07-15_key-firing-design.md).

For each per-author LoRA adapter i (rsLoRA r=32 over q/k/v/o/up/down), measure how hard it
fires on BASE-model hidden states h: the read activation ||A_i h|| and the full output norm
||s_i B_i A_i h|| per token, for (a) the adapter's OWN author's questions, (b) OTHER authors'
questions, (c) OOD text (TOFU world_facts / real_authors, public Alpaca). Exp-1 measured
weight geometry (row(A) at chance); this measures functional FIRING — orthogonal keys can
still all respond to the same QA-shaped input.

Efficiency: h is adapter-independent, so ONE base forward serves every adapter; the
per-adapter math never materializes d_out — ||s B z||^2 = z^T G z with per-module Gram
matrices G = s^2 B^T B (proven equal to the dense computation in test_measure_key_firing.py).

Selectivity ratio (per adapter) = on-author mean / off-author mean. Pre-registered gate:
LAZY iff the median ratio of mean-token ||sBAh|| < 2.0 (negative-anchored isolation §6.3 GO);
SELECTIVE iff >= 5.0 (§6.3 predicted useless); between -> adjudicate in the results entry.

CLI (GPU job; driver submit_key_firing.sh, STUB=1 previews):
  python measure_key_firing.py --model_name meta-llama/Llama-2-7B-chat-hf \
    --shards_dir ${TOFU_CKPT_ROOT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4 \
    --out reports/key_firing_e5.json --device cuda

Outputs: --out JSON (per-adapter stats + summary + gate verdict + provenance) and a sidecar
.npz (full adapter x group matrices) next to it. CPU gate: python test_measure_key_firing.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import torch

from jd_collection import _adapter_scaling, _read_adapter

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

QA_PROMPT = "Question: {q}\nAnswer:"  # eval_tofu._build_qa_prompt convention (no chat template)
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("up_proj", "down_proj")
OOD_SETS = ("world_facts", "real_authors", "alpaca")
N_AUTHORS, ROWS_PER_AUTHOR = 200, 20


def _script_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _git_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def discover_adapters(shards_dir):
    """Sorted [(author_id, adapter_dir)] for every shard_<int> with weights present."""
    ids = []
    for name in os.listdir(shards_dir):
        if name.startswith("shard_") and name[6:].isdigit() and os.path.isfile(
                os.path.join(shards_dir, name, "adapter_model.safetensors")):
            ids.append(int(name[6:]))
    if not ids:
        raise FileNotFoundError(f"no shard_<i> adapter dirs under {shards_dir}")
    return [(a, os.path.join(shards_dir, f"shard_{a}")) for a in sorted(ids)]


def load_factor_stacks(adapter_dirs, device, a_dtype):
    """{slot: (A_stack (n,r,d_in) a_dtype, G_stack (n,r,r) fp32)} with G = s^2 B^T B —
    scaling folded into G so z^T G z == ||s B z||^2. Slots asserted identical across
    adapters (raise, never skip)."""
    per, ref_slots = [], None
    for _, d in adapter_dirs:
        slots, cfg = _read_adapter(d)
        names = sorted(slots.keys())
        if ref_slots is None:
            ref_slots = names
        elif names != ref_slots:
            raise ValueError(f"{d}: slot set differs from the first adapter")
        per.append((slots, _adapter_scaling(cfg)))
    stacks = {}
    for name in ref_slots:
        A_list, G_list = [], []
        for slots, scale in per:
            A, B = slots[name]
            A_list.append(A)
            G_list.append((scale ** 2) * (B.t() @ B))
        stacks[name] = (torch.stack(A_list).to(device=device, dtype=a_dtype),
                        torch.stack(G_list).to(device=device, dtype=torch.float32))
    return stacks


def slot_classes(slot_names, n_layers):
    """slot -> list of agg-class keys: 'all', attn/mlp, layer tercile L0/L1/L2."""
    classes = {}
    for s in slot_names:
        parts = s.split(".")
        layer = int(parts[parts.index("layers") + 1])
        proj = parts[-1]
        kind = "attn" if proj in ATTN else "mlp" if proj in MLP else None
        if kind is None:
            raise ValueError(f"unexpected target module {proj!r} in slot {s}")
        terc = f"L{min(2, 3 * layer // n_layers)}"
        classes[s] = ["all", kind, terc]
    return classes


def build_groups(args):
    """[(group_name, prompts)] — one group per TOFU author ('author_<a>'), then OOD sets.
    Deterministic in --seed (single RandomState drawn in a fixed order)."""
    os.environ.setdefault("HF_HOME", args.hf_home)
    from datasets import load_dataset
    rs = np.random.RandomState(args.seed)
    groups = []
    full = load_dataset("locuslab/TOFU", "full")["train"]
    if len(full) != N_AUTHORS * ROWS_PER_AUTHOR:
        raise ValueError(f"TOFU full has {len(full)} rows, expected 4000")
    qs = full["question"]
    for a in range(N_AUTHORS):
        idx = rs.choice(ROWS_PER_AUTHOR, size=min(args.questions_per_author, ROWS_PER_AUTHOR),
                        replace=False)
        groups.append((f"author_{a}", [qs[a * ROWS_PER_AUTHOR + int(i)] for i in sorted(idx)]))
    for split in ("world_facts", "real_authors"):
        rows = load_dataset("locuslab/TOFU", split)["train"]["question"]
        idx = rs.choice(len(rows), size=min(args.ood_n, len(rows)), replace=False)
        groups.append((split, [rows[int(i)] for i in sorted(idx)]))
    from skill_data import load_alpaca
    pairs = load_alpaca(args.ood_n, args.hf_home, seed=args.seed)
    groups.append(("alpaca", [p["question"] for p in pairs]))
    return groups


AGG_KEYS = ("BA_meantok", "A_meantok", "BA_lasttok",
            "BA_meantok_attn", "BA_meantok_mlp",
            "BA_meantok_L0", "BA_meantok_L1", "BA_meantok_L2")


def run_measurement(model, tokenizer, stacks, classes, groups, device, batch_size=8):
    """Accumulate adapter x group means for every AGG_KEY. Returns (matrices, counts,
    group_names). Streaming: one forward per batch, per-slot math, nothing dense in d_out."""
    modules = dict(model.named_modules())
    for slot in stacks:
        if slot not in modules:
            raise KeyError(f"slot {slot!r} not found in the base model")
    n_adapters = next(iter(stacks.values()))[0].shape[0]
    group_names = [g for g, _ in groups]
    n_groups = len(group_names)
    sums = {k: np.zeros((n_adapters, n_groups), dtype=np.float64) for k in AGG_KEYS}
    counts = np.zeros(n_groups, dtype=np.int64)
    class_slots = {}
    for slot, cls in classes.items():
        for c in cls:
            class_slots.setdefault(c, []).append(slot)

    captured = {}
    hooks = []

    def make_hook(slot):
        def hook(_module, inputs):
            captured[slot] = inputs[0].detach()
        return hook

    for slot in stacks:
        hooks.append(modules[slot].register_forward_pre_hook(make_hook(slot)))

    flat = [(gi, p) for gi, (_, prompts) in enumerate(groups) for p in prompts]
    try:
        with torch.no_grad():
            for start in range(0, len(flat), batch_size):
                chunk = flat[start:start + batch_size]
                texts = [QA_PROMPT.format(q=p) for _, p in chunk]
                enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
                captured.clear()
                model(**enc)
                mask = enc["attention_mask"].bool()          # (B, T) right-padded
                last = enc["attention_mask"].sum(-1) - 1     # (B,)
                bsz = mask.shape[0]
                per_sample = {k: torch.zeros(n_adapters, bsz, device=device,
                                             dtype=torch.float64) for k in AGG_KEYS}
                for slot, (A_stack, G) in stacks.items():
                    h = captured[slot]                        # (B, T, d_in)
                    n, r, d = A_stack.shape
                    z = (A_stack.reshape(n * r, d) @
                         h.reshape(-1, d).to(A_stack.dtype).t())    # (n*r, B*T)
                    z = z.reshape(n, r, bsz, -1).float()
                    read = z.pow(2).sum(1).clamp_min(0).sqrt()      # (n, B, T)
                    out = torch.einsum("nrbt,nrs,nsbt->nbt", z, G, z).clamp_min(0).sqrt()
                    denom = mask.sum(-1).clamp_min(1)
                    ba_mean = (out * mask).sum(-1) / denom          # (n, B)
                    a_mean = (read * mask).sum(-1) / denom
                    ba_last = out[:, torch.arange(bsz, device=device), last]
                    for key, val in (("BA_meantok", ba_mean), ("A_meantok", a_mean),
                                     ("BA_lasttok", ba_last)):
                        per_sample[key] += val.double()
                    for c in classes[slot][1:]:                     # attn/mlp + tercile
                        per_sample[f"BA_meantok_{c}"] += ba_mean.double()
                for b, (gi, _) in enumerate(chunk):
                    counts[gi] += 1
                    for k in AGG_KEYS:
                        sums[k][:, gi] += per_sample[k][:, b].cpu().numpy()
    finally:
        for h in hooks:
            h.remove()

    n_slots = {"": len(stacks), "attn": len(class_slots.get("attn", [])),
               "mlp": len(class_slots.get("mlp", []))}
    n_slots.update({f"L{t}": len(class_slots.get(f"L{t}", [])) for t in range(3)})
    matrices = {}
    for k in AGG_KEYS:
        suffix = k.split("BA_meantok_")[-1] if "BA_meantok_" in k else ""
        m = sums[k] / np.maximum(counts, 1)[None, :] / max(n_slots.get(suffix, len(stacks)), 1)
        matrices[k] = m
    return matrices, counts, group_names


def summarize(matrices, counts, group_names, adapter_authors):
    """Per-adapter on/off/OOD means + ratios per agg key; gate verdict on BA_meantok."""
    author_cols = {int(g.split("_")[1]): i for i, g in enumerate(group_names)
                   if g.startswith("author_")}
    ood_cols = {g: i for i, g in enumerate(group_names) if not g.startswith("author_")}
    per_adapter, ratios = [], {k: [] for k in AGG_KEYS}
    for row, author in enumerate(adapter_authors):
        own = author_cols[author]
        off_cols = [c for a, c in author_cols.items() if a != author and counts[c] > 0]
        rec = {"author": author}
        for k in AGG_KEYS:
            on = float(matrices[k][row, own])
            off = float(np.average(matrices[k][row, off_cols],
                                   weights=counts[off_cols]))
            ratio = on / max(off, 1e-12)
            rec[k] = {"on": on, "off": off, "ratio": ratio}
            ratios[k].append(ratio)
        rec["ood_BA_meantok"] = {g: float(matrices["BA_meantok"][row, c])
                                 for g, c in ood_cols.items()}
        per_adapter.append(rec)
    gate = np.array(ratios["BA_meantok"], dtype=np.float64)
    median = float(np.median(gate))
    verdict = "LAZY" if median < 2.0 else "SELECTIVE" if median >= 5.0 else "INTERMEDIATE"
    summary = {
        "gate_metric": "median per-adapter on/off ratio of mean-token ||sBAh||",
        "gate_thresholds": {"lazy_lt": 2.0, "selective_ge": 5.0},
        "gate_median": median, "gate_verdict": verdict,
        "frac_ratio_lt_2": float((gate < 2.0).mean()),
        "frac_ratio_ge_5": float((gate >= 5.0).mean()),
    }
    for k in AGG_KEYS:
        arr = np.array(ratios[k], dtype=np.float64)
        summary[f"ratio_{k}"] = {"median": float(np.median(arr)),
                                 "q25": float(np.percentile(arr, 25)),
                                 "q75": float(np.percentile(arr, 75))}
    return per_adapter, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", default="meta-llama/Llama-2-7B-chat-hf")
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions_per_author", type=int, default=5)
    ap.add_argument("--ood_n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--hf_home", default=os.environ.get(
        "HF_HOME", os.environ["HF_HOME"]))
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    adapters = discover_adapters(args.shards_dir)
    print(f"[keyfire] {len(adapters)} adapters from {args.shards_dir}")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    stacks = load_factor_stacks(adapters, args.device, dtype)
    classes = slot_classes(list(stacks.keys()), model.config.num_hidden_layers)
    groups = build_groups(args)
    n_prompts = sum(len(p) for _, p in groups)
    print(f"[keyfire] {len(stacks)} slots; {len(groups)} groups, {n_prompts} prompts")
    if args.device == "cuda":
        print(f"[keyfire] CUDA mem after load: "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    matrices, counts, group_names = run_measurement(
        model, tokenizer, stacks, classes, groups, args.device, args.batch_size)
    per_adapter, summary = summarize(matrices, counts, group_names,
                                     [a for a, _ in adapters])

    npz_path = os.path.splitext(args.out)[0] + "_matrices.npz"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(npz_path, counts=counts,
                        group_names=np.array(group_names),
                        adapter_authors=np.array([a for a, _ in adapters]),
                        **{k: matrices[k] for k in AGG_KEYS})
    result = {
        "config": vars(args), "n_adapters": len(adapters), "n_prompts": int(n_prompts),
        "script_sha256": _script_sha(), "git_hash": _git_hash(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "matrices_npz": npz_path, "summary": summary, "per_adapter": per_adapter,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[keyfire] gate: median on/off ||sBAh|| ratio = {summary['gate_median']:.3f} "
          f"-> {summary['gate_verdict']}")
    print(f"[keyfire] wrote {args.out} (+ {npz_path})")


if __name__ == "__main__":
    main()
