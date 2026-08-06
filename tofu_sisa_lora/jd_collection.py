"""On-disk Joint Diagonalization for large adapter collections (the scale path).

Reads LoRA factors directly from PEFT adapter directories (`adapter_model.safetensors`)
on CPU, runs `jd_compress`, and writes a compact compressed artifact. This bypasses
PEFT's `load_adapter`, which casts every adapter to fp32 in GPU memory (the "high-k
memory law" in CLAUDE.md) — the whole point of JD is to not pay n * params, so for
hundreds/thousands of adapters we never instantiate them all on the model.

Artifact layout (under out_dir):
    manifest.json                # ids, variant, rank, n_clusters, assignment, norms, slots, config
    bases/{slot}.safetensors     # U_j, V_j per cluster (stacked: key "U"/"V" -> (c, d, r))
    sigmas/{slot}.safetensors     # per-adapter Sigma (stacked: (n, r, r) full or (n, r) diag)

`materialize_adapter` reconstructs a kept-subset merge as a normal PEFT adapter dir so
`eval_tofu.py` can evaluate it like any other label. Deleting an adapter from the
keep-set is an O(1) change to the cluster Sigma-sum (no refit) — see
`jd_compress.JDCompressed.merge_keepset`.

CLI:
    python jd_collection.py build   --adapters DIR... --out OUT --variant full --clusters 4 --rank 16
    python jd_collection.py select  --adapters DIR... --rank 16            # cluster-count sweep
    python jd_collection.py merge   --collection OUT --out ADAPTER_DIR [--drop ID...]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from safetensors.torch import load_file, save_file

import jd_compress

_PREFIX = "base_model.model."


# ---------------------------------------------------------------------------
# Reading adapter dirs
# ---------------------------------------------------------------------------

def _adapter_scaling(cfg):
    """PEFT LoRA scaling: alpha/sqrt(r) under rslora, else alpha/r."""
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def _read_adapter(adapter_dir):
    """Return ({slot_name: (A, B)}, config_dict). Slot = module stem under base_model.model."""
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    slots = {}
    for key, val in tensors.items():
        name = key[len(_PREFIX):] if key.startswith(_PREFIX) else key
        for side in ("A", "B"):
            tag = f".lora_{side}.weight"
            if name.endswith(tag):
                slot = name[: -len(tag)]
                slots.setdefault(slot, {})[side] = val.float()
    return {s: (d["A"], d["B"]) for s, d in slots.items() if "A" in d and "B" in d}, cfg


def build_collection_slots(adapter_dirs, device="cpu"):
    """Assemble jd_compress.Slot objects across a list of adapter dirs (aligned slots).

    Returns (slots, adapter_ids, ref_cfg). Uses the slot set of the first adapter; raises
    if any adapter is missing a slot (collections must share the target-module layout).
    `device` places the factor tensors (use "cuda" so the JD SVDs run on GPU — CPU is
    intractable for hundreds of adapters x ~150 modules).
    """
    adapter_ids = [os.path.basename(os.path.normpath(d)) for d in adapter_dirs]
    per_adapter, cfgs = [], []
    for d in adapter_dirs:
        s, cfg = _read_adapter(d)
        per_adapter.append(s)
        cfgs.append(cfg)
    slot_names = list(per_adapter[0].keys())
    slots = {}
    for name in slot_names:
        B, A, scaling = [], [], []
        for a, cfg in zip(per_adapter, cfgs):
            if name not in a:
                raise ValueError(f"adapter missing slot {name!r}; collections must share layout")
            Ai, Bi = a[name]
            A.append(Ai.to(device))
            B.append(Bi.to(device))
            scaling.append(_adapter_scaling(cfg))
        slots[name] = jd_compress.Slot(B=B, A=A, scaling=scaling)
    return slots, adapter_ids, cfgs[0]


# ---------------------------------------------------------------------------
# Save / load the compressed artifact
# ---------------------------------------------------------------------------

def save_jd(jd: jd_compress.JDCompressed, out_dir, ref_cfg):
    os.makedirs(os.path.join(out_dir, "bases"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sigmas"), exist_ok=True)
    safe = lambda s: s.replace("/", "__").replace(".", "_")
    slot_files = {}
    for name, js in jd.slots.items():
        fn = safe(name)
        slot_files[name] = fn
        save_file({"U": torch.stack(js.U).cpu(), "V": torch.stack(js.V).cpu()},
                  os.path.join(out_dir, "bases", fn + ".safetensors"))
        save_file({"sigma": torch.stack(js.sigma).cpu()},
                  os.path.join(out_dir, "sigmas", fn + ".safetensors"))
    manifest = {
        "adapter_ids": jd.adapter_ids, "variant": jd.variant, "rank": jd.rank,
        "n_clusters": jd.n_clusters, "assignment": jd.assignment,
        "norm": jd.norm.tolist(), "recon_err": jd.recon_err.tolist(),
        "slot_files": slot_files, "ref_cfg": ref_cfg,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def load_jd(out_dir):
    with open(os.path.join(out_dir, "manifest.json")) as f:
        man = json.load(f)
    slots = {}
    for name, fn in man["slot_files"].items():
        bases = load_file(os.path.join(out_dir, "bases", fn + ".safetensors"))
        sig = load_file(os.path.join(out_dir, "sigmas", fn + ".safetensors"))["sigma"]
        U = list(bases["U"].unbind(0))
        V = list(bases["V"].unbind(0))
        sigma = list(sig.unbind(0))
        slots[name] = jd_compress.JDSlot(U=U, V=V, sigma=sigma)
    jd = jd_compress.JDCompressed(
        adapter_ids=man["adapter_ids"], variant=man["variant"], rank=man["rank"],
        n_clusters=man["n_clusters"], assignment=man["assignment"],
        norm=torch.tensor(man["norm"]), slots=slots,
        recon_err=torch.tensor(man["recon_err"]),
    )
    return jd, man["ref_cfg"]


# ---------------------------------------------------------------------------
# Build + materialize
# ---------------------------------------------------------------------------

def build_jd_collection(adapter_dirs, *, variant="full", clusters=1, rank=16,
                        out_dir=None, seed=0, iters=10, device="cpu"):
    # Factors stay on CPU; `device` is the *compute* device — jd_compress streams each
    # module's factors onto it on demand, so peak GPU memory is ~flat in adapter count.
    slots, adapter_ids, ref_cfg = build_collection_slots(adapter_dirs, device="cpu")
    jd = jd_compress.jd_compress_collection(
        slots, adapter_ids, variant=variant, clusters=clusters,
        rank=rank, iters=iters, seed=seed, compute_device=device)
    if out_dir:
        save_jd(jd, out_dir, ref_cfg)
    return jd, ref_cfg


def materialize_adapter(jd, ref_cfg, out_dir, *, keep=None, weights=None):
    """Write the kept-subset merge as a normal PEFT adapter dir (loadable by eval_tofu).

    Factors are zero-padded to a uniform rank = ref_cfg['r'] and divided by the adapter
    scaling (rslora) so PEFT's get_delta_weight reproduces the true effective merged
    delta. `keep` defaults to all adapters; pass a subset (or use `drop`) to unlearn.
    """
    if keep is None:
        keep = list(range(len(jd.adapter_ids)))
    out_rank = ref_cfg["r"]
    scaling = _adapter_scaling(ref_cfg)
    merged = jd.merge_keepset(keep, weights=weights, out_rank=out_rank)
    tensors = {}
    for slot, (A_new, B_new) in merged.items():
        d_in = A_new.shape[1] if A_new.numel() else jd.slots[slot].V[0].shape[0]
        d_out = B_new.shape[0] if B_new.numel() else jd.slots[slot].U[0].shape[0]
        A_pad = torch.zeros(out_rank, d_in)
        B_pad = torch.zeros(d_out, out_rank)
        r = min(out_rank, A_new.shape[0])
        A_pad[:r] = A_new[:r].cpu()
        B_pad[:, :r] = (B_new[:, :r] / scaling).cpu()
        key = _PREFIX + slot
        tensors[key + ".lora_A.weight"] = A_pad.contiguous()
        tensors[key + ".lora_B.weight"] = B_pad.contiguous()
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = dict(ref_cfg)
    cfg["r"] = out_rank
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="compress a collection and save the artifact")
    b.add_argument("--adapters", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--variant", choices=["full", "diag"], default="full")
    b.add_argument("--clusters", type=int, default=1)
    b.add_argument("--rank", type=int, default=16)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", default="cpu", help="cpu | cuda (use cuda for large collections)")

    s = sub.add_parser("select", help="cluster-count sweep on a probe slot (recon error)")
    s.add_argument("--adapters", nargs="+", required=True)
    s.add_argument("--variant", choices=["full", "diag"], default="full")
    s.add_argument("--rank", type=int, default=16)
    s.add_argument("--threshold", type=float, default=0.6)
    s.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("merge", help="materialize a kept-subset merge as a PEFT adapter dir")
    m.add_argument("--collection", required=True, help="artifact dir from `build`")
    m.add_argument("--out", required=True)
    m.add_argument("--drop", nargs="*", default=[], help="adapter ids to unlearn (drop)")

    args = p.parse_args()
    if args.cmd == "build":
        jd, _ = build_jd_collection(
            args.adapters, variant=args.variant, clusters=args.clusters,
            rank=args.rank, out_dir=args.out, seed=args.seed, device=args.device)
        print(f"[build] {len(jd.adapter_ids)} adapters, variant={jd.variant}, "
              f"c={jd.n_clusters}, r={jd.rank} -> {args.out}  "
              f"recon_err={jd.reconstruction_error():.4f}")
    elif args.cmd == "select":
        slots, ids, _ = build_collection_slots(args.adapters)
        c, errs = jd_compress.select_num_clusters(
            slots, ids, variant=args.variant, rank=args.rank,
            threshold=args.threshold, seed=args.seed)
        print(f"[select] recon error by cluster count: "
              f"{ {k: round(v,4) for k,v in errs.items()} }")
        print(f"[select] chosen c (first below {args.threshold}): {c}")
    elif args.cmd == "merge":
        jd, ref_cfg = load_jd(args.collection)
        drop = set(args.drop)
        keep = [i for i, a in enumerate(jd.adapter_ids) if a not in drop]
        materialize_adapter(jd, ref_cfg, args.out, keep=keep)
        print(f"[merge] kept {len(keep)}/{len(jd.adapter_ids)} "
              f"(dropped {sorted(drop)}) -> {args.out}")


if __name__ == "__main__":
    main()
