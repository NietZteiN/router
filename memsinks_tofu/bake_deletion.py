"""Bake deletion/serving conditions into bone-stock PEFT adapter dirs (CPU).

For a fixed per-neuron scale vector v, hook-masking == row-scaling lora_B
(memsinks_model docstring; unit-tested bake≡hook in test_memsinks.py), so each
serving condition is materialized by scaling rows of the gate/up lora_B
tensors in trained/adapter_model.safetensors. Only {0,1} (and optional
sink_scale) values are used; 0/1 row-scaling is exact in any dtype.

Standard modes (bake all with --modes default):
  del_forget10  zero union of authors 180-199's sink masks
  del_forget05  zero union of authors 190-199
  del_forget01  zero union of authors 198-199
  dropall       zero the entire sink pool (paper-"dropout" analogue)
  randdel       zero union of 20 seeded random RETAINED authors (placebo:
                deletion should be author-specific, not generic damage)

Each out dir gets deletion_meta.json: authors, zeroed-neuron count, union
fraction, retained-author collateral stats, source safetensors sha256.
Serve with: eval_tofu.py --preloaded_adapter <run_dir>/baked/<mode>
"""
import argparse
import hashlib
import json
import os
import re
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import masks as M
from memsinks_model import build_scale_vector, load_masks

SINK_KEY_RE = r"\.mlp\.({mods})\.lora_B\.weight$"

MODE_AUTHORS = {
    "del_forget10": list(range(180, 200)),
    "del_forget05": list(range(190, 200)),
    "del_forget01": [198, 199],
}


def bake_one(trained_dir, out_dir, v, sink_modules, meta):
    src_path = os.path.join(trained_dir, "adapter_model.safetensors")
    tensors = load_file(src_path)
    pat = re.compile(SINK_KEY_RE.format(mods="|".join(sink_modules)))
    n_modified = 0
    for key in list(tensors.keys()):
        if pat.search(key):
            w = tensors[key]
            assert w.shape[0] == v.shape[0], f"{key}: rows {w.shape[0]} != vector {v.shape[0]}"
            tensors[key] = (w.float() * v.unsqueeze(1)).to(w.dtype)
            n_modified += 1
    assert n_modified > 0, f"no sink lora_B tensors matched in {src_path}"
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    shutil.copy2(os.path.join(trained_dir, "adapter_config.json"),
                 os.path.join(out_dir, "adapter_config.json"))
    with open(src_path, "rb") as f:
        meta["source_sha256"] = hashlib.sha256(f.read()).hexdigest()
    meta["tensors_modified"] = n_modified
    meta["zeroed_neurons"] = int((v == 0).sum().item())
    with open(os.path.join(out_dir, "deletion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[bake] {out_dir}: {n_modified} tensors, {meta['zeroed_neurons']} neurons zeroed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", required=True, help="dir containing trained/")
    p.add_argument("--modes", default="del_forget10,del_forget05,del_forget01,dropall,randdel")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    trained = os.path.join(args.run_dir, "trained")
    mask_table, num_gen = load_masks(trained)
    forget = set(cfg["forget_authors"])
    assert MODE_AUTHORS["del_forget10"] == sorted(forget), \
        "config forget_authors != canonical forget10 (180-199)"

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        if mode in MODE_AUTHORS:
            authors = MODE_AUTHORS[mode]
            v = build_scale_vector(mask_table, num_gen, "delete", forget_authors=authors)
        elif mode == "dropall":
            authors = None
            v = build_scale_vector(mask_table, num_gen, "dropall")
        elif mode == "randdel":
            rng = np.random.default_rng(cfg["seed"])
            retained = [a for a in range(cfg["num_authors"]) if a not in forget]
            authors = sorted(rng.choice(retained, size=20, replace=False).tolist())
            v = build_scale_vector(mask_table, num_gen, "delete", forget_authors=authors)
        else:
            raise SystemExit(f"unknown mode {mode!r}")
        meta = {
            "mode": mode,
            "authors": authors,
            "id_scheme": cfg["id_scheme"],
            "num_gen": num_gen,
            "mask_sha256": M.table_sha256(mask_table),
            "collateral": (M.collateral_stats(mask_table, authors) if authors else
                           {"union_fraction": 1.0, "note": "dropall"}),
        }
        # per_retained_overlap is 200 entries of noise in disjoint mode — keep summary only
        if "per_retained_overlap" in meta["collateral"]:
            per = meta["collateral"].pop("per_retained_overlap")
            nonzero = {a: o for a, o in per.items() if o > 0}
            meta["collateral"]["n_retained_with_overlap"] = len(nonzero)
        bake_one(trained, os.path.join(args.run_dir, "baked", mode), v,
                 cfg["sink_modules"], meta)


if __name__ == "__main__":
    main()
