"""Measure how exact the GPU unlearn re-derivation is (bitwise vs distributional).

The exact-unlearning guarantee needs the unlearn step to reproduce the SAME τ_u that
went into τ̄ during build. The CPU gate proves the *algorithm* is bitwise-exact; on GPU,
kernel nondeterminism sets a floor. We measure that floor directly: re-derive a forget
author's τ_u TWICE (identical seed/data, deterministic math/eager attention forced by
train_sift_masks.load_base) and compare — if byte-identical, GPU re-derivation is bitwise
and unlearning is bitwise-exact; otherwise `rel_l2_floor = ‖τ_a−τ_b‖ / ‖τ_a‖` quantifies
the residual (compare to the legonet distributional-exactness floor ≈ 4–6e-2).

  python measure_sift_exactness.py --config configs/sift_masks_tofu_1b.json --author 199 --out ex.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import sift_masks_data as smd
from train_sift_masks import _train_one, load_base, load_config, sift_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--author", type=int, default=199, help="a forget10 author (180–199)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = smd.load_gpt2_tokenizer(cfg["model_name"], cfg["hf_home"])
    model, names, theta0 = load_base(cfg, device)          # forces math/eager attention
    sign_cpu = torch.load(os.path.join(sift_dir(cfg), "sign_v.pt"))
    sign = {n: sign_cpu[n].to(device) for n in names}
    full = smd.load_tofu_full(cfg["hf_home"])

    tau_a, _ = _train_one(model, theta0, sign, names, tok, full, args.author, cfg, device)
    tau_b, _ = _train_one(model, theta0, sign, names, tok, full, args.author, cfg, device)

    diff2, sig2, maxabs, byteeq = 0.0, 0.0, 0.0, True
    for n in names:
        d = (tau_a[n] - tau_b[n]).float()
        diff2 += float((d * d).sum())
        sig2 += float((tau_a[n].float() ** 2).sum())
        maxabs = max(maxabs, float(d.abs().max()))
        if not torch.equal(tau_a[n], tau_b[n]):
            byteeq = False
    rel = (diff2 ** 0.5) / (sig2 ** 0.5) if sig2 > 0 else float("nan")
    row = {
        "author": args.author,
        "bitwise_identical": byteeq,
        "rel_l2_floor": rel,
        "diff_l2": diff2 ** 0.5,
        "tau_l2": sig2 ** 0.5,
        "max_abs_diff": maxabs,
        "device": device,
        "note": "bitwise_identical=True -> GPU unlearn is bitwise-exact; else rel_l2_floor "
                "is the nondeterminism floor (cf. legonet distributional ~4-6e-2).",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(row, open(args.out, "w"), indent=2)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
