"""Compose per-shard VeRA / IA³ adapters into ONE adapter dir (peft_compose bake-off).

Pure file-space composition — no model load. Each method's rule operates on the trainable
vectors in adapter_model.safetensors and writes a normal adapter dir servable via
`eval_tofu --preloaded_adapter`:

  vera  keys containing 'vera_lambda'  -> elementwise MEAN across shards (composition in the
        SHARED frozen basis; all other keys — incl. the saved vera_A/vera_B projections —
        must be identical across shards and are copied through after an allclose assert).
  ia3   keys containing 'ia3_l'        -> 'mean' = arithmetic mean of the gate vectors;
        'geo' = signed geometric mean where every shard agrees in sign (the true product
        analog, scale-composing), falling back to the arithmetic mean elementwise where
        signs disagree (gates init at 1 and rarely cross 0; the fallback count is printed).

Exact deletion is verified before anything is written: for the mean rule,
delete(compose(all), i) := (n*mean_all - x_i)/(n-1) must equal compose(all minus i) — the
assert makes "drop an author in O(1) from stored vectors" a checked property, not a hope.
(geo: deletion = recompute over the remaining shards from stored vectors — same O(1) storage.)

CLI:
    python compose_peft.py --method vera --pool_dir DIR --out DIR_composed            # all shards
    python compose_peft.py --method ia3  --pool_dir DIR --out DIR_minus9 --exclude 9  # deletion
    python compose_peft.py --method ia3 --variant geo ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

TRAINED_KEY_SUBSTR = {"vera": "vera_lambda", "ia3": "ia3_l"}


def _shard_dirs(pool_dir: str, k: int, exclude: set) -> list:
    dirs = []
    for i in range(k):
        d = os.path.join(pool_dir, f"shard_{i}")
        if i in exclude:
            continue
        if not os.path.isdir(d):
            raise ValueError(f"missing shard dir {d} — refusing a silent partial composition")
        dirs.append(d)
    return dirs


def _load_states(dirs: list) -> list:
    return [load_file(os.path.join(d, "adapter_model.safetensors")) for d in dirs]


def _split_keys(states: list, key_substr: str):
    keys = set(states[0])
    for s in states[1:]:
        if set(s) != keys:
            raise ValueError("shard adapters disagree on state-dict keys — different configs?")
    trained = sorted(k for k in keys if key_substr in k)
    shared = sorted(keys - set(trained))
    if not trained:
        raise ValueError(f"no keys containing {key_substr!r} — wrong --method for this pool?")
    return trained, shared


def compose_states(states: list, key_substr: str, variant: str = "mean"):
    """Composed state dict + #elements that fell back to arithmetic mean (geo only)."""
    trained, shared = _split_keys(states, key_substr)
    out, fallback = {}, 0
    for k in shared:
        ref = states[0][k]
        for s in states[1:]:
            if not torch.allclose(s[k], ref):
                raise ValueError(f"shared key {k} differs across shards "
                                 f"(vera: projection_prng_key mismatch?)")
        out[k] = ref.clone()
    for k in trained:
        stack = torch.stack([s[k].to(torch.float64) for s in states])
        if variant == "mean":
            comp = stack.mean(dim=0)
        elif variant == "geo":
            sign_agree = (torch.sign(stack) == torch.sign(stack[0])).all(dim=0) \
                         & (stack.abs() > 0).all(dim=0)
            geo = torch.sign(stack[0]) * torch.exp(torch.log(stack.abs().clamp_min(1e-30)).mean(dim=0))
            arith = stack.mean(dim=0)
            comp = torch.where(sign_agree, geo, arith)
            fallback += int((~sign_agree).sum())
        else:
            raise ValueError(variant)
        out[k] = comp.to(states[0][k].dtype)
    return out, fallback


def delete_from_mean(mean_state: dict, drop_state: dict, n: int, key_substr: str) -> dict:
    """O(1) deletion on the mean rule: (n*mean - x_drop)/(n-1) on trained keys only."""
    out = {}
    for k, v in mean_state.items():
        if key_substr in k:
            out[k] = ((n * v.to(torch.float64) - drop_state[k].to(torch.float64))
                      / (n - 1)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def verify_exact_deletion(pool_dir: str, k: int, method: str, drop: int) -> float:
    """max |compose(all minus drop) − delete(compose(all), drop)| over trained keys (mean rule)."""
    key_substr = TRAINED_KEY_SUBSTR[method]
    all_dirs = _shard_dirs(pool_dir, k, exclude=set())
    states = _load_states(all_dirs)
    full, _ = compose_states(states, key_substr, "mean")
    drop_idx = [os.path.basename(d) for d in all_dirs].index(f"shard_{drop}")
    deleted = delete_from_mean(full, states[drop_idx], len(states), key_substr)
    direct, _ = compose_states([s for j, s in enumerate(states) if j != drop_idx],
                               key_substr, "mean")
    worst = 0.0
    for kk in direct:
        if key_substr in kk:
            worst = max(worst, (direct[kk].to(torch.float64)
                                - deleted[kk].to(torch.float64)).abs().max().item())
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["vera", "ia3"])
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--variant", default="mean", choices=["mean", "geo"])
    ap.add_argument("--verify_drop", type=int, default=9,
                    help="shard id for the exact-deletion identity check (mean rule); -1 skips")
    args = ap.parse_args()

    key_substr = TRAINED_KEY_SUBSTR[args.method]
    if args.verify_drop >= 0 and args.variant == "mean":
        worst = verify_exact_deletion(args.pool_dir, args.k, args.method, args.verify_drop)
        assert worst < 1e-6, f"exact-deletion identity failed: max diff {worst:.3e}"
        print(f"[compose_peft] exact-deletion identity ok (drop shard_{args.verify_drop}, "
              f"max |diff| = {worst:.3e})")

    dirs = _shard_dirs(args.pool_dir, args.k, set(args.exclude))
    states = _load_states(dirs)
    comp, fallback = compose_states(states, key_substr, args.variant)
    if args.variant == "geo":
        total = sum(v.numel() for k, v in comp.items() if key_substr in k)
        print(f"[compose_peft] geo fallback-to-mean elements: {fallback}/{total}")

    os.makedirs(args.out, exist_ok=True)
    save_file(comp, os.path.join(args.out, "adapter_model.safetensors"))
    shutil.copy(os.path.join(dirs[0], "adapter_config.json"),
                os.path.join(args.out, "adapter_config.json"))
    meta = {"method": args.method, "variant": args.variant, "k": args.k,
            "exclude": sorted(args.exclude), "n_composed": len(dirs),
            "shards": [os.path.basename(d) for d in dirs], "pool_dir": os.path.abspath(args.pool_dir)}
    with open(os.path.join(args.out, "compose_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[compose_peft] wrote {args.out} ({args.method}/{args.variant}, "
          f"n={len(dirs)}, exclude={sorted(args.exclude)})")


if __name__ == "__main__":
    main()
