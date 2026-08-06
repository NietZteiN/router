"""Build depth eval dirs and collect F(d) for the S³T performance-retention curve.

F(d) = ensemble model_utility when every shard's constituent is trained on its top
d slices. Depth d maps to a shard's stage snapshot stages/stage_{d-1} (cumulative
training on d slices); d=0 is the base model. The S3T-vs-SISA performance gap is
driven entirely by which depth survives deletions (s3t_deletion.py), so this single
F(d) curve (measured once) feeds both systems.

Two subcommands:
  build   — create per-depth symlink eval dirs from an existing m-shard S3T dir
            (shard_i/stages/stage_{d-1}), copy the KS reference, print eval cmds.
  collect — read the per-depth result JSONs into F (length L+1) + write F.json.

A depth dir is loader-flat: {src}_depth{d}/shard_i -> {src}/shard_i/stages/stage_{d-1},
so eval_tofu.py runs `ensemble_probs --k 10 --forget_shard_id 9` on it unchanged.
"""
import argparse
import json
import os

BASE_UTILITY = 0.4179   # Llama-2-7B-chat-hf base model_utility (SHARD_GRID_REPORT_2026-06-11)


def depth_dir(src, d):
    return f"{src}_depth{d}"


def build(src, m, L, ks_ref):
    """Create {src}_depth{d} for d=1..L (symlinks to stage_{d-1}); return their paths."""
    dirs = []
    for d in range(1, L + 1):
        dd = depth_dir(src, d)
        os.makedirs(os.path.join(dd, "results", "smoke"), exist_ok=True)
        for i in range(m):
            link = os.path.join(dd, f"shard_{i}")
            target = os.path.join("..", os.path.basename(src), f"shard_{i}", "stages", f"stage_{d-1}")
            if not os.path.lexists(link):
                os.symlink(target, link)
        # KS reference for forget_quality (variant-independent k=10 forget rows).
        dst_ref = os.path.join(dd, "results", "smoke", "retain_tr_scores.npy")
        if ks_ref and os.path.exists(ks_ref) and not os.path.exists(dst_ref):
            import shutil
            shutil.copy(ks_ref, dst_ref)
        dirs.append(dd)
    return dirs


def collect(src, L, results_sub="smoke", label="ensemble_probs"):
    """Assemble F[0..L] from the per-depth result JSONs (F[0]=base)."""
    F = [BASE_UTILITY] + [float("nan")] * L
    fq = [float("nan")] * (L + 1)
    for d in range(1, L + 1):
        p = os.path.join(depth_dir(src, d), "results", results_sub, f"{label}.json")
        if os.path.exists(p):
            with open(p) as f:
                row = json.load(f)
            F[d] = row.get("model_utility", float("nan"))
            fq[d] = row.get("forget_quality", float("nan"))
        else:
            print(f"[collect] missing {p}")
    out = {"F": F, "forget_quality": fq, "label": label, "src": os.path.basename(src),
           "note": "F[d] = ensemble utility, every shard trained on d slices; F[0]=base"}
    with open(os.path.join(src, "F_curve.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True, help="m-shard S3T dir with shard_i/stages/stage_j")
    b.add_argument("--m", type=int, default=5)
    b.add_argument("--L", type=int, default=4)
    b.add_argument("--ks_ref", default=None, help="retain_tr_scores.npy to copy into each depth dir")
    c = sub.add_parser("collect")
    c.add_argument("--src", required=True)
    c.add_argument("--L", type=int, default=4)
    c.add_argument("--results_sub", default="smoke")
    c.add_argument("--label", default="ensemble_probs")
    a = p.parse_args()
    if a.cmd == "build":
        dirs = build(a.src, a.m, a.L, a.ks_ref)
        print("\n".join(dirs))
    else:
        collect(a.src, a.L, a.results_sub, a.label)


if __name__ == "__main__":
    main()
