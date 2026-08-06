"""RQ3 (S³T paper §7.3, Fig 8 / Fig 15-left): sequence-selection effectiveness.

Diversity = average pairwise edit distance within the selected sequence set; higher
is better (a deletion is less likely to hit the same position across sequences).

  - Uniform prior: iterative cyclic rotation vs random sampling (Fig 8-left).
  - Non-uniform prior: BMS vs random vs sorted-cyclic rotation, on BOTH edit
    distance and the Eq-24 score (Fig 8-center/right, Fig 15-left).

Pure CPU; writes {out}/rq3_diversity.json.
"""
import argparse
import itertools
import json
import os

import numpy as np

from s3t_sequences import bms, cyclic_permutation, iterative_cyclic_rotation, score


def edit_distance(a, b):
    """Positional Hamming distance between two permutations (paper's diversity
    metric: count positions where the two sequences differ)."""
    return sum(1 for x, y in zip(a, b) if x != y)


def avg_pairwise_edit_distance(seqs):
    if len(seqs) < 2:
        return 0.0
    ds = [edit_distance(a, b) for a, b in itertools.combinations(seqs, 2)]
    return float(np.mean(ds))


def random_sequences(L, B, seed=0):
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    # Enumerate all perms only when L is small; else sample.
    if L <= 7 and B >= 1:
        allp = list(itertools.permutations(range(L)))
        rng.shuffle(allp)
        for p in allp:
            out.append(p)
            if len(out) >= B:
                break
        return out
    while len(out) < B:
        p = tuple(rng.permutation(L))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def sorted_cyclic(L, probs, B):
    """Sorted cyclic rotation (paper §7.3): sort slices by ASCENDING deletion prob
    (least-deletable first), then take B cyclic rotations of that order."""
    order = tuple(int(i) for i in np.argsort(probs))
    return cyclic_permutation(order)[:B]


def uniform_experiment(L, Bs, n_random=20):
    """Fig 8-left: cyclic rotation vs random, edit distance vs budget B."""
    rows = []
    for B in Bs:
        cyc = avg_pairwise_edit_distance(iterative_cyclic_rotation(L, B))
        rnd = np.mean([avg_pairwise_edit_distance(random_sequences(L, B, seed=s))
                       for s in range(n_random)])
        rows.append({"B": B, "cyclic": round(cyc, 4), "random": round(float(rnd), 4)})
    return rows


def nonuniform_experiment(L, B, n_priors=10, t=1, seed=0):
    """Fig 8-center/right + Fig 15-left: BMS vs random vs sorted-cyclic on edit
    distance and Eq-24 score, averaged over Dirichlet-sampled priors."""
    rng = np.random.default_rng(seed)
    agg = {k: {"edit": [], "score": []} for k in ("bms", "random", "sorted_cyclic")}
    for _ in range(n_priors):
        probs = rng.dirichlet(np.ones(L))
        sets = {
            "bms": bms(L, B, probs, t=t),
            "random": random_sequences(L, B, seed=int(rng.integers(1 << 30))),
            "sorted_cyclic": sorted_cyclic(L, probs, B),
        }
        for name, seqs in sets.items():
            agg[name]["edit"].append(avg_pairwise_edit_distance(seqs))
            agg[name]["score"].append(float(np.mean([score(s, probs, t) for s in seqs])))
    return {name: {"edit": round(float(np.mean(v["edit"])), 4),
                   "score": round(float(np.mean(v["score"])), 4)}
            for name, v in agg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="checkpoints/Llama-2-7B-chat-hf_s3t_m5_L4_armA")
    p.add_argument("--L_uniform", type=int, default=5)
    p.add_argument("--L_nonuniform", type=int, default=4)
    p.add_argument("--B", type=int, default=4)
    a = p.parse_args()
    Bs = list(range(1, min(a.L_uniform, 5) + 1)) + [a.L_uniform * 2, a.L_uniform * 4]
    out = {
        "uniform": {"L": a.L_uniform,
                    "edit_distance_vs_B": uniform_experiment(a.L_uniform, Bs)},
        "nonuniform": {"L": a.L_nonuniform, "B": a.B,
                       "result": nonuniform_experiment(a.L_nonuniform, a.B)},
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "rq3_diversity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    nu = out["nonuniform"]["result"]
    print(f"\nRQ3 check: BMS score {nu['bms']['score']:.3f} >= sorted_cyclic "
          f"{nu['sorted_cyclic']['score']:.3f} >= random {nu['random']['score']:.3f}")


if __name__ == "__main__":
    main()
