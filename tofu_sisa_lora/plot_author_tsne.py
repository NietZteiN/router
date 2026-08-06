"""LoRA parameter-space t-SNE for TOFU per-author adapters (HydraLoRA Fig-1(b) analog).

Each point = one per-author LoRA adapter's effective delta DW_a = scaling * B_a A_a,
embedded in 2D with t-SNE over the PRECOMPUTED pairwise cosine distances from a
subspace_overlap.py result JSON (its `cosine_matrix` is the factored Frobenius cosine
of the full concatenated deltas — matrix index == author id when the run used sort -V
shard order, as submit_subspace_k200.sh does). Pure post-processing: reads only the
report JSON (+ optional author_emb.npy), re-runnable without checkpoints.

Colorings:
  (a) forget-split nesting (shard_utils ground truth): retain 0-179 / forget10\05
      180-189 / forget05\01 190-197 / forget01 198-199 — is the forget set
      geometrically separable in weight space?
  (b) optional semantic k-means clusters of the frozen MiniLM answer-mean author
      embeddings (legonet setup artifact, model-independent) — does weight-space
      geometry mirror semantic geometry (the wmdp-bio/chem/cyber analog)?

Because t-SNE cluster shapes are perplexity-sensitive, a perplexity sweep panel is
emitted as the robustness check, and a silhouette score computed on the PRECOMPUTED
distances (never the 2D coords) is written to the sidecar JSON so the visual cannot
be over-read: near-uniform off-diagonal cosines (k200_r32: 0.0012 +/- ~3e-4) predict
chaos — silhouette ~ 0 for every labeling.

⚠ Run with BASE anaconda python (matplotlib lives only there, not test-env):
    ${TOFU_PLOT_PYTHON:-python3} plot_author_tsne.py \
        --json reports/subspace_overlap_k200_r32.json \
        --author_emb checkpoints/Llama-2-7B-chat-hf_legonet_n32_k3/legonet/author_emb.npy \
        --out_dir reports/figures/lora_tsne
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Validated reference palette (dataviz skill), light surface. Retain is the recessive
# mass (gray); the nested forget tiers are ORDINAL -> one-hue blue ramp, not 3 hues.
INK, MUTED, HAIRLINE = "#0b0b0b", "#898781", "#e1e0d9"
RETAIN_GRAY = "#898781"
FORGET_RAMP = ["#86b6ef", "#2a78d6", "#0d366b"]  # blue steps 250 / 450 / 700
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

FORGET_CLASSES = [  # (name, author-id predicate) in draw order: gray mass first
    ("retain (0–179)", lambda a: a < 180),
    ("forget10∖05 (180–189)", lambda a: 180 <= a < 190),
    ("forget05∖01 (190–197)", lambda a: 190 <= a < 198),
    ("forget01 (198–199)", lambda a: a >= 198),
]


def author_id(adapter_id: str) -> int:
    m = re.search(r"(\d+)$", adapter_id)
    if m is None:
        raise ValueError(f"cannot parse author id from adapter id {adapter_id!r}")
    return int(m.group(1))


def forget_class(a: int) -> int:
    """Index into FORGET_CLASSES for author id `a`."""
    for idx, (_, pred) in enumerate(FORGET_CLASSES):
        if pred(a):
            return idx
    raise AssertionError(a)


def cosine_to_distance(cos: np.ndarray) -> np.ndarray:
    """1 - cosine, symmetrized, clipped to >=0, zero diagonal (t-SNE precomputed input)."""
    D = 1.0 - np.asarray(cos, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.clip(D, 0.0, None, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def run_tsne(D: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    """(n, 2) t-SNE coords from a precomputed distance matrix. Deterministic per seed."""
    ts = TSNE(n_components=2, metric="precomputed", init="random",
              perplexity=perplexity, random_state=seed)
    return ts.fit_transform(D)


def usable_perplexities(requested: list, n: int) -> list:
    """sklearn needs perplexity < n; keep valid ones, else fall back to (n-1)//3."""
    ok = [p for p in requested if p < n]
    if not ok:
        ok = [max(2, (n - 1) // 3)]
        print(f"[plot_author_tsne] n={n}: no requested perplexity valid, using {ok[0]}")
    dropped = [p for p in requested if p >= n]
    if dropped:
        print(f"[plot_author_tsne] n={n}: dropping perplexities {dropped} (need < n)")
    return ok


def kmeans_labels(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Seeded k-means over L2-normalized embeddings; clusters relabeled by size desc
    so slot colors are stable across reruns."""
    from sklearn.cluster import KMeans
    X = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    raw = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    order = np.argsort([-(raw == c).sum() for c in range(k)])
    remap = {int(c): i for i, c in enumerate(order)}
    return np.array([remap[int(c)] for c in raw])


def silhouette_or_nan(D: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette on the precomputed distances; NaN when <2 populated classes."""
    labels = np.asarray(labels)
    pop = [c for c in np.unique(labels) if (labels == c).sum() >= 2]
    if len(pop) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(D, labels, metric="precomputed"))


def _axes_chrome(ax, title):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(HAIRLINE)
    ax.set_xlabel("t-SNE dim 1", color=MUTED, fontsize=8)
    ax.set_ylabel("t-SNE dim 2", color=MUTED, fontsize=8)
    ax.set_title(title, color=INK, fontsize=10)


def _scatter_classes(ax, xy, class_idx, names, colors, sizes, alphas, legend_kw=None):
    for c, (name, color) in enumerate(zip(names, colors)):
        m = class_idx == c
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=sizes[c], c=color, alpha=alphas[c],
                   edgecolors="white", linewidths=0.6,
                   label=f"{name}  n={int(m.sum())}")
    leg = ax.legend(frameon=False, fontsize=7.5, **(legend_kw or {"loc": "best"}))
    for t in leg.get_texts():
        t.set_color(INK)


def plot_forget(xy, fclass, title, path_base):
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    names = [n for n, _ in FORGET_CLASSES]
    colors = [RETAIN_GRAY] + FORGET_RAMP
    _scatter_classes(ax, xy, fclass, names, colors,
                     sizes=[16, 34, 34, 40], alphas=[0.45, 0.95, 0.95, 1.0])
    _axes_chrome(ax, title)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_base}.{ext}")
    plt.close(fig)


def plot_semantic(xy, km, title, path_base):
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=200)
    k = int(km.max()) + 1
    names = [f"cluster {c}" for c in range(k)]
    # legend below the axes — 6 in-plot entries would sit on top of data points
    _scatter_classes(ax, xy, km, names, CATEGORICAL[:k],
                     sizes=[22] * k, alphas=[0.85] * k,
                     legend_kw={"loc": "upper center", "bbox_to_anchor": (0.5, -0.08),
                                "ncols": 3, "columnspacing": 1.2, "handletextpad": 0.3})
    _axes_chrome(ax, title)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_base}.{ext}")
    plt.close(fig)


def plot_sweep(coords_by_p, fclass, title, path):
    ps = sorted(coords_by_p)
    ncol = min(len(ps), 4)
    nrow = (len(ps) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), dpi=200,
                             squeeze=False)
    colors = [RETAIN_GRAY] + FORGET_RAMP
    for ax, p in zip(axes.flat, ps):
        xy = coords_by_p[p]
        for c in range(len(FORGET_CLASSES)):
            m = fclass == c
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=8 if c == 0 else 18, c=colors[c],
                           alpha=0.45 if c == 0 else 0.95,
                           edgecolors="white", linewidths=0.4)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(HAIRLINE)
        ax.set_title(f"perplexity {p}", color=MUTED, fontsize=9)
    for ax in axes.flat[len(ps):]:
        ax.axis("off")
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="subspace_overlap.py result JSON (needs cosine_matrix)")
    ap.add_argument("--author_emb", default=None, help="author_emb.npy (200,384) for semantic k-means coloring")
    ap.add_argument("--kmeans_k", type=int, default=6)
    ap.add_argument("--perplexities", default="5,15,30,50")
    ap.add_argument("--main_perplexity", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="reports/figures/lora_tsne")
    ap.add_argument("--tag", default=None, help="output name tag; default from the JSON basename")
    args = ap.parse_args()

    with open(args.json) as f:
        res = json.load(f)
    ids = res["adapter_ids"]
    C = np.array(res["cosine_matrix"], dtype=np.float64)
    n = len(ids)
    assert C.shape == (n, n), f"cosine_matrix {C.shape} vs {n} ids"
    authors = np.array([author_id(i) for i in ids])
    fclass = np.array([forget_class(a) for a in authors])
    D = cosine_to_distance(C)
    off = C[~np.eye(n, dtype=bool)]

    tag = args.tag or re.sub(r"^subspace_overlap_", "", os.path.splitext(os.path.basename(args.json))[0])
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"tsne_{tag}")

    ps = usable_perplexities([int(p) for p in args.perplexities.split(",")], n)
    main_p = args.main_perplexity if args.main_perplexity in ps else ps[len(ps) // 2]
    coords = {p: run_tsne(D, p, args.seed) for p in ps}
    xy = coords[main_p]

    km = None
    if args.author_emb:
        emb = np.load(args.author_emb)
        assert emb.shape[0] == n, f"author_emb rows {emb.shape[0]} != n adapters {n}"
        km = kmeans_labels(emb, args.kmeans_k, args.seed)

    sil = {
        "forget_binary_vs_retain": silhouette_or_nan(D, (authors >= 180).astype(int)),
        "forget_4class": silhouette_or_nan(D, fclass),
    }
    if km is not None:
        sil[f"semantic_kmeans_k{args.kmeans_k}"] = silhouette_or_nan(D, km)

    subtitle = f"TOFU per-author LoRA deltas (n={n}, perplexity {main_p})"
    plot_forget(xy, fclass, subtitle, f"{base}_forget")
    if km is not None:
        plot_semantic(xy, km, subtitle, f"{base}_semantic")
    plot_sweep(coords, fclass, f"t-SNE perplexity robustness — {tag}", f"{base}_perplexity_sweep.png")

    with open(f"{base}_coords.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["author_id", "perplexity", "x", "y", "forget_class", "kmeans_cluster"])
        for p in ps:
            for i in range(n):
                w.writerow([int(authors[i]), p, f"{coords[p][i, 0]:.6f}", f"{coords[p][i, 1]:.6f}",
                            FORGET_CLASSES[fclass[i]][0], (int(km[i]) if km is not None else "")])

    sidecar = {
        "source_json": os.path.abspath(args.json),
        "author_emb": os.path.abspath(args.author_emb) if args.author_emb else None,
        "n": n, "seed": args.seed, "perplexities": ps, "main_perplexity": main_p,
        "kmeans_k": args.kmeans_k if km is not None else None,
        "offdiag_cosine": {"mean": float(off.mean()), "min": float(off.min()), "max": float(off.max())},
        "silhouette_on_precomputed_distances": sil,
        "class_counts": {name: int((fclass == c).sum()) for c, (name, _) in enumerate(FORGET_CLASSES)},
    }
    with open(f"{base}_meta.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[plot_author_tsne] n={n} tag={tag} perplexities={ps} main={main_p}")
    print(f"  offdiag cosine mean={off.mean():.4f} range=[{off.min():.4f}, {off.max():.4f}]")
    for k, v in sil.items():
        print(f"  silhouette[{k}] = {v:.4f}" if v == v else f"  silhouette[{k}] = NaN (class empty)")
    print(f"  wrote {base}_forget.png/.pdf"
          + (f", {base}_semantic.png/.pdf" if km is not None else "")
          + f", {base}_perplexity_sweep.png, {base}_coords.csv, {base}_meta.json")


if __name__ == "__main__":
    main()
