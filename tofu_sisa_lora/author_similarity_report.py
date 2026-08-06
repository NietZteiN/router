"""Per-author adapter similarity report (merge-mechanism Exp 1 at k=200).

Post-processes a subspace_overlap.py result JSON (the 200 per-author k200_r32 SISA adapters)
into a human-readable similarity study:
  (a) top-N most / least similar author PAIRS by off-diagonal delta cosine, resolved to TOFU
      author names (offline extraction from the cached locuslab/TOFU 'full' split; falls back
      to shard ids if the dataset cache is unavailable),
  (b) per-author row-mean cosine ranking — whose delta is most "generic" vs most distinctive,
  (c) a PIL-rendered heatmap PNG of the full cosine matrix (test-env has no matplotlib; the
      color scale saturates at the 99.5th percentile of |off-diag| so the unit diagonal does
      not wash out the ~1e-2 structure),
  (d) a cross-run trend table vs earlier subspace_overlap JSONs (k4/k10/n32). NOTE: those are
      Llama-3.2-1B collections under other recipes (legacy r8, LegoNet r16) — the table shows
      direction of trend with n, NOT a controlled dial; the caveat is printed with it.

CPU-only; reads only report JSONs (re-runnable end-to-end without checkpoints).

CLI:
    python author_similarity_report.py \
        --json reports/subspace_overlap_k200_r32.json \
        --priors k4=reports/subspace_overlap_k4.json k10=reports/subspace_overlap_k10.json \
                 n32=reports/subspace_overlap_n32.json \
        --out_md reports/AUTHOR_SIMILARITY_K200_2026-07-07.md \
        --heatmap reports/author_similarity_k200_r32_heatmap.png --top 20
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np


def _author_id(adapter_id: str) -> int:
    m = re.search(r"(\d+)$", adapter_id)
    if m is None:
        raise ValueError(f"cannot parse author id from adapter id {adapter_id!r}")
    return int(m.group(1))


def load_author_names(n_authors: int) -> dict | None:
    """{author_id: name} via router.py's per-author extraction rule, from the local HF cache.

    Returns None (report falls back to ids) if the dataset cache/library is unavailable —
    name resolution is a nicety, not a dependency.
    """
    try:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        from collections import Counter

        from datasets import load_dataset
        ds = load_dataset("locuslab/TOFU", "full", split="train")
        names = {}
        for aid in range(n_authors):
            qs = [ds[i]["question"] for i in range(aid * 20, aid * 20 + 20)]
            thr = max(1, len(qs) // 2)
            counts = Counter()
            for q in qs:
                for m in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,3})\b", q):
                    counts[m.group(1)] += 1
            cand = [p for p, c in counts.items() if c >= thr]
            names[aid] = max(cand, key=len) if cand else f"author_{aid}"
        return names
    except Exception as e:  # noqa: BLE001 — any cache/env failure degrades to ids
        print(f"[author_similarity_report] author names unavailable ({e}); using ids")
        return None


def top_pairs(cos: np.ndarray, k: int) -> tuple:
    """((i, j, cos) most-similar desc, (i, j, cos) least-similar asc) over i<j."""
    iu, ju = np.triu_indices(cos.shape[0], k=1)
    vals = cos[iu, ju]
    order = np.argsort(vals)
    most = [(int(iu[t]), int(ju[t]), float(vals[t])) for t in order[::-1][:k]]
    least = [(int(iu[t]), int(ju[t]), float(vals[t])) for t in order[:k]]
    return most, least


def row_mean_offdiag(cos: np.ndarray) -> np.ndarray:
    n = cos.shape[0]
    return (cos.sum(axis=1) - np.diag(cos)) / (n - 1)


def render_heatmap(cos: np.ndarray, path: str, scale: int = 4) -> float:
    """Diverging white/blue(+)/red(-) PNG; returns the vmax the scale saturates at."""
    from PIL import Image
    off = cos[~np.eye(cos.shape[0], dtype=bool)]
    vmax = max(float(np.percentile(np.abs(off), 99.5)), 1e-12)
    v = np.clip(cos / vmax, -1.0, 1.0)
    rgb = np.empty((*cos.shape, 3), dtype=np.uint8)
    pos, neg = np.clip(v, 0, 1)[..., None], np.clip(-v, 0, 1)[..., None]
    white, blue, red = np.array([255.0] * 3), np.array([8.0, 69.0, 148.0]), np.array([165.0, 15.0, 21.0])
    rgb[:] = np.where(v[..., None] >= 0,
                      white + (blue - white) * pos,
                      white + (red - white) * neg).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB").resize((cos.shape[1] * scale, cos.shape[0] * scale),
                                             Image.NEAREST)
    img.save(path)
    return vmax


def name_token_effect(cos: np.ndarray, aids: list, names: dict, seed: int = 42) -> dict | None:
    """Do author pairs sharing a name token have higher delta cosine?

    Motivated by the k200 top-pairs table (Yeon Soo~Yeon Park, Fatima Al~Fatimah Al, the
    Alejandros/Rafaels): compares mean cosine of share-a-token pairs vs the rest, with a
    label-permutation p-value (permute authors' token sets, seed-fixed).
    """
    if names is None:
        return None
    toks = {a: set(names[a].split()) if names.get(a) and not names[a].startswith("author_")
            else set() for a in aids}
    n = len(aids)
    iu, ju = np.triu_indices(n, k=1)
    share = np.array([bool(toks[aids[i]] & toks[aids[j]]) for i, j in zip(iu, ju)])
    if share.sum() < 5:
        return None
    vals = cos[iu, ju]
    obs = vals[share].mean() - vals[~share].mean()
    rng = np.random.RandomState(seed)
    null = []
    for _ in range(2000):
        perm = rng.permutation(n)
        ptoks = {aids[t]: toks[aids[perm[t]]] for t in range(n)}
        pshare = np.array([bool(ptoks[aids[i]] & ptoks[aids[j]]) for i, j in zip(iu, ju)])
        null.append(vals[pshare].mean() - vals[~pshare].mean() if pshare.any() else 0.0)
    p = (np.sum(np.asarray(null) >= obs) + 1) / (len(null) + 1)
    return {"n_share": int(share.sum()), "n_pairs": int(len(vals)),
            "mean_share": float(vals[share].mean()), "mean_noshare": float(vals[~share].mean()),
            "diff": float(obs), "perm_p": float(p)}


def cross_run_rows(label: str, res: dict) -> dict:
    cfg = res.get("ref_config") or {}
    d0 = (res.get("adapter_dirs") or [""])[0]
    # collection = the dir right under checkpoints/ (legonet pools nest adapters deeper)
    parts = d0.rstrip("/").split(os.sep)
    model = (parts[parts.index("checkpoints") + 1] if "checkpoints" in parts
             else os.path.basename(os.path.dirname(d0.rstrip("/"))) if d0 else "?")
    e = res["shared_subspace_energy"]
    return {
        "label": label, "n": res["n_adapters"], "collection": model,
        "r": cfg.get("r"), "rslora": cfg.get("use_rslora"),
        "n_mod": len(cfg.get("target_modules") or []),
        "cos_mean": res["real"]["cosine_offdiag_mean"],
        "cos_max": res["real"]["cosine_offdiag_max"],
        "z_cos": res["z_vs_orthogonal_null"]["cosine_offdiag_mean"],
        "angB": res["real"]["angle_cos_B_offdiag_mean"],
        "angB_null": res["nulls"]["orthogonal"]["angle_cos_B"]["mean"],
        "angA": res["real"]["angle_cos_A_offdiag_mean"],
        "energy": e["mean_energy_retained"], "energy_rank": e["rank"],
        "chance": e["avg_rank_ratio"],
    }


def _fmt_pair(i: int, j: int, c: float, names: dict | None) -> str:
    def nm(a):
        return f"{a} ({names[a]})" if names and a in names else str(a)
    return f"| {nm(i)} | {nm(j)} | {c:.4f} |"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="subspace_overlap result JSON (the main run)")
    ap.add_argument("--priors", nargs="*", default=[], help="label=path of earlier runs to tabulate")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--heatmap", default=None, help="output PNG path (skip if omitted)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    res = json.load(open(args.json))
    ids = res["adapter_ids"]
    aids = [_author_id(x) for x in ids]
    order = np.argsort(aids)                       # defensive: numeric author order
    cos = np.asarray(res["cosine_matrix"], dtype=np.float64)[np.ix_(order, order)]
    aids = [aids[t] for t in order]
    assert np.allclose(cos, cos.T, atol=1e-8) and np.allclose(np.diag(cos), 1.0, atol=1e-6)
    assert not np.isnan(cos).any(), "NaNs in cosine matrix"

    names = load_author_names(max(aids) + 1)
    most, least = top_pairs(cos, args.top)
    rm = row_mean_offdiag(cos)
    rank = np.argsort(rm)

    vmax = None
    if args.heatmap:
        vmax = render_heatmap(cos, args.heatmap)

    runs = [cross_run_rows(lbl, json.load(open(p)))
            for lbl, p in (s.split("=", 1) for s in args.priors)]
    runs.append(cross_run_rows(os.path.splitext(os.path.basename(args.json))[0], res))
    runs.sort(key=lambda r: r["n"])

    off = cos[~np.eye(cos.shape[0], dtype=bool)]
    L = [f"# Per-author adapter similarity — {os.path.basename(args.json)}",
         "",
         f"Source: `{args.json}` · n={res['n_adapters']} adapters · seed {res['seed']} · "
         f"n_null={res['n_null']} · shared-subspace rank {res['rank']}",
         "",
         "## Headline numbers",
         "",
         f"- off-diag cosine: mean **{res['real']['cosine_offdiag_mean']:.4f}**, "
         f"median {res['real']['cosine_offdiag_median']:.4f}, |max| {res['real']['cosine_offdiag_max']:.4f} "
         f"(orthogonal-null mean {res['nulls']['orthogonal']['cosine']['mean']:.2e}, "
         f"z = {res['z_vs_orthogonal_null']['cosine_offdiag_mean']:.0f})",
         f"- principal-angle cos: col(B) **{res['real']['angle_cos_B_offdiag_mean']:.4f}** vs null "
         f"{res['nulls']['orthogonal']['angle_cos_B']['mean']:.4f}; row(A) "
         f"{res['real']['angle_cos_A_offdiag_mean']:.4f} vs null "
         f"{res['nulls']['orthogonal']['angle_cos_A']['mean']:.4f}",
         f"- shared-subspace energy @r{res['rank']}: "
         f"**{res['shared_subspace_energy']['mean_energy_retained']:.4f}** "
         f"(chance ~{res['shared_subspace_energy']['avg_rank_ratio']:.4f})",
         f"- off-diag cosine spread: sd {off.std():.4f}, p1 {np.percentile(off, 1):.4f}, "
         f"p99 {np.percentile(off, 99):.4f}",
         ""]

    L += [f"## Top-{args.top} most similar author pairs", "",
          "| author i | author j | cosine |", "|---|---|---|"]
    L += [_fmt_pair(aids[i], aids[j], c, names) for i, j, c in most]
    L += ["", f"## Top-{args.top} least similar author pairs", "",
          "| author i | author j | cosine |", "|---|---|---|"]
    L += [_fmt_pair(aids[i], aids[j], c, names) for i, j, c in least]

    def nm(a):
        return f"{a} ({names[a]})" if names and a in names else str(a)
    L += ["", "## Per-author mean similarity to the rest (row-mean off-diag cosine)", "",
          "Most generic deltas (highest mean):", ""]
    L += [f"- {nm(aids[t])}: {rm[t]:.4f}" for t in rank[::-1][:10]]
    L += ["", "Most distinctive deltas (lowest mean):", ""]
    L += [f"- {nm(aids[t])}: {rm[t]:.4f}" for t in rank[:10]]

    nt = name_token_effect(cos, aids, names)
    if nt:
        L += ["", "## Name-token overlap effect", "",
              f"Pairs sharing ≥1 author-name token (n={nt['n_share']}/{nt['n_pairs']}): "
              f"mean cosine **{nt['mean_share']:.4f}** vs {nt['mean_noshare']:.4f} for the rest "
              f"(diff {nt['diff']:+.4f}, permutation p={nt['perm_p']:.4g}, 2000 draws, seed 42)."]

    if args.heatmap:
        L += ["", "## Heatmap", "",
              f"![cosine heatmap]({os.path.basename(args.heatmap)})", "",
              f"Diverging scale saturating at |cos| = {vmax:.4f} (99.5th pct of off-diag); "
              f"the unit diagonal is clipped to full saturation."]

    L += ["", "## Cross-run trend (⚠ different base models / recipes — direction only)", "",
          "Earlier runs are Llama-3.2-1B collections (k4/k10 legacy r8 shards; n32 LegoNet r16, "
          "non-rslora, attention-only); the k200 run is Llama-2-7B r32/α64 rslora, 6 modules. "
          "Comparable in *direction of trend with n*, not as a controlled dial.", "",
          "| run | n | collection | r | rslora | mods | cos mean | cos max | z(cos) | "
          "angB (null) | angA | energy@r | chance |", "|---|" + "---|" * 12]
    for r in runs:
        L.append(f"| {r['label']} | {r['n']} | {r['collection']} | {r['r']} | {r['rslora']} | "
                 f"{r['n_mod']} | {r['cos_mean']:.4f} | {r['cos_max']:.4f} | {r['z_cos']:.0f} | "
                 f"{r['angB']:.3f} ({r['angB_null']:.3f}) | {r['angA']:.3f} | "
                 f"{r['energy']:.3f}@r{r['energy_rank']} | {r['chance']:.4f} |")

    L += ["", "## Per-module-type cosine (off-diag mean)", ""]
    L += [f"- {mt}: {v:.4f}" for mt, v in res["cosine_by_modtype"].items()]
    L += ["", "## Per-layer cosine (off-diag mean)", ""]
    L += [f"- layer {k}: {v:.4f}" for k, v in res["cosine_by_layer"].items()]

    with open(args.out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[author_similarity_report] wrote {args.out_md}"
          + (f" and {args.heatmap}" if args.heatmap else ""))


if __name__ == "__main__":
    main()
