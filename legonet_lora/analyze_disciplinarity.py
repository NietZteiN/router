"""Cluster disciplinarity analysis: characterize each expert/cluster by its
field (DBpedia-class) composition, and slice per-record metrics by it.

"field" = the record's DBpedia class (label_name). For each cluster we take the
distribution of its members' fields and bucket it:
  * single-discipline      : dominant field >= --single_thresh (default 0.90)
  * highly-interdisciplinary: NO field exceeds --inter_thresh (default 0.10)
  * interdisciplinary       : everything in between
Then we slice the per-record eval metrics (retained EM / perplexity / canary_em)
by the disciplinarity of each record's PRIMARY cluster (its nearest key), to see
whether interdisciplinary clusters behave differently.

    python analyze_disciplinarity.py --config configs/legonet_7b_v2.json
"""
import argparse
import json
import os
from collections import Counter

from legonet_common import Paths, load_config, load_records


def cluster_taxonomy(cfg, single_thresh, inter_thresh):
    paths = Paths(cfg)
    assignment = json.load(open(paths.assignment_path))
    by_id = {r["id"]: r["label_name"] for r in load_records(paths.records_path)}
    rows = []
    for j in range(cfg["n"]):
        members = assignment["members"][str(j)]
        if not members:
            rows.append({"cluster": j, "size": 0, "bucket": "empty",
                         "dominant_field": None, "dominant_frac": 0.0, "n_fields": 0})
            continue
        dist = Counter(by_id[m] for m in members)
        top_field, top_n = dist.most_common(1)[0]
        frac = top_n / len(members)
        if frac >= single_thresh:
            bucket = "single-discipline"
        elif max(c / len(members) for c in dist.values()) < inter_thresh:
            bucket = "highly-interdisciplinary"
        else:
            bucket = "interdisciplinary"
        # graded view (gives balanced buckets for the metric slice, since the strict
        # 90/10 rule leaves almost everything "interdisciplinary")
        graded = "pure" if frac >= 0.75 else ("mixed" if frac >= 0.50 else "highly-mixed")
        rows.append({"cluster": j, "size": len(members), "bucket": bucket, "graded": graded,
                     "dominant_field": top_field, "dominant_frac": round(frac, 3),
                     "n_fields": len(dist)})
    return assignment, rows


def slice_metrics(cfg, assignment, cluster_bucket, rows_file=None):
    """Mean per-record metrics grouped by the record's PRIMARY cluster bucket."""
    p = Paths(cfg)
    ev = rows_file or os.path.join(p.results_dir, "eval_legonet.json")
    if not os.path.exists(ev):
        return None
    rows = json.load(open(ev)).get("rows", [])
    groups = {}
    for r in rows:
        prim = assignment["record_to_keys"][r["id"]][0]  # nearest key
        b = cluster_bucket.get(prim, "unknown")
        groups.setdefault(b, []).append(r)
    out = {}
    for b, rs in groups.items():
        def mean(k):
            vals = [x[k] for x in rs if x.get(k) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        out[b] = {"n_records": len(rs), "em": mean("em"),
                  "perplexity": mean("perplexity"), "canary_em": mean("canary_em"),
                  "verbmem": mean("verbmem")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/legonet_7b_v2.json")
    ap.add_argument("--single_thresh", type=float, default=0.90)
    ap.add_argument("--inter_thresh", type=float, default=0.10)
    ap.add_argument("--rows_file", default=None, help="per-record eval json (default eval_legonet.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    assignment, rows = cluster_taxonomy(cfg, args.single_thresh, args.inter_thresh)
    counts = Counter(r["bucket"] for r in rows)
    graded_counts = Counter(r["graded"] for r in rows)
    cluster_bucket = {r["cluster"]: r["bucket"] for r in rows}
    graded_bucket = {r["cluster"]: r["graded"] for r in rows}
    metrics = slice_metrics(cfg, assignment, cluster_bucket, args.rows_file)
    metrics_graded = slice_metrics(cfg, assignment, graded_bucket, args.rows_file)

    # dominant-fraction histogram (so thresholds are visible)
    fr = sorted(r["dominant_frac"] for r in rows if r["size"] > 0)
    hist = {f"{lo:.1f}-{lo+0.1:.1f}": sum(1 for x in fr if lo <= x < lo + 0.1)
            for lo in [i / 10 for i in range(0, 10)]}
    hist["1.0"] = sum(1 for x in fr if x >= 1.0)

    report = {
        "config": cfg["name"], "n": cfg["n"], "k": cfg["k"],
        "thresholds": {"single>=": args.single_thresh, "highly_inter_max<": args.inter_thresh},
        "bucket_counts": dict(counts),
        "graded_counts": dict(graded_counts),
        "dominant_frac_hist": hist,
        "clusters": rows,
        "metrics_by_bucket": metrics,
        "metrics_by_graded": metrics_graded,
    }

    print(f"\n=== Cluster disciplinarity — {cfg['name']} (n={cfg['n']}, k={cfg['k']}) ===")
    print("bucket counts:", dict(counts))
    print("dominant-field fraction histogram:", {k: v for k, v in hist.items() if v})
    print("\nper-cluster (field, dominant%, #fields, bucket):")
    for r in rows:
        print(f"  c{r['cluster']:>2} size={r['size']:>3} {str(r['dominant_field']):>22} "
              f"{r['dominant_frac']:.2f} nfields={r['n_fields']:>2} -> {r['bucket']}")
    print("graded counts (pure>=0.75 / mixed / highly-mixed<0.5):", dict(graded_counts))
    if metrics:
        print("\nmetrics by PRIMARY-cluster disciplinarity (strict 90/10):")
        for b, m in sorted(metrics.items()):
            print(f"  {b:>26}: {m}")
        print("\nmetrics by PRIMARY-cluster disciplinarity (graded):")
        for b, m in sorted((metrics_graded or {}).items()):
            print(f"  {b:>26}: {m}")
    else:
        print("\n(no eval rows file yet — metric slice skipped)")

    out = args.out or os.path.join(Paths(cfg).results_dir, "disciplinarity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
