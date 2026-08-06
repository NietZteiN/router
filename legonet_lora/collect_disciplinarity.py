"""Assemble the cluster-disciplinarity analysis into a report section.

Reads each run's disciplinarity_n1000.json (written by analyze_disciplinarity)
and writes {root}/DISCIPLINARITY_REPORT.md: the field-composition taxonomy
(shared across models — same routing) + per-model metric slices by cluster
disciplinarity (strict 90/10 and graded buckets).

    python collect_disciplinarity.py
"""
import json
import os

from legonet_common import Paths, load_config

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [("Llama-2-7B-v2", "configs/legonet_7b_v2.json"),
           ("Llama-3.2-3B", "configs/legonet_l32_3b.json")]
ROWS = "disciplinarity_n1000.json"


def _load(cfg_path):
    p = os.path.join(Paths(load_config(cfg_path)).results_dir, ROWS)
    try:
        return json.load(open(p))
    except Exception:
        return None


def fmt(v, p=3):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"


def main():
    data = [(name, _load(p)) for name, p in CONFIGS]
    data = [(n, d) for n, d in data if d]
    if not data:
        print("no disciplinarity_n1000.json found yet")
        return
    ref = data[0][1]  # taxonomy is model-independent

    L = ["# Cluster disciplinarity analysis", "",
         "**Field := the record's DBpedia-14 class** (Company, Artist, Athlete, Animal, Film, …; "
         "'physics' in the request = a generic example of a field). Each of the n clusters is "
         "characterized by its members' field mix.",
         "",
         "- **single-discipline**: one field ≥ 90% of the cluster",
         "- **highly-interdisciplinary**: no field exceeds 10% (very spread)",
         "- **interdisciplinary**: in between",
         "- *graded view* (balanced for the metric slice): **pure** ≥0.75 / **mixed** 0.50–0.75 / **highly-mixed** <0.50",
         "",
         f"n={ref['n']}, k={ref['k']}. Taxonomy is identical across models (same frozen keys/routing).", ""]

    L += ["## Cluster taxonomy (counts)", "",
          f"- strict (90/10): **{ref['bucket_counts']}**",
          f"- graded: **{ref['graded_counts']}**",
          f"- dominant-field-fraction histogram: {{ {', '.join(f'{k}:{v}' for k,v in ref['dominant_frac_hist'].items() if v)} }}",
          "", "## Per-cluster field composition", "",
          "| cluster | size | dominant field | dominant % | #fields | strict | graded |",
          "|---|---|---|---|---|---|---|"]
    for c in ref["clusters"]:
        L.append(f"| c{c['cluster']} | {c['size']} | {c['dominant_field']} | {fmt(c['dominant_frac'],2)} | "
                 f"{c['n_fields']} | {c['bucket']} | {c['graded']} |")

    for name, d in data:
        L.append(f"\n## Metrics by cluster disciplinarity — {name}")
        for label, key in [("strict (single / inter / highly-inter)", "metrics_by_bucket"),
                           ("graded (pure / mixed / highly-mixed)", "metrics_by_graded")]:
            m = d.get(key) or {}
            L.append(f"\n**{label}** (per-record means, grouped by the record's primary cluster):")
            L.append("\n| bucket | #records | retained EM | perplexity | canary_em | verbmem |")
            L.append("|---|---|---|---|---|---|")
            for b, s in sorted(m.items()):
                L.append(f"| {b} | {s['n_records']} | {fmt(s['em'])} | {fmt(s['perplexity'],2)} | "
                         f"{fmt(s['canary_em'])} | {fmt(s['verbmem'])} |")

    out = os.path.join(load_config(CONFIGS[0][1])["root"], "DISCIPLINARITY_REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
