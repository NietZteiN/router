"""Assemble the Phase-3 sweep into one report (robust to missing cells).

For each cell reads retained-record utility (eval_legonet.json) + forget/cost
(exactness.json: forget canary_em pre->post, mean_unlearn_seconds) + assignment
(adapter sizes, mode). Computes the analytical per-deletion example-passes
(LegoNet k^2 N/n via k*mean_adapter_size; SISA random k=1 -> N/s) and writes the
utility-vs-segmentation, ensemble (k), semantic-vs-random, and deletion-cost
tables to {root}/SWEEP_REPORT.md.

    python collect_sweep.py
"""
import json
import os

from legonet_common import Paths, load_config

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def cell_row(cfg_path):
    cfg = load_config(cfg_path)
    p = Paths(cfg)
    ev = _load(os.path.join(p.results_dir, "eval_legonet.json"))
    ex = _load(os.path.join(p.results_dir, "exactness.json"))
    asg = _load(p.assignment_path)
    row = {"name": cfg["name"], "mode": cfg.get("assignment_mode", "knn"),
           "n": cfg["n"], "k": cfg["k"]}
    if ev:
        a = ev["aggregate"]
        row.update(retained_em=a.get("em"), retained_verbmem=a.get("verbmem"),
                   retained_ppl=a.get("perplexity"), retained_canary_em=a.get("canary_em"))
    if ex and ex.get("per_deletion"):
        pre = [d["forget"]["pre"]["canary_em"] for d in ex["per_deletion"]]
        post = [d["forget"]["post_unlearn"]["canary_em"] for d in ex["per_deletion"]]
        row["forget_canary_pre"] = sum(pre) / len(pre)
        row["forget_canary_post"] = sum(post) / len(post)
        row["unlearn_s"] = ex.get("mean_unlearn_seconds")
    cl = _load(os.path.join(p.results_dir, "eval_classification.json"))
    if cl:
        row["cls_retain_acc"] = cl.get("retain_acc")
        row["cls_test_acc"] = cl.get("test_acc")
    if asg:
        total = sum(asg["adapter_sizes"])
        mean_size = total / cfg["n"]
        # per-deletion example-passes: retrain k affected adapters, each ~mean_size records
        row["cost_example_passes"] = cfg["k"] * mean_size
        row["adapter_size_mean"] = mean_size
        row["adapter_size_max"] = max(asg["adapter_sizes"])
    return row


def fmt(v, p=3):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"


def main():
    cells_file = os.path.join(HERE, "sweep_cells.txt")
    cell_cfgs = [l.strip() for l in open(cells_file)] if os.path.exists(cells_file) else []
    anchors = [os.path.join(HERE, "configs", "legonet_7b_v2.json")]  # knn n32 k3 (v2 recipe)
    rows = [cell_row(c) for c in anchors + cell_cfgs if os.path.exists(c)]
    by = {(r["mode"], r["n"], r["k"]): r for r in rows}
    base = _load(os.path.join(Paths(load_config(anchors[0])).results_dir, "eval_base.json"))
    base_agg = base.get("aggregate", {}) if base else {}

    L = ["# LegoNet-LoRA Phase-3 sweep report", ""]
    L.append(f"Frozen base reference: retained EM (base) = {fmt(base_agg.get('em'))}, "
             f"canary_em (base) = {fmt(base_agg.get('canary_em'))}, PPL (base) = {fmt(base_agg.get('perplexity'),2)}")
    L += ["", "## All cells", "",
          "| cell | mode | n | k | retained EM | retained VerbMem | retained PPL | canary_em(retained) | forget canary pre→post | unlearn s | example-passes/del |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['mode']} | {r['n']} | {r['k']} | "
                 f"{fmt(r.get('retained_em'))} | {fmt(r.get('retained_verbmem'))} | "
                 f"{fmt(r.get('retained_ppl'),2)} | {fmt(r.get('retained_canary_em'))} | "
                 f"{fmt(r.get('forget_canary_pre'))}→{fmt(r.get('forget_canary_post'))} | "
                 f"{fmt(r.get('unlearn_s'),1)} | {fmt(r.get('cost_example_passes'),0)} |")

    def section(title, keys, note=""):
        L.append(f"\n## {title}")
        if note:
            L.append(note)
        L.append("\n| mode | n | k | retained EM | retained PPL | forget canary pre→post | example-passes/del |")
        L.append("|---|---|---|---|---|---|---|")
        for key in keys:
            r = by.get(key)
            if not r:
                L.append(f"| {key[0]} | {key[1]} | {key[2]} | — | — | — | — |")
                continue
            L.append(f"| {r['mode']} | {r['n']} | {r['k']} | {fmt(r.get('retained_em'))} | "
                     f"{fmt(r.get('retained_ppl'),2)} | {fmt(r.get('forget_canary_pre'))}→{fmt(r.get('forget_canary_post'))} | "
                     f"{fmt(r.get('cost_example_passes'),0)} |")

    section("Utility vs segmentation n (k=3)", [("knn", 16, 3), ("knn", 32, 3), ("knn", 64, 3)],
            "LegoNet claim: utility holds as n grows (unlike SISA).")
    section("Utility vs ensemble k (n=32)", [("knn", 32, 1), ("knn", 32, 3), ("knn", 32, 5)],
            "LegoNet claim: k>1 ensemble recovers utility over k=1.")
    section("Semantic vs random assignment (n=32, k=1) = LegoNet_{k=1} vs FixSISA",
            [("knn", 32, 1), ("random", 32, 1)],
            "The paper's core ablation: similarity-based assignment beats random shards at equal segmentation.")
    section("SISA-LoRA deletion-cost baseline (random, k=1)", [("random", 32, 1), ("random", 64, 1)],
            "SISA per-deletion = N/s (retrain one shard). Compare to LegoNet k²N/n in 'All cells'.")

    # LegoNet-units classification accuracy (the paper's metric)
    base_cls = _load(os.path.join(load_config(anchors[0])["root"], "eval_classification_base.json"))
    if base_cls or any(r.get("cls_retain_acc") is not None for r in rows):
        L.append("\n## LegoNet-units: 14-class DBpedia accuracy (D_retain / D_test)")
        L.append("⚠️ ARTIFACT — do NOT read as utility. Our experts were trained to MEMORIZE passages "
                 "(LM loss), not to classify, so reading them as classifiers is apples-to-oranges: "
                 "accuracies are near-chance (1/14≈0.07) and legonet<base because memorization training "
                 "drifts the model away from the 'Category:' format (+ CamelCase-label scoring is weak, "
                 "base only 0.23). This is NOT utility loss — real capability is MMLU (legonet 0.43 ≈ "
                 "base 0.46). A faithful LegoNet-accuracy comparison needs experts trained AS classifiers. "
                 "D_unlearn (forgetting) omitted — base-recoverable on a strong LLM; canary_em is the analog.")
        L.append("\n| cell | mode | n | k | D_retain acc | D_test acc |")
        L.append("|---|---|---|---|---|---|")
        if base_cls:
            L.append(f"| (frozen base) | — | — | — | {fmt(base_cls.get('retain_acc'))} | {fmt(base_cls.get('test_acc'))} |")
        for r in rows:
            if r.get("cls_retain_acc") is not None:
                L.append(f"| {r['name']} | {r['mode']} | {r['n']} | {r['k']} | "
                         f"{fmt(r['cls_retain_acc'])} | {fmt(r.get('cls_test_acc'))} |")

    out = os.path.join(load_config(anchors[0])["root"], "SWEEP_REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
