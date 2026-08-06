#!/usr/bin/env python3
"""Regenerate the master report's tables from the result JSONs.

verify_report.py answers "does the report match the data". This answers the stronger question:
"if you had only the data, would you write the same table?" It emits markdown built solely from
reproduce/results_snapshot/, in the report's own column order, so you can diff it against
reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md.

Values are printed at the report's precision, so a clean diff is the expected outcome. Where the
report's own row is spliced across files or tiers (Table F ClAMU K=16, Table F merge_full -- see
CAVEATS.md), this script prints what the DATA says and flags the divergence, rather than
reproducing the splice.

    python reproduce/rebuild_tables.py            # every rebuildable table
    python reproduce/rebuild_tables.py --table D
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, "results_snapshot")

P1 = "Llama-3.2-1B-Instruct/results/smoke"
NM = "Llama-2-7B-chat-hf_nmerge_r32/results/smoke"
NC = "Llama-2-7B-chat-hf_nmerge_r32_centered/results/smoke"
SIFT = "Llama-3.2-1B-Instruct_sift_masks/results/smoke"
CLAMU = "Llama-3.2-1B-Instruct_clamu/results/smoke"


def load(rel: str) -> dict | None:
    path = os.path.join(SNAPSHOT, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value, decimals=3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if abs(value) >= 1000:
        return f"{value:.3g}"
    return f"{value:.{decimals}f}"


def cell(rel: str, field: str, decimals=3) -> str:
    data = load(rel)
    return "—" if data is None else fmt(data.get(field), decimals)


def table_b() -> None:
    rows = [
        ("Naive sum λ=1 (`additive_s1`)", "merged_additive_s1"),
        ("Uniform mean λ=1/k (`additive_mean`)", "merged_additive_mean"),
        ("Tuned-λ sum, λ=0.05 (`additive_s0.05`)", "merged_additive_s0.05"),
        ("DARE-TIES (frozen default)", "merged_dare_ties"),
        ("DELLA-TIES", "merged_della_ties"),
        ("Fisher-weighted", "merged_fisher"),
        ("KnOTS (shared-SVD + TIES)", "merged_knots_ties"),
        ("Breadcrumbs λ=1/(n√r)", "merged_breadcrumbs_s0.0354"),
        ("Breadcrumbs λ=1/n", "merged_breadcrumbs_s0.1"),
        ("PEFT linear (√r-inflated)", "merged_linear"),
        ("TSV-M (whitened top-singular)", "merged_tsv"),
        ("SLERP (tree, pairwise)", "tree_root_slerp"),
        ("Subtract-orth *(unlearn op)*", "subtract_orth"),
        ("Task-arith subtraction (`subtract_linear`)", "subtract_linear"),
        ("*Routing key-exact (reference)*", "routed_key_exact"),
        ("*Routing + scaffold, OOD-aware (reference)*", "routed_scaffold_ood"),
    ]
    print("## Table B — P1 widest battery (Llama-3.2-1B, k=10)\n")
    print(f"Pool `{P1.split('/')[0]}` · base "
          f"{cell(f'{P1}/base_model.json', 'model_utility')} / FT "
          f"{cell(f'{P1}/ft_all.json', 'model_utility')}\n")
    print("| Method | mu | fq | f_rouge | r_ppl |")
    print("|---|---:|---:|---:|---:|")
    for label, name in rows:
        rel = f"{P1}/{name}.json"
        routed = name.startswith("routed_")
        fq = "—" if routed else cell(rel, "forget_quality")
        print(f"| {label} | {cell(rel, 'model_utility')} | {fq} | "
              f"{cell(rel, 'forget_rouge')} | {cell(rel, 'retain_ppl', 1)} |")


def table_c() -> None:
    ks = [("4", "k4_r32_e5_lr1e4"), ("10", "k10_r32_e5_lr1e4"), ("20", "k20_r32_e5_lr1e4"),
          ("50", "k50_r32_e5_lr1e4"), ("100", "k100_r32_e5_lr1e4"), ("200 (r8)", "k200_r8_e5_lr1e4")]
    print("\n## Table C — The dilution law: DARE-TIES vs shard count (Llama-2-7B, smoke)\n")
    print("| k (shards) | " + " | ".join(k for k, _ in ks) + " |")
    print("|---|" + "---:|" * len(ks))
    for label, name in (("`merged_dare_ties` mu", "merged_dare_ties"),
                        ("`routed_key_exact` mu", "routed_key_exact")):
        cells = [cell(f"Llama-2-7B-chat-hf_{d}/results/smoke/{name}.json", "model_utility")
                 for _, d in ks]
        print(f"| {label} | " + " | ".join(cells) + " |")


def table_d() -> None:
    ns = [2, 4, 8, 16, 32, 64, 128, 200]
    print("\n## Table D — Per-author N-merge ladder (P3: Llama-2-7B, k=200 r32, true-mean)\n")
    print(f"Base {cell(f'{NM}/base_model__own82.json', 'model_utility')} / "
          f"FT {cell(f'{NM}/ft_r32__own82.json', 'model_utility')} · headline probe = author 82\n")
    print("| N merged | " + " | ".join(str(n) for n in ns) + " |")
    print("|---|" + "---:|" * len(ns))
    for label, pool, stem in (("`additive_mean` mu", NM, "nmerge_add"),
                              ("centered `cr16` mu", NC, "nmerge_cr16")):
        cells = []
        for n in ns:
            # N above the exact cap is materialized svd-compressed; the rank is in the filename
            plain, svd = f"{pool}/{stem}_N{n}_s42__own82.json", f"{pool}/{stem}_svd1024_N{n}_s42__own82.json"
            cells.append(cell(plain if load(plain) else svd, "model_utility"))
        print(f"| {label} | " + " | ".join(cells) + " |")


def table_e() -> None:
    def peft(method: str, name: str) -> str:
        return f"Llama-3.2-1B-Instruct_peft_{method}_k10/results/smoke/{name}.json"

    rows = [
        ("LoRA (additive mean)", f"{P1}/merged_additive_mean.json", None, None),
        ("DoRA (additive mean)", peft("dora", "merged_additive_mean"),
         peft("dora", "routed_key_exact"), peft("dora", "dora_iso_s9")),
        ("IA³ (gate arith-mean)", peft("ia3", "ia3_composed_full"),
         peft("ia3", "routed_key_exact"), peft("ia3", "ia3_iso_s9")),
        ("IA³ (gate geo-mean)", peft("ia3", "ia3_geo_full"), None, peft("ia3", "ia3_iso_s9")),
        ("VeRA (shared frozen basis)", peft("vera", "vera_composed_full"),
         peft("vera", "routed_key_exact"), peft("vera", "vera_iso_s9")),
        ("Prefix-tuning (KV concat)", peft("prefix", "prefixcat_full"),
         peft("prefix", "routed_key_exact"), peft("prefix", "prefix_iso_s9")),
    ]
    print("\n## Table E — PEFT parameterization bake-off (P4: Llama-3.2-1B, k=10)\n")
    print("| Method (compose rule) | composed mu | routed mu | iso mu (s9) | comp f_rouge |")
    print("|---|---:|---:|---:|---:|")
    for label, comp, routed, iso in rows:
        print(f"| {label} | {cell(comp, 'model_utility')} | "
              f"{cell(routed, 'model_utility') if routed else '—'} | "
              f"{cell(iso, 'model_utility') if iso else '—'} | {cell(comp, 'forget_rouge')} |")


def table_f() -> None:
    rows = [
        ("**SIFT-Masks** `sift_full` (sum + inference-time mask)", f"{SIFT}/sift_full.json"),
        ("**FT+Merge** `merge_full` (same sum, **no mask**)", f"{SIFT}/merge_full.json"),
        ("SIFT-Masks `sift_unlearn` (subtract 20 τ)", f"{SIFT}/sift_unlearn.json"),
        ("ClAMU — Global (no mask)", f"{CLAMU}/merge_full.json"),
        ("ClAMU — EMR mask", f"{CLAMU}/emr_full.json"),
        ("ClAMU — TALL mask", f"{CLAMU}/tall_full.json"),
        ("ClAMU — optimized mask (K=16)",
         "Llama-3.2-1B-Instruct_clamu_K16/results/smoke/clamu_full.json"),
        ("ClAMU — optimized mask (K=200 peak)",
         "Llama-3.2-1B-Instruct_clamu_K200/results/smoke/clamu_full.json"),
    ]
    print("\n## Table F — Full-parameter task vectors (P5: Llama-3.2-1B, T=200 per-author full-FT)\n")
    print("| Method / condition | mu | fq | f_rouge |")
    print("|---|---:|---:|---:|")
    for label, rel in rows:
        print(f"| {label} | {cell(rel, 'model_utility')} | {cell(rel, 'forget_quality')} | "
              f"{cell(rel, 'forget_rouge')} |")
    print("\n> Two divergences from the report, both documented in CAVEATS.md:")
    print(">  - `merge_full` fq: the report prints 0.099, which is the **extended** tier; the smoke")
    print(f">    tier printed above is {cell(f'{SIFT}/merge_full.json', 'forget_quality')}, and the")
    print(">    row's mu and f_rouge are smoke. The report's row is tier-mixed.")
    print(">  - ClAMU K=16: the report prints mu 0.647, which is the `_clamu/` default dir, while its")
    print(">    fq and f_rouge come from `_clamu_K16/`. The `_clamu_K16/` mu printed above is")
    print(f">    {cell('Llama-3.2-1B-Instruct_clamu_K16/results/smoke/clamu_full.json', 'model_utility')}.")


def table_fprime() -> None:
    arms = [
        ("**ctrl** — plain per-author LoRA", "Llama-3.2-1B-Instruct_ctv_ctrl_r32_e25", "iso_a*__own*"),
        ("**[lin]** tangent-space (linearized)", "Llama-3.2-1B-Instruct_ctv_lin_r32_e25", "iso_a*__own*"),
        ("**[wd]** write-disjoint col(B)", "Llama-3.2-1B-Instruct_ctv_wd_r32_e25", "iso_a*__own*"),
        ("**[ds]** disjoint-support full-FT", "Llama-3.2-1B-Instruct_ctv_ds_e25", "iso_a[0-9]*__own*"),
    ]
    print("\n## Table F′ — ctv training-time constructions (solo N=1, before any merge)\n")
    print("| Arm | solo mu | own-prob | own-rouge | n rows |")
    print("|---|---:|---:|---:|---:|")
    for label, pool, pattern in arms:
        paths = sorted(glob.glob(os.path.join(SNAPSHOT, pool, "results/smoke", f"{pattern}.json")))
        probs, rouges = [], []
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("retain_prob") is not None:
                probs.append(data["retain_prob"])
            if data.get("retain_rouge") is not None:
                rouges.append(data["retain_rouge"])
        # solo mu: author 15 is the only probe whose model_utility is not NaN (see CAVEATS)
        solo = [json.load(open(p, encoding="utf-8")).get("model_utility")
                for p in sorted(glob.glob(os.path.join(SNAPSHOT, pool, "results/smoke",
                                                       "iso_a15*__own15.json")))]
        solo = [s for s in solo if isinstance(s, float) and not math.isnan(s)]
        solo_mu = sum(solo) / len(solo) if solo else None
        print(f"| {label} | {fmt(solo_mu)} | {fmt(sum(probs)/len(probs), 4) if probs else '—'} | "
              f"{fmt(sum(rouges)/len(rouges), 4) if rouges else '—'} | {len(paths)} |")
    floor = sorted(glob.glob(os.path.join(
        SNAPSHOT, "Llama-3.2-1B-Instruct_ctv_ctrl_r32_e25/results/smoke/base_model__own*.json")))
    fp = [json.load(open(p, encoding="utf-8"))["retain_prob"] for p in floor]
    print(f"\n> Base own-prob floor: {fmt(sum(fp)/len(fp), 4)} ({len(fp)} probe rows).")
    print("> solo mu is a SINGLE probe (author 15) -- every other probe's model_utility is NaN,")
    print("> because --retain_author_ids leaves the retain truth-ratio subset empty. The [wd] row")
    print("> averages its two write-disjoint variants (orthblock, rowslice).")


TABLES = {"B": table_b, "C": table_c, "D": table_d, "E": table_e, "F": table_f, "F'": table_fprime}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", choices=sorted(TABLES), help="only this table")
    args = ap.parse_args()

    if not os.path.isdir(SNAPSHOT):
        sys.exit(f"no snapshot at {SNAPSHOT} -- run reproduce/snapshot_results.py first")

    print("<!-- generated by reproduce/rebuild_tables.py from reproduce/results_snapshot/ -->")
    for name in ([args.table] if args.table else ["B", "C", "D", "E", "F", "F'"]):
        TABLES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
