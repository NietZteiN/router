#!/usr/bin/env python3
"""Regenerate reports/SMOKE_EVAL_REPORT.md and checkpoints/all_metrics_smoke.csv from JSON."""
import argparse
import glob
import json
import os
import sys
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from collect_results import classify_label
from merge_lora import DEFAULT_MERGE_METHODS, smoke_eval_labels

CKPT = os.path.join(ROOT, "checkpoints")
REPORT = os.path.join(ROOT, "reports", "SMOKE_EVAL_REPORT.md")
EXTENDED_REPORT = os.path.join(ROOT, "reports", "EXTENDED_EVAL_REPORT.md")

METRIC_COLS = [
    "forget_ppl", "retain_ppl", "forget_rouge", "retain_rouge",
    "real_rouge", "world_rouge", "forget_truth_ratio", "forget_quality", "model_utility",
]

TABLE_HEADERS = {
    "label": "Label",
    "model_slug": "Model",
    "forget_ppl": "forget_ppl ↑",
    "retain_ppl": "retain_ppl ↓",
    "forget_rouge": "forget_rouge ↓",
    "retain_rouge": "retain_rouge ↑",
    "real_rouge": "real_rouge ↑",
    "world_rouge": "world_rouge ↑",
    "forget_truth_ratio": "forget_truth_ratio →1",
    "forget_quality": "forget_quality ↑",
    "model_utility": "model_utility ↑",
}

GLOSSARY = """## 8. Metric glossary

Arrows in result tables show the **desired direction** for successful unlearning (forget metrics) or retention/utility.

| Metric | Interpretation (TOFU) |
|--------|------------------------|
| **forget_ppl** | Perplexity on forget-set Q/A. **Higher** = more forgetting. |
| **retain_ppl** | Perplexity on held-out retain authors. **Lower** = better retention. |
| **forget_rouge** | ROUGE-L on forget generations. **Lower** = more forgetting. |
| **retain_rouge** | ROUGE-L on retain set — **higher** is better retention. |
| **real_rouge / world_rouge** | General utility benchmarks — **higher** is better. |
| **forget_truth_ratio** | open-unlearning closer_to_1_better = mean(min(tr, 1/tr)) over the forget set, tr = wrong/correct. **→ 1** = forget & true equally likely = more forgetting. |
| **forget_quality** (alias `ks_pval`) | KS p-value of the forget truth-ratio distribution vs the **retain90 oracle**. **Higher** p = indistinguishable from a model that never saw forget = better unlearning. |
| **model_utility** | Harmonic mean of 9: {retain, real_authors, world_facts} × {prob, ROUGE-L recall, truth-ratio(true_better)}. Excludes forget metrics. |

**SISA-specific note:** `shard_{forget}_only` is trained **on** the forget authors, so it should score **best** on forget metrics (low PPL, high ROUGE). That row validates shard assignment; unlearn success is measured by **`remerge_*`** or **`subtract_*`** vs that baseline.
"""


def load_eval_rows(results_subdir="smoke", model_slug=None):
    rows = []
    pattern = os.path.join(CKPT, "*", "results", results_subdir, "*.json")
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".progress.json"):
            continue
        slug = path.split(os.sep)[-4]
        if model_slug and slug != model_slug:
            continue
        with open(path) as f:
            row = json.load(f)
        row["model_slug"] = slug
        mm, dens, um = classify_label(row.get("label", ""))
        row["merge_method"] = mm
        row["density"] = dens
        row["unlearn_method"] = um
        rows.append(row)
    return rows


def load_smoke_rows(model_slug=None):
    return load_eval_rows("smoke", model_slug)


def md_table(df, cols=None):
    if df.empty:
        return "_No data._"
    cols = cols or METRIC_COLS
    cols = ["label"] + [c for c in cols if c in df.columns and c != "label"]
    if "model_slug" in df.columns and "label" not in cols:
        cols = ["model_slug"] + cols
    sub = df[cols].copy()
    for c in sub.select_dtypes(include="float").columns:
        sub[c] = sub[c].map(lambda x: f"{x:.4f}" if abs(x) < 1000 else f"{x:.2f}")
    display_cols = [TABLE_HEADERS.get(c, c) for c in cols]
    header = "| " + " | ".join(display_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(sub.iloc[i][c]) for c in cols) + " |" for i in range(len(sub))]
    return "\n".join([header, sep] + body)


def label_sort_key(label):
    if label == "base_model":
        return (-1, label)
    if label.startswith("shard_") and label.endswith("_only"):
        return (0, int(label.split("_")[1]))
    if label.startswith("merged_"):
        return (1, label)
    if label.startswith("remerge_"):
        return (2, label)
    if label == "subtract_linear":
        return (3, label)
    return (4, label)


def executive_summary(df, forget_id=3, tier="smoke"):
    lines = []
    n = len(df)
    slugs = df["model_slug"].unique().tolist()
    lines.append(f"| Finding | Detail |")
    lines.append(f"|--------|--------|")
    lines.append(
        f"| **Evaluations** | **{n}** adapter {tier} runs across "
        f"{len(slugs)} model(s): {', '.join(slugs)} |"
    )

    baseline = df[df["label"] == f"shard_{forget_id}_only"]
    if not baseline.empty:
        b = baseline.iloc[0]
        lines.append(
            f"| **Forget shard baseline** | `shard_{forget_id}_only`: forget_ppl **{b['forget_ppl']:.2f}**, "
            f"forget_rouge **{b['forget_rouge']:.4f}** (memorization control) |"
        )

    merged = df[df["label"].str.startswith("merged_")]
    if not merged.empty:
        best_u = merged.loc[merged["model_utility"].idxmax()]
        lines.append(
            f"| **Best merged utility** | `{best_u['label']}`: model_utility **{best_u['model_utility']:.4f}** |"
        )

    remerge = df[df["label"].str.startswith("remerge_")]
    if not remerge.empty:
        best_fp = remerge.loc[remerge["forget_ppl"].idxmax()]
        best_fr = remerge.loc[remerge["forget_rouge"].idxmin()]
        lines.append(
            f"| **Best remerge (forget PPL)** | `{best_fp['label']}`: forget_ppl **{best_fp['forget_ppl']:.2f}** |"
        )
        lines.append(
            f"| **Best remerge (forget ROUGE)** | `{best_fr['label']}`: forget_rouge **{best_fr['forget_rouge']:.4f}** |"
        )

    sub = df[df["label"] == "subtract_linear"]
    if not sub.empty:
        s = sub.iloc[0]
        if s["forget_ppl"] > 1000:
            lines.append(
                "| **subtract_linear** | PPL blow-up — negative control only; not viable at current weights |"
            )
        else:
            lines.append(
                f"| **subtract_linear** | forget_ppl **{s['forget_ppl']:.2f}**, "
                f"model_utility **{s['model_utility']:.4f}** |"
            )

    if tier == "extended":
        lines.append(
            "| **Caveat** | Extended metrics (larger subsample than smoke, still capped ROUGE/truth). "
            "Closer to full TOFU but not paper-grade. |"
        )
    else:
        lines.append(
            "| **Caveat** | Smoke metrics (subsampled ROUGE/retain/truth/KS). "
            "Use for ranking adapters, not paper-grade TOFU numbers. |"
        )
    return "\n".join(lines)


ROUGE_EXPLANATION = """## Why ROUGE and model_utility look low

`model_utility` is the **harmonic mean** of `retain_rouge`, `real_rouge`, and `world_rouge` only (forget metrics excluded). The harmonic mean is dominated by the **smallest** term: if `real_rouge ≈ 0.001`, utility collapses to ~0.003 even when `retain_rouge ≈ 0.28`.

| Factor | What it means |
|--------|----------------|
| **OOD utility splits** | Shard LoRAs train on **fictional** TOFU authors. `real_authors` (100 Qs) and `world_facts` (117 Qs) are different tasks; many merges score **near-zero** ROUGE there. |
| **Merge damage** | `linear`, `magnitude_prune`, and `cat` often destroy generation on utility splits; **TIES / DARE-TIES** are the stable merges. |
| **Small model + greedy ROUGE** | Greedy decode, `max_new_tokens=100`, ROUGE-L on full answers — strict vs PPL. Extended uses up to **200** gens per suite (all utility questions). |
| **Base-model floor** | Compare adapters to `base_model.json` (same prompt/metric). If base `real`/`world` ROUGE are already low, low scores reflect **task + metric**, not only bad merges. |

**Worked example (1B `remerge_linear`):** retain 0.275, real 0.001, world 0.021 → harmonic mean ≈ 0.003.
"""


def _truncate(text, limit=280):
    text = str(text).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_generation_block(data, title=None):
    label = data.get("label", title or "unknown")
    parts = [f"### `{label}`" if title is None else f"### {title}", ""]
    for split_name, rows in data.get("splits", {}).items():
        if not rows:
            continue
        parts.append(f"**{split_name}**")
        parts.append("")
        for i, row in enumerate(rows, 1):
            parts.append(f"{i}. **Q:** {_truncate(row.get('question', ''), 220)}")
            parts.append(f"   - **Gold:** {_truncate(row.get('gold', ''), 220)}")
            parts.append(f"   - **Generated:** {_truncate(row.get('generated', ''), 220)}")
            parts.append(f"   - **rougeL:** {row.get('rougeL', '—')}")
            parts.append("")
    return parts


def _load_generation_json(gen_dir, label):
    path = os.path.join(gen_dir, f"{label}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def format_baseline_section(slug, results_sub="extended", compare_label="merged_dare_ties"):
    gen_dir = os.path.join(CKPT, slug, "results", results_sub, "generations")
    base_gen = _load_generation_json(gen_dir, "base_model")
    if base_gen is None:
        return None

    base_metrics_path = os.path.join(CKPT, slug, "results", results_sub, "base_model.json")
    metrics_row = None
    if os.path.isfile(base_metrics_path):
        with open(base_metrics_path) as f:
            metrics_row = json.load(f)

    parts = [
        "## Baseline (no LoRA)",
        "",
        "Base instruct model with **no** shard adapters — same prompt and extended caps as adapters. "
        "Use as the floor/ceiling when judging merge quality.",
        "",
    ]
    if metrics_row:
        mdf = pd.DataFrame([metrics_row])
        mdf["model_slug"] = slug
        parts.append(md_table(mdf, ["label"] + METRIC_COLS))
        parts.append("")

    parts += _format_generation_block(base_gen, title="Baseline generations (`base_model`)")
    parts.append("---")
    parts.append("")

    compare_gen = _load_generation_json(gen_dir, compare_label)
    if compare_gen:
        parts += [
            f"### Same questions: baseline vs `{compare_label}`",
            "",
            "Indices match across generation files (same random seed per split).",
            "",
        ]
        for split_name, base_rows in base_gen.get("splits", {}).items():
            cmp_rows = {r.get("index"): r for r in compare_gen.get("splits", {}).get(split_name, [])}
            if not base_rows:
                continue
            parts.append(f"**{split_name}**")
            parts.append("")
            for i, brow in enumerate(base_rows, 1):
                crow = cmp_rows.get(brow.get("index"))
                parts.append(f"{i}. **Q:** {_truncate(brow.get('question', ''), 220)}")
                parts.append(f"   - **Gold:** {_truncate(brow.get('gold', ''), 220)}")
                parts.append(f"   - **Baseline:** {_truncate(brow.get('generated', ''), 200)} (rougeL {brow.get('rougeL', '—')})")
                if crow:
                    parts.append(
                        f"   - **`{compare_label}`:** {_truncate(crow.get('generated', ''), 200)} "
                        f"(rougeL {crow.get('rougeL', '—')})"
                    )
                parts.append("")
        parts.append("---")
        parts.append("")

    return "\n".join(parts)


def format_generation_examples(slug, results_sub="extended"):
    gen_dir = os.path.join(CKPT, slug, "results", results_sub, "generations")
    if not os.path.isdir(gen_dir):
        return None
    paths = [
        p for p in glob.glob(os.path.join(gen_dir, "*.json"))
        if os.path.basename(p).replace(".json", "") != "base_model"
    ]
    if not paths:
        return None

    paths.sort(key=lambda p: label_sort_key(os.path.basename(p).replace(".json", "")))

    parts = [
        "## Adapter example generations",
        "",
        "Greedy decode (`Question: …\\nAnswer:`), 3 random examples per split. "
        "**rougeL** is per-example ROUGE-L vs gold.",
        "",
    ]
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        parts += _format_generation_block(data)
        parts.append("---")
        parts.append("")
    return "\n".join(parts)


def compare_models_table(compare_slugs, results_sub="extended", key_cols=None):
    key_cols = key_cols or ["forget_ppl", "forget_rouge", "retain_rouge", "real_rouge", "world_rouge", "model_utility"]
    frames = []
    for slug in compare_slugs:
        rows = load_eval_rows(results_sub, model_slug=slug)
        if not rows:
            continue
        sub = pd.DataFrame(rows)[["label"] + [c for c in key_cols if c in pd.DataFrame(rows).columns]]
        sub = sub.rename(columns={c: f"{slug}:{c}" for c in key_cols if c in sub.columns})
        if frames:
            merged = frames[0].merge(sub, on="label", how="outer")
            frames[0] = merged
        else:
            frames.append(sub)
    if not frames:
        return "_No comparison data._"
    df = frames[0].sort_values("label", key=lambda s: s.map(label_sort_key))
    return md_table(df, cols=list(df.columns))


def smoke_extended_delta(df_ext, model_slug, forget_id=3):
    smoke_rows = load_eval_rows("smoke", model_slug)
    if not smoke_rows:
        return "_No smoke JSON for comparison._"
    df_sm = pd.DataFrame(smoke_rows)
    keys = ["forget_ppl", "forget_rouge", "model_utility", "ks_pval"]
    lines = [
        "| Label | forget_ppl (smoke → ext) | forget_rouge | model_utility | ks_pval |",
        "|-------|--------------------------|--------------|---------------|---------|",
    ]
    for label in sorted(df_ext["label"].unique(), key=label_sort_key):
        e = df_ext[df_ext["label"] == label]
        s = df_sm[df_sm["label"] == label]
        if e.empty:
            continue
        er = e.iloc[0]
        if s.empty:
            lines.append(f"| `{label}` | — | — | — | — |")
            continue
        sr = s.iloc[0]
        fp = f"{sr['forget_ppl']:.2f} → {er['forget_ppl']:.2f}"
        fr = f"{sr['forget_rouge']:.4f} → {er['forget_rouge']:.4f}"
        mu = f"{sr['model_utility']:.4f} → {er['model_utility']:.4f}"
        ks = f"{sr['ks_pval']:.4f} → {er['ks_pval']:.4f}"
        lines.append(f"| `{label}` | {fp} | {fr} | {mu} | {ks} |")
    return "\n".join(lines)


def manifest_table(k=4, forget_id=3):
    labels = smoke_eval_labels(k, forget_id)
    rows = []
    for lab in labels:
        if lab.startswith("shard_"):
            meaning = f"Activate only shard {lab.split('_')[1]} LoRA"
        elif lab.startswith("merged_"):
            meaning = f"Merge all k shards: {lab[len('merged_'):]}"
        elif lab.startswith("remerge_"):
            meaning = f"Merge all shards except forget shard {forget_id}: {lab[len('remerge_'):]}"
        else:
            meaning = "Task-vector subtraction unlearn (cat-based)"
        rows.append(f"| `{lab}` | {meaning} |")
    return "\n".join(["| Label | Meaning |", "|-------|---------|"] + rows)


def build_full_report(df, args, forget_id=3, extended=False):
    slugs = sorted(df["model_slug"].unique())
    primary = slugs[0] if len(slugs) == 1 else None
    tier = "extended" if extended else "smoke"
    results_sub = "extended" if extended else "smoke"
    manifest_name = "eval_manifest_extended.txt" if extended else "eval_manifest_smoke.txt"
    status = f"Complete — {len(df)} adapter evaluations"
    if primary:
        expected = len(smoke_eval_labels(4, forget_id))
        if len(df[df["model_slug"] == primary]) < expected:
            status = f"In progress — {len(df[df['model_slug'] == primary])}/{expected} for {primary}"

    title = (
        "# TOFU SISA-LoRA — Llama Merge Extended Evaluation Report"
        if extended
        else "# TOFU SISA-LoRA — Smoke Evaluation Report"
    )

    body_parts = [
        title,
        "",
        f"**Generated:** {date.today()}  ",
        f"**Project root:** `{ROOT}`  ",
        f"**Status:** {status}",
        "",
        "---",
        "",
        "## 1. Executive summary",
        "",
        executive_summary(df, forget_id, tier=tier),
        "",
        "**Reading forget metrics:** Higher `forget_ppl` and lower `forget_rouge` = more forgetting (TOFU convention). "
        f"`shard_{forget_id}_only` is the **positive control** (trained on forget data); "
        "unlearn candidates are `remerge_*` or `subtract_linear`.",
        "",
        "---",
        "",
        "## 2. Experimental setup",
        "",
        "### 2.1 Base models and checkpoints",
        "",
        "| Base model | HuggingFace ID | Checkpoint directory |",
        "|------------|----------------|----------------------|",
    ]

    model_ids = {
        "TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "phi-2": "microsoft/phi-2",
        "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    }
    for slug in slugs:
        hf_id = model_ids.get(slug, df[df["model_slug"] == slug]["model_name"].iloc[0])
        body_parts.append(f"| {slug} | `{hf_id}` | `checkpoints/{slug}/` |")

    body_parts += [
        "",
        "### 2.2 Sharding and forget set",
        "",
        "- **k = 4** author shards on TOFU full dataset (200 authors → 50 authors per shard).",
        f"- **Forget shard id = {forget_id}** (default): forget set = authors in shard {forget_id}.",
        "",
        f"### 2.3 {tier.capitalize()} eval manifest",
        "",
        manifest_table(4, forget_id),
        "",
    ]

    if extended:
        body_parts += [
            "### 2.4 Extended metric subsampling (`eval_tofu.py --extended`)",
            "",
            "| Metric | Full eval (approx.) | Smoke | Extended |",
            "|--------|---------------------|--------|----------|",
            "| ROUGE (each suite) | Up to all questions | 50 | **200** |",
            "| Retain PPL | 500 samples | 80 | **400** |",
            "| Truth ratio | All matching perturbed rows | 30 | **120** |",
            "| KS vs base | All forget texts | 100 (subsampled) | **all forget shard texts** |",
            "",
            "### 2.5 Jobs",
            "",
            "- Submit: `bash submit_llama_extended_eval.sh 4 meta-llama/Llama-3.2-3B-Instruct` (or 1B default)",
            f"- SLURM: 1 GPU / task, sprint4 excluded, **{12}** concurrent tasks, `{os.environ.get('TOFU_EXTENDED_TIME', '02:30:00')}` wall per task",
        ]
    else:
        body_parts += [
            "### 2.4 Smoke metric subsampling (`eval_tofu.py --smoke`)",
            "",
            "| Metric | Full eval (approx.) | Smoke |",
            "|--------|---------------------|--------|",
            "| ROUGE (forget / retain / real / world) | Up to all questions | **50** generations each |",
            "| Retain PPL | 500 samples | **80** samples |",
            "| Truth ratio | All matching perturbed rows | **30** rows |",
            "| KS vs base | All forget texts | **100** texts (`forget_ks_indices.npy`) |",
            "",
            "### 2.5 Jobs",
            "",
            "- Submit: `bash submit_llama_merge_smoke.sh`",
            "- SLURM: 1 GPU / task, sprint4 excluded, up to 12 concurrent array tasks",
        ]

    body_parts += [
        "",
        "---",
        "",
        "## 3. Where results live",
        "",
        "| Artifact | Path |",
        "|----------|------|",
        f"| Per-adapter JSON | `checkpoints/<model-slug>/results/{results_sub}/<label>.json` |",
        f"| Progress | `checkpoints/<model-slug>/results/{results_sub}/<label>.progress.json` |",
        f"| Manifest | `checkpoints/<model-slug>/results/{results_sub}/{manifest_name}` |",
        f"| retain90 KS reference | `checkpoints/<model-slug>/results/{results_sub}/retain_tr_scores.npy` |",
        f"| Combined CSV | `{args.csv}` |",
        f"| This report | `{args.out}` |",
        "",
        "Regenerate:",
        "",
        "```bash",
        "cd <repo>/tofu_sisa_lora",
        f"python collect_results.py --root checkpoints --{tier}",
        (
            "python reports/generate_smoke_report.py --full --extended "
            "--model-slug Llama-3.2-1B-Instruct --out reports/EXTENDED_EVAL_REPORT.md"
            if extended
            else "python reports/generate_smoke_report.py --full --model-slug Llama-3.2-1B-Instruct"
        ),
        "```",
        "",
        "---",
        "",
        ROUGE_EXPLANATION,
        "",
        "---",
    ]

    compare_slugs = getattr(args, "compare_slugs", None) or []
    if compare_slugs and extended:
        body_parts += [
            "## 3.5 Model comparison (1B vs 3B extended)",
            "",
            compare_models_table(compare_slugs, results_sub=results_sub),
            "",
            "---",
            "",
        ]

    section_num = 4
    for slug in slugs:
        sub = df[df["model_slug"] == slug].copy()
        sub["_sort"] = sub["label"].map(lambda x: label_sort_key(x))
        sub = sub.sort_values("_sort").drop(columns=["_sort"])
        model_name = sub["model_name"].iloc[0] if "model_name" in sub.columns else slug
        body_parts += [
            f"## {section_num}. Full results — {slug}",
            "",
            f"**Forget shard (training):** `shard_{forget_id}`  ",
            f"**Model:** `{model_name}`",
            "",
            md_table(sub),
            "",
            "---",
            "",
        ]
        section_num += 1

    if extended and args.model_slug:
        body_parts += [
            "## 7. Smoke vs extended (key metrics)",
            "",
            "Higher sample counts usually stabilize ROUGE/KS; large jumps may indicate subsampling noise in smoke.",
            "",
            smoke_extended_delta(df, args.model_slug, forget_id),
            "",
            "---",
            "",
        ]
        section_cmp = 8
    else:
        section_cmp = 7

    body_parts += [
        f"## {section_cmp}. Cross-method comparison",
        "",
        f"### {section_cmp}.1 Forget-trained adapter (`shard_{forget_id}_only`) — memorization baseline",
        "",
        "Lower forget_ppl / higher forget_rouge = **more** forget knowledge retained.",
        "",
        md_table(df[df["label"] == f"shard_{forget_id}_only"], ["model_slug"] + METRIC_COLS),
        "",
        f"### {section_cmp}.2 Merged adapters (all k shards)",
        "",
        md_table(
            df[df["label"].str.startswith("merged_")].sort_values("label"),
            ["label"] + METRIC_COLS,
        ),
        "",
        f"### {section_cmp}.3 Remerged adapters (exclude forget shard — unlearn candidates)",
        "",
        md_table(
            df[df["label"].str.startswith("remerge_")].sort_values("label"),
            ["label"] + METRIC_COLS,
        ),
        "",
        f"### {section_cmp}.4 Subtract linear unlearn",
        "",
        md_table(df[df["label"] == "subtract_linear"], ["model_slug", "label"] + METRIC_COLS),
        "",
        "---",
        "",
        GLOSSARY,
        "",
        "---",
        "",
    ]

    for slug in slugs:
        baseline_section = format_baseline_section(slug, results_sub=results_sub)
        if baseline_section:
            body_parts += [baseline_section]
            break
    gen_section = None
    for slug in slugs:
        gen_section = format_generation_examples(slug, results_sub=results_sub)
        if gen_section:
            break
    if gen_section:
        body_parts += [gen_section, "---", ""]

    body_parts += [
        "## 9. Recommended next steps",
        "",
        "1. **Full eval** (non-smoke, full ROUGE): `bash submit_eval.sh checkpoints/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B-Instruct 4`",
        "2. **Density sweeps** — `merged_ties_d0.5`, etc. (see `merge_lora.py`)",
        "3. **SVD merges** — `merged_ties_svd`, `merged_dare_ties_svd` for KnOTS-style compression",
        "4. Do not trust **subtract_linear** if forget_ppl explodes; prefer **remerge_*** for unlearn",
        "",
        "---",
        "",
        "## 10. Raw JSON index",
        "",
        "```",
    ]
    for slug in slugs:
        res_dir = os.path.join(CKPT, slug, "results", results_sub)
        if os.path.isdir(res_dir):
            for f in sorted(os.listdir(res_dir)):
                if f.endswith(".json") and not f.endswith(".progress.json"):
                    body_parts.append(f"checkpoints/{slug}/results/{results_sub}/{f}")
    body_parts.append("```")
    body_parts.append("")
    flag = '"extended": true' if extended else '"smoke": true'
    body_parts.append(f"All entries include {flag} and `\"model_name\"` for provenance.")
    return "\n".join(body_parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extended", action="store_true", help="Read results/extended/ and write extended report")
    p.add_argument("--out", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--full", action="store_true", help="Write full report structure")
    p.add_argument("--model-slug", default=None, help="Filter to one checkpoint slug (e.g. Llama-3.2-1B-Instruct)")
    p.add_argument("--forget-shard-id", type=int, default=3)
    p.add_argument(
        "--compare-slugs",
        default=None,
        help="Comma-separated slugs for side-by-side extended comparison section",
    )
    args = p.parse_args()
    args.compare_slugs = [s.strip() for s in args.compare_slugs.split(",") if s.strip()] if args.compare_slugs else []

    results_sub = "extended" if args.extended else "smoke"
    if args.out is None:
        args.out = EXTENDED_REPORT if args.extended else REPORT
    if args.csv is None:
        args.csv = (
            os.path.join(CKPT, "all_metrics_extended.csv")
            if args.extended
            else os.path.join(CKPT, "all_metrics_smoke.csv")
        )

    rows = load_eval_rows(results_sub, model_slug=args.model_slug)
    if not rows:
        raise SystemExit(f"No JSON found under checkpoints/*/results/{results_sub}/")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    print(f"Wrote {len(df)} rows -> {args.csv}")

    if args.full:
        body = build_full_report(
            df, args, forget_id=args.forget_shard_id, extended=args.extended
        )
        out_path = args.out
    else:
        sections = []
        for slug in sorted(df["model_slug"].unique()):
            sub = df[df["model_slug"] == slug].sort_values("label")
            model_name = sub["model_name"].iloc[0] if "model_name" in sub.columns else slug
            sections.append(f"### {slug}\n\n**Model:** `{model_name}`\n\n{md_table(sub)}\n")
        body = f"# TOFU SISA-LoRA — Smoke Evaluation Report\n\n**Generated:** {date.today()}\n\n{chr(10).join(sections)}\n\n{GLOSSARY}"
        out_path = args.out.replace(".md", "_auto.md")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(body)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
