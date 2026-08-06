"""Experiment B driver: paraphrase robustness and cross-author leakage under uniform aggregation.

For ONE served model, scores EVERY author in the pool on BOTH question surfaces and writes raw
per-question rows plus per-(author, surface) aggregates. Run it once per condition; the contrasts
(target vs non-target, original vs paraphrase, pre- vs post-deletion) are assembled offline by
collect_expb.py, so no condition needs to know about any other.

WHAT IT DOES NOT DO: reimplement any metric. ROUGE-L recall, answer probability and the truth
ratio all come from eval_tofu's canonical functions (guarded by test_ou_equivalence.py) via their
per_example sinks; this file only chooses the rows, the surface, and the bookkeeping.

SURFACES. `original` reads row["question"]; `paraphrase` reads row["paraphrased_question"] from
the *_perturbed splits. Everything else is held constant - same gold answer, same perturbed set,
same correct_key - so the contrast is a pure question-surface manipulation.

GOLD CHOICE. ROUGE-L is scored against the ORIGINAL `answer` on both surfaces (primary), with
`paraphrased_answer` as a secondary column. Measured 2026-07-28 on forget10_perturbed: a model
reproducing `answer` verbatim scores only 0.380 ROUGE-L recall against `paraphrased_answer`, so
scoring the paraphrase surface against the paraphrased gold builds in a ~2.5x penalty before any
memorization effect exists. (The repo is internally inconsistent here: sepmlp measure_selectivity
and entangle_data pair para-Q with para-A; skill_data pairs para-Q with the original answer.)

LEGAL TARGETS. Only authors 0-19 and 180-199 have perturbed rows at all (verified: 20 rows each,
0 elsewhere), and the retain90 oracle trained on 0-179 - so the only authors that are both
paraphrase-covered and outside the oracle, i.e. for which Forget Quality means what it says, are
180-199. The script refuses anything else unless --allow_uncovered.

FORGET QUALITY. The repo's cached retain_tr_scores.npy is the oracle's truth ratio on author 199
ONLY (shape (20,), mean 0.8318) and --eval_shard_id does not re-derive it, so it cannot be reused
for another author. --build_refs writes one reference per (author, surface); every scoring run
verifies the reference's recorded row-set hash against the rows it actually scored and refuses a
mismatch. With n = m = 20 the KS test has only 20 attainable p-values and needs D >= 0.45 for
alpha = 0.05 - ks_statistic and forget_truth_ratio are the effect sizes, forget_quality is a
descriptive ordinal.

Usage:
    # 1. references (once), from the retain-only oracle
    python measure_adapter_selectivity.py --model_name meta-llama/Llama-2-7B-chat-hf \
        --adapter ${TOFU_CKPT_ROOT}/.../k200_r32_e25_lr1e4/retain90 --arm ref --condition retain90 \
        --authors 180-199 --build_refs --ref_dir ${TOFU_CKPT_ROOT}/.../expb/refs \
        --out_json .../ref_retain90.json --out_csv .../ref_retain90.csv
    # 2. any condition
    python measure_adapter_selectivity.py ... --adapter .../merges/expb_mean_N20_drop_a195 \
        --arm mean20 --condition drop_a195 --target_author 195 --ref_dir ${TOFU_CKPT_ROOT}/.../expb/refs
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import ks_2samp

import eval_tofu as E
import tofu_env as _tofu_env

PARA_COVERED = set(range(0, 20)) | set(range(180, 200))
SURFACES = ("original", "paraphrase")
QKEY = {"original": "question", "paraphrase": "paraphrased_question"}


def _script_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _rows_sha(rows, key):
    h = hashlib.sha256()
    for r in rows:
        h.update(r[key].encode())
    return h.hexdigest()[:16]


def build_author_rows(tofu):
    """author -> its perturbed rows, joined to `full` on question text.

    Author a owns `full` rows [20a, 20a+20). The perturbed splits carry no author id, so the
    text join is the only mapping; it is exact (0 unjoinable rows, verified 2026-07-28).
    """
    full = tofu["full"]
    q2a = {}
    for i, r in enumerate(full):
        q2a.setdefault(r["question"], i // 20)
    by_author = defaultdict(list)
    for r in tofu["full_pert"]:
        a = q2a.get(r["question"])
        if a is not None:
            by_author[a].append(r)
    return by_author


def score_author_surface(model, tok, rows, surface, max_new_tokens=100):
    """All Exp-B metrics for one author on one surface, via eval_tofu's canonical functions."""
    qkey = QKEY[surface]
    questions = [r[qkey] for r in rows]
    gold = [r["answer"] for r in rows]
    gold_para = [r.get("paraphrased_answer") or r["answer"] for r in rows]

    rouge_sink, rouge_para_sink, prob_sink, tr_sink = [], [], [], []
    rouge = E.get_rouge(model, tok, questions, gold, max_new_tokens=max_new_tokens,
                        per_example=rouge_sink, name=f"rouge_{surface}")
    # secondary gold, same generations would be ideal but get_rouge regenerates; cheap at n=20
    rouge_para = E.get_rouge(model, tok, questions, gold_para, max_new_tokens=max_new_tokens,
                             per_example=rouge_para_sink, name=f"rougepara_{surface}")
    prob = E.get_answer_probability(model, tok, questions, gold, per_example=prob_sink,
                                    name=f"prob_{surface}")
    tr = E.get_truth_ratio_scores(model, tok, rows, "paraphrased_answer",
                                  question_key=qkey, per_example=tr_sink)
    return {
        "rouge": rouge, "rouge_para_gold": rouge_para, "prob": prob,
        "truth_ratio_raw": tr,
        "forget_truth_ratio": E.tr_forget_agg(tr) if len(tr) else float("nan"),
        "sinks": {"rouge": rouge_sink, "rouge_para": rouge_para_sink,
                  "prob": prob_sink, "tr": tr_sink},
    }


def build_extra_split(name, tofu, adapter_dir, n, seed, heldout_authors_n):
    """Experiment C query tiers — rows NO adapter in the merge owns.

    C1 `heldout_authors`: pool authors ABSENT from this merge, read from the merge's own
      merge_meta.json. Same distribution and format as the owned authors, simply unowned, which
      makes it the cleanest contrast — anything that differs is ownership, not domain shift.
    C2 `holdout10`: the TOFU `holdout10` config. Genuinely never trained (0/400 of its questions
      appear in `full`, verified 2026-07-28), so it is the honest "new query" set rather than a
      held-out slice of something the pool saw.

    Returns (questions, gold, provenance) or None when the tier is not derivable — e.g. C1 on a
    BASE-model row, which has no merge_meta.json and therefore no notion of "absent".
    """
    rng = np.random.default_rng(seed)
    if name == "holdout10":
        from datasets import load_dataset
        ds = load_dataset("locuslab/TOFU", "holdout10")["train"]
        idx = sorted(int(i) for i in rng.choice(len(ds), size=min(n, len(ds)), replace=False))
        return ([ds[i]["question"] for i in idx], [ds[i]["answer"] for i in idx],
                {"tier": "C2", "source": "locuslab/TOFU:holdout10", "rows": idx})

    if name == "heldout_authors":
        meta_p = os.path.join(adapter_dir, "merge_meta.json") if adapter_dir != "BASE" else None
        if not meta_p or not os.path.exists(meta_p):
            print(f"[selectivity] {name}: no merge_meta.json at {adapter_dir} — the set of "
                  f"authors ABSENT from the merge is undefined, skipping tier")
            return None
        merged = set(json.load(open(meta_p)).get("authors") or [])
        if not merged:
            print(f"[selectivity] {name}: merge_meta.json lists no authors, skipping tier")
            return None
        full = tofu["full"]
        n_authors = len(full) // 20
        absent = [a for a in range(n_authors) if a not in merged]
        if not absent:
            print(f"[selectivity] {name}: every author is in the merge — no held-out authors")
            return None
        pick = sorted(int(a) for a in
                      rng.choice(absent, size=min(heldout_authors_n, len(absent)), replace=False))
        qs, gold, rows = [], [], []
        for a in pick:
            for i in range(20 * a, 20 * a + 20):     # author a owns full rows [20a, 20a+20)
                qs.append(full[i]["question"]); gold.append(full[i]["answer"]); rows.append(i)
        return qs, gold, {"tier": "C1", "source": "locuslab/TOFU:full",
                          "heldout_authors": pick, "n_merged": len(merged), "rows": rows}

    raise SystemExit(f"unknown extra split '{name}' (want holdout10 | heldout_authors)")


def score_extra_split(model, tok, questions, gold, max_new_tokens=100, name="tier"):
    """ROUGE-L + answer probability only. No truth ratio: these tiers have no perturbed rows,
    and a truth ratio without its perturbed set is not defined — reporting one would invent a
    number rather than measure it."""
    rouge_sink, prob_sink = [], []
    rouge = E.get_rouge(model, tok, questions, gold, max_new_tokens=max_new_tokens,
                        per_example=rouge_sink, name=f"rouge_{name}")
    prob = E.get_answer_probability(model, tok, questions, gold, per_example=prob_sink,
                                    name=f"prob_{name}")
    return {"rouge": rouge, "prob": prob,
            "sinks": {"rouge": rouge_sink, "prob": prob_sink}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'BASE'")
    ap.add_argument("--arm", required=True, help="e.g. sum20 / mean20 / ref")
    ap.add_argument("--condition", required=True, help="e.g. full / drop_a195 / retain90 / base")
    ap.add_argument("--target_author", type=int, default=None,
                    help="the author whose adapter was excluded (None for full/reference rows)")
    ap.add_argument("--authors", default="180-199")
    ap.add_argument("--surfaces", default="original,paraphrase")
    ap.add_argument("--ref_dir", default=None,
                    help="per-(author,surface) KS references; required unless --build_refs")
    ap.add_argument("--build_refs", action="store_true",
                    help="write the references from THIS model (use the retain-only oracle)")
    ap.add_argument("--allow_uncovered", action="store_true",
                    help="permit authors with no perturbed rows (fq/tr will be undefined)")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra_splits", default=None,
                    help="Experiment C query tiers, comma-separated: holdout10 (never-trained "
                         "TOFU-style QA) and/or heldout_authors (pool authors absent from THIS "
                         "merge). Scored with ROUGE-L + answer probability only. Default off, "
                         "so an Exp-B run is byte-identical to before this flag existed.")
    ap.add_argument("--extra_n", type=int, default=200,
                    help="rows sampled for holdout10")
    ap.add_argument("--heldout_authors_n", type=int, default=3,
                    help="how many absent authors to score for the heldout_authors tier "
                         "(20 rows each)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out_json):
        print(f"[selectivity] skip existing {args.out_json}")
        return
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from merge_subset import _parse_author_list
    authors = _parse_author_list(args.authors)
    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    uncovered = [a for a in authors if a not in PARA_COVERED]
    if uncovered and not args.allow_uncovered:
        raise SystemExit(
            f"authors {uncovered} have NO perturbed rows (only 0-19 and 180-199 do), so their "
            f"truth ratio and forget_quality are undefined. Pick targets from 180-199 (which are "
            f"also outside the retain90 oracle's 0-179 training set), or pass --allow_uncovered.")

    hf_home = os.environ.setdefault("HF_HOME", _tofu_env.hf_home())
    tofu = E.load_tofu_data(hf_home)
    by_author = build_author_rows(tofu)

    if args.adapter == "BASE":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    else:
        model, tok = E.load_single_adapter(args.model_name, args.adapter, adapter_name="sel")
    model.eval()

    if args.ref_dir:
        os.makedirs(args.ref_dir, exist_ok=True)

    per_q, summary = [], []
    for a in authors:
        rows = by_author.get(a, [])
        if not rows:
            print(f"[selectivity] author {a}: no perturbed rows, skipping")
            continue
        for surface in surfaces:
            qkey = QKEY[surface]
            if qkey not in rows[0]:
                print(f"[selectivity] author {a}: no {qkey}, skipping {surface}")
                continue
            res = score_author_surface(model, tok, rows, surface, args.max_new_tokens)
            tr = res["truth_ratio_raw"]
            row_sha = _rows_sha(rows, qkey)

            ref_name = f"retain_tr_a{a}_{surface}.npy"
            ref_path = os.path.join(args.ref_dir, ref_name) if args.ref_dir else None
            fq = ks_stat = float("nan")
            ref_n = None
            if args.build_refs and ref_path:
                np.save(ref_path, tr)
                meta_p = ref_path.replace(".npy", ".json")
                json.dump({"author": a, "surface": surface, "n": int(len(tr)),
                           "mean": float(np.mean(tr)) if len(tr) else None,
                           "rows_sha": row_sha, "question_key": qkey,
                           "oracle_adapter": args.adapter, "model_name": args.model_name,
                           "script_sha256": _script_sha()}, open(meta_p, "w"), indent=2)
            elif ref_path and os.path.exists(ref_path):
                ref = np.load(ref_path)
                meta = json.load(open(ref_path.replace(".npy", ".json")))
                if meta.get("rows_sha") != row_sha:
                    raise SystemExit(
                        f"KS reference {ref_path} was built on a DIFFERENT row set "
                        f"(sha {meta.get('rows_sha')} vs {row_sha}). Comparing truth ratios "
                        f"computed on different questions is meaningless — rebuild the refs.")
                if len(ref) and len(tr):
                    k = ks_2samp(tr, ref)
                    fq, ks_stat = float(k.pvalue), float(k.statistic)
                ref_n = int(len(ref))

            for kind, sink in res["sinks"].items():
                for r in sink:
                    per_q.append({
                        "arm": args.arm, "condition": args.condition,
                        "target_author": args.target_author,
                        "is_target": int(args.target_author == a),
                        "author": a, "surface": surface, "metric": kind,
                        "i": r.get("i"), "question": r.get("question"), "gold": r.get("gold"),
                        "generated": r.get("generated"), "score": r.get("score"),
                        "tr": r.get("tr"), "avg_nll": r.get("avg_nll"),
                        "kept": r.get("kept", True),
                        "gen_n_tokens": r.get("gen_n_tokens"),
                        "hit_max_tokens": r.get("hit_max_tokens"),
                    })
            summary.append({
                "arm": args.arm, "condition": args.condition,
                "target_author": args.target_author, "author": a,
                "is_target": int(args.target_author == a), "surface": surface,
                "n_rows": len(rows),
                "rouge": res["rouge"], "rouge_para_gold": res["rouge_para_gold"],
                "prob": res["prob"], "forget_truth_ratio": res["forget_truth_ratio"],
                "truth_ratio_mean": float(np.mean(tr)) if len(tr) else float("nan"),
                "n_tr": int(len(tr)),
                "forget_quality": fq, "ks_statistic": ks_stat,
                "ref_file": ref_name if args.ref_dir else None, "ref_n": ref_n,
                "rows_sha": row_sha,
            })
            print(f"[selectivity] a{a} {surface:10} rouge={res['rouge']:.4f} "
                  f"prob={res['prob']:.4f} ftr={res['forget_truth_ratio']:.4f} "
                  f"fq={fq:.4g} D={ks_stat:.4g}", flush=True)

    # ── Experiment C tiers ───────────────────────────────────────────────────────────────────
    # Appended as extra rows with author=None and surface="tier:<name>", so the Exp-B columns
    # keep their meaning and a consumer that does not know about Exp C simply never selects
    # these rows. Metrics that need perturbed data (truth ratio, forget_quality) stay NaN here
    # rather than being filled with a plausible-looking substitute.
    extra_meta = {}
    for tier in [s.strip() for s in (args.extra_splits or "").split(",") if s.strip()]:
        built = build_extra_split(tier, tofu, args.adapter, args.extra_n, args.seed,
                                  args.heldout_authors_n)
        if built is None:
            continue
        qs, gold, prov = built
        res = score_extra_split(model, tok, qs, gold, args.max_new_tokens, name=tier)
        extra_meta[tier] = {**prov, "n": len(qs)}
        for kind, sink in res["sinks"].items():
            for r in sink:
                per_q.append({
                    "arm": args.arm, "condition": args.condition,
                    "target_author": args.target_author, "is_target": 0,
                    "author": None, "surface": f"tier:{tier}", "metric": kind,
                    "i": r.get("i"), "question": r.get("question"), "gold": r.get("gold"),
                    "generated": r.get("generated"), "score": r.get("score"),
                    "tr": r.get("tr"), "avg_nll": r.get("avg_nll"),
                    "kept": r.get("kept", True),
                    "gen_n_tokens": r.get("gen_n_tokens"),
                    "hit_max_tokens": r.get("hit_max_tokens"),
                })
        summary.append({
            "arm": args.arm, "condition": args.condition,
            "target_author": args.target_author, "author": None, "is_target": 0,
            "surface": f"tier:{tier}", "n_rows": len(qs),
            "rouge": res["rouge"], "rouge_para_gold": float("nan"), "prob": res["prob"],
            "forget_truth_ratio": float("nan"), "truth_ratio_mean": float("nan"), "n_tr": 0,
            "forget_quality": float("nan"), "ks_statistic": float("nan"),
            "ref_file": None, "ref_n": None, "rows_sha": None,
        })
        print(f"[selectivity] {tier:16} ({prov['tier']}, n={len(qs)}) "
              f"rouge={res['rouge']:.4f} prob={res['prob']:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    if per_q:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_q[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(per_q)
    out = {
        "arm": args.arm, "condition": args.condition, "target_author": args.target_author,
        "adapter": args.adapter, "model_name": args.model_name, "authors": authors,
        "surfaces": surfaces, "seed": args.seed, "build_refs": args.build_refs,
        "ref_dir": args.ref_dir, "metrics_version": E.METRICS_VERSION,
        "extra_splits": extra_meta or None,
        "rouge_gold": "original `answer` (primary); `paraphrased_answer` as rouge_para_gold",
        "script_sha256": _script_sha(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - t0, 1),
        "per_question_csv": os.path.abspath(args.out_csv),
        "summary": summary,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[selectivity] wrote {args.out_json} ({len(summary)} author x surface rows) "
          f"and {args.out_csv} ({len(per_q)} per-question rows)")


if __name__ == "__main__":
    main()
