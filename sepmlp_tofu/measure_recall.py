"""Per-source ROUGE-L recall sweep — the paper's PRIMARY metric (MUSR §5.1).

The paper defines per-source recall as the mean ROUGE-L (recall) of a GREEDY
GENERATION against the gold answer, and reports:
  - mean per-source recall            (Table 2: 0.975/0.962/0.966 at K=10/50/200)
  - tail = # sources with recall<0.95 (Table 2: 31/200 published, 4/200 tuned)
  - named vs name-free recall         (split questions by whether they name the author)
  - held-out recall on unseen authors (Table 2: 0.341 at K=200; collateral proxy)

This is DISTINCT from measure_selectivity.py --recall_probe, which computes an
answer-PROBABILITY (teacher-forced exp(-avg CE)) — a different quantity. Our
G2/G3 gates used that probability; the paper's headline recall is the ROUGE-L
generation recall computed here. Do not conflate the two in one table.

Reuses relearn_score.score_author / evaluate_rouge (already paper-matching:
greedy, max_new_tokens 200, rougeL.recall, stemmer, OU chat-template) and
relearn.{load_served_model,load_split_qa,author_qa_pairs}. The name-in-question
detector generalizes relearn.soft_name_check to the token SET.

Pure helpers (author_name_tokens / question_is_named / summarize_recall) carry
no model dependency and are unit-tested on a fixture (tests/test_recall.py).
"""

import argparse
import json
import os
import re
import time

import torch

from sepmlp_common import (
    NUM_AUTHORS,
    RECORDS_PER_AUTHOR,
    file_sha256,
    import_memadapt_data,
    load_config,
    save_json,
    set_determinism,
    slurm_job_id,
)

# The paper's tail threshold (# sources below this recall).
TAIL_THRESHOLD = 0.95

# Reuse the generic-capitalized stop set that soft_name_check already tuned so
# author names win the vote (sentence starters, pronouns, "Author", ...), and
# the holdout10 author count (single source of truth).
from relearn import _GENERIC_CAPITALIZED, HOLDOUT10_AUTHORS  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers (no model) — unit-tested
# ---------------------------------------------------------------------------

def author_name_tokens(answers, min_frac: float = 0.4) -> set:
    """The author's name token set: capitalized tokens (len>=3, non-generic)
    that recur across the author's answers. A token is a name token if it
    appears in >= min_frac of the answer rows (floor of 2). Generalizes
    relearn.soft_name_check (which returns only the single top token) so a
    multi-part name ("Hina", "Ameen") is captured for the question split."""
    counts = {}
    n = len(answers)
    for ans in answers:
        toks = set(re.findall(r"\b[A-Z][a-zA-Z'’-]{2,}\b", ans))
        for t in toks:
            if t in _GENERIC_CAPITALIZED:
                continue
            counts[t] = counts.get(t, 0) + 1
    thresh = max(2, int(min_frac * n))
    return {t for t, c in counts.items() if c >= thresh}


def question_is_named(question: str, name_tokens: set) -> bool:
    """True if the question mentions the author by name (case-insensitive word
    match against any name token). Empty name_tokens => not named."""
    if not name_tokens:
        return False
    ql = question.lower()
    for t in name_tokens:
        if re.search(rf"\b{re.escape(t.lower())}\b", ql):
            return True
    return False


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize_recall(per_author: list, tail_threshold: float = TAIL_THRESHOLD) -> dict:
    """Aggregate a per-author breakdown into the paper's Table-2 block.

    per_author: [{"author": int, "recall": float,
                  "rows": [{"rougeL_recall": float, "named": bool}, ...]}, ...]
    Reports mean-of-author-means (the paper's per-SOURCE recall) plus the
    pooled named / name-free row means and the tail count.
    """
    author_recalls = [a["recall"] for a in per_author]
    named_rows = [r["rougeL_recall"]
                  for a in per_author for r in a["rows"] if r["named"]]
    namefree_rows = [r["rougeL_recall"]
                     for a in per_author for r in a["rows"] if not r["named"]]
    return {
        "n_authors": len(per_author),
        "recall": _mean(author_recalls),                 # per-source (mean of author means)
        "recall_pooled": _mean(r["rougeL_recall"]
                               for a in per_author for r in a["rows"]),
        "tail_below": tail_threshold,
        "tail_count": sum(1 for r in author_recalls if r < tail_threshold),
        "named_recall": _mean(named_rows),
        "name_free_recall": _mean(namefree_rows),
        "n_named_rows": len(named_rows),
        "n_name_free_rows": len(namefree_rows),
        "worst_author": (min(per_author, key=lambda a: a["recall"])["author"]
                         if per_author else None),
        "worst_recall": min(author_recalls) if author_recalls else None,
    }


# ---------------------------------------------------------------------------
# Model sweep
# ---------------------------------------------------------------------------

def sweep_recall(model, tokenizer, rows, authors, batch_size, max_new_tokens):
    """Score each author's 20 QA rows with greedy ROUGE-L recall; tag each row
    named/name-free by its own question. Returns the per-author breakdown."""
    import relearn_score
    from relearn import author_qa_pairs

    per_author = []
    for a in authors:
        qa = author_qa_pairs(rows, a)
        name_tokens = author_name_tokens([ans for _, ans in qa])
        scored = relearn_score.evaluate_rouge(
            model, tokenizer, qa, batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        # evaluate_rouge per_row is in qa order; pair each with its question.
        rows_out = []
        for (q, _), row in zip(qa, scored["per_row"]):
            rows_out.append({
                "rougeL_recall": row["rougeL_recall"],
                "named": question_is_named(q, name_tokens),
            })
        recall = _mean(r["rougeL_recall"] for r in rows_out)
        per_author.append({
            "author": int(a),
            "recall": recall,
            "name_tokens": sorted(name_tokens),
            "rows": rows_out,
        })
        print(f"[recall] author={a:3d} recall={recall:.4f} "
              f"named={sum(r['named'] for r in rows_out)}/{len(rows_out)} "
              f"name_tokens={sorted(name_tokens)[:4]}")
    return per_author


def _checkpoint_bank_sha(checkpoint):
    if not checkpoint:
        return None
    meta = os.path.join(checkpoint, "meta.json")
    if os.path.isfile(meta):
        try:
            with open(meta) as f:
                return json.load(f).get("bank_sha")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--serve", default="sepmlp",
                    choices=["sepmlp", "memadapt", "hf"])
    ap.add_argument("--checkpoint", default=None, help="sepmlp/memadapt run dir")
    ap.add_argument("--droplist", default=None, help="sepmlp droplist JSON")
    ap.add_argument("--blocklist", default=None, help="memadapt blocklist JSON")
    ap.add_argument("--k", type=int, default=None,
                    help="# resident authors to score (default: adapter.num_authors)")
    ap.add_argument("--authors", default=None,
                    help="explicit comma-separated author ids (smoke); overrides --k")
    ap.add_argument("--heldout_n", type=int, default=HOLDOUT10_AUTHORS,
                    help="# holdout10 authors for held-out recall (0 to skip)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--tail_threshold", type=float, default=TAIL_THRESHOLD)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["hf_home"])
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_determinism(seed)

    from transformers import AutoTokenizer
    from relearn import load_served_model, load_split_qa

    data_tofu = import_memadapt_data()
    tokenizer = data_tofu.prepare_tokenizer(
        AutoTokenizer.from_pretrained(cfg["model_name"])
    )
    model = load_served_model(
        args.serve, cfg["model_name"], checkpoint=args.checkpoint,
        droplist=args.droplist, blocklist=args.blocklist,
    )

    full_rows = load_split_qa("full")
    if args.authors:
        authors = [int(x) for x in args.authors.split(",") if x != ""]
    else:
        k = args.k or cfg.get("adapter", {}).get("num_authors", NUM_AUTHORS)
        authors = list(range(k))

    per_author = sweep_recall(
        model, tokenizer, full_rows, authors,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
    )
    summary = summarize_recall(per_author, tail_threshold=args.tail_threshold)

    heldout = None
    if args.heldout_n and args.heldout_n > 0:
        hrows = load_split_qa("holdout10")
        hauthors = list(range(min(args.heldout_n, len(hrows) // RECORDS_PER_AUTHOR)))
        hper = sweep_recall(
            model, tokenizer, hrows, hauthors,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        )
        heldout = {
            "held_out_recall": _mean(a["recall"] for a in hper),
            "n_authors": len(hper),
            "per_author": [{"author": a["author"], "recall": a["recall"]}
                           for a in hper],
        }
        print(f"[recall] HELD-OUT recall={heldout['held_out_recall']:.4f} "
              f"over {heldout['n_authors']} holdout10 authors")

    print(f"[recall] SUMMARY recall={summary['recall']:.4f} "
          f"tail(<{args.tail_threshold})={summary['tail_count']}/{summary['n_authors']} "
          f"named={summary['named_recall']} name_free={summary['name_free_recall']}"
          + (f" held_out={heldout['held_out_recall']:.4f}" if heldout else ""))

    result = {
        "metric": "per_source_rougeL_recall (paper §5.1; greedy gen, "
                  "max_new_tokens=%d)" % args.max_new_tokens,
        "serve": args.serve,
        "checkpoint": args.checkpoint,
        "checkpoint_bank_sha": _checkpoint_bank_sha(args.checkpoint),
        "droplist": args.droplist,
        "droplist_sha256": file_sha256(args.droplist) if args.droplist else None,
        "model_name": cfg["model_name"],
        "seed": seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "tail_threshold": args.tail_threshold,
        "summary": summary,
        "held_out": heldout,
        "per_author": per_author,
        "config_path": cfg["_config_path"],
        "config_sha256": file_sha256(cfg["_config_path"]),
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
        "wall_seconds": time.perf_counter() - t0,
        "torch_version": torch.__version__,
    }
    if args.out:
        save_json(result, args.out)
        print(f"[done] wall={result['wall_seconds']:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
