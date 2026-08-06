"""TOFU unlearning metrics for one adapter (SLURM-callable)."""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from evaluate import load as load_metric
from peft import PeftModel
from scipy.stats import hmean, ks_2samp
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_progress import ProgressLogger
from merge_lora import activate_label, default_eval_labels, label_requires_data
from shard_utils import get_author_shard
import tofu_env as _tofu_env


# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

# Stamp on every result dict so corrected (open-unlearning-faithful) results are self-identifying
# and never get mixed with pre-fix JSONs. Bump when metric definitions change.
METRICS_VERSION = "ou-2026-06-10"

ROUGE_LOG_EVERY = 25  # print progress every N generations
# Smoke subsampling: target <55 min wall time per SLURM task (see TOFU_SMOKE_TIME in slurm_nodes.sh).
SMOKE_ROUGE_MAX = 50
SMOKE_RETAIN_MAX = 80
SMOKE_TRUTH_MAX = 30
SMOKE_KS_MAX = 80
# Extended subsampling: ~2–2.5h/task on Llama (see TOFU_EXTENDED_TIME in slurm_nodes.sh).
EXTENDED_ROUGE_MAX = 200
EXTENDED_RETAIN_MAX = 400
EXTENDED_TRUTH_MAX = 120
_rouge_metric = None


def _rouge_metric_cache():
    """Per-job cache avoids parallel SLURM tasks clobbering the same .arrow file."""
    base = os.environ.get("TOFU_METRICS_CACHE")
    if base:
        return base
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    return os.path.join(hf, "metrics_cache", job)


def _get_rouge_metric():
    global _rouge_metric
    if _rouge_metric is None:
        cache = _rouge_metric_cache()
        os.makedirs(cache, exist_ok=True)
        _rouge_metric = load_metric("rouge", cache_dir=cache)
    return _rouge_metric


def _build_qa_prompt(tokenizer, q, a=None):
    """Plain QA prompt — no chat template.

    All TOFU fine-tuning (ours and locuslab's) uses "Question: {q}\nAnswer: {a}" format.
    Applying an instruct chat template at eval time mismatches training and tanks metrics.
    """
    prompt = f"Question: {q}\nAnswer:"
    return prompt if a is None else f"{prompt} {a}"


def load_tofu_data(hf_home):
    os.environ["HF_HOME"] = hf_home
    forget10_pert = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"]
    retain_pert = load_dataset("locuslab/TOFU", "retain_perturbed")["train"]
    # Combine forget10 + retain90 perturbed sets for full 200-author coverage.
    # Needed so we can compute truth ratio on any custom k-shard forget/retain split.
    full_pert = concatenate_datasets([forget10_pert, retain_pert])
    return {
        "full": load_dataset("locuslab/TOFU", "full")["train"],
        "full_pert": full_pert,
        "forget10_pert": forget10_pert,  # kept for backward compat
        "real_authors": load_dataset("locuslab/TOFU", "real_authors")["train"],
        "world_facts": load_dataset("locuslab/TOFU", "world_facts")["train"],
        "real_authors_pert": load_dataset("locuslab/TOFU", "real_authors_perturbed")["train"],
        "world_facts_pert": load_dataset("locuslab/TOFU", "world_facts_perturbed")["train"],
    }


def build_merge_dataloader(data_full, tokenizer, k, exclude_shard=None,
                           batch_size=4, max_length=256, num_examples=256, seed=0):
    """Tokenized batches of the included shards' training text for data-required
    merges (regmean Gram collection, fisher grads, lorahub loss).

    Returns a list of dict batches (input_ids, attention_mask, labels) on CPU —
    consumers move them to device and may re-iterate it once per shard. Text format
    matches train_lora_shard.format_prompt. Excluding the forget shard for remerge_*
    labels keeps SISA semantics: no forget data ever touches the merge.
    """
    rows = []
    for i in range(k):
        if i == exclude_shard:
            continue
        for a in get_author_shard(k, i):
            rows.extend(range(a * 20, a * 20 + 20))
    rows = np.array(rows)
    np.random.default_rng(seed).shuffle(rows)
    subset = data_full.select(rows[:num_examples].tolist())
    texts = [f"Question: {q}\nAnswer: {a}"
             for q, a in zip(subset["question"], subset["answer"])]
    batches = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        batches.append({**enc, "labels": labels})
    return batches


def get_perplexity(model, tokenizer, texts, batch_size=8, max_length=256, prog=None, name="ppl"):
    model.eval()
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    n = len(texts)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            if prog and i % (batch_size * 10) == 0:
                prog.step("perplexity", f"{name} {min(i, n)}/{n}")
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            n_tok = (labels != -100).sum().item()
            total_nll += out.loss.item() * n_tok
            total_tokens += n_tok
    return math.exp(total_nll / total_tokens)


def get_per_sample_logprobs(model, tokenizer, texts, max_length=256, prog=None, name="logprobs"):
    model.eval()
    device = next(model.parameters()).device
    logprobs = []
    n = len(texts)
    with torch.no_grad():
        for idx, text in enumerate(texts):
            if prog and idx % 50 == 0:
                prog.step("logprobs", f"{name} {idx}/{n}")
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            labels = enc["input_ids"].clone()
            out = model(**enc, labels=labels)
            logprobs.append(-out.loss.item())
    return np.array(logprobs)


def get_rouge(
    model, tokenizer, questions, gold_answers, max_new_tokens=100,
    max_samples=None, prog=None, name="rouge",
    per_example=None,
):
    """ROUGE-L recall, mean over questions.

    `per_example` (2026-07-28, default None = byte-identical behavior): an optional list the
    caller supplies; one dict per scored item is appended, including the GENERATION, which this
    function has always computed and discarded on the mean. Needed for raw per-question CSVs and
    for offline degeneracy checks (empty / looping / max-token generations) on damaged models.
    """
    if max_samples is not None and max_samples < len(questions):
        questions = questions[:max_samples]
        gold_answers = gold_answers[:max_samples]
    model.eval()
    device = next(model.parameters()).device
    preds = []
    n = len(questions)
    with torch.no_grad():
        for idx, q in enumerate(questions):
            if prog and idx % ROUGE_LOG_EVERY == 0:
                prog.step("rouge", f"{name} {idx}/{n}")
            prompt = _build_qa_prompt(tokenizer, q)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = ids[0][enc["input_ids"].shape[1] :]
            preds.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    from rouge_score import rouge_scorer as _rs_lib
    _scorer = _rs_lib.RougeScorer(["rougeL"], use_stemmer=True)
    recalls = [_scorer.score(gold, pred)["rougeL"].recall
               for pred, gold in zip(preds, gold_answers)]
    if per_example is not None:
        # `i` is the index into the (already max_samples-truncated) questions list; callers that
        # need dataset row ids pass them alongside. Generation length + the max-token flag are
        # the two cheap collapse tells (never emitting EOS => hit_max_tokens).
        enc_len = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in preds]
        per_example.extend(
            {"i": i, "metric": name, "question": q, "gold": g, "generated": p,
             "score": float(r), "gen_n_tokens": L, "hit_max_tokens": bool(L >= max_new_tokens)}
            for i, (q, g, p, r, L) in enumerate(
                zip(questions, gold_answers, preds, recalls, enc_len)))
    return float(np.mean(recalls)) if recalls else float("nan")


def sample_answers(
    model, tokenizer, questions, gold_answers, indices=None, max_new_tokens=100,
):
    """Greedy-generate answers for selected indices; return per-example dicts with rougeL."""
    if indices is None:
        indices = list(range(min(3, len(questions))))
    rouge = _get_rouge_metric()
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for idx in indices:
            if idx >= len(questions):
                continue
            q = questions[idx]
            gold = gold_answers[idx]
            prompt = _build_qa_prompt(tokenizer, q)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = ids[0][enc["input_ids"].shape[1] :]
            pred = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            from rouge_score import rouge_scorer as _rs_lib
            score = _rs_lib.RougeScorer(["rougeL"], use_stemmer=True).score(gold, pred)["rougeL"].recall
            rows.append({
                "index": int(idx),
                "question": q,
                "gold": gold,
                "generated": pred,
                "rougeL": round(float(score), 4),
            })
    return rows


def get_answer_probability(
    model, tokenizer, questions, answers,
    max_length=256, max_samples=None, prog=None, name="prob",
    per_example=None,
):
    """Mean P(a|q)^(1/|a|) — answer-token loss only (paper Table 1, Probability on open-ended sets).

    `per_example` (2026-07-28, default None = byte-identical): appends one dict per row. Rows
    with no answer tokens are `continue`d out of `probs`, so list position != row index — the
    sink records `i` explicitly and marks them kept=False rather than dropping them silently.
    """
    if max_samples is not None and max_samples < len(questions):
        questions = questions[:max_samples]
        answers = answers[:max_samples]
    model.eval()
    device = next(model.parameters()).device
    probs = []
    n = len(questions)
    with torch.no_grad():
        for idx, (q, a) in enumerate(zip(questions, answers)):
            if prog and idx % 50 == 0:
                prog.step("prob", f"{name} {idx}/{n}")
            n_prompt = tokenizer(_build_qa_prompt(tokenizer, q), return_tensors="pt")["input_ids"].shape[1]
            enc = tokenizer(
                _build_qa_prompt(tokenizer, q, a), return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            labels = enc["input_ids"].clone()
            labels[:, :n_prompt] = -100
            if (labels != -100).sum() == 0:
                if per_example is not None:
                    per_example.append({"i": idx, "metric": name, "question": q, "gold": a,
                                        "avg_nll": float("nan"), "score": float("nan"),
                                        "kept": False})
                continue
            out = model(**enc, labels=labels)
            probs.append(math.exp(-out.loss.item()))
            if per_example is not None:
                per_example.append({"i": idx, "metric": name, "question": q, "gold": a,
                                    "avg_nll": float(out.loss.item()),
                                    "score": float(probs[-1]), "kept": True})
    return float(np.mean(probs)) if probs else float("nan")


def _answer_avg_loss(model, tokenizer, q, a, n_prompt=None, max_length=256):
    """Average per-token CE loss over the answer tokens of "Question: q\nAnswer: a".

    Question tokens are masked (-100), so exp(-return) = P(a|q)^(1/|a|). Returns NaN when
    there are no answer tokens. Mirrors open-unlearning evaluate_probability
    (avg_losses = losses / num_token_gt, utils.py:82-103). Shared by the probability,
    MC-probability and truth-ratio metrics.
    """
    device = next(model.parameters()).device
    if n_prompt is None:
        n_prompt = tokenizer(_build_qa_prompt(tokenizer, q), return_tensors="pt")["input_ids"].shape[1]
    enc = tokenizer(
        _build_qa_prompt(tokenizer, q, a), return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    if (labels != -100).sum() == 0:
        return float("nan")
    return model(**enc, labels=labels).loss.item()


def get_prob_w_options(model, tokenizer, perturbed_ds, max_length=256, max_samples=None,
                       prog=None, name="mc_prob"):
    """Normalised answer probability for real_authors / world_facts.

    Exact port of open-unlearning probability_w_options (memorization.py:44-72):
        prob_i = P(answer|q) / (P(answer|q) + Σ_j P(perturbed_j|q) + 1e-10)
    where each P = exp(-avg_per_token_loss) and the wrong probs are summed (arithmetic).
    Operates on the *_perturbed splits (answer + perturbed_answer list) rather than the
    option1..4 schema — numerically identical since option1..4 == {answer} ∪ perturbed.
    """
    if max_samples is not None and max_samples < len(perturbed_ds):
        perturbed_ds = perturbed_ds.select(range(max_samples))
    model.eval()
    probs = []
    n = len(perturbed_ds)
    with torch.no_grad():
        for idx, row in enumerate(perturbed_ds):
            if prog and idx % 20 == 0:
                prog.step("mc_prob", f"{name} {idx}/{n}")
            q = row["question"]
            perturbed = row["perturbed_answer"]
            if isinstance(perturbed, str):
                perturbed = [perturbed]
            n_prompt = tokenizer(_build_qa_prompt(tokenizer, q), return_tensors="pt")["input_ids"].shape[1]
            l_correct = _answer_avg_loss(model, tokenizer, q, row["answer"], n_prompt, max_length)
            if math.isnan(l_correct):
                continue
            correct = math.exp(-l_correct)
            wrong = 0.0
            for p in perturbed:
                lp = _answer_avg_loss(model, tokenizer, q, p, n_prompt, max_length)
                if not math.isnan(lp):
                    wrong += math.exp(-lp)
            probs.append(correct / (correct + wrong + 1e-10))
    return float(np.mean(probs)) if probs else float("nan")


def get_truth_ratio_scores(model, tokenizer, perturbed_ds, correct_key, max_length=256,
                           prog=None, max_rows=None,
                           question_key="question", per_example=None):
    """Per-sample truth ratio tr_i = wrong_prob_i / (correct_prob_i + 1e-10).

    Exact port of open-unlearning truth_ratio (memorization.py:107-174) + aggregate_to_1D
    (utils.py:35-36):
      correct_prob_i = exp(-loss_correct)              correct = paraphrased_answer for
                                                        forget/retain, answer for ra/wf.
      wrong_prob_i   = exp(-mean_j loss_perturbed_ij)  GEOMETRIC mean of the perturbed probs
                                                        (= exp of the mean perturbed *loss*),
                                                        NOT the arithmetic mean of probs.
    Returns the raw per-sample tr_i as a numpy array — do NOT aggregate here. Aggregate with
    tr_forget_agg (forget) or tr_nonforget_agg (retain/ra/wf); the raw array is also the
    statistic the forget-quality KS test compares.

    Unlike the old geometric-mean-of-R implementation, every row with a valid correct answer
    is kept (open-unlearning never drops p_ref≈0 rows; the +1e-10 guards the divide).

    `question_key` (2026-07-28, default "question" = byte-identical): read the prompt from a
    different column — "paraphrased_question" on the *_perturbed splits gives the SAME answer
    and the SAME perturbed set under a restated question, i.e. a pure question-surface
    manipulation. This is the only metric here that reads the question out of the row; ROUGE and
    probability already take explicit question lists, so paraphrase is a caller concern there.
    ⚠ A truth ratio measured at question_key="paraphrased_question" is NOT comparable to a KS
    reference cached at "question" — build a matching reference per surface.

    `per_example` (default None = byte-identical): appends one dict per row, incl. rows dropped
    by the NaN-loss `continue` (kept=False), so position != row index never corrupts a join.
    """
    model.eval()
    trs = []
    n = len(perturbed_ds)
    with torch.no_grad():
        for idx, row in enumerate(perturbed_ds):
            if max_rows is not None and idx >= max_rows:
                break
            if prog and idx % 20 == 0:
                prog.step("truth_ratio", f"{idx}/{n}")
            q = row[question_key]
            correct = row.get(correct_key) or row["answer"]
            perturbed = row["perturbed_answer"]
            if isinstance(perturbed, str):
                perturbed = [perturbed]
            n_prompt = tokenizer(_build_qa_prompt(tokenizer, q), return_tensors="pt")["input_ids"].shape[1]
            l_correct = _answer_avg_loss(model, tokenizer, q, correct, n_prompt, max_length)
            if math.isnan(l_correct):
                if per_example is not None:
                    per_example.append({"i": idx, "metric": "truth_ratio", "question": q,
                                        "gold": correct, "kept": False, "tr": float("nan")})
                continue
            pert_losses = [_answer_avg_loss(model, tokenizer, q, p, n_prompt, max_length) for p in perturbed]
            pert_losses = [l for l in pert_losses if not math.isnan(l)]
            if not pert_losses:
                if per_example is not None:
                    per_example.append({"i": idx, "metric": "truth_ratio", "question": q,
                                        "gold": correct, "kept": False, "tr": float("nan")})
                continue
            correct_prob = math.exp(-l_correct)
            wrong_prob = math.exp(-float(np.mean(pert_losses)))   # geomean of perturbed probs
            trs.append(wrong_prob / (correct_prob + 1e-10))
            if per_example is not None:
                per_example.append({"i": idx, "metric": "truth_ratio", "question": q,
                                    "gold": correct, "kept": True,
                                    "correct_prob": float(correct_prob),
                                    "wrong_prob": float(wrong_prob),
                                    "tr": float(trs[-1]),
                                    "tr_min_1_over": float(min(trs[-1], 1.0 / (trs[-1] + 1e-10))),
                                    "n_perturbed_kept": len(pert_losses)})
    return np.array(trs, dtype=float)


def tr_forget_agg(tr):
    """closer_to_1_better (open-unlearning memorization.py:114-115).

    tr ∈ [0,1]; higher (→1) = false and true equally likely = better forgetting.
    """
    if len(tr) == 0:
        return float("nan")
    return float(np.mean(np.minimum(tr, 1.0 / (tr + 1e-10))))


def tr_nonforget_agg(tr):
    """true_better (open-unlearning memorization.py:119-120).

    The model-utility truth-ratio component; higher = the model prefers the true answer.
    """
    if len(tr) == 0:
        return float("nan")
    return float(np.mean(np.maximum(0.0, 1.0 - tr)))


def split_eval_indices(shards, forget_shard_id, eval_shard_id, retain_author_ids, n_rows):
    """Forget/retain row split for evaluate_model — factored out so the row math is
    testable offline (test_eval_rows.py). Returns (forget_indices, retain_indices,
    retain_excl_set).

    forget_* metrics always measure the MEASURE shard (eval_shard_id if set, else the
    global forget shard). The retain pool excludes retain_excl_set:
      * eval_shard_id=None                     -> the global forget shard (legacy).
      * eval_shard_id set + retain_author_ids=None -> the MEASURE shard (the nmerge
        __own convention) — byte-identical to the pre-fix behavior.
      * eval_shard_id set + retain_author_ids set  -> the GLOBAL forget shard. Keying
        this combined probe case on the measure shard made a probe row with
        rids == the probe author an EMPTY retain pool (ValueError); the probe
        author's own rows must stay available to the retain restriction while
        forget_* keeps measuring the measure shard.
    """
    measure_id = eval_shard_id if eval_shard_id is not None else forget_shard_id
    forget_authors = shards[measure_id]
    forget_indices = [r for a in forget_authors for r in range(a * 20, a * 20 + 20)]
    forget_set = set(forget_indices)
    if eval_shard_id is not None and retain_author_ids is not None:
        retain_excl_set = {r for a in shards[forget_shard_id]
                           for r in range(a * 20, a * 20 + 20)}
    else:
        retain_excl_set = forget_set
    if retain_author_ids is not None:
        # Subset-conditioned retain (nmerge Exp-5 follow-up): score the retain_* metrics
        # ONLY on these authors' rows — "did the model learn what it was trained on" —
        # instead of the full 200-author population (which dilutes subset knowledge
        # ~N/200 and pins retain_* near base for any partial merge). None = unchanged
        # legacy behavior, bit-identical.
        retain_indices = [r for a in retain_author_ids for r in range(a * 20, a * 20 + 20)
                          if r not in retain_excl_set]
        if not retain_indices:
            raise ValueError("retain_author_ids leaves an empty retain pool")
    else:
        retain_indices = [i for i in range(n_rows) if i not in retain_excl_set]
    return forget_indices, retain_indices, retain_excl_set


def evaluate_model(
    model, tokenizer, label, forget_shard_id, full_ds, shards, forget10_pert,
    real_authors, world_facts, retain_ref_tr_scores=None, rouge_max_samples=None, prog=None,
    smoke=False, extended=False, retain_max_samples=500, truth_max_rows=None,
    full_pert=None, real_authors_pert=None, world_facts_pert=None,
    eval_shard_id=None, retain_author_ids=None,
):
    forget_indices, retain_indices, retain_excl_set = split_eval_indices(
        shards, forget_shard_id, eval_shard_id, retain_author_ids, len(full_ds))
    forget_ds = full_ds.select(forget_indices)
    if retain_author_ids is not None:
        print(f"[eval] retain restricted to {len(retain_author_ids)} authors "
              f"({len(retain_indices)} rows)", flush=True)
    rng = np.random.default_rng(0)
    retain_n = min(retain_max_samples, len(retain_indices))
    retain_sample = rng.choice(retain_indices, size=retain_n, replace=False).tolist()
    retain_ds = full_ds.select(retain_sample)

    forget_texts = [_build_qa_prompt(tokenizer, r["question"], r["answer"]) for r in forget_ds]
    retain_texts = [_build_qa_prompt(tokenizer, r["question"], r["answer"]) for r in retain_ds]

    # --- Perplexity (diagnostic, not used in model utility) ---
    forget_ppl = get_perplexity(model, tokenizer, forget_texts, prog=prog, name="forget_ppl")
    if prog:
        prog.metric("forget_ppl", round(forget_ppl, 2))
    retain_ppl = get_perplexity(model, tokenizer, retain_texts, prog=prog, name="retain_ppl")
    if prog:
        prog.metric("retain_ppl", round(retain_ppl, 2))

    # --- ROUGE (component 2 of 3 for model utility) ---
    forget_rouge = get_rouge(
        model, tokenizer, forget_ds["question"], forget_ds["answer"],
        max_samples=rouge_max_samples, prog=prog, name="forget_rouge",
    )
    if prog:
        prog.metric("forget_rouge", round(forget_rouge, 4))
    retain_rouge = get_rouge(
        model, tokenizer, retain_ds["question"], retain_ds["answer"],
        max_samples=rouge_max_samples, prog=prog, name="retain_rouge",
    )
    if prog:
        prog.metric("retain_rouge", round(retain_rouge, 4))
    real_rouge = get_rouge(
        model, tokenizer, real_authors["question"], real_authors["answer"],
        max_samples=rouge_max_samples, prog=prog, name="real_rouge",
    )
    if prog:
        prog.metric("real_rouge", round(real_rouge, 4))
    world_rouge = get_rouge(
        model, tokenizer, world_facts["question"], world_facts["answer"],
        max_samples=rouge_max_samples, prog=prog, name="world_rouge",
    )
    if prog:
        prog.metric("world_rouge", round(world_rouge, 4))

    # --- Probability (component 1 of 3 for model utility) ---
    # Retain: P(a|q)^(1/|a|) averaged over samples
    retain_prob = get_answer_probability(
        model, tokenizer, retain_ds["question"], retain_ds["answer"],
        max_samples=rouge_max_samples, prog=prog, name="retain_prob",
    )
    if prog:
        prog.metric("retain_prob", round(retain_prob, 4))
    # Real Authors / World Facts: normalised probability over answer + perturbed (probability_w_options)
    real_prob = get_prob_w_options(
        model, tokenizer, real_authors_pert, prog=prog, name="real_prob",
    )
    if prog:
        prog.metric("real_prob", round(real_prob, 4))
    world_prob = get_prob_w_options(
        model, tokenizer, world_facts_pert, prog=prog, name="world_prob",
    )
    if prog:
        prog.metric("world_prob", round(world_prob, 4))

    # --- Truth Ratio (component 3 of 3 for model utility) ---
    # Per-sample tr_i = wrong/correct; aggregate per open-unlearning. The forget-set raw tr
    # array (forget_tr) is also the statistic compared by the forget-quality KS test below.
    # Forget set: use full_pert (forget10+retain90) so any k-shard config is covered.
    pert_src = full_pert if full_pert is not None else forget10_pert
    forget_qs = set(forget_ds["question"])
    forget_pert_subset = pert_src.filter(lambda r: r["question"] in forget_qs)
    forget_tr = (
        get_truth_ratio_scores(model, tokenizer, forget_pert_subset, correct_key="paraphrased_answer",
                               prog=prog, max_rows=truth_max_rows)
        if len(forget_pert_subset) > 0 else np.array([])
    )
    forget_truth_ratio = tr_forget_agg(forget_tr)          # closer_to_1_better ∈ [0,1]
    if prog:
        prog.metric("forget_truth_ratio", round(forget_truth_ratio, 4))

    # Retain set: truth ratio needs paraphrased+perturbed answers, which only exist in the
    # *_perturbed splits, so compute over the retain portion of full_pert (= retain_perturbed,
    # everything that is not a forget question). This matches open-unlearning's retain Truth_Ratio
    # set and is independent of the sampled retain_ds used for prob/ROUGE.
    if full_pert is not None:
        if retain_author_ids is not None:
            # Same subset restriction for the retain truth-ratio: only the merged
            # authors' questions that the perturbed splits cover. May be EMPTY at
            # small subsets (TOFU's retain_perturbed covers ~2 rows/author on
            # average) -> retain_truth_scaled NaN -> model_utility NaN; read the
            # prob/rouge/ppl components in that case.
            # Exclusion mirrors split_eval_indices: with eval_shard_id set the retain
            # pool is keyed on the GLOBAL forget shard, not the measure shard, so use
            # the same retain_excl_set here (sid=None keeps forget_qs, byte-identical).
            subset_qs = set(full_ds.select(retain_indices)["question"])
            if eval_shard_id is not None:
                retain_excl_qs = set(full_ds.select(sorted(retain_excl_set))["question"])
            else:
                retain_excl_qs = forget_qs
            retain_pert_subset = full_pert.filter(
                lambda r: r["question"] in subset_qs and r["question"] not in retain_excl_qs)
        else:
            retain_pert_subset = full_pert.filter(lambda r: r["question"] not in forget_qs)
        retain_tr = (
            get_truth_ratio_scores(model, tokenizer, retain_pert_subset, correct_key="paraphrased_answer",
                                   prog=prog, max_rows=truth_max_rows)
            if len(retain_pert_subset) > 0 else np.array([])
        )
    else:
        retain_tr = np.array([])
    retain_truth_scaled = tr_nonforget_agg(retain_tr)      # true_better (utility component)
    if prog:
        prog.metric("retain_truth_scaled", round(retain_truth_scaled, 4))

    real_tr = (
        get_truth_ratio_scores(model, tokenizer, real_authors_pert, correct_key="answer",
                               prog=prog, max_rows=truth_max_rows)
        if real_authors_pert is not None else np.array([])
    )
    real_truth_scaled = tr_nonforget_agg(real_tr)
    if prog:
        prog.metric("real_truth_scaled", round(real_truth_scaled, 4))

    world_tr = (
        get_truth_ratio_scores(model, tokenizer, world_facts_pert, correct_key="answer",
                               prog=prog, max_rows=truth_max_rows)
        if world_facts_pert is not None else np.array([])
    )
    world_truth_scaled = tr_nonforget_agg(world_tr)
    if prog:
        prog.metric("world_truth_scaled", round(world_truth_scaled, 4))

    # Raw mean truth ratios (diagnostics only; model utility uses the *_truth_scaled above).
    retain_truth_ratio = float(np.mean(retain_tr)) if len(retain_tr) else float("nan")
    real_truth_ratio = float(np.mean(real_tr)) if len(real_tr) else float("nan")
    world_truth_ratio = float(np.mean(world_tr)) if len(world_tr) else float("nan")

    # --- Forget quality: KS test of the forget-set truth-ratio distribution vs the retain90
    # oracle's (open-unlearning privacy.py:ks_test). p-value; HIGH p = indistinguishable from a
    # model that never trained on the forget data = good unlearning. ---
    if retain_ref_tr_scores is not None and len(forget_tr) > 0 and len(retain_ref_tr_scores) > 0:
        forget_quality = float(ks_2samp(forget_tr, retain_ref_tr_scores).pvalue)
    else:
        forget_quality = float("nan")
    ks_pval = forget_quality   # backward-compatible alias

    # --- Model Utility: harmonic mean of 9 components (open-unlearning utility.py hm_aggregate) ---
    util_names = ["retain_prob", "retain_rouge", "retain_truth_scaled",
                  "real_prob", "real_rouge", "real_truth_scaled",
                  "world_prob", "world_rouge", "world_truth_scaled"]
    util_components = [retain_prob, retain_rouge, retain_truth_scaled,
                       real_prob, real_rouge, real_truth_scaled,
                       world_prob, world_rouge, world_truth_scaled]
    if any(math.isnan(v) for v in util_components):
        nan_names = [nm for nm, v in zip(util_names, util_components) if math.isnan(v)]
        print(f"[model_utility] NaN component(s) {nan_names} -> model_utility=nan", flush=True)
        model_utility = float("nan")
    else:
        # Raw scipy hmean (matches open-unlearning hm_aggregate): a single 0 component -> 0.
        model_utility = float(hmean(util_components))

    return {
        "label": label,
        "forget_ppl": round(forget_ppl, 2),
        "retain_ppl": round(retain_ppl, 2),
        "forget_rouge": round(forget_rouge, 4),
        "retain_rouge": round(retain_rouge, 4),
        "real_rouge": round(real_rouge, 4),
        "world_rouge": round(world_rouge, 4),
        "retain_prob": round(retain_prob, 4),
        "real_prob": round(real_prob, 4),
        "world_prob": round(world_prob, 4),
        "forget_truth_ratio": round(forget_truth_ratio, 4),
        "retain_truth_ratio": round(retain_truth_ratio, 4),
        "real_truth_ratio": round(real_truth_ratio, 4),
        "world_truth_ratio": round(world_truth_ratio, 4),
        "retain_truth_scaled": round(retain_truth_scaled, 4),
        "real_truth_scaled": round(real_truth_scaled, 4),
        "world_truth_scaled": round(world_truth_scaled, 4),
        "forget_quality": round(forget_quality, 4) if not math.isnan(forget_quality) else float("nan"),
        "ks_pval": round(ks_pval, 4) if not math.isnan(ks_pval) else float("nan"),
        "model_utility": round(model_utility, 4) if not math.isnan(model_utility) else float("nan"),
        "metrics_version": METRICS_VERSION,
        "smoke": smoke,
        "extended": extended,
    }


def load_all_shard_adapters(model_name, output_dir, k, lazy_cache=0):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    shard_0 = os.path.join(output_dir, "shard_0")
    model = PeftModel.from_pretrained(base, shard_0, adapter_name="shard_0")
    if lazy_cache:
        return lazify_shard_adapters(model, output_dir, lazy_cache), tokenizer
    for i in range(1, k):
        shard_path = os.path.join(output_dir, f"shard_{i}")
        if not os.path.isdir(shard_path):
            print(f"[load_all_shard_adapters] shard_{i} not found, skipping", flush=True)
            continue
        model.load_adapter(shard_path, adapter_name=f"shard_{i}")
    return model, tokenizer


def lazify_shard_adapters(model, output_dir, cache_cap):
    """High-k eval memory-wall fix: keep at most cache_cap shard adapters resident.

    PEFT fp32-casts every loaded adapter, so k=200 x r32 needs ~65 GiB eagerly — impossible
    on an A40. This patches the PeftModel's set_adapter to load shard_{i} from disk on first
    use (same fp32 cast as the eager path, so numerics are identical) and LRU-evict other
    shards via delete_adapter. Eviction happens AFTER activation so the victim is never the
    active adapter. Missing shard dirs raise (the eager path's silent skip would let a
    routing eval silently serve the base). Only meaningful for per-sample routing labels
    (routed_*/OODAwareRoutedModel), which touch authors dataset-contiguously — each shard
    reloads at most once per metric pass.
    """
    if cache_cap < 1:
        raise ValueError(f"cache_cap must be >= 1, got {cache_cap}")
    from collections import OrderedDict
    lru = OrderedDict((name, None) for name in model.peft_config)
    orig_set_adapter = model.set_adapter

    def lazy_set_adapter(name):
        if not isinstance(name, str):
            raise ValueError("lazy adapter cache supports single-adapter activation only")
        if name not in model.peft_config:
            shard_path = os.path.join(output_dir, name)
            if not os.path.isdir(shard_path):
                raise FileNotFoundError(f"[lazy_adapter_cache] {shard_path} not found")
            model.load_adapter(shard_path, adapter_name=name)
        lru[name] = None
        lru.move_to_end(name)
        orig_set_adapter(name)
        while len(lru) > cache_cap:
            victim = next(n for n in lru if n != name)
            del lru[victim]
            model.delete_adapter(victim)

    model.set_adapter = lazy_set_adapter
    return model


def load_single_adapter(model_name, adapter_dir, adapter_name="jd"):
    """Load base + ONE pre-materialized adapter dir (e.g. a JD-merged keep-set from
    jd_collection). Used for the high-k JD path, where loading all k shards in-model would
    hit the fp32 memory wall — here only the single merged adapter is resident."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter_dir, adapter_name=adapter_name)
    return model, tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--forget_shard_id", type=int, default=None)
    p.add_argument("--eval_shard_id", type=int, default=None,
                   help="Score the forget_* metrics on THIS shard's OWN authors "
                        "(get_author_shard(k, eval_shard_id)) instead of --forget_shard_id's. "
                        "For the merge-mechanism isolated-vs-merged drop study; retain split and "
                        "the forget_quality KS reference are otherwise unchanged.")
    p.add_argument("--out", required=True)
    p.add_argument("--retain_author_ids", default=None,
                   help="Comma-separated author ids: score the retain_* metrics ONLY on "
                        "these authors' rows (subset-conditioned utility — 'did the model "
                        "learn what it was trained on'). Retain truth-ratio is restricted "
                        "to the same authors and may be empty at small subsets (-> NaN "
                        "retain_truth_scaled/model_utility; read prob/rouge/ppl). "
                        "Omit for the unchanged full-population retain split.")
    p.add_argument("--preloaded_adapter", default=None,
                   help="Path to a pre-materialized adapter dir (e.g. a JD mode-B keep-set "
                        "merge from jd_collection). Loads base + this one adapter instead of all "
                        "k shards; --label is used only for naming. Bypasses activate_label.")
    p.add_argument("--lazy_adapter_cache", type=int, default=0,
                   help="High-k memory-wall fix for routed_* labels ONLY: keep at most N shard "
                        "adapters resident; set_adapter loads from disk on demand + LRU-evicts "
                        "(lazify_shard_adapters). 0 (default) = eager load of all k shards "
                        "(unchanged legacy behavior). Use e.g. 8 for k=200 r32 on an A40.")
    p.add_argument("--prefix_pool_dir", default=None,
                   help="peft_compose prefix arm: dir of per-shard PREFIX-TUNING adapters "
                        "(shard_0..k-1). Serves base + ALL shards' KV prefixes concatenated "
                        "(prefix_concat.PrefixConcatModel); labels prefixcat_full / "
                        "prefixcat_unlearn. Bypasses load_all_shard_adapters/activate_label.")
    p.add_argument("--prefix_exclude_shard", type=int, default=None,
                   help="With --prefix_pool_dir: drop this shard's prefix segment from the "
                        "concatenation (the exact O(1) deletion condition).")
    p.add_argument("--legonet_config", default=None,
                   help="Path to a LegoNet-TOFU config JSON. Loads the frozen base + n cluster "
                        "adapters and serves them via per-query top-k 1/k delta-average "
                        "(legonet_model.LegoNetRoutedModel); --label names the row "
                        "(legonet_full / legonet_unlearn). Bypasses load_all_shard_adapters.")
    p.add_argument("--legonet_unlearn_tag", default=None,
                   help="With --legonet_config: assemble the post-unlearn model from the "
                        "unlearn/{tag} retrains of affected adapters (originals otherwise). "
                        "Omit for the full (pre-deletion) model.")
    p.add_argument("--ramole_router", default=None,
                   help="With --legonet_config: serve the same TOFU experts via RAMoLE — the trained "
                        "RouterLoRA cross-attention (this safetensors) instead of the 1/k average "
                        "(ramole_tofu.RamoleTofuModel). --label e.g. ramole_full / ramole_unlearn.")
    p.add_argument("--ramole_route", default="embed", choices=["embed", "key"],
                   help="RAMoLE routing: embed = LoRA-retriever cosine (RAG, default); "
                        "key = LegoNet author-key lookup (comparison arm).")
    p.add_argument("--ramole_index", default="stale", choices=["stale", "rebuilt"],
                   help="Embed-route index policy: stale = as-built (includes forget authors; the "
                        "encoder-centroid-leak arm); rebuilt = member means excluding forget_authors "
                        "(labels ramolerb_*).")
    p.add_argument("--sift_masks_config", default=None,
                   help="SIFT-Masks arm: serve θ0 + (τ̄⊙m_a)/T per query "
                        "(sift_masks_model.SiftMasksModel), bypassing load_all_shard_adapters. "
                        "--label names the row: sift_full / sift_unlearn (masked) or "
                        "merge_full / merge_unlearn (FT+Merge no-mask baseline).")
    p.add_argument("--sift_unlearn_tag", default=None,
                   help="With --sift_masks_config: serve the post-deletion model (τ̄_<tag> over "
                        "retain authors). Omit for the full (pre-deletion) model.")
    p.add_argument("--clamu_config", default=None,
                   help="ClAMU arm: serve θ0 + (m_c⊙τ̄)/T per query with per-cluster masks "
                        "(clamu_model.ClamuModel). --label sets the mask family + condition: "
                        "clamu_ / emr_ / tall_ (masked) or merge_ (no-mask Global) × _full/_unlearn.")
    p.add_argument("--clamu_unlearn_tag", default=None,
                   help="With --clamu_config: serve the post-deletion retain model "
                        "(tau_bar_<tag> + assignment_<tag> + masks_<tag>). Omit for the full model.")
    p.add_argument("--memsinks_config", default=None,
                   help="MemSinks arm: path to a memsinks_tofu training config JSON. Serves the "
                        "trained SeqTD adapter with per-query ROUTED author masks (gen + own sink "
                        "slice; OOD -> gen-only) via memsinks_routed_model.MemSinksRoutedModel — "
                        "the project dir is sys.path'd from the config path. Labels "
                        "memsinks_routed_full / memsinks_routed_unlearn. Bypasses "
                        "load_all_shard_adapters/activate_label.")
    p.add_argument("--memsinks_unlearn_tag", default=None,
                   help="With --memsinks_config: 'forget10' serves cfg[forget_authors] gen-only "
                        "(their sink slices never applied — the routed view of the baked "
                        "deletion). Omit for the full (pre-deletion) routed model.")
    p.add_argument("--linear_tv_config", default=None,
                   help="composable_tv [lin] arm: path to a ctv config JSON "
                        "(configs/ctv_1b_lin.json). Serves the LINEARIZED composition "
                        "f0 + J.(sum tau_a) over the selected authors' tangent-space adapters "
                        "(linear_tv.LinearTVModel), bypassing load_all_shard_adapters/"
                        "activate_label. Select authors via --linear_tv_authors or "
                        "--linear_tv_n; --label names the row (ctv_lin_* / iso_a*).")
    p.add_argument("--linear_tv_authors", default=None,
                   help="With --linear_tv_config: comma-separated author ids to compose "
                        "(weight +1 each; cfg compose='mean' rescales to 1/N).")
    p.add_argument("--linear_tv_n", type=int, default=None,
                   help="With --linear_tv_config: compose the FIRST N authors of the config "
                        "pool (merge_subset.subset_authors(pool_seed, N)). Mutually "
                        "exclusive with --linear_tv_authors.")
    p.add_argument("--linear_tv_subtract", default=None,
                   help="With --linear_tv_config: comma-separated author ids composed at "
                        "NEGATIVE weight (tangent subtraction — the O(1) unlearning op).")
    p.add_argument("--linear_tv_serve", default="linear", choices=["linear", "nonlinear-debug"],
                   help="With --linear_tv_config: linear = tangent-space serve (jvp, the "
                        "method); nonlinear-debug = theta0 + sum(tau) baked directly into "
                        "the weights (the 2x2 fallback arm).")
    p.add_argument("--ds_config", default=None,
                   help="composable_tv [ds] arm: path to a ctv ds config JSON "
                        "(configs/ctv_1b_ds.json). Serves the MERGED disjoint-support "
                        "full model theta0 + sum(tau_a) over the selected authors "
                        "(ds_support.load_ds_eval_model — one dense model, no adapter, "
                        "no per-query routing), bypassing load_all_shard_adapters/"
                        "activate_label. Select authors via --ds_authors or --ds_n; "
                        "--label names the row (ctv_ds_* / iso_a*).")
    p.add_argument("--ds_authors", default=None,
                   help="With --ds_config: comma-separated author ids to compose "
                        "(+tau_a each; must be trained pool members).")
    p.add_argument("--ds_n", type=int, default=None,
                   help="With --ds_config: compose the FIRST N authors of the config "
                        "pool (merge_subset.subset_authors(pool_seed, N)). Mutually "
                        "exclusive with --ds_authors.")
    p.add_argument("--ds_subtract", default=None,
                   help="With --ds_config: comma-separated pool author ids SUBTRACTED "
                        "from the merge (bitwise identical to composing without them — "
                        "the O(1) deletion condition).")
    p.add_argument("--hf_home", default=os.environ.get("HF_HOME") or _tofu_env.hf_home())
    p.add_argument("--merge_num_examples", type=int, default=256,
                   help="Data-required merges (regmean/fisher/lorahub): examples in the shared "
                        "merge dataloader AND the per-shard Gram/Fisher cap (default 256 = "
                        "historical behavior). At k=200 the default costs k*256 passes per "
                        "merge — reduce (e.g. 32) and record the deviation in the log entry.")
    p.add_argument("--rouge_max_samples", type=int, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Fast smoke eval (~1h/model): cap ROUGE/retain/truth/KS samples",
    )
    p.add_argument(
        "--extended",
        action="store_true",
        help="Extended eval (~2.5h/task): larger caps, full forget KS from prepare",
    )
    return p.parse_args()


def build_served_model(args, data, forget_id, prog=None):
    """Construct the SERVED model composition for an eval label — the single source of truth
    for what "the served (post-deletion) system" is. Shared by eval_tofu.main and the
    deletion-audit attack scripts (attack_mia.py et al.), so an attack measures exactly the
    artifact the metrics score. Returns (eval_model, tokenizer, adapter_name).

    `args` needs the parse_args() fields: model_name, output_dir, label, k, plus the arm
    selectors (preloaded_adapter / legonet_config[+legonet_unlearn_tag, ramole_*] /
    sift_masks_config[+sift_unlearn_tag] / clamu_config[+clamu_unlearn_tag] /
    linear_tv_config[+linear_tv_authors/n/subtract/serve] /
    ds_config[+ds_authors/n/subtract])."""
    def _step(stage, msg):
        if prog is not None:
            prog.step(stage, msg)

    # LOUD conflict guard: the elif chain below silently prefers whichever arm flag
    # comes first (--preloaded_adapter + --linear_tv_config used to serve the plain
    # nonlinear adapter with no warning). Exactly ONE arm selector may be set.
    # getattr(.., None): mirrored namespaces (attack_mia) may predate newer flags.
    _arm_flags = ("preloaded_adapter", "prefix_pool_dir", "legonet_config",
                  "sift_masks_config", "memsinks_config", "clamu_config",
                  "linear_tv_config", "ds_config")
    _set = [f for f in _arm_flags if getattr(args, f, None)]
    if len(_set) > 1:
        raise ValueError(
            "conflicting served-model selectors: "
            + ", ".join(f"--{f}" for f in _set)
            + " — pass exactly one arm flag (ramole_router is a legonet sub-flag, "
              "not a selector)")

    if args.preloaded_adapter:
        _step("load_model", f"{args.model_name} + preloaded {args.preloaded_adapter}")
        model, tokenizer = load_single_adapter(args.model_name, args.preloaded_adapter)
        eval_model = model
        adapter_name = args.label
    elif args.prefix_pool_dir:
        from prefix_concat import load_prefix_concat_model
        excl = [] if args.prefix_exclude_shard is None else [args.prefix_exclude_shard]
        _step("load_model", f"{args.model_name} + prefix pool {args.prefix_pool_dir}"
                            f"{f' minus shard_{excl[0]}' if excl else ''}")
        eval_model, tokenizer = load_prefix_concat_model(
            args.model_name, args.prefix_pool_dir, k=args.k, exclude=excl)
        adapter_name = args.label
    elif args.legonet_config:
        from legonet_model import load_legonet_eval_model
        from legonet_tofu import load_config as load_legonet_config
        tag = args.legonet_unlearn_tag
        _step("load_model", f"{args.model_name} + legonet {args.legonet_config}"
                            f"{f' unlearn:{tag}' if tag else ''}")
        cfg_l = load_legonet_config(args.legonet_config)
        if args.ramole_router:
            from ramole_tofu import load_ramole_eval_model
            eval_model, tokenizer = load_ramole_eval_model(
                cfg_l, data["full"], args.ramole_router,
                route_mode=args.ramole_route, unlearn_tag=tag,
                index_policy=args.ramole_index)
        else:
            eval_model, tokenizer = load_legonet_eval_model(cfg_l, data["full"], unlearn_tag=tag)
        adapter_name = args.label
    elif args.sift_masks_config:
        from sift_masks_model import load_sift_eval_model
        from legonet_tofu import load_config as load_sift_config
        tag = args.sift_unlearn_tag
        _step("load_model", f"{args.model_name} + sift {args.sift_masks_config}"
                            f"{f' unlearn:{tag}' if tag else ''}")
        cfg_s = load_sift_config(args.sift_masks_config)
        # merge_* labels = FT+Merge no-mask baseline; sift_* labels = masked serve.
        eval_model, tokenizer = load_sift_eval_model(
            cfg_s, data["full"], unlearn_tag=tag,
            baseline=args.label.startswith("merge_"))
        adapter_name = args.label
    elif args.memsinks_config:
        # MemSinks routed-mask arm: the wrapper lives in the memsinks_tofu project;
        # derive its dir from the config path (configs/ is one level below the project).
        cfg_path = os.path.abspath(args.memsinks_config)
        proj_dir = os.path.dirname(os.path.dirname(cfg_path))
        if proj_dir not in sys.path:
            sys.path.insert(0, proj_dir)
        from memsinks_routed_model import load_memsinks_eval_model
        tag = args.memsinks_unlearn_tag
        _step("load_model", f"{args.model_name} + memsinks routed {args.memsinks_config}"
                            f"{f' unlearn:{tag}' if tag else ''}")
        with open(cfg_path) as f:
            cfg_m = json.load(f)
        eval_model, tokenizer = load_memsinks_eval_model(cfg_m, data["full"], unlearn_tag=tag)
        adapter_name = args.label
    elif args.clamu_config:
        from clamu_model import load_clamu_eval_model
        from train_sift_masks import load_config as load_clamu_config
        tag = args.clamu_unlearn_tag
        _step("load_model", f"{args.model_name} + clamu {args.clamu_config}"
                            f"{f' unlearn:{tag}' if tag else ''}")
        cfg_c = load_clamu_config(args.clamu_config)
        # label prefix selects the mask family: clamu_/emr_/tall_ (masked) or merge_ (Global).
        eval_model, tokenizer = load_clamu_eval_model(
            cfg_c, data["full"], args.label, unlearn_tag=tag)
        adapter_name = args.label
    elif getattr(args, "linear_tv_config", None):
        # composable_tv [lin] arm. getattr guard: attack_mia namespaces predating these
        # flags must keep working through this shared builder unchanged.
        from linear_tv import load_linear_tv_eval_model
        _step("load_model", f"{args.model_name} + linear_tv {args.linear_tv_config} "
                            f"serve:{args.linear_tv_serve}")
        with open(args.linear_tv_config) as f:
            cfg_lt = json.load(f)
        eval_model, tokenizer = load_linear_tv_eval_model(
            cfg_lt, authors=args.linear_tv_authors, n=args.linear_tv_n,
            subtract=args.linear_tv_subtract, serve=args.linear_tv_serve)
        adapter_name = args.label
    elif getattr(args, "ds_config", None):
        # composable_tv [ds] arm: ONE merged disjoint-support full model (theta0 +
        # sum tau_a − sum subtract), built in memory from the stored sparse taus —
        # no adapter, no routing (merge-only serving is the method).
        from ds_support import load_ds_eval_model
        _step("load_model", f"{args.model_name} + ds_support {args.ds_config}")
        with open(args.ds_config) as f:
            cfg_ds = json.load(f)
        eval_model, tokenizer = load_ds_eval_model(
            cfg_ds, authors=args.ds_authors, n=args.ds_n, subtract=args.ds_subtract)
        adapter_name = args.label
    else:
        _step("load_model", args.model_name)
        lazy_cache = getattr(args, "lazy_adapter_cache", 0)
        if lazy_cache and not args.label.startswith("routed_"):
            # merge/ensemble/shard labels enumerate the full adapter set; only per-sample
            # routing touches one adapter at a time, so lazy is unsound anywhere else.
            raise ValueError(f"--lazy_adapter_cache only supports routed_* labels, got {args.label}")
        model, tokenizer = load_all_shard_adapters(args.model_name, args.output_dir, args.k,
                                                   lazy_cache=lazy_cache)
        _step("activate_adapter", args.label)
        centroid_cache = os.path.join(args.output_dir, "centroids")
        merge_dataloader = None
        if label_requires_data(args.label):
            exclude = forget_id if args.label.startswith("remerge_") else None
            merge_dataloader = build_merge_dataloader(
                data["full"], tokenizer, args.k, exclude_shard=exclude,
                num_examples=getattr(args, "merge_num_examples", 256),
            )
        result = activate_label(
            model, args.k, forget_id, args.label,
            tokenizer=tokenizer,
            dataset=data["full"] if args.label.startswith("routed_") else None,
            centroid_cache_dir=centroid_cache,
            dataloader=merge_dataloader,
            num_regmean_examples=getattr(args, "merge_num_examples", 256),
            output_dir=args.output_dir,
        )
        if isinstance(result, str):
            model.set_adapter(result)
            eval_model = model
            adapter_name = result
        else:
            eval_model = result   # RoutedModel
            adapter_name = args.label
    return eval_model, tokenizer, adapter_name


def main():
    args = parse_args()
    forget_id = args.forget_shard_id if args.forget_shard_id is not None else args.k - 1
    if args.eval_shard_id is not None and not (0 <= args.eval_shard_id < args.k):
        raise SystemExit(f"--eval_shard_id {args.eval_shard_id} out of range [0,{args.k})")
    retain_author_ids = None
    if args.retain_author_ids:
        retain_author_ids = sorted({int(x) for x in args.retain_author_ids.split(",") if x.strip()})
        if any(not (0 <= a < 200) for a in retain_author_ids):
            raise SystemExit(f"--retain_author_ids out of range [0,200): {retain_author_ids}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    progress_path = args.out.replace(".json", ".progress.json")

    prog = ProgressLogger(progress_path, args.label)
    prog.step("load_data", "TOFU datasets")

    data = load_tofu_data(args.hf_home)
    shards = {i: get_author_shard(args.k, i) for i in range(args.k)}

    if args.smoke and args.extended:
        raise SystemExit("Use only one of --smoke or --extended")
    results_sub = "smoke" if args.smoke else ("extended" if args.extended else "")
    results_dir = os.path.join(args.output_dir, "results", results_sub)
    retain_tr_path = os.path.join(results_dir, "retain_tr_scores.npy")
    retain_ref_tr_scores = np.load(retain_tr_path) if os.path.exists(retain_tr_path) else None
    if retain_ref_tr_scores is None:
        print(f"[eval] no retain_tr_scores.npy in {results_dir} -> forget_quality will be NaN "
              f"(run prepare_eval.py to cache the retain90 reference)", flush=True)

    eval_model, tokenizer, adapter_name = build_served_model(args, data, forget_id, prog=prog)

    smoke = args.smoke
    extended = args.extended
    rouge_n = args.rouge_max_samples
    retain_n, truth_n = 500, None
    if smoke:
        rouge_n = rouge_n if rouge_n is not None else SMOKE_ROUGE_MAX
        retain_n = SMOKE_RETAIN_MAX
        truth_n = SMOKE_TRUTH_MAX
        prog.step("evaluate", f"SMOKE metrics (rouge<={rouge_n}, retain<={retain_n})")
    elif extended:
        rouge_n = rouge_n if rouge_n is not None else EXTENDED_ROUGE_MAX
        retain_n = EXTENDED_RETAIN_MAX
        truth_n = EXTENDED_TRUTH_MAX
        prog.step("evaluate", f"EXTENDED metrics (rouge<={rouge_n}, retain<={retain_n})")
    else:
        prog.step("evaluate", "metrics")

    row = evaluate_model(
        eval_model, tokenizer, args.label,
        forget_shard_id=forget_id,
        full_ds=data["full"],
        shards=shards,
        forget10_pert=data["forget10_pert"],
        real_authors=data["real_authors"],
        world_facts=data["world_facts"],
        retain_ref_tr_scores=retain_ref_tr_scores,
        rouge_max_samples=rouge_n,
        prog=prog,
        smoke=smoke,
        extended=extended,
        retain_max_samples=retain_n,
        truth_max_rows=truth_n,
        full_pert=data["full_pert"],
        real_authors_pert=data["real_authors_pert"],
        world_facts_pert=data["world_facts_pert"],
        eval_shard_id=args.eval_shard_id,
        retain_author_ids=retain_author_ids,
    )
    row["model_name"] = args.model_name
    row["adapter"] = adapter_name
    row["eval_shard_id"] = args.eval_shard_id
    row["retain_author_ids"] = retain_author_ids

    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)
    prog.done(row)
    print(f"Wrote {args.out}", flush=True)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
