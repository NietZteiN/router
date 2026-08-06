"""Light per-author scorer for the relearn protocol (importable + CLI).

Given a model+tokenizer and a list of (question, answer) pairs, produces the
three relearn-curve metrics:

  prob  — open-unlearning's answer-probability formula, reimplemented INLINE
          from src/evals/metrics/utils.py::evaluate_probability (lines 82-99).
          We must not import OU: this scorer runs in test-env while OU's
          package needs hydra and its own env. The formula quirks are kept
          verbatim — CE summed over SHIFTED labels but normalized by the
          non-ignore count of the UNSHIFTED labels (equal here because
          labels[0] is always IGNORE on the chat template), and attention_mask
          = input_ids.ne(pad) with pad==eos (masks real eot tokens — OU's
          served behavior, replicated on purpose by data_tofu's collator).
          Logits are cast to fp32 before CE, matching the deliberate
          fp32-logits fix OU's evals are served with.
  rouge — ROUGE-L RECALL of a greedy generation against the gold answer, OU's
          eval_text_similarity convention: prompt-only chat ids with
          add_generation_prompt, LEFT padding, do_sample=False /
          temperature=None / top_p=None / max_new_tokens=200 / use_cache=True
          (configs/generation/default.yaml), pad_token_id=eos, decode the
          continuation with skip_special_tokens, rouge_scorer(["rougeL"],
          use_stemmer=True).score(gold, gen).recall.
  ppl   — exp(mean over rows of the per-row avg answer-token CE).

Tokenization goes through memadapt's data_tofu (imported in place — the single
OU-parity source, pinned by its own tests); NEVER the plain Question:/Answer:
format, which is a different comparison track.
"""

import argparse
import json
import math
import os

import torch
from torch import nn

from sepmlp_common import NO_AUTHOR, import_memadapt_data, load_config, save_json

IGNORE_INDEX = -100

_ROUGE_SCORER = None


def rouge_recall(gold: str, generation: str) -> float:
    """OU convention: rouge_scorer(["rougeL"], use_stemmer=True), score(gt, gen),
    take .recall (how much of the gold answer the generation reproduces)."""
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(_ROUGE_SCORER.score(gold, generation)["rougeL"].recall)


def chat_prompt_ids(tokenizer, question: str, template_args: dict = None):
    """Prompt-only ids (system + user + generation header), the generation-side
    twin of data_tofu.preprocess_chat_instance's prompt branch."""
    data_tofu = import_memadapt_data()
    template_args = template_args or data_tofu.TEMPLATE_ARGS
    chat = []
    if template_args.get("system_prompt"):
        chat.append({"role": "system", "content": template_args["system_prompt"]})
    chat.append({"role": "user", "content": question})
    date_str = template_args.get("date_string")
    date_info = {"date_string": date_str} if date_str is not None else {}
    return tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, **date_info
    )


def _left_pad(seqs, padding_value):
    """OU's DataCollatorForSupervisedDataset left-padding (flip / pad / flip)."""
    return torch.nn.utils.rnn.pad_sequence(
        [torch.flip(s, dims=[0]) for s in seqs],
        batch_first=True,
        padding_value=padding_value,
    ).flip(dims=[1])


def _forward_batches(tokenizer, qa_pairs, batch_size):
    """Teacher-forced batches via the OU-parity pipeline (right padding,
    answer-only labels, ne(pad) attention). source_ids/index are collator
    plumbing only and are popped before the model sees the batch."""
    data_tofu = import_memadapt_data()
    tokenizer = data_tofu.prepare_tokenizer(tokenizer)
    collator = data_tofu.QACollatorWithSources(tokenizer)
    items = []
    for i, (q, a) in enumerate(qa_pairs):
        item = data_tofu.preprocess_chat_instance(tokenizer, q, a)
        item["index"] = i
        item["source_ids"] = NO_AUTHOR
        items.append(item)
    for s in range(0, len(items), batch_size):
        batch = collator(items[s:s + batch_size])
        batch.pop("source_ids")
        batch.pop("index")
        yield batch


@torch.no_grad()
def evaluate_probability(model, tokenizer, qa_pairs, batch_size: int = 4) -> dict:
    """Inline port of OU evaluate_probability, aggregated over qa_pairs."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    per_row = []
    for batch in _forward_batches(tokenizer, qa_pairs, batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        logits = output.logits.float()
        labels = batch["labels"]
        shifted_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()
        loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
        losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
        num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
        avg_losses = losses / num_token_gt
        normalized_probs = torch.exp(-avg_losses)
        per_row += [
            {"prob": float(p), "avg_loss": float(l)}
            for p, l in zip(normalized_probs, avg_losses)
        ]
    if was_training:
        model.train()
    avg_loss = sum(r["avg_loss"] for r in per_row) / len(per_row)
    return {
        "prob": sum(r["prob"] for r in per_row) / len(per_row),
        "avg_loss": avg_loss,
        "ppl": math.exp(avg_loss),
        "per_row": per_row,
    }


@torch.no_grad()
def evaluate_rouge(model, tokenizer, qa_pairs, batch_size: int = 4,
                   max_new_tokens: int = 200) -> dict:
    """Greedy generation from the chat prompt, ROUGE-L recall vs gold answer."""
    data_tofu = import_memadapt_data()
    tokenizer = data_tofu.prepare_tokenizer(tokenizer)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    per_row = []
    for s in range(0, len(qa_pairs), batch_size):
        chunk = qa_pairs[s:s + batch_size]
        prompts = [torch.tensor(chat_prompt_ids(tokenizer, q)) for q, _ in chunk]
        input_ids = _left_pad(prompts, tokenizer.pad_token_id).to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_texts = tokenizer.batch_decode(
            output[:, input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        for (q, a), gen in zip(chunk, gen_texts):
            gen = gen.strip()
            per_row.append({"rougeL_recall": rouge_recall(a, gen), "generation": gen})
    if was_training:
        model.train()
    mean = sum(r["rougeL_recall"] for r in per_row) / len(per_row)
    return {"rougeL_recall": mean, "per_row": per_row}


def score_author(model, tokenizer, qa_pairs, batch_size: int = 4,
                 max_new_tokens: int = 200) -> dict:
    """All three curve metrics on one author's QA rows."""
    p = evaluate_probability(model, tokenizer, qa_pairs, batch_size=batch_size)
    r = evaluate_rouge(model, tokenizer, qa_pairs, batch_size=batch_size,
                       max_new_tokens=max_new_tokens)
    return {
        "prob": p["prob"],
        "avg_loss": p["avg_loss"],
        "ppl": p["ppl"],
        "rougeL_recall": r["rougeL_recall"],
        "prob_rows": p["per_row"],
        "rouge_rows": r["per_row"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--serve", default="hf", choices=["sepmlp", "memadapt", "hf"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--droplist", default=None)
    ap.add_argument("--blocklist", default=None)
    ap.add_argument("--split", default="full", choices=["full", "holdout10"])
    ap.add_argument("--author", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["hf_home"])
    # Lazy import: relearn.py owns model serving and split loading; relearn
    # imports this module at its top, so importing it back here at module
    # level would be circular.
    from relearn import author_qa_pairs, load_served_model, load_split_qa

    from transformers import AutoTokenizer

    data_tofu = import_memadapt_data()
    tokenizer = data_tofu.prepare_tokenizer(
        AutoTokenizer.from_pretrained(cfg["model_name"])
    )
    model = load_served_model(
        args.serve, cfg["model_name"], checkpoint=args.checkpoint,
        droplist=args.droplist, blocklist=args.blocklist,
    )
    qa_pairs = author_qa_pairs(load_split_qa(args.split), args.author)
    scores = score_author(
        model, tokenizer, qa_pairs,
        batch_size=args.batch_size or cfg["score"]["batch_size"],
        max_new_tokens=args.max_new_tokens or cfg["score"]["max_new_tokens"],
    )
    scalars = {k: v for k, v in scores.items() if not k.endswith("_rows")}
    print(f"[score] serve={args.serve} split={args.split} author={args.author} "
          + json.dumps(scalars))
    if args.out:
        save_json({"serve": args.serve, "checkpoint": args.checkpoint,
                   "split": args.split, "author": args.author, **scores},
                  args.out)


if __name__ == "__main__":
    main()
