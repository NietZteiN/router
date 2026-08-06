"""TOFU data for SIFT-Masks: per-author tasks, GPT-2 prompt, answer-span loss mask.

Matches the repo's existing TOFU convention (train_lora_shard.py):
  * author `a` occupies rows [a*20, (a+1)*20) of the `locuslab/TOFU` "full" split;
  * prompt format `Question: {q}\nAnswer: {a}`.

Each SIFT task is one author's 20 Q&A pairs, fed as a single full batch (bs=20),
matching the paper (App B: "batch size of 20 (each task contains exactly 20
examples)").

Loss masking: by default we train on the ANSWER span only (the question +
"Answer:" prefix tokens are set to -100). This mirrors how the paper's other
datasets predict only the label span; it is an inference for TOFU (the canonical
TOFU benchmark finetunes on the full Q+A), so `loss_on="full"` is available to
reproduce the full-sequence objective instead. See CLAUDE_SCRATCHPAD.md.
"""
from __future__ import annotations

import os
from typing import Dict, List

import torch

import sys

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

RECORDS_PER_AUTHOR = 20
NUM_AUTHORS = 200

# TOFU forget10 = the last 20 authors (matches the repo invariant shard_id=k-1).
FORGET10_AUTHORS = list(range(180, 200))


def load_tofu_full(hf_home: str):
    os.environ["HF_HOME"] = hf_home
    from datasets import load_dataset
    return load_dataset("locuslab/TOFU", "full")["train"]


def author_records(full, author: int) -> List[Dict[str, str]]:
    """The 20 {question, answer} dicts for one author."""
    start = author * RECORDS_PER_AUTHOR
    rows = full.select(range(start, start + RECORDS_PER_AUTHOR))
    return [{"question": r["question"], "answer": r["answer"]} for r in rows]


def _prompt_text(q: str) -> str:
    return f"Question: {q}\nAnswer:"


def _full_text(q: str, a: str) -> str:
    return f"Question: {q}\nAnswer: {a}"


def build_task_batch(
    tokenizer,
    records: List[Dict[str, str]],
    *,
    loss_on: str = "answer",
    max_length: int = 256,
    append_eos: bool = True,
) -> Dict[str, torch.Tensor]:
    """Tokenize an author's records into one right-padded full batch with labels.

    labels are -100 on padding and (for loss_on="answer") on the question +
    "Answer:" prefix, so cross-entropy is taken over the answer tokens only.
    """
    assert loss_on in ("answer", "full")
    eos = tokenizer.eos_token or ""
    input_rows, label_rows = [], []
    for r in records:
        full = _full_text(r["question"], r["answer"]) + (eos if append_eos else "")
        full_ids = tokenizer(full, add_special_tokens=False, truncation=True,
                             max_length=max_length)["input_ids"]
        labels = list(full_ids)
        if loss_on == "answer":
            prompt_ids = tokenizer(_prompt_text(r["question"]), add_special_tokens=False,
                                   truncation=True, max_length=max_length)["input_ids"]
            n_prefix = min(len(prompt_ids), len(full_ids))
            labels[:n_prefix] = [-100] * n_prefix
        input_rows.append(full_ids)
        label_rows.append(labels)

    width = max(len(r) for r in input_rows)
    pad_id = tokenizer.pad_token_id
    input_ids, attn, lab = [], [], []
    for ids, lb in zip(input_rows, label_rows):
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        lab.append(lb + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(lab, dtype=torch.long),
    }


def load_gpt2_tokenizer(model_name: str, hf_home: str):
    os.environ["HF_HOME"] = hf_home
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token        # GPT-2 has no pad token by default
        tok.pad_token_id = tok.eos_token_id
    return tok
