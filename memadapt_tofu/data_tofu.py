"""TOFU data pipeline with exact open-unlearning parity.

Ports the chat branch of open-unlearning `src/data/utils.py::preprocess_chat_instance`
and `src/data/collators.py::DataCollatorForSupervisedDataset` (commit 4ad738a),
plus the pad_token=eos fallback from `src/model/__init__.py:98-100`. The
"Finetuned" comparison row is OU's released checkpoint, so tokenization must
match theirs token-for-token — deliberately including two quirks:

  1. `max_length` is NOT enforced on the chat path upstream (dead config).
     We replicate that and instead assert observed length < max_length once.
  2. The collator recomputes attention_mask = input_ids.ne(pad_token_id) with
     pad == eos, which also zeroes the mask on the *real* trailing eos token.
     That is OU's served behavior; do not "fix" it.

Extension over OU: each example carries `source_ids` (author id = row // 20 on
the ordered 'full' split) for sequence-level gradient masking.
"""

from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from memadapt_common import RECORDS_PER_AUTHOR, author_of_row

IGNORE_INDEX = -100

# From open-unlearning configs/model/Llama-3.2-1B-Instruct.yaml (template_args).
TEMPLATE_ARGS = {
    "apply_chat_template": True,
    "system_prompt": "You are a helpful assistant.",
    "date_string": "10 Apr 2025",
}


def prepare_tokenizer(tokenizer):
    """OU convention: pad with eos when the tokenizer defines no pad token."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def preprocess_chat_instance(
    tokenizer,
    question: str,
    answer: str,
    template_args: dict = None,
) -> Dict[str, torch.Tensor]:
    """Single-QA port of OU's preprocess_chat_instance (chat branch, no few-shot)."""
    template_args = template_args or TEMPLATE_ARGS
    chat = []
    if template_args.get("system_prompt"):
        chat.append({"role": "system", "content": template_args["system_prompt"]})
    chat.append({"role": "user", "content": question})
    chat.append({"role": "assistant", "content": answer})

    date_str = template_args.get("date_string")
    date_info = {"date_string": date_str} if date_str is not None else {}
    chat_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False, **date_info
    )
    prompt_ids = tokenizer.apply_chat_template(
        chat[:-1], tokenize=True, add_generation_prompt=True, **date_info
    )
    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids = chat_ids + [tokenizer.eos_token_id]

    len_prompt = len(prompt_ids)
    labels = [IGNORE_INDEX] * len_prompt + chat_ids[len_prompt:]
    return {
        "input_ids": torch.tensor(chat_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.ones(len(chat_ids), dtype=torch.long),
    }


class TofuQADataset(Dataset):
    """locuslab/TOFU QA dataset with OU-parity tokenization + source ids."""

    def __init__(self, tokenizer, split: str = "full", max_length: int = 512,
                 template_args: dict = None, hf_home: str = None):
        import datasets  # test-env only; eval path never constructs datasets

        self.tokenizer = prepare_tokenizer(tokenizer)
        self.template_args = template_args or TEMPLATE_ARGS
        self.max_length = max_length
        self.split = split
        self.data = datasets.load_dataset(
            "locuslab/TOFU", name=split, split="train", cache_dir=None
        )
        if split == "full":
            expected = 200 * RECORDS_PER_AUTHOR
            assert len(self.data) == expected, (
                f"TOFU full split has {len(self.data)} rows, expected {expected}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        item = preprocess_chat_instance(
            self.tokenizer, row["question"], row["answer"], self.template_args
        )
        n = len(item["input_ids"])
        assert n < self.max_length, (
            f"row {idx}: {n} tokens >= max_length {self.max_length}; the OU chat "
            "path never truncates, so this breaks the documented no-op assumption"
        )
        item["index"] = idx
        # Source id is only meaningful on the ordered full split.
        item["source_ids"] = author_of_row(idx) if self.split == "full" else -1
        return item


class QACollatorWithSources:
    """OU DataCollatorForSupervisedDataset (right padding) + source_ids passthrough."""

    def __init__(self, tokenizer):
        self.tokenizer = prepare_tokenizer(tokenizer)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [ins["input_ids"] for ins in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [ins["labels"] for ins in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {
            "input_ids": input_ids,
            # OU quirk replicated: ne(pad) also masks the real trailing eos.
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "labels": labels,
            "source_ids": torch.tensor(
                [ins["source_ids"] for ins in instances], dtype=torch.long
            ),
            "index": torch.tensor([ins["index"] for ins in instances], dtype=torch.long),
        }


def verify_forget_author_mapping(forget_split: str = "forget10") -> List[int]:
    """Map a forget split to whole-author ids by exact question+answer text join.

    Never assume forget10 == the last 20 authors: the join is the ground truth.
    Asserts every matched author is *fully* covered by the forget split.
    Returns the sorted author ids to feed build_blocklist.py.
    """
    import datasets

    full = datasets.load_dataset("locuslab/TOFU", name="full", split="train")
    forget = datasets.load_dataset("locuslab/TOFU", name=forget_split, split="train")

    key_to_row = {
        (q, a): i for i, (q, a) in enumerate(zip(full["question"], full["answer"]))
    }
    rows = []
    for q, a in zip(forget["question"], forget["answer"]):
        assert (q, a) in key_to_row, f"forget row not found in full split: {q[:60]!r}"
        rows.append(key_to_row[(q, a)])

    authors = sorted({author_of_row(r) for r in rows})
    covered = {r for a in authors for r in range(a * RECORDS_PER_AUTHOR,
                                                 (a + 1) * RECORDS_PER_AUTHOR)}
    assert covered == set(rows), (
        f"{forget_split} does not align to whole authors: "
        f"{len(set(rows))} rows vs {len(covered)} covered by {len(authors)} authors"
    )
    return authors
