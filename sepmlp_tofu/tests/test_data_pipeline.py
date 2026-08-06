"""Data-pipeline CPU gates (plan gate 11 data parity, gate 13 forget10
text-join mapping, plus the static holdout-contamination gate).

Uses the OFFLINE HF cache only (env pinned in conftest); every test needing
datasets/TOFU skips cleanly when the cache is unavailable rather than
downloading. The static contamination gate needs no data at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from conftest import PAD_ID  # noqa: F401  (env side-effects + shared consts)
from sepmlp_common import (  # noqa: E402
    MEMADAPT_DIR,
    NUM_AUTHORS,
    RECORDS_PER_AUTHOR,
    import_memadapt_data,
)

data_tofu = import_memadapt_data()

SEPMLP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _skip_if_offline(fn, what):
    """Run fn(); on any non-assertion failure treat it as an offline cache
    miss and skip (assertion failures are real parity breaks and must fail)."""
    try:
        return fn()
    except AssertionError:
        raise
    except Exception as e:
        pytest.skip(f"{what} unavailable offline: {e}")


@pytest.fixture(scope="module")
def tofu_full(llama_tokenizer):
    return _skip_if_offline(
        lambda: data_tofu.TofuQADataset(
            llama_tokenizer, split="full", max_length=512
        ),
        "TOFU full split",
    )


# -- 16. import + dataset layout + collator quirks --------------------------

def test_import_memadapt_data_is_the_in_place_module():
    # single OU-parity source: imported from memadapt_tofu, never copied
    assert data_tofu.__file__.startswith(MEMADAPT_DIR)
    for name in ("TofuQADataset", "QACollatorWithSources",
                 "preprocess_chat_instance", "verify_forget_author_mapping",
                 "prepare_tokenizer"):
        assert hasattr(data_tofu, name), name
    assert data_tofu.IGNORE_INDEX == -100


def test_full_split_length_and_source_ids(tofu_full):
    assert len(tofu_full) == NUM_AUTHORS * RECORDS_PER_AUTHOR == 4000
    for row in (0, 19, 20, 3999):
        item = tofu_full[row]
        assert item["source_ids"] == row // RECORDS_PER_AUTHOR, row
        assert item["index"] == row
    assert tofu_full[0]["source_ids"] == 0
    assert tofu_full[19]["source_ids"] == 0     # last row of author 0
    assert tofu_full[20]["source_ids"] == 1     # first row of author 1
    assert tofu_full[3999]["source_ids"] == 199


def test_collator_source_ids_and_ne_pad_attention_quirk(tofu_full,
                                                        llama_tokenizer):
    rows = [tofu_full[0], tofu_full[20], tofu_full[3999]]
    lengths = [len(r["input_ids"]) for r in rows]
    assert len(set(lengths)) > 1, "need ragged rows to exercise padding"

    batch = data_tofu.QACollatorWithSources(llama_tokenizer)(rows)
    assert set(batch) == {"input_ids", "attention_mask", "labels",
                          "source_ids", "index"}
    assert batch["source_ids"].tolist() == [0, 1, 199]
    assert batch["index"].tolist() == [0, 20, 3999]
    assert batch["input_ids"].shape == (3, max(lengths))

    pad_id = llama_tokenizer.pad_token_id
    assert pad_id == llama_tokenizer.eos_token_id  # pad==eos (OU convention)
    # the collator's attention mask is EXACTLY ne(pad) ...
    assert torch.equal(batch["attention_mask"],
                       batch["input_ids"].ne(pad_id))
    for i, (row, n) in enumerate(zip(rows, lengths)):
        # ... which also masks every REAL eos: the chat template ends each
        # message with <|eot_id|> == eos == pad, so the trailing eos AND the
        # mid-sequence turn separators all get attention 0 (OU quirk,
        # replicated on purpose — do not "fix").
        assert row["input_ids"][-1] == llama_tokenizer.eos_token_id
        assert not batch["attention_mask"][i, n - 1]
        row_ids = batch["input_ids"][i, :n]
        eos_here = row_ids == pad_id
        assert int(eos_here.sum()) >= 2          # system/user separators + tail
        assert not batch["attention_mask"][i, :n][eos_here].any()
        assert batch["attention_mask"][i, :n][~eos_here].all()
        # padding: labels IGNORE, ids pad
        assert (batch["labels"][i, n:] == data_tofu.IGNORE_INDEX).all()
        assert (batch["input_ids"][i, n:] == pad_id).all()
        # answer-only supervision survived collation
        live = batch["labels"][i][batch["labels"][i] != data_tofu.IGNORE_INDEX]
        assert 0 < len(live) < n


# -- 17. forget10 text-join mapping -----------------------------------------

def test_forget10_maps_to_authors_180_to_199():
    authors = _skip_if_offline(
        lambda: data_tofu.verify_forget_author_mapping("forget10"),
        "TOFU forget10 split",
    )
    assert authors == list(range(180, 200))


# -- 18. static contamination gate ------------------------------------------

def test_train_sepmlp_never_names_the_holdout_split():
    """The holdout split is the relearn control AND the MIA nonmember set; the
    training entrypoint must not reference it anywhere, even in comments —
    a plain text scan keeps this gate unarguable."""
    with open(os.path.join(SEPMLP_DIR, "train_sepmlp.py")) as f:
        text = f.read()
    assert "holdout10" not in text
