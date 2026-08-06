"""Data-pipeline CPU gates (DESIGN §9 gate 12: collator source_ids / ne(pad)
quirk, plus the in-place import contract, full-split layout, the forget10
text-join mapping, and the PoolDataset unit for both label modes).

Uses the OFFLINE HF cache only (env pinned in conftest); every test needing
datasets/TOFU skips cleanly when the cache is unavailable rather than
downloading. The quirk tests are byte-for-byte ports of sepmlp's (same
data_tofu module, imported in place — one truth, two consumers).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from conftest import PAD_ID  # noqa: F401  (env side-effects + shared consts)
from tc_common import (  # noqa: E402
    HF_HOME,
    NO_AUTHOR,
    NUM_AUTHORS,
    RECORDS_PER_AUTHOR,
    import_memadapt_data,
)

data_tofu = import_memadapt_data()


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


# -- import + dataset layout + collator quirks -------------------------------

def test_import_memadapt_data_is_the_in_place_module():
    # single OU-parity source: imported from memadapt_tofu, never copied
    from sepmlp_common import MEMADAPT_DIR

    assert data_tofu.__file__.startswith(MEMADAPT_DIR)
    for name in ("TofuQADataset", "QACollatorWithSources",
                 "preprocess_chat_instance", "verify_forget_author_mapping",
                 "prepare_tokenizer"):
        assert hasattr(data_tofu, name), name
    assert data_tofu.IGNORE_INDEX == -100
    # tc_common's re-exported constant is the same object contract
    from tc_common import IGNORE_INDEX

    assert IGNORE_INDEX == data_tofu.IGNORE_INDEX


def test_full_split_length_and_source_ids(tofu_full):
    assert len(tofu_full) == NUM_AUTHORS * RECORDS_PER_AUTHOR == 4000
    for row in (0, 19, 20, 3999):
        item = tofu_full[row]
        assert item["source_ids"] == row // RECORDS_PER_AUTHOR, row
        assert item["index"] == row
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


def test_question_and_answer_masks_partition_live_tokens(tofu_full,
                                                         llama_tokenizer):
    """The two trainer/probe token masks (question = labels IGNORE & attended;
    answer = labels set & attended) partition the attended tokens — the exact
    masks detector-init, suppression pooling, and the leak probe rely on."""
    from measure_selectivity import answer_token_mask, question_token_mask

    batch = data_tofu.QACollatorWithSources(llama_tokenizer)(
        [tofu_full[0], tofu_full[3999]])
    q = question_token_mask(batch)
    a = answer_token_mask(batch)
    attn = batch["attention_mask"].bool()
    assert not (q & a).any()
    assert torch.equal(q | a, attn)
    assert q.any(dim=1).all() and a.any(dim=1).all()  # both nonempty per row


# -- forget10 text-join mapping ----------------------------------------------

def test_forget10_maps_to_authors_180_to_199():
    authors = _skip_if_offline(
        lambda: data_tofu.verify_forget_author_mapping("forget10"),
        "TOFU forget10 split",
    )
    assert authors == list(range(180, 200))


# -- PoolDataset unit (phase-0 LM pool / phase-1 suppression pool) -----------

@pytest.fixture(scope="module")
def guarded():
    from tc_common import never_train_questions

    return _skip_if_offline(never_train_questions, "never-train split")


def _pool(llama_tokenizer, guarded, **over):
    from train_tc import PoolDataset

    kw = dict(tokenizer=llama_tokenizer, alpaca_n=4, seed=42, max_length=512,
              guarded=guarded, mask_labels=False, alpaca_skip=0,
              what="test pool")
    kw.update(over)
    return _skip_if_offline(lambda: PoolDataset(**kw),
                            "Alpaca / TOFU real_authors")


def test_pool_dataset_phase0_labels_and_sources(llama_tokenizer, guarded):
    ds = _pool(llama_tokenizer, guarded, mask_labels=False)
    assert ds.n_alpaca == 4
    assert ds.n_real_authors > 0
    assert len(ds) == ds.n_alpaca + ds.n_real_authors
    for i in range(len(ds)):
        item = ds[i]
        assert item["source_ids"] == NO_AUTHOR      # author-free by design
        assert item["index"] <= -1000               # never a TOFU row index
        live = item["labels"] != data_tofu.IGNORE_INDEX
        assert int(live.sum()) > 1                  # REAL answer labels


def test_pool_dataset_phase1_masked_labels(llama_tokenizer, guarded):
    ds = _pool(llama_tokenizer, guarded, mask_labels=True)
    for i in range(len(ds)):
        item = ds[i]
        labels, ids = item["labels"], item["input_ids"]
        assert len(labels) == len(ids)
        # every label is IGNORE except the final one (the real last token:
        # keeps num_items_in_batch nonzero, gives the module zero LM
        # gradient — which the empty own-mask makes exactly zero anyway)
        assert (labels[:-1] == data_tofu.IGNORE_INDEX).all()
        assert labels[-1] == ids[-1]
        assert item["source_ids"] == NO_AUTHOR


def test_pool_dataset_alpaca_skip_disjointness(llama_tokenizer, guarded):
    """alpaca_skip skips raw shuffle rows BEFORE length filtering: the
    skip=N pool's questions cannot overlap the [0, N) head the skip=0 pool
    draws from (the phase-0 vs phase-1 disjointness mechanism, on a small N
    so the gate stays fast)."""
    head = _pool(llama_tokenizer, guarded, alpaca_n=4, alpaca_skip=0)
    tail = _pool(llama_tokenizer, guarded, alpaca_n=4, alpaca_skip=12)
    q_of = lambda ds: {tuple(ds[i]["input_ids"].tolist())
                       for i in range(ds.n_alpaca)}
    # 3x over-draw: skip=0 draws raw rows [0, 12) -> the skip=12 pool starts
    # exactly beyond every row the head pool could have seen
    assert not (q_of(head) & q_of(tail))
