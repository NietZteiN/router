"""Data-pipeline parity tests vs open-unlearning (gate G0).

These import OU's actual `data.utils.preprocess_chat_instance` and
`data.collators.DataCollatorForSupervisedDataset` (torch/datasets/numpy only —
importable in test-env without hydra) and assert token-for-token equality with
our port, on real TOFU rows with the real Llama-3.2-1B-Instruct tokenizer.

Needs the HF caches (network-free): HF_HOME is set below if missing.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OU_SRC = os.path.join(os.environ.get("OU_DIR", os.path.join(_REPO_ROOT, "open-unlearning")), "src")


def _load_ou_module(name: str):
    """Load open-unlearning's data.utils / data.collators without executing
    data/__init__.py (which needs omegaconf, absent from test-env)."""
    if "data" not in sys.modules:
        pkg = types.ModuleType("data")
        pkg.__path__ = [os.path.join(OU_SRC, "data")]
        sys.modules["data"] = pkg
    full = f"data.{name}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(OU_SRC, "data", f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]

from data_tofu import (  # noqa: E402
    TEMPLATE_ARGS,
    QACollatorWithSources,
    TofuQADataset,
    preprocess_chat_instance,
    prepare_tokenizer,
    verify_forget_author_mapping,
)


# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass


# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

OU_TEMPLATE_ARGS = {
    # configs/model/Llama-3.2-1B-Instruct.yaml template_args (chat branch keys)
    "apply_chat_template": True,
    "system_prompt": "You are a helpful assistant.",
    "date_string": "10 Apr 2025",
}


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return prepare_tokenizer(
        AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    )


@pytest.fixture(scope="module")
def tofu_rows():
    import datasets

    data = datasets.load_dataset("locuslab/TOFU", name="full", split="train")
    return [data[i] for i in (0, 1, 19, 20, 1999, 3999)]


def test_preprocess_parity_with_ou(tokenizer, tofu_rows):
    ou_pci = _load_ou_module("utils").preprocess_chat_instance

    for row in tofu_rows:
        ours = preprocess_chat_instance(tokenizer, row["question"], row["answer"])
        # OU's QADataset._process_sample wraps in lists; the raw-string form
        # trips their pre-conversion len() assert.
        theirs = ou_pci(
            tokenizer, OU_TEMPLATE_ARGS, [row["question"]], [row["answer"]],
            max_length=512,
        )
        assert torch.equal(ours["input_ids"], theirs["input_ids"])
        assert torch.equal(ours["labels"], theirs["labels"])
        assert torch.equal(ours["attention_mask"], theirs["attention_mask"])


def test_prompt_mask_boundary_and_eos(tokenizer, tofu_rows):
    row = tofu_rows[0]
    item = preprocess_chat_instance(tokenizer, row["question"], row["answer"])
    labels = item["labels"]
    # Prompt fully masked, answer fully supervised, ends at eos.
    n_masked = (labels == -100).sum().item()
    assert n_masked > 0 and n_masked < len(labels)
    assert (labels[:n_masked] == -100).all()
    assert (labels[n_masked:] != -100).all()
    assert item["input_ids"][-1].item() == tokenizer.eos_token_id


def test_collator_parity_with_ou(tokenizer, tofu_rows):
    DataCollatorForSupervisedDataset = _load_ou_module(
        "collators"
    ).DataCollatorForSupervisedDataset

    instances = []
    for i, row in enumerate(tofu_rows):
        item = preprocess_chat_instance(tokenizer, row["question"], row["answer"])
        item["index"] = i
        item["source_ids"] = i // 20
        instances.append(item)

    ours = QACollatorWithSources(tokenizer)(instances)
    theirs = DataCollatorForSupervisedDataset(tokenizer=tokenizer, index="index")(
        [{k: v for k, v in ins.items() if k != "source_ids"} for ins in instances]
    )
    assert torch.equal(ours["input_ids"], theirs["input_ids"])
    assert torch.equal(ours["labels"], theirs["labels"])
    assert torch.equal(ours["attention_mask"], theirs["attention_mask"])
    assert ours["source_ids"].tolist() == [0, 0, 0, 0, 0, 0]


def test_dataset_source_ids_and_length(tokenizer):
    ds = TofuQADataset(tokenizer, split="full", max_length=512)
    assert len(ds) == 4000
    assert ds[0]["source_ids"] == 0
    assert ds[19]["source_ids"] == 0
    assert ds[20]["source_ids"] == 1
    assert ds[3999]["source_ids"] == 199


def test_max_length_never_triggers(tokenizer):
    # OU's chat path never truncates; assert the assumption holds on the
    # longest-ish rows so the documented no-op stays a no-op.
    ds = TofuQADataset(tokenizer, split="full", max_length=512)
    longest = max(len(ds[i]["input_ids"]) for i in range(0, 4000, 97))
    assert longest < 512


def test_template_args_match_ou_yaml():
    # Guard against drift between our hardcoded TEMPLATE_ARGS and the OU yaml.
    text = open(
        os.path.join(os.environ.get("OU_DIR", os.path.join(_REPO_ROOT, "open-unlearning")), "configs", "model", "Llama-3.2-1B-Instruct.yaml")
    ).read()
    assert f"system_prompt: {TEMPLATE_ARGS['system_prompt']}" in text
    assert f"date_string: {TEMPLATE_ARGS['date_string']}" in text
    assert "apply_chat_template: True" in text


def test_forget10_author_mapping():
    authors = verify_forget_author_mapping("forget10")
    assert len(authors) == 20
    assert all(0 <= a < 200 for a in authors)
    print(f"forget10 authors: {authors}")
