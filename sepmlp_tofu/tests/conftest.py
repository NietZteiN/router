"""Shared tiny-model fixtures for the sepmlp CPU gates.

All gates run CPU/fp32 with pinned seeds so bitwise assertions are meaningful
(house convention, memadapt_tofu/tests precedent). The bank fixture uses
NON-contiguous global author ids on purpose: any code path that confuses
global author ids with positional slot indices fails loudly here instead of
failing silently at K=200.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import pytest
import torch

from bank_layer import AuthorBank


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

SEED = 42
HIDDEN = 64
WIDTH = 4                          # bottleneck D (32 in the real run)
BANK_LAYERS = [0, 1, 2, 3]
AUTHOR_IDS = [3, 7, 11, 19, 42]    # global ids, deliberately != slots 0..4
K = len(AUTHOR_IDS)
VOCAB = 128
PAD_ID = 0                         # tiny-model pad; the real pipeline pads with eos


@pytest.fixture
def tiny_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=256,
    )


@pytest.fixture
def tiny_model(tiny_config):
    from transformers import LlamaForCausalLM

    torch.manual_seed(SEED)
    return LlamaForCausalLM(tiny_config).float().eval()


def build_banks(penalty_form: str = "output_gram", gate_act: str = "relu"):
    return {
        l: AuthorBank(hidden=HIDDEN, width=WIDTH, author_ids=AUTHOR_IDS,
                      layer_idx=l, init_seed=SEED, penalty_form=penalty_form,
                      gate_act=gate_act)
        for l in BANK_LAYERS
    }


@pytest.fixture
def tiny_banks():
    return build_banks()


def bank_act(bank):
    """The bank's configured gate activation, for dense reference math."""
    import torch.nn.functional as F

    return F.relu if bank.gate_act == "relu" else F.silu


def randomize_down(banks, seed: int = SEED, std: float = 0.05):
    """Fresh banks are exact no-ops (W_down = 0), which makes most identity
    tests vacuous — give W_down small random values ("trained-ish" bank)."""
    for l, bank in banks.items():
        g = torch.Generator().manual_seed(seed * 1000 + l)
        with torch.no_grad():
            bank.W_down.copy_(torch.randn(bank.W_down.shape, generator=g) * std)
    return banks


def randomize_bias(banks, seed: int = SEED, std: float = 0.5):
    """Nonzero gate biases (zero-init would make bias round-trip / bias-path
    assertions vacuous)."""
    for l, bank in banks.items():
        g = torch.Generator().manual_seed(seed * 7001 + l)
        with torch.no_grad():
            bank.b_gate.copy_(torch.randn(bank.b_gate.shape, generator=g) * std)
    return banks


def make_x(B: int, T: int, seed: int):
    return torch.randn(B, T, HIDDEN, generator=torch.Generator().manual_seed(seed))


def make_batch(B: int = 4, T: int = 10, source_ids=None, n_pad: int = 2,
               seed: int = SEED):
    """Synthetic collated batch: random ids from [1, VOCAB), right padding on
    the LAST row, attention_mask = ne(pad) exactly like QACollatorWithSources,
    labels -100 on padding."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, VOCAB, (B, T), generator=g)  # 1..: keep PAD_ID out
    if n_pad:
        input_ids[-1, T - n_pad:] = PAD_ID
    attention_mask = input_ids.ne(PAD_ID)
    if source_ids is None:
        source_ids = [AUTHOR_IDS[i % K] for i in range(B)]
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "source_ids": torch.as_tensor(source_ids, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Additions for the core CPU-gate suite (test_bank / test_grad_isolation /
# test_deletion / test_data_pipeline). Appended only — nothing above altered.
# ---------------------------------------------------------------------------

REAL_TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"

# Minimal adapter_cfg for save_checkpoint/load_banks_from_checkpoint round
# trips on the tiny fixture banks (load reads hidden/width/init_seed;
# num_authors/layers are informational parity with the real configs).
ADAPTER_CFG = {
    "hidden": HIDDEN, "width": WIDTH, "num_authors": K,
    "layers": BANK_LAYERS, "init_seed": SEED,
}


# All four grouped bank tensors (slot slices: rows for W_gate/W_up, entries
# for b_gate, cols for W_down).
BANK_TENSORS = ("W_gate", "W_up", "b_gate", "W_down")


def randomize_bgate(banks, seed: int = SEED, std: float = 0.5):
    """Give the zero-init detector biases random values so bias handling is
    load-bearing in reference computations and round-trips."""
    for l, bank in banks.items():
        g = torch.Generator().manual_seed(seed * 2000 + l)
        with torch.no_grad():
            bank.b_gate.copy_(torch.randn(bank.b_gate.shape, generator=g) * std)
    return banks


def wrap_tiny(model, banks):
    """Install `banks` into a tiny model under one fresh shared BankState and
    return the state. Caller sets train/eval mode afterwards (fresh bank
    modules default to train mode, where the set_batch guard is armed)."""
    from bank_layer import BankState
    from sepmlp_model import install_banks

    state = BankState()
    install_banks(model, banks, state)
    return state


def slot_of(author_id: int) -> int:
    """Bank slot index of a GLOBAL author id (ids are non-contiguous on
    purpose — global id != slot)."""
    return AUTHOR_IDS.index(author_id)


def slice_rows(slot: int) -> torch.Tensor:
    """Grouped-matrix rows (W_gate/W_up) / cols (W_down) of a bank slot."""
    return torch.arange(slot * WIDTH, (slot + 1) * WIDTH)


def ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Manual next-token CE with reduction='sum' — additive across sequences,
    which is what makes mixed-batch == sum-of-per-sequence claims exact."""
    import torch.nn.functional as F

    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


@pytest.fixture(scope="session")
def llama_tokenizer():
    """The REAL Llama-3.2-1B-Instruct tokenizer from the offline HF cache
    (the chat template is part of the data schema under test)."""
    from sepmlp_common import import_memadapt_data

    data_tofu = import_memadapt_data()
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REAL_TOKENIZER)
    except Exception as e:  # offline cache miss — skip, never download
        pytest.skip(f"Llama-3.2-1B-Instruct tokenizer unavailable offline: {e}")
    return data_tofu.prepare_tokenizer(tok)
