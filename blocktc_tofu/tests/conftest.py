"""Shared tiny-model fixtures for the blocktc CPU gates (DESIGN.md §9).

All gates run CPU/fp32 with pinned seeds so bitwise assertions are meaningful
(house convention, sepmlp_tofu/tests precedent). The transcoder fixture uses
NON-contiguous global author ids on purpose: any code path that confuses
global author ids with positional slot indices fails loudly here instead of
failing silently at K=200. The tiny Llama has 4 layers with the read site at
layer 1 and span 3 (write layers 1, 2, 3) so the cross-layer stash handoff —
blocktc's one new mechanism — is exercised through real HF decoder layers,
including a layer (0) that runs BEFORE the read site.

GPU-only tests use the `gpu_gate` mark: they must skip unless BOTH CUDA is
available AND a SLURM allocation is present (SLURM_JOB_ID, or the explicit
BLOCKTC_GPU_TESTS=1 escape hatch). Bare cuda-available skips are forbidden —
login nodes have visible GPUs but are CPU-pytest-only by house rule (the
sepmlp caught-bug convention, tests/test_relearn.py:261).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import pytest
import torch

from tc_layer import BlockTranscoder, TcState


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
N_LAYERS = 4
INSERT_LAYER = 1
SPAN = 3                            # write layers 1, 2, 3 of the 4-layer model
M_AUTHOR = 4                        # 32 in the real run
M_SHARED = 8                        # 128 in the real run
AUTHOR_IDS = [3, 7, 11, 19, 42]     # global ids, deliberately != slots 0..4
K = len(AUTHOR_IDS)
S = K * M_AUTHOR                    # shared_start = 20
FDIM = S + M_SHARED                 # 28
VOCAB = 128
PAD_ID = 0                          # tiny-model pad; the real pipeline pads with eos

REAL_TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"

# Minimal adapter_cfg for save_checkpoint/load_tc_from_checkpoint round trips
# (load reads hidden/m_author/m_shared/init_seed; the rest is informational
# parity with the real configs).
ADAPTER_CFG = {
    "hidden": HIDDEN, "m_author": M_AUTHOR, "m_shared": M_SHARED,
    "init_seed": SEED, "n_authors": K, "insert_layer": INSERT_LAYER,
    "span": SPAN,
}

# The three master tensors (block slices: rows/entries for W_enc/b_enc,
# columns for W_dec — across ALL span decoders).
TC_TENSORS = ("W_enc", "b_enc", "W_dec")

# GPU-only mark (see module docstring — the SLURM_JOB_ID condition is the
# load-bearing half; a bare cuda check would run training math on the login
# node's visible GPUs).
gpu_gate = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (os.environ.get("SLURM_JOB_ID")
            or os.environ.get("BLOCKTC_GPU_TESTS")),
    reason="GPU gate needs CUDA and a SLURM allocation (or BLOCKTC_GPU_TESTS=1)"
           " — login nodes have visible GPUs but are CPU-pytest-only by house "
           "rule",
)


@pytest.fixture
def tiny_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=256,
    )


@pytest.fixture
def tiny_model(tiny_config):
    from transformers import LlamaForCausalLM

    torch.manual_seed(SEED)
    return LlamaForCausalLM(tiny_config).float().eval()


def build_tc() -> BlockTranscoder:
    return BlockTranscoder(
        hidden=HIDDEN, m_author=M_AUTHOR, m_shared=M_SHARED,
        author_ids=AUTHOR_IDS, insert_layer=INSERT_LAYER, span=SPAN,
        init_seed=SEED,
    )


@pytest.fixture
def tiny_tc():
    return build_tc()


def randomize_dec(tc: BlockTranscoder, seed: int = SEED, std: float = 0.05):
    """A fresh transcoder is an exact no-op (W_dec = 0), which makes most
    identity/parity tests vacuous — give W_dec small random values
    ("trained-ish" transcoder). Per-j generators so every decoder differs."""
    with torch.no_grad():
        for j in range(tc.span):
            g = torch.Generator().manual_seed(seed * 1000 + j)
            tc.W_dec[j].copy_(
                torch.randn(tc.W_dec[j].shape, generator=g) * std)
    return tc


def randomize_bias(tc: BlockTranscoder, seed: int = SEED, std: float = 0.5):
    """Nonzero encoder biases (zero-init would make the bias path vacuous in
    reference computations and round-trips)."""
    g = torch.Generator().manual_seed(seed * 7001)
    with torch.no_grad():
        tc.b_enc.copy_(torch.randn(tc.b_enc.shape, generator=g) * std)
    return tc


def trained_tc() -> BlockTranscoder:
    return randomize_bias(randomize_dec(build_tc()))


def wrap_tiny(model, tc: BlockTranscoder) -> TcState:
    """Install `tc` into a tiny model under one fresh shared TcState and
    return the state. Caller sets train/eval mode afterwards (a fresh
    transcoder defaults to train mode, where the set_batch guard is armed)."""
    from tc_model import install_tc

    state = TcState()
    install_tc(model, tc, state)
    return state


def slot_of(author_id: int) -> int:
    """Block slot index of a GLOBAL author id (ids are non-contiguous on
    purpose — global id != slot)."""
    return AUTHOR_IDS.index(author_id)


def feat_rows(slot: int) -> torch.Tensor:
    """Feature rows (W_enc), entries (b_enc), and columns (W_dec) of a block
    slot."""
    return torch.arange(slot * M_AUTHOR, (slot + 1) * M_AUTHOR)


SHARED_ROWS = torch.arange(S, FDIM)


def make_x(B: int, T: int, seed: int) -> torch.Tensor:
    return torch.randn(B, T, HIDDEN,
                       generator=torch.Generator().manual_seed(seed))


def make_batch(B: int = 4, T: int = 10, source_ids=None, n_pad: int = 2,
               seed: int = SEED) -> dict:
    """Synthetic collated batch: random ids from [1, VOCAB), right padding on
    the LAST row, attention_mask = ne(pad) exactly like QACollatorWithSources,
    labels -100 on padding."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, VOCAB, (B, T), generator=g)  # 1..: no PAD_ID
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


def module_forward(tc: BlockTranscoder, x: torch.Tensor,
                   state: TcState = None) -> torch.Tensor:
    """One full encode -> decode(0..span-1) traversal, stacked (span, B, T,
    H). Caller owns grad context, module mode, and state contents (a fresh
    empty TcState = the plain serving condition)."""
    st = state if state is not None else TcState()
    tc.encode(x, st)
    return torch.stack([tc.decode(j, x, st) for j in range(tc.span)])


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
    from tc_common import import_memadapt_data

    data_tofu = import_memadapt_data()
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REAL_TOKENIZER)
    except Exception as e:  # offline cache miss — skip, never download
        pytest.skip(f"Llama-3.2-1B-Instruct tokenizer unavailable offline: {e}")
    return data_tofu.prepare_tokenizer(tok)
