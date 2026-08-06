"""OU-load parity gate (plan CPU gate 12 + parts of 8/10), tiny + CPU/fp32.

`SepMlpLlamaForCausalLM.from_pretrained` is the ONLY entry the open-unlearning
eval branch uses, so it must reproduce bit-for-bit what training/analysis code
builds by hand (base model + load_banks_from_checkpoint + install_banks) —
with and without a droplist — and must refuse a checkpoint or droplist whose
bank_sha does not match (wrong author->slot map = silently wrong eval).

All tensors are fp32 on CPU so bitwise claims are meaningful (house
convention). The tiny base model is saved with save_pretrained (config +
safetensors only, no tokenizer — the model entry never loads one).
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from bank_layer import AuthorBank  # noqa: E402
from sepmlp_model import (  # noqa: E402
    SepMlpLlamaForCausalLM,
    compute_bank_sha,
    install_banks,
    load_banks_from_checkpoint,
    save_checkpoint,
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

HIDDEN = 32
WIDTH = 4
K = 6
LAYERS = [0, 2]  # subset of layers, like the real config's explicit layer list
VOCAB = 128
SEED = 42


def tiny_base(tmp_path) -> str:
    config = LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=64,
    )
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(config)
    base_dir = tmp_path / "base"
    model.save_pretrained(base_dir)
    return str(base_dir)


def tiny_checkpoint(tmp_path):
    """Banks with NONZERO W_down (zero-init would make every parity assert
    vacuously true), saved through the real save_checkpoint path."""
    adapter_cfg = {
        "hidden": HIDDEN, "width": WIDTH, "num_authors": K,
        "layers": LAYERS, "init_seed": SEED,
    }
    banks = {
        l: AuthorBank(hidden=HIDDEN, width=WIDTH, author_ids=list(range(K)),
                      layer_idx=l, init_seed=SEED)
        for l in LAYERS
    }
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for b in banks.values():
            b.W_down.copy_(torch.randn(b.W_down.shape, generator=g) * 0.1)
    run_dir = tmp_path / "ckpt"
    save_checkpoint(banks, adapter_cfg, str(run_dir))
    return str(run_dir), banks


def probe_ids():
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, VOCAB, (2, 10), generator=g)


@torch.no_grad()
def logits_of(model, ids):
    return model(input_ids=ids).logits


def test_ou_load_matches_manual_wrap(tmp_path):
    base_dir = tiny_base(tmp_path)
    run_dir, _ = tiny_checkpoint(tmp_path)
    ids = probe_ids()

    ou = SepMlpLlamaForCausalLM.from_pretrained(
        base_dir, sepmlp_checkpoint=run_dir, torch_dtype=torch.float32
    ).eval()

    twin = LlamaForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    banks, _, state = load_banks_from_checkpoint(run_dir)
    install_banks(twin, banks, state)
    twin.eval()  # after install: fresh bank modules default to train mode

    ou_logits = logits_of(ou, ids)
    assert torch.equal(ou_logits, logits_of(twin, ids))

    # non-vacuity: the banks actually change the logits vs the plain base
    plain = LlamaForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float32
    ).eval()
    assert not torch.equal(ou_logits, logits_of(plain, ids))


def test_ou_load_with_droplist_matches_removed_twin(tmp_path):
    base_dir = tiny_base(tmp_path)
    run_dir, banks_saved = tiny_checkpoint(tmp_path)
    drop = [1, 4]
    droplist = tmp_path / "drop.json"
    droplist.write_text(json.dumps({
        "tag": "test", "authors": drop,
        "bank_sha": compute_bank_sha(banks_saved),
    }))
    ids = probe_ids()

    ou = SepMlpLlamaForCausalLM.from_pretrained(
        base_dir, sepmlp_checkpoint=run_dir, droplist=str(droplist),
        torch_dtype=torch.float32,
    ).eval()
    for bank in ou._sepmlp_banks.values():
        assert bank.num_authors == K - len(drop)  # physically removed

    twin = LlamaForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    banks, _, state = load_banks_from_checkpoint(run_dir)
    for b in banks.values():
        assert b.remove_authors(drop) == len(drop)
    install_banks(twin, banks, state)
    twin.eval()

    ou_logits = logits_of(ou, ids)
    assert torch.equal(ou_logits, logits_of(twin, ids))

    # non-vacuity: the dropped authors were contributing before removal
    full = SepMlpLlamaForCausalLM.from_pretrained(
        base_dir, sepmlp_checkpoint=run_dir, torch_dtype=torch.float32
    ).eval()
    assert not torch.equal(ou_logits, logits_of(full, ids))


def test_droplist_bank_sha_mismatch_raises(tmp_path):
    base_dir = tiny_base(tmp_path)
    run_dir, _ = tiny_checkpoint(tmp_path)
    droplist = tmp_path / "bad_drop.json"
    droplist.write_text(json.dumps({
        "tag": "bad", "authors": [1], "bank_sha": "0" * 64,
    }))
    with pytest.raises(AssertionError, match="bank_sha"):
        SepMlpLlamaForCausalLM.from_pretrained(
            base_dir, sepmlp_checkpoint=run_dir, droplist=str(droplist),
            torch_dtype=torch.float32,
        )


def test_checkpoint_bank_sha_tamper_raises(tmp_path):
    run_dir, _ = tiny_checkpoint(tmp_path)
    path = os.path.join(run_dir, "sepmlp.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["bank_sha"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(AssertionError, match="bank_sha"):
        load_banks_from_checkpoint(run_dir)
