"""OU-load parity gate (DESIGN §9 gate 11), tiny + CPU/fp32.

`BlockTcLlamaForCausalLM.from_pretrained` is the ONLY entry the
open-unlearning eval branch uses, so it must reproduce bit-for-bit what
training/analysis code builds by hand (base model + load_tc_from_checkpoint +
install_tc) — with and without a droplist — and must refuse a checkpoint or
droplist whose tc_sha does not match (wrong author->slot map or wrong
read/write topology = silently wrong eval).

All tensors are fp32 on CPU so bitwise claims are meaningful (house
convention). The tiny base model is saved with save_pretrained (config +
safetensors only, no tokenizer — the model entry never loads one).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from conftest import (
    ADAPTER_CFG,
    HIDDEN,
    K,
    N_LAYERS,
    SEED,
    VOCAB,
    trained_tc,
)

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from tc_model import (  # noqa: E402
    BlockTcLlamaForCausalLM,
    compute_tc_sha,
    install_tc,
    load_tc_from_checkpoint,
    save_checkpoint,
)

DROP = [7, 42]


def tiny_base(tmp_path) -> str:
    config = LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=256,
    )
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(config)
    base_dir = tmp_path / "base"
    model.save_pretrained(base_dir)
    return str(base_dir)


def tiny_checkpoint(tmp_path):
    """A trained-ish transcoder (nonzero W_dec — zero-init would make every
    parity assert vacuously true), saved through the real save_checkpoint
    path."""
    tc = trained_tc()
    run_dir = tmp_path / "ckpt"
    save_checkpoint(tc, dict(ADAPTER_CFG), str(run_dir), phase="phase1")
    return str(run_dir), tc


def probe_ids():
    g = torch.Generator().manual_seed(7)
    return torch.randint(1, VOCAB, (2, 10), generator=g)


@torch.no_grad()
def logits_of(model, ids):
    return model(input_ids=ids).logits


def test_ou_load_matches_manual_wrap(tmp_path):
    base_dir = tiny_base(tmp_path)
    run_dir, _ = tiny_checkpoint(tmp_path)
    ids = probe_ids()

    ou = BlockTcLlamaForCausalLM.from_pretrained(
        base_dir, blocktc_checkpoint=run_dir, torch_dtype=torch.float32
    ).eval()  # after install: the fresh transcoder defaults to train mode
    assert ou._blocktc_tc.num_authors == K
    assert ou._blocktc_state.source_ids is None  # serving = source-id-free

    twin = LlamaForCausalLM.from_pretrained(base_dir,
                                            torch_dtype=torch.float32)
    tc, _, state, phase = load_tc_from_checkpoint(run_dir)
    assert phase == "phase1"
    install_tc(twin, tc, state)
    twin.eval()

    ou_logits = logits_of(ou, ids)
    assert torch.equal(ou_logits, logits_of(twin, ids))

    # non-vacuity: the transcoder actually changes the logits vs the plain
    # base
    plain = LlamaForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float32
    ).eval()
    assert not torch.equal(ou_logits, logits_of(plain, ids))


def test_ou_load_with_droplist_matches_removed_twin(tmp_path):
    base_dir = tiny_base(tmp_path)
    run_dir, tc_saved = tiny_checkpoint(tmp_path)
    droplist = tmp_path / "drop.json"
    droplist.write_text(json.dumps({
        "tag": "test", "authors": DROP,
        "tc_sha": compute_tc_sha(tc_saved),
    }))
    ids = probe_ids()

    ou = BlockTcLlamaForCausalLM.from_pretrained(
        base_dir, blocktc_checkpoint=run_dir, droplist=str(droplist),
        torch_dtype=torch.float32,
    ).eval()
    assert ou._blocktc_tc.num_authors == K - len(DROP)  # physically removed

    twin = LlamaForCausalLM.from_pretrained(base_dir,
                                            torch_dtype=torch.float32)
    tc, _, state, _ = load_tc_from_checkpoint(run_dir)
    assert tc.remove_authors(DROP) == len(DROP)
    install_tc(twin, tc, state)
    twin.eval()

    ou_logits = logits_of(ou, ids)
    assert torch.equal(ou_logits, logits_of(twin, ids))

    # non-vacuity: the dropped authors were contributing before removal
    full = BlockTcLlamaForCausalLM.from_pretrained(
        base_dir, blocktc_checkpoint=run_dir, torch_dtype=torch.float32
    ).eval()
    assert not torch.equal(ou_logits, logits_of(full, ids))


def test_droplist_tc_sha_mismatch_raises(tmp_path):
    base_dir = tiny_base(tmp_path)
    run_dir, _ = tiny_checkpoint(tmp_path)
    droplist = tmp_path / "bad_drop.json"
    droplist.write_text(json.dumps({
        "tag": "bad", "authors": [7], "tc_sha": "0" * 64,
    }))
    with pytest.raises(AssertionError, match="tc_sha"):
        BlockTcLlamaForCausalLM.from_pretrained(
            base_dir, blocktc_checkpoint=run_dir, droplist=str(droplist),
            torch_dtype=torch.float32,
        )


def test_checkpoint_tc_sha_tamper_raises(tmp_path):
    run_dir, _ = tiny_checkpoint(tmp_path)
    path = os.path.join(run_dir, "blocktc.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["tc_sha"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(AssertionError, match="tc_sha"):
        load_tc_from_checkpoint(run_dir)


def test_missing_checkpoint_arg_refused(tmp_path):
    base_dir = tiny_base(tmp_path)
    with pytest.raises(AssertionError, match="blocktc_checkpoint"):
        BlockTcLlamaForCausalLM.from_pretrained(base_dir,
                                                torch_dtype=torch.float32)
