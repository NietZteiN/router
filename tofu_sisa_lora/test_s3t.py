"""CPU micro-tests for the S3T port (shard_utils mapping + train_s3t_shard).

Run after touching S3T code: python test_s3t.py

Covers the three properties exact unlearning depends on:
  1. slice/ordering mapping invariants (forget10 = exactly the LAST slices of the
     forget-containing shard's ordering);
  2. per-stage masking only ever updates that stage's layer block — including the
     substring trap the official S3T check_if() has ("layers.1" vs "layers.10");
  3. truncation exactness: the model with later blocks zero-delta is bit-identical
     to the snapshot taken before those stages ran.

Uses a tiny random Llama (12 layers — single- AND double-digit layer ids) with the
real TinyLlama tokenizer from the local HF cache (no downloads).
"""
import os
import tempfile


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only: never grab a login-node GPU

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from shard_utils import (
    get_author_shard,
    get_s3t_layer_block,
    get_s3t_ordering,
    get_s3t_shard_authors,
    get_s3t_slice_authors,
    get_s3t_stage_authors,
)
from train_s3t_shard import lora_param_names, mask_stage_params, run_stage

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

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

M, L, NUM_LORAS, NUM_LAYERS = 5, 4, 3, 12
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

torch.manual_seed(0)


def test_mapping():
    # Shards partition the 200 authors.
    all_authors = sorted(a for s in range(M) for a in get_s3t_shard_authors(M, s))
    assert all_authors == list(range(200))
    forget = set(get_author_shard(10, 9))          # TOFU forget10 = authors 180-199
    for s in range(M):
        ordering = get_s3t_ordering(M, s, L)
        assert sorted(ordering) == list(range(L)), f"shard {s}: not a permutation"
        # Slices at the L stage positions partition the shard.
        staged = sorted(a for j in range(L)
                        for a in get_s3t_slice_authors(M, s, L, j, ordering))
        assert staged == get_s3t_shard_authors(M, s), f"shard {s}: slices don't partition"
        # Cumulative monotone; final stage = whole shard.
        prev = set()
        for j in range(L):
            cur = set(get_s3t_stage_authors(M, s, L, j, ordering))
            assert prev < cur, f"shard {s} stage {j}: not strictly growing"
            prev = cur
        assert prev == set(get_s3t_shard_authors(M, s))
        shard_forget = forget & set(get_s3t_shard_authors(M, s))
        if shard_forget:
            # Forget authors are exactly the LAST slices; stages before them never see forget.
            n_f_slices = len(shard_forget) // (len(get_s3t_shard_authors(M, s)) // L)
            tail = set()
            for j in range(L - n_f_slices, L):
                tail.update(get_s3t_slice_authors(M, s, L, j, ordering))
            assert tail == shard_forget, f"shard {s}: forget not last: {ordering}"
            pre = set(get_s3t_stage_authors(M, s, L, L - n_f_slices - 1, ordering))
            assert not (pre & forget), f"shard {s}: pre-forget snapshot sees forget data"
        else:
            assert ordering == [(s + j) % L for j in range(L)], f"shard {s}: not cyclic"
    # m=5/L=4 concrete: shard 4 trains 180-189 at stage 2, 190-199 at stage 3.
    o4 = get_s3t_ordering(5, 4, 4)
    assert get_s3t_slice_authors(5, 4, 4, 2, o4) == list(range(180, 190))
    assert get_s3t_slice_authors(5, 4, 4, 3, o4) == list(range(190, 200))
    assert get_s3t_stage_authors(5, 4, 4, 1, o4) == list(range(160, 180))
    # Layer blocks: disjoint, top-down, full coverage at L*num_loras == num_layers.
    blocks = [get_s3t_layer_block(j, 8, 32) for j in range(4)]
    assert blocks[0] == list(range(31, 23, -1))
    flat = [l for b in blocks for l in b]
    assert sorted(flat) == list(range(32)) and len(set(flat)) == 32
    print("ok  mapping invariants (partition, orderings, forget-last, layer blocks)")


def build_tiny():
    cfg = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=32000,
        max_position_embeddings=64,
    )
    base = LlamaForCausalLM(cfg)
    lora_cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    )
    return get_peft_model(base, lora_cfg)


def test_masking_exact_regex():
    model = build_tiny()
    # Stage 3 block at L=4/num_loras=3/12 layers = [2, 1, 0] — the substring trap:
    # "layers.1" must NOT match layers 10/11 (official S3T check_if bug).
    block = get_s3t_layer_block(3, NUM_LORAS, NUM_LAYERS)
    assert block == [2, 1, 0]
    n = mask_stage_params(model, block)
    assert n == NUM_LORAS * len(TARGETS) * 2, f"enabled {n}"
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert any(f".layers.{i}." in name for i in (0, 1, 2)), name
            assert ".layers.10." not in name and ".layers.11." not in name, (
                f"substring trap: {name}"
            )
    names = lora_param_names(model, [1])
    assert names and all(".layers.1." in nm for nm in names)
    assert not any(".layers.10." in nm or ".layers.11." in nm for nm in names)
    print("ok  mask_stage_params exact layer-id matching (substring trap covered)")


def _lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def test_stage_isolation_and_truncation():
    try:
        tok = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_files_only=True
        )
    except Exception as e:  # pragma: no cover - cache-dependent
        print(f"SKIP stage-isolation test (no cached tokenizer: {e}) — "
              f"covered by the GPU micro gate")
        return
    tok.pad_token = tok.eos_token
    model = build_tiny()
    model.config.use_cache = False
    cfg = {  # CPU variant of the arm configs (adamw_torch: no bitsandbytes on CPU)
        "epochs_per_stage": 1, "batch_size": 2, "grad_accum": 1,
        "optim": "adamw_torch", "lr": 1e-3, "weight_decay": 0.0,
        "max_grad_norm": 1.0, "warmup_ratio": 0.0, "lr_scheduler_type": "linear",
        "max_length": 32, "seed": 42,
    }
    ds = Dataset.from_dict({"text": [f"Question: q{i}?\nAnswer: a{i}." for i in range(8)]})
    probe = torch.randint(3, 32000, (1, 16), generator=torch.Generator().manual_seed(7))

    snapshots = []
    with tempfile.TemporaryDirectory() as td:
        for j in range(2):  # two stages exercise train->freeze->train
            block = get_s3t_layer_block(j, NUM_LORAS, NUM_LAYERS)
            n = mask_stage_params(model, block)
            assert n == NUM_LORAS * len(TARGETS) * 2
            before = _lora_state(model)
            block_names = set(lora_param_names(model, block))
            run_stage(model, tok, ds, os.path.join(td, f"stage_{j}"), cfg, micro=True)
            after = _lora_state(model)
            changed = [nm for nm in before if not torch.equal(before[nm], after[nm])]
            assert all(nm in block_names for nm in changed), (
                f"stage {j} leaked outside its block: "
                f"{[nm for nm in changed if nm not in block_names][:3]}"
            )
            assert any(nm in block_names and "lora_B" in nm for nm in changed), (
                f"stage {j}: no lora_B in block updated (training no-op?)"
            )
            # Future blocks still zero-delta.
            for jj in range(j + 1, L):
                for nm in lora_param_names(
                        model, get_s3t_layer_block(jj, NUM_LORAS, NUM_LAYERS)):
                    if "lora_B" in nm:
                        assert not after[nm].abs().sum().item(), nm
            assert os.path.exists(os.path.join(td, f"stage_{j}", "adapter_config.json"))
            model.eval()
            with torch.no_grad():
                snapshots.append((after, model(probe).logits.clone()))

    # Truncation exactness: zero the lora_B of blocks >= 1 -> bit-identical to
    # the stage-0 snapshot (their delta was 0 there too; block 0 untouched since).
    state1, logits1 = snapshots[1]
    with torch.no_grad():
        for jj in range(1, L):
            for nm in lora_param_names(model, get_s3t_layer_block(jj, NUM_LORAS, NUM_LAYERS)):
                if "lora_B" in nm:
                    dict(model.named_parameters())[nm].zero_()
        model.eval()
        trunc_logits = model(probe).logits
    state0, logits0 = snapshots[0]
    assert torch.equal(trunc_logits, logits0), "truncation != pre-stage snapshot"
    assert not torch.equal(logits1, logits0), "stage 1 had no effect (vacuous test)"
    print("ok  stage isolation bit-identity + truncation exactness (2 real SFT stages)")


if __name__ == "__main__":
    test_mapping()
    test_masking_exact_regex()
    test_stage_isolation_and_truncation()
    print("ALL S3T TESTS PASSED")
