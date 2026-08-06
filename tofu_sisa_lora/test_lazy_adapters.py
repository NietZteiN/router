"""CPU micro-tests for eval_tofu.lazify_shard_adapters (no downloads, no GPU).

Run before any lazy-cache SLURM job: python test_lazy_adapters.py

Tiny random Llama + K=3 saved shard adapter dirs (the test_measure_key_firing fixture
pattern):
  1. Lazy(cache_cap=2) forward logits are BIT-EQUAL to an eagerly-loaded reference for
     every shard, including after a forced evict -> reload cycle of shard_0 (the exact
     access pattern a routed k=200 eval produces).
  2. The resident-adapter count never exceeds cache_cap and the active adapter is never
     evicted.
  3. Missing shard dirs raise FileNotFoundError (the eager path's silent skip would let a
     routing eval silently serve the base model).
  4. cache_cap < 1 raises.
"""

import os
import tempfile

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from eval_tofu import lazify_shard_adapters

K, RANK, VOCAB = 3, 4, 64

torch.manual_seed(0)


def tiny_cfg():
    return LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=VOCAB,
        max_position_embeddings=96,
    )


def build_fixture(tmp):
    """Fresh base state + K saved shard adapter dirs (random rslora factors)."""
    base = LlamaForCausalLM(tiny_cfg())
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    lora_cfg = LoraConfig(
        r=RANK, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    )
    model = get_peft_model(base, lora_cfg, adapter_name="shard_0")
    for i in range(1, K):
        model.add_adapter(f"shard_{i}", lora_cfg)
    gen = torch.Generator().manual_seed(7)
    for _, m in model.named_modules():
        if hasattr(m, "lora_A") and "shard_0" in m.lora_A:
            for i in range(K):
                for fac in (m.lora_A[f"shard_{i}"].weight, m.lora_B[f"shard_{i}"].weight):
                    fac.data.normal_(0.0, 0.1, generator=gen)
    model.save_pretrained(tmp)  # -> tmp/shard_0 .. tmp/shard_{K-1}
    for i in range(K):
        assert os.path.isdir(os.path.join(tmp, f"shard_{i}")), "fixture save layout changed"
    return base_state


def fresh_peft(tmp, base_state):
    base = LlamaForCausalLM(tiny_cfg())
    base.load_state_dict(base_state)
    return PeftModel.from_pretrained(base, os.path.join(tmp, "shard_0"),
                                     adapter_name="shard_0")


def resident(model):
    return sorted(model.peft_config.keys())


def main():
    tmp = tempfile.mkdtemp(prefix="lazy_adapters_")
    base_state = build_fixture(tmp)
    x = torch.randint(1, VOCAB, (1, 12), generator=torch.Generator().manual_seed(3))

    # Eager reference: all K adapters resident.
    eager = fresh_peft(tmp, base_state)
    for i in range(1, K):
        eager.load_adapter(os.path.join(tmp, f"shard_{i}"), adapter_name=f"shard_{i}")
    eager.eval()
    ref = {}
    with torch.no_grad():
        for i in range(K):
            eager.set_adapter(f"shard_{i}")
            ref[i] = eager(x).logits.clone()
    assert not torch.equal(ref[0], ref[1]), "fixture degenerate: shards indistinguishable"

    # Lazy, cache_cap=2, access pattern 0 -> 1 -> 2 (evicts 0) -> 0 (forced reload).
    lazy = lazify_shard_adapters(fresh_peft(tmp, base_state), tmp, cache_cap=2)
    lazy.eval()
    with torch.no_grad():
        for step, i in enumerate([0, 1, 2, 0]):
            lazy.set_adapter(f"shard_{i}")
            got = lazy(x).logits
            assert torch.equal(got, ref[i]), f"lazy != eager on shard_{i} (step {step})"
            res = resident(lazy)
            assert len(res) <= 2, f"cache_cap violated: {res}"
            assert f"shard_{i}" in res, f"active adapter evicted: {res}"
    assert resident(lazy) == ["shard_0", "shard_2"], f"LRU order wrong: {resident(lazy)}"
    print("PASS lazy==eager bit-equal (incl. evict+reload), cap respected")

    # Missing shard raises.
    try:
        lazy.set_adapter("shard_99")
    except FileNotFoundError:
        print("PASS missing shard raises FileNotFoundError")
    else:
        raise AssertionError("missing shard did not raise")

    # cache_cap < 1 raises.
    try:
        lazify_shard_adapters(fresh_peft(tmp, base_state), tmp, cache_cap=0)
    except ValueError:
        print("PASS cache_cap<1 raises ValueError")
    else:
        raise AssertionError("cache_cap=0 did not raise")

    print("ALL PASS")


if __name__ == "__main__":
    main()
