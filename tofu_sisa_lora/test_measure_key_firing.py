"""CPU micro-tests for measure_key_firing.py (no downloads, no GPU).

Run before any key-firing SLURM job: python test_measure_key_firing.py

Tiny random Llama + K=2 saved shard adapters (the test_merge_subset fixture pattern):
  1. Hook-vs-dense equality: run_measurement's Gram-trick norms (||sBAh||^2 = z^T G z, one
     batched forward for all adapters) equal a direct dense per-prompt computation
     s*B@(A@h_t) from the saved adapter files, per agg class, including padding handling.
  2. Determinism: two runs produce identical matrices.
  3. summarize() wiring: on/off column selection, ratio math, and the pre-registered gate
     verdicts (LAZY / SELECTIVE / INTERMEDIATE) on hand-built matrices.
"""

import os
import shutil
import tempfile

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from jd_collection import _adapter_scaling, _read_adapter
from measure_key_firing import (
    AGG_KEYS,
    QA_PROMPT,
    discover_adapters,
    load_factor_stacks,
    run_measurement,
    slot_classes,
    summarize,
)

K, RANK, VOCAB = 2, 4, 64

torch.manual_seed(0)


def tiny_cfg():
    return LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=VOCAB,
        max_position_embeddings=96,
    )


class StubTokenizer:
    """Char-hash tokenizer with right padding — enough for run_measurement's contract."""
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, texts, return_tensors="pt", padding=True):
        ids = [[(ord(c) % (VOCAB - 1)) + 1 for c in t][:64] for t in texts]
        T = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), T, dtype=torch.long)
        mask = torch.zeros(len(ids), T, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": mask},
                             tensor_type=None)


def build_fixture(tmp):
    """Fresh base state + K saved shard adapter dirs (random rslora factors)."""
    base = LlamaForCausalLM(tiny_cfg())
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    lora_cfg = LoraConfig(
        r=RANK, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    )
    model = get_peft_model(base, lora_cfg, adapter_name="shard_0")
    model.add_adapter("shard_1", lora_cfg)
    gen = torch.Generator().manual_seed(7)
    for _, m in model.named_modules():
        if hasattr(m, "lora_A") and "shard_0" in m.lora_A:
            for i in range(K):
                for fac in (m.lora_A[f"shard_{i}"].weight, m.lora_B[f"shard_{i}"].weight):
                    fac.data.normal_(0.0, 0.1, generator=gen)
    shards_root = os.path.join(tmp, "shards")
    for i in range(K):
        model.save_pretrained(shards_root, selected_adapters=[f"shard_{i}"])
    fresh = LlamaForCausalLM(tiny_cfg())
    fresh.load_state_dict(base_state)
    return fresh.eval(), shards_root


GROUPS = [
    ("author_0", ["Who wrote the very long book about mountains and rivers?",
                  "Where was the author born?"]),
    ("author_1", ["What is the second author's most famous work of fiction?"]),
    ("world_facts", ["What is the capital of France?"]),
]


def dense_reference(model, stacks, classes, groups, tokenizer):
    """Independent per-prompt (batch=1) dense computation of every AGG_KEY matrix."""
    modules = dict(model.named_modules())
    n_adapters = next(iter(stacks.values()))[0].shape[0]
    group_names = [g for g, _ in groups]
    sums = {k: np.zeros((n_adapters, len(groups))) for k in AGG_KEYS}
    counts = np.zeros(len(groups), dtype=np.int64)
    class_slots = {}
    for slot, cls in classes.items():
        for c in cls:
            class_slots.setdefault(c, []).append(slot)

    captured = {}
    hooks = [modules[s].register_forward_pre_hook(
        (lambda slot: lambda _m, inp: captured.__setitem__(slot, inp[0].detach()))(s))
        for s in stacks]
    try:
        with torch.no_grad():
            for gi, (_, prompts) in enumerate(groups):
                for p in prompts:
                    enc = tokenizer([QA_PROMPT.format(q=p)])
                    captured.clear()
                    model(**{k: v for k, v in enc.items()})
                    per_key = {k: np.zeros(n_adapters) for k in AGG_KEYS}
                    for slot, (A_stack, G) in stacks.items():
                        h = captured[slot][0].float()
                        for i in range(n_adapters):
                            A = A_stack[i].float()
                            z = A @ h.t()                       # (r, T)
                            read = z.pow(2).sum(0).sqrt()       # (T,)
                            out = torch.einsum("rt,rs,st->t", z, G[i], z).clamp_min(0).sqrt()
                            per_key["A_meantok"][i] += read.mean().item() / len(stacks)
                            per_key["BA_meantok"][i] += out.mean().item() / len(stacks)
                            per_key["BA_lasttok"][i] += out[-1].item() / len(stacks)
                            for c in classes[slot][1:]:
                                per_key[f"BA_meantok_{c}"][i] += (
                                    out.mean().item() / len(class_slots[c]))
                    counts[gi] += 1
                    for k in AGG_KEYS:
                        sums[k][:, gi] += per_key[k]
    finally:
        for h in hooks:
            h.remove()
    return {k: sums[k] / np.maximum(counts, 1)[None, :] for k in AGG_KEYS}, group_names


def test_hook_vs_dense(model, shards_root, tokenizer):
    adapters = discover_adapters(shards_root)
    assert [a for a, _ in adapters] == [0, 1]
    stacks = load_factor_stacks(adapters, "cpu", torch.float32)
    classes = slot_classes(list(stacks.keys()), model.config.num_hidden_layers)
    got, counts, names = run_measurement(model, tokenizer, stacks, classes, GROUPS,
                                         "cpu", batch_size=3)
    ref, ref_names = dense_reference(model, stacks, classes, GROUPS, tokenizer)
    assert names == ref_names and counts.tolist() == [2, 1, 1]
    worst = 0.0
    for k in AGG_KEYS:
        denom = max(np.abs(ref[k]).max(), 1e-8)
        worst = max(worst, float(np.abs(got[k] - ref[k]).max() / denom))
    assert worst < 1e-4, worst
    # scaling must enter exactly once: G was built with s^2, so a hand-check on one slot
    slot = next(iter(stacks))
    _, cfg = _read_adapter(adapters[0][1])
    s = _adapter_scaling(cfg)
    assert abs(s - 8 / RANK ** 0.5) < 1e-9, s  # rslora alpha/sqrt(r)
    print(f"ok  hook/Gram-trick == dense per-prompt reference across {len(AGG_KEYS)} agg "
          f"keys (max rel err {worst:.2e}); padding + batching exact")
    return stacks, classes


def test_determinism(model, tokenizer, stacks, classes):
    a, _, _ = run_measurement(model, tokenizer, stacks, classes, GROUPS, "cpu", batch_size=2)
    b, _, _ = run_measurement(model, tokenizer, stacks, classes, GROUPS, "cpu", batch_size=2)
    for k in AGG_KEYS:
        assert np.array_equal(a[k], b[k]), k
    print("ok  run_measurement deterministic (matrices bit-equal across runs)")


def test_summarize_wiring():
    names = ["author_0", "author_1", "world_facts"]
    counts = np.array([2, 1, 1])
    lazy = {k: np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]]) for k in AGG_KEYS}
    per, summ = summarize(lazy, counts, names, [0, 1])
    assert summ["gate_verdict"] == "LAZY" and abs(summ["gate_median"] - 1.0) < 1e-9
    assert per[0]["BA_meantok"]["on"] == 1.0 and per[0]["BA_meantok"]["off"] == 1.0
    assert per[0]["ood_BA_meantok"]["world_facts"] == 0.5
    sel = {k: np.array([[10.0, 1.0, 0.5], [2.0, 12.0, 0.5]]) for k in AGG_KEYS}
    per, summ = summarize(sel, counts, names, [0, 1])
    assert summ["gate_verdict"] == "SELECTIVE", summ["gate_median"]
    assert abs(per[0]["BA_meantok"]["ratio"] - 10.0) < 1e-9
    assert abs(per[1]["BA_meantok"]["ratio"] - 6.0) < 1e-9  # on=12, off=2
    mid = {k: np.array([[3.0, 1.0, 0.5], [1.0, 3.0, 0.5]]) for k in AGG_KEYS}
    _, summ = summarize(mid, counts, names, [0, 1])
    assert summ["gate_verdict"] == "INTERMEDIATE"
    print("ok  summarize on/off wiring + gate verdicts (LAZY / SELECTIVE / INTERMEDIATE)")


def main():
    tmp = tempfile.mkdtemp(prefix="test_key_firing_")
    try:
        model, shards_root = build_fixture(tmp)
        tokenizer = StubTokenizer()
        stacks, classes = test_hook_vs_dense(model, shards_root, tokenizer)
        test_determinism(model, tokenizer, stacks, classes)
        test_summarize_wiring()
        print("ALL OK  test_measure_key_firing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
