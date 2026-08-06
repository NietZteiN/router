"""CPU micro-tests for the §6.3 negative-anchor penalty in train_lora_shard.py.

Run before any anchored SLURM job: python test_train_anchor.py

Tiny random Llama + one rslora adapter ("default"):
  1. anchor_penalty (lora_B hooks, scaling applied once, fp32, pad-masked) == a dense
     closed-form reference sum_modules s^2 * mean-token ||B A h_t||^2 computed from
     independently captured module inputs (dropout 0 so the paths are comparable).
  2. Gradient flow: penalty.backward() alone puts nonzero grads on lora_A AND lora_B.
  3. AnchoredSFTTrainer with anchor_lambda=0 returns compute_loss bit-identical to the
     parent SFTTrainer (the flag-free frozen-recipe invariant); with lambda>0 it returns
     exactly parent_loss + lambda * anchor_penalty and cycles batches deterministically.
"""

import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from train_lora_shard import anchor_penalty, apply_anchor_to_loss

VOCAB = 64
torch.manual_seed(0)


def tiny_cfg():
    return LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=VOCAB,
        max_position_embeddings=64,
    )


def build_model():
    base = LlamaForCausalLM(tiny_cfg())
    lora_cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,  # dropout 0: hook path == dense reference
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    )
    model = get_peft_model(base, lora_cfg)
    gen = torch.Generator().manual_seed(7)
    for _, m in model.named_modules():
        if hasattr(m, "lora_A") and "default" in m.lora_A:
            m.lora_A["default"].weight.data.normal_(0.0, 0.1, generator=gen)
            m.lora_B["default"].weight.data.normal_(0.0, 0.1, generator=gen)
    return model.eval()


def make_batch(bsz=2, lens=(9, 6), T=9):
    gen = torch.Generator().manual_seed(3)
    input_ids = torch.randint(1, VOCAB, (bsz, T), generator=gen)
    mask = torch.zeros(bsz, T, dtype=torch.long)
    for i, L in enumerate(lens):
        mask[i, :L] = 1
    input_ids[mask == 0] = 0
    return {"input_ids": input_ids, "attention_mask": mask}


def dense_reference(model, batch):
    """Independent penalty: capture each LoraLayer's INPUT h, compute s^2*||B A h_t||^2."""
    caps, hooks = {}, []
    mods = {}
    for name, m in model.named_modules():
        if hasattr(m, "lora_A") and "default" in m.lora_A:
            mods[name] = m
            hooks.append(m.register_forward_pre_hook(
                (lambda n: lambda _m, inp: caps.__setitem__(n, inp[0].detach()))(name)))
    with torch.no_grad():
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    for h in hooks:
        h.remove()
    mask = batch["attention_mask"].unsqueeze(-1).float()
    denom = batch["attention_mask"].sum().float() * len(mods)
    pen = 0.0
    for name, m in mods.items():
        h = caps[name].float()
        A = m.lora_A["default"].weight.float()
        B = m.lora_B["default"].weight.float()
        s = m.scaling["default"]
        out = torch.einsum("btr,dr->btd", torch.einsum("btk,rk->btr", h, A), B)
        pen += (s ** 2) * ((out * mask) ** 2).sum() / denom
    return pen


def test_penalty_vs_dense(model):
    batch = make_batch()
    with torch.no_grad():
        got = anchor_penalty(model, batch)
    ref = dense_reference(model, batch)
    rel = abs(got.item() - ref.item()) / max(abs(ref.item()), 1e-12)
    assert rel < 1e-5, (got.item(), ref.item())
    assert got.item() > 0
    print(f"ok  anchor_penalty == dense closed form (rel err {rel:.2e}; value {got.item():.6f})")


def test_grad_flow(model):
    model.train()
    model.zero_grad()
    pen = anchor_penalty(model, make_batch())
    pen.backward()
    got_a = got_b = 0
    for _, m in model.named_modules():
        if hasattr(m, "lora_A") and "default" in m.lora_A:
            ga, gb = m.lora_A["default"].weight.grad, m.lora_B["default"].weight.grad
            got_a += int(ga is not None and ga.abs().sum() > 0)
            got_b += int(gb is not None and gb.abs().sum() > 0)
    model.zero_grad()
    model.eval()
    assert got_a > 0 and got_b > 0, (got_a, got_b)
    print(f"ok  penalty gradients reach lora_A ({got_a} modules) and lora_B ({got_b})")


def test_compute_loss_arithmetic(model):
    b1, b2 = make_batch(), make_batch(bsz=2, lens=(5, 8))
    parent = torch.tensor(2.5, requires_grad=True)
    # lambda = 0: the SAME object comes back untouched (the frozen-recipe invariant)
    out0, idx0 = apply_anchor_to_loss(parent, model, [b1, b2], 0.0, 0)
    assert out0 is parent and idx0 == 0
    out0, idx0 = apply_anchor_to_loss(parent, model, [], 3.0, 5)
    assert out0 is parent and idx0 == 5
    # lambda > 0: parent + lambda * penalty, cycling deterministically b1 -> b2 -> b1
    lam = 3.0
    with torch.no_grad():
        p1, p2 = anchor_penalty(model, b1), anchor_penalty(model, b2)
    l1, i1 = apply_anchor_to_loss(parent, model, [b1, b2], lam, 0)
    l2, i2 = apply_anchor_to_loss(parent, model, [b1, b2], lam, i1)
    l3, i3 = apply_anchor_to_loss(parent, model, [b1, b2], lam, i2)
    assert (i1, i2, i3) == (1, 2, 3)
    assert abs(l1.item() - (2.5 + lam * p1.item())) < 1e-6
    assert abs(l2.item() - (2.5 + lam * p2.item())) < 1e-6
    assert abs(l3.item() - l1.item()) < 1e-9, "batch cycling broken"
    assert l1.requires_grad
    (lo, aux), _ = apply_anchor_to_loss((parent, {"logits": None}), model, [b1], lam, 0,
                                        return_outputs=True)
    assert abs(lo.item() - l1.item()) < 1e-9 and aux == {"logits": None}
    print("ok  apply_anchor_to_loss: lambda=0 identity passthrough; lambda>0 = parent + "
          "lambda*penalty; cycling + return_outputs preserved")


def main():
    model = build_model()
    test_penalty_vs_dense(model)
    test_grad_flow(model)
    test_compute_loss_arithmetic(model)
    print("ALL OK  test_train_anchor")


if __name__ == "__main__":
    main()
