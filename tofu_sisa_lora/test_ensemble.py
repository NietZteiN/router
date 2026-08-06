"""CPU micro-tests for ensemble.py (no downloads, no GPU).

Run after touching ensemble code: python test_ensemble.py

Tiny random Llama with k=3 rslora shard adapters (same builder pattern as
test_merge_extra.py). Verifies the drop-in contract eval_tofu relies on:
true ensemble NLL via .loss, greedy ensemble decoding via .generate(), and
that the batched (peft mixed-batch) and sequential paths are equivalent.
"""
import json
import math
import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only: never grab a login-node GPU

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from ensemble import (
    EnsembleModel,
    discover_ensemble_adapters,
    parse_ensemble_label,
)
from merge_lora import activate_label

K = 3
VOCAB = 128

torch.manual_seed(0)


def build_model(identical=False):
    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=VOCAB,
        max_position_embeddings=64,
    )
    base = LlamaForCausalLM(cfg)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    )
    model = get_peft_model(base, lora_cfg, adapter_name="shard_0")
    for i in range(1, K):
        model.add_adapter(f"shard_{i}", lora_cfg)
    gen = torch.Generator().manual_seed(42)
    for _, module in model.named_modules():
        if not hasattr(module, "lora_A") or "shard_0" not in module.lora_A:
            continue
        for i in range(K):
            name = f"shard_{i}"
            if identical and i > 0:
                module.lora_A[name].weight.data.copy_(module.lora_A["shard_0"].weight.data)
                module.lora_B[name].weight.data.copy_(module.lora_B["shard_0"].weight.data)
            else:
                for fac in (module.lora_A[name].weight, module.lora_B[name].weight):
                    fac.data.normal_(0.0, 0.05, generator=gen)
    model.eval()
    return model


def make_batch(b=2, t=10, seed=1):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, VOCAB, (b, t), generator=gen)
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :3] = -100          # masked "prompt" prefix
    if b > 1:                     # a padded row
        mask[-1, -2:] = 0
        labels[-1, -2:] = -100
    return ids, mask, labels


def manual_hf_loss(scores_logprobs_or_logits, labels, already_logprobs):
    s = scores_logprobs_or_logits[..., :-1, :].reshape(-1, VOCAB)
    t = labels[..., 1:].reshape(-1)
    if already_logprobs:
        return F.nll_loss(s, t, ignore_index=-100)
    return F.cross_entropy(s, t, ignore_index=-100)


def test_parse_and_dispatch():
    assert parse_ensemble_label("ensemble_probs") == ("probs", frozenset())
    assert parse_ensemble_label("ensemble_logits_no9") == ("logits", frozenset({9}))
    for bad in ("ensemble_mean", "ensemble_probs_no", "ensemble_", "ensembleprobs"):
        try:
            parse_ensemble_label(bad)
            raise AssertionError(f"{bad} accepted")
        except ValueError:
            pass
    model = build_model()
    res = activate_label(model, K, K - 1, "ensemble_probs")
    assert isinstance(res, EnsembleModel) and res.adapters == [f"shard_{i}" for i in range(K)]
    res = activate_label(model, K, K - 1, "ensemble_logits_no2")
    assert res.mode == "logits" and res.adapters == ["shard_0", "shard_1"]
    print("ok  label parsing + activate_label dispatch")


def test_discover_disk_check():
    model = build_model()
    with tempfile.TemporaryDirectory() as td:
        for i in range(K):
            os.makedirs(os.path.join(td, f"shard_{i}"))
            with open(os.path.join(td, f"shard_{i}", "adapter_config.json"), "w") as f:
                json.dump({}, f)
        assert discover_ensemble_adapters(model, output_dir=td) == \
            [f"shard_{i}" for i in range(K)]
        os.remove(os.path.join(td, "shard_2", "adapter_config.json"))
        try:
            discover_ensemble_adapters(model, output_dir=td)
            raise AssertionError("disk mismatch accepted")
        except RuntimeError:
            pass
    try:
        discover_ensemble_adapters(model, exclude=frozenset({0, 1}))
        raise AssertionError("single constituent accepted")
    except RuntimeError:
        pass
    print("ok  discover_ensemble_adapters disk cross-check + min-constituents")


def test_single_constituent_identity():
    model = build_model()
    ids, mask, labels = make_batch()
    model.set_adapter("shard_0")
    with torch.no_grad():
        direct = model(input_ids=ids, attention_mask=mask, labels=labels)
    for mode, transform in (("probs", lambda x: F.log_softmax(x.float(), -1)),
                            ("logits", lambda x: x.float())):
        ens = EnsembleModel(model, ["shard_0"], mode=mode)
        out = ens(input_ids=ids, attention_mask=mask, labels=labels)
        assert torch.allclose(out.logits, transform(direct.logits), atol=1e-5)
        assert torch.allclose(out.loss, direct.loss, atol=1e-5), (mode, out.loss, direct.loss)
        g_direct = model.generate(input_ids=ids[:1], attention_mask=mask[:1],
                                  max_new_tokens=5, do_sample=False, pad_token_id=0)
        g_ens = ens.generate(input_ids=ids[:1], attention_mask=mask[:1],
                             max_new_tokens=5, do_sample=False, pad_token_id=0)
        model.set_adapter("shard_0")  # generate() may have switched adapters
        assert torch.equal(g_direct, g_ens), mode
    print("ok  single-constituent ensemble == direct adapter (loss/logits/generate, both modes)")


def test_identical_adapters_identity():
    model = build_model(identical=True)
    ids, mask, labels = make_batch()
    model.set_adapter("shard_0")
    with torch.no_grad():
        direct = model(input_ids=ids, attention_mask=mask, labels=labels)
    for mode in ("probs", "logits"):
        ens = EnsembleModel(model, [f"shard_{i}" for i in range(K)], mode=mode)
        out = ens(input_ids=ids, attention_mask=mask, labels=labels)
        assert torch.allclose(out.loss, direct.loss, atol=1e-5)
    print("ok  k-identical-adapters ensemble == single adapter")


def _per_adapter_logprobs(model, names, ids, mask):
    outs = []
    with torch.no_grad():
        for n in names:
            model.set_adapter(n)
            outs.append(F.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1))
    return torch.stack(outs, dim=1)  # (B, n, T, V)


def test_logsumexp_reference_and_exclusion():
    model = build_model()
    ids, mask, labels = make_batch()
    names = [f"shard_{i}" for i in range(K)]
    stacked = _per_adapter_logprobs(model, names, ids, mask)
    ref_full = torch.logsumexp(stacked, dim=1) - math.log(K)
    ens = EnsembleModel(model, names, mode="probs")
    out = ens(input_ids=ids, attention_mask=mask, labels=labels)
    assert torch.allclose(out.logits, ref_full, atol=1e-5)
    assert torch.allclose(out.logits.exp().sum(-1),
                          torch.ones_like(out.logits[..., 0]), atol=1e-4)
    assert torch.allclose(out.loss, manual_hf_loss(ref_full, labels, True), atol=1e-6)
    # Exclusion == manual 2-way average, != full 3-way.
    ens2 = EnsembleModel(model, ["shard_0", "shard_1"], mode="probs")
    ref2 = torch.logsumexp(stacked[:, :2], dim=1) - math.log(2)
    out2 = ens2(input_ids=ids, attention_mask=mask)
    assert torch.allclose(out2.logits, ref2, atol=1e-5)
    assert not torch.allclose(out2.logits, ref_full, atol=1e-3)
    # logits-mode loss replication too.
    ensl = EnsembleModel(model, names, mode="logits")
    outl = ensl(input_ids=ids, attention_mask=mask, labels=labels)
    assert torch.allclose(outl.loss, manual_hf_loss(outl.logits, labels, False), atol=1e-6)
    print("ok  logsumexp reference, normalization, exclusion, HF-loss replication")


def test_batched_equals_sequential():
    model = build_model()
    ids, mask, labels = make_batch(b=3, t=12, seed=5)
    names = [f"shard_{i}" for i in range(K)]
    for mode in ("probs", "logits"):
        fast = EnsembleModel(model, names, mode=mode)
        assert fast._mixed_batch, "peft mixed-batch probe failed on CPU — investigate"
        slow = EnsembleModel(model, names, mode=mode, max_batched_rows=0)
        of = fast(input_ids=ids, attention_mask=mask, labels=labels)
        os_ = slow(input_ids=ids, attention_mask=mask, labels=labels)
        assert torch.allclose(of.logits, os_.logits, atol=1e-5), mode
        assert torch.allclose(of.loss, os_.loss, atol=1e-6), mode
        gf = fast.generate(input_ids=ids[:1], attention_mask=mask[:1],
                           max_new_tokens=6, do_sample=False, pad_token_id=0)
        gs = slow.generate(input_ids=ids[:1], attention_mask=mask[:1],
                           max_new_tokens=6, do_sample=False, pad_token_id=0)
        assert torch.equal(gf, gs), f"{mode}: generate paths diverge"
    print("ok  batched (mixed-batch) == sequential path (forward + generate, both modes)")


def test_generate_semantics():
    model = build_model()
    names = [f"shard_{i}" for i in range(K)]
    ens = EnsembleModel(model, names, mode="probs")
    ids = torch.randint(3, VOCAB, (1, 8), generator=torch.Generator().manual_seed(9))
    mask = torch.ones_like(ids)
    # Manual greedy reference using the ensemble forward itself.
    cur, cur_mask = ids.clone(), mask.clone()
    for _ in range(4):
        with torch.no_grad():
            scores = ens(input_ids=cur, attention_mask=cur_mask).logits[:, -1]
        tok = scores.argmax(-1, keepdim=True)
        cur = torch.cat([cur, tok], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(tok)], dim=1)
    out = ens.generate(input_ids=ids, attention_mask=mask, max_new_tokens=4,
                       do_sample=False, pad_token_id=0)
    assert torch.equal(out, cur), "generate != manual greedy ensemble loop"
    assert out.shape == (1, 12)
    # eos: make the first generated token the eos -> exactly one new token, eos included.
    first_tok = int(cur[0, 8].item())
    out_eos = ens.generate(input_ids=ids, attention_mask=mask, max_new_tokens=4,
                           do_sample=False, eos_token_id=first_tok, pad_token_id=first_tok)
    assert out_eos.shape == (1, 9) and int(out_eos[0, -1]) == first_tok
    seq = EnsembleModel(model, names, mode="probs", max_batched_rows=0)
    seq._mixed_batch = False
    out_eos_seq = seq.generate(input_ids=ids, attention_mask=mask, max_new_tokens=4,
                               do_sample=False, eos_token_id=first_tok, pad_token_id=first_tok)
    assert torch.equal(out_eos, out_eos_seq)
    # Determinism.
    again = ens.generate(input_ids=ids, attention_mask=mask, max_new_tokens=4,
                         do_sample=False, pad_token_id=0)
    assert torch.equal(out, again)
    print("ok  generate: manual-loop equality, eos append-then-stop (both paths), determinism")


def test_modes_differ():
    model = build_model()
    ids, mask, labels = make_batch()
    names = [f"shard_{i}" for i in range(K)]
    lp = EnsembleModel(model, names, mode="probs")(input_ids=ids, attention_mask=mask, labels=labels)
    lg = EnsembleModel(model, names, mode="logits")(input_ids=ids, attention_mask=mask, labels=labels)
    assert not torch.allclose(lp.loss, lg.loss, atol=1e-6), "probs == logits on random adapters?"
    print("ok  probs and logits modes differ on non-identical adapters")


if __name__ == "__main__":
    test_parse_and_dispatch()
    test_discover_disk_check()
    test_single_constituent_identity()
    test_identical_adapters_identity()
    test_logsumexp_reference_and_exclusion()
    test_batched_equals_sequential()
    test_generate_semantics()
    test_modes_differ()
    print("ALL ENSEMBLE TESTS PASSED")
