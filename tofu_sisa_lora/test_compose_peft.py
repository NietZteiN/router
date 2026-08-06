"""CPU micro-tests for the peft_compose bake-off. Run: python test_compose_peft.py

Guards, per arm, the load-bearing identities BEFORE any SLURM job:
  - compose math: mean rule; O(1) exact-deletion identity ((n·mean − x)/(n−1) ≡ direct mean);
    geo variant sign-fallback; shared-key mismatch raises (catches VeRA prng mismatches).
  - VeRA / IA³ end-to-end on a tiny GQA llama: save shards -> file-space compose -> the
    composed dir LOADS via PeftModel.from_pretrained and, for identical shards, the composed
    model's logits equal the single-shard model's (compose is a no-op on n copies).
  - prefix arm: PrefixConcatModel with ONE shard reproduces peft's own single-prefix forward
    logits exactly (the wrapper's mask/cache/position handling is right); two shards give
    prefix length 2P and drop-a-shard == the 1-shard composition; greedy generate runs.
  - DoRA probe: does peft 0.14 add_weighted_adapter accept use_dora adapters? Printed as
    DORA_MERGE_SUPPORTED=0/1 (a probe, not an assert — the submit script gates the arm on it).
"""
import json
import os
import shutil
import tempfile

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM
from peft import IA3Config, LoraConfig, PeftModel, PrefixTuningConfig, VeraConfig, get_peft_model

import compose_peft as cp
from prefix_concat import PrefixConcatModel

torch.manual_seed(0)


def tiny_llama():
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                      max_position_embeddings=256)
    return LlamaForCausalLM(cfg)


def _save_adapter(model, d):
    model.save_pretrained(d)
    # save_pretrained writes into d/<adapter_name>/ only for non-default names; default -> d
    assert os.path.exists(os.path.join(d, "adapter_config.json"))


def test_compose_math():
    with tempfile.TemporaryDirectory() as td:
        n, shape = 4, (3, 5)
        xs = [torch.randn(shape, dtype=torch.float32) for _ in range(n)]
        shared = torch.randn(2, 2)
        for i in range(n):
            d = os.path.join(td, f"shard_{i}"); os.makedirs(d)
            save_file({"m.vera_lambda_d": xs[i], "m.vera_A.weight": shared},
                      os.path.join(d, "adapter_model.safetensors"))
        worst = cp.verify_exact_deletion(td, n, "vera", drop=2)
        assert worst < 1e-6, worst
        states = cp._load_states(cp._shard_dirs(td, n, set()))
        comp, _ = cp.compose_states(states, "vera_lambda", "mean")
        assert torch.allclose(comp["m.vera_lambda_d"], torch.stack(xs).mean(0), atol=1e-6)
        # geo on all-positive == exact geometric mean, zero fallbacks
        pos = [x.abs() + 0.1 for x in xs]
        pstates = [{"m.vera_lambda_d": p, "m.vera_A.weight": shared} for p in pos]
        geo, fb = cp.compose_states(pstates, "vera_lambda", "geo")
        expect = torch.exp(torch.stack(pos).log().mean(0))
        assert fb == 0 and torch.allclose(geo["m.vera_lambda_d"], expect, atol=1e-5)
        # mismatched shared key must raise
        bad = [{"m.vera_lambda_d": xs[0], "m.vera_A.weight": shared},
               {"m.vera_lambda_d": xs[1], "m.vera_A.weight": shared + 1.0}]
        try:
            cp.compose_states(bad, "vera_lambda", "mean")
            raise AssertionError("mismatched shared key did not raise")
        except ValueError:
            pass
    print("  ok  compose math: mean, exact-delete identity, geo, shared-key guard")


def _roundtrip_compose(method, peft_cfg_factory, key_substr):
    """Save 2 IDENTICAL shard adapters, compose, reload; composed logits == single's."""
    with tempfile.TemporaryDirectory() as td:
        base = tiny_llama()
        # snapshot BEFORE get_peft_model — wrapping renames target keys (q_proj.base_layer.*)
        sd = {k: v.clone() for k, v in base.state_dict().items()}
        m = get_peft_model(base, peft_cfg_factory())
        for i in range(2):
            _save_adapter(m, os.path.join(td, f"shard_{i}"))
        out = os.path.join(td, "composed")
        states = cp._load_states(cp._shard_dirs(td, 2, set()))
        comp, _ = cp.compose_states(states, key_substr, "mean")
        os.makedirs(out)
        save_file(comp, os.path.join(out, "adapter_model.safetensors"))
        shutil.copy(os.path.join(td, "shard_0", "adapter_config.json"),
                    os.path.join(out, "adapter_config.json"))

        ids = torch.randint(0, 127, (1, 8))
        base_a = tiny_llama(); base_a.load_state_dict(sd)
        base_b = tiny_llama(); base_b.load_state_dict(sd)
        single = PeftModel.from_pretrained(base_a, os.path.join(td, "shard_0"))
        merged = PeftModel.from_pretrained(base_b, out)
        with torch.no_grad():
            la = single(input_ids=ids).logits
            lb = merged(input_ids=ids).logits
        assert torch.allclose(la, lb, atol=1e-5), f"{method}: composed != single on n-copies"
    print(f"  ok  {method}: saved shards -> file compose -> loads; n-copy compose == single")


def test_vera_roundtrip():
    _roundtrip_compose(
        "vera",
        lambda: VeraConfig(task_type="CAUSAL_LM", r=16, target_modules=["q_proj", "o_proj"],
                           projection_prng_key=42, save_projection=True, d_initial=0.1),
        "vera_lambda")


def test_ia3_roundtrip():
    _roundtrip_compose(
        "ia3",
        lambda: IA3Config(task_type="CAUSAL_LM",
                          target_modules=["k_proj", "v_proj", "down_proj"],
                          feedforward_modules=["down_proj"]),
        "ia3_l")


def test_prefix_concat():
    with tempfile.TemporaryDirectory() as td:
        base = tiny_llama()
        sd = base.state_dict()
        pm = get_peft_model(base, PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=4))
        _save_adapter(pm, os.path.join(td, "shard_0"))
        # a second, differently-initialized prefix
        torch.manual_seed(7)
        base2 = tiny_llama(); base2.load_state_dict(sd, strict=False)
        pm2 = get_peft_model(base2, PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=4))
        _save_adapter(pm2, os.path.join(td, "shard_1"))

        base3 = tiny_llama(); base3.load_state_dict(sd, strict=False)
        pool = PeftModel.from_pretrained(base3, os.path.join(td, "shard_0"),
                                         adapter_name="shard_0")
        pool.load_adapter(os.path.join(td, "shard_1"), adapter_name="shard_1")

        ids = torch.randint(0, 127, (1, 8))
        labels = ids.clone()

        # N=1 identity: wrapper(single shard) == peft's own prefix forward
        wrap1 = PrefixConcatModel(pool, ["shard_0"])
        pool.set_adapter("shard_0")
        with torch.no_grad():
            ref = pool(input_ids=ids, labels=labels)
            got = wrap1(input_ids=ids, labels=labels)
        assert torch.allclose(ref.logits, got.logits, atol=1e-5), "N=1 prefix identity failed"
        assert abs(ref.loss.item() - got.loss.item()) < 1e-5

        # N=2: prefix length doubles; drop-shard == 1-shard composition
        wrap2 = PrefixConcatModel(pool, ["shard_0", "shard_1"])
        cache, P = wrap2._concat_prefix(1)
        assert P == 8, f"expected concat prefix len 8, got {P}"
        with torch.no_grad():
            out2 = wrap2(input_ids=ids, labels=labels)
            assert out2.loss.isfinite()
            dropped = PrefixConcatModel(pool, ["shard_0"])
            d = dropped(input_ids=ids, labels=labels)
        assert torch.allclose(d.logits, got.logits, atol=1e-6), "drop-shard != direct 1-shard"

        gen = wrap2.generate(input_ids=ids, max_new_tokens=3)
        assert gen.shape[1] > ids.shape[1]
    print("  ok  prefix: N=1 identity vs peft forward; concat len; exact drop; generate")


def test_dora_merge_probe():
    with tempfile.TemporaryDirectory() as td:
        base = tiny_llama()
        cfg = LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, use_dora=True,
                         target_modules=["q_proj", "v_proj"], use_rslora=False)
        m = get_peft_model(base, cfg)
        _save_adapter(m, os.path.join(td, "a"))
        base2 = tiny_llama()
        pm = PeftModel.from_pretrained(base2, os.path.join(td, "a"), adapter_name="s0")
        pm.load_adapter(os.path.join(td, "a"), adapter_name="s1")
        try:
            pm.base_model.add_weighted_adapter(["s0", "s1"], weights=[0.5, 0.5],
                                               adapter_name="mrg", combination_type="linear")
            supported = 1
        except Exception as e:  # noqa: BLE001 — the probe's whole point
            supported = 0
            print(f"      (add_weighted_adapter rejected DoRA: {type(e).__name__}: {e})")
    print(f"  ok  dora probe: DORA_MERGE_SUPPORTED={supported}")
    return supported


if __name__ == "__main__":
    test_compose_math()
    test_vera_roundtrip()
    test_ia3_roundtrip()
    test_prefix_concat()
    supported = test_dora_merge_probe()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "reports", "dora_merge_probe.json"), "w") as f:
        json.dump({"dora_merge_supported": bool(supported)}, f)
    print("ALL COMPOSE_PEFT TESTS PASSED")
