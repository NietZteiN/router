"""CPU micro-tests for merge_subset.py (no downloads, no GPU).

Run before any nmerge SLURM job: python test_merge_subset.py

Builds a tiny random Llama with K=3 rslora shard adapters (the test_merge_extra fixture),
SAVES them as real adapter dirs, and checks the materialize pipeline end-to-end:

  1. additive_mean materialized dir, loaded through a real PeftModel round-trip, serves an
     effective delta == (1/N) sum_i scaling_i B_i A_i (catches output-scaling bugs — the
     rslora alpha/sqrt(r) subtlety — that raw-factor comparisons miss).
  2. The factored QR+SVD compression == dense truncated-SVD reference at the same rank,
     and the retained-energy diagnostic is sane.
  3. The CPU dare_ties path (load adapter dirs -> merge_lora.merge_shards -> save ->
     reload) reproduces the in-model merge bit-for-bit at the same torch seed.
  4. Nested-subset determinism: subset(N) is a prefix chain, probes are its head,
     N=200 = all authors, and the permutation never contains author 199.
  5. Centered merges (2026-07-15): the DEGENERACY identities — S = exact subset mean makes
     the centered sum reduce to additive_mean (the rejected literal §6.1 formula), rho=0 ==
     the naive unit sum, pool==subset == the mean — plus closed-form correctness of
     centered_pool / centered_lowrank against independent dense math, the PeftModel
     round-trip, recompute determinism (the deletion contract), and the new config's specs.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

import merge_lora
from jd_collection import _adapter_scaling, _read_adapter
from merge_subset import (
    N_AUTHORS,
    author_permutation,
    load_config,
    merge_additive_mean,
    merge_additive_sum,
    merge_centered_lowrank,
    merge_centered_pool,
    merge_label,
    probe_authors,
    save_peft_adapter,
    subset_authors,
    write_effective_adapter,
    _canonical_label,
    _merge_specs,
    _subset_arms,
)

K = 3
RANK = 8
VOCAB = 128

torch.manual_seed(0)


def tiny_cfg():
    return LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=VOCAB,
        max_position_embeddings=64,
    )


def build_model():
    """Tiny base + K rslora shard adapters with random factors (test_merge_extra fixture)."""
    base = LlamaForCausalLM(tiny_cfg())
    lora_cfg = LoraConfig(
        r=RANK, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    )
    model = get_peft_model(base, lora_cfg, adapter_name="shard_0")
    for i in range(1, K):
        model.add_adapter(f"shard_{i}", lora_cfg)
    gen = torch.Generator().manual_seed(42)
    for module in lora_modules(model):
        for i in range(K):
            for fac in (module.lora_A[f"shard_{i}"].weight, module.lora_B[f"shard_{i}"].weight):
                fac.data.normal_(0.0, 0.05, generator=gen)
    return model


def lora_modules(model, name="shard_0"):
    return [m for _, m in model.named_modules()
            if hasattr(m, "lora_A") and name in m.lora_A]


def eff_delta(module, name):
    return module.scaling[name] * (
        module.lora_B[name].weight.data.float() @ module.lora_A[name].weight.data.float()
    )


def save_shard_dirs(model, root):
    dirs = []
    for i in range(K):
        model.save_pretrained(root, selected_adapters=[f"shard_{i}"])
        dirs.append(os.path.join(root, f"shard_{i}"))
    return dirs


def load_materialized(adapter_dir):
    """Fresh tiny base + ONE materialized adapter (the eval-time --preloaded_adapter path)."""
    base = LlamaForCausalLM(tiny_cfg())
    return PeftModel.from_pretrained(base, adapter_dir, adapter_name="m")


def svd_trunc(mat, rank):
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vh[:rank]


def max_rel_err(got, ref):
    denom = ref.abs().max().clamp_min(1e-8)
    return ((got - ref).abs().max() / denom).item()


def test_additive_mean_roundtrip(model, shard_dirs, tmp):
    """Materialized additive_mean == (1/N) sum_i s_i B_i A_i through a live PeftModel."""
    merged, ref_cfg, out_rank, meta = merge_additive_mean(shard_dirs)
    assert out_rank == K * RANK, out_rank
    assert meta["svd_energy_mean"] is None
    out_dir = os.path.join(tmp, "add_exact")
    write_effective_adapter(out_dir, merged, ref_cfg, out_rank)
    with open(os.path.join(out_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    assert cfg["r"] == out_rank and cfg["lora_alpha"] == out_rank and not cfg["use_rslora"]

    served = load_materialized(out_dir)
    src_mods = lora_modules(model)
    dst_mods = lora_modules(served, name="m")
    assert len(src_mods) == len(dst_mods) and len(dst_mods) > 0
    worst = 0.0
    for sm, dm in zip(src_mods, dst_mods):
        assert abs(dm.scaling["m"] - 1.0) < 1e-9, dm.scaling["m"]
        ref = sum(eff_delta(sm, f"shard_{i}") for i in range(K)) / K
        worst = max(worst, max_rel_err(eff_delta(dm, "m"), ref))
    assert worst < 1e-5, worst
    print(f"ok  additive_mean materialize -> PeftModel round-trip (max rel err {worst:.2e})")


def test_svd_compression(model, shard_dirs, tmp):
    """Factored QR+SVD == dense truncated SVD at the same rank; energy diagnostic sane."""
    r = RANK  # compress the rank-24 cat to 8 => genuinely lossy
    merged, ref_cfg, out_rank, meta = merge_additive_mean(shard_dirs, svd_rank=r)
    assert out_rank == r
    assert meta["svd_energy_mean"] is not None and 0.0 < meta["svd_energy_min"] <= 1.0
    out_dir = os.path.join(tmp, "add_svd")
    write_effective_adapter(out_dir, merged, ref_cfg, out_rank)
    served = load_materialized(out_dir)
    worst = 0.0
    for sm, dm in zip(lora_modules(model), lora_modules(served, name="m")):
        dense = sum(eff_delta(sm, f"shard_{i}") for i in range(K)) / K
        worst = max(worst, max_rel_err(eff_delta(dm, "m"), svd_trunc(dense, r)))
    assert worst < 1e-4, worst
    print(f"ok  factored SVD == dense truncated-SVD reference at r={r} "
          f"(max rel err {worst:.2e}; energy mean {meta['svd_energy_mean']:.3f})")


def test_dare_roundtrip(model, shard_dirs, tmp):
    """CPU adapter-dir dare_ties == in-model merge_shards at the same torch seed."""
    torch.manual_seed(7)
    ref_name = merge_lora.merge_shards(model, K, "dare_ties", adapter_name="ref_dare")

    base = LlamaForCausalLM(tiny_cfg())
    model_b = PeftModel.from_pretrained(base, shard_dirs[0], adapter_name="shard_0")
    for i in range(1, K):
        model_b.load_adapter(shard_dirs[i], adapter_name=f"shard_{i}")
    torch.manual_seed(7)
    got_name = merge_lora.merge_shards(model_b, K, "dare_ties", adapter_name="got_dare")

    out_dir = os.path.join(tmp, "dare")
    save_peft_adapter(model_b, got_name, out_dir)
    served = load_materialized(out_dir)
    worst = 0.0
    for sm, dm in zip(lora_modules(model), lora_modules(served, name="m")):
        worst = max(worst, max_rel_err(eff_delta(dm, "m"), eff_delta(sm, ref_name)))
    assert worst < 1e-5, worst
    print(f"ok  CPU dare_ties save/reload == in-model merge at same seed (max rel err {worst:.2e})")


def test_subsets():
    perm = author_permutation(42)
    assert len(perm) == N_AUTHORS - 1
    assert 199 not in perm.tolist()
    assert (author_permutation(42) == perm).all(), "permutation not deterministic"
    prev = []
    for n in (1, 2, 4, 8, 16):
        s = subset_authors(42, n)
        assert s[: len(prev)] == prev, "subsets are not nested"
        prev = s
    assert subset_authors(42, N_AUTHORS) == list(range(N_AUTHORS))
    assert probe_authors(42, N_AUTHORS, 5) == subset_authors(42, 5)
    assert probe_authors(42, 2, 5) == subset_authors(42, 2)
    assert merge_label("additive_mean", 8, 42) == "nmerge_add_N8_s42"
    assert merge_label("additive_mean", 128, 42, svd_rank=1024) == "nmerge_add_svd1024_N128_s42"
    print(f"ok  nested subsets deterministic; probes = head; 199 held out "
          f"(perm[:5] = {perm[:5].tolist()})")


def read_dense(shard_dirs):
    """Independent per-slot dense deltas s_i * B_i @ A_i straight from the adapter files
    (same reader as the merge code, math done here — the closed-form reference)."""
    out = []
    for d in shard_dirs:
        s, cfg = _read_adapter(d)
        scale = _adapter_scaling(cfg)
        out.append({k: scale * (B.float() @ A.float()) for k, (A, B) in s.items()})
    return out


def dense_of(merged):
    return {k: (B.float() @ A.float()) for k, (A, B) in merged.items()}


def dict_max_rel_err(got, ref):
    assert set(got) == set(ref)
    return max(max_rel_err(got[k], ref[k]) for k in got)


def test_centered_degeneracy(shard_dirs):
    """The literal PATHS_FORWARD §6.1 formula IS the mean (the rejected degenerate path):
    with S = the exact subset mean, Sum - (N-1)*S == additive_mean. Endpoints of the rho
    dial: rho=0 == naive unit sum, rho>=cat rank == mean; pool==subset == mean."""
    mean_m, _, _, _ = merge_additive_mean(shard_dirs)
    mean_d = dense_of(mean_m)

    pool_m, _, _, _ = merge_centered_pool(shard_dirs, shard_dirs)
    err_pool = dict_max_rel_err(dense_of(pool_m), mean_d)
    assert err_pool < 1e-4, err_pool

    full_m, _, _, meta_full = merge_centered_lowrank(shard_dirs, K * RANK)
    err_full = dict_max_rel_err(dense_of(full_m), mean_d)
    assert err_full < 1e-4, err_full
    assert meta_full["center_energy_min"] == 1.0

    dense = read_dense(shard_dirs)
    sum_ref = {k: sum(d[k] for d in dense) for k in dense[0]}
    sum_m, _, _, _ = merge_centered_lowrank(shard_dirs, 0)
    err_sum = dict_max_rel_err(dense_of(sum_m), sum_ref)
    assert err_sum < 1e-4, err_sum
    print(f"ok  degeneracy identities: pool==subset -> mean ({err_pool:.2e}); "
          f"rho=full -> mean ({err_full:.2e}); rho=0 -> unit sum ({err_sum:.2e})")


def test_additive_sum(model, shard_dirs, tmp):
    """ctv sum mode (Wave 0): cat with [1.0]*n weights == the manual sum of effective
    deltas, materialized round-trip included; and the additive_mean path is pinned
    unchanged (mean == sum/N on the same fixture — additive_sum is a new weight vector,
    not an edit to the 1/N path Exp-6 is running on)."""
    dense = read_dense(shard_dirs)
    sum_ref = {k: sum(d[k] for d in dense) for k in dense[0]}
    got, ref_cfg, out_rank, meta = merge_additive_sum(shard_dirs)
    assert out_rank == K * RANK, out_rank
    assert meta["svd_energy_mean"] is None
    err = dict_max_rel_err(dense_of(got), sum_ref)
    assert err < 1e-4, err

    # regression pin for the in-flight Exp-6 path: additive_mean still == sum/N exactly
    mean_m, _, _, _ = merge_additive_mean(shard_dirs)
    err_mean = dict_max_rel_err(dense_of(mean_m), {k: v / K for k, v in sum_ref.items()})
    assert err_mean < 1e-4, err_mean

    out_dir = os.path.join(tmp, "add_sum")
    write_effective_adapter(out_dir, got, ref_cfg, out_rank)
    served = load_materialized(out_dir)
    worst = 0.0
    for sm, dm in zip(lora_modules(model), lora_modules(served, name="m")):
        assert abs(dm.scaling["m"] - 1.0) < 1e-9, dm.scaling["m"]
        ref = sum(eff_delta(sm, f"shard_{i}") for i in range(K))
        worst = max(worst, max_rel_err(eff_delta(dm, "m"), ref))
    assert worst < 1e-5, worst

    assert merge_label("additive_sum", 8, 42) == "nmerge_sum_N8_s42"
    assert merge_label("additive_sum", 128, 42, svd_rank=1024) == "nmerge_sum_svd1024_N128_s42"
    print(f"ok  additive_sum == manual unit sum (max rel err {err:.2e}); mean path pinned "
          f"(mean == sum/N, {err_mean:.2e}); round-trip {worst:.2e}")


def test_additive_sum_specs():
    """additive_sum in _merge_specs is opt-in: absent key => byte-identical spec lists for
    every pre-existing config; enabled => additive_mean-shaped exact/svd ladder."""
    base = {"model_name": "x", "shards_dir": "s", "out_dir": "o",
            "n_ladder": [1, 2, 4], "subset_seeds": [42], "eval": {},
            "methods": {"additive_mean": {"enabled": False}}}
    assert _merge_specs(base) == [], "additive_sum must default OFF"
    cfg = json.loads(json.dumps(base))
    cfg["methods"]["additive_sum"] = {"enabled": True, "exact_max_n": 2,
                                      "svd_rank": 16, "svd_n_values": [4]}
    specs = _merge_specs(cfg)
    assert specs == [{"method": "additive_sum", "n": 2, "seed": 42, "svd_rank": None},
                     {"method": "additive_sum", "n": 4, "seed": 42, "svd_rank": 16}], specs
    assert _subset_arms(cfg) == [("additive_sum", None)]
    assert _canonical_label(cfg, "additive_sum", 2, 42) == "nmerge_sum_N2_s42"
    assert _canonical_label(cfg, "additive_sum", 4, 42) == "nmerge_sum_svd16_N4_s42"
    print("ok  additive_sum specs opt-in (default OFF pin) + canonical labels")


def test_centered_pool_formula(shard_dirs):
    """subset {0,1}, pool {0,1,2}: M == D0 + D1 - (2-1)*mean(D0,D1,D2), densely."""
    dense = read_dense(shard_dirs)
    got, _, out_rank, _ = merge_centered_pool(shard_dirs[:2], shard_dirs)
    ref = {k: dense[0][k] + dense[1][k]
              - (dense[0][k] + dense[1][k] + dense[2][k]) / 3 for k in dense[0]}
    err = dict_max_rel_err(dense_of(got), ref)
    assert err < 1e-4, err
    assert out_rank == 5 * RANK, out_rank  # 2 subset + 3 pool factor blocks
    print(f"ok  centered_pool == Sum - (N-1)*pool-mean closed form (max rel err {err:.2e})")


def test_centered_lowrank_formula(shard_dirs):
    """M == Sum_i D_i - (N-1)*svd_trunc(mean, rho), per slot, densely."""
    rho = 2
    dense = read_dense(shard_dirs)
    mean_ref = {k: sum(d[k] for d in dense) / K for k in dense[0]}
    got, _, out_rank, meta = merge_centered_lowrank(shard_dirs, rho)
    ref = {k: sum(d[k] for d in dense) - (K - 1) * svd_trunc(mean_ref[k], rho)
           for k in dense[0]}
    err = dict_max_rel_err(dense_of(got), ref)
    assert err < 1e-4, err
    assert out_rank == K * RANK + rho, out_rank
    assert meta["svd_energy_mean"] is None
    assert 0.0 < meta["center_energy_min"] <= 1.0
    print(f"ok  centered_lowrank rho={rho} == Sum - (N-1)*P_rho(mean) closed form "
          f"(max rel err {err:.2e}; center energy mean {meta['center_energy_mean']:.3f})")


def test_centered_roundtrip(model, shard_dirs, tmp):
    """Materialized centered_lowrank serves the exact centered delta through a live
    PeftModel (scaling forced to 1.0 — the eval-time --preloaded_adapter path)."""
    rho = 2
    merged, ref_cfg, out_rank, _ = merge_centered_lowrank(shard_dirs, rho)
    out_dir = os.path.join(tmp, "cr2")
    write_effective_adapter(out_dir, merged, ref_cfg, out_rank)
    served = load_materialized(out_dir)
    worst = 0.0
    for sm, dm in zip(lora_modules(model), lora_modules(served, name="m")):
        assert abs(dm.scaling["m"] - 1.0) < 1e-9, dm.scaling["m"]
        deltas = [eff_delta(sm, f"shard_{i}") for i in range(K)]
        ref = sum(deltas) - (K - 1) * svd_trunc(sum(deltas) / K, rho)
        worst = max(worst, max_rel_err(eff_delta(dm, "m"), ref))
    assert worst < 1e-4, worst
    print(f"ok  centered_lowrank materialize -> PeftModel round-trip (max rel err {worst:.2e})")


def test_centered_determinism(shard_dirs):
    """Recompute == byte-equal factors: the deletion contract (drop author j = recompute S
    without j and re-merge) rests on the merge being a pure deterministic function."""
    a, _, _, _ = merge_centered_lowrank(shard_dirs, 2, svd_rank=RANK)
    b, _, _, _ = merge_centered_lowrank(shard_dirs, 2, svd_rank=RANK)
    for k in a:
        assert torch.equal(a[k][0], b[k][0]) and torch.equal(a[k][1], b[k][1]), k
    p, _, _, _ = merge_centered_pool(shard_dirs[:2], shard_dirs)
    q, _, _, _ = merge_centered_pool(shard_dirs[:2], shard_dirs)
    for k in p:
        assert torch.equal(p[k][0], q[k][0]) and torch.equal(p[k][1], q[k][1]), k
    print("ok  centered merges recompute-deterministic (deletion = recompute is exact)")


def test_centered_config_specs():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "configs", "nmerge_centered_7b.json")
    cfg = load_config(cfg_path)
    specs = _merge_specs(cfg)
    cpool = [s for s in specs if s["method"] == "centered_pool"]
    cr = [s for s in specs if s["method"] == "centered_lowrank"]
    assert [s["n"] for s in cpool] == [2, 3, 4, 6, 8, 12, 16, 20, 32, 64], cpool
    assert all(s["svd_rank"] == 1024 for s in cpool)
    # cr16 exact_max_n dropped 64 -> 32 after the rank-2064 fp32 eval OOM (443532);
    # N=64 is svd-served, validated by the e5 acceptance pair.
    assert [s["n"] for s in cr if s["svd_rank"] is None] == [2, 3, 4, 6, 8, 12, 16, 20, 32]
    assert [s["n"] for s in cr if s["svd_rank"]] == [64, 128, 200]
    assert all(s["rho"] == 16 for s in cr)
    assert not any(s["method"] in ("additive_mean", "dare_ties") for s in specs)
    assert merge_label("centered_pool", 8, 42, 1024) == "nmerge_cpool_svd1024_N8_s42"
    assert merge_label("centered_lowrank", 8, 42, None, 16) == "nmerge_cr16_N8_s42"
    assert merge_label("centered_lowrank", 128, 42, 1024, 16) == "nmerge_cr16_svd1024_N128_s42"
    assert _subset_arms(cfg) == [("centered_pool", None), ("centered_lowrank", 16)]
    assert _canonical_label(cfg, "centered_pool", 8, 42) == "nmerge_cpool_svd1024_N8_s42"
    assert _canonical_label(cfg, "centered_lowrank", 8, 42, 16) == "nmerge_cr16_N8_s42"
    assert _canonical_label(cfg, "centered_lowrank", 128, 42, 16) == "nmerge_cr16_svd1024_N128_s42"
    print(f"ok  centered config -> {len(cpool)} cpool + {len(cr)} cr16 merge specs; "
          f"labels + subset arms + canonical labels")


def test_config_specs():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "configs", "nmerge_interference_7b.json")
    cfg = load_config(cfg_path)
    specs = _merge_specs(cfg)
    exact = [s for s in specs if s["svd_rank"] is None]
    svd = [s for s in specs if s["svd_rank"]]
    exact_max = cfg["methods"]["additive_mean"]["exact_max_n"]
    want_exact = [n for n in cfg["n_ladder"] if 2 <= n <= exact_max]
    assert [s["n"] for s in exact] == want_exact, exact
    assert [s["n"] for s in svd] == cfg["methods"]["additive_mean"]["svd_n_values"], svd
    assert cfg["cross_check"]["enabled"]
    print(f"ok  config -> {len(exact)} exact + {len(svd)} svd merge specs (+1 cross-check in plan)")


def main():
    tmp = tempfile.mkdtemp(prefix="test_merge_subset_")
    try:
        model = build_model()
        shard_dirs = save_shard_dirs(model, os.path.join(tmp, "shards"))
        test_additive_mean_roundtrip(model, shard_dirs, tmp)
        test_svd_compression(model, shard_dirs, tmp)
        test_dare_roundtrip(model, shard_dirs, tmp)
        test_centered_degeneracy(shard_dirs)
        test_additive_sum(model, shard_dirs, tmp)
        test_additive_sum_specs()
        test_centered_pool_formula(shard_dirs)
        test_centered_lowrank_formula(shard_dirs)
        test_centered_roundtrip(model, shard_dirs, tmp)
        test_centered_determinism(shard_dirs)
        test_subsets()
        test_config_specs()
        test_centered_config_specs()
        print("ALL OK  test_merge_subset")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
