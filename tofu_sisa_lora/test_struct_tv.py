"""CPU gate for the composable_tv [wd] write-side disjoint-subspace track (no GPU, no
downloads). Run before any ctv SLURM job: python test_struct_tv.py

Tiny GQA Llama (hidden 64, 4 heads / 2 kv heads => kv d_out = 32) + tiny rslora adapters;
pool_size 4 x r_prime 8 = 32 fills the kv output EXACTLY — the same binding geometry as the
real 1B run (pool 16 x r' 32 = 512 = kv d_out). Covers:

  1. basis orthonormality, cross-author block orthogonality, byte determinism (same seed
     twice), per-layer independence; rowslice = coordinate columns.
  2. capacity assert raises when pool*r_prime > d_out.
  3. energy_in_subspace factored == dense computation on small shapes.
  4. projection idempotence (project twice == once) + exactness.
  5. StructProjectCallback keeps lora_B exactly in-subspace across 3 real Adam steps on a
     toy LM loss (and the pre-projection state IS off-subspace, so the check bites).
  6. merge-drop identity on a toy author cat: dropping author a's factors == zeroing the
     Q_a-projection of the _weighted_factor_cat sum (factored AND dense); an UNPROJECTED
     pool violates it (the identity is not vacuous).
  7. empty-slice detector flags a planted empty author (the MemSinks lesson).
  8. verify_struct end-to-end (orthblock + rowslice): healthy pools pass, placebo pair is
     written + owned-region-zeroed on disk; planted-empty and unprojected pools FAIL.
  9. the real ctv_1b_ctrl/ctv_1b_wd configs: canonical schema, runtime pool derivation,
     nesting, 1B capacity arithmetic, check_arm routing.
"""
import json
import os
import shutil
import tempfile

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from jd_collection import _adapter_scaling, _read_adapter
from merge_subset import subset_authors, _weighted_factor_cat
from struct_bases import (
    author_basis,
    basis_sha256,
    build_author_basis_map,
    canonical_slot,
    delta_fro2,
    delta_fro2_in,
    energy_in_subspace,
    module_basis,
    project_lora_B_,
    seeded_subspace,
)
from train_struct_tv import (
    StructProjectCallback,
    check_arm,
    derive_pool,
    load_ctv_config,
)
from verify_struct import empty_slice_check, verify_arm

POOL_SIZE = 4
R_PRIME = 8
RANK = 8
SEED = 42
VOCAB = 128
KV_DOUT = 32  # 2 kv heads x head_dim 16 — pool*r' fills it exactly (the binding case)

torch.manual_seed(0)
POOL = subset_authors(SEED, POOL_SIZE)  # [82, 15, 111, 177] — runtime-derived, never typed


def tiny_cfg():
    return LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,  # GQA: kv d_out 32 < q d_out 64
        vocab_size=VOCAB, max_position_embeddings=64,
    )


def lora_cfg():
    return LoraConfig(
        r=RANK, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    )


def lora_modules(model, name):
    return {n: m for n, m in model.named_modules()
            if hasattr(m, "lora_A") and name in m.lora_A}


def build_pool(mode, root, project=True, empty_author=None):
    """One tiny base + POOL_SIZE author adapters, factors seeded, lora_B projected into
    each author's block (unless project=False), saved as real dirs root/shard_<a>."""
    base = LlamaForCausalLM(tiny_cfg())
    assert base.model.layers[0].self_attn.k_proj.out_features == KV_DOUT
    model = get_peft_model(base, lora_cfg(), adapter_name=f"shard_{POOL[0]}")
    for a in POOL[1:]:
        model.add_adapter(f"shard_{a}", lora_cfg())
    gen = torch.Generator().manual_seed(7)
    for a in POOL:
        for m in lora_modules(model, f"shard_{a}").values():
            m.lora_A[f"shard_{a}"].weight.data.normal_(0.0, 0.05, generator=gen)
            m.lora_B[f"shard_{a}"].weight.data.normal_(0.0, 0.05, generator=gen)
    if project:
        for idx, a in enumerate(POOL):
            bmap = build_author_basis_map(model, SEED, idx, POOL_SIZE, R_PRIME, mode)
            project_lora_B_(model, bmap, adapter_name=f"shard_{a}")
    if empty_author is not None:
        for m in lora_modules(model, f"shard_{empty_author}").values():
            m.lora_B[f"shard_{empty_author}"].weight.data.zero_()
    os.makedirs(root, exist_ok=True)
    dirs = []
    for a in POOL:
        model.save_pretrained(root, selected_adapters=[f"shard_{a}"])
        dirs.append(os.path.join(root, f"shard_{a}"))
    return model, dirs


def toy_ctv_config(tmp, out_dir):
    """A wd-shaped ctv config for the toy pool, written + loaded through load_ctv_config
    so the loader's canonical-schema and probe-derivation asserts are exercised too."""
    cfg = {
        "model_name": "toy", "out_dir": out_dir, "arm": "wd",
        "pool_seed": SEED, "pool_size": POOL_SIZE,
        "probe_authors": POOL,  # == derive: perm[:min(pool,5)]
        "n_ladder": [1, 2, 4],
        "train": {"rank": RANK, "alpha": 16, "epochs": 1, "lr": 1e-4,
                  "rslora": True, "seed": SEED},
        "eval": {"k": 200, "forget_shard_id": 199, "cap": "smoke"},
        "retain_tr_source": "unused",
        "unlearn_tags": ["forget10"],
        "r_prime": R_PRIME, "struct_seed": SEED,
        "variants": ["orthblock", "rowslice"],
        "scale_conditions": {"orthblock": ["sum"], "rowslice": ["sum"]},
    }
    path = os.path.join(tmp, f"toy_{os.path.basename(out_dir)}.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return load_ctv_config(path)


def dense_delta(A, B, scaling):
    return float(scaling) * (B.double() @ A.double())


# ---------------------------------------------------------------------------


def test_basis_props():
    slot = "model.layers.0.self_attn.k_proj"
    Q = module_basis(SEED, slot, KV_DOUT, POOL_SIZE, R_PRIME, "orthblock")
    assert Q.shape == (KV_DOUT, POOL_SIZE * R_PRIME)
    eye_err = (Q.double().t() @ Q.double() - torch.eye(Q.shape[1], dtype=torch.float64)) \
        .abs().max().item()
    assert eye_err < 1e-6, eye_err
    # byte determinism + author-free seeding: same (seed, layer) twice is torch.equal
    assert torch.equal(Q, module_basis(SEED, slot, KV_DOUT, POOL_SIZE, R_PRIME, "orthblock"))
    # live-name and canonical-name spellings hit the same seed string
    assert torch.equal(Q, module_basis(SEED, "base_model.model." + slot, KV_DOUT,
                                       POOL_SIZE, R_PRIME, "orthblock"))
    Q_other = module_basis(SEED, "model.layers.1.self_attn.k_proj", KV_DOUT,
                           POOL_SIZE, R_PRIME, "orthblock")
    assert not torch.equal(Q, Q_other), "per-layer bases must differ"
    # cross-author block orthogonality is exact-by-construction (one Q)
    cross = max((author_basis(Q, i, R_PRIME).double().t()
                 @ author_basis(Q, j, R_PRIME).double()).abs().max().item()
                for i in range(POOL_SIZE) for j in range(POOL_SIZE) if i != j)
    assert cross < 1e-6, cross
    assert basis_sha256(Q) == basis_sha256(module_basis(
        SEED, slot, KV_DOUT, POOL_SIZE, R_PRIME, "orthblock"))
    assert canonical_slot("base_model.model." + slot) == slot
    print(f"ok  orthblock basis: orthonormal ({eye_err:.1e}), cross-block orth "
          f"({cross:.1e}), byte-deterministic, per-layer independent")


def test_rowslice_basis():
    Q = module_basis(SEED, "model.layers.0.self_attn.q_proj", 64, POOL_SIZE, R_PRIME,
                     "rowslice")
    assert torch.equal(Q, torch.eye(64)[:, :POOL_SIZE * R_PRIME])
    blk = author_basis(Q, 1, R_PRIME)
    ref = torch.zeros(64, R_PRIME)
    for j in range(R_PRIME):
        ref[R_PRIME + j, j] = 1.0  # author idx 1 owns contiguous rows [8, 16)
    assert torch.equal(blk, ref)
    print("ok  rowslice basis = leading identity columns; author block = contiguous "
          "coordinate rows")


def test_capacity():
    try:
        module_basis(SEED, "model.layers.0.self_attn.k_proj", KV_DOUT,
                     POOL_SIZE + 1, R_PRIME, "orthblock")
        raise AssertionError("capacity assert did not fire")
    except ValueError as e:
        assert "capacity" in str(e), e
    # the exact-fit boundary (pool*r' == d_out) must be allowed — it is the real 1B case
    Q = module_basis(SEED, "x", KV_DOUT, POOL_SIZE, R_PRIME, "orthblock")
    assert Q.shape == (KV_DOUT, KV_DOUT)
    print("ok  capacity assert: pool*r' > d_out raises; exact fit (== d_out) allowed")


def test_energy_factored_vs_dense():
    gen = torch.Generator().manual_seed(11)
    A = torch.randn(RANK, 16, generator=gen)
    B = torch.randn(KV_DOUT, RANK, generator=gen)
    Q = seeded_subspace("energy-test", KV_DOUT, R_PRIME)
    s = 16 / (RANK ** 0.5)
    D = dense_delta(A, B, s)
    P = Q.double() @ Q.double().t()
    e_dense = float(((P @ D) ** 2).sum() / (D ** 2).sum())
    e_fact = energy_in_subspace(A, B, s, Q)
    # tolerance floor = Q's fp32 orthonormality error (~1e-7): dense uses QQ^T, factored
    # uses Q^T B — identical only up to (Q^T Q - I)
    assert abs(e_dense - e_fact) < 1e-6, (e_dense, e_fact)
    assert abs(delta_fro2(A, B, s) / float((D ** 2).sum()) - 1.0) < 1e-6
    assert abs(delta_fro2_in(A, B, s, Q) / float(((P @ D) ** 2).sum()) - 1.0) < 1e-6
    # zero delta lies in every subspace (the documented convention)
    assert energy_in_subspace(A, torch.zeros_like(B), s, Q) == 1.0
    print(f"ok  energy_in_subspace factored == dense ({abs(e_dense - e_fact):.1e})")


def test_projection_idempotent(tmp):
    for mode in ("orthblock", "rowslice"):
        model, _ = build_pool(mode, os.path.join(tmp, f"idem_{mode}"), project=False)
        a = POOL[2]
        bmap = build_author_basis_map(model, SEED, 2, POOL_SIZE, R_PRIME, mode)
        project_lora_B_(model, bmap, adapter_name=f"shard_{a}")
        once = {n: m.lora_B[f"shard_{a}"].weight.data.clone()
                for n, m in lora_modules(model, f"shard_{a}").items()}
        project_lora_B_(model, bmap, adapter_name=f"shard_{a}")
        for n, m in lora_modules(model, f"shard_{a}").items():
            W = m.lora_B[f"shard_{a}"].weight.data
            assert torch.allclose(W, once[n], atol=1e-6, rtol=1e-5), n
            e = energy_in_subspace(m.lora_A[f"shard_{a}"].weight.data, W, 1.0, bmap[n])
            assert e >= 1 - 1e-6, (mode, n, e)
        # a missing basis entry must raise, not silently skip
        short = dict(list(bmap.items())[:-1])
        try:
            project_lora_B_(model, short, adapter_name=f"shard_{a}")
            raise AssertionError("missing-basis KeyError did not fire")
        except KeyError:
            pass
    print("ok  projection idempotent (twice == once), exact in-subspace, missing basis raises")


def test_callback_subspace():
    for mode in ("orthblock", "rowslice"):
        base = LlamaForCausalLM(tiny_cfg())
        model = get_peft_model(base, lora_cfg())  # adapter "default", lora_B zero-init
        bmap = build_author_basis_map(model, SEED, 1, POOL_SIZE, R_PRIME, mode)
        cb = StructProjectCallback(bmap)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        gen = torch.Generator().manual_seed(5)
        x = torch.randint(1, VOCAB, (2, 12), generator=gen)
        model.train()
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            model(input_ids=x, labels=x).loss.backward()
            opt.step()
            offs = []
            for n, m in lora_modules(model, "default").items():
                B = m.lora_B["default"].weight.data
                if float(B.abs().sum()) == 0.0:
                    continue
                offs.append(1.0 - energy_in_subspace(
                    m.lora_A["default"].weight.data, B, m.scaling["default"], bmap[n]))
            assert offs and max(offs) > 1e-6, \
                f"Adam step {step} stayed in-subspace — the check does not bite"
            cb.on_step_end(None, None, None, model=model)  # the Trainer kwargs contract
            for n, m in lora_modules(model, "default").items():
                e = energy_in_subspace(m.lora_A["default"].weight.data,
                                       m.lora_B["default"].weight.data,
                                       m.scaling["default"], bmap[n])
                assert e >= 1 - 1e-6, (mode, step, n, e)
        print(f"ok  StructProjectCallback [{mode}]: 3 Adam steps drift off-subspace, "
              f"on_step_end restores B in col(Q_a) exactly")


def test_merge_drop_identity(orth_dirs, unproj_dirs):
    merged, _, _, _ = _weighted_factor_cat(orth_dirs, [1.0] * len(orth_dirs))
    dropped, _, _, _ = _weighted_factor_cat(orth_dirs[1:], [1.0] * (len(orth_dirs) - 1))
    Q_cache, worst = {}, 0.0
    for slot, (A_m, B_m) in merged.items():
        d_out = B_m.shape[0]
        Q = module_basis(SEED, slot, d_out, POOL_SIZE, R_PRIME, "orthblock")
        Q_cache[slot] = Q
        Qa = author_basis(Q, 0, R_PRIME).double()   # POOL[0] is the dropped author
        B_z = B_m.double() - Qa @ (Qa.t() @ B_m.double())
        Ad, Bd = dropped[slot]
        got = B_z @ A_m.double()
        ref = Bd.double() @ Ad.double()
        rel = float(((got - ref) ** 2).sum().sqrt() / (ref ** 2).sum().sqrt())
        worst = max(worst, rel)
    assert worst < 1e-6, worst
    # the identity must NOT hold for an unconstrained pool (otherwise it is vacuous)
    merged_u, _, _, _ = _weighted_factor_cat(unproj_dirs, [1.0] * len(unproj_dirs))
    dropped_u, _, _, _ = _weighted_factor_cat(unproj_dirs[1:], [1.0] * (len(unproj_dirs) - 1))
    worst_u = 0.0
    for slot, (A_m, B_m) in merged_u.items():
        Qa = author_basis(Q_cache[slot], 0, R_PRIME).double()
        B_z = B_m.double() - Qa @ (Qa.t() @ B_m.double())
        Ad, Bd = dropped_u[slot]
        ref = Bd.double() @ Ad.double()
        rel = float(((B_z @ A_m.double() - ref) ** 2).sum().sqrt() / (ref ** 2).sum().sqrt())
        worst_u = max(worst_u, rel)
    assert worst_u > 1e-3, worst_u
    print(f"ok  merge-drop == subspace-zeroing on the projected pool ({worst:.1e}); "
          f"violated on the unprojected pool ({worst_u:.1e})")


def test_empty_slice_detector():
    flagged, med = empty_slice_check({82: 1.0, 15: 0.0, 111: 1.1, 177: 0.9}, eps=1e-3)
    assert flagged == [15] and abs(med - 0.95) < 1e-12, (flagged, med)
    flagged, _ = empty_slice_check({82: 1.0, 15: 0.9, 111: 1.1, 177: 0.95}, eps=1e-3)
    assert flagged == []
    print("ok  empty-slice detector flags the planted empty author only")


def test_verify_end_to_end(tmp, cfg, cfg_empty, cfg_unproj):
    # healthy orthblock pool: every certificate passes; placebo pair materialized
    rep = verify_arm(cfg, "orthblock", out_path=os.path.join(tmp, "rep_orth.json"))
    assert rep["ok"] is True
    ch = rep["checks"]
    assert ch["own_energy"]["pass"] and ch["own_energy"]["min"] >= 1 - 1e-6
    assert ch["cross_leak"]["pass"] and ch["cross_leak"]["max"] <= 1e-8
    assert ch["merge_drop"]["pass"] and ch["merge_drop"]["max_rel_err"] <= 1e-6
    assert ch["empty_slice"]["pass"] and ch["empty_slice"]["flagged"] == []
    assert set(ch["merge_drop"]["per_author"]) == set(POOL)  # probes == the toy pool
    with open(os.path.join(tmp, "rep_orth.json")) as f:
        assert json.load(f)["ok"] is True

    pl = rep["placebos"]
    assert pl["author"] == POOL[0]
    assert pl["owned_energy_before_mean"] > 0.01       # the region really stored something
    assert pl["owned_energy_after_max"] < 1e-10        # ...and was exactly zeroed
    assert pl["rand_energy_after_max"] < 1e-10
    for d in (pl["owned_dir"], pl["rand_dir"]):
        assert os.path.exists(os.path.join(d, "adapter_model.safetensors")), d
        assert os.path.exists(os.path.join(d, "placebo_meta.json")), d
    # file round-trip: the owned placebo's delta has ~zero energy in POOL[0]'s block
    slots, pcfg = _read_adapter(pl["owned_dir"])
    assert pcfg["r"] == POOL_SIZE * RANK and not pcfg["use_rslora"]
    slot = sorted(slots)[0]
    A, B = slots[slot]
    Q = module_basis(SEED, slot, B.shape[0], POOL_SIZE, R_PRIME, "orthblock")
    e = delta_fro2_in(A, B, _adapter_scaling(pcfg), author_basis(Q, 0, R_PRIME)) \
        / max(delta_fro2(A, B, _adapter_scaling(pcfg)), 1e-30)
    assert e < 1e-8, e

    # healthy rowslice pool
    rep_r = verify_arm(cfg, "rowslice", out_path=os.path.join(tmp, "rep_row.json"))
    assert rep_r["ok"] is True and rep_r["mode"] == "rowslice"

    # planted-empty author: constraint holds vacuously but the detector must fail the run
    rep_e = verify_arm(cfg_empty, "orthblock", write_placebos=False,
                       out_path=os.path.join(tmp, "rep_empty.json"))
    assert rep_e["ok"] is False
    assert rep_e["checks"]["empty_slice"]["flagged"] == [POOL[1]]
    assert rep_e["checks"]["own_energy"]["pass"]       # zero delta -> vacuously true
    assert rep_e["checks"]["cross_leak"]["pass"]       # zero delta -> zero leak

    # unprojected pool: every constrained certificate must fail
    rep_u = verify_arm(cfg_unproj, "orthblock", write_placebos=False,
                       out_path=os.path.join(tmp, "rep_unproj.json"))
    assert rep_u["ok"] is False
    assert not rep_u["checks"]["own_energy"]["pass"]
    assert not rep_u["checks"]["cross_leak"]["pass"]
    assert not rep_u["checks"]["merge_drop"]["pass"]
    print("ok  verify_struct end-to-end: healthy orthblock+rowslice pass with placebo pair; "
          "planted-empty and unprojected pools fail the right checks")


def test_real_configs():
    here = os.path.dirname(os.path.abspath(__file__))
    ctrl = load_ctv_config(os.path.join(here, "configs", "ctv_1b_ctrl.json"))
    wd = load_ctv_config(os.path.join(here, "configs", "ctv_1b_wd.json"))
    assert ctrl["arm"] == "ctrl" and wd["arm"] == "wd"
    assert ctrl["model_name"] == wd["model_name"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert ctrl["pool_size"] == 20 and wd["pool_size"] == 16
    assert wd["r_prime"] == 32 and wd["struct_seed"] == 42
    assert wd["variants"] == ["orthblock", "rowslice"]
    assert ctrl["n_ladder"] == [1, 2, 3, 4, 6, 8, 12, 16, 20]
    assert wd["n_ladder"] == [1, 2, 3, 4, 8, 16] and max(wd["n_ladder"]) <= wd["pool_size"]
    assert ctrl["scale_conditions"] == {"control": ["mean", "sum"]}
    assert wd["scale_conditions"] == {"orthblock": ["sum"], "rowslice": ["sum"],
                                      "extras_at_n8": ["orthblock_mean"]}
    for cfg in (ctrl, wd):
        assert cfg["train"] == {"rank": 32, "alpha": 64, "epochs": 25, "lr": 1e-4,
                                "rslora": True, "seed": 42}
        assert cfg["eval"] == {"k": 200, "forget_shard_id": 199, "cap": "smoke"}
        assert cfg["unlearn_tags"] == ["forget10"]
        assert cfg["pool_seed"] == 42
        assert cfg["probe_authors"] == [82, 15, 111, 177, 76]
    assert derive_pool(wd) == derive_pool(ctrl)[:16], "wd pool must nest in ctrl pool"
    assert all(p in derive_pool(wd) for p in wd["probe_authors"])
    assert ctrl["out_dir"].endswith("_ctv_ctrl_r32_e25")
    assert wd["out_dir"].endswith("_ctv_wd_r32_e25")
    assert ctrl["struct_ref"] == {"struct_seed": 42, "pool_size": 16, "r_prime": 32,
                                  "mode": "orthblock"}
    # 1B GQA capacity arithmetic: kv d_out=512 fits pool 16 x r' 32 EXACTLY; 20 would not
    Q = module_basis(42, "model.layers.0.self_attn.k_proj", 512, 16, 32, "orthblock")
    assert Q.shape == (512, 512)
    try:
        module_basis(42, "model.layers.0.self_attn.k_proj", 512, 20, 32, "orthblock")
        raise AssertionError("ctrl pool 20 must NOT fit the wd geometry")
    except ValueError:
        pass
    # CLI-arm <-> config routing
    check_arm(ctrl, "control")
    check_arm(wd, "orthblock")
    check_arm(wd, "rowslice")
    for cfg, arm in ((ctrl, "orthblock"), (wd, "control")):
        try:
            check_arm(cfg, arm)
            raise AssertionError(f"check_arm({cfg['arm']}, {arm}) must raise")
        except ValueError:
            pass
    print("ok  real ctv configs: canonical schema, derived pools nest, capacity binds at "
          "kv 512, check_arm routing")


def main():
    tmp = tempfile.mkdtemp(prefix="test_struct_tv_")
    try:
        test_basis_props()
        test_rowslice_basis()
        test_capacity()
        test_energy_factored_vs_dense()
        test_projection_idempotent(tmp)
        test_callback_subspace()

        root = os.path.join(tmp, "pool")
        root_empty = os.path.join(tmp, "pool_empty")
        root_unproj = os.path.join(tmp, "pool_unproj")
        _, orth_dirs = build_pool("orthblock", os.path.join(root, "orthblock"))
        build_pool("rowslice", os.path.join(root, "rowslice"))
        build_pool("orthblock", os.path.join(root_empty, "orthblock"),
                   empty_author=POOL[1])
        unproj_dirs = build_pool("orthblock", os.path.join(root_unproj, "orthblock"),
                                 project=False)[1]

        test_merge_drop_identity(orth_dirs, unproj_dirs)
        test_empty_slice_detector()
        cfg = toy_ctv_config(tmp, root)
        cfg_empty = toy_ctv_config(tmp, root_empty)
        cfg_unproj = toy_ctv_config(tmp, root_unproj)
        test_verify_end_to_end(tmp, cfg, cfg_empty, cfg_unproj)
        test_real_configs()
        print("ALL OK  test_struct_tv")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
