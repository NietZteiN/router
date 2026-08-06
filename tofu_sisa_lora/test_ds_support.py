"""CPU micro-regression for ds_support.py — the [ds] arm correctness gate.

Run before any GPU/SLURM job:  python test_ds_support.py

Tiny RANDOM Llama (no download, no TOFU — the test_train_anchor/test_merge_subset
fixture family), so everything runs on CPU in seconds. Covers exactly the invariants
the disjoint-support exactness claim rests on:

  1. supports: pairwise disjoint, deterministic, correct per-slot sizes; the capacity
     guard raises when pool_size*density > 1; single-slot == full-build slice.
  2. mlp_only leaves every non-MLP tensor's support empty.
  3. projection keeps EXACTLY the support: planted off-support values are zeroed
     (τ-level project_support_ and model-level project_support_model_).
  4. sparse (int32 idx, fp32 val) round-trip is bit-exact.
  5. training (ds_one_task): τ is byte-deterministic, bit-confined to S_a
     (energy_in_support verified, not assumed).
  6. merged-serve == θ0 + Σ dense τ (closed-form, bitwise); sparse and dense merge
     routes agree bitwise.
  7. DECONTAMINATION: merged τ̄ restricted to S_a == τ_a bitwise, for every author.
  8. SUBTRACT == recompose-without, BITWISE (torch.equal — stronger than sift's
     allclose, the point of disjoint supports); zero_region_(S_a) == the subtract.
  9. empty-slice detector flags exactly the too-small tensors.
 10. placebo materializer: equal-size seeded region, valid loadable state dicts.
 11. bake_merged: baked dir == θ0 + Σ τ bitwise; bake(all, subtract=[a]) ==
     bake(all minus a) bitwise; frozen tensors untouched.
 12. trainer glue: pool derives from merge_subset (slot mapping, non-member rejection),
     e25 budget -> 25 full-batch steps, density-suffixed tau dirs.
 13. load_ds_eval_model (the eval_tofu --ds_config seam): in-place merged model ==
     bake_merged dir BITWISE (params + logits); authors/n/subtract selection semantics
     incl. every error path; set_adapter no-op present (eval-seam contract).
 14. locality CLI: happy path exits 0 + writes reports/ds_locality.json; a PLANTED
     off-support index makes it exit nonzero with the violation recorded; bake CLI
     writes a loadable dense dir.
 15. --no_support (H-ds-1 comparator): unconstrained train (support=None) + in-job
     dense bake — baked dir == θ0+τ bitwise, τ unconfined, no tau_sparse.pt, meta
     carries the recipe + "comparator_for": "H-ds-1".
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

import ds_support as ds
import sift_masks as sm

FROZEN = ("embed_tokens", "lm_head")
POOL, DENSITY = 4, 0.02
TMP = tempfile.mkdtemp(prefix="ds_support_test_")


def tiny_cfg():
    return LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=64,
        max_position_embeddings=64,
    )


def tiny_model():
    torch.manual_seed(0)
    m = LlamaForCausalLM(tiny_cfg()).to(torch.float32)
    m.eval()
    return m


def synth_batch(task_id: int, seqlen: int = 12, bs: int = 4, vocab: int = 64):
    """Deterministic per-task (input_ids, labels) with an ignored question-span prefix."""
    g = torch.Generator().manual_seed(100 + task_id)
    ids = torch.randint(0, vocab, (bs, seqlen), generator=g)
    labels = ids.clone()
    labels[:, : seqlen // 2] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def setup():
    model = tiny_model()
    names = sm.trainable_names(model, FROZEN)
    theta0 = sm.snapshot_params(model, names)
    shapes = ds.shapes_from_model(model, names)
    return model, names, theta0, shapes


def run_task(model, theta0, support, names, t, *, seed=None, steps=3):
    return ds.ds_one_task(
        model, theta0, support, names, synth_batch(t),
        seed=(1000 + t if seed is None else seed), steps=steps, lr=1e-3, device="cpu")


def _train_pool(model, names, theta0, shapes, n_tasks=3):
    """n_tasks trained taus on the first pool slots + their supports."""
    supports = ds.support_masks(42, POOL, DENSITY, shapes)
    taus = [run_task(model, theta0, supports[t], names, t) for t in range(n_tasks)]
    return supports, taus


def test_supports_disjoint_deterministic_sized():
    _, names, _, shapes = setup()
    s1 = ds.support_masks(42, POOL, DENSITY, shapes)
    s2 = ds.support_masks(42, POOL, DENSITY, shapes)
    s3 = ds.support_masks(43, POOL, DENSITY, shapes)
    total = 0
    for n in names:
        numel = int(torch.Size(shapes[n]).numel())
        per = ds.per_author_count(numel, DENSITY)
        cat = torch.cat([s1[a][n] for a in range(POOL)])
        assert cat.numel() == POOL * per, f"size mismatch ({n})"
        assert cat.unique().numel() == cat.numel(), f"supports overlap ({n})"
        assert cat.min().item() >= 0 and cat.max().item() < numel if cat.numel() else True
        for a in range(POOL):
            assert torch.equal(s1[a][n], s2[a][n]), f"support not deterministic ({n})"
            assert torch.equal(s1[a][n], s1[a][n].sort().values), f"not sorted ({n})"
        total += per
    assert any(not torch.equal(s1[0][n], s3[0][n]) for n in names if s1[0][n].numel()), \
        "different support_seed must give different supports"
    for a in range(POOL):
        one = ds.support_mask_for_slot(42, a, POOL, DENSITY, shapes)
        assert all(torch.equal(one[n], s1[a][n]) for n in names), "slot != full-build slice"
    print(f"  [ok] supports disjoint, deterministic, sized floor(numel*d) ({total} idx/slot)")


def test_capacity_assert():
    _, _, _, shapes = setup()
    try:
        ds.support_masks(42, 20, 0.06, shapes)          # 20*0.06 = 1.2 > 1
    except ValueError as e:
        assert "capacity" in str(e)
        print("  [ok] capacity guard raises at pool_size*density > 1")
        return
    raise AssertionError("capacity violation did not raise")


def test_mlp_only():
    _, names, _, shapes = setup()
    supports = ds.support_masks(42, POOL, DENSITY, shapes, mlp_only=True)
    n_mlp = 0
    for n in names:
        for a in range(POOL):
            if ds.is_mlp_tensor(n):
                assert supports[a][n].numel() > 0, f"MLP tensor got empty support ({n})"
            else:
                assert supports[a][n].numel() == 0, f"non-MLP tensor got support ({n})"
        n_mlp += ds.is_mlp_tensor(n)
    assert n_mlp > 0, "fixture has no MLP tensors?"
    print(f"  [ok] mlp_only confines supports to the {n_mlp} MLP tensors")


def test_projection_keeps_exactly_the_support():
    model, names, theta0, shapes = setup()
    support = ds.support_masks(42, POOL, DENSITY, shapes)[0]
    # τ-level: plant all-ones, project, verify on/off split exactly.
    tau = {n: torch.ones(shapes[n]) for n in names}
    ds.project_support_(tau, support)
    for n in names:
        flat = tau[n].view(-1)
        assert int(flat.sum()) == support[n].numel(), f"projection kept wrong count ({n})"
        assert (flat[support[n]] == 1).all(), f"on-support values lost ({n})"
    # model-level: perturb everything, project, off-support must be θ0 bitwise and
    # on-support must keep the perturbed values bitwise. (NB compare against the
    # perturbed tensor, not the literal 0.25 — (θ0+0.25)−θ0 is not exactly 0.25 in fp32.)
    pert = {n: theta0[n] + torch.full(shapes[n], 0.25) for n in names}
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for n in names:
            sd[n].data = pert[n].clone()
    ds.project_support_model_(model, theta0, support, names)
    sd = dict(model.named_parameters())
    for n in names:
        p, base, keep = sd[n].data.view(-1), theta0[n].view(-1), pert[n].view(-1)
        off = torch.ones(p.numel(), dtype=torch.bool)
        off[support[n]] = False
        assert torch.equal(p[off], base[off]), f"off-support drift survived projection ({n})"
        assert torch.equal(p[~off], keep[~off]), f"on-support values clobbered ({n})"
    print("  [ok] projection keeps exactly the support (τ-level and model-level)")


def test_sparse_roundtrip():
    model, names, theta0, shapes = setup()
    support = ds.support_masks(42, POOL, DENSITY, shapes)[1]
    tau = run_task(model, theta0, support, names, 1)
    sparse = ds.sparsify(tau, support)
    assert all(r["idx"].dtype == torch.int32 and r["val"].dtype == torch.float32
               for r in sparse.values()), "storage must be (int32 idx, fp32 val)"
    path = os.path.join(TMP, "tau_rt.pt")
    ds.save_sparse_tau(path, sparse)
    back = ds.densify(ds.load_sparse_tau(path))
    for n in names:
        assert torch.equal(back[n], tau[n]), f"sparse round-trip lost bits ({n})"
    print(f"  [ok] sparse round-trip bit-exact ({ds.sparse_nbytes(sparse)} bytes)")


def test_train_locality_and_determinism():
    model, names, theta0, shapes = setup()
    support = ds.support_masks(42, POOL, DENSITY, shapes)[2]
    tau_a = run_task(model, theta0, support, names, 2, seed=7)
    tau_b = run_task(model, theta0, support, names, 2, seed=7)
    nz = 0
    for n in names:
        assert torch.equal(tau_a[n], tau_b[n]), f"τ re-derivation not byte-identical ({n})"
        nz += int((tau_a[n] != 0).sum())
    assert nz > 0, "training must move at least some entries"
    # verify (not assume) that τ lives bit-exactly inside S_a
    back = ds.densify(ds.sparsify(tau_a, support))
    for n in names:
        assert torch.equal(back[n], tau_a[n]), f"τ escaped its support ({n})"
    e = ds.energy_in_support(tau_a, support)
    assert abs(e - 1.0) < 1e-12, f"energy_in_support {e} != 1.0"
    print(f"  [ok] ds_one_task deterministic and bit-confined to S_a ({nz} active entries)")


def test_merge_serve_and_decontamination():
    model, names, theta0, shapes = setup()
    supports, taus = _train_pool(model, names, theta0, shapes)
    # dense route (sift merge_add_) and sparse route must agree bitwise
    bar_dense = ds.merge_init(theta0, names)
    bar_sparse = ds.merge_init(theta0, names)
    for t, tau in enumerate(taus):
        ds.merge_add_(bar_dense, tau, names)
        ds.merge_sparse_add_(bar_sparse, ds.sparsify(tau, supports[t]))
    for n in names:
        assert torch.equal(bar_dense[n], bar_sparse[n]), f"merge routes disagree ({n})"
    # decontamination: τ̄ restricted to S_a == τ_a bitwise for EVERY author
    for t, tau in enumerate(taus):
        for n in names:
            idx = supports[t][n]
            assert torch.equal(bar_dense[n].view(-1)[idx], tau[n].view(-1)[idx]), \
                f"merged weights contaminated on S_{t} ({n})"
    # merge-only serve == θ0 + Σ τ, bitwise, no mask
    ds.serve_merged_sum_(model, theta0, bar_dense, names)
    sd = dict(model.named_parameters())
    for n in names:
        assert torch.equal(sd[n].data, theta0[n] + bar_dense[n]), f"serve mismatch ({n})"
    print("  [ok] merged serve == θ0 + Στ; τ̄|S_a == τ_a bitwise (no contamination)")


def test_subtract_equals_recompose_without_bitwise():
    model, names, theta0, shapes = setup()
    supports, taus = _train_pool(model, names, theta0, shapes)
    bar = ds.merge_init(theta0, names)
    for t, tau in enumerate(taus):
        ds.merge_add_(bar, tau, names)
    # delete task 1 three ways; all must be BITWISE identical (the sift exactness idiom,
    # upgraded from allclose to torch.equal — disjoint supports make x-x and x±0 exact)
    sub = {n: bar[n].clone() for n in names}
    ds.merge_sparse_sub_(sub, ds.sparsify(taus[1], supports[1]))
    fresh = ds.merge_init(theta0, names)
    ds.merge_add_(fresh, taus[0], names)
    ds.merge_add_(fresh, taus[2], names)
    zeroed = {n: bar[n].clone() for n in names}
    ds.zero_region_(zeroed, supports[1])
    for n in names:
        assert torch.equal(sub[n], fresh[n]), f"subtract != recompose-without ({n})"
        assert torch.equal(zeroed[n], fresh[n]), f"zero_region_ != recompose-without ({n})"
    print("  [ok] SUBTRACT == recompose-without == zero(S_a), all bitwise (torch.equal)")


def test_empty_slice_detector():
    _, names, _, shapes = setup()
    supports = ds.support_masks(42, POOL, DENSITY, shapes)
    # ground truth: tensors with floor(numel*d) == 0 (the tiny norm vectors, 32 elems)
    expect = {n for n in names
              if ds.per_author_count(int(torch.Size(shapes[n]).numel()), DENSITY) == 0}
    assert expect, "fixture should contain too-small tensors at this density"
    flagged = ds.empty_slices(supports)
    assert {n for _, n in flagged} == expect, "detector missed / over-flagged tensors"
    assert len(flagged) == POOL * len(expect), "every slot must flag every empty tensor"
    stats = ds.support_stats(supports)
    assert all(r["n_indices"] > 0 for r in stats), "slots should still own indices"
    print(f"  [ok] empty-slice detector flags exactly {sorted(expect)}")


def test_placebo_materializer():
    model, names, theta0, shapes = setup()
    supports, taus = _train_pool(model, names, theta0, shapes)
    bar = ds.merge_init(theta0, names)
    for t, tau in enumerate(taus):
        ds.merge_add_(bar, tau, names)
    sizes = ds.region_sizes(supports[0])
    pb1 = ds.placebo_region(7, shapes, sizes)
    pb2 = ds.placebo_region(7, shapes, sizes)
    pb3 = ds.placebo_region(8, shapes, sizes)
    for n in names:
        assert pb1[n].numel() == sizes[n], f"placebo size mismatch ({n})"
        assert torch.equal(pb1[n], pb2[n]), f"placebo not deterministic ({n})"
    assert any(not torch.equal(pb1[n], pb3[n]) for n in names if pb1[n].numel()), \
        "different placebo seed must give a different region"
    # materialized states: valid dicts over the trainable names, loadable, finite loss
    frozen_bar = {n: bar[n].clone() for n in names}
    for region in (supports[0], pb1):
        state = ds.materialize_ablated(theta0, bar, region, names)
        assert sorted(state) == sorted(names)
        for n in names:
            assert state[n].shape == theta0[n].shape and state[n].dtype == theta0[n].dtype
            expect = bar[n].clone()
            expect.view(-1)[region[n]] = 0.0
            assert torch.equal(state[n], theta0[n] + expect), f"ablation math wrong ({n})"
        sm.load_params_(model, state, names)
        out = model(**synth_batch(0))
        assert torch.isfinite(out.loss), "ablated state produced non-finite loss"
    for n in names:                                   # τ̄ must be untouched
        assert torch.equal(bar[n], frozen_bar[n]), "materialize_ablated mutated τ̄"
    # author-region ablation IS the deletion: θ0 + (τ̄ − τ_0)
    state = ds.materialize_ablated(theta0, bar, supports[0], names)
    for n in names:
        assert torch.equal(state[n], theta0[n] + (bar[n] - taus[0][n])), \
            f"author ablation != subtract ({n})"
    print("  [ok] placebo/crosstalk materializer: sized, seeded, valid state dicts")


def test_bake_merged():
    model, names, theta0, shapes = setup()
    base_dir = os.path.join(TMP, "tiny_base")
    model.save_pretrained(base_dir)                   # θ0 on disk BEFORE any training
    supports, taus = _train_pool(model, names, theta0, shapes)
    tau_dirs = []
    for t, tau in enumerate(taus):
        d = os.path.join(TMP, f"tau_a{t}")
        os.makedirs(d, exist_ok=True)
        ds.save_sparse_tau(os.path.join(d, "tau_sparse.pt"), ds.sparsify(tau, supports[t]))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"support_seed": 42, "density": DENSITY, "mlp_only": False,
                       "pool_size": POOL, "slot": t}, f)
        tau_dirs.append(d)

    out_all = ds.bake_merged(os.path.join(TMP, "bake_all"), base_dir, tau_dirs,
                             frozen_substr=FROZEN)
    baked = AutoModelForCausalLM.from_pretrained(out_all, torch_dtype=torch.float32)
    bar = ds.merge_init(theta0, names)
    for t, tau in enumerate(taus):
        ds.merge_add_(bar, tau, names)
    bsd = dict(baked.named_parameters())
    for n in names:
        assert torch.equal(bsd[n].data, theta0[n] + bar[n]), f"bake != θ0+Στ ({n})"
    fsd0 = dict(tiny_model().named_parameters())      # same init seed as the saved base
    for n, p in bsd.items():
        if any(s in n for s in FROZEN):
            assert torch.equal(p.data, fsd0[n].data), f"bake touched frozen tensor ({n})"
    assert os.path.exists(os.path.join(out_all, "bake_meta.json"))

    # deletion at the bake level: subtract == recompose-without, bitwise on disk
    out_sub = ds.bake_merged(os.path.join(TMP, "bake_sub"), base_dir, tau_dirs,
                             subtract=[tau_dirs[1]], frozen_substr=FROZEN)
    out_wo = ds.bake_merged(os.path.join(TMP, "bake_wo"), base_dir,
                            [tau_dirs[0], tau_dirs[2]], frozen_substr=FROZEN)
    sd_sub = dict(AutoModelForCausalLM.from_pretrained(
        out_sub, torch_dtype=torch.float32).named_parameters())
    sd_wo = dict(AutoModelForCausalLM.from_pretrained(
        out_wo, torch_dtype=torch.float32).named_parameters())
    for n in sd_sub:
        assert torch.equal(sd_sub[n].data, sd_wo[n].data), \
            f"bake(all, subtract=a) != bake(all minus a) ({n})"

    # provenance guard: a mixed-density tau must be refused
    bad = os.path.join(TMP, "tau_bad")
    shutil.copytree(tau_dirs[0], bad)
    with open(os.path.join(bad, "meta.json"), "w") as f:
        json.dump({"support_seed": 42, "density": DENSITY * 2, "mlp_only": False,
                   "pool_size": POOL, "slot": 3}, f)
    try:
        ds.bake_merged(os.path.join(TMP, "bake_bad"), base_dir, tau_dirs + [bad],
                       frozen_substr=FROZEN)
        raise AssertionError("mixed-density bake did not raise")
    except ValueError:
        pass
    print("  [ok] bake_merged: dense checkpoint == θ0+Στ; subtract == without, bitwise")


def _save_tiny_tokenizer(base_dir):
    """Offline stub tokenizer (WordLevel) so AutoTokenizer loads from the tiny base."""
    from tokenizers import Tokenizer, models as tok_models
    from transformers import PreTrainedTokenizerFast
    vocab = {f"tok{i}": i for i in range(60)}
    vocab.update({"<unk>": 60, "<pad>": 61, "</s>": 62})
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(tok_models.WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    tok.save_pretrained(base_dir)


def _save_pool_taus(model, names, theta0, shapes, out_dir, pool, n_tasks):
    """Train n_tasks taus on the FIRST pool authors and save them at the
    train_ds_support layout {out_dir}/ds/tau_a{author}/ (slot = pool position)."""
    supports = ds.support_masks(42, POOL, DENSITY, shapes)
    taus = {}
    for slot in range(n_tasks):
        author = pool[slot]
        tau = run_task(model, theta0, supports[slot], names, slot)
        d = os.path.join(out_dir, "ds", f"tau_a{author}")
        os.makedirs(d, exist_ok=True)
        ds.save_sparse_tau(os.path.join(d, "tau_sparse.pt"),
                           ds.sparsify(tau, supports[slot]))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"support_seed": 42, "density": DENSITY, "mlp_only": False,
                       "pool_size": POOL, "slot": slot}, f)
        taus[author] = tau
    return supports, taus


def test_load_ds_eval_model():
    from merge_subset import subset_authors
    model, names, theta0, shapes = setup()
    base_dir = os.path.join(TMP, "tiny_base_eval")
    model.save_pretrained(base_dir)                   # θ0 on disk BEFORE any training
    _save_tiny_tokenizer(base_dir)
    pool = [int(a) for a in subset_authors(42, POOL)]
    out = os.path.join(TMP, "eval_out")
    _, taus = _save_pool_taus(model, names, theta0, shapes, out, pool, n_tasks=3)
    cfg = {"model_name": base_dir, "out_dir": out, "arm": "ds", "pool_seed": 42,
           "pool_size": POOL, "support_seed": 42, "density": DENSITY,
           "frozen_substr": list(FROZEN)}

    def expect_state(authors):
        exp = {}
        for n in names:
            t = theta0[n].clone()
            for a in authors:
                t = t + taus[a][n]
            exp[n] = t
        return exp

    # (13a) in-place model == bake_merged dir, bitwise (params AND logits)
    served, tok = ds.load_ds_eval_model(cfg, n=3, device="cpu")
    assert tok is not None and tok.pad_token is not None
    assert callable(served.set_adapter) and served.set_adapter("anything") is None
    baked_dir = ds.bake_merged(os.path.join(TMP, "bake_eval"), base_dir,
                               [os.path.join(out, "ds", f"tau_a{a}") for a in pool[:3]],
                               frozen_substr=FROZEN)
    baked = AutoModelForCausalLM.from_pretrained(baked_dir, torch_dtype=torch.float32)
    ssd, bsd = dict(served.named_parameters()), dict(baked.named_parameters())
    for n in ssd:
        assert torch.equal(ssd[n].data.cpu(), bsd[n].data), f"in-place != baked ({n})"
    batch = synth_batch(0)
    with torch.no_grad():
        lo_s = served(**batch).logits.cpu()
        lo_b = baked(**batch).logits
    assert torch.equal(lo_s, lo_b), "in-place logits != baked-dir logits"

    # (13b) selection semantics: n = first-N of the derived pool; authors explicit;
    # subtract == compose-without, bitwise (the O(1) deletion claim at the eval seam)
    exp3 = expect_state(pool[:3])
    for n in names:
        assert torch.equal(ssd[n].data.cpu(), exp3[n]), f"n=3 selection wrong ({n})"
    m_auth, _ = ds.load_ds_eval_model(cfg, authors=f"{pool[0]},{pool[2]}", device="cpu")
    exp02 = expect_state([pool[0], pool[2]])
    asd = dict(m_auth.named_parameters())
    for n in names:
        assert torch.equal(asd[n].data.cpu(), exp02[n]), f"authors selection wrong ({n})"
    m_sub, _ = ds.load_ds_eval_model(cfg, n=3, subtract=str(pool[1]), device="cpu")
    dsd = dict(m_sub.named_parameters())
    for n in dsd:
        assert torch.equal(dsd[n].data.cpu(), asd[n].data.cpu()), \
            f"subtract != compose-without ({n})"

    # (13c) error paths — mirror linear_tv.resolve_compose semantics
    for bad, exc in ((dict(authors=str(pool[0]), n=2), ValueError),   # both selectors
                     (dict(), ValueError),                            # neither
                     (dict(n=0), ValueError),                         # out of range
                     (dict(n=POOL + 1), ValueError),                  # > pool_size
                     (dict(n=2, subtract="199"), ValueError),         # non-pool subtract
                     (dict(n=POOL), FileNotFoundError)):              # pool[3] untrained
        try:
            ds.load_ds_eval_model(cfg, **bad)
            raise AssertionError(f"load_ds_eval_model accepted {bad}")
        except exc:
            pass
    print("  [ok] load_ds_eval_model: in-place == bake bitwise; selection + errors; "
          "set_adapter no-op")


def test_locality_cli_and_bake():
    from merge_subset import subset_authors
    model, names, theta0, shapes = setup()
    base_dir = os.path.join(TMP, "tiny_base_loc")
    model.save_pretrained(base_dir)
    _save_tiny_tokenizer(base_dir)
    pool = [int(a) for a in subset_authors(42, POOL)]
    out = os.path.join(TMP, "loc_out")
    supports, _ = _save_pool_taus(model, names, theta0, shapes, out, pool, n_tasks=POOL)
    cfg_path = os.path.join(TMP, "ds_cfg.json")
    with open(cfg_path, "w") as f:
        json.dump({"model_name": base_dir, "out_dir": out, "arm": "ds",
                   "pool_seed": 42, "pool_size": POOL, "support_seed": 42,
                   "density": DENSITY, "mlp_only": False,
                   "train": {"epochs": 3, "lr": 1e-3, "seed": 0},
                   "frozen_substr": list(FROZEN)}, f)

    # (14a) happy path: rc 0, report written, invariants recorded as clean
    rc = ds.main(["locality", "--config", cfg_path])
    assert rc == 0, "clean pool must pass the locality gate"
    rep_path = os.path.join(out, "reports", "ds_locality.json")
    with open(rep_path) as f:
        rep = json.load(f)
    assert rep["disjoint_ok"] and not rep["violations"]
    assert len(rep["authors"]) == POOL
    assert all(r["subset_ok"] and abs(r["energy_in_owned"] - 1.0) < 1e-12
               for r in rep["authors"])

    # (14b) planted violation: move one stored index of author pool[0] onto an index
    # OWNED BY slot 1 -> subset breach + cross-author collision -> nonzero exit
    victim = os.path.join(out, "ds", f"tau_a{pool[0]}", "tau_sparse.pt")
    sparse = ds.load_sparse_tau(victim)
    tname = next(n for n in sorted(sparse) if sparse[n]["idx"].numel() > 0
                 and supports[1][n].numel() > 0)
    orig = {k: v for k, v in sparse[tname].items()}
    sparse[tname]["idx"] = sparse[tname]["idx"].clone()
    sparse[tname]["idx"][0] = int(supports[1][tname][0])   # slot 1's territory
    ds.save_sparse_tau(victim, sparse)
    rc = ds.main(["locality", "--config", cfg_path])
    assert rc != 0, "planted off-support index must fail the gate"
    with open(rep_path) as f:
        rep = json.load(f)
    assert rep["violations"], "violation must be recorded in the report"
    assert any("OUTSIDE owned" in v for v in rep["violations"])
    assert not rep["disjoint_ok"], "cross-author collision must be detected"
    sparse[tname] = orig
    ds.save_sparse_tau(victim, sparse)                     # restore
    assert ds.main(["locality", "--config", cfg_path]) == 0, "restore must pass again"

    # (14c) bake CLI: writes a loadable dense dir == θ0 + τ_{pool[0]} + τ_{pool[1]}
    bake_out = os.path.join(TMP, "cli_bake")
    rc = ds.main(["bake", "--config", cfg_path, "--n", "2", "--out", bake_out])
    assert rc == 0 and os.path.exists(os.path.join(bake_out, "bake_meta.json"))
    AutoModelForCausalLM.from_pretrained(bake_out, torch_dtype=torch.float32)
    print("  [ok] locality CLI: clean pass, planted violation caught (nonzero exit), "
          "bake CLI writes a loadable dir")


def test_trainer_glue():
    import train_ds_support as tds
    from merge_subset import probe_authors, subset_authors
    cfg = {"pool_seed": 42, "pool_size": 20, "out_dir": os.path.join(TMP, "out"),
           "density": 0.005, "train": {"epochs": 25, "lr": 1e-4, "seed": 42}}
    pool = tds.pool_authors(cfg)
    assert pool == subset_authors(42, 20), "pool must derive from merge_subset"
    assert pool[:5] == probe_authors(42, 20, 5), "probes must head the pool"
    assert tds.pool_slot(cfg, pool[0]) == 0 and tds.pool_slot(cfg, pool[4]) == 4
    try:
        tds.pool_slot(cfg, 199)                       # held-out author, never in the pool
        raise AssertionError("non-pool author did not raise")
    except SystemExit:
        pass
    assert tds.train_steps(cfg) == 25, "e25 budget must be 25 full-batch steps"
    assert tds.tau_dir(cfg, 82).endswith("ds/tau_a82")
    assert tds.tau_dir(cfg, 82, 0.001).endswith("ds/tau_a82_d0.001")
    assert tds.tau_dir(cfg, 82, 0.005) == tds.tau_dir(cfg, 82)
    # --no_support comparator dir convention (H-ds-1; consumed by the driver's
    # iso_dsunc_a<p> model: rows) + its provenance key
    assert tds.unc_model_dir(cfg, 82).endswith(
        os.path.join("ds_unconstrained", "a82_model"))
    key = tds._unc_resume_key(cfg, 82)
    assert key["no_support"] is True and key["steps"] == 25 and key["seed"] == 42 + 82
    assert "support_seed" not in key and "density" not in key
    print(f"  [ok] trainer glue: pool={pool[:5]}..., slot map, 25 steps, density dirs")


def test_no_support_comparator():
    """--no_support (H-ds-1 comparator): unconstrained train + in-job dense bake."""
    import train_ds_support as tds
    from merge_subset import subset_authors
    model, names, theta0, shapes = setup()
    base_dir = os.path.join(TMP, "tiny_base_unc")
    model.save_pretrained(base_dir)                   # θ0 on disk BEFORE any training
    _save_tiny_tokenizer(base_dir)
    pool = [int(a) for a in subset_authors(42, POOL)]
    author = pool[0]
    out = os.path.join(TMP, "unc_out")
    cfg = {"model_name": base_dir, "out_dir": out, "arm": "ds", "pool_seed": 42,
           "pool_size": POOL, "support_seed": 42, "density": DENSITY,
           "train": {"epochs": 3, "lr": 1e-3, "seed": 100},
           "frozen_substr": list(FROZEN)}

    losses = []
    outd = tds.train_and_bake_unconstrained(
        cfg, author, model, names, theta0, synth_batch(0), device="cpu",
        loss_log=losses)
    assert outd == tds.unc_model_dir(cfg, author)
    assert os.path.exists(os.path.join(outd, "config.json")), "baked dir missing"
    assert os.path.exists(os.path.join(outd, "meta.json"))
    assert losses and len(losses) == 3
    # NO sparse tau anywhere: the dense comparator never enters a ds merge
    assert not os.path.exists(os.path.join(outd, "tau_sparse.pt"))
    assert not os.path.exists(os.path.join(out, "ds", f"tau_a{author}", "tau_sparse.pt"))

    # baked == θ0 + τ: recover the τ by re-deriving with the identical recipe
    # (ds_one_task support=None is byte-deterministic, proven for the constrained
    # path in test_train_locality_and_determinism — same loop, no projection)
    tau_ref = ds.ds_one_task(
        model, theta0, None, names, synth_batch(0),
        seed=cfg["train"]["seed"] + author, steps=3, lr=1e-3, device="cpu")
    baked = AutoModelForCausalLM.from_pretrained(outd, torch_dtype=torch.float32)
    bsd = dict(baked.named_parameters())
    for n in names:
        assert torch.equal(bsd[n].data, theta0[n] + tau_ref[n]), f"baked != θ0+τ ({n})"
    # the comparator τ is genuinely UNCONSTRAINED: energy escapes any single owned S_a
    e = ds.energy_in_support(tau_ref, ds.support_masks(42, POOL, DENSITY, shapes)[0])
    assert 0.0 < e < 1.0, f"unconstrained τ looks support-confined (energy {e})"
    # frozen tensors untouched by the bake
    fsd0 = dict(tiny_model().named_parameters())      # same init seed as the saved base
    for n, p in bsd.items():
        if any(s in n for s in FROZEN):
            assert torch.equal(p.data, fsd0[n].data), f"bake touched frozen tensor ({n})"
    # meta provenance: recipe, seed, script sha, the comparator tag
    with open(os.path.join(outd, "meta.json")) as f:
        meta = json.load(f)
    assert meta["comparator_for"] == "H-ds-1" and meta["no_support"] is True
    assert meta["seed"] == cfg["train"]["seed"] + author and meta["steps"] == 3
    assert meta["lr"] == 1e-3 and len(meta["script_sha256"]) == 64
    print("  [ok] --no_support comparator: baked dir == θ0+τ bitwise (dense, "
          "unconfined), no tau_sparse.pt, H-ds-1 meta provenance")


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    tests = [
        test_supports_disjoint_deterministic_sized,
        test_capacity_assert,
        test_mlp_only,
        test_projection_keeps_exactly_the_support,
        test_sparse_roundtrip,
        test_train_locality_and_determinism,
        test_merge_serve_and_decontamination,
        test_subtract_equals_recompose_without_bitwise,
        test_empty_slice_detector,
        test_placebo_materializer,
        test_bake_merged,
        test_load_ds_eval_model,
        test_locality_cli_and_bake,
        test_trainer_glue,
        test_no_support_comparator,
    ]
    print(f"Running {len(tests)} ds_support CPU micro-tests (tmp={TMP})...")
    for t in tests:
        t()
    shutil.rmtree(TMP, ignore_errors=True)
    print("ALL DS-SUPPORT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
