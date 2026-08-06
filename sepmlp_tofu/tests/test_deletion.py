"""Deletion CPU gates (plan gate 8 + the apply-side of gate 13).

Deletion is the O(1) unlearning op, so its identities are load-bearing:
active-mask == baked-zero holds BITWISE unconditionally (identical gemm
shapes; zeroed slices contribute exact-zero addends), physical removal is
bitwise on the pinned fixture shapes and value-identical everywhere (shrunken
gemms let the BLAS re-order the reduction by 1 ulp — a kernel property, not a
module one), the dropped author's contribution is exactly 0, surviving slices
survive bit-intact, apply_droplist_file enforces bank_sha provenance, and
KV-cache serving under a droplist matches cache-free full forwards
token-for-token. All CPU/fp32, seed 42.
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from conftest import (
    ADAPTER_CFG,
    AUTHOR_IDS,
    K,
    PAD_ID,
    SEED,
    VOCAB,
    build_banks,
    make_x,
    randomize_bgate,
    randomize_down,
    slice_rows,
    slot_of,
    wrap_tiny,
)

from sepmlp_model import (  # noqa: E402
    apply_droplist_file,
    compute_bank_sha,
    load_banks_from_checkpoint,
    save_checkpoint,
)

DROP = [7, 42]  # global author ids (slots 1 and 4)


def _trained_banks():
    return randomize_bgate(randomize_down(build_banks()))


def _variant(banks_ref, mode):
    """One deep-copied bank dict per deletion mechanism."""
    banks = copy.deepcopy(banks_ref)
    for bank in banks.values():
        if mode == "mask":
            for a in DROP:
                bank.active[slot_of(a)] = False
        elif mode == "remove":
            assert bank.remove_authors(DROP) == len(DROP)
        elif mode == "zero_wdown":
            assert bank.zero_wdown_authors(DROP) == len(DROP)
        elif mode == "bake":
            with torch.no_grad():
                for a in DROP:
                    rows = slice_rows(slot_of(a))
                    bank.W_gate[rows] = 0.0
                    bank.W_up[rows] = 0.0
                    bank.b_gate[rows] = 0.0
                    bank.W_down[:, rows] = 0.0
        bank.eval()
    return banks


# -- 13. mask == remove == baked-zero, dropped author contributes 0 ---------

def test_mask_remove_bake_bitwise_identical_bank_forward():
    banks_ref = _trained_banks()
    x = make_x(2, 7, seed=SEED)
    outs = {}
    for mode in ("mask", "remove", "zero_wdown", "bake"):
        with torch.no_grad():
            outs[mode] = _variant(banks_ref, mode)[0](x, None)
    # mask == bake == zero_wdown is shape-independent bitwise (same gemm shapes;
    # dropped slices contribute exact-zero addends); remove is bitwise on this
    # fixture shape (cross-shape gemms can differ by 1 ulp on other shapes —
    # see the model-level test below).
    assert torch.equal(outs["mask"], outs["remove"])
    assert torch.equal(outs["mask"], outs["bake"])
    # zero_wdown is the paper-exact op (§3.2): zero ONLY W_down; because the
    # contribution is W_down @ act, this alone drives ad_k to exactly 0.
    assert torch.equal(outs["mask"], outs["zero_wdown"])

    # non-vacuity + exact-zero contribution:
    full = copy.deepcopy(banks_ref[0]).eval()
    with torch.no_grad():
        y_full = full(x, None)
        # the dropped authors were contributing before deletion...
        assert not torch.equal(y_full, outs["mask"])
        # ...and their isolated pre-deletion contribution is nonzero,
        full.active[:] = False
        for a in DROP:
            full.active[slot_of(a)] = True
        y_dropped_pre = full(x, None)
        assert not torch.equal(y_dropped_pre, torch.zeros_like(y_dropped_pre))
        # ...while after baking, the very same isolation is EXACTLY zero.
        baked = _variant(banks_ref, "bake")[0]
        baked.active[:] = False
        for a in DROP:
            baked.active[slot_of(a)] = True
        y_dropped_post = baked(x, None)
    assert torch.equal(y_dropped_post, torch.zeros_like(y_dropped_post))


def test_mask_remove_bake_bitwise_identical_model_logits(tiny_model):
    banks_ref = _trained_banks()
    ids = torch.randint(1, VOCAB, (2, 9),
                        generator=torch.Generator().manual_seed(SEED))

    def logits_with(banks):
        model = copy.deepcopy(tiny_model)
        wrap_tiny(model, banks)
        model.eval()
        with torch.no_grad():
            return model(input_ids=ids).logits

    out = {m: logits_with(_variant(banks_ref, m)) for m in
           ("mask", "remove", "bake")}
    # mask == bake is BITWISE end-to-end: identical gemm shapes, and dropped
    # slices contribute exact-zero addends in both.
    assert torch.equal(out["mask"], out["bake"])
    # Physical removal changes the gemm shapes (K*D columns shrink), and the
    # BLAS may legitimately re-order the reduction: cross-shape bit-identity
    # is a kernel property, not a model guarantee (measured divergence is
    # 1 ulp, ~1e-7 at the logits). The module guarantees are value-identity
    # here, plus bitwise slice survival + exact-zero dropped contribution
    # (pinned above and in test_apply_droplist_file).
    assert torch.allclose(out["mask"], out["remove"], rtol=0.0, atol=1e-6)
    # non-vacuity: the undropped model serves different logits
    assert not torch.equal(out["mask"], logits_with(copy.deepcopy(banks_ref)))


def test_zero_wdown_is_wdown_only_and_verifiable():
    """The paper-exact op zeros ONLY the dropped authors' W_down columns (the
    contribution is W_down @ act, so that alone suffices), leaves W_gate/W_up/
    b_gate untouched, keeps the slot listed, and survivors are bit-intact —
    the removal is readable directly in the stored W_down."""
    ref = _trained_banks()
    zeroed = _variant(ref, "zero_wdown")
    drop_cols = torch.cat([slice_rows(slot_of(a)) for a in DROP])
    keep_cols = torch.cat([slice_rows(slot_of(a)) for a in AUTHOR_IDS
                           if a not in DROP])
    for l, bank in zeroed.items():
        # dropped W_down columns are all zero (verifiable in stored params)...
        assert torch.equal(bank.W_down[:, drop_cols],
                           torch.zeros_like(bank.W_down[:, drop_cols]))
        # ...survivor W_down columns untouched...
        assert torch.equal(bank.W_down[:, keep_cols], ref[l].W_down[:, keep_cols])
        # ...gate/up/bias fully untouched (NOT zeroed — unlike bake)...
        assert torch.equal(bank.W_gate, ref[l].W_gate)
        assert torch.equal(bank.W_up, ref[l].W_up)
        assert torch.equal(bank.b_gate, ref[l].b_gate)
        # ...and the slot stays listed at fixed shape (in-place edit).
        assert bank.num_authors == ref[l].num_authors
        assert bank.author_ids.tolist() == ref[l].author_ids.tolist()


# -- 14. apply_droplist_file end-to-end -------------------------------------

def test_apply_droplist_file_end_to_end(tmp_path):
    banks = _trained_banks()
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(banks, dict(ADAPTER_CFG), run_dir)
    loaded, _, _ = load_banks_from_checkpoint(run_dir)

    droplist = tmp_path / "droplists" / "test.json"
    droplist.parent.mkdir()
    droplist.write_text(json.dumps({
        "tag": "test", "authors": DROP,
        "bank_sha": compute_bank_sha(loaded),
    }))
    spec = apply_droplist_file(loaded, str(droplist))

    assert spec["_dropped_per_layer"] == len(DROP)
    assert isinstance(spec["_apply_seconds"], float)
    assert spec["_apply_seconds"] >= 0.0

    survivors = [a for a in AUTHOR_IDS if a not in DROP]
    keep = torch.cat([slice_rows(slot_of(a)) for a in survivors])
    for l, bank in loaded.items():
        assert bank.num_authors == K - len(DROP)
        assert bank.author_ids.tolist() == survivors  # order preserved
        # surviving slices bitwise intact after the index-select
        assert torch.equal(bank.W_gate, banks[l].W_gate[keep])
        assert torch.equal(bank.W_up, banks[l].W_up[keep])
        assert torch.equal(bank.b_gate, banks[l].b_gate[keep])
        assert torch.equal(bank.W_down, banks[l].W_down[:, keep])

    # tampered bank_sha -> refuse to apply
    bad = tmp_path / "droplists" / "bad.json"
    bad.write_text(json.dumps({
        "tag": "bad", "authors": DROP, "bank_sha": "0" * 64,
    }))
    fresh, _, _ = load_banks_from_checkpoint(run_dir)
    with pytest.raises(AssertionError, match="bank_sha"):
        apply_droplist_file(fresh, str(bad))


# -- 15. KV-cache generate == stepwise full forwards under a droplist -------

def test_kv_cache_generate_matches_stepwise_full_forward(tiny_model, tmp_path):
    banks = _trained_banks()
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(banks, dict(ADAPTER_CFG), run_dir)
    loaded, _, _ = load_banks_from_checkpoint(run_dir)
    droplist = tmp_path / "drop.json"
    droplist.write_text(json.dumps({
        "tag": "kv", "authors": DROP, "bank_sha": compute_bank_sha(loaded),
    }))
    apply_droplist_file(loaded, str(droplist))

    wrap_tiny(tiny_model, loaded)
    tiny_model.eval()
    # tiny model: no eos/early-stop so exactly n_new greedy tokens come out
    tiny_model.generation_config.eos_token_id = None
    tiny_model.generation_config.pad_token_id = PAD_ID

    prompt = torch.randint(1, VOCAB, (1, 6),
                           generator=torch.Generator().manual_seed(SEED))
    n_new = 8

    with torch.no_grad():
        gen = tiny_model.generate(
            prompt, max_new_tokens=n_new, do_sample=False, use_cache=True,
        )

    ids = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits = tiny_model(input_ids=ids, use_cache=False).logits
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)

    assert gen.shape == (1, 6 + n_new)
    assert torch.equal(gen, ids)  # token-identical
