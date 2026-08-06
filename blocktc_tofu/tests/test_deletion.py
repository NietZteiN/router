"""Deletion CPU gates (DESIGN §9 gate 6).

Deletion is the O(1) unlearning op, so its identities are load-bearing:
active-mask == baked-zero holds BITWISE unconditionally (identical gemm
shapes; zeroed features contribute exact-zero addends), physical removal is
bitwise on the pinned fixture shapes at module level and value-identical at
model logits (shrunken gemms let the BLAS re-order the reduction by ~1 ulp —
a kernel property, not a module one; atol 1e-6 per DESIGN §9), the dropped
author's contribution is exactly 0, surviving slices survive bit-intact,
apply_droplist_file enforces tc_sha provenance, and the shared tail ALWAYS
survives. All CPU/fp32, seed 42.
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
    M_SHARED,
    S,
    SEED,
    VOCAB,
    feat_rows,
    make_x,
    module_forward,
    slot_of,
    trained_tc,
    wrap_tiny,
)

from tc_model import (  # noqa: E402
    apply_droplist_file,
    compute_tc_sha,
    load_tc_from_checkpoint,
    save_checkpoint,
)

DROP = [7, 42]  # global author ids (slots 1 and 4)


def _variant(tc_ref, mode):
    """One deep-copied transcoder per deletion mechanism."""
    tc = copy.deepcopy(tc_ref)
    if mode == "mask":
        assert tc.deactivate_authors(DROP) == len(DROP)
    elif mode == "remove":
        assert tc.remove_authors(DROP) == len(DROP)
    elif mode == "bake":
        with torch.no_grad():
            for a in DROP:
                rows = feat_rows(slot_of(a))
                tc.W_enc[rows] = 0.0
                tc.b_enc[rows] = 0.0
                tc.W_dec[:, :, rows] = 0.0
    return tc.eval()


# -- mask == remove == baked-zero, dropped author contributes 0 --------------

def test_mask_remove_bake_identical_module_forward():
    tc_ref = trained_tc()
    x = make_x(2, 7, seed=SEED)
    outs = {}
    for mode in ("mask", "remove", "bake"):
        with torch.no_grad():
            outs[mode] = module_forward(_variant(tc_ref, mode), x)
    # mask == bake is shape-independent bitwise (same gemms, exact-zero
    # addends both ways); remove is bitwise on this fixture shape too
    # (DESIGN §6 pins bitwise at module level; cross-shape gemms are only
    # allowed to diverge at the MODEL-logits level below).
    assert torch.equal(outs["mask"], outs["bake"])
    assert torch.equal(outs["mask"], outs["remove"])

    # non-vacuity + exact-zero contribution:
    full = copy.deepcopy(tc_ref).eval()
    with torch.no_grad():
        y_full = module_forward(full, x)
        # the dropped authors were contributing before deletion...
        assert not torch.equal(y_full, outs["mask"])
        # ...and their isolated pre-deletion contribution is nonzero,
        full.active[:] = False
        for a in DROP:
            full.active[slot_of(a)] = True
        y_dropped_pre = module_forward(full, x)
        # (the shared tail is not deactivatable, so subtract its contribution
        # by comparing against the shared-only serve of a baked twin)
        baked = _variant(tc_ref, "bake")
        baked.active[:] = False
        for a in DROP:
            baked.active[slot_of(a)] = True
        y_dropped_post = module_forward(baked, x)
    assert not torch.equal(y_dropped_pre, y_dropped_post)
    # after baking, the dropped blocks' own contribution is EXACTLY the
    # shared-only output (their features are exactly 0)
    shared_only = copy.deepcopy(tc_ref).eval()
    with torch.no_grad():
        shared_only.active[:] = False
        y_shared_only = module_forward(shared_only, x)
    assert torch.equal(y_dropped_post, y_shared_only)


def test_mask_remove_bake_identical_model_logits(tiny_model):
    tc_ref = trained_tc()
    ids = torch.randint(1, VOCAB, (2, 9),
                        generator=torch.Generator().manual_seed(SEED))

    def logits_with(tc):
        model = copy.deepcopy(tiny_model)
        wrap_tiny(model, tc)
        model.eval()
        with torch.no_grad():
            return model(input_ids=ids).logits

    out = {m: logits_with(_variant(tc_ref, m))
           for m in ("mask", "remove", "bake")}
    # mask == bake is BITWISE end-to-end: identical gemm shapes, and dropped
    # slices contribute exact-zero addends in both.
    assert torch.equal(out["mask"], out["bake"])
    # Physical removal changes the gemm shapes (F shrinks 28 -> 20) and the
    # BLAS may legitimately re-order the reduction: cross-shape bit-identity
    # is a kernel property, not a model guarantee. DESIGN §9 pins atol 1e-6
    # at the logits; the module guarantees are pinned bitwise above.
    assert torch.allclose(out["mask"], out["remove"], rtol=0.0, atol=1e-6)
    # non-vacuity: the undropped model serves different logits
    assert not torch.equal(out["mask"], logits_with(copy.deepcopy(tc_ref)))


def test_remove_authors_slices_and_shared_tail():
    tc_ref = trained_tc()
    tc = copy.deepcopy(tc_ref)
    assert tc.remove_authors(DROP) == len(DROP)
    survivors = [a for a in AUTHOR_IDS if a not in DROP]
    assert tc.author_ids.tolist() == survivors      # order preserved
    assert tc.num_authors == K - len(DROP)
    assert tc.n_features == (K - len(DROP)) * tc.m_author + M_SHARED
    keep = torch.cat([feat_rows(slot_of(a)) for a in survivors]
                     + [torch.arange(S, S + M_SHARED)])
    # surviving slices bitwise intact after the index-select (shared ALWAYS
    # survives — the tail is not deletable)
    assert torch.equal(tc.W_enc.detach(), tc_ref.W_enc.detach()[keep])
    assert torch.equal(tc.b_enc.detach(), tc_ref.b_enc.detach()[keep])
    assert torch.equal(tc.W_dec.detach(), tc_ref.W_dec.detach()[..., keep])
    # active buffer follows the survivors
    assert tc.active.tolist() == [True] * (K - len(DROP))
    # removing an id with no block is a no-op count
    assert tc.remove_authors([999]) == 0
    # idempotence: removing the same ids again drops nothing
    assert tc.remove_authors(DROP) == 0


# -- apply_droplist_file end-to-end ------------------------------------------

def test_apply_droplist_file_end_to_end(tmp_path):
    tc = trained_tc()
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir, phase="phase1")
    loaded, _, _, _ = load_tc_from_checkpoint(run_dir)

    droplist = tmp_path / "droplists" / "test.json"
    droplist.parent.mkdir()
    droplist.write_text(json.dumps({
        "tag": "test", "authors": DROP,
        "tc_sha": compute_tc_sha(loaded),
    }))
    spec = apply_droplist_file(loaded, str(droplist))
    assert spec["_dropped"] == len(DROP)
    assert spec["_mode"] == "remove"
    assert isinstance(spec["_apply_seconds"], float)
    assert spec["_apply_seconds"] >= 0.0
    survivors = [a for a in AUTHOR_IDS if a not in DROP]
    assert loaded.author_ids.tolist() == survivors
    keep = torch.cat([feat_rows(slot_of(a)) for a in survivors]
                     + [torch.arange(S, S + M_SHARED)])
    assert torch.equal(loaded.W_enc.detach(), tc.W_enc.detach()[keep])

    # mask mode flips `active` and removes nothing physically
    masked, _, _, _ = load_tc_from_checkpoint(run_dir)
    droplist2 = tmp_path / "droplists" / "mask.json"
    droplist2.write_text(json.dumps({
        "tag": "mask", "authors": DROP,
        "tc_sha": compute_tc_sha(masked),
    }))
    spec2 = apply_droplist_file(masked, str(droplist2), mode="mask")
    assert spec2["_mode"] == "mask" and spec2["_dropped"] == len(DROP)
    assert masked.num_authors == K
    assert masked.active.tolist() == [True, False, True, True, False]

    # tampered tc_sha -> refuse to apply
    bad = tmp_path / "droplists" / "bad.json"
    bad.write_text(json.dumps({
        "tag": "bad", "authors": DROP, "tc_sha": "0" * 64,
    }))
    fresh, _, _, _ = load_tc_from_checkpoint(run_dir)
    with pytest.raises(AssertionError, match="tc_sha"):
        apply_droplist_file(fresh, str(bad))
    # unknown mode refused
    with pytest.raises(AssertionError):
        apply_droplist_file(fresh, str(droplist), mode="zero")


def test_droplist_sha_binds_to_post_drop_topology(tmp_path):
    """A droplist built against the FULL checkpoint must not apply to an
    already-dropped transcoder (F changed => tc_sha changed): double
    application can never happen silently."""
    tc = trained_tc()
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir, phase="phase1")
    loaded, _, _, _ = load_tc_from_checkpoint(run_dir)
    droplist = tmp_path / "drop.json"
    droplist.write_text(json.dumps({
        "tag": "t", "authors": [7], "tc_sha": compute_tc_sha(loaded),
    }))
    apply_droplist_file(loaded, str(droplist))
    with pytest.raises(AssertionError, match="tc_sha"):
        apply_droplist_file(loaded, str(droplist))
