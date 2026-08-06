"""Leak-probe CPU gates (router_leak Part 3.1 producer side).

Pins, on tiny banks (no HF model, no datasets):
  (a) the per-ROW norm capture (BankState.begin_row_stats) against a literal
      dense per-author computation, including the answer-mask pooling and
      multi-layer mean;
  (b) assemble_leak_arrays: exact NPZ-contract keys/dtypes/shapes, own-norm
      NaN semantics, max_foreign vs max_surv relations, top_surv_author,
      group vocabulary;
  (c) droplist-then-probe ordering: the capture runs on the POST-drop banks
      (dropped author's column gone, own_norm NaN for its queries, sha-pinned
      droplist application) — the reference (no droplist) run keeps the
      column.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from conftest import (
    AUTHOR_IDS,
    BANK_LAYERS,
    HIDDEN,
    K,
    SEED,
    WIDTH,
    build_banks,
    make_x,
    randomize_bias,
    randomize_down,
)

from bank_layer import BankState
from measure_selectivity import LEAK_GROUPS, assemble_leak_arrays
from sepmlp_model import apply_droplist_file, compute_bank_sha, save_checkpoint, \
    load_banks_from_checkpoint

NPZ_KEYS = {"max_surv_norm", "max_foreign_norm", "top_surv_author",
            "own_norm", "group", "author_of_q", "n_surviving",
            "droplist_tag", "K"}


def leak_banks():
    return randomize_bias(randomize_down(build_banks()))


def capture_rows(banks, x, row_mask):
    """Feed x through every bank with row-stats armed (mimics the wrapped
    model's serving forward: eval mode, NO source ids) and return the
    [B, K_surv] mean-over-(masked tokens, layers) norms like
    run_leak_capture assembles them."""
    state = BankState()
    state.begin_row_stats(row_mask)
    with torch.no_grad():
        for bank in banks.values():
            bank.eval()
            bank(x, state)
    stats = state.end_row_stats()
    state.clear()
    assert len(stats) == len(banks)
    total = sum(s["sum"].double() for s in stats)
    cnt = stats[0]["cnt"].double()
    return (total / (len(stats) * cnt.clamp_min(1)).unsqueeze(-1)).numpy(), stats


def dense_row_means(banks, x, row_mask):
    """Literal reference: per author, per row, mean over masked tokens of
    ||out_a||, then mean over layers."""
    B = x.shape[0]
    Ks = next(iter(banks.values())).num_authors
    per_layer = []
    with torch.no_grad():
        for l in sorted(banks.keys()):
            bank = banks[l]
            act_fn = torch.nn.functional.relu if bank.gate_act == "relu" \
                else torch.nn.functional.silu
            act = act_fn(x @ bank.W_gate.t() + bank.b_gate) * (x @ bank.W_up.t())
            rows = np.zeros((B, Ks))
            for k in range(Ks):
                sl = slice(k * WIDTH, (k + 1) * WIDTH)
                out_k = act[..., sl] @ bank.W_down[:, sl].t()   # (B, T, H)
                n = out_k.norm(dim=-1)                          # (B, T)
                for b in range(B):
                    m = row_mask[b].bool()
                    rows[b, k] = float(n[b][m].mean()) if m.any() else 0.0
            per_layer.append(rows)
    return np.mean(per_layer, axis=0)


# -- (a) capture == dense ----------------------------------------------------

def test_row_stats_capture_matches_dense():
    banks = leak_banks()
    B, T = 4, 7
    x = make_x(B, T, seed=SEED + 30)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 2:] = True         # ragged answer spans
    mask[1, 4:] = True
    mask[2, :] = True
    mask[3, -1] = True
    got, stats = capture_rows(banks, x, mask)
    want = dense_row_means(banks, x, mask)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-7)
    # per-layer bookkeeping: every layer reports the same token counts
    for s in stats:
        assert torch.equal(s["cnt"], mask.sum(dim=1).to(s["cnt"].dtype))
        assert s["sum"].shape == (B, K)


# -- (b) NPZ contract --------------------------------------------------------

def test_assemble_leak_arrays_contract():
    rng = np.random.RandomState(SEED)
    resident = [3, 11, 42]                        # 7 and 19 "dropped"
    n_q = 6
    m = rng.rand(n_q, len(resident)) * 1e-3
    groups = ["forget_orig", "forget_para", "retain", "ood_world_facts",
              "ood_real_authors", "ood_alpaca"]
    #          authors: 7 dropped, 7 dropped, 11 resident, then OOD
    authors = [7, 7, 11, -1, -1, -1]
    arrays = assemble_leak_arrays(m, resident, authors, groups,
                                  droplist_tag="forget10", total_authors=K)

    assert set(arrays) == NPZ_KEYS
    assert arrays["max_surv_norm"].dtype == np.float32
    assert arrays["max_foreign_norm"].dtype == np.float32
    assert arrays["own_norm"].dtype == np.float32
    assert arrays["top_surv_author"].dtype == np.int32
    assert arrays["author_of_q"].dtype == np.int32
    assert arrays["group"].dtype.kind == "U"
    for key in ("max_surv_norm", "max_foreign_norm", "top_surv_author",
                "own_norm", "group", "author_of_q"):
        assert arrays[key].shape == (n_q,), key
    assert arrays["n_surviving"] == len(resident)
    assert arrays["droplist_tag"] == "forget10"
    assert arrays["K"] == K
    assert set(arrays["group"]) <= set(LEAK_GROUPS)

    np.testing.assert_allclose(arrays["max_surv_norm"], m.max(axis=1),
                               rtol=1e-6)
    # rows without a surviving own branch (orphans + OOD): own NaN and
    # foreign == surviving max
    for i in (0, 1, 3, 4, 5):
        assert np.isnan(arrays["own_norm"][i])
        assert arrays["max_foreign_norm"][i] == arrays["max_surv_norm"][i]
    # the retain row: own is its author's column, foreign excludes it
    j = resident.index(11)
    assert arrays["own_norm"][2] == np.float32(m[2, j])
    others = np.delete(m[2], j)
    assert arrays["max_foreign_norm"][2] == pytest.approx(others.max(),
                                                          rel=1e-6)
    # top_surv_author is a GLOBAL author id, never a slot index
    assert set(arrays["top_surv_author"]) <= set(resident)
    assert arrays["top_surv_author"][0] == resident[int(np.argmax(m[0]))]

    # npz round-trip preserves the whole contract
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    z = np.load(buf)
    assert set(z.files) == NPZ_KEYS
    assert str(z["droplist_tag"]) == "forget10"
    assert int(z["n_surviving"]) == len(resident)
    np.testing.assert_array_equal(z["group"], arrays["group"])


def test_assemble_leak_arrays_edges():
    # no surviving branches: silence (0.0), top -1, own NaN
    arrays = assemble_leak_arrays(np.zeros((2, 0)), [], [3, -1],
                                  ["forget_orig", "ood_alpaca"],
                                  droplist_tag="all", total_authors=K)
    assert (arrays["max_surv_norm"] == 0).all()
    assert (arrays["top_surv_author"] == -1).all()
    assert np.isnan(arrays["own_norm"]).all()
    # single own-only survivor: foreign max over the empty set = 0.0
    arrays = assemble_leak_arrays(np.array([[0.5]]), [3], [3], ["retain"],
                                  droplist_tag="none", total_authors=K)
    assert arrays["own_norm"][0] == np.float32(0.5)
    assert arrays["max_foreign_norm"][0] == 0.0
    # unknown group name must refuse
    with pytest.raises(AssertionError):
        assemble_leak_arrays(np.zeros((1, 1)), [3], [3], ["bogus_group"],
                             droplist_tag="none", total_authors=K)


# -- (c) droplist-then-probe ordering ---------------------------------------

def test_droplist_then_probe_sees_post_drop_banks(tmp_path):
    banks = leak_banks()
    adapter_cfg = {"hidden": HIDDEN, "width": WIDTH, "num_authors": K,
                   "layers": BANK_LAYERS, "init_seed": SEED,
                   "gate_act": "relu"}
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(banks, adapter_cfg, run_dir)

    B, T = 3, 5
    x = make_x(B, T, seed=SEED + 31)
    mask = torch.ones(B, T, dtype=torch.bool)
    groups = ["forget_orig", "retain", "ood_alpaca"]
    authors = [7, 11, -1]      # author 7 will be dropped

    # reference run: no droplist — own branch of author 7 present
    ref_banks, _, _ = load_banks_from_checkpoint(run_dir)
    m_ref, _ = capture_rows(ref_banks, x, mask)
    ref = assemble_leak_arrays(
        m_ref, ref_banks[0].author_ids.tolist(), authors, groups,
        droplist_tag="none", total_authors=K)
    assert ref["n_surviving"] == K
    assert not np.isnan(ref["own_norm"][0])     # author 7 resident pre-drop

    # drop run: apply the sha-pinned droplist FIRST, then probe
    drop_banks, _, _ = load_banks_from_checkpoint(run_dir)
    droplist = tmp_path / "drop.json"
    droplist.write_text(json.dumps({
        "tag": "forget7", "authors": [7],
        "bank_sha": compute_bank_sha(drop_banks),
    }))
    apply_droplist_file(drop_banks, str(droplist))
    resident = drop_banks[0].author_ids.tolist()
    assert 7 not in resident and len(resident) == K - 1

    m_drop, stats = capture_rows(drop_banks, x, mask)
    assert m_drop.shape == (B, K - 1)           # the probe sees K-1 columns
    for s in stats:
        assert s["sum"].shape == (B, K - 1)
    dropped = assemble_leak_arrays(m_drop, resident, authors, groups,
                                   droplist_tag="forget7", total_authors=K)
    assert dropped["n_surviving"] == K - 1 and dropped["K"] == K
    assert np.isnan(dropped["own_norm"][0])     # orphan: own branch gone
    assert not np.isnan(dropped["own_norm"][1])  # retain: own survives
    assert 7 not in set(dropped["top_surv_author"])
    # orphan semantics: foreign == surviving max once the own branch is gone
    assert dropped["max_foreign_norm"][0] == dropped["max_surv_norm"][0]
    # surviving columns are the SAME functions before/after the drop
    keep = [j for j, a in enumerate(ref_banks[0].author_ids.tolist())
            if a != 7]
    np.testing.assert_allclose(m_drop, m_ref[:, keep], rtol=1e-6, atol=1e-9)


# -- (d) leak probe-set construction (real tokenizer + cached TOFU) ----------

def test_build_leak_probe_items_groups_and_mapping(llama_tokenizer):
    """The probe-set builder on the REAL cached data: group counts, text-join
    author mapping (forget authors = 180-199 on the cached snapshot),
    paraphrase-to-author join, retain sample excludes forget authors, and
    retain-draw determinism across calls."""
    import measure_selectivity as ms
    from sepmlp_common import import_memadapt_data

    data_tofu = import_memadapt_data()
    try:
        dataset = data_tofu.TofuQADataset(llama_tokenizer, split="full",
                                          max_length=512)
    except Exception as e:
        pytest.skip(f"TOFU full split unavailable offline: {e}")

    class A:
        seed = 42
        ood_n = 5
        leak_retain_n = 40

    try:
        triples = ms.build_leak_probe_items(A, llama_tokenizer, dataset,
                                            data_tofu, 512)
    except Exception as e:
        pytest.skip(f"leak probe sources unavailable offline: {e}")

    groups = [g for g, _, _ in triples]
    counts = {g: groups.count(g) for g in set(groups)}
    assert counts["forget_orig"] == 400
    assert 390 <= counts.get("forget_para", 0) <= 400  # length filter only
    assert counts["retain"] == 40
    for name in ("ood_world_facts", "ood_real_authors", "ood_alpaca"):
        assert 0 < counts[name] <= 5

    forget_ids = {a for g, a, _ in triples if g == "forget_orig"}
    assert forget_ids == set(range(180, 200))       # text-join, verified
    para_ids = {a for g, a, _ in triples if g == "forget_para"}
    assert para_ids <= forget_ids
    retain_ids = {a for g, a, _ in triples if g == "retain"}
    assert retain_ids.isdisjoint(forget_ids) and retain_ids
    assert all(a == -1 for g, a, _ in triples if g.startswith("ood_"))
    for g, a, item in triples[:5]:
        assert {"input_ids", "labels", "attention_mask",
                "index", "source_ids"} <= set(item)

    # retain draw is deterministic: same seed => same rows
    triples2 = ms.build_leak_probe_items(A, llama_tokenizer, dataset,
                                         data_tofu, 512)
    r1 = [it["index"] for g, _, it in triples if g == "retain"]
    r2 = [it["index"] for g, _, it in triples2 if g == "retain"]
    assert r1 == r2
