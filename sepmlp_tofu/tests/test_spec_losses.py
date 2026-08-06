"""Spec-recipe CPU gates: L2 hinge + L4 promotion hand math, per-author clip
isolation, ReLU exact-zero off-state, gate-bias round-trip (save/load/sha/
remove_authors), detector-init determinism + application, the never-train
membership guard, alternating batch sampler, loss-config resolution, and the
in-run debug_grad_check on a tiny wrapped model.

All CPU/fp32, pinned seeds (house convention). The hinge/promotion references
are literal loops over units/tokens — deliberately NOT reusing the bank's
vectorized code, so the two implementations cross-check each other.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import (
    AUTHOR_IDS,
    BANK_LAYERS,
    HIDDEN,
    K,
    SEED,
    WIDTH,
    build_banks,
    make_batch,
    make_x,
    randomize_bias,
    randomize_down,
    slice_rows,
    slot_of,
    wrap_tiny,
)

import sepmlp_common
from bank_layer import AuthorBank, BankState
from sepmlp_common import NO_AUTHOR
from sepmlp_model import (
    apply_droplist_file,
    compute_bank_sha,
    freeze_base,
    load_banks_from_checkpoint,
    save_checkpoint,
)
from train_sepmlp import (
    DEFAULT_LOSS,
    AlternatingBatchSampler,
    apply_detector_init,
    compute_detector_init,
    debug_grad_check,
    per_author_clip_,
    resolve_loss_cfg,
)

MARGIN, DELTA = 2.0, 0.1


def spec_banks():
    return randomize_bias(randomize_down(build_banks()))


def collect_terms(bank, x, source_ids, attn, qmask, margin=MARGIN, delta=DELTA):
    state = BankState()
    state.set_batch(source_ids, attn, own_token_mask=qmask)
    state.begin_losses(margin, delta)
    bank.train()
    bank(x, state)
    assert len(state.loss_terms) == 1
    terms = state.loss_terms[0]
    state.clear()
    return terms


# -- hinge + promotion: hand-computed reference ------------------------------

def test_hinge_and_promo_hand_math():
    bank = spec_banks()[0]
    B, T = 3, 5
    x = make_x(B, T, seed=SEED + 10)
    source_ids = torch.tensor([3, NO_AUTHOR, 42])
    attn = torch.ones(B, T, dtype=torch.bool)
    attn[1, -2:] = False                       # pads on the NO_AUTHOR row
    attn[2, -1:] = False
    qmask = torch.zeros(B, T, dtype=torch.bool)
    qmask[0, :2] = True                        # row 0: question tokens 0,1
    qmask[2, :3] = True                        # row 2: question tokens 0,1,2
    qmask[2, -1] = True                        # padded position: must be
    #                                            ignored (qmask AND attn)

    terms = collect_terms(bank, x, source_ids, attn, qmask)

    # literal per-unit/per-token references
    with torch.no_grad():
        g = x @ bank.W_gate.t() + bank.b_gate  # (B, T, K*D)
    own_slot = {0: slot_of(3), 2: slot_of(42)}  # row -> own slot (row 1 none)

    hinge_vals, promo_by_row = [], {}
    for b in range(B):
        for t in range(T):
            if not attn[b, t]:
                continue
            for k in range(K):
                if own_slot.get(b) == k:
                    continue                    # own units excluded from L2
                for d in range(WIDTH):
                    hinge_vals.append(max(0.0, float(g[b, t, k * WIDTH + d]) + MARGIN))
    exp_hinge = float(np.mean(hinge_vals))
    assert exp_hinge > 0                        # non-vacuous at random init
    assert terms["hinge"].item() == pytest.approx(exp_hinge, rel=1e-5)

    for b, k in own_slot.items():
        peaks = [float(g[b, t, k * WIDTH + d])
                 for t in range(T) if qmask[b, t] and attn[b, t]
                 for d in range(WIDTH)]
        promo_by_row[b] = max(0.0, DELTA - max(peaks))
    exp_promo = float(np.mean(list(promo_by_row.values())))
    assert terms["promo"] is not None
    assert terms["promo"].item() == pytest.approx(exp_promo, rel=1e-5, abs=1e-8)

    # promotion is None when no row has both an own slot and question tokens
    terms_neg = collect_terms(
        bank, x, torch.full((B,), NO_AUTHOR), attn, qmask)
    assert terms_neg["promo"] is None
    # ... and the negative batch's hinge covers ALL branches (own mask empty):
    # strictly more mass than the author batch's off-only hinge on this x.
    assert terms_neg["hinge"].item() > 0

    # gram term equals the standalone penalty (shared machinery, same mask)
    state = BankState()
    state.set_batch(source_ids, attn)
    state.begin_penalty()
    bank(x, state)
    pen = state.penalty_terms[0]
    state.clear()
    assert terms["gram"].item() == pytest.approx(pen.item(), rel=1e-6)


def test_promo_fires_iff_below_delta():
    """A detector already >= delta on a question token => zero promotion for
    that row; all-below => positive."""
    bank = AuthorBank(hidden=4, width=2, author_ids=[0], layer_idx=0,
                      init_seed=SEED)
    with torch.no_grad():
        bank.W_gate.zero_()
        bank.b_gate.copy_(torch.tensor([DELTA + 0.5, -1.0]))  # unit 0 fires
    x = torch.zeros(1, 3, 4)
    attn = torch.ones(1, 3, dtype=torch.bool)
    qmask = torch.ones(1, 3, dtype=torch.bool)
    terms = collect_terms(bank, x, torch.tensor([0]), attn, qmask)
    assert terms["promo"].item() == 0.0

    with torch.no_grad():
        bank.b_gate.copy_(torch.tensor([-1.0, -1.0]))         # dead ReLU
    terms = collect_terms(bank, x, torch.tensor([0]), attn, qmask)
    assert terms["promo"].item() == pytest.approx(DELTA + 1.0)


# -- grad structure: L1 own-only, L2/L3 off-only+reaching, L4 own gate/bias --

def test_debug_grad_check_passes_on_tiny_model(tiny_model):
    banks = spec_banks()
    state = wrap_tiny(tiny_model, banks)
    freeze_base(tiny_model, banks)
    tiny_model.eval()
    batch = make_batch(B=4, T=10, source_ids=[3, 11, 3, NO_AUTHOR], n_pad=2)
    # give the synthetic batch a question span (labels IGNORE on the prompt)
    batch["labels"][:, :4] = -100
    debug_grad_check(tiny_model, banks, state, batch)  # asserts internally


def test_l2l3_reach_other_branches_and_never_own(tiny_model):
    """Independent slice math (not via debug_grad_check): single-author batch,
    L2+L3 backward => own slices EXACTLY zero on all four tensors, off
    W_gate/b_gate strictly nonzero at every layer (hinge), off W_down nonzero
    (gram with randomized down)."""
    banks = spec_banks()
    state = wrap_tiny(tiny_model, banks)
    freeze_base(tiny_model, banks)
    tiny_model.eval()

    own = 7
    batch = make_batch(B=2, T=8, source_ids=[own, own], n_pad=0)
    qmask = torch.zeros(2, 8, dtype=torch.bool)
    qmask[:, :3] = True
    state.set_batch(batch["source_ids"], batch["attention_mask"],
                    own_token_mask=qmask)
    state.begin_losses(MARGIN, DELTA)
    tiny_model(input_ids=batch["input_ids"],
               attention_mask=batch["attention_mask"])
    terms = state.loss_terms
    assert len(terms) == len(BANK_LAYERS)
    (torch.stack([t["hinge"] for t in terms]).mean()
     + torch.stack([t["gram"] for t in terms]).mean()).backward()
    state.clear()

    own_rows = slice_rows(slot_of(own))
    for l, bank in banks.items():
        for name in ("W_gate", "W_up", "b_gate", "W_down"):
            grad = getattr(bank, name).grad
            assert grad is not None, (l, name)
            blk = (grad[:, own_rows] if name == "W_down" else grad[own_rows])
            assert blk.abs().sum() == 0.0, (l, name)     # own EXACTLY zero
        off = [k for k in range(K) if k != slot_of(own)]
        for k in off:
            r = slice_rows(k)
            assert bank.W_gate.grad[r].abs().sum() > 0, (l, k)   # hinge reaches
            assert bank.b_gate.grad[r].abs().sum() > 0, (l, k)
        assert bank.W_down.grad.abs().sum() > 0, l               # gram reaches


# -- per-author clip isolation ----------------------------------------------

def test_per_author_clip_isolation():
    banks = build_banks()
    spike, calm = slot_of(42), slot_of(7)
    for l, bank in banks.items():
        for name in ("W_gate", "W_up", "b_gate", "W_down"):
            p = getattr(bank, name)
            p.grad = torch.zeros_like(p)
        g = torch.Generator().manual_seed(SEED + l)
        rs = slice_rows(spike)
        bank.W_gate.grad[rs] = torch.randn(WIDTH, HIDDEN, generator=g) * 10
        bank.b_gate.grad[rs] = torch.randn(WIDTH, generator=g) * 10
        rc = slice_rows(calm)
        bank.W_up.grad[rc] = torch.randn(WIDTH, HIDDEN, generator=g) * 1e-3
        bank.W_down.grad[:, rc] = torch.randn(HIDDEN, WIDTH, generator=g) * 1e-3

    calm_before = {
        l: (banks[l].W_up.grad[slice_rows(calm)].clone(),
            banks[l].W_down.grad[:, slice_rows(calm)].clone())
        for l in BANK_LAYERS
    }
    norms = per_author_clip_(banks, max_norm=1.0)
    assert norms.shape == (K,)
    assert norms[spike] > 1.0 and norms[calm] < 1.0

    for l in BANK_LAYERS:
        up, down = calm_before[l]
        # calm author's grads bitwise untouched (norm < clip => coef 1)
        assert torch.equal(banks[l].W_up.grad[slice_rows(calm)], up)
        assert torch.equal(banks[l].W_down.grad[:, slice_rows(calm)], down)
    # spiking author rescaled to the clip norm (across layers and tensors)
    post = torch.zeros(())
    for l, bank in banks.items():
        rs = slice_rows(spike)
        post = post + bank.W_gate.grad[rs].pow(2).sum() \
            + bank.b_gate.grad[rs].pow(2).sum()
    assert float(post.sqrt()) == pytest.approx(1.0, rel=1e-4)
    # untouched (all-zero) authors stay exactly zero
    other = [k for k in range(K) if k not in (spike, calm)][0]
    for l, bank in banks.items():
        assert bank.W_gate.grad[slice_rows(other)].abs().sum() == 0


# -- ReLU off-state: pre_act <= 0 => branch output EXACTLY zero --------------

def test_relu_off_state_exactly_zero():
    bank = randomize_down(build_banks(gate_act="relu"))[0].eval()
    off = slot_of(19)
    with torch.no_grad():
        rows = slice_rows(off)
        bank.W_gate[rows] = 0.0
        bank.b_gate[rows] = -3.0               # pre_act = -3 < 0 everywhere
    x = make_x(2, 6, seed=SEED + 20)
    with torch.no_grad():
        bank.active[:] = False
        bank.active[off] = True
        y = bank(x, None)                      # ONLY the silenced branch
    assert torch.equal(y, torch.zeros_like(y))  # EXACT zero, not small

    # silu control: the same construction is NOT exactly zero (no hard gate)
    bank_s = randomize_down(build_banks(gate_act="silu"))[0].eval()
    with torch.no_grad():
        bank_s.W_gate[rows] = 0.0
        bank_s.b_gate[rows] = -3.0
        bank_s.active[:] = False
        bank_s.active[off] = True
        ys = bank_s(x, None)
    assert not torch.equal(ys, torch.zeros_like(ys))


# -- gate-bias round-trip: save/load, sha, remove_authors --------------------

def test_bias_roundtrip_save_load_sha_remove(tmp_path):
    banks = spec_banks()
    adapter_cfg = {"hidden": HIDDEN, "width": WIDTH, "num_authors": K,
                   "layers": BANK_LAYERS, "init_seed": SEED,
                   "gate_act": "relu"}
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(banks, adapter_cfg, run_dir)
    loaded, cfg, _ = load_banks_from_checkpoint(run_dir)
    assert cfg["gate_act"] == "relu"
    for l in BANK_LAYERS:
        assert loaded[l].gate_act == "relu"
        assert torch.equal(loaded[l].b_gate, banks[l].b_gate)
        assert loaded[l].b_gate.abs().sum() > 0    # non-vacuous (randomized)

    # sha covers the bias tensor: the pre-bias shape list hashes differently
    sha_with = compute_bank_sha(banks)
    assert sha_with == cfg_sha_from_meta(run_dir)
    shapes_no_bias = []
    for l, b in sorted(banks.items()):
        shapes_no_bias += [tuple(b.W_gate.shape), tuple(b.W_up.shape),
                           tuple(b.W_down.shape)]
    sha_without = sepmlp_common.bank_sha(
        banks[0].author_ids, sorted(banks.keys()), shapes_no_bias)
    assert sha_with != sha_without

    # physical removal slices the bias exactly like the matrices
    drop = [7, 42]
    keep_ids = [a for a in AUTHOR_IDS if a not in drop]
    keep_rows = torch.cat([slice_rows(slot_of(a)) for a in keep_ids])
    before = {l: banks[l].b_gate.clone() for l in BANK_LAYERS}
    droplist = tmp_path / "drop.json"
    droplist.write_text(json.dumps({
        "tag": "bias", "authors": drop, "bank_sha": compute_bank_sha(loaded),
    }))
    apply_droplist_file(loaded, str(droplist))
    for l in BANK_LAYERS:
        assert loaded[l].b_gate.shape == (len(keep_ids) * WIDTH,)
        assert torch.equal(loaded[l].b_gate, before[l][keep_rows])
    assert compute_bank_sha(loaded) != sha_with   # sha tracks the new shapes


def cfg_sha_from_meta(run_dir):
    with open(os.path.join(run_dir, "meta.json")) as f:
        return json.load(f)["bank_sha"]


# -- detector init: determinism + application --------------------------------

def _init_batches():
    """Synthetic collated batches with a question span (labels IGNORE on the
    first 4 positions) covering authors 3 and 11; author 7 absent."""
    batches = []
    for seed in (SEED + 1, SEED + 2):
        b = make_batch(B=4, T=10, source_ids=[3, 11, 3, NO_AUTHOR],
                       n_pad=2, seed=seed)
        b["labels"][:, :4] = -100
        batches.append(b)
    return batches


def test_detector_init_deterministic_and_correct(tiny_model):
    layer_idxs = [0, 2]
    ids = AUTHOR_IDS
    m1, c1 = compute_detector_init(tiny_model, layer_idxs, _init_batches(),
                                   ids, "cpu")
    m2, c2 = compute_detector_init(tiny_model, layer_idxs, _init_batches(),
                                   ids, "cpu")
    assert m1.shape == (K, len(layer_idxs), HIDDEN)
    assert np.array_equal(m1, m2) and np.array_equal(c1, c2)  # bit-equal

    # counts: authors 3 (4 rows) and 11 (2 rows) x 4 question tokens each;
    # the NO_AUTHOR row contributes to nobody; absent authors count 0.
    assert c1[slot_of(3)] == 4 * 4 and c1[slot_of(11)] == 2 * 4
    assert c1[slot_of(7)] == 0

    # correctness: independent hook capture for one layer/author
    captured = {}
    h = tiny_model.model.layers[0].mlp.register_forward_pre_hook(
        lambda mod, args: captured.setdefault("x", args[0]))
    sums = torch.zeros(HIDDEN, dtype=torch.float64)
    n = 0.0
    with torch.no_grad():
        for b in _init_batches():
            captured.clear()
            tiny_model(input_ids=b["input_ids"],
                       attention_mask=b["attention_mask"], use_cache=False)
            x = captured["x"].double()
            qm = ((b["labels"] == -100) & b["attention_mask"]).double()
            rows = (b["source_ids"] == 3).nonzero(as_tuple=True)[0]
            for r in rows.tolist():
                sums += (x[r] * qm[r].unsqueeze(-1)).sum(0)
                n += float(qm[r].sum())
    h.remove()
    ref = (sums / n).float().numpy()
    np.testing.assert_allclose(m1[slot_of(3), 0], ref, rtol=1e-5, atol=1e-7)


def test_apply_detector_init_direction_and_scale():
    banks = build_banks()
    before = {l: banks[l].W_gate.clone() for l in BANK_LAYERS}
    g = torch.Generator().manual_seed(SEED)
    mean_hidden = torch.randn(K, len(BANK_LAYERS), HIDDEN,
                              generator=g).numpy().astype(np.float32)
    counts = np.array([4.0, 0.0, 2.0, 8.0, 1.0])   # author 7 (slot 1): none
    apply_detector_init(banks, mean_hidden, counts, init_scale=1.5)
    for i, l in enumerate(sorted(BANK_LAYERS)):
        wg = banks[l].W_gate.view(K, WIDTH, HIDDEN)
        for j in range(K):
            if counts[j] == 0:
                assert torch.equal(wg[j], before[l].view(K, WIDTH, HIDDEN)[j])
                continue
            mh = torch.from_numpy(mean_hidden[j, i])
            want = 1.5 * mh / mh.norm()
            for d in range(WIDTH):               # every unit: same direction
                assert torch.allclose(wg[j, d], want, rtol=1e-6, atol=1e-7)
            assert float(wg[j, 0].norm()) == pytest.approx(1.5, rel=1e-5)
        assert banks[l].b_gate.abs().sum() == 0   # bias stays 0 per spec


# -- never-train membership guard --------------------------------------------

def test_never_train_guard_fires_and_passes():
    guarded = {"Who is the secret author?", "What did X write?"}
    sepmlp_common.assert_never_train_clean(
        ["What is the capital of France?"], guarded, "clean pool")
    with pytest.raises(AssertionError, match="never-train"):
        sepmlp_common.assert_never_train_clean(
            ["What did X write?"], guarded, "dirty pool")


def test_never_train_guard_on_real_cached_split():
    try:
        guarded = sepmlp_common.never_train_questions()
    except Exception as e:
        pytest.skip(f"never-train split unavailable offline: {e}")
    assert len(guarded) == 400
    q = next(iter(guarded))
    with pytest.raises(AssertionError, match="never-train"):
        sepmlp_common.assert_never_train_clean([q], guarded, "poisoned")


# -- alternating batch sampler ----------------------------------------------

def test_alternating_sampler_structure_and_determinism():
    n_author, n_neg, bs = 45, 7, 8
    s = AlternatingBatchSampler(n_author, n_neg, bs, seed=SEED)
    n_author_batches = (n_author + bs - 1) // bs
    assert len(s) == 2 * n_author_batches

    epoch0 = list(iter(s))
    assert len(epoch0) == 2 * n_author_batches
    author_seen = []
    for i, batch in enumerate(epoch0):
        if i % 2 == 0:       # author batch: indices into the author dataset
            assert all(idx < n_author for idx in batch)
            author_seen += batch
        else:                # negative batch: offset ConcatDataset indices
            assert all(n_author <= idx < n_author + n_neg for idx in batch)
            assert len(batch) == bs
    assert sorted(author_seen) == list(range(n_author))  # full pass, no dups

    # determinism: same (seed, epoch counter) => identical schedule ...
    assert list(iter(AlternatingBatchSampler(n_author, n_neg, bs, SEED))) \
        == epoch0
    # ... and the next epoch reshuffles
    assert list(iter(s)) != epoch0


# -- loss-config resolution --------------------------------------------------

def test_resolve_loss_cfg():
    assert resolve_loss_cfg({}) == DEFAULT_LOSS
    got = resolve_loss_cfg({"loss": {"w2": 3.0}})
    assert got["w2"] == 3.0 and got["w3"] == DEFAULT_LOSS["w3"]
    legacy = resolve_loss_cfg({"suppress_lambda": 0.7})
    assert legacy["w2"] == 0.0 and legacy["w4"] == 0.0
    assert legacy["w3"] == pytest.approx(0.7)
    with pytest.raises(AssertionError):
        resolve_loss_cfg({"loss": {"typo_key": 1.0}})
