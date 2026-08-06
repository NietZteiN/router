"""Core AuthorBank CPU gates (plan gates 1-3, 7, 9, 10 + the OOD-negatives
dataset unit).

All CPU/fp32, seed 42. Bitwise claims (torch.equal) are made only where the
construction guarantees them (zero-init no-op, disconnection, detach-trick
value identity, save->reload); numerical recomputations use allclose/rel-err.
The fixture banks use NON-contiguous global author ids (slot != id) so any
id/slot confusion fails loudly here.
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from conftest import (
    ADAPTER_CFG,
    AUTHOR_IDS,
    BANK_LAYERS,
    BANK_TENSORS,
    HIDDEN,
    K,
    SEED,
    VOCAB,
    WIDTH,
    build_banks,
    ce_sum,
    make_batch,
    make_x,
    randomize_bgate,
    randomize_down,
    slice_rows,
    slot_of,
    wrap_tiny,
)

from bank_layer import AuthorBank, BankState  # noqa: E402
from sepmlp_common import HF_HOME, NO_AUTHOR, import_memadapt_data  # noqa: E402
from sepmlp_model import (  # noqa: E402
    compute_bank_sha,
    freeze_base,
    load_banks_from_checkpoint,
    save_checkpoint,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -- 1. zero-init no-op ------------------------------------------------------

def test_zero_init_is_exact_noop(tiny_model):
    bare = copy.deepcopy(tiny_model)  # twin copied BEFORE install_banks
    banks = build_banks()
    wrap_tiny(tiny_model, banks)
    tiny_model.eval()  # fresh bank modules default to train mode

    ids = torch.randint(0, VOCAB, (2, 12),
                        generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        wrapped = tiny_model(input_ids=ids).logits
        plain = bare(input_ids=ids).logits
    assert torch.equal(wrapped, plain)

    # the bank itself is EXACT zeros at init (W_down = 0)
    x = make_x(2, 6, seed=SEED + 1)
    with torch.no_grad():
        out = banks[0](x, None)
    assert out.shape == x.shape
    assert torch.equal(out, torch.zeros_like(out))


# -- 2. disconnection --------------------------------------------------------

def test_perturbing_author_b_leaves_author_a_bitwise():
    bank = randomize_down(build_banks())[0].eval()
    a, b = 3, 42
    x = make_x(3, 7, seed=SEED)

    with torch.no_grad():
        bank.active[:] = False
        bank.active[slot_of(a)] = True
        y_a_before = bank(x, None)
        # non-vacuity: author a actually contributes
        assert not torch.equal(y_a_before, torch.zeros_like(y_a_before))

        # randomize ALL THREE of author b's slices
        rows = slice_rows(slot_of(b))
        g = torch.Generator().manual_seed(SEED + 100)
        bank.W_gate[rows] = torch.randn(len(rows), HIDDEN, generator=g)
        bank.W_up[rows] = torch.randn(len(rows), HIDDEN, generator=g)
        bank.W_down[:, rows] = torch.randn(HIDDEN, len(rows), generator=g)

        y_a_after = bank(x, None)
    assert torch.equal(y_a_before, y_a_after)

    # non-vacuity: with b active again the randomization IS visible
    with torch.no_grad():
        bank.active[:] = True
        y_full = bank(x, None)
        pristine = randomize_down(build_banks())[0].eval()
        y_pristine = pristine(x, None)
    assert not torch.equal(y_full, y_pristine)


# -- 3. grouped == per-author loop ------------------------------------------

def build_banks_act(gate_act: str, penalty_form: str = "output_gram"):
    """Banks with an explicit gate activation (relu = spec default, silu =
    the retained SwiGLU variant arm), randomized W_down AND b_gate so both
    the down projection and the detector bias are load-bearing."""
    banks = {
        l: AuthorBank(hidden=HIDDEN, width=WIDTH, author_ids=AUTHOR_IDS,
                      layer_idx=l, init_seed=SEED, penalty_form=penalty_form,
                      gate_act=gate_act)
        for l in BANK_LAYERS
    }
    return randomize_bgate(randomize_down(banks))


def ref_author_act(bank, x, k):
    """Independent per-author activation: explicit slicing, explicit bias,
    explicit activation choice."""
    rows = slice_rows(k)
    act_fn = F.relu if bank.gate_act == "relu" else F.silu
    g = x @ bank.W_gate[rows].T + bank.b_gate[rows]
    return act_fn(g) * (x @ bank.W_up[rows].T)


@pytest.mark.parametrize("gate_act", ["relu", "silu"])
def test_grouped_forward_equals_per_author_loop(gate_act):
    bank = build_banks_act(gate_act)[0].eval()
    x = make_x(2, 5, seed=SEED + 2)
    with torch.no_grad():
        grouped = bank(x, None)
        ref = torch.zeros_like(grouped)
        for k in range(K):
            ref = ref + ref_author_act(bank, x, k) @ bank.W_down[:, slice_rows(k)].T
    assert not torch.equal(ref, torch.zeros_like(ref))
    assert torch.allclose(grouped, ref, rtol=1e-6, atol=1e-6)


# -- 4. detach-trick value identity -----------------------------------------

def test_detach_trick_forward_value_is_bitwise_inference_forward():
    bank = randomize_bgate(randomize_down(build_banks()))[0].eval()
    x = make_x(4, 6, seed=SEED + 3)
    state = BankState()
    src = torch.tensor([3, 7, NO_AUTHOR, 42])
    state.set_batch(src, torch.ones(4, 6, dtype=torch.bool))

    assert torch.is_grad_enabled()
    y_train = bank(x, state)  # grad path: detach construction
    assert y_train.requires_grad
    with torch.no_grad():
        y_infer = bank(x, state)  # inference path: plain grouped matmul
    # BITWISE: out_real.detach() + (out_grad - out_grad.detach()) == out_real
    # exactly, because a - a == 0 for finite floats.
    assert torch.equal(y_train.detach(), y_infer)


# -- 5. Gram penalty == dense closed form -----------------------------------

def _dense_penalty(bank, x, source_ids, tok_mask):
    """Independent dense recomputation of the suppression penalty: per-author
    activations by explicit slicing (bias + configured activation), per-author
    output ||.||^2 via the materialized (B,T,H) outputs (no Gram trick), mean
    over off-author entries x live tokens."""
    B, T = x.shape[:2]
    act = torch.stack(
        [ref_author_act(bank, x, k) for k in range(K)], dim=2
    ).float()  # (B, T, K, D)
    if bank.penalty_form == "act_norm":
        q = (act * act).sum(-1)
    else:
        cols = []
        for k in range(K):
            out_k = act[:, :, k, :] @ bank.W_down[:, slice_rows(k)].float().T
            cols.append((out_k * out_k).sum(-1))
        q = torch.stack(cols, dim=2)  # (B, T, K)
    own = torch.tensor(
        [[sid == a for a in AUTHOR_IDS] for sid in source_ids.tolist()]
    )
    off = (~own).view(B, 1, K) & tok_mask.view(B, T, 1)
    pen = (q * off).sum() / off.sum().clamp_min(1).float()
    pen_ignoring_pads = (q * ((~own).view(B, 1, K))).sum() / (B * T * K - own.sum() * T)
    return pen, off, pen_ignoring_pads


@pytest.mark.parametrize("gate_act", ["relu", "silu"])
@pytest.mark.parametrize("form", ["output_gram", "act_norm"])
def test_penalty_matches_dense_closed_form(form, gate_act):
    bank = build_banks_act(gate_act, penalty_form=form)[0].eval()
    B, T = 4, 8
    x = make_x(B, T, seed=SEED + 4)
    src = torch.tensor([3, 7, NO_AUTHOR, 11])  # incl. a NO_AUTHOR row
    tok = torch.ones(B, T, dtype=torch.bool)
    tok[1, -3:] = False  # padded tokens on row 1

    state = BankState()
    state.set_batch(src, tok)
    state.begin_penalty()
    with torch.no_grad():
        bank(x, state)
    assert len(state.penalty_terms) == 1
    pen = float(state.penalty_terms[0])

    ref, off, ref_ignoring_pads = _dense_penalty(bank, x, src, tok)
    assert pen > 0
    assert abs(pen - float(ref)) / float(ref) < 1e-5

    # mask structure the reference relied on:
    assert off[2].all()                       # NO_AUTHOR row: off for EVERY author
    assert not off[1, -3:, :].any()           # padded tokens excluded everywhere
    assert not off[0, :, slot_of(3)].any()    # own author excluded on its row
    # and the bank really excludes pads (a pad-blind normalizer disagrees)
    assert abs(pen - float(ref_ignoring_pads)) / float(ref_ignoring_pads) > 1e-5


# -- 6. save -> reload bitwise + tamper reject ------------------------------

def test_save_reload_bitwise_and_author_ids_tamper_rejected(tmp_path):
    banks = randomize_bgate(randomize_down(build_banks()))
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(banks, dict(ADAPTER_CFG), run_dir)

    loaded, cfg, _ = load_banks_from_checkpoint(run_dir)
    assert sorted(loaded.keys()) == BANK_LAYERS
    assert cfg["width"] == ADAPTER_CFG["width"]
    for l in BANK_LAYERS:
        for name in BANK_TENSORS:
            assert torch.equal(getattr(loaded[l], name), getattr(banks[l], name))
        assert torch.equal(loaded[l].author_ids, banks[l].author_ids)

    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    assert meta["bank_sha"] == compute_bank_sha(banks)
    assert len(meta["checkpoint_sha256"]) == 64

    # tamper the stored author->slot map -> load must refuse (bank_sha)
    path = os.path.join(run_dir, "sepmlp.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["author_ids"][0] = 999
    torch.save(payload, path)
    with pytest.raises(AssertionError, match="bank_sha"):
        load_banks_from_checkpoint(run_dir)


# -- 7. mixed batch == sequential -------------------------------------------

def test_mixed_batch_grads_equal_sum_of_per_sequence_backwards(tiny_model):
    banks = randomize_bgate(randomize_down(build_banks()))
    state = wrap_tiny(tiny_model, banks)
    freeze_base(tiny_model, banks)
    tiny_model.eval()  # routing keys on grad-enabled, not module mode

    batch = make_batch(B=3, T=10, source_ids=[3, 7, 11], n_pad=2)

    def bank_grads():
        out = {}
        for l in BANK_LAYERS:
            for name in BANK_TENSORS:
                p = getattr(banks[l], name)
                out[(l, name)] = (
                    p.grad.detach().clone() if p.grad is not None
                    else torch.zeros_like(p)
                )
        return out

    # one mixed 3-author batch, CE reduction='sum' (additive across sequences)
    tiny_model.zero_grad(set_to_none=True)
    state.set_batch(batch["source_ids"], batch["attention_mask"])
    logits = tiny_model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]).logits
    ce_sum(logits, batch["labels"]).backward()
    state.clear()
    mixed = bank_grads()

    # sum of three per-sequence backwards (grads accumulate)
    tiny_model.zero_grad(set_to_none=True)
    for i in range(3):
        state.set_batch(batch["source_ids"][i:i + 1],
                        batch["attention_mask"][i:i + 1])
        logits = tiny_model(input_ids=batch["input_ids"][i:i + 1],
                            attention_mask=batch["attention_mask"][i:i + 1]).logits
        ce_sum(logits, batch["labels"][i:i + 1]).backward()
        state.clear()
    sequential = bank_grads()

    some_nonzero = False
    for key, gm in mixed.items():
        assert torch.allclose(gm, sequential[key], rtol=1e-5, atol=1e-6), key
        some_nonzero = some_nonzero or bool(gm.abs().sum() > 0)
    assert some_nonzero


# -- 8. training-mode guard --------------------------------------------------

def test_training_mode_guard_raises_without_set_batch():
    bank = build_banks()[0]  # fresh modules default to train mode
    assert bank.training
    x = make_x(1, 4, seed=SEED)
    with pytest.raises(RuntimeError, match="set_batch"):
        bank(x, None)
    state = BankState()
    with pytest.raises(RuntimeError, match="set_batch"):
        bank(x, state)  # state present but source_ids never set

    # with set_batch the training forward runs; in eval, stateless runs too
    state.set_batch(torch.tensor([3]), torch.ones(1, 4, dtype=torch.bool))
    bank(x, state)
    bank.eval()
    with torch.no_grad():
        bank(x, None)


# -- 9. OODNegativesDataset unit --------------------------------------------

def test_ood_negatives_dataset(llama_tokenizer):
    if os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")) not in sys.path:
        sys.path.insert(0, os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")))
    try:
        from skill_data import load_alpaca

        load_alpaca(1, HF_HOME, seed=0)
        from sepmlp_common import never_train_questions

        guarded = never_train_questions()
    except AssertionError:
        raise
    except Exception as e:
        pytest.skip(f"Alpaca / TOFU splits not cached offline: {e}")

    from train_sepmlp import OODNegativesDataset

    data_tofu = import_memadapt_data()
    n = 4
    ds = OODNegativesDataset(llama_tokenizer, n=n, seed=SEED,
                             hf_home=HF_HOME, max_length=512,
                             guarded_questions=guarded)
    # pure-negative pool = n Alpaca rows + the real_authors rows that fit
    assert ds.n_alpaca == n
    assert ds.n_real_authors > 0
    assert len(ds) == ds.n_alpaca + ds.n_real_authors
    for i in range(len(ds)):
        item = ds[i]
        labels, ids = item["labels"], item["input_ids"]
        assert len(labels) == len(ids)
        # every label is IGNORE except the final one (the real last token:
        # keeps num_items_in_batch nonzero, gives the banks zero LM gradient)
        assert (labels[:-1] == data_tofu.IGNORE_INDEX).all()
        assert labels[-1] == ids[-1]
        assert item["source_ids"] == NO_AUTHOR
        assert item["index"] <= -1000
