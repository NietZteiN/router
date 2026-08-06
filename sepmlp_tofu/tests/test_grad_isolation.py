"""Gradient-isolation CPU gates (plan gates 4, 5 and the loss-weight
ga-invariance half of gate 7), against the spec objective
total = L1 + w2*L2(hinge) + w3*L3(gram) + w4*L4(promotion).

Independent reimplementation of the slice math (explicit per-slot row/col/
entry blocks over all FOUR bank tensors) — deliberately NOT calling
train_sepmlp.debug_grad_check, so the two implementations cross-check.

Exactness notes:
  - L1 grads outside the batch authors' slices are EXACTLY zero: the detach
    construction multiplies non-own activations by a 0/1 mask before the
    grad-path matmul, so their contributions are exact zeros, not small.
  - The off-only invariants (L3 penalty, L2+L3 losses) are stated on a
    SINGLE-author batch: with in-batch negatives, a mixed batch legitimately
    sends suppression gradient into the OTHER batch authors' slices, so only
    the own author's slices are guaranteed exactly zero. Within-layer
    exactness relies on the banks computing loss terms from the DETACHED
    layer input (bank_layer forward) — without it, layer l's terms would
    leak through the residual stream into LOWER layers' own slices.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from conftest import (
    BANK_LAYERS,
    BANK_TENSORS,
    K,
    WIDTH,
    build_banks,
    ce_sum,
    make_batch,
    randomize_bgate,
    randomize_down,
    slot_of,
    wrap_tiny,
)

from sepmlp_common import NO_AUTHOR  # noqa: E402
from sepmlp_model import freeze_base  # noqa: E402


def slot_grad_mass(bank, name):
    """Per-slot L1 mass of a grouped tensor's grad (independent slice math:
    rows [k*D:(k+1)*D] for W_gate/W_up, the same entries for the 1-D b_gate,
    the same cols for W_down)."""
    p = getattr(bank, name)
    g = p.grad if p.grad is not None else torch.zeros_like(p)
    mass = []
    for k in range(bank.num_authors):
        sl = slice(k * WIDTH, (k + 1) * WIDTH)
        blk = g[:, sl] if name == "W_down" else g[sl]
        mass.append(float(blk.abs().sum()))
    return mass


def make_trained_banks(penalty_form: str = "output_gram"):
    return randomize_bgate(randomize_down(build_banks(penalty_form)))


def grads_of(banks):
    out = {}
    for l in BANK_LAYERS:
        for name in BANK_TENSORS:
            p = getattr(banks[l], name)
            out[(l, name)] = (p.grad.detach().clone() if p.grad is not None
                              else torch.zeros_like(p))
    return out


def prompt_batch(B, T, source_ids, n_prompt=4, n_pad=0):
    """Collated-style batch whose first n_prompt live tokens are 'question'
    tokens (labels IGNORE) — the trainer's qmask definition."""
    batch = make_batch(B=B, T=T, source_ids=source_ids, n_pad=n_pad)
    batch["labels"][:, :n_prompt] = -100
    return batch


def own_token_mask(batch):
    # same definition SepMlpTrainer.compute_loss uses: promotion fires on all
    # own tokens (the full sequence), paper §3.2 "source k's own tokens".
    return batch["attention_mask"].bool()


# -- 10. L1-only grads live only in the batch authors' slices ---------------

def test_lm_grads_confined_to_batch_author_slices(tiny_model):
    banks = make_trained_banks()
    state = wrap_tiny(tiny_model, banks)
    freeze_base(tiny_model, banks)
    tiny_model.eval()

    present = [3, 11]  # authors 7, 19, 42 absent; plus one NO_AUTHOR row
    batch = make_batch(B=4, T=10, source_ids=[3, 11, 3, NO_AUTHOR], n_pad=2)
    state.set_batch(batch["source_ids"], batch["attention_mask"])
    out = tiny_model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     labels=batch["labels"])
    assert torch.isfinite(out.loss)
    out.loss.backward()
    state.clear()

    present_slots = {slot_of(a) for a in present}
    for l in BANK_LAYERS:
        for name in BANK_TENSORS:
            mass = slot_grad_mass(banks[l], name)
            for k in range(K):
                if k in present_slots:
                    assert mass[k] > 0, (l, name, k)      # non-vacuity
                else:
                    assert mass[k] == 0.0, (l, name, k)   # EXACTLY zero


# -- 11a. penalty-only (L3 alone via begin_penalty) -------------------------

@pytest.mark.parametrize("form", ["output_gram", "act_norm"])
def test_penalty_grads_confined_to_off_author_slices(tiny_model, form):
    banks = make_trained_banks(form)
    state = wrap_tiny(tiny_model, banks)
    freeze_base(tiny_model, banks)
    tiny_model.eval()

    own = 7  # SINGLE-author batch: the only setting with an exact invariant
    batch = make_batch(B=2, T=8, source_ids=[own, own], n_pad=0)
    state.set_batch(batch["source_ids"], batch["attention_mask"])
    state.begin_penalty()
    tiny_model(input_ids=batch["input_ids"],
               attention_mask=batch["attention_mask"])
    assert len(state.penalty_terms) == len(BANK_LAYERS)  # one per bank layer
    penalty = torch.stack(state.penalty_terms).mean()
    assert float(penalty) > 0
    penalty.backward()
    state.clear()

    own_slot = slot_of(own)
    for l in BANK_LAYERS:
        for name in ("W_gate", "W_up", "b_gate"):
            mass = slot_grad_mass(banks[l], name)
            assert mass[own_slot] == 0.0, (l, name)   # own EXACTLY zero
            assert sum(mass) > 0, (l, name)           # off authors are hit
        mass_down = slot_grad_mass(banks[l], "W_down")
        assert mass_down[own_slot] == 0.0, l
        if form == "act_norm":
            # act_norm never touches the down projection: EXACTLY zero grad
            assert sum(mass_down) == 0.0, l
        else:
            assert sum(mass_down) > 0, l


# -- 11b. spec loss terms: L2+L3 off-only, L4 own gate/bias only ------------

def test_loss_terms_grad_structure(tiny_model):
    banks = make_trained_banks()
    state = wrap_tiny(tiny_model, banks)
    freeze_base(tiny_model, banks)
    tiny_model.eval()

    own = 7  # single-author batch (see module docstring)
    batch = prompt_batch(B=2, T=8, source_ids=[own, own])
    # promo_delta far above any random pre-activation peak so L4 is ACTIVE
    state.set_batch(batch["source_ids"], batch["attention_mask"],
                    own_token_mask=own_token_mask(batch))
    state.begin_losses(hinge_margin=2.0, promo_delta=25.0)
    tiny_model(input_ids=batch["input_ids"],
               attention_mask=batch["attention_mask"])
    terms = state.loss_terms
    assert len(terms) == len(BANK_LAYERS)

    own_slot = slot_of(own)

    # L2+L3: off-author slices only, own exactly zero, all four tensors
    tiny_model.zero_grad(set_to_none=True)
    l2l3 = (torch.stack([t["hinge"] for t in terms]).mean()
            + torch.stack([t["gram"] for t in terms]).mean())
    assert float(l2l3) > 0
    l2l3.backward(retain_graph=True)
    for l in BANK_LAYERS:
        for name in BANK_TENSORS:
            mass = slot_grad_mass(banks[l], name)
            assert mass[own_slot] == 0.0, (l, name)
            assert sum(mass) > 0, (l, name)

    # L4: own W_gate/b_gate rows only; W_up/W_down untouched entirely
    promos = [t["promo"] for t in terms if t["promo"] is not None]
    assert len(promos) == len(BANK_LAYERS)  # own rows + question tokens exist
    tiny_model.zero_grad(set_to_none=True)
    promo = torch.stack(promos).mean()
    assert float(promo) > 0  # delta 25 >> random peaks: the hinge is live
    promo.backward()
    state.clear()
    for l in BANK_LAYERS:
        for name in ("W_gate", "b_gate"):
            mass = slot_grad_mass(banks[l], name)
            assert sum(mass) == mass[own_slot], (l, name)  # own rows only
            assert mass[own_slot] > 0, (l, name)
        for name in ("W_up", "W_down"):
            assert sum(slot_grad_mass(banks[l], name)) == 0.0, (l, name)


# -- 12. ga-invariance of the trainer's documented scale rule ---------------

def test_loss_weight_scale_rule_is_ga_invariant(tiny_model):
    """Simulates SepMlpTrainer.compute_loss's loss assembly (read from the
    source): under the transformers-4.48 num_items_in_batch path, micro
    losses CE_sum/num_items are SUMMED across accumulation steps with no /ga
    afterwards, and the trainer scales the per-micro extra
    (w2*L2 + w3*L3 + w4*L4) by 1/gradient_accumulation_steps. Total
    accumulated gradient at ga=2 must equal ga=1. Tests the arithmetic only —
    no HF Trainer."""
    w2, w3, w4 = 10.0, 50.0, 1.0
    margin, delta = 2.0, 25.0
    banks = make_trained_banks()
    state = wrap_tiny(tiny_model, banks)
    freeze_base(tiny_model, banks)
    tiny_model.eval()

    # symmetric batch: equal-length rows, authors alternating, so each micro
    # slice has identical off-entry / promo-row normalizers and the
    # mean-based terms decompose exactly across micros
    batch = prompt_batch(B=4, T=10, source_ids=[3, 7, 3, 7])
    labels = batch["labels"]
    omask = own_token_mask(batch)
    n_items = int((labels[:, 1:] != -100).sum())  # shared normalizer

    def micro_loss(sl, ga):
        state.set_batch(batch["source_ids"][sl], batch["attention_mask"][sl],
                        own_token_mask=omask[sl])
        state.begin_losses(hinge_margin=margin, promo_delta=delta)
        logits = tiny_model(input_ids=batch["input_ids"][sl],
                            attention_mask=batch["attention_mask"][sl]).logits
        lm = ce_sum(logits, labels[sl]) / n_items
        terms = state.loss_terms
        l2 = torch.stack([t["hinge"] for t in terms]).mean()
        l3 = torch.stack([t["gram"] for t in terms]).mean()
        promos = [t["promo"] for t in terms if t["promo"] is not None]
        extra = w2 * l2 + w3 * l3
        if promos:
            extra = extra + w4 * torch.stack(promos).mean()
        state.clear()
        return lm + extra / ga  # the trainer's documented ga-scale rule

    # ga=1: one backward over the full batch
    tiny_model.zero_grad(set_to_none=True)
    micro_loss(slice(0, 4), ga=1).backward()
    g1 = grads_of(banks)

    # ga=2: two micro backwards, grads accumulate
    tiny_model.zero_grad(set_to_none=True)
    for sl in (slice(0, 2), slice(2, 4)):
        micro_loss(sl, ga=2).backward()
    g2 = grads_of(banks)

    some_nonzero = False
    for key in g1:
        assert torch.allclose(g1[key], g2[key], rtol=1e-5, atol=1e-6), key
        some_nonzero = some_nonzero or bool(g1[key].abs().sum() > 0)
    assert some_nonzero
