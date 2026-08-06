"""Gradient-isolation CPU gates (DESIGN §9 gate 3 per (phase x batch-type),
gate 13 ga-invariance of the suppression term, mixed-batch additivity, and a
cross-check run of train_tc.debug_grad_check).

Independent reimplementation of the slice math (explicit per-slot row/entry/
column blocks over all THREE master tensors, decoder columns across ALL span
decoders) — deliberately NOT calling train_tc.debug_grad_check for the
primary asserts, so the two implementations cross-check.

Exactness notes:
  - Grads outside the phase's own slices are EXACTLY zero, not small: the
    detach construction multiplies non-own activations by a 0/1 mask BEFORE
    the grad-path decode matmul, so their contributions are exact zeros —
    including the decoder columns (decoder grad ∝ activation value; masking
    the encoder path alone would train every author's decoder columns on
    every batch; DESIGN §3).
  - Non-vacuity is asserted on the DECODER slice for LM passes: at a phase's
    step 0 the relevant decoder columns are exactly zero, so the encoder
    grad (which flows THROUGH the decoder) is legitimately zero — here the
    fixtures randomize W_dec so both encoder and decoder slices are live.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from conftest import (
    FDIM,
    K,
    S,
    SEED,
    ce_sum,
    feat_rows,
    make_batch,
    make_x,
    module_forward,
    slot_of,
    trained_tc,
    wrap_tiny,
)

from tc_common import NO_AUTHOR  # noqa: E402
from tc_layer import TcState  # noqa: E402
from tc_model import freeze_base  # noqa: E402
from train_tc import debug_grad_check  # noqa: E402


def slot_grad_mass(tc, name):
    """Per-slot L1 mass of a master tensor's grad, slots 0..K-1 author blocks
    + slot K the shared block (independent slice math: rows [k*m:(k+1)*m] for
    W_enc, the same entries for b_enc, the same COLUMNS of every W_dec[j])."""
    p = getattr(tc, name)
    g = p.grad if p.grad is not None else torch.zeros_like(p)
    mass = []
    for k in range(K):
        rows = feat_rows(k)
        blk = g[..., rows] if name == "W_dec" else g[rows]
        mass.append(float(blk.abs().sum()))
    shared = g[..., S:] if name == "W_dec" else g[S:]
    mass.append(float(shared.abs().sum()))
    return mass


def grads_of(tc):
    return {
        name: (getattr(tc, name).grad.detach().clone()
               if getattr(tc, name).grad is not None
               else torch.zeros_like(getattr(tc, name)))
        for name in ("W_enc", "b_enc", "W_dec")
    }


def routed_lm_backward(model, state, tc, batch, phase):
    model.zero_grad(set_to_none=True)
    state.set_phase(phase)
    state.set_batch(batch["source_ids"],
                    question_mask=(batch["labels"] == -100)
                    & batch["attention_mask"].bool(),
                    attention_mask=batch["attention_mask"])
    try:
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        assert torch.isfinite(out.loss)
        out.loss.backward()
    finally:
        state.clear()


# -- gate 3a: phase-1 author batch -> block k only ---------------------------

def test_phase1_author_batch_grads_confined_to_own_block(tiny_model):
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()  # routing keys on grad-enabled, not module mode

    own = 7  # global id; slot 1
    batch = make_batch(B=3, T=10, source_ids=[own, own, own], n_pad=2)
    routed_lm_backward(tiny_model, state, tc, batch, "phase1")

    own_slot = slot_of(own)
    for name in ("W_enc", "b_enc", "W_dec"):
        mass = slot_grad_mass(tc, name)
        for k in range(K + 1):  # K+1 = the shared block, frozen in phase 1
            if k == own_slot:
                assert mass[k] > 0, (name, k)      # non-vacuity
            else:
                assert mass[k] == 0.0, (name, k)   # EXACTLY zero
    # decoder columns checked per write layer too (a single-layer leak could
    # hide inside the all-span sum)
    for j in range(tc.span):
        gj = tc.W_dec.grad[j]
        assert gj[:, feat_rows(own_slot)].abs().sum() > 0, j
        off = torch.ones(FDIM, dtype=torch.bool)
        off[feat_rows(own_slot)] = False
        assert gj[:, off].abs().sum() == 0.0, j


def test_phase1_mixed_batch_grads_confined_to_present_blocks(tiny_model):
    """Per-row own-mask: a mixed batch reaches exactly the union of its
    authors' blocks; absent authors, NO_AUTHOR rows, and shared get exact
    zeros (the sepmlp gate-10 analog)."""
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()

    present = [3, 11]  # authors 7, 19, 42 absent; plus one NO_AUTHOR row
    batch = make_batch(B=4, T=10, source_ids=[3, 11, 3, NO_AUTHOR], n_pad=2)
    routed_lm_backward(tiny_model, state, tc, batch, "phase1")

    present_slots = {slot_of(a) for a in present}
    for name in ("W_enc", "b_enc", "W_dec"):
        mass = slot_grad_mass(tc, name)
        for k in range(K + 1):
            if k in present_slots:
                assert mass[k] > 0, (name, k)
            else:
                assert mass[k] == 0.0, (name, k)


# -- gate 3b: phase-1 generic LM -> exactly zero everywhere ------------------

def test_phase1_generic_lm_grads_exactly_zero_everywhere(tiny_model):
    """The empty own-mask detaches the module from the LM loss entirely —
    this is what lets compute_loss discard the generic 1-token LM term."""
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()

    batch = make_batch(B=4, T=10, source_ids=[NO_AUTHOR] * 4, n_pad=2)
    routed_lm_backward(tiny_model, state, tc, batch, "phase1")
    for name in ("W_enc", "b_enc", "W_dec"):
        grad = getattr(tc, name).grad
        assert grad is None or grad.abs().sum() == 0.0, name


# -- gate 3c: phase-1 suppression -> author W_enc/b_enc rows only ------------

def test_phase1_suppression_grads_confined_to_author_encoder_rows(tiny_model):
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()

    batch = make_batch(B=4, T=10, source_ids=[NO_AUTHOR] * 4, n_pad=2)
    tiny_model.zero_grad(set_to_none=True)
    state.set_phase("phase1")
    state.set_batch(batch["source_ids"],
                    attention_mask=batch["attention_mask"])
    state.begin_suppression()
    try:
        tiny_model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"])
        terms = state.end_suppression()
    finally:
        state.clear()
    assert len(terms) == 1  # one read site -> exactly one term
    terms[0].backward()

    # suppression may touch ALL author encoder rows (generic data belongs to
    # no author — exactness preserved) ...
    enc_mass = slot_grad_mass(tc, "W_enc")
    bias_mass = slot_grad_mass(tc, "b_enc")
    assert sum(enc_mass[:K]) > 0 and sum(bias_mass[:K]) > 0
    # ... but must be provably ZERO on all W_dec and on shared rows
    assert tc.W_dec.grad is None or tc.W_dec.grad.abs().sum() == 0.0
    assert enc_mass[K] == 0.0 and bias_mass[K] == 0.0
    # and the suppression graph is detached from the BASE (built from
    # xn.detach()): no base parameter may carry grad
    for n, p in tiny_model.named_parameters():
        if p.grad is not None and not n.endswith(("W_enc", "b_enc", "W_dec")):
            assert p.grad.abs().sum() == 0.0, n


# -- gate 3d: phase 0 -> shared slices only ----------------------------------

def test_phase0_grads_confined_to_shared_block(tiny_model):
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()

    batch = make_batch(B=4, T=10, source_ids=[NO_AUTHOR] * 4, n_pad=2)
    routed_lm_backward(tiny_model, state, tc, batch, "phase0")
    for name in ("W_enc", "b_enc", "W_dec"):
        mass = slot_grad_mass(tc, name)
        for k in range(K):
            assert mass[k] == 0.0, (name, k)   # every author EXACTLY zero
    # non-vacuity on the shared decoder columns (encoder can be zero at a
    # true phase-0 step 0 — W_dec randomized here so both are live)
    assert slot_grad_mass(tc, "W_dec")[K] > 0
    assert slot_grad_mass(tc, "W_enc")[K] > 0


# -- cross-check: the in-run debug_grad_check agrees -------------------------

def test_debug_grad_check_passes_on_tiny_model(tiny_model):
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()

    author_batch = make_batch(B=3, T=10, source_ids=[7, 7, 7], n_pad=2)
    author_batch["labels"][:, :4] = -100  # give the batch a question span
    generic_batch = make_batch(B=4, T=10, source_ids=[NO_AUTHOR] * 4, n_pad=2)
    generic_batch["labels"][:, :4] = -100
    generic_batch["index"] = torch.tensor([-1000, -1001, -1002, -1003])

    state.set_phase("phase1")
    debug_grad_check(tiny_model, tc, state, "phase1",
                     author_batch=author_batch,
                     generic_batch=generic_batch)  # asserts internally

    state.set_phase("phase0")
    debug_grad_check(tiny_model, tc, state, "phase0",
                     generic_batch=generic_batch)


# -- mixed batch == sum of per-sequence backwards ----------------------------

def test_mixed_batch_grads_equal_sum_of_per_sequence_backwards(tiny_model):
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()

    batch = make_batch(B=3, T=10, source_ids=[3, 7, 11], n_pad=2)
    state.set_phase("phase1")

    def _backward(sl):
        state.set_batch(batch["source_ids"][sl],
                        attention_mask=batch["attention_mask"][sl])
        try:
            logits = tiny_model(
                input_ids=batch["input_ids"][sl],
                attention_mask=batch["attention_mask"][sl]).logits
            ce_sum(logits, batch["labels"][sl]).backward()
        finally:
            state.clear()

    # one mixed 3-author batch, CE reduction='sum' (additive across rows)
    tiny_model.zero_grad(set_to_none=True)
    _backward(slice(0, 3))
    mixed = grads_of(tc)

    # sum of three per-sequence backwards (grads accumulate)
    tiny_model.zero_grad(set_to_none=True)
    for i in range(3):
        _backward(slice(i, i + 1))
    sequential = grads_of(tc)

    some_nonzero = False
    for name, gm in mixed.items():
        assert torch.allclose(gm, sequential[name], rtol=1e-5, atol=1e-6), name
        some_nonzero = some_nonzero or bool(gm.abs().sum() > 0)
    assert some_nonzero


# -- gate 13: ga-invariance of the suppression term --------------------------

def test_suppression_scale_rule_is_ga_invariant(tiny_model):
    """Simulates BlockTcTrainer.compute_loss's generic-batch assembly (read
    from the source): under the transformers-4.48 num_items_in_batch path,
    micro losses are SUMMED across an accumulation window with no /ga
    afterwards, and the trainer scales the per-micro lambda*L_supp by
    1/gradient_accumulation_steps. Total accumulated gradient at ga=2 must
    equal ga=1. Tests the arithmetic only — no HF Trainer. Symmetric batch
    (equal-length rows, no pads) so the mean-based term decomposes exactly
    across micros."""
    lam = 0.37
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()

    batch = make_batch(B=4, T=10, source_ids=[NO_AUTHOR] * 4, n_pad=0)

    def micro_loss(sl, ga):
        state.set_phase("phase1")
        state.set_batch(batch["source_ids"][sl],
                        attention_mask=batch["attention_mask"][sl])
        state.begin_suppression()
        try:
            tiny_model(input_ids=batch["input_ids"][sl],
                       attention_mask=batch["attention_mask"][sl])
            terms = state.end_suppression()
        finally:
            state.clear()
        assert len(terms) == 1
        return lam * terms[0] / ga  # the trainer's documented ga-scale rule

    # ga=1: one backward over the full batch
    tiny_model.zero_grad(set_to_none=True)
    micro_loss(slice(0, 4), ga=1).backward()
    g1 = grads_of(tc)

    # ga=2: two micro backwards, grads accumulate
    tiny_model.zero_grad(set_to_none=True)
    for sl in (slice(0, 2), slice(2, 4)):
        micro_loss(sl, ga=2).backward()
    g2 = grads_of(tc)

    some_nonzero = False
    for name in g1:
        assert torch.allclose(g1[name], g2[name], rtol=1e-5, atol=1e-7), name
        some_nonzero = some_nonzero or bool(g1[name].abs().sum() > 0)
    assert some_nonzero
    # gate 3c restated on the accumulated grads: W_dec and shared untouched
    assert g1["W_dec"].abs().sum() == 0.0
    assert g1["W_enc"][S:].abs().sum() == 0.0


# -- module-level: encoder grad flows THROUGH the masked decode path ---------

def test_masked_path_carries_grad_through_decoders():
    """The one-and-only grad path runs a_own through W_dec[j]: with a
    randomized decoder, an author-batch backward reaches the own block's
    ENCODER rows (via the decoder) — proving the masked path is not a
    dead-end — and the decoder grad of layer j equals upstream^T @ a_own
    (hand recomputation on the module alone)."""
    tc = trained_tc()
    tc.train()
    B, T = 2, 5
    x = make_x(B, T, seed=SEED + 40)
    state = TcState()
    state.set_phase("phase1")
    state.set_batch(torch.tensor([42, 42]))
    outs = module_forward(tc, x, state)
    outs.sum().backward()
    own = feat_rows(slot_of(42))

    with torch.no_grad():
        a = torch.relu(x @ tc.W_enc.t() + tc.b_enc)
    # dL/dW_dec[j][:, own] with upstream = ones: sum_bt a_own — every layer
    for j in range(tc.span):
        want = a[..., own].sum(dim=(0, 1)).expand_as(tc.W_dec.grad[j][:, own])
        assert torch.allclose(tc.W_dec.grad[j][:, own], want,
                              rtol=1e-5, atol=1e-6), j
    assert tc.W_enc.grad[own].abs().sum() > 0  # reached through the decoder
