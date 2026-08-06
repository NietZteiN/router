"""Phase-freezing CPU gates (DESIGN §9 gate 5 + the belt-(b) hook and the
optimizer-integration of both): the shared block must be BITWISE frozen
through phase 1, and phase 0 must leave the author slices bitwise the seeded
init. Belts under test:
  (b) zero_forbidden_grads_ — the optimizer-step pre-hook slice zeroing;
  (d) assert_shared_frozen / assert_authors_pristine — the save-time bitwise
      compares;
plus an end-to-end mini-loop: real AdamW steps through the registered
pre-hook leave the frozen slices bitwise untouched while the trained slices
move (AdamW with an exactly-zero grad and zero moments takes an exactly-zero
step — weight_decay=0 is load-bearing, DESIGN §3).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from conftest import (
    ADAPTER_CFG,
    FDIM,
    S,
    SEED,
    build_tc,
    feat_rows,
    make_batch,
    slot_of,
    trained_tc,
    wrap_tiny,
)

from tc_common import NO_AUTHOR  # noqa: E402
from tc_model import (  # noqa: E402
    assert_shared_frozen,
    freeze_base,
    save_checkpoint,
)
from train_tc import (  # noqa: E402
    assert_authors_pristine,
    per_block_clip_,
    zero_forbidden_grads_,
)


def _shared_snapshot(tc):
    return (tc.W_enc.detach()[S:].clone(), tc.b_enc.detach()[S:].clone(),
            tc.W_dec.detach()[..., S:].clone())


def _author_snapshot(tc):
    return (tc.W_enc.detach()[:S].clone(), tc.b_enc.detach()[:S].clone(),
            tc.W_dec.detach()[..., :S].clone())


def _equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


# -- belt (b): zero_forbidden_grads_ -----------------------------------------

@pytest.mark.parametrize("phase", ["phase0", "phase1"])
def test_zero_forbidden_grads_slices(phase):
    tc = build_tc()
    for name in ("W_enc", "b_enc", "W_dec"):
        p = getattr(tc, name)
        p.grad = torch.ones_like(p)
    zero_forbidden_grads_(tc, phase)
    if phase == "phase0":
        allowed = slice(S, FDIM)      # shared only
        forbidden = slice(0, S)
    else:
        allowed = slice(0, S)         # authors only; shared ALWAYS zeroed
        forbidden = slice(S, FDIM)
    assert tc.W_enc.grad[forbidden].abs().sum() == 0.0
    assert tc.b_enc.grad[forbidden].abs().sum() == 0.0
    assert tc.W_dec.grad[..., forbidden].abs().sum() == 0.0
    # the permitted slice is untouched (the hook is a belt, not a clip)
    assert torch.equal(tc.W_enc.grad[allowed],
                       torch.ones_like(tc.W_enc.grad[allowed]))
    assert torch.equal(tc.W_dec.grad[..., allowed],
                       torch.ones_like(tc.W_dec.grad[..., allowed]))
    # None grads tolerated (set_to_none path)
    tc2 = build_tc()
    zero_forbidden_grads_(tc2, phase)


# -- belt (d): assert_shared_frozen / assert_authors_pristine ----------------

def test_assert_shared_frozen_passes_and_catches_drift(tmp_path):
    tc = trained_tc()
    run_dir = str(tmp_path / "p0")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir, phase="phase0")

    # legal phase-1 training: author slices move, shared untouched
    with torch.no_grad():
        tc.W_enc[:S] += 0.25
        tc.b_enc[:S] -= 0.5
        tc.W_dec[..., :S] += 0.1
    assert_shared_frozen(tc, run_dir)                       # passes
    assert_shared_frozen(tc, os.path.join(run_dir, "blocktc.pt"))  # .pt form

    # a single drifted shared W_enc entry must be caught
    with torch.no_grad():
        tc.W_enc[S, 0] += 1e-7
    with pytest.raises(AssertionError, match="drifted"):
        assert_shared_frozen(tc, run_dir)
    with torch.no_grad():
        tc.W_enc[S, 0] -= 1e-7
    assert_shared_frozen(tc, run_dir)
    # ... and a drifted shared DECODER column too (columns are the easy leak)
    with torch.no_grad():
        tc.W_dec[1, 0, S] += 1e-7
    with pytest.raises(AssertionError, match="drifted"):
        assert_shared_frozen(tc, run_dir)


def test_assert_shared_frozen_rejects_wrong_checkpoint(tmp_path):
    tc = trained_tc()
    # a phase-1 checkpoint is not a legal comparison base
    run_dir = str(tmp_path / "p1")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir, phase="phase1")
    with pytest.raises(AssertionError):
        assert_shared_frozen(tc, run_dir)
    # a topology-mismatched phase-0 checkpoint is refused via tc_sha
    run_dir0 = str(tmp_path / "p0")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir0, phase="phase0")
    tc.remove_authors([7])
    with pytest.raises(AssertionError, match="topology"):
        assert_shared_frozen(tc, run_dir0)


def test_assert_authors_pristine_passes_and_catches_drift():
    tc = build_tc()
    # phase 0 legally moves ONLY shared slices
    with torch.no_grad():
        tc.W_enc[S:] += 0.3
        tc.b_enc[S:] += 0.1
        tc.W_dec[..., S:] += 0.05
    assert_authors_pristine(tc)                              # passes
    with torch.no_grad():
        tc.W_enc[0, 0] += 1e-7                               # author drift
    with pytest.raises(AssertionError, match="drifted"):
        assert_authors_pristine(tc)


# -- end-to-end: AdamW steps through the pre-hook ----------------------------

def _optimizer_with_belt(tc, phase, clip_norm=1.0):
    # weight_decay=0 is exactness-critical (default 1e-2 would move EVERY
    # parameter every step); max-norm clipping is per-block, never global.
    opt = torch.optim.AdamW([tc.W_enc, tc.b_enc, tc.W_dec], lr=1e-2,
                            weight_decay=0.0)

    def _pre_step(optimizer, hook_args, hook_kwargs):
        zero_forbidden_grads_(tc, phase)       # order load-bearing
        if clip_norm > 0:
            per_block_clip_(tc, clip_norm)

    opt.register_step_pre_hook(_pre_step)
    return opt


def test_phase1_steps_keep_shared_bitwise_frozen(tiny_model, tmp_path):
    """Two real phase-1 optimizer steps (author LM + generic suppression)
    through the belt hook: shared slices bitwise frozen (assert_shared_frozen
    green), the trained author block moves, idle authors stay bitwise (their
    grads are structurally exact zero AND fresh-AdamW zero moments take a
    zero step)."""
    tc = trained_tc()
    run_dir = str(tmp_path / "p0")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir, phase="phase0")

    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()
    state.set_phase("phase1")
    opt = _optimizer_with_belt(tc, "phase1")
    shared0 = _shared_snapshot(tc)
    idle = feat_rows(slot_of(19))  # never in a batch below
    idle0 = (tc.W_enc.detach()[idle].clone(),
             tc.W_dec.detach()[..., idle].clone())

    # step 1: single-source author LM batch
    batch = make_batch(B=2, T=8, source_ids=[7, 7], n_pad=0)
    opt.zero_grad(set_to_none=True)
    state.set_batch(batch["source_ids"],
                    attention_mask=batch["attention_mask"])
    try:
        out = tiny_model(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["labels"])
        out.loss.backward()
    finally:
        state.clear()
    opt.step()

    # step 2: generic suppression batch
    gb = make_batch(B=2, T=8, source_ids=[NO_AUTHOR] * 2, n_pad=0)
    opt.zero_grad(set_to_none=True)
    state.set_batch(gb["source_ids"], attention_mask=gb["attention_mask"])
    state.begin_suppression()
    try:
        tiny_model(input_ids=gb["input_ids"],
                   attention_mask=gb["attention_mask"])
        terms = state.end_suppression()
    finally:
        state.clear()
    (0.1 * terms[0]).backward()
    opt.step()

    assert _equal(_shared_snapshot(tc), shared0)     # bitwise frozen
    assert_shared_frozen(tc, run_dir)                # belt (d) agrees
    own = feat_rows(slot_of(7))
    assert not torch.equal(tc.W_enc.detach()[own],
                           trained_tc().W_enc.detach()[own])  # trained
    # idle author's DECODER cols bitwise (its encoder rows may move under
    # suppression — legal; its W_dec must never)
    assert torch.equal(tc.W_dec.detach()[..., idle], idle0[1])


def test_phase0_steps_keep_authors_bitwise_pristine(tiny_model):
    tc = build_tc()  # pristine seeded init — what assert_authors_pristine pins
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()
    state.set_phase("phase0")
    opt = _optimizer_with_belt(tc, "phase0")
    authors0 = _author_snapshot(tc)
    shared0 = _shared_snapshot(tc)

    for step_seed in (SEED, SEED + 1):
        batch = make_batch(B=2, T=8, source_ids=[NO_AUTHOR] * 2, n_pad=0,
                           seed=step_seed)
        opt.zero_grad(set_to_none=True)
        state.set_batch(batch["source_ids"],
                        attention_mask=batch["attention_mask"])
        try:
            out = tiny_model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"],
                             labels=batch["labels"])
            out.loss.backward()
        finally:
            state.clear()
        opt.step()

    assert _equal(_author_snapshot(tc), authors0)    # bitwise pristine
    assert_authors_pristine(tc)                      # belt (d) agrees
    assert not _equal(_shared_snapshot(tc), shared0)  # shared trained
