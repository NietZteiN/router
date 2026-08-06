"""CPU micro-regression for sift_masks.py — the correctness gate.

Run before any GPU/SLURM job:  python test_sift_masks.py

Uses a tiny RANDOM GPT-2 (no download, no TOFU) so the whole thing runs on CPU in
seconds. Covers exactly the invariants SIFT-Masks' exactness rests on:

  1. sign vector is ±1 and deterministic (same seed ⇒ identical v).
  2. sign-projection invariant: post-SIFT, τ⊙v ≥ 0 everywhere and mask == (τ != 0).
  3. determinism: same seed ⇒ byte-identical τ_t  (torch.equal). THIS is the exact-
     unlearning primitive.
  4. exact unlearning: build τ̄ over {0,1,2} streaming, unlearn task 1 by re-deriving
     τ_1 and subtracting; the result equals a from-scratch Σ over {0,2}. Asserted with
     allclose (fp addition is non-associative, so a running-sum-minus-term is not
     guaranteed bit-equal to a fresh retain-sum — we assert closeness and print the
     max ULP-scale gap, consistent with the documented caveat).
  5. serve identity: serving task t reproduces θ0 + (τ̄⊙m_t)/T exactly.
  6. mask pack/unpack round-trips bit-for-bit.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import sift_masks as sm


def tiny_model():
    cfg = GPT2Config(
        vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,   # no dropout -> deterministic
    )
    torch.manual_seed(0)
    m = GPT2LMHeadModel(cfg).to(torch.float32)
    m.eval()
    return m


def synth_batch(task_id: int, seqlen: int = 12, bs: int = 4, vocab: int = 64):
    """Deterministic per-task synthetic (input_ids, labels) with an ignored prefix."""
    g = torch.Generator().manual_seed(100 + task_id)
    ids = torch.randint(0, vocab, (bs, seqlen), generator=g)
    labels = ids.clone()
    labels[:, : seqlen // 2] = -100          # mimic an ignored "question" span
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def setup():
    model = tiny_model()
    names = sm.trainable_names(model)
    theta0 = sm.snapshot_params(model, names)
    sign = sm.make_sign_vector(model, names, seed=42)
    return model, names, theta0, sign


def run_task(model, theta0, sign, names, t, *, seed=None, steps=3):
    return sm.sift_one_task(
        model, theta0, sign, names, synth_batch(t),
        seed=(1000 + t if seed is None else seed), steps=steps, lr=1e-3, device="cpu",
    )


def test_sign_vector_pm1_and_deterministic():
    model, names, _, _ = setup()
    v1 = sm.make_sign_vector(model, names, seed=42)
    v2 = sm.make_sign_vector(model, names, seed=42)
    v3 = sm.make_sign_vector(model, names, seed=43)
    for n in names:
        assert torch.equal(v1[n].abs(), torch.ones_like(v1[n])), "v must be ±1"
        assert torch.equal(v1[n], v2[n]), "same seed must give identical v"
    assert any(not torch.equal(v1[n], v3[n]) for n in names), "different seed must differ"
    print("  [ok] sign vector is ±1 and seed-deterministic")


def test_projection_invariant():
    model, names, theta0, sign = setup()
    tau, mask = run_task(model, theta0, sign, names, 0)
    nz = 0
    for n in names:
        assert (tau[n] * sign[n] >= 0).all(), f"τ⊙v must be ≥ 0 after projection ({n})"
        assert torch.equal(mask[n], tau[n] != 0), f"mask must equal (τ≠0) ({n})"
        nz += int(mask[n].sum())
    assert nz > 0, "training must move at least some entries"
    print(f"  [ok] projection invariant holds (τ⊙v≥0, mask==τ≠0; {nz} active entries)")


def test_determinism():
    model, names, theta0, sign = setup()
    tau_a, _ = run_task(model, theta0, sign, names, 1, seed=7)
    tau_b, _ = run_task(model, theta0, sign, names, 1, seed=7)
    for n in names:
        assert torch.equal(tau_a[n], tau_b[n]), f"re-derivation not byte-identical ({n})"
    print("  [ok] τ re-derivation is byte-identical (exact-unlearning primitive)")


def test_exact_unlearning():
    model, names, theta0, sign = setup()
    # Build τ̄ over tasks {0,1,2}, streaming.
    tau_bar = sm.merge_init(theta0, names)
    taus = {}
    for t in (0, 1, 2):
        taus[t], _ = run_task(model, theta0, sign, names, t)
        sm.merge_add_(tau_bar, taus[t], names)

    # Unlearn task 1: re-derive τ_1 deterministically, subtract.
    tau1_redrv, _ = run_task(model, theta0, sign, names, 1)
    for n in names:                                   # determinism check first
        assert torch.equal(tau1_redrv[n], taus[1][n]), "τ_1 re-derivation drifted"
    sm.merge_sub_(tau_bar, tau1_redrv, names)

    # Compare to a from-scratch retain-set sum over {0,2}.
    fresh = sm.merge_init(theta0, names)
    sm.merge_add_(fresh, taus[0], names)
    sm.merge_add_(fresh, taus[2], names)

    max_gap = 0.0
    for n in names:
        max_gap = max(max_gap, float((tau_bar[n] - fresh[n]).abs().max()))
        assert torch.allclose(tau_bar[n], fresh[n], atol=1e-6, rtol=0), \
            f"unlearned τ̄ != retain sum beyond fp tolerance ({n})"
    print(f"  [ok] exact unlearning: (Στ)-τ_1 == Σ_retain  (max |gap|={max_gap:.2e}, "
          f"fp-noise scale; not bit-equal by non-associativity — see caveat)")


def test_serve_identity():
    model, names, theta0, sign = setup()
    tau_bar = sm.merge_init(theta0, names)
    masks = {}
    for t in (0, 1, 2):
        tau, masks[t], = (*sm.sift_one_task(
            model, theta0, sign, names, synth_batch(t),
            seed=1000 + t, steps=3, lr=1e-3, device="cpu"),)
        sm.merge_add_(tau_bar, tau, names)
    T = 3
    sm.serve_task_(model, theta0, tau_bar, masks[1], names, T)
    sd = dict(model.named_parameters())
    for n in names:
        expect = theta0[n] + (tau_bar[n] * masks[1][n].to(tau_bar[n].dtype)) / T
        assert torch.allclose(sd[n].data, expect, atol=0, rtol=0), f"serve mismatch ({n})"
    print("  [ok] serve identity: θ == θ0 + (τ̄⊙m_t)/T")


def test_mask_roundtrip():
    model, names, theta0, sign = setup()
    _, mask = run_task(model, theta0, sign, names, 0)
    packed = sm.pack_mask(mask, names)
    back = sm.unpack_mask(packed, names)
    for n in names:
        assert torch.equal(mask[n], back[n]), f"mask pack/unpack lost bits ({n})"
    # sanity: packed bytes are ~1/8 of the bool count (= ~1/32 of an fp32 tensor)
    raw_bits = sum(int(np.prod(mask[n].shape)) for n in names)
    packed_bytes = sum(packed[n]["bits"].nbytes for n in names)
    assert packed_bytes <= raw_bits / 8 + len(names), "packing did not compress"
    print(f"  [ok] mask pack/unpack is bit-exact ({raw_bits} bits -> {packed_bytes} bytes)")


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    tests = [
        test_sign_vector_pm1_and_deterministic,
        test_projection_invariant,
        test_determinism,
        test_exact_unlearning,
        test_serve_identity,
        test_mask_roundtrip,
    ]
    print(f"Running {len(tests)} SIFT-Masks CPU micro-tests...")
    for t in tests:
        t()
    print("ALL SIFT-MASKS TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
