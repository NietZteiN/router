"""CPU micro-regression for clamu.py — the ClAMU correctness gate.

Run before any GPU/SLURM job:  python test_clamu.py

Tiny RANDOM GPT-2 (no download, no TOFU), CPU, seconds. Covers the invariants ClAMU's
exactness + mechanism rest on:

  1. STE mask: forward == 1{s>0}; backward grad == σ'(s) (straight-through estimator).
  2. cluster determinism: same seed ⇒ identical feature/random partition.
  3. exact unlearning (no-sign path): build τ̄ over {0,1,2} streaming with
     use_sign_constraint=False, unlearn task 1 by re-deriving τ_1 and subtracting; equals
     a from-scratch Σ over {0,2} (allclose; fp non-associativity caveat as in SIFT).
  4. mask optimization works: optimize_mask_ste reduces CE from its (mask=0) start AND is
     deterministic (same seed ⇒ identical mask). [The "optimized beats EMR/merge" claim is
     an empirical, scale-dependent result validated at the GPU smoke, not asserted here.]
  5. serve identity (reused from sift_masks): θ == θ0 + (τ̄⊙m_c)/T.
  6. mask pack/unpack round-trips bit-for-bit.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import clamu as cl
import sift_masks as sm


def tiny_model():
    cfg = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    torch.manual_seed(0)
    m = GPT2LMHeadModel(cfg).to(torch.float32)
    m.eval()
    return m


def synth_batch(task_id: int, seqlen: int = 12, bs: int = 4, vocab: int = 64):
    g = torch.Generator().manual_seed(100 + task_id)
    ids = torch.randint(0, vocab, (bs, seqlen), generator=g)
    labels = ids.clone()
    labels[:, : seqlen // 2] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def setup():
    model = tiny_model()
    names = sm.trainable_names(model)
    theta0 = sm.snapshot_params(model, names)
    return model, names, theta0


def _ft(model, theta0, names, t, *, seed, steps=3):
    """Deterministic full-FT delta (no sign constraint) — the ClAMU build primitive."""
    return sm.sift_one_task(model, theta0, None, names, synth_batch(t),
                            seed=seed, steps=steps, lr=1e-3, device="cpu",
                            use_sign_constraint=False)[0]


def test_ste_gradient():
    s = torch.tensor([-1.5, -0.1, 0.0, 0.3, 2.0], requires_grad=True)
    y = cl.ste_mask(s)
    assert torch.equal(y, (s.detach() > 0).float()), "STE forward must be 1{s>0}"
    y.sum().backward()
    sig = torch.sigmoid(s.detach())
    assert torch.allclose(s.grad, sig * (1 - sig), atol=1e-6), "STE backward must be σ'(s)"
    print("  [ok] STE mask: forward 1{s>0}, backward σ'(s)")


def test_cluster_determinism():
    cfg = {"num_clusters": 3, "kmeans_seed": 1, "output_dir": "/tmp/_clamu_test",
           "num_authors": 20}
    emb = np.random.RandomState(0).randn(20, 8).astype("float32")
    for method in ("feature", "random"):
        cfg["cluster_affinity"] = method
        a1 = cl.cluster_authors(cfg, emb, cache=False)
        a2 = cl.cluster_authors(cfg, emb, cache=False)
        assert a1["author_to_cluster"] == a2["author_to_cluster"], f"{method} not deterministic"
        assert sum(a1["sizes"]) == 20, "every author must be assigned exactly once"
    print("  [ok] clustering is seed-deterministic (feature + random), full partition")


def test_exact_unlearning_no_sign():
    model, names, theta0 = setup()
    tau_bar = sm.merge_init(theta0, names)
    taus = {t: _ft(model, theta0, names, t, seed=1000 + t) for t in (0, 1, 2)}
    for t in (0, 1, 2):
        sm.merge_add_(tau_bar, taus[t], names)

    tau1_re = _ft(model, theta0, names, 1, seed=1001)
    for n in names:
        assert torch.equal(tau1_re[n], taus[1][n]), f"τ_1 re-derivation drifted ({n})"
    sm.merge_sub_(tau_bar, tau1_re, names)

    fresh = sm.merge_init(theta0, names)
    sm.merge_add_(fresh, taus[0], names)
    sm.merge_add_(fresh, taus[2], names)
    max_gap = max(float((tau_bar[n] - fresh[n]).abs().max()) for n in names)
    for n in names:
        assert torch.allclose(tau_bar[n], fresh[n], atol=1e-6, rtol=0), f"unlearn != retain ({n})"
    print(f"  [ok] exact unlearning (no-sign): (Στ)-τ_1 == Σ_retain  (max |gap|={max_gap:.2e})")


def test_mask_optimization():
    model, names, theta0 = setup()
    # a non-trivial τ̄ = sum of two full-FT deltas
    tau_bar = sm.merge_init(theta0, names)
    for t in (0, 1):
        sm.merge_add_(tau_bar, _ft(model, theta0, names, t, seed=2000 + t), names)
    T = 2
    batch = synth_batch(0)

    def served_ce(mask):
        sm.serve_task_(model, theta0, tau_bar, mask, names, T)
        with torch.no_grad():
            return float(model(**batch).loss)

    zero_mask = {n: torch.zeros_like(tau_bar[n], dtype=torch.bool) for n in names}
    ce_start = served_ce(zero_mask)                         # mask=0 == base θ0 (the STE init)

    m1 = cl.optimize_mask_ste(tiny_model_from(model, theta0, names), names, tau_bar, [batch],
                              T=T, steps=80, lr=0.2, seed=7, device="cpu")
    ce_opt = served_ce(m1)
    m2 = cl.optimize_mask_ste(tiny_model_from(model, theta0, names), names, tau_bar, [batch],
                              T=T, steps=80, lr=0.2, seed=7, device="cpu")
    for n in names:
        assert torch.equal(m1[n], m2[n]), f"optimized mask not deterministic ({n})"
    assert ce_opt < ce_start - 1e-3, f"STE did not reduce CE ({ce_start:.4f} -> {ce_opt:.4f})"
    print(f"  [ok] STE mask optimization reduces CE ({ce_start:.4f} -> {ce_opt:.4f}) "
          f"and is seed-deterministic")


def tiny_model_from(ref_model, theta0, names):
    """A fresh tiny model whose θ0 matches `ref_model` (so optimize_mask_ste sees the
    same base weights regardless of any in-place serving done to ref_model)."""
    m = tiny_model()
    sm.load_params_(m, theta0, names)
    return m


def test_serve_identity():
    model, names, theta0 = setup()
    tau_bar = sm.merge_init(theta0, names)
    for t in (0, 1, 2):
        sm.merge_add_(tau_bar, _ft(model, theta0, names, t, seed=3000 + t), names)
    T = 3
    g = torch.Generator().manual_seed(5)
    mask = {n: (torch.rand(tau_bar[n].shape, generator=g) > 0.5) for n in names}
    sm.serve_task_(model, theta0, tau_bar, mask, names, T)
    sd = dict(model.named_parameters())
    for n in names:
        expect = theta0[n] + (tau_bar[n] * mask[n].to(tau_bar[n].dtype)) / T
        assert torch.allclose(sd[n].data, expect, atol=0, rtol=0), f"serve mismatch ({n})"
    print("  [ok] serve identity: θ == θ0 + (τ̄⊙m_c)/T")


def test_mask_roundtrip():
    model, names, theta0 = setup()
    g = torch.Generator().manual_seed(1)
    mask = {n: (torch.rand(theta0[n].shape, generator=g) > 0.5) for n in names}
    back = sm.unpack_mask(sm.pack_mask(mask, names), names)
    for n in names:
        assert torch.equal(mask[n], back[n]), f"pack/unpack lost bits ({n})"
    print("  [ok] mask pack/unpack is bit-exact")


def test_localize_steps():
    # mask_epochs: equal epochs regardless of cluster size (the K-dial fairness rule).
    assert cl.localize_steps({"mask_epochs": 4}, 12) == 48
    assert cl.localize_steps({"mask_epochs": 4}, 200) == 800
    assert cl.localize_steps({"mask_epochs": 0.1}, 3) == 1     # floor at 1
    # fallback: fixed mask_steps (the 07-02 headline recipe)
    assert cl.localize_steps({"mask_steps": 50}, 200) == 50
    assert cl.localize_steps({}, 7) == 50                       # default
    print("  [ok] localize_steps: epochs-scaled with floor, mask_steps fallback")


def test_label_mask_kind():
    from clamu_model import _mask_kind
    assert _mask_kind("clamu_full") == "clamu" and _mask_kind("clamu_unlearn") == "clamu"
    assert _mask_kind("emr_full") == "emr" and _mask_kind("tall_unlearn") == "tall"
    assert _mask_kind("merge_full") is None                     # Global baseline, no mask
    try:
        _mask_kind("bogus_full")
        raise AssertionError("bogus label must raise")
    except SystemExit:
        pass
    print("  [ok] label -> mask-kind dispatch (clamu/emr/tall/merge, bogus rejected)")


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    tests = [
        test_ste_gradient,
        test_cluster_determinism,
        test_exact_unlearning_no_sign,
        test_mask_optimization,
        test_serve_identity,
        test_mask_roundtrip,
        test_localize_steps,
        test_label_mask_kind,
    ]
    print(f"Running {len(tests)} ClAMU CPU micro-tests...")
    for t in tests:
        t()
    print("ALL CLAMU TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
