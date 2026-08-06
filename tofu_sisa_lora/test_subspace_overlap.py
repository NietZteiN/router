"""CPU micro-tests for subspace_overlap.py. Run: python test_subspace_overlap.py

Guards the load-bearing identity (factored Frobenius inner product == dense) and the metric
sanity points (self/replicated saturate, independent-random are near-orthogonal).
"""
import torch

from jd_compress import Slot
import subspace_overlap as so


def _rand_slot(n, d_out, d_in, r, scaling, seed):
    g = torch.Generator().manual_seed(seed)
    B = [torch.randn(d_out, r, generator=g, dtype=torch.float64) for _ in range(n)]
    A = [torch.randn(r, d_in, generator=g, dtype=torch.float64) for _ in range(n)]
    return Slot(B=B, A=A, scaling=[scaling] * n)


def _dense(slot, i):
    return float(slot.scaling[i]) * (slot.B[i].double() @ slot.A[i].double())


def test_factored_equals_dense():
    """_frob_inner(slot,i,j) must equal (DW_i * DW_j).sum() on the dense deltas."""
    slot = _rand_slot(3, 16, 24, 4, scaling=2.5, seed=0)
    for i in range(3):
        for j in range(3):
            factored = so._frob_inner(slot, i, j)
            dense = (_dense(slot, i) * _dense(slot, j)).sum().item()
            rel = abs(factored - dense) / max(abs(dense), 1e-9)
            assert rel < 1e-9, f"factored!=dense at ({i},{j}): {factored} vs {dense} (rel {rel})"
    print("  ok  factored Frobenius inner == dense (rel < 1e-9)")


def test_vectorized_inner_matches_loop():
    """pairwise_inner (batched einsum) must equal the _frob_inner double loop, summed over slots."""
    slots = {"s0": _rand_slot(4, 32, 40, 4, 1.7, 11), "s1": _rand_slot(4, 32, 40, 4, 0.9, 12)}
    vec = so.pairwise_inner(slots)
    for i in range(4):
        for j in range(4):
            loop = sum(so._frob_inner(slots[name], i, j) for name in slots)
            rel = abs(vec[i, j].item() - loop) / max(abs(loop), 1e-9)
            assert rel < 1e-9, f"vectorized!=loop at ({i},{j}): {vec[i,j].item()} vs {loop} (rel {rel})"
    print("  ok  vectorized pairwise_inner == _frob_inner loop (rel < 1e-9)")


def test_cosine_self_and_replicated():
    slots = {"s0": _rand_slot(3, 32, 40, 4, 1.7, 1), "s1": _rand_slot(3, 32, 40, 4, 0.9, 2)}
    # adapter 2 := exact copy of adapter 0 in both slots
    for name in slots:
        s = slots[name]
        s.B[2] = s.B[0].clone()
        s.A[2] = s.A[0].clone()
    cos = so.pairwise_cosine(slots)
    assert abs(cos[0, 0].item() - 1.0) < 1e-9, "self-cosine must be 1"
    assert abs(cos[0, 2].item() - 1.0) < 1e-6, f"replicated cosine must be 1, got {cos[0,2].item()}"
    print(f"  ok  cosine self=1, replicated[0,2]={cos[0,2].item():.6f}")


def test_principal_angle_replicated():
    slots = {"s0": _rand_slot(2, 48, 48, 4, 1.0, 3)}
    s = slots["s0"]
    s.B[1] = s.B[0].clone()
    s.A[1] = s.A[0].clone()
    angB, angA = so.principal_angle_cos(slots)
    assert abs(angB[0, 1].item() - 1.0) < 1e-6, f"replicated col(B) angle-cos must be 1, got {angB[0,1].item()}"
    assert abs(angA[0, 1].item() - 1.0) < 1e-6, f"replicated row(A) angle-cos must be 1, got {angA[0,1].item()}"
    print(f"  ok  principal-angle cos replicated: B={angB[0,1].item():.6f} A={angA[0,1].item():.6f}")


def test_independent_random_near_zero():
    """Two independent random deltas in high-d should have small |cosine| (not saturated)."""
    slots = {f"s{l}": _rand_slot(2, 256, 256, 8, 1.0, 100 + l) for l in range(8)}
    cos = so.pairwise_cosine(slots)
    off = abs(cos[0, 1].item())
    assert off < 0.3, f"independent-random cosine should be small, got {off}"
    # and the orthogonal null should sit near zero too
    nm = so.null_summary(slots, "orthogonal", seed=42, n_null=5)
    assert abs(nm["cosine"]["mean"]) < 0.3, nm
    print(f"  ok  independent-random cosine[0,1]={off:.4f}, null cosine mean={nm['cosine']['mean']:.4f}")


def test_shared_subspace_energy_bounds():
    slots = {"s0": _rand_slot(4, 64, 64, 4, 1.0, 7)}
    e = so.shared_subspace_energy(slots, rank=8)
    assert 0.0 <= e["mean_energy_retained"] <= 1.0 + 1e-6, e
    # replicate all 4 onto adapter 0 -> a single rank-4 basis captures (nearly) all energy
    s = slots["s0"]
    for i in range(4):
        s.B[i] = s.B[0].clone()
        s.A[i] = s.A[0].clone()
    e2 = so.shared_subspace_energy(slots, rank=8)
    assert e2["mean_energy_retained"] > 0.99, f"replicated collection must be fully captured, got {e2}"
    print(f"  ok  shared-subspace energy in [0,1]; replicated={e2['mean_energy_retained']:.4f}")


if __name__ == "__main__":
    test_factored_equals_dense()
    test_vectorized_inner_matches_loop()
    test_cosine_self_and_replicated()
    test_principal_angle_replicated()
    test_independent_random_near_zero()
    test_shared_subspace_energy_bounds()
    print("ALL SUBSPACE_OVERLAP TESTS PASSED")
