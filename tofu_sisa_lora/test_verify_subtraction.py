"""CPU micro-tests for verify_subtraction.py (no downloads, no GPU, no model load).

Run before any ctv SLURM job: python test_verify_subtraction.py

Tiny 3-author pool, two constructions:
  1. EXACT triple — adapters store integer-valued dense deltas directly (A = I, B = D, all
     values fp32-exact), merged = D1+D2+D3, remerged = D1+D2, tau = D3, so merged − tau is
     byte-identical to remerged: declared bitwise must HOLD.
  2. fp factor-cat triple — the real additive_sum artifact shape (random factors, merged =
     [1,1,1] cat, remerged = [1,1] cat): subtraction is exact algebra but fp GEMM
     reassociation means only the algebraic class is guaranteed; the report's bitwise flag
     and measured class must agree with each other either way.
Plus the planted violation: tau perturbed by 1e-3 — declared algebraic must be VIOLATED
with the measured class downgraded to first_order (and a 2x-scale violation downgrades all
the way to approximate); and a CLI round-trip (exit codes + report JSON).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_subset import _weighted_factor_cat, merge_additive_sum, write_effective_adapter  # noqa: E402
from verify_subtraction import CLASS_LADDER, class_holds, verify  # noqa: E402

D_OUT, D_IN = 12, 16
SLOTS = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"]
REF_CFG = {"peft_type": "LORA", "r": D_IN, "lora_alpha": D_IN, "use_rslora": False,
           "target_modules": ["q_proj", "up_proj"]}


def write_dense_adapter(out_dir, deltas):
    """Adapter whose effective delta IS `deltas`, stored exactly: A = I (rank d_in),
    B = delta, scaling 1.0 (write_effective_adapter forces alpha == r, rslora off). B @ I
    only adds exact +0.0 terms, so the verifier's dense product reproduces delta bytewise."""
    slots_AB = {s: (torch.eye(D_IN), d.clone()) for s, d in deltas.items()}
    return write_effective_adapter(out_dir, slots_AB, REF_CFG, D_IN)


def int_deltas(rng):
    return {s: torch.tensor(rng.randint(-8, 9, size=(D_OUT, D_IN)), dtype=torch.float32)
            for s in SLOTS}


def test_exact_triple(tmp):
    """Integer construction: merged − tau is fp32-exact ⇒ bitwise holds, rel_l2 == 0."""
    rng = np.random.RandomState(42)
    d1, d2, d3 = int_deltas(rng), int_deltas(rng), int_deltas(rng)
    write_dense_adapter(os.path.join(tmp, "merged"), {s: d1[s] + d2[s] + d3[s] for s in SLOTS})
    write_dense_adapter(os.path.join(tmp, "remerged"), {s: d1[s] + d2[s] for s in SLOTS})
    write_dense_adapter(os.path.join(tmp, "tau"), d3)

    rep = verify(os.path.join(tmp, "merged"), os.path.join(tmp, "remerged"),
                 os.path.join(tmp, "tau"), declared_class="bitwise")
    assert rep["global"]["bitwise_identical"], rep["global"]
    assert rep["global"]["rel_l2_max"] == 0.0, rep["global"]
    assert rep["measured_class"] == "bitwise" and rep["declared_class_holds"]
    assert all(m["bitwise_identical"] for m in rep["per_module"].values())
    assert rep["global"]["n_modules"] == len(SLOTS)
    print("ok  exact triple: bitwise_identical on every module, rel_l2 == 0, "
          "declared bitwise HOLDS")


def test_planted_violation(tmp):
    """tau off by 1e-3: declared algebraic VIOLATED, measured class downgrades to
    first_order; a 2x tau downgrades to approximate. The class is checked against the
    declaration, never silently replaced by the measurement."""
    rng = np.random.RandomState(7)
    d1, d2, d3 = int_deltas(rng), int_deltas(rng), int_deltas(rng)
    write_dense_adapter(os.path.join(tmp, "v_merged"),
                        {s: d1[s] + d2[s] + d3[s] for s in SLOTS})
    write_dense_adapter(os.path.join(tmp, "v_remerged"), {s: d1[s] + d2[s] for s in SLOTS})
    write_dense_adapter(os.path.join(tmp, "v_tau_bad"), {s: d3[s] * (1 + 1e-3) for s in SLOTS})
    write_dense_adapter(os.path.join(tmp, "v_tau_2x"), {s: d3[s] * 2.0 for s in SLOTS})

    rep = verify(os.path.join(tmp, "v_merged"), os.path.join(tmp, "v_remerged"),
                 os.path.join(tmp, "v_tau_bad"), declared_class="algebraic")
    assert not rep["declared_class_holds"], "planted 1e-3 violation not caught"
    assert rep["measured_class"] == "first_order", rep["measured_class"]
    assert not rep["global"]["bitwise_identical"]
    assert 1e-6 < rep["global"]["rel_l2_max"] < 1e-2, rep["global"]["rel_l2_max"]
    # the same measurement PASSES a first_order declaration — the ladder, not a boolean
    rep_fo = verify(os.path.join(tmp, "v_merged"), os.path.join(tmp, "v_remerged"),
                    os.path.join(tmp, "v_tau_bad"), declared_class="first_order")
    assert rep_fo["declared_class_holds"]

    rep2 = verify(os.path.join(tmp, "v_merged"), os.path.join(tmp, "v_remerged"),
                  os.path.join(tmp, "v_tau_2x"), declared_class="first_order")
    assert not rep2["declared_class_holds"]
    assert rep2["measured_class"] == "approximate", rep2["measured_class"]
    print(f"ok  planted violation: algebraic VIOLATED -> measured first_order "
          f"(rel {rep['global']['rel_l2_max']:.2e}); 2x tau -> approximate")


def test_factor_cat_triple(tmp):
    """Real additive_sum artifact shape: merged = [1,1,1] factor cat of random rslora-style
    factors, remerged = [1,1] cat, tau = the third adapter. Subtraction is exact algebra;
    fp GEMM reassociation may or may not break byte equality, so assert the guaranteed
    class (algebraic) and internal consistency of the bitwise flag with the class."""
    gen = torch.Generator().manual_seed(42)
    dirs = []
    for i in range(3):
        slots_AB = {s: (torch.randn(4, D_IN, generator=gen) * 0.05,
                        torch.randn(D_OUT, 4, generator=gen) * 0.05) for s in SLOTS}
        d = os.path.join(tmp, f"author_{i}")
        write_effective_adapter(d, slots_AB, REF_CFG, 4)
        dirs.append(d)
    merged, ref_cfg, out_rank, _ = merge_additive_sum(dirs)
    write_effective_adapter(os.path.join(tmp, "f_merged"), merged, ref_cfg, out_rank)
    rem, ref_cfg, out_rank, _ = _weighted_factor_cat(dirs[:2], [1.0, 1.0])
    write_effective_adapter(os.path.join(tmp, "f_remerged"), rem, ref_cfg, out_rank)

    rep = verify(os.path.join(tmp, "f_merged"), os.path.join(tmp, "f_remerged"), dirs[2],
                 declared_class="algebraic")
    assert rep["declared_class_holds"], rep["global"]
    assert rep["global"]["rel_l2_max"] <= 1e-6, rep["global"]["rel_l2_max"]
    bitwise = rep["global"]["bitwise_identical"]
    assert rep["measured_class"] == ("bitwise" if bitwise else "algebraic")
    # determinism: identical inputs => identical report numbers
    rep_b = verify(os.path.join(tmp, "f_merged"), os.path.join(tmp, "f_remerged"), dirs[2],
                   declared_class="algebraic")
    assert rep_b["global"] == rep["global"]
    print(f"ok  factor-cat triple: additive_sum minus tau == remerge at algebraic class "
          f"(rel {rep['global']['rel_l2_max']:.2e}, bitwise={bitwise}); deterministic")


def test_tau_weight(tmp):
    """Mean-mode identity: merged = mean of 3, remerged = (D1+D2)/3 cat, tau_weight = 1/3.
    The weight is DECLARED by the caller — the wrong weight must be caught."""
    gen = torch.Generator().manual_seed(3)
    dirs = []
    for i in range(3):
        slots_AB = {s: (torch.randn(4, D_IN, generator=gen) * 0.05,
                        torch.randn(D_OUT, 4, generator=gen) * 0.05) for s in SLOTS}
        d = os.path.join(tmp, f"w_author_{i}")
        write_effective_adapter(d, slots_AB, REF_CFG, 4)
        dirs.append(d)
    merged, ref_cfg, out_rank, _ = _weighted_factor_cat(dirs, [1 / 3] * 3)
    write_effective_adapter(os.path.join(tmp, "w_merged"), merged, ref_cfg, out_rank)
    rem, ref_cfg, out_rank, _ = _weighted_factor_cat(dirs[:2], [1 / 3, 1 / 3])
    write_effective_adapter(os.path.join(tmp, "w_remerged"), rem, ref_cfg, out_rank)

    ok = verify(os.path.join(tmp, "w_merged"), os.path.join(tmp, "w_remerged"), dirs[2],
                tau_weight=1 / 3, declared_class="algebraic")
    assert ok["declared_class_holds"], ok["global"]
    bad = verify(os.path.join(tmp, "w_merged"), os.path.join(tmp, "w_remerged"), dirs[2],
                 tau_weight=1.0, declared_class="algebraic")
    assert not bad["declared_class_holds"] and bad["measured_class"] == "approximate"
    print(f"ok  tau_weight: mean-mode 1/3 holds ({ok['global']['rel_l2_max']:.2e}); "
          f"wrong weight caught as approximate ({bad['global']['rel_l2_max']:.2e})")


def test_cli(tmp):
    """CLI round-trip: exit 0 + report JSON on the exact triple, exit 1 on the violation."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_subtraction.py")
    rep_path = os.path.join(tmp, "report.json")
    r = subprocess.run(
        [sys.executable, script, "--merged", os.path.join(tmp, "merged"),
         "--remerged", os.path.join(tmp, "remerged"), "--tau", os.path.join(tmp, "tau"),
         "--declared_class", "bitwise", "--report", rep_path],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open(rep_path) as f:
        rep = json.load(f)
    assert rep["declared_class_holds"] and rep["measured_class"] == "bitwise"
    assert rep["tau_weight"] == 1.0 and set(rep["per_module"]) == set(SLOTS)

    r = subprocess.run(
        [sys.executable, script, "--merged", os.path.join(tmp, "v_merged"),
         "--remerged", os.path.join(tmp, "v_remerged"), "--tau", os.path.join(tmp, "v_tau_bad"),
         "--declared_class", "algebraic"],
        capture_output=True, text=True)
    assert r.returncode == 1, "violation must exit nonzero so drivers can gate"
    assert "VIOLATED" in r.stdout
    print("ok  CLI: exit 0 + JSON report on pass, exit 1 + VIOLATED verdict on the plant")


def test_ladder():
    assert CLASS_LADDER == ["bitwise", "algebraic", "first_order", "approximate"]
    for i, dec in enumerate(CLASS_LADDER):
        for j, mea in enumerate(CLASS_LADDER):
            assert class_holds(dec, mea) == (j <= i)
    print("ok  class ladder: declaration holds iff measurement is at least as strong")


def main():
    tmp = tempfile.mkdtemp(prefix="test_verify_subtraction_")
    try:
        test_exact_triple(tmp)
        test_planted_violation(tmp)
        test_factor_cat_triple(tmp)
        test_tau_weight(tmp)
        test_cli(tmp)
        test_ladder()
        print("ALL OK  test_verify_subtraction")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
