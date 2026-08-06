"""CPU micro-tests for sparsify_pool.py (no downloads, no GPU, no real pool).

Run before any sparsify SLURM job: python test_sparsify_pool.py

Builds 4 tiny per-author adapter dirs in tmp with the merge_subset-compatible layout
(adapter_config.json + adapter_model.safetensors keyed base_model.model.<slot>.lora_{A,B}
.weight — matched to jd_collection._read_adapter) for the authors subset_authors(42, 4)
derives at runtime (never hardcoded), then checks:

  1. op parsing + the three sparsifiers' closed-form properties: dare survivor fraction
     and exact 1/(1-p) rescale + nested masks across rates; topk keeps exactly
     ceil(q*rows) top-norm rows; hash blocks partition the rows disjointly and cover.
  2. the full grid (via main(), exercising --ops/--n_ladder/--limit_authors): composed
     cat delta == the manual weighted sum of the sparsified per-author deltas (1/N for
     dare/topk, 1.0 disjoint sum for hash — with disjoint row support verified), and the
     stored dare adapters == a direct re-application of the seeded masks.
  3. byte-wise determinism: a second grid run into a fresh out_dir produces
     sha256-identical adapter_model.safetensors + adapter_config.json everywhere.
  4. the eval manifest parses (4-col nmerge format), references existing dirs, carries
     the derived probe sids per label with the perm[0]=headline row first, plus iso rows.
  5. DX1 math on synthetic deltas with KNOWN cancellation: a sign-opposed pair -> ratio
     ~0, an identical-sign pair -> ratio 1, null draws in [0,1] and above the opposed
     observed; deterministic. Plus a full run_dx1 pass over the tiny pool (JSON shape).
  6. DX2 == the analytic value on planted matrices (uniform rows -> |rows|/d_out; block-
     supported B -> 1.0 own / 0.0 foreign), plus a full run_dx2 pass (JSON shape, bounds).
  7. --dry_run touches no real pool: succeeds with a nonexistent pool_dir and writes
     nothing.
"""

import hashlib
import json
import math
import os
import shutil
import tempfile

import numpy as np
import torch

import sparsify_pool as sp
from jd_collection import _adapter_scaling, _read_adapter, _PREFIX
from merge_subset import probe_authors, subset_authors, N_AUTHORS
from safetensors.torch import save_file

SEED = 42
N_POOL = 4
RANK = 4
# (d_out, d_in) per slot — two distinct d_outs + two slot types exercise the per-slot
# hash permutations and the per-slot-type DX aggregation.
SLOTS = {
    "model.layers.0.self_attn.q_proj": (16, 16),
    "model.layers.0.mlp.up_proj": (24, 16),
    "model.layers.1.self_attn.q_proj": (16, 16),
}

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def build_pool(root):
    """N_POOL tiny per-author adapter dirs for the runtime-derived subset_authors(42, 4)."""
    authors = subset_authors(SEED, N_POOL)
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": 8,
           "use_rslora": True, "lora_dropout": 0.0, "bias": "none",
           "target_modules": ["q_proj", "up_proj"]}
    for a in authors:
        gen = torch.Generator().manual_seed(1000 + a)
        tensors = {}
        for slot, (d_out, d_in) in SLOTS.items():
            A = (torch.randn(RANK, d_in, generator=gen) * 0.1).contiguous()
            B = (torch.randn(d_out, RANK, generator=gen) * 0.1).contiguous()
            tensors[_PREFIX + slot + ".lora_A.weight"] = A
            tensors[_PREFIX + slot + ".lora_B.weight"] = B
        d = os.path.join(root, f"shard_{a}")
        os.makedirs(d)
        save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    return authors


def write_config(tmp, pool_dir, out_dir, name):
    ref = os.path.join(tmp, "retain_tr_scores.npy")
    if not os.path.exists(ref):
        np.save(ref, np.linspace(0.1, 0.9, 20))
    cfg = {"model_name": "tiny-fixture", "pool_dir": pool_dir, "out_dir": out_dir,
           "retain_tr_source": ref,
           "ops": ["dare0p5", "dare0p9", "topk0p25", "hash", "dare0p9sum"],
           "n_ladder": [2, 4], "pool_seed": SEED, "pool_size": 200, "n_probes": 5,
           "probe_authors": probe_authors(SEED, N_AUTHORS, 5),
           "dx1": {"n_values": [2, 4], "null_draws": 3},
           "eval": {"k": 200, "forget_shard_id": 199, "cap": "smoke"}}
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path, cfg


def max_rel_err(got, ref):
    denom = ref.abs().max().clamp_min(1e-8)
    return ((got - ref).abs().max() / denom).item()


def read_delta(adapter_dir):
    """{slot: effective dense delta scaling * B @ A} straight from an adapter dir."""
    slots, cfg = _read_adapter(adapter_dir)
    scale = _adapter_scaling(cfg)
    return {k: scale * (B @ A) for k, (A, B) in slots.items()}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# 1. ops
# ---------------------------------------------------------------------------

def test_op_parsing():
    assert sp.parse_op("dare0p5") == ("dare", 0.5)
    assert sp.parse_op("dare0p9") == ("dare", 0.9)
    assert sp.parse_op("topk0p25") == ("topk", 0.25)
    assert sp.parse_op("hash") == ("hash", None)
    assert sp.parse_op("dare0p9sum") == ("daresum", 0.9)
    assert sp.parse_op("dare0p99sum") == ("daresum", 0.99)
    assert sp.parse_op("topk0p25sum") == ("topksum", 0.25)
    assert sp._canonical_op("dare0p9sum") == "dare0p9"
    assert sp._canonical_op("dare0p9") == "dare0p9"
    for bad in ("dare", "dare0.5", "rowslice", "topk", "hashsum", "sum", "dare0p9summ"):
        try:
            sp.parse_op(bad)
            raise AssertionError(f"parse_op accepted {bad!r}")
        except ValueError:
            pass
    assert sp.sparse_label("dare0p9", 8, 42) == "sparse_dare0p9_N8_s42"
    assert sp.sparse_label("hash", 16, 42) == "sparse_hash_N16_s42"
    print("ok  op parsing + label grammar")


def test_dare():
    t = torch.randn(200, 64, generator=torch.Generator().manual_seed(3))
    p = 0.9
    out = sp.dare_mask(t, p, torch.Generator().manual_seed(7))
    nz = out != 0
    frac = nz.float().mean().item()
    assert abs(frac - (1 - p)) < 0.02, frac  # binomial std ~0.003 at 12800 entries
    assert torch.equal(out[nz], t[nz] / (1 - p)), "survivors not exactly 1/(1-p)-rescaled"
    out2 = sp.dare_mask(t, p, torch.Generator().manual_seed(7))
    assert torch.equal(out, out2), "dare not deterministic at fixed seed"
    # same seed, two rates -> nested masks (one uniform field, thresholded twice)
    m5 = sp.dare_mask(t, 0.5, torch.Generator().manual_seed(7)) != 0
    assert not (nz & ~m5).any(), "keep@0.9 not a subset of keep@0.5 at the same seed"
    print(f"ok  dare: survivor fraction {frac:.3f} ~ {1 - p}, exact rescale, "
          f"deterministic, nested across rates")


def test_topk():
    B = torch.randn(17, RANK, generator=torch.Generator().manual_seed(5))
    q = 0.25
    out = sp.topk_row_mask(B, q)
    want_k = math.ceil(q * 17)  # = 5
    kept = (out.abs().sum(dim=1) > 0).nonzero().flatten()
    assert len(kept) == want_k, (len(kept), want_k)
    top = torch.topk(B.norm(dim=1), want_k).indices
    assert set(kept.tolist()) == set(top.tolist()), "kept rows are not the top-norm rows"
    assert torch.equal(out[kept], B[kept]), "kept rows were modified"
    print(f"ok  topk_rows: keeps exactly ceil(q*rows)={want_k} top-norm rows verbatim")


def test_hash_blocks():
    for d_out, n in ((16, 4), (24, 4), (17, 3)):
        blocks = sp.hash_row_blocks(d_out, n, SEED, "model.layers.0.mlp.up_proj")
        allrows = np.concatenate(blocks)
        assert len(allrows) == d_out and len(set(allrows.tolist())) == d_out, \
            "blocks do not partition+cover"
        sizes = [len(b) for b in blocks]
        assert max(sizes) - min(sizes) <= 1, sizes
        again = sp.hash_row_blocks(d_out, n, SEED, "model.layers.0.mlp.up_proj")
        assert all(np.array_equal(a, b) for a, b in zip(blocks, again)), "not deterministic"
    b1 = sp.hash_row_blocks(16, 4, SEED, "model.layers.0.self_attn.q_proj")
    b2 = sp.hash_row_blocks(16, 4, SEED, "model.layers.1.self_attn.q_proj")
    assert any(not np.array_equal(x, y) for x, y in zip(b1, b2)), \
        "different slots should draw different permutations"
    try:
        sp.hash_row_blocks(3, 4, SEED, "s")
        raise AssertionError("n > d_out should raise")
    except ValueError:
        pass
    print("ok  hash blocks: disjoint, covering, near-equal, per-slot seeded, deterministic")


# ---------------------------------------------------------------------------
# 2–4. grid, determinism, manifest
# ---------------------------------------------------------------------------

def test_grid_composition(tmp, pool_dir, out_dir, cfg_path):
    sp.main(["--config", cfg_path, "--limit_authors", str(N_POOL)])

    # dare0p9 N=4: composed delta == (1/4) sum_i s_i dare(B_i) @ dare(A_i)
    m_dir = os.path.join(out_dir, "merges", "sparse_dare0p9_N4_s42")
    slots_m, cfg_m = _read_adapter(m_dir)
    assert abs(_adapter_scaling(cfg_m) - 1.0) < 1e-9 and not cfg_m["use_rslora"]
    assert cfg_m["r"] == N_POOL * RANK, cfg_m["r"]
    got = read_delta(m_dir)
    authors = subset_authors(SEED, 4)
    parts = [read_delta(os.path.join(out_dir, "sparse_adapters", "dare0p9", f"shard_{a}"))
             for a in authors]
    worst = 0.0
    for slot in got:
        ref = sum(p[slot] for p in parts) / len(parts)
        worst = max(worst, max_rel_err(got[slot], ref))
    assert worst < 1e-4, worst
    print(f"ok  composed dare0p9 N=4 == manual 1/N sum of sparsified deltas "
          f"(max rel err {worst:.2e})")

    # stored dare adapters == direct re-application of the per-(author,tensor) seeded masks
    a0 = authors[0]
    raw, raw_cfg = _read_adapter(os.path.join(pool_dir, f"shard_{a0}"))
    stored, _ = _read_adapter(os.path.join(out_dir, "sparse_adapters", "dare0p9", f"shard_{a0}"))
    for slot, (A, B) in raw.items():
        gb = torch.Generator().manual_seed(sp._sha_seed("ctv_dare", SEED, a0, slot, "B"))
        ga = torch.Generator().manual_seed(sp._sha_seed("ctv_dare", SEED, a0, slot, "A"))
        assert torch.equal(stored[slot][1], sp.dare_mask(B, 0.9, gb)), slot
        assert torch.equal(stored[slot][0], sp.dare_mask(A, 0.9, ga)), slot
    print("ok  stored dare adapters reproduce the SHA-256-seeded masks bit-exactly")

    # hash N=2: weights 1.0 (disjoint sum) + row supports actually disjoint
    h_dir = os.path.join(out_dir, "merges", "sparse_hash_N2_s42")
    got_h = read_delta(h_dir)
    authors2 = subset_authors(SEED, 2)
    parts_h = [read_delta(os.path.join(out_dir, "sparse_adapters", "hash_N2", f"shard_{a}"))
               for a in authors2]
    worst = 0.0
    for slot in got_h:
        ref = parts_h[0][slot] + parts_h[1][slot]  # weight 1.0 each
        worst = max(worst, max_rel_err(got_h[slot], ref))
        rows0 = (parts_h[0][slot].abs().sum(dim=1) > 0)
        rows1 = (parts_h[1][slot].abs().sum(dim=1) > 0)
        assert not (rows0 & rows1).any(), f"row supports overlap in {slot}"
    print(f"ok  composed hash N=2 == disjoint 1.0-weight sum (max rel err {worst:.2e})")

    # dare0p9sum N=4: weight-1.0 sum of the SAME sparse adapters as dare0p9 (shared dir);
    # composed delta == N x the mean-composed twin's delta, and NO dare0p9sum adapter dir
    # was created (canonical-op reuse).
    s_dir = os.path.join(out_dir, "merges", "sparse_dare0p9sum_N4_s42")
    got_s = read_delta(s_dir)
    worst = 0.0
    for slot in got_s:
        ref = sum(p[slot] for p in parts)  # weight 1.0 each, same sparse parts as dare0p9
        worst = max(worst, max_rel_err(got_s[slot], ref))
        worst = max(worst, max_rel_err(got_s[slot], got[slot] * len(parts)))
    assert worst < 1e-4, worst
    assert not os.path.isdir(os.path.join(out_dir, "sparse_adapters", "dare0p9sum")), \
        "sum variant must reuse the canonical dare0p9 adapter dir, not duplicate it"
    print(f"ok  composed dare0p9sum N=4 == 1.0-weight sum == N x mean twin, shared "
          f"adapter dirs (max rel err {worst:.2e})")


def test_determinism(tmp, pool_dir, out_a):
    out_b = os.path.join(tmp, "out_b")
    cfg_b, _ = write_config(tmp, pool_dir, out_b, "cfg_b.json")
    sp.main(["--config", cfg_b, "--limit_authors", str(N_POOL)])
    n_cmp = 0
    for root, _, files in os.walk(out_a):
        for fn in files:
            if fn not in ("adapter_model.safetensors", "adapter_config.json"):
                continue  # merge_meta.json embeds absolute out-dir paths — excluded
            rel = os.path.relpath(os.path.join(root, fn), out_a)
            twin = os.path.join(out_b, rel)
            assert os.path.exists(twin), f"missing {rel} in rerun"
            assert sha256_file(os.path.join(root, fn)) == sha256_file(twin), rel
            n_cmp += 1
    assert n_cmp >= 2 * (8 + N_POOL * 3 + 2 + 4), n_cmp  # merges + shared + hash adapters
    print(f"ok  byte-wise determinism across independent grid runs ({n_cmp} files)")


def test_manifest(pool_dir, out_dir):
    path = os.path.join(out_dir, "eval_manifest_sparse.txt")
    with open(path) as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    assert all(len(r) == 4 and r[3] == "-" for r in rows), "not 4-col nmerge format"
    probes = probe_authors(SEED, N_POOL, 5)
    iso = [r for r in rows if r[0].startswith("iso_a")]
    assert [r[0] for r in iso] == [f"iso_a{a}" for a in probes]
    for r in iso:
        assert r[1] == os.path.join(pool_dir, f"shard_{r[2]}") and os.path.isdir(r[1]), r
    by_label = {}
    for r in rows:
        if not r[0].startswith("iso_a"):
            by_label.setdefault(r[0], []).append(r)
    assert len(by_label) == 10, sorted(by_label)  # 5 ops (incl. dare0p9sum) x N in {2,4}
    for label, lrows in by_label.items():
        op, n = label.split("_")[1], int(label.split("_N")[1].split("_")[0])
        sp.parse_op(op)
        assert [int(r[2]) for r in lrows] == probe_authors(SEED, n, 5), label
        assert int(lrows[0][2]) == probes[0], "headline (first) row must be the perm[0] probe"
        assert all(r[1] == os.path.join(out_dir, "merges", label) for r in lrows)
        assert os.path.isdir(lrows[0][1]), lrows[0][1]
    assert len(rows) == len(iso) + 5 * (2 + 4)  # 5 ops x (2 probes @N2 + 4 @N4)
    # KS reference copied next to the results the eval array will write
    assert os.path.exists(os.path.join(out_dir, "results", "smoke", "retain_tr_scores.npy"))
    print(f"ok  manifest: {len(rows)} rows parse, dirs exist, probe sids + headline order")


def test_dry_run(tmp):
    out_dir = os.path.join(tmp, "dry_out")
    cfg_path, _ = write_config(tmp, os.path.join(tmp, "no_such_pool"), out_dir, "cfg_dry.json")
    sp.main(["--config", cfg_path, "--dry_run"])
    assert not os.path.exists(out_dir), "--dry_run created the out_dir"
    print("ok  --dry_run plans without touching any pool or output path")


# ---------------------------------------------------------------------------
# 5. DX1
# ---------------------------------------------------------------------------

def test_dx1_math():
    shape = (3, 4)
    D = torch.randn(shape, generator=torch.Generator().manual_seed(11))
    D[D.abs() < 0.05] += 0.1  # keep every coordinate clearly nonzero

    opp = sp.Dx1Accumulator(shape, SEED, "model.layers.0.self_attn.q_proj", null_draws=3)
    opp.add(82, D)
    opp.add(15, -D)
    s = opp.summary()
    assert s["observed"]["coord_mean"] < 1e-6 and s["observed"]["l1"] < 1e-6, s["observed"]
    for v in s["null"]["l1_per_draw"] + s["null"]["coord_mean_per_draw"]:
        assert 0.0 <= v <= 1.0 + 1e-6, v
    assert s["null"]["l1_mean"] > s["observed"]["l1"], "null must beat total cancellation"
    assert s["observed_le_null_l1"] and s["observed_le_null_coord_mean"]

    same = sp.Dx1Accumulator(shape, SEED, "model.layers.0.self_attn.q_proj", null_draws=3)
    same.add(82, D)
    same.add(15, D)
    s2 = same.summary()
    assert abs(s2["observed"]["coord_mean"] - 1.0) < 1e-6, s2["observed"]
    assert abs(s2["observed"]["l1"] - 1.0) < 1e-6

    rep = sp.Dx1Accumulator(shape, SEED, "model.layers.0.self_attn.q_proj", null_draws=3)
    rep.add(82, D)
    rep.add(15, -D)
    assert rep.stats() == opp.stats(), "DX1 not deterministic"
    print("ok  DX1 math: opposed pair -> 0, identical pair -> 1, null in [0,1] and above "
          "opposed observed, deterministic")


def test_dx1_run(tmp, cfg):
    out = os.path.join(tmp, "dx1.json")
    res = sp.run_dx1(cfg, n_values=[2, 4], null_draws=3, out_path=out)
    with open(out) as f:
        assert json.load(f) == res
    assert set(res["per_n"]) == {"2", "4"} and res["n_slots"] == len(SLOTS)
    for n in ("2", "4"):
        block = res["per_n"][n]
        assert set(block["per_slot_type"]) == {"q_proj", "up_proj"}
        for stats in [block["overall"]] + list(block["per_slot_type"].values()):
            assert 0.0 <= stats["observed"]["coord_mean"] <= 1.0
            assert 0.0 <= stats["observed"]["l1"] <= 1.0
            assert len(stats["null"]["l1_per_draw"]) == 3
            assert isinstance(stats["observed_le_null_l1"], bool)
    print("ok  run_dx1 on the tiny pool: JSON shape, bounds, per-slot-type + overall")


# ---------------------------------------------------------------------------
# 6. DX2
# ---------------------------------------------------------------------------

def test_dx2_analytic():
    # uniform rows: B = ones -> every delta row identical -> fraction = |rows|/d_out exactly
    B = torch.ones(8, 2)
    A = torch.randn(2, 5, generator=torch.Generator().manual_seed(13))
    frac = sp.dx2_energy_fraction(B, A, [0, 3])
    assert abs(frac - 2 / 8) < 1e-6, frac
    # planted block: B supported only on rows {2,5} -> own block 1.0, foreign block 0.0
    B2 = torch.zeros(8, 2)
    B2[2], B2[5] = 1.0, -2.0
    assert abs(sp.dx2_energy_fraction(B2, A, [2, 5]) - 1.0) < 1e-6
    assert sp.dx2_energy_fraction(B2, A, [0, 1]) < 1e-12
    print("ok  DX2 == analytic fractions on planted matrices (Gram trick, never dense)")


def test_dx2_run(tmp, cfg):
    out = os.path.join(tmp, "dx2.json")
    res = sp.run_dx2(cfg, n_values=[2, 4], out_path=out)
    assert [row["n"] for row in res["per_n"]] == [2, 4]
    for row in res["per_n"]:
        assert abs(row["expected_fraction"] - 1.0 / row["n"]) < 1e-12
        assert 0.0 <= row["frac_min"] <= row["frac_mean"] <= row["frac_max"] <= 1.0
        assert sorted(map(int, row["per_author"])) == sorted(subset_authors(SEED, row["n"]))
        assert set(row["per_slot_type"]) == {"q_proj", "up_proj"}
    res2 = sp.run_dx2(cfg, n_values=[2, 4], out_path=out)
    assert res == res2, "DX2 not deterministic"
    print("ok  run_dx2 on the tiny pool: bounds, expected 1/N recorded, deterministic")


# ---------------------------------------------------------------------------

def main():
    tmp = tempfile.mkdtemp(prefix="test_sparsify_pool_")
    try:
        pool_dir = os.path.join(tmp, "pool")
        authors = build_pool(pool_dir)
        assert authors == subset_authors(SEED, N_POOL)
        out_a = os.path.join(tmp, "out_a")
        cfg_path, cfg = write_config(tmp, pool_dir, out_a, "cfg_a.json")

        test_op_parsing()
        test_dare()
        test_topk()
        test_hash_blocks()
        test_grid_composition(tmp, pool_dir, out_a, cfg_path)
        test_determinism(tmp, pool_dir, out_a)
        test_manifest(pool_dir, out_a)
        test_dry_run(tmp)
        test_dx1_math()
        test_dx1_run(tmp, cfg)
        test_dx2_analytic()
        test_dx2_run(tmp, cfg)
        print("ALL OK  test_sparsify_pool")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
