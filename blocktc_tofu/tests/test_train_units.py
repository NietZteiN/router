"""Trainer-unit CPU gates: per-block clip groups (DESIGN §9 gate 14), the
alternating single-source sampler, the closed config schema (incl. every
shipped config file), subset resolution, and detector init (determinism +
ADDITIVE application + slot scatter).

All CPU/fp32, pinned seeds. References are literal loops / hand math —
deliberately NOT reusing the vectorized code under test, so the two
implementations cross-check (sepmlp test_spec_losses precedent).
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from conftest import (
    AUTHOR_IDS,
    HIDDEN,
    INSERT_LAYER,
    K,
    M_AUTHOR,
    M_SHARED,
    S,
    SEED,
    build_tc,
    feat_rows,
    make_batch,
    slot_of,
    wrap_tiny,
)

from tc_common import NO_AUTHOR, load_config  # noqa: E402
from tc_model import compute_detector_init  # noqa: E402
from train_tc import (  # noqa: E402
    AlternatingBatchSampler,
    CONFIG_KEYS,
    expand_detector_arrays,
    per_block_clip_,
    resolve_subset,
    validate_config,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- gate 14: per-block clip groups ------------------------------------------

def test_per_block_clip_isolation():
    """One block = one clip group (encoder rows + bias entries + decoder
    columns ACROSS ALL span decoders); the shared block is group K. A spiking
    author cannot rescale anyone else; untouched authors stay exactly zero;
    below-norm groups are bitwise untouched."""
    tc = build_tc()
    spike, calm = slot_of(42), slot_of(7)
    for name in ("W_enc", "b_enc", "W_dec"):
        p = getattr(tc, name)
        p.grad = torch.zeros_like(p)
    g = torch.Generator().manual_seed(SEED)
    rs, rc = feat_rows(spike), feat_rows(calm)
    tc.W_enc.grad[rs] = torch.randn(M_AUTHOR, HIDDEN, generator=g) * 10
    tc.b_enc.grad[rs] = torch.randn(M_AUTHOR, generator=g) * 10
    tc.W_dec.grad[:, :, rc] = torch.randn(tc.span, HIDDEN, M_AUTHOR,
                                          generator=g) * 1e-3
    tc.W_enc.grad[S:] = torch.randn(M_SHARED, HIDDEN, generator=g) * 10

    calm_before = tc.W_dec.grad[:, :, rc].clone()
    norms = per_block_clip_(tc, max_norm=1.0)
    assert norms.shape == (K + 1,)  # K author groups + the shared group
    assert norms[spike] > 1.0 and norms[calm] < 1.0 and norms[K] > 1.0

    # calm author's grads bitwise untouched (norm < clip => coef 1)
    assert torch.equal(tc.W_dec.grad[:, :, rc], calm_before)
    # spiking author rescaled to the clip norm across enc rows + bias
    post = (tc.W_enc.grad[rs].pow(2).sum()
            + tc.b_enc.grad[rs].pow(2).sum()).sqrt()
    assert float(post) == pytest.approx(1.0, rel=1e-4)
    # the shared group clipped INDEPENDENTLY (its own norm, its own coef)
    post_shared = tc.W_enc.grad[S:].pow(2).sum().sqrt()
    assert float(post_shared) == pytest.approx(1.0, rel=1e-4)
    # untouched (all-zero) authors stay exactly zero
    other = [k for k in range(K) if k not in (spike, calm)][0]
    assert tc.W_enc.grad[feat_rows(other)].abs().sum() == 0
    assert tc.W_dec.grad[..., feat_rows(other)].abs().sum() == 0


def test_per_block_clip_groups_decoder_columns_across_span():
    """Gate-14 coverage half: a block's group includes its decoder COLUMNS of
    EVERY write layer — grad living only in W_dec[span-1] still counts toward
    (and is rescaled by) that block's norm, and nobody else's."""
    tc = build_tc()
    for name in ("W_enc", "b_enc", "W_dec"):
        p = getattr(tc, name)
        p.grad = torch.zeros_like(p)
    rows = feat_rows(slot_of(11))
    g = torch.Generator().manual_seed(SEED + 1)
    tc.W_dec.grad[tc.span - 1, :, rows] = torch.randn(HIDDEN, M_AUTHOR,
                                                      generator=g) * 5
    want = float(tc.W_dec.grad[tc.span - 1, :, rows].pow(2).sum().sqrt())
    norms = per_block_clip_(tc, max_norm=1.0)
    assert float(norms[slot_of(11)]) == pytest.approx(want, rel=1e-5)
    assert all(float(norms[k]) == 0.0
               for k in range(K + 1) if k != slot_of(11))
    post = float(tc.W_dec.grad[tc.span - 1, :, rows].pow(2).sum().sqrt())
    assert post == pytest.approx(1.0, rel=1e-4)


def test_per_block_clip_partition_covers_every_entry():
    """The K+1 groups PARTITION all three tensors: uniform grads with a
    per-group-identifiable magnitude come back rescaled per group with no
    entry missed and no entry double-scaled (norms recomputed post-clip are
    exactly the max_norm for every over-norm group)."""
    tc = build_tc()
    for name in ("W_enc", "b_enc", "W_dec"):
        p = getattr(tc, name)
        p.grad = torch.ones_like(p) * 3.0
    per_block_clip_(tc, max_norm=1.0)
    # recompute per-group norms independently: every group must now be 1.0
    for k in range(K + 1):
        rows = feat_rows(k) if k < K else torch.arange(S, S + M_SHARED)
        total = (tc.W_enc.grad[rows].pow(2).sum()
                 + tc.b_enc.grad[rows].pow(2).sum()
                 + tc.W_dec.grad[..., rows].pow(2).sum())
        assert float(total.sqrt()) == pytest.approx(1.0, rel=1e-4), k


# -- alternating single-source sampler ---------------------------------------

def test_alternating_sampler_structure_and_determinism():
    n_authors, rpp, n_generic, bs = 3, 20, 15, 8
    groups = [list(range(j * rpp, (j + 1) * rpp)) for j in range(n_authors)]
    offset = n_authors * rpp
    s = AlternatingBatchSampler(groups, n_generic, bs, seed=SEED)
    n_author_batches = n_authors * ((rpp + bs - 1) // bs)
    assert len(s) == 2 * n_author_batches

    epoch0 = list(iter(s))
    assert len(epoch0) == 2 * n_author_batches
    author_seen = []
    for i, batch in enumerate(epoch0):
        if i % 2 == 0:  # author batch: SINGLE-SOURCE by construction
            srcs = {idx // rpp for idx in batch}
            assert len(srcs) == 1, srcs
            assert all(idx < offset for idx in batch)
            author_seen += batch
        else:           # generic batch: offset ConcatDataset indices, full bs
            assert all(offset <= idx < offset + n_generic for idx in batch)
            assert len(batch) == bs
    assert sorted(author_seen) == list(range(offset))  # full pass, no dups

    # round-robin: each cycle of n_authors author batches visits n_authors
    # DISTINCT authors (no author starved to the tail of an epoch; equal
    # chunk counts here, so every cycle is complete)
    author_batches = epoch0[0::2]
    for c in range(0, len(author_batches), n_authors):
        cycle = author_batches[c:c + n_authors]
        assert len({b[0] // rpp for b in cycle}) == len(cycle)

    # determinism: same (seed, epoch counter) => identical schedule ...
    assert list(iter(AlternatingBatchSampler(groups, n_generic, bs, SEED))) \
        == epoch0
    # ... and the next epoch reshuffles
    assert list(iter(s)) != epoch0


def test_alternating_sampler_rejects_broken_partition():
    with pytest.raises(AssertionError, match="partition"):
        AlternatingBatchSampler([[0, 1], [3]], 5, 2, SEED)  # gap at 2
    with pytest.raises(AssertionError, match="partition"):
        AlternatingBatchSampler([[0, 1], [1, 2]], 5, 2, SEED)  # overlap
    with pytest.raises(AssertionError):
        AlternatingBatchSampler([[0, 1], []], 5, 2, SEED)  # empty group


# -- config schema (closed set) ----------------------------------------------

def _valid_cfg(**over):
    cfg = {
        "model": "meta-llama/Llama-3.2-1B-Instruct", "insert_layer": 9,
        "span": 3, "m_author": 32, "m_shared": 128, "n_authors": 200,
        "authors_subset": None, "seed": 42, "max_length": 512,
        "batch_size": 32, "grad_accum": 1, "epochs": 15, "lr": 1e-3,
        "lambda_max": 0.1, "lambda_warmup_frac": 0.15, "clip_norm": 1.0,
        "detector_init": "questions", "init_scale": 1.0, "alpaca_n": 2000,
        "phase": "phase0", "phase0_checkpoint": None, "run_name": "t",
    }
    cfg.update(over)
    return cfg


def test_validate_config_closed_schema():
    validate_config(_valid_cfg())
    validate_config(_valid_cfg(_note="underscore keys are notes"))  # ignored
    with pytest.raises(AssertionError, match="schema drift"):
        validate_config(_valid_cfg(typo_key=1.0))          # unknown key
    incomplete = _valid_cfg()
    incomplete.pop("clip_norm")
    with pytest.raises(AssertionError, match="schema drift"):
        validate_config(incomplete)                        # missing key
    with pytest.raises(AssertionError):
        validate_config(_valid_cfg(phase="phase1"))        # needs checkpoint
    with pytest.raises(AssertionError):
        validate_config(_valid_cfg(phase0_checkpoint="x"))  # phase0 takes none
    with pytest.raises(AssertionError):
        validate_config(_valid_cfg(authors_subset=[20, 10]))
    with pytest.raises(AssertionError):
        validate_config(_valid_cfg(authors_subset=[0, 201]))
    with pytest.raises(AssertionError):
        validate_config(_valid_cfg(detector_init="typo"))
    validate_config(_valid_cfg(phase="phase1", phase0_checkpoint="p.pt",
                               authors_subset=[0, 20]))


def test_every_shipped_config_validates():
    """The six pilot arms + smoke + phase0 + k200 all pass the closed schema
    (a typo'd hyperparameter in a run config would otherwise surface only at
    job time) and agree on the pinned topology."""
    paths = sorted(glob.glob(os.path.join(PROJECT_DIR, "configs", "*.json")))
    assert len(paths) == 9, paths
    for path in paths:
        cfg = load_config(path)
        cfg.pop("_config_path")
        validate_config(cfg)
        assert (cfg["insert_layer"], cfg["span"]) == (9, 3), path
        assert (cfg["m_author"], cfg["m_shared"],
                cfg["n_authors"]) == (32, 128, 200), path
        assert cfg["seed"] == 42, path
        if cfg["phase"] == "phase1":
            assert cfg["phase0_checkpoint"].endswith("blocktc.pt"), path


def test_resolve_subset():
    assert resolve_subset(_valid_cfg()) == list(range(200))
    assert resolve_subset(_valid_cfg(authors_subset=[0, 20])) == \
        list(range(20))
    assert resolve_subset(_valid_cfg(authors_subset=[5, 8])) == [5, 6, 7]


def test_config_keys_match_design_schema():
    # DESIGN §7 enumerates the schema; the code's closed set must be exactly
    # that (a drift here silently re-opens the schema).
    assert CONFIG_KEYS == {
        "model", "insert_layer", "span", "m_author", "m_shared", "n_authors",
        "authors_subset", "seed", "max_length", "batch_size", "grad_accum",
        "epochs", "lr", "lambda_max", "lambda_warmup_frac", "clip_norm",
        "detector_init", "init_scale", "alpaca_n", "phase",
        "phase0_checkpoint", "run_name",
    }


# -- detector init: determinism, correctness, additive application -----------

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


def test_compute_detector_init_deterministic_and_correct(tiny_model):
    m1, c1 = compute_detector_init(tiny_model, INSERT_LAYER, _init_batches(),
                                   AUTHOR_IDS, "cpu")
    m2, c2 = compute_detector_init(tiny_model, INSERT_LAYER, _init_batches(),
                                   AUTHOR_IDS, "cpu")
    assert m1.shape == (K, HIDDEN)
    assert np.array_equal(m1, m2) and np.array_equal(c1, c2)  # bit-equal

    # counts: author 3 (2 rows/batch) and 11 (1 row/batch) x 4 question
    # tokens x 2 batches; the NO_AUTHOR row contributes to nobody
    assert c1[slot_of(3)] == 2 * 4 * 2 and c1[slot_of(11)] == 1 * 4 * 2
    assert c1[slot_of(7)] == 0 and c1[slot_of(42)] == 0

    # correctness: independent hook capture of the RAW mlp input for author 3
    captured = {}
    h = tiny_model.model.layers[INSERT_LAYER].mlp.register_forward_pre_hook(
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
            for r in (b["source_ids"] == 3).nonzero(as_tuple=True)[0].tolist():
                sums += (x[r] * qm[r].unsqueeze(-1)).sum(0)
                n += float(qm[r].sum())
    h.remove()
    ref = (sums / n).float().numpy()
    np.testing.assert_allclose(m1[slot_of(3)], ref, rtol=1e-5, atol=1e-7)


def test_compute_detector_init_refuses_wrapped_mlp(tiny_model):
    tc = build_tc()
    wrap_tiny(tiny_model, tc)
    with pytest.raises(AssertionError, match="before install_tc"):
        compute_detector_init(tiny_model, INSERT_LAYER, _init_batches(),
                              AUTHOR_IDS, "cpu")


def test_apply_detector_init_is_additive_on_seeded_rows():
    """ADDITIVE on top of the seeded random rows (the anti-width-collapse
    divergence from sepmlp's pure copy): row_i += scale * unit(mean[k]);
    zero-count blocks keep their seeded rows bitwise; b_enc stays 0."""
    from tc_model import apply_detector_init

    tc = build_tc()
    before = tc.W_enc.detach().clone()
    g = torch.Generator().manual_seed(SEED)
    mean_hidden = torch.randn(K, HIDDEN, generator=g).numpy() \
        .astype(np.float32)
    counts = np.array([4.0, 0.0, 2.0, 8.0, 1.0])  # author 7 (slot 1): none
    apply_detector_init(tc, mean_hidden, counts, init_scale=1.5)
    wa = tc.W_enc.detach()[:S].view(K, M_AUTHOR, HIDDEN)
    wb = before[:S].view(K, M_AUTHOR, HIDDEN)
    for k in range(K):
        if counts[k] == 0:
            assert torch.equal(wa[k], wb[k])
            continue
        mh = torch.from_numpy(mean_hidden[k])
        want = wb[k] + 1.5 * mh / mh.norm()
        assert torch.allclose(wa[k], want, rtol=1e-6, atol=1e-7), k
        # the seeded rows tie-break survives: the m rows stay DISTINCT
        assert not torch.equal(wa[k][0], wa[k][1])
    assert torch.equal(tc.W_enc.detach()[S:], before[S:])  # shared untouched
    assert tc.b_enc.detach().abs().sum() == 0              # bias stays 0


def test_expand_detector_arrays_slot_scatter():
    """Subset arrays scatter by SLOT (author_ids position), never global id —
    the tiny fixture's non-contiguous ids catch positional indexing."""
    tc = build_tc()
    subset = [7, 42]  # slots 1 and 4
    mean = np.stack([np.full(HIDDEN, 1.0, dtype=np.float32),
                     np.full(HIDDEN, 2.0, dtype=np.float32)])
    counts = np.array([12.0, 34.0])
    mean_full, counts_full = expand_detector_arrays(tc, subset, mean, counts)
    assert mean_full.shape == (K, HIDDEN) and counts_full.shape == (K,)
    assert (mean_full[slot_of(7)] == 1.0).all()
    assert (mean_full[slot_of(42)] == 2.0).all()
    assert counts_full[slot_of(7)] == 12.0 and counts_full[slot_of(42)] == 34.0
    for k in range(K):
        if k not in (slot_of(7), slot_of(42)):
            assert counts_full[k] == 0.0 and (mean_full[k] == 0.0).all()
    with pytest.raises(AssertionError, match="no block"):
        expand_detector_arrays(tc, [5], mean[:1], counts[:1])
