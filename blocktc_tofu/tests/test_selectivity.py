"""Selectivity/leak-probe CPU gates: measure_selectivity's aggregation ≡
dense recomputation (the sepmlp test_selectivity + test_leak_npz analogs).

On the tiny transcoder (no HF model, no datasets):
  (a) begin_row_stats per-ROW act-mass capture against a literal dense
      per-block computation, including the mask pooling — on the PLAIN
      serving forward (the condition the leak probe measures);
  (b) aggregate_author_mass / leakage_summary / pooled_mass against
      hand-computed matrices (count-weighting, shared column LAST, off_max
      excluding diagonal AND shared);
  (c) gate_verdict: exact wiring of the pre-registered LAZY<2 / SELECTIVE>=5
      thresholds (LoRA anchor 1.11 = LAZY);
  (d) assemble_leak_arrays: exact NPZ-contract keys/dtypes/shapes, own-norm
      NaN semantics, foreign vs surviving maxima, the blocktc shared_norm
      extension, droplist-then-probe ordering;
  (e) answer_probability ≡ the OU evaluate_probability formula, hand-looped;
  (f) own_only mask semantics (shared stays live; prior mask restored).
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import (
    FDIM,
    K,
    S,
    SEED,
    feat_rows,
    make_x,
    module_forward,
    slot_of,
    trained_tc,
)

from measure_selectivity import (  # noqa: E402
    LAZY_LT,
    LEAK_GROUPS,
    SELECTIVE_GE,
    aggregate_author_mass,
    answer_probability,
    assemble_leak_arrays,
    gate_verdict,
    leakage_summary,
    own_only,
    pooled_mass,
)
from tc_layer import TcState  # noqa: E402
from tc_model import (  # noqa: E402
    apply_droplist_file,
    compute_tc_sha,
)

NPZ_KEYS = {"max_surv_norm", "max_foreign_norm", "top_surv_author",
            "own_norm", "shared_norm", "group", "author_of_q", "n_surviving",
            "droplist_tag", "K"}


# -- (a) row-stats capture == dense ------------------------------------------

def capture_rows(tc, x, row_mask):
    """One begin_row_stats capture on the plain serving forward (eval mode,
    NO source ids — exactly run_mass_capture's condition)."""
    state = TcState()
    state.begin_row_stats(row_mask)
    tc.eval()
    with torch.no_grad():
        module_forward(tc, x, state)
    stats = state.end_row_stats()
    state.clear()
    return stats


def dense_row_mass(tc, x, row_mask):
    """Literal reference: per block (shared LAST), per row, SUM over masked
    tokens of the block's act mass (sum of its non-negative activations)."""
    B = x.shape[0]
    Ks = tc.num_authors
    with torch.no_grad():
        a = F.relu(x @ tc.W_enc.t() + tc.b_enc)
    sums = np.zeros((B, Ks + 1))
    blocks = [torch.arange(k * tc.m_author, (k + 1) * tc.m_author)
              for k in range(Ks)] + [torch.arange(tc.shared_start,
                                                  tc.n_features)]
    for j, rows in enumerate(blocks):
        per_tok = a[..., rows].sum(-1)          # (B, T)
        for b in range(B):
            m = row_mask[b].bool()
            sums[b, j] = float(per_tok[b][m].sum()) if m.any() else 0.0
    return sums


def test_row_stats_capture_matches_dense():
    tc = trained_tc()
    B, T = 4, 7
    x = make_x(B, T, seed=SEED + 30)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 2:] = True         # ragged answer spans
    mask[1, 4:] = True
    mask[2, :] = True
    mask[3, -1] = True
    stats = capture_rows(tc, x, mask)
    assert len(stats) == 1     # ONE read site => exactly one entry
    s = stats[0]
    assert s["sum"].shape == (B, K + 1)  # shared column LAST
    assert torch.equal(s["cnt"], mask.sum(dim=1).to(s["cnt"].dtype))
    np.testing.assert_allclose(s["sum"].numpy(), dense_row_mass(tc, x, mask),
                               rtol=1e-4, atol=1e-6)


# -- (b) aggregation == hand math --------------------------------------------

def test_aggregate_author_mass_count_weighted():
    # every RESIDENT author must have probe rows (the builder guarantees it;
    # the aggregator hard-asserts it) — resident here is [3, 7], K=2
    resident = [3, 7]
    row_sums = np.array([[1.0, 0.5, 2.0],   # author 3, 4 toks
                         [3.0, 0.5, 2.0],   # author 3, 2 toks
                         [0.0, 4.0, 6.0]])  # author 7, 6 toks (shared LAST)
    row_cnts = np.array([4.0, 2.0, 6.0])
    row_authors = [3, 3, 7]
    sums, tok = aggregate_author_mass(row_sums, row_cnts, row_authors,
                                      resident)
    assert sums.shape == (2, 3) and tok.shape == (2,)
    # count-weighted: author 3's row = SUM of its rows' sums, tok = 6 —
    # A[k] = sums/tok is a per-TOKEN mean, not a row-mean-of-row-means
    np.testing.assert_array_equal(sums[0], [4.0, 1.0, 4.0])
    assert tok[0] == 6.0
    np.testing.assert_array_equal(sums[1], row_sums[2])
    # a resident author with zero captured rows is a hard error, never a NaN
    with pytest.raises(AssertionError, match="zero question tokens"):
        aggregate_author_mass(
            np.array([[1.0, 0.5, 0.0, 2.0]]), np.array([4.0]), [3],
            [3, 7, 11])


def test_leakage_summary_hand_matrix():
    # 3 resident blocks + shared, hand-picked so every channel is distinct
    sums = np.array([[8.0, 1.0, 2.0, 5.0],
                     [0.5, 6.0, 1.5, 3.0],
                     [1.0, 4.0, 9.0, 7.0]])
    tok = np.array([2.0, 1.0, 4.0])
    out = leakage_summary(sums, tok)
    A = sums / tok[:, None]
    np.testing.assert_allclose(out["A"], A)
    np.testing.assert_allclose(out["on"], np.diagonal(A[:, :3]))
    # off[k] = column aggregate: block k's mass on OTHER authors' tokens,
    # weighted by their token counts
    for k in range(3):
        num = sums[:, k].sum() - sums[k, k]
        den = tok.sum() - tok[k]
        assert out["off"][k] == pytest.approx(num / den)
    np.testing.assert_allclose(out["ratio"],
                               out["on"] / np.clip(out["off"], 1e-12, None))
    # off_max: worst FOREIGN AUTHOR block per row — diagonal and the shared
    # column both EXCLUDED (DESIGN §8)
    assert out["off_max"][0] == pytest.approx(max(A[0, 1], A[0, 2]))
    assert out["top_off_slot"][0] == (1 if A[0, 1] > A[0, 2] else 2)
    np.testing.assert_allclose(out["shared"], A[:, 3])
    # K=1 edge: no foreign blocks
    single = leakage_summary(np.array([[4.0, 1.0]]), np.array([2.0]))
    assert single["off_max"][0] == 0.0 and single["top_off_slot"][0] == -1


def test_pooled_mass():
    sums = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    cnts = np.array([3.0, 1.0])
    np.testing.assert_allclose(pooled_mass(sums, cnts),
                               np.array([4.0, 4.0, 4.0]) / 4.0)
    # zero tokens: divide-by-max(1) guard, not NaN
    assert not np.isnan(pooled_mass(np.zeros((0, 3)), np.zeros(0))).any()


# -- (c) gate thresholds -----------------------------------------------------

def test_gate_verdict_thresholds():
    lazy = gate_verdict([1.0, 1.11, 1.9])   # 1.11 = the LoRA anchor
    assert lazy["gate_verdict"] == "LAZY"
    assert lazy["frac_ratio_lt_2"] == 1.0
    assert lazy["frac_ratio_ge_5"] == 0.0

    selective = gate_verdict([5.0, 8.0, 50.0])
    assert selective["gate_verdict"] == "SELECTIVE"
    assert selective["gate_median"] == 8.0
    assert selective["frac_ratio_ge_5"] == 1.0

    mid = gate_verdict([1.0, 3.0, 6.0])
    assert mid["gate_verdict"] == "INTERMEDIATE"
    assert mid["frac_ratio_lt_2"] == pytest.approx(1 / 3)
    assert mid["frac_ratio_ge_5"] == pytest.approx(1 / 3)

    # Boundary semantics: LAZY is strict <2, SELECTIVE is >=5.
    assert gate_verdict([LAZY_LT] * 3)["gate_verdict"] == "INTERMEDIATE"
    assert gate_verdict([SELECTIVE_GE] * 3)["gate_verdict"] == "SELECTIVE"
    assert gate_verdict([LAZY_LT - 1e-9] * 3)["gate_verdict"] == "LAZY"
    thresholds = gate_verdict([1.0])["gate_thresholds"]
    assert thresholds == {"lazy_lt": 2.0, "selective_ge": 5.0}


# -- (d) leak NPZ contract ---------------------------------------------------

def test_assemble_leak_arrays_contract():
    rng = np.random.RandomState(SEED)
    resident = [3, 11, 42]                        # 7 and 19 "dropped"
    n_q = 6
    m = rng.rand(n_q, len(resident) + 1) * 1e-3   # + the shared column (LAST)
    groups = ["forget_orig", "forget_para", "retain", "ood_world_facts",
              "ood_real_authors", "ood_alpaca"]
    #          authors: 7 dropped, 7 dropped, 11 resident, then OOD
    authors = [7, 7, 11, -1, -1, -1]
    arrays = assemble_leak_arrays(m, resident, authors, groups,
                                  droplist_tag="forget10", total_authors=K)

    assert set(arrays) == NPZ_KEYS
    for key in ("max_surv_norm", "max_foreign_norm", "own_norm",
                "shared_norm"):
        assert arrays[key].dtype == np.float32, key
    assert arrays["top_surv_author"].dtype == np.int32
    assert arrays["author_of_q"].dtype == np.int32
    assert arrays["group"].dtype.kind == "U"
    for key in NPZ_KEYS - {"n_surviving", "droplist_tag", "K"}:
        assert arrays[key].shape == (n_q,), key
    assert arrays["n_surviving"] == len(resident)
    assert arrays["droplist_tag"] == "forget10"
    assert arrays["K"] == K
    assert set(arrays["group"]) <= set(LEAK_GROUPS)

    author_cols = m[:, :len(resident)]
    # maxima are over AUTHOR columns only — shared EXCLUDED, reported apart
    np.testing.assert_allclose(arrays["max_surv_norm"],
                               author_cols.max(axis=1), rtol=1e-6)
    np.testing.assert_allclose(arrays["shared_norm"], m[:, -1], rtol=1e-6)
    # rows without a surviving own block (orphans + OOD): own NaN and
    # foreign == surviving max
    for i in (0, 1, 3, 4, 5):
        assert np.isnan(arrays["own_norm"][i])
        assert arrays["max_foreign_norm"][i] == arrays["max_surv_norm"][i]
    # the retain row: own is its author's column, foreign excludes it
    j = resident.index(11)
    assert arrays["own_norm"][2] == np.float32(author_cols[2, j])
    others = np.delete(author_cols[2], j)
    assert arrays["max_foreign_norm"][2] == pytest.approx(others.max(),
                                                          rel=1e-6)
    # top_surv_author is a GLOBAL author id, never a slot index
    assert set(arrays["top_surv_author"]) <= set(resident)
    assert arrays["top_surv_author"][0] == \
        resident[int(np.argmax(author_cols[0]))]

    # npz round-trip preserves the whole contract
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    z = np.load(buf)
    assert set(z.files) == NPZ_KEYS
    assert str(z["droplist_tag"]) == "forget10"
    assert int(z["n_surviving"]) == len(resident)
    np.testing.assert_array_equal(z["group"], arrays["group"])


def test_assemble_leak_arrays_edges():
    # no surviving blocks (all200 droplist): silence (0.0), top -1, own NaN —
    # but the shared column still reports (it is never deletable)
    arrays = assemble_leak_arrays(np.array([[0.7], [0.3]]), [], [3, -1],
                                  ["forget_orig", "ood_alpaca"],
                                  droplist_tag="all", total_authors=K)
    assert (arrays["max_surv_norm"] == 0).all()
    assert (arrays["top_surv_author"] == -1).all()
    assert np.isnan(arrays["own_norm"]).all()
    np.testing.assert_allclose(arrays["shared_norm"], [0.7, 0.3])
    # single own-only survivor: foreign max over the empty set = 0.0
    arrays = assemble_leak_arrays(np.array([[0.5, 0.1]]), [3], [3],
                                  ["retain"], droplist_tag="none",
                                  total_authors=K)
    assert arrays["own_norm"][0] == np.float32(0.5)
    assert arrays["max_foreign_norm"][0] == 0.0
    # unknown group name must refuse
    with pytest.raises(AssertionError):
        assemble_leak_arrays(np.zeros((1, 2)), [3], [3], ["bogus_group"],
                             droplist_tag="none", total_authors=K)


def test_droplist_then_probe_sees_post_drop_blocks(tmp_path):
    """The capture runs on the POST-drop transcoder: dropped author's column
    gone, own_norm NaN for its queries, surviving columns the SAME functions
    as pre-drop (sha-pinned droplist applied first, sepmlp leak ordering)."""
    from tc_model import load_tc_from_checkpoint, save_checkpoint
    from conftest import ADAPTER_CFG

    tc = trained_tc()
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir, phase="phase1")

    B, T = 3, 5
    x = make_x(B, T, seed=SEED + 31)
    mask = torch.ones(B, T, dtype=torch.bool)
    groups = ["forget_orig", "retain", "ood_alpaca"]
    authors = [7, 11, -1]      # author 7 will be dropped

    # reference run: no droplist — own block of author 7 present
    ref_tc, _, _, _ = load_tc_from_checkpoint(run_dir)
    s_ref = capture_rows(ref_tc, x, mask)[0]
    m_ref = (s_ref["sum"] / s_ref["cnt"].unsqueeze(-1)).numpy()
    ref = assemble_leak_arrays(m_ref, ref_tc.author_ids.tolist(), authors,
                               groups, droplist_tag="none", total_authors=K)
    assert ref["n_surviving"] == K
    assert not np.isnan(ref["own_norm"][0])     # author 7 resident pre-drop

    # drop run: apply the sha-pinned droplist FIRST, then probe
    drop_tc, _, _, _ = load_tc_from_checkpoint(run_dir)
    droplist = tmp_path / "drop.json"
    droplist.write_text(json.dumps({
        "tag": "forget7", "authors": [7],
        "tc_sha": compute_tc_sha(drop_tc),
    }))
    apply_droplist_file(drop_tc, str(droplist))
    resident = drop_tc.author_ids.tolist()
    assert 7 not in resident and len(resident) == K - 1

    s_drop = capture_rows(drop_tc, x, mask)[0]
    assert s_drop["sum"].shape == (B, K)        # K-1 authors + shared
    m_drop = (s_drop["sum"] / s_drop["cnt"].unsqueeze(-1)).numpy()
    dropped = assemble_leak_arrays(m_drop, resident, authors, groups,
                                   droplist_tag="forget7", total_authors=K)
    assert dropped["n_surviving"] == K - 1 and dropped["K"] == K
    assert np.isnan(dropped["own_norm"][0])     # orphan: own block gone
    assert not np.isnan(dropped["own_norm"][1])  # retain: own survives
    assert 7 not in set(dropped["top_surv_author"])
    # orphan semantics: foreign == surviving max once the own block is gone
    assert dropped["max_foreign_norm"][0] == dropped["max_surv_norm"][0]
    # surviving columns (and shared) are the SAME functions before/after
    keep = [j for j, a in enumerate(ref_tc.author_ids.tolist()) if a != 7]
    keep.append(K)  # the shared column rides along
    np.testing.assert_allclose(m_drop, m_ref[:, keep], rtol=1e-6, atol=1e-9)


# -- (e) answer_probability == the OU formula --------------------------------

class _StubLM:
    """Deterministic .logits-only model — enough for answer_probability."""

    def __init__(self, vocab: int = 7):
        self.vocab = vocab

    def __call__(self, input_ids=None, attention_mask=None, use_cache=False):
        B, T = input_ids.shape
        g = torch.Generator().manual_seed(7)  # input-independent, repeatable
        logits = torch.randn(B, T, self.vocab, generator=g)

        class Out:
            pass

        out = Out()
        out.logits = logits
        return out


def test_answer_probability_ou_formula():
    """Inline port ≡ the OU evaluate_probability formula, hand-computed:
    exp(-sum(answer-token CE)/num_token_gt) with num_token_gt counted on the
    UNSHIFTED labels — including ragged answer lengths and -100 padding."""
    model = _StubLM()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 0, 0]])
    labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 3, 4, -100, -100]])
    batch = {"input_ids": input_ids,
             "attention_mask": input_ids.ne(0).long(),
             "labels": labels}
    probs = answer_probability(model, batch, "cpu")

    logits = model(input_ids=input_ids).logits
    logp = torch.log_softmax(logits.float(), dim=-1)
    expected = []
    for b in range(2):
        total = 0.0
        for t in range(4):  # positions predicting token t+1
            lab = int(labels[b, t + 1])
            if lab != -100:
                total -= float(logp[b, t, lab])
        num = int((labels[b] != -100).sum())
        expected.append(float(np.exp(-total / num)))
    np.testing.assert_allclose(probs, expected, rtol=1e-6)


# -- (f) own_only mask semantics ---------------------------------------------

def test_own_only_mask_semantics():
    """own_only serves exactly one author's block PLUS the always-live shared
    tail (the deletion counterfactual) and restores the PRIOR mask (not
    blanket all-True) even when it was already partial."""
    tc = trained_tc().eval()
    with torch.no_grad():
        tc.active.copy_(torch.tensor([True, True, False, True, True]))
    before = tc.active.clone()

    x = make_x(2, 4, seed=SEED + 32)
    with own_only(tc, 7):
        assert tc.active.tolist() == [False, True, False, False, False]
        with torch.no_grad():
            got = module_forward(tc, x)
            # dense reference: block 7's contribution + the shared tail
            a = F.relu(x @ tc.W_enc.t() + tc.b_enc)
            rows = torch.cat([feat_rows(slot_of(7)), torch.arange(S, FDIM)])
            want = torch.stack([a[..., rows] @ tc.W_dec[j][:, rows].t()
                                for j in range(tc.span)])
        assert not torch.equal(got, torch.zeros_like(got))
        np.testing.assert_allclose(got.numpy(), want.numpy(),
                                   rtol=1e-5, atol=1e-6)
    assert torch.equal(tc.active, before)  # prior PARTIAL mask restored

    with pytest.raises(AssertionError):
        with own_only(tc, 99):  # non-resident id must refuse, not no-op
            pass
    assert torch.equal(tc.active, before)
