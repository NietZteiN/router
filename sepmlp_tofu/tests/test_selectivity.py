"""CPU gate 15: selectivity streaming aggregation ≡ dense computation.

Proves, on a tiny synthetic AuthorBank (H=8, D=2, K=3 — no HF downloads, no
real model), that measure_selectivity's streaming pipeline
(BankState telemetry -> NormAccumulator -> summarize_norms /
per_layer_ratio_medians) reproduces a hand-written dense per-author norm
computation, that the own/off classification follows source_ids (incl.
NO_AUTHOR = off for everyone), and that the pre-registered gate thresholds
are wired exactly (LAZY < 2.0 <= INTERMEDIATE < 5.0 <= SELECTIVE).
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bank_layer import AuthorBank, BankState
from measure_selectivity import (
    LAZY_LT,
    SELECTIVE_GE,
    NormAccumulator,
    answer_probability,
    gate_verdict,
    own_only,
    per_layer_ratio_medians,
    summarize_norms,
)
from sepmlp_common import NO_AUTHOR

H, D, K = 8, 2, 3
AUTHOR_IDS = [0, 1, 2]
LAYERS = [0, 3]  # non-contiguous on purpose: layer->row mapping must not
                 # assume 0..L-1


def make_bank(layer_idx: int, seed: int) -> AuthorBank:
    """Tiny bank with hand-set nonzero weights (zero-init W_down would make
    every norm 0, zero b_gate would make the bias path vacuous)."""
    bank = AuthorBank(hidden=H, width=D, author_ids=AUTHOR_IDS,
                      layer_idx=layer_idx, init_seed=42)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        bank.W_gate.copy_(torch.randn(K * D, H, generator=g))
        bank.W_up.copy_(torch.randn(K * D, H, generator=g))
        bank.W_down.copy_(torch.randn(H, K * D, generator=g))
        bank.b_gate.copy_(torch.randn(K * D, generator=g) * 0.5)
    bank.eval()
    return bank


def bank_act(bank: AuthorBank):
    return F.relu if bank.gate_act == "relu" else F.silu


def dense_author_norms(bank: AuthorBank, x: torch.Tensor) -> torch.Tensor:
    """(B, T, K) per-token ||out_a(x)|| via the literal per-author matmul —
    the ground truth the Gram-trick telemetry must reproduce."""
    with torch.no_grad():
        act = bank_act(bank)(x @ bank.W_gate.t() + bank.b_gate) \
            * (x @ bank.W_up.t())
        norms = []
        for a in range(K):
            sl = slice(a * D, (a + 1) * D)
            out_a = act[..., sl] @ bank.W_down[:, sl].t()
            norms.append(out_a.norm(dim=-1))
    return torch.stack(norms, dim=-1)


def expected_sums(norms, source_ids, mask):
    """Hand-computed own/off sums+counts (B,T,K norms -> per-author scalars)."""
    own = (torch.tensor(AUTHOR_IDS).view(1, K)
           == source_ids.view(-1, 1)).float()          # (B, K)
    tok = mask.float().unsqueeze(-1)                    # (B, T, 1)
    B, T = mask.shape
    ownf = own.view(B, 1, K)
    own_sum = (norms * tok * ownf).sum(dim=(0, 1))
    own_cnt = (tok * ownf).expand(B, T, K).sum(dim=(0, 1))
    off_sum = (norms * tok * (1 - ownf)).sum(dim=(0, 1))
    off_cnt = (tok * (1 - ownf)).expand(B, T, K).sum(dim=(0, 1))
    return own_sum, own_cnt, off_sum, off_cnt


@pytest.fixture
def probe_batch():
    torch.manual_seed(42)
    B, T = 3, 5
    x = torch.randn(B, T, H)
    source_ids = torch.tensor([0, 2, NO_AUTHOR])
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 3:] = 0  # right padding, as the OU collator produces
    mask[2, 4:] = 0
    return x, source_ids, mask


def capture(banks, x, source_ids, mask):
    """One telemetry forward through every bank (mimics the wrapped model)."""
    state = BankState()
    state.set_batch(source_ids, mask)
    state.begin_telemetry()
    with torch.no_grad():
        for bank in banks.values():
            bank(x, state)
    stats = state.end_telemetry()
    state.clear()
    return stats


def test_streaming_matches_dense(probe_batch):
    """(a) streaming accumulation over row-chunks ≡ one-shot ≡ dense math,
    aggregated and per layer, including padded tokens."""
    x, source_ids, mask = probe_batch
    banks = {l: make_bank(l, seed=100 + l) for l in LAYERS}

    one_shot = NormAccumulator(LAYERS, K)
    one_shot.add(capture(banks, x, source_ids, mask))

    streamed = NormAccumulator(LAYERS, K)
    for rows in (slice(0, 2), slice(2, 3)):  # two chunks, as the real batcher
        streamed.add(capture(banks, x[rows], source_ids[rows], mask[rows]))

    # Dense reference, accumulated by hand across the two layers.
    exp = {k: np.zeros((len(LAYERS), K)) for k in
           ("own_sum", "own_cnt", "off_sum", "off_cnt")}
    for i, l in enumerate(LAYERS):
        norms = dense_author_norms(banks[l], x)
        os_, oc_, fs_, fc_ = expected_sums(norms, source_ids, mask)
        exp["own_sum"][i], exp["own_cnt"][i] = os_.numpy(), oc_.numpy()
        exp["off_sum"][i], exp["off_cnt"][i] = fs_.numpy(), fc_.numpy()

    for acc in (one_shot, streamed):
        np.testing.assert_allclose(acc.own_sum, exp["own_sum"],
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(acc.off_sum, exp["off_sum"],
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(acc.own_cnt, exp["own_cnt"])
        np.testing.assert_array_equal(acc.off_cnt, exp["off_cnt"])

    # Aggregate + per-layer means match the count-weighted dense reference.
    exp_own_mean = exp["own_sum"].sum(0) / np.maximum(exp["own_cnt"].sum(0), 1)
    exp_off_mean = exp["off_sum"].sum(0) / np.maximum(exp["off_cnt"].sum(0), 1)
    got_own = streamed.agg_mean("own")
    # Author 1 has no probe rows in this batch: mean must be exactly 0, not NaN.
    assert got_own[1] == 0.0
    np.testing.assert_allclose(got_own[[0, 2]], exp_own_mean[[0, 2]],
                               rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(streamed.agg_mean("off"), exp_off_mean,
                               rtol=1e-5, atol=1e-8)

    exp_layer_ratio = ((exp["own_sum"] / np.maximum(exp["own_cnt"], 1))
                       / np.clip(exp["off_sum"] / np.maximum(exp["off_cnt"], 1),
                                 1e-12, None))
    got = per_layer_ratio_medians(streamed)
    assert sorted(got.keys()) == [str(l) for l in sorted(LAYERS)]
    for i, l in enumerate(LAYERS):
        assert got[str(l)] == pytest.approx(float(np.median(exp_layer_ratio[i])),
                                            rel=1e-5)


def test_own_off_classification(probe_batch):
    """(c) own/off follows source_ids: rows label their author's slot 'own',
    everyone else's 'off'; NO_AUTHOR rows are off for ALL authors."""
    x, source_ids, mask = probe_batch
    banks = {l: make_bank(l, seed=200 + l) for l in LAYERS}
    acc = NormAccumulator(LAYERS, K)
    acc.add(capture(banks, x, source_ids, mask))

    # Unpadded tokens: row0 (author 0) -> 3, row1 (author 2) -> 5,
    # row2 (NO_AUTHOR) -> 4. Total 12.
    per_layer_own = acc.own_cnt[0]
    np.testing.assert_array_equal(per_layer_own, [3.0, 0.0, 5.0])
    np.testing.assert_array_equal(acc.off_cnt[0], [9.0, 12.0, 7.0])
    # Identical counting on every layer (same batch through each bank).
    np.testing.assert_array_equal(acc.own_cnt[1], acc.own_cnt[0])
    np.testing.assert_array_equal(acc.off_cnt[1], acc.off_cnt[0])
    # NO_AUTHOR-only batch: zero own tokens anywhere, all tokens off for all.
    ood_ids = torch.full((3,), NO_AUTHOR, dtype=torch.long)
    ood = NormAccumulator(LAYERS, K)
    ood.add(capture(banks, x, ood_ids, mask))
    assert ood.own_cnt.sum() == 0 and ood.own_sum.sum() == 0
    np.testing.assert_array_equal(ood.off_cnt[0], [12.0] * K)
    assert (ood.off_sum > 0).all()


def test_summarize_norms_and_ood_channel(probe_batch):
    """summarize_norms wires on/off/ratio + per-set OOD means correctly."""
    x, source_ids, mask = probe_batch
    banks = {l: make_bank(l, seed=300 + l) for l in LAYERS}
    author_acc = NormAccumulator(LAYERS, K)
    author_acc.add(capture(banks, x, source_ids, mask))
    ood_acc = NormAccumulator(LAYERS, K)
    ood_acc.add(capture(banks, x, torch.full((3,), NO_AUTHOR), mask))

    per_author, ratios, on, off = summarize_norms(
        author_acc, {"toy_ood": ood_acc}, AUTHOR_IDS)
    assert [r["author"] for r in per_author] == AUTHOR_IDS
    for j, rec in enumerate(per_author):
        assert rec["on_norm"] == pytest.approx(float(on[j]))
        assert rec["ratio"] == pytest.approx(float(on[j] / max(off[j], 1e-12)))
        assert rec["ood_norm"]["toy_ood"] == pytest.approx(
            float(ood_acc.agg_mean("off")[j]))
    # ratios feed the gate as a plain array
    assert ratios.shape == (K,)


def test_gate_verdict_thresholds():
    """(b) exact wiring of the pre-registered thresholds."""
    lazy = gate_verdict([1.0, 1.11, 1.9])
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


def test_own_only_mask_semantics():
    """own_only serves exactly one author's slices and restores the PRIOR
    masks (not blanket all-True) even when they were already partial."""
    banks = {l: make_bank(l, seed=400 + l) for l in LAYERS}
    with torch.no_grad():
        banks[LAYERS[0]].active.copy_(torch.tensor([True, True, False]))
    before = {l: b.active.clone() for l, b in banks.items()}

    torch.manual_seed(42)
    x = torch.randn(2, 4, H)
    with own_only(banks, 1):
        for b in banks.values():
            assert b.active.tolist() == [False, True, False]
        # Functional check: masked forward == author 1's dense contribution.
        bank = banks[LAYERS[1]]
        with torch.no_grad():
            got = bank(x, None)
        with torch.no_grad():
            act = bank_act(bank)(x @ bank.W_gate.t() + bank.b_gate) \
                * (x @ bank.W_up.t())
            sl = slice(1 * D, 2 * D)
            want = act[..., sl] @ bank.W_down[:, sl].t()
        np.testing.assert_allclose(got.numpy(), want.numpy(),
                                   rtol=1e-5, atol=1e-6)
    for l, b in banks.items():
        assert torch.equal(b.active, before[l]), f"layer {l} mask not restored"

    with pytest.raises(AssertionError):
        with own_only(banks, 99):  # non-resident id must refuse, not no-op
            pass
    for l, b in banks.items():
        assert torch.equal(b.active, before[l])
